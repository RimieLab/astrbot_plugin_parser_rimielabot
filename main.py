# main.py

import asyncio
import re
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import (
    At,
    Forward,
    Image,
    Json,
    Node,
    Nodes,
    Plain,
    Reply,
)
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from .core.arbiter import ArbiterContext, EmojiLikeArbiter
from .core.clean import CacheCleaner
from .core.config import PluginConfig
from .core.debounce import Debouncer
from .core.download import Downloader
from .core.exception import ParseException, TipException
from .core.parsers import BaseParser, BilibiliParser
from .core.render import Renderer
from .core.sender import MessageSender
from .core.utils import extract_json_url, has_ffmpeg


class ParserPlugin(Star):
    _SUCCESS_EMOJI_ID = 124  # /hd（OK）
    _SUCCESS_EMOJI_TYPE = "1"

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config, context=context)
        # 渲染器
        self.renderer = Renderer(self.cfg)
        # 下载器
        self.downloader = Downloader(self.cfg)
        # 防抖器
        self.debouncer = Debouncer(self.cfg)
        # 仲裁器
        self.arbiter = EmojiLikeArbiter()
        # 消息发送器
        self.sender = MessageSender(self.cfg, self.renderer)
        # 缓存清理器
        self.cleaner = CacheCleaner(self.cfg)
        # 关键词 -> Parser 映射
        self.parser_map: dict[str, BaseParser] = {}
        # 关键词 -> 正则 列表
        self.key_pattern_list: list[tuple[str, re.Pattern[str]]] = []
        # /测试转发 收集会话：umo -> {owner, next, items, created_at}
        self._forward_test_sessions: dict[str, dict] = {}

    @staticmethod
    def _read_version() -> str:
        try:
            metadata = Path(__file__).with_name("metadata.yaml")
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("version:"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return "unknown"

    async def initialize(self):
        """加载、重载插件时触发"""
        logger.info(
            f"[parser] 插件版本 {self._read_version()}，"
            f"ffmpeg={'可用' if has_ffmpeg() else '不可用（动图将按视频发送）'}"
        )
        # 加载渲染器资源
        await asyncio.to_thread(Renderer.load_resources)
        # 注册解析器
        self._register_parser()

    async def terminate(self):
        """插件卸载时触发"""
        # 关下载器里的会话
        await self.downloader.close()
        # 关所有解析器里的会话 (去重后的实例)
        unique_parsers = set(self.parser_map.values())
        for parser in unique_parsers:
            await parser.close_session()
        # 关缓存清理器
        await self.cleaner.stop()

    def _register_parser(self):
        """注册解析器（以 parser.enable 为唯一启用来源）"""
        # 所有 Parser 子类
        all_subclass = BaseParser.get_all_subclass()
        enabled_platforms = set(self.cfg.parser.enabled_platforms())

        enabled_classes: list[type[BaseParser]] = []
        enabled_names: list[str] = []
        for cls in all_subclass:
            platform_name = cls.platform.name

            if platform_name not in enabled_platforms:
                logger.debug(f"[parser] 平台未启用或未配置: {platform_name}")
                continue

            enabled_classes.append(cls)
            enabled_names.append(platform_name)

            # 一个平台一个 parser 实例
            parser = cls(self.cfg, self.downloader)

            # 关键词 → parser
            for keyword, _ in cls._key_patterns:
                self.parser_map[keyword] = parser

        enabled_text = "、".join(enabled_names) if enabled_names else "无"
        logger.info(f"[parser] 启用平台: {enabled_text}")

        # -------- 关键词-正则表（统一生成） --------
        patterns: list[tuple[str, re.Pattern[str]]] = []

        for cls in enabled_classes:
            for kw, pat in cls._key_patterns:
                patterns.append((kw, re.compile(pat) if isinstance(pat, str) else pat))

        # 长关键词优先，避免短词抢匹配
        patterns.sort(key=lambda x: -len(x[0]))

        self.key_pattern_list = patterns

        logger.debug(f"[parser] 关键词-正则对已生成: {[kw for kw, _ in patterns]}")

    def _get_parser_by_type(self, parser_type):
        for parser in self.parser_map.values():
            if isinstance(parser, parser_type):
                return parser
        raise ValueError(f"未找到类型为 {parser_type} 的 parser 实例")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息的统一入口"""
        umo = event.unified_msg_origin

        # 白名单
        if self.cfg.whitelist and umo not in self.cfg.whitelist:
            return

        # 黑名单
        if self.cfg.blacklist and umo in self.cfg.blacklist:
            return

        # 消息链
        chain = event.get_messages()
        if not chain:
            return

        seg1 = chain[0]
        text = event.message_str

        # /测试转发：用数字回复消息进行标记
        session = self._forward_test_sessions.get(umo)
        if session is not None:
            if time.monotonic() - session["created_at"] > 600:
                self._forward_test_sessions.pop(umo, None)
                session = None
            elif str(event.get_sender_id()) != session["owner"]:
                session = None

        if session is not None:
            marker = text.strip()
            reply_seg = next(
                (seg for seg in chain if isinstance(seg, Reply)), None
            )
            if reply_seg is not None and marker.isdigit():
                number = int(marker)
                if number != session["next"]:
                    await event.send(
                        event.chain_result(
                            [
                                Plain(
                                    f"请按顺序标记，下一条应为 "
                                    f"{session['next']}"
                                )
                            ]
                        )
                    )
                    return
                content = [
                    seg
                    for seg in (reply_seg.chain or [])
                    if not isinstance(seg, Reply)
                ]
                if not content:
                    await event.send(
                        event.chain_result([Plain("无法读取被回复消息的内容")])
                    )
                    return
                nodes = await self._expand_forward_nodes(event, content)
                if nodes == []:
                    await event.send(
                        event.chain_result(
                            [Plain("这条合并转发展开失败，未标记")]
                        )
                    )
                    return
                if nodes is not None:
                    session["items"][number] = {
                        "kind": "nodes",
                        "nodes": nodes,
                    }
                else:
                    session["items"][number] = {
                        "kind": "components",
                        "content": content,
                        "sender_id": reply_seg.sender_id,
                        "nickname": reply_seg.sender_nickname,
                        "time": reply_seg.time,
                    }
                session["next"] = number + 1
                await event.send(
                    event.chain_result([Plain(f"已标记 {number}")])
                )
                return

        # 指定机制：专门@其他bot的消息不解析
        self_id = event.get_self_id()
        mentioned_ids: set[str] = set()
        for seg in chain:
            if isinstance(seg, At):
                mentioned_ids.add(str(seg.qq))
            elif isinstance(seg, Plain):
                mentioned_ids.update(re.findall(r"<@!?([^>\s]+)>", seg.text))
        if (
            self.cfg.require_at_in_group
            and not isinstance(event, AiocqhttpMessageEvent)
            and event.get_message_type() == MessageType.GROUP_MESSAGE
            and self_id not in mentioned_ids
        ):
            return
        if (
            mentioned_ids
            and self_id not in mentioned_ids
            and "qq_official" not in mentioned_ids
        ):
            return

        # 卡片解析：扫描整条消息链，兼容 @ + JSON 卡片等组合消息。
        for seg in chain:
            if not isinstance(seg, Json):
                continue
            parsed_url = extract_json_url(seg.data)
            logger.debug(f"解析Json组件: {parsed_url}")
            if parsed_url:
                text = parsed_url
                break

        # 引用解析
        reply_seg = next((seg for seg in chain if isinstance(seg, Reply)), None)
        if self.cfg.enable_reply_parse and reply_seg and reply_seg.chain:
            reply_texts = []
            for seg in reply_seg.chain:
                if isinstance(seg, Plain):
                    reply_texts.append(seg.text)
                elif isinstance(seg, Json):
                    reply_texts.append(extract_json_url(seg.data))
            if reply_texts:
                reply_text = "".join(t for t in reply_texts if t)
                if reply_text:
                    text = reply_text

        if not text:
            return

        # 核心匹配逻辑 ：关键词 + 正则双重判定，汇集了所有解析器的正则对。
        keyword: str = ""
        searched: re.Match[str] | None = None
        for kw, pat in self.key_pattern_list:
            if kw not in text:
                continue
            if m := pat.search(text):
                keyword, searched = kw, m
                break
        if searched is None:
            return
        logger.debug(f"匹配结果: {keyword}, {searched}")

        # 仲裁机制
        if isinstance(event, AiocqhttpMessageEvent) and not event.is_private_chat():
            raw = event.message_obj.raw_message
            if not isinstance(raw, dict):
                logger.warning(f"Unexpected raw_message type: {type(raw)}")
                return
            is_win = await self.arbiter.compete(
                bot=event.bot,
                ctx=ArbiterContext(
                    message_id=int(raw["message_id"]),
                    msg_time=int(raw["time"]),
                    self_id=int(raw["self_id"]),
                ),
            )
            if not is_win:
                logger.debug("Bot在仲裁中输了, 跳过解析")
                return
            logger.debug("Bot在仲裁中胜出, 准备解析...")

        # 基于link防抖
        link = searched.group(0)
        if self.debouncer.hit_link(umo, link):
            logger.warning(f"[链接防抖] 链接 {link} 在防抖时间内，跳过解析")
            return

        # 解析
        logger.info(f"[解析开始] {link}")
        try:
            parse_res = await self.parser_map[keyword].parse(keyword, searched)
        except TipException as exc:
            logger.info(f"[解析提示] {link}: {exc.message}")
            await event.send(event.chain_result([Plain(exc.message)]))
            return
        except ParseException as exc:
            logger.warning(f"[解析失败] {link}: {exc}")
            return
        except Exception:
            logger.exception(f"[解析异常] {link}")
            return

        logger.info(
            f"[解析完成] {link}: 内容={len(parse_res.contents)}, "
            f"分组={len(parse_res.send_groups)}"
        )

        # 基于资源ID防抖
        resource_id = parse_res.get_resource_id()
        if self.debouncer.hit_resource(umo, resource_id):
            logger.warning(f"[资源防抖] 资源 {resource_id} 在防抖时间内，跳过发送")
            return

        # 发送
        if await self.sender.send_parse_result(event, parse_res):
            logger.info(f"[解析完成] 已发送 {link}")
            await self._react_success(event)
        else:
            logger.warning(f"[发送失败] {link}")

    async def _react_success(self, event: AstrMessageEvent) -> None:
        """解析并发送成功后，在原消息上回应 /hd（OK）表情。"""
        if not isinstance(event, AiocqhttpMessageEvent) or event.is_private_chat():
            return
        raw = event.message_obj.raw_message
        if not isinstance(raw, dict):
            return
        message_id = raw.get("message_id")
        if message_id is None:
            return
        try:
            await event.bot.set_msg_emoji_like(
                message_id=int(message_id),
                emoji_id=self._SUCCESS_EMOJI_ID,
                emoji_type=self._SUCCESS_EMOJI_TYPE,
                set=True,
            )
        except Exception as e:
            logger.debug(f"添加成功表情失败: {e}")

    @staticmethod
    def _as_node_dict(message: dict) -> dict:
        if message.get("type") == "node":
            return message
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        sender = message.get("sender")
        if not isinstance(sender, dict):
            sender = data.get("sender")
        if not isinstance(sender, dict):
            sender = {}
        content = message.get("content")
        if content is None:
            content = data.get("content")
        if not isinstance(content, list):
            content = [message] if message.get("type") else []
        user_id = (
            sender.get("user_id")
            or data.get("user_id")
            or message.get("user_id")
            or "0"
        )
        nickname = (
            sender.get("nickname")
            or sender.get("card")
            or data.get("nickname")
            or message.get("nickname")
            or ""
        )
        return {
            "type": "node",
            "data": {
                "user_id": str(user_id),
                "nickname": str(nickname),
                "content": content,
            },
        }

    async def _expand_forward_nodes(
        self, event: AstrMessageEvent, chain: list
    ) -> list[dict] | None:
        """把被标记消息里的合并转发展开成普通节点；不是转发则返回 None。"""
        if not isinstance(event, AiocqhttpMessageEvent):
            return None
        nodes_comp = next(
            (seg for seg in chain if isinstance(seg, Nodes)), None
        )
        if nodes_comp is not None:
            payload = await nodes_comp.to_dict()
            return payload.get("messages") or []

        forward = next(
            (seg for seg in chain if isinstance(seg, Forward)), None
        )
        if forward is None:
            return None
        try:
            resp = await event.bot.call_action(
                "get_forward_msg", id=str(forward.id)
            )
        except Exception as e:
            logger.warning(f"[测试转发] 展开聊天记录失败: {e}")
            return []
        messages = None
        if isinstance(resp, dict):
            messages = resp.get("messages")
            data = resp.get("data")
            if messages is None and isinstance(data, dict):
                messages = data.get("messages")
        elif isinstance(resp, list):
            messages = resp
        if not isinstance(messages, list):
            return []
        return [
            self._as_node_dict(item)
            for item in messages
            if isinstance(item, dict)
        ]

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("测试转发")
    async def forward_test_begin(self, event: AstrMessageEvent):
        """开始收集要合并转发的消息。"""
        umo = event.unified_msg_origin
        self._forward_test_sessions[umo] = {
            "owner": str(event.get_sender_id()),
            "next": 1,
            "items": {},
            "created_at": time.monotonic(),
        }
        yield event.plain_result(
            "已开始测试转发：请依次回复要合并的消息，并发送数字 1、2、3…"
            "完成后发送 /测试转发开始"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("测试转发取消")
    async def forward_test_cancel(self, event: AstrMessageEvent):
        self._forward_test_sessions.pop(event.unified_msg_origin, None)
        yield event.plain_result("已取消测试转发")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("测试转发开始")
    async def forward_test_finish(self, event: AstrMessageEvent):
        umo = event.unified_msg_origin
        session = self._forward_test_sessions.pop(umo, None)
        if session is None:
            yield event.plain_result("没有进行中的测试转发，请先发送 /测试转发")
            return
        items = session["items"]
        if not items:
            yield event.plain_result("还没有标记任何消息")
            return
        messages: list[dict] = []
        for number in sorted(items):
            item = items[number]
            if item.get("kind") == "nodes":
                messages.extend(item.get("nodes") or [])
                continue
            node = Node(
                content=item["content"],
                uin=str(item["sender_id"] or event.get_self_id()),
                name=item["nickname"] or "解析器",
            )
            messages.append(await node.to_dict())
        if not messages:
            yield event.plain_result("没有可转发的消息")
            return
        try:
            if event.get_group_id():
                await event.bot.call_action(
                    "send_group_forward_msg",
                    group_id=event.get_group_id(),
                    messages=messages,
                )
            else:
                await event.bot.call_action(
                    "send_private_forward_msg",
                    user_id=event.get_sender_id(),
                    messages=messages,
                )
            logger.info(f"[测试转发] 已合并 {len(messages)} 个节点")
        except Exception as e:
            logger.warning(f"[测试转发] 合并转发失败: {e}")
            yield event.plain_result("合并转发失败，请查看日志")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("开启解析")
    async def open_parser(self, event: AstrMessageEvent):
        """开启当前会话的解析"""
        umo = event.unified_msg_origin
        self.cfg.remove_blacklist(umo)
        yield event.plain_result("当前会话的解析已开启")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("关闭解析")
    async def close_parser(self, event: AstrMessageEvent):
        """关闭当前会话的解析"""
        umo = event.unified_msg_origin
        self.cfg.add_blacklist(umo)
        yield event.plain_result("当前会话的解析已关闭")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("登录B站", alias={"blogin", "登录b站"})
    async def login_bilibili(self, event: AstrMessageEvent):
        """扫码登录B站"""
        parser: BilibiliParser = self._get_parser_by_type(BilibiliParser)  # type: ignore
        qrcode = await parser.login.login_with_qrcode()
        yield event.chain_result([Image.fromBytes(qrcode)])
        async for msg in parser.login.check_qr_state():
            yield event.plain_result(msg)

    async def parse_url(self, url: str):
        """供其他插件调用的公开解析 API：传入 URL 解析并返回 ParseResult 对象"""
        for kw, pat in self.key_pattern_list:
            if kw in url and (m := pat.search(url)):
                return await self.parser_map[kw].parse(kw, m)
        raise ValueError(f"未匹配到支持的解析器: {url}")

    async def parse_and_send(self, event: AstrMessageEvent, url: str) -> bool:
        """供其他插件调用的公开方法：解析指定 URL 并直接发送至 event 对应会话"""
        result = await self.parse_url(url)
        return await self.sender.send_parse_result(event, result)

    @filter.llm_tool(name="parse_media_url")
    async def llm_parse_media_url(self, event: AstrMessageEvent, url: str) -> str:
        """
        解析用户提供的视频、图集或音频链接并自动发送解析结果。

        Args:
            url(string): 待解析的视频或图文等媒体链接
        """
        try:
            await self.parse_and_send(event, url)
            return "链接解析成功并已发送"
        except Exception as e:
            logger.error(f"[LLM Tool] 解析链接失败: {url}, error: {e}")
            return f"解析链接失败: {e}"
