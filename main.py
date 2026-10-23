# main.py

import asyncio
import re
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Forward, Image, Json, Plain, Reply
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
from .core.protected_forward import ProtectedForwardStore
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
        # 保护聊天记录（QQ 发送被拦截时使用）
        self.protected_forward = ProtectedForwardStore(
            self.cfg.data_dir / "protected_forward.json"
        )
        # 消息发送器
        self.sender = MessageSender(
            self.cfg, self.renderer, self.protected_forward
        )
        # 缓存清理器
        self.cleaner = CacheCleaner(self.cfg)
        # 关键词 -> Parser 映射
        self.parser_map: dict[str, BaseParser] = {}
        # 关键词 -> 正则 列表
        self.key_pattern_list: list[tuple[str, re.Pattern[str]]] = []
        # /dumpmsg 采集模式：记录下一条消息的结构
        self._dump_next_umo: str | None = None
        # /保护记录 采集模式：记录下一条特殊聊天记录
        self._protect_capture_umo: str | None = None

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

        # 采集模式只接受真正带消息链的消息，忽略“对方正在输入”等 notice
        if self._dump_next_umo == umo:
            self._dump_next_umo = None
            self._log_message_chain("采集", event)
            return

        if self._protect_capture_umo == umo:
            self._protect_capture_umo = None
            forward = self._find_forward(chain)
            if forward is None:
                await event.send(
                    event.chain_result([Plain("未找到聊天记录，请重新发送 /保护记录")])
                )
                return
            ok = await self._save_protected_forward(event, forward.id)
            await event.send(
                event.chain_result(
                    [
                        Plain(
                            "保护记录已保存"
                            if ok
                            else "保护记录保存失败，请查看日志"
                        )
                    ]
                )
            )
            return

        seg1 = chain[0]
        text = event.message_str

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
    def _find_forward(chain) -> Forward | None:
        for seg in chain:
            if isinstance(seg, Forward):
                return seg
            if isinstance(seg, Reply) and getattr(seg, "chain", None):
                for sub in seg.chain:
                    if isinstance(sub, Forward):
                        return sub
        return None

    async def _save_protected_forward(
        self, event: AstrMessageEvent, forward_id: str
    ) -> bool:
        if not isinstance(event, AiocqhttpMessageEvent):
            return False
        forward_id = str(forward_id)
        if not forward_id:
            return False

        messages = None
        try:
            resp = await event.bot.call_action(
                "get_forward_msg", id=forward_id
            )
            if isinstance(resp, dict):
                messages = resp.get("messages")
                data = resp.get("data")
                if messages is None and isinstance(data, dict):
                    messages = data.get("messages")
            elif isinstance(resp, list):
                messages = resp
        except Exception as e:
            logger.warning(f"[保护记录] 获取聊天记录详情失败（仍保存 id）: {e}")

        if not isinstance(messages, list):
            messages = None

        await asyncio.to_thread(
            self.protected_forward.save, forward_id, messages
        )
        logger.info(
            f"[保护记录] 已保存 forward_id={forward_id}, "
            f"详情节点数={len(messages) if messages else 0}"
        )
        return True

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("保护记录", alias={"保护转发"})
    async def protect_record(self, event: AstrMessageEvent):
        """保存一条特殊聊天记录，发送超时时与失败资源一起合并转发。"""
        if not isinstance(event, AiocqhttpMessageEvent):
            yield event.plain_result("保护记录仅支持 QQ/NapCat")
            return
        forward = self._find_forward(event.get_messages())
        if forward is None:
            self._protect_capture_umo = event.unified_msg_origin
            yield event.plain_result("请在下一条消息发送特殊聊天记录")
            return
        ok = await self._save_protected_forward(event, forward.id)
        yield event.plain_result(
            "保护记录已保存" if ok else "保护记录保存失败，请查看日志"
        )

    async def _send_forward_payload(
        self,
        event: AiocqhttpMessageEvent,
        messages: list[dict],
    ):
        group_id = event.get_group_id()
        if group_id:
            return await event.bot.call_action(
                "send_group_forward_msg",
                group_id=group_id,
                messages=messages,
            )
        return await event.bot.call_action(
            "send_private_forward_msg",
            user_id=event.get_sender_id(),
            messages=messages,
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("保护记录发送", alias={"查看保护记录"})
    async def send_protected_record(self, event: AstrMessageEvent):
        """把已保存的保护记录发回当前会话，方便检查内容是否正确。"""
        if not isinstance(event, AiocqhttpMessageEvent):
            yield event.plain_result("保护记录仅支持 QQ/NapCat")
            return

        forward_id = self.protected_forward.load_id()
        if not forward_id:
            yield event.plain_result("尚未保存保护记录")
            return

        protect_node = {
            "type": "node",
            "data": {
                "user_id": str(event.get_self_id()),
                "nickname": "解析器",
                "content": [
                    {"type": "forward", "data": {"id": forward_id}}
                ],
            },
        }
        try:
            await self._send_forward_payload(event, [protect_node])
            yield event.plain_result(
                f"已发送原保护记录，forward_id={forward_id}"
            )
            return
        except Exception as e:
            logger.warning(f"[保护记录] 按 id 回发失败: {e}")

        messages = self.protected_forward.load_messages()
        if not messages:
            yield event.plain_result(
                "原记录回发失败，且没有保存的展开内容"
            )
            return
        try:
            await self._send_forward_payload(event, messages)
            yield event.plain_result(
                "原记录 id 回发失败，已发送保存的展开内容"
            )
        except Exception as e:
            logger.warning(f"[保护记录] 回发展开内容失败: {e}")
            yield event.plain_result("保护记录回发失败，请查看日志")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("清除保护记录")
    async def clear_protected_record(self, event: AstrMessageEvent):
        self.protected_forward.clear()
        yield event.plain_result("保护记录已清除")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("保护记录状态")
    async def protected_record_status(self, event: AstrMessageEvent):
        forward_id = self.protected_forward.load_id()
        if not forward_id:
            yield event.plain_result("尚未保存保护记录")
            return
        messages = self.protected_forward.load_messages() or []
        logger.info(
            f"[保护记录] forward_id={forward_id}, 详情节点数={len(messages)}"
        )
        for index, node in enumerate(messages):
            data = node.get("data") if isinstance(node, dict) else None
            content = data.get("content") if isinstance(data, dict) else None
            if not isinstance(content, list):
                content = []
            types = [
                item.get("type") if isinstance(item, dict) else type(item).__name__
                for item in content
            ]
            logger.info(f"[保护记录] 节点 {index}: types={types}")
        yield event.plain_result(
            f"已保存保护记录 forward_id={forward_id}，"
            f"详情节点数={len(messages)}"
        )

    @staticmethod
    def _describe_segment(seg: object) -> str:
        name = type(seg).__name__
        parts: list[str] = []
        for attr in ("type", "file", "url", "path", "text", "data", "nodes", "chain"):
            value = getattr(seg, attr, None)
            if value is None:
                continue
            if attr in ("nodes", "chain") and isinstance(value, (list, tuple)):
                parts.append(f"{attr}={[type(item).__name__ for item in value]}")
            else:
                parts.append(f"{attr}={str(value)[:300]}")
        return f"{name}({', '.join(parts)})"

    def _log_message_chain(self, tag: str, event: AstrMessageEvent) -> None:
        chain = event.get_messages()
        logger.info(f"[调试] {tag} 消息链: {[self._describe_segment(seg) for seg in chain]}")
        raw = getattr(event.message_obj, "raw_message", None)
        if raw is not None:
            logger.info(f"[调试] {tag} raw_message: {str(raw)[:1000]}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("dumpmsg", alias={"调试消息"})
    async def dump_msg(self, event: AstrMessageEvent):
        """输出当前消息链结构；发送 /dumpmsg 采集 后记录下一条消息。"""
        self._log_message_chain("当前", event)
        if "采集" in event.message_str:
            self._dump_next_umo = event.unified_msg_origin
            yield event.plain_result("已开启采集：请发送特殊聊天记录")
        else:
            yield event.plain_result("已输出当前消息链调试日志")

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
