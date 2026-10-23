from itertools import chain
from pathlib import Path

from astrbot.api import logger
from astrbot.core.message.components import (
    BaseMessageComponent,
    File,
    Image,
    Node,
    Nodes,
    Plain,
    Record,
    Video,
)
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from .config import PluginConfig
from .data import (
    AudioContent,
    DynamicContent,
    FileContent,
    GraphicsContent,
    ImageContent,
    ParseResult,
    SendGroup,
    TextContent,
    VideoContent,
)
from .exception import (
    DownloadException,
    DownloadLimitException,
    DurationLimitException,
    SizeLimitException,
    ZeroSizeException,
)
from .protected_forward import ProtectedForwardStore
from .render import Renderer
from .utils import has_ffmpeg, transcode_video_audio_to_lc, video_to_gif


class MessageSender:
    """
    消息发送器

    职责：
    - 根据解析结果（ParseResult）规划发送策略
    - 控制是否渲染卡片、是否强制合并转发
    - 将不同类型的内容转换为 AstrBot 消息组件并发送

    重要原则：
    - 不在此处做解析
    - 不在此处决定“内容是什么”
    - 只负责“怎么发”
    """

    def __init__(
        self,
        config: PluginConfig,
        renderer: Renderer,
        protected_forward: ProtectedForwardStore | None = None,
    ):
        self.cfg = config
        self.renderer = renderer
        self.protected_forward = protected_forward

    def _to_file_uri(self, path: Path) -> str:
        if not path.is_absolute():
            path = path.resolve()
        return path.as_uri()

    @staticmethod
    def _image_from_path(path: Path) -> Image:
        return Image.fromFileSystem(str(path))

    @staticmethod
    def _video_from_path(path: Path) -> Video:
        return Video.fromFileSystem(str(path))

    @staticmethod
    def _record_from_path(path: Path) -> Record:
        return Record.fromFileSystem(str(path))

    @staticmethod
    def _iter_contents(result: ParseResult):
        return chain(result.contents, result.repost.contents if result.repost else ())

    def _build_send_plan(
        self,
        result: ParseResult,
        contents: list | tuple | None = None,
        *,
        force_merge_override: bool | None = None,
        render_card_override: bool | None = None,
    ) -> dict:
        """
        根据解析结果生成发送计划（plan）

        plan 只做“策略决策”，不做任何 IO 或发送动作。
        后续发送流程严格按 plan 执行，避免逻辑分散。
        """
        light, heavy = [], []

        # 合并主内容 + 转发内容，统一参与发送策略计算
        iterable = contents if contents is not None else self._iter_contents(result)
        for cont in iterable:
            match cont:
                case ImageContent() | GraphicsContent() | TextContent():
                    light.append(cont)
                case VideoContent() | AudioContent() | FileContent() | DynamicContent():
                    heavy.append(cont)
                case _:
                    light.append(cont)

        # 基础卡片渲染条件：单条重媒体配置或显式指定
        is_single_heavy = len(heavy) == 1 and not light
        render_card = is_single_heavy and self.cfg.single_heavy_render_card
        if render_card_override is not None:
            render_card = render_card_override

        # 实际媒体段数量
        media_count = len(light) + len(heavy)

        # 达到阈值后，强制合并转发，避免刷屏
        force_merge = (media_count + (1 if render_card else 0)) >= self.cfg.forward_threshold
        if force_merge_override is not None:
            force_merge = force_merge_override

        # NapCat 的伪造合并转发只会上传图片/语音为 base64，
        # 普通视频仍是 file:// 本地路径，会导致“上传转发消息资源失败”。
        # DynamicContent 会先转成 GIF 图片；没有 ffmpeg 时只能按视频发送。
        ffmpeg_ok = has_ffmpeg()
        if force_merge and any(
            isinstance(cont, VideoContent)
            or (isinstance(cont, DynamicContent) and not ffmpeg_ok)
            for cont in heavy
        ):
            force_merge = False

        # 合并转发时始终内联预览卡片（除非显式禁用）
        if force_merge and render_card_override is not False:
            render_card = True

        return {
            "light": light,
            "heavy": heavy,
            "render_card": render_card,
            # 预览卡片：仅在“渲染卡片 + 不合并”时独立发送
            "preview_card": render_card and not force_merge,
            "force_merge": force_merge,
        }

    async def _send_preview_card(
        self,
        event: AstrMessageEvent,
        result: ParseResult,
        plan: dict,
    ) -> bool:
        """
        发送预览卡片（独立消息）

        场景：
        - 只有一个重媒体
        - 未触发合并转发
        - 卡片作为“预览”，不与正文混合

        返回是否真的发出了卡片，供调用方判断该组是否算发送成功。
        """
        if not plan["preview_card"]:
            return False

        image_path = await self.renderer.render_card(result)
        if not image_path:
            return False

        try:
            logger.info("[发送] 预览卡片")
            await event.send(event.chain_result([self._image_from_path(image_path)]))
            return True
        except Exception as e:
            if self._is_napcat_timeout(e):
                if await self._try_protected_forward(
                    event, [self._image_from_path(image_path)]
                ):
                    return True
                # NapCat 常见假超时：卡片通常已经送达，按成功处理避免重复兜底
                logger.warning(f"发送预览卡片超时（NapCat 可能已送达）： error={e}")
                return True
            logger.error(f"发送预览卡片失败： error={e}")
            return False

    async def _build_segments(
        self,
        result: ParseResult,
        plan: dict,
    ) -> list[BaseMessageComponent]:
        """
        根据发送计划构建消息段列表

        这里负责：
        - 下载媒体
        - 转换为 AstrBot 消息组件
        """
        segs: list[BaseMessageComponent] = []

        # 合并转发时，卡片以内联形式作为一个消息段参与合并
        if plan["render_card"] and plan["force_merge"]:
            if image_path := await self.renderer.render_card(result):
                segs.append(self._image_from_path(image_path))

        # 轻媒体处理
        for cont in plan["light"]:
            if isinstance(cont, TextContent):
                if cont.text:
                    segs.append(Plain(cont.text))
                continue

            try:
                path: Path = await cont.get_path()
            except (DownloadLimitException, ZeroSizeException):
                continue
            except DownloadException:
                if self.cfg.show_download_fail_tip:
                    segs.append(Plain("此项媒体下载失败"))
                continue

            match cont:
                case ImageContent():
                    segs.append(self._image_from_path(path))
                case GraphicsContent() as g:
                    segs.append(self._image_from_path(path))
                    # GraphicsContent 允许携带补充文本
                    if g.text:
                        segs.append(Plain(g.text))
                    if g.alt:
                        segs.append(Plain(g.alt))

        # 重媒体处理
        for cont in plan["heavy"]:
            try:
                path: Path = await cont.get_path()
            except (SizeLimitException, DurationLimitException) as exc:
                if self.cfg.show_download_fail_tip:
                    message = (
                        "此项媒体超过时长限制"
                        if isinstance(exc, DurationLimitException)
                        else "此项媒体超过大小限制"
                    )
                    segs.append(Plain(message))
                continue
            except DownloadException:
                if self.cfg.show_download_fail_tip:
                    segs.append(Plain("此项媒体下载失败"))
                continue

            match cont:
                case VideoContent():
                    path = await transcode_video_audio_to_lc(path)
                    segs.append(self._video_from_path(path))
                case DynamicContent() as dynamic:
                    gif_path = dynamic.gif_path
                    if gif_path is None or not gif_path.exists():
                        gif_path = await video_to_gif(path)
                        dynamic.gif_path = gif_path
                    if gif_path is None:
                        logger.warning(
                            f"[动图] GIF 转换失败，改为按视频发送: {path.name}"
                        )
                        segs.append(self._video_from_path(path))
                        continue
                    segs.append(self._image_from_path(gif_path))
                case AudioContent():
                    segs.append(
                        File(name=path.name, file=self._to_file_uri(path))
                        if self.cfg.audio_to_file
                        else self._record_from_path(path)
                    )
                case FileContent():
                    segs.append(File(name=path.name, file=self._to_file_uri(path)))

        return segs

    def _merge_segments_if_needed(
        self,
        event: AstrMessageEvent,
        segs: list[BaseMessageComponent],
        force_merge: bool,
    ) -> list[BaseMessageComponent]:
        """
        根据策略决定是否将消息段合并为转发节点

        合并后的消息结构：
        - 每个原始消息段成为一个 Node
        - 统一使用机器人自身身份
        """
        if not force_merge or not segs:
            return segs

        nodes = Nodes([])
        self_id = event.get_self_id()

        for seg in segs:
            nodes.nodes.append(Node(uin=self_id, name="解析器", content=[seg]))

        return [nodes]

    @staticmethod
    def _build_text_fallback(result: ParseResult) -> list[BaseMessageComponent]:
        lines: list[str] = []
        if result.header:
            lines.append(result.header)
        if result.text:
            lines.append(result.text)
        elif result.extra.get("info"):
            lines.append(str(result.extra["info"]))

        text = "\n".join(line for line in lines if line).strip()
        return [Plain(text)] if text else []

    def _resolve_groups(self, result: ParseResult) -> list[SendGroup]:
        if result.send_groups:
            return result.send_groups
        return [SendGroup(contents=list(MessageSender._iter_contents(result)))]

    async def _send_group(
        self,
        event: AstrMessageEvent,
        result: ParseResult,
        group: SendGroup,
    ) -> bool:
        plan = self._build_send_plan(
            result,
            group.contents,
            force_merge_override=group.force_merge,
            render_card_override=group.render_card,
        )

        logger.info(
            f"[发送] 计划: 轻媒体={len(plan['light'])}, "
            f"重媒体={len(plan['heavy'])}, 卡片={plan['render_card']}, "
            f"合并={plan['force_merge']}"
        )
        card_sent = await self._send_preview_card(event, result, plan)

        segs = await self._build_segments(result, plan)
        # 兜底：转换失败时可能混入 Video，绝不能塞进伪造合并转发
        if any(isinstance(seg, Video) for seg in segs):
            plan["force_merge"] = False
        segs = self._merge_segments_if_needed(event, segs, plan["force_merge"])

        # 只有卡片、没有其他消息段时，卡片发出即视为该组发送成功，
        # 否则会错误触发 send_parse_result 的文字兜底（Pixiv 正文/标签重复）。
        if not segs:
            return card_sent

        # QQ/NapCat 一条消息通常只处理一个视频，多个视频会只发出第一个。
        # 含多个视频或视频混在图片里时，拆成多条发送。
        if self._should_split_video(segs):
            return await self._send_segments_individually(event, segs, card_sent)

        try:
            logger.info(f"[发送] 消息段: {[type(seg).__name__ for seg in segs]}")
            await event.send(event.chain_result(segs))
            return True
        except Exception as e:
            if self._is_napcat_timeout(e) and await self._try_protected_forward(
                event, segs
            ):
                return True
            self._log_send_error(e, segs)
            return card_sent

    @staticmethod
    def _should_split_video(segs: list[BaseMessageComponent]) -> bool:
        video_count = sum(isinstance(seg, Video) for seg in segs)
        return video_count > 0 and (video_count > 1 or len(segs) > video_count)

    async def _send_segments_individually(
        self,
        event: AstrMessageEvent,
        segs: list[BaseMessageComponent],
        card_sent: bool,
    ) -> bool:
        batches: list[list[BaseMessageComponent]] = []
        non_video = [seg for seg in segs if not isinstance(seg, Video)]
        if non_video:
            batches.append(non_video)
        batches.extend([seg] for seg in segs if isinstance(seg, Video))

        logger.info(f"[发送] 分批发送: {len(batches)} 条")
        sent = card_sent
        for index, batch in enumerate(batches):
            try:
                logger.info(
                    f"[发送] 第 {index + 1}/{len(batches)} 条: "
                    f"{[type(seg).__name__ for seg in batch]}"
                )
                await event.send(event.chain_result(batch))
                sent = True
            except Exception as e:
                if self._is_napcat_timeout(e) and await self._try_protected_forward(
                    event, batch
                ):
                    sent = True
                    continue
                self._log_send_error(e, batch)
        return sent

    @staticmethod
    def _video_as_file(seg: BaseMessageComponent) -> BaseMessageComponent:
        """合并转发里的本地视频 NapCat 无法上传，改用 File 节点。"""
        if not isinstance(seg, Video):
            return seg
        path = getattr(seg, "path", "") or ""
        if not path:
            file_uri = getattr(seg, "file", "") or ""
            if file_uri.startswith("file://"):
                from urllib.parse import unquote, urlparse

                path = unquote(urlparse(file_uri).path)
        if not path:
            return seg
        video_path = Path(path)
        if not video_path.is_file():
            return seg
        logger.info(f"[保护转发] 视频改用文件节点: {video_path.name}")
        return File(name=video_path.name, file=video_path.as_uri())

    async def _try_protected_forward(
        self,
        event: AstrMessageEvent,
        segs: list[BaseMessageComponent],
    ) -> bool:
        """把失败资源附在保护聊天记录后面，用合并转发再试一次。"""
        if not self.protected_forward or not self.protected_forward.exists():
            return False
        if not isinstance(event, AiocqhttpMessageEvent):
            return False
        try:
            forward_id = self.protected_forward.load_id()
            if not forward_id:
                return False
            protected_segs = [self._video_as_file(seg) for seg in segs]
            node = Node(
                content=protected_segs,
                uin=str(event.get_self_id()),
                name="解析器",
            )
            node_dict = await node.to_dict()
            # 外层合并转发里放两个嵌套的“保护记录”卡片，再接失败资源
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
            messages = [protect_node, protect_node, node_dict]
            logger.info(
                f"[保护转发] 使用 forward_id={forward_id}, "
                f"节点类型={[item.get('type') for item in messages]}"
            )
            group_id = event.get_group_id()
            if group_id:
                await event.bot.call_action(
                    "send_group_forward_msg",
                    group_id=group_id,
                    messages=messages,
                )
            else:
                await event.bot.call_action(
                    "send_private_forward_msg",
                    user_id=event.get_sender_id(),
                    messages=messages,
                )
            logger.info(f"[保护转发] 发送成功，共 {len(messages)} 个节点")
            return True
        except Exception as e:
            logger.warning(f"[保护转发] 发送失败: {e}")
            return False

    def _log_send_error(
        self, exc: Exception, segs: list[BaseMessageComponent]
    ) -> None:
        seg_meta = self._collect_seg_meta(segs)
        if self._is_napcat_timeout(exc):
            # NapCat 偶发假超时：日志报错但消息通常已经送达
            logger.warning(
                f"发送超时（NapCat 可能已送达）： error={exc}, segments={seg_meta}"
            )
        else:
            logger.error(f"发送解析结果失败： error={exc}, segments={seg_meta}")

    @staticmethod
    def _is_napcat_timeout(exc: Exception) -> bool:
        text = str(exc)
        return "onMsgInfoListUpdate" in text or "Timeout: NTEvent" in text

    @staticmethod
    def _collect_seg_meta(segs: list[BaseMessageComponent]) -> list[dict[str, str]]:
        """提取消息段元信息，用于失败日志定位。"""
        meta: list[dict[str, str]] = []

        for seg in segs:
            item = {"type": seg.__class__.__name__}
            for attr in ("file", "path", "url"):
                value = getattr(seg, attr, None)
                if value:
                    item["media"] = str(value)
                    break
            meta.append(item)

        return meta

    async def send_parse_result(
        self,
        event: AstrMessageEvent,
        result: ParseResult,
    ) -> bool:
        """
        发送解析结果的统一入口

        执行顺序固定：
        1. 构建发送计划
        2. 发送预览卡片（如有）
        3. 构建消息段
        4. 必要时合并转发
        5. 最终发送
        """
        groups = self._resolve_groups(result)

        sent = False
        for group in groups:
            sent = await self._send_group(event, result, group) or sent

        if sent:
            return True

        # 所有分组都没发出去时，才退化为纯文本提示
        segs = self._build_text_fallback(result)
        if not segs:
            logger.warning("发送结果为空，不执行发送")
            return False

        try:
            await event.send(event.chain_result(segs))
        except Exception as e:
            seg_meta = self._collect_seg_meta(segs)
            logger.error(f"发送解析结果失败： error={e}, segments={seg_meta}")
            return False
        return False
