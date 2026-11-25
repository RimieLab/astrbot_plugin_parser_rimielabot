import re
from typing import ClassVar

import msgspec
import yt_dlp
from aiohttp import ClientError
from msgspec import Struct

from astrbot.api import logger

from ..config import PluginConfig
from ..cookie import CookieJar
from ..data import TextContent
from ..download import Downloader, detect_js_runtimes
from ..exception import ParseException, TipException
from .base import BaseParser, Platform, handle


class YouTubeParser(BaseParser):
    # 平台信息
    platform: ClassVar[Platform] = Platform(name="youtube", display_name="油管")

    def __init__(self, config: PluginConfig, downloader: Downloader):
        super().__init__(config, downloader)
        self.mycfg = config.parser.youtube
        self.headers.update(
            {
                "Referer": "https://www.youtube.com/",
                # 旧的 Chrome/55 UA 容易被 YouTube 媒体 CDN 判为异常请求
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
            }
        )
        self.cookiejar = CookieJar(config, self.mycfg, domain="youtube.com")
        if not self.mycfg.cookies:
            logger.warning("油管Cookie未配置，将无法解析相关媒体")
        else:
            expired = bool(self.cookiejar.cookies) and all(
                cookie.is_expired() for cookie in self.cookiejar.cookies
            )
            logger.info(
                f"[油管] Cookies 已配置，数量={len(self.cookiejar.cookies)}，"
                f"{'已全部过期' if expired else '文件状态正常'}"
            )
            if expired:
                logger.warning("[油管] Cookies 已过期，请重新导出后更新配置")
        if not detect_js_runtimes():
            logger.warning(
                "[油管] 未检测到 JS 运行时（Deno/Node 22+/Bun），"
                "YouTube 签名挑战无法求解，解析会报 "
                "'Requested format is not available'。"
                "请安装 Deno/Node.js 22+，或执行 "
                "pip install \"yt-dlp[default,deno]\" 后重启"
            )

    # 从匹配到的链接里提取视频 ID
    _VIDEO_ID_PATTERN = re.compile(
        r"(?:watch\?(?:[^\s#]*&)?v=|shorts/|youtu\.be/)([A-Za-z\d_-]{6,})"
    )

    @classmethod
    def canonical_url(cls, matched: str) -> str:
        """把匹配到的片段规范成单视频链接。

        电台/混音链接（v=ID&list=RD...&start_radio=1）会带上播放列表参数，
        直接交给 yt-dlp 会被当成播放列表处理，这里只保留视频 ID。
        """
        match = cls._VIDEO_ID_PATTERN.search(matched)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
        return matched

    @handle("youtu", r"youtu\.be/[A-Za-z\d\._\?%&\+\-=/#]+")
    @handle(
        "youtube",
        # watch?v=ID 允许后面再跟 list=RD.../start_radio 等参数（电台、混音列表链接）
        r"youtube\.com/(?:watch(?:\?[^\s#]*?v=[A-Za-z\d_\-]{6,})"
        r"|shorts/[A-Za-z\d_\-]+)",
    )
    async def _parse_video(self, searched: re.Match[str]):
        return await self.parse_video(searched)

    async def _extract_info(self, url: str):
        try:
            return await self.downloader.ytdlp_extract_info(
                url,
                cookiefile=self.cookiejar.cookie_file,
                headers=self.headers,
                proxy=self.proxy,
            )
        except yt_dlp.utils.DownloadError as exc:
            message = str(exc)
            lowered = message.lower()
            if (
                "sign in to confirm" in lowered
                or "not a bot" in lowered
                or "page needs to be reloaded" in lowered
            ):
                raise TipException(
                    "YouTube 要求登录/人机验证：请更新油管 Cookies，"
                    "或给 YouTube 配置干净的住宅代理后重试"
                ) from exc
            if (
                "requested format is not available" in lowered
                or "n challenge solving failed" in lowered
                or "signature solving failed" in lowered
                or "only images are available" in lowered
            ):
                version = getattr(getattr(yt_dlp, "version", None), "__version__", "unknown")
                logger.warning(
                    "[油管] 无可用视频格式：请确认运行环境已安装 Deno 或 "
                    f"Node.js 22+，并更新 yt-dlp（当前 {version}）；"
                    "否则 YouTube 只会返回 storyboard 图片"
                )
                raise ParseException(
                    "YouTube 无可用视频格式：请更新 yt-dlp 并安装 "
                    "Node.js 22+/Deno 后重启"
                ) from exc
            raise ParseException(f"YouTube 解析失败: {message[:300]}") from exc

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """秒数格式化为 m:ss 或 h:mm:ss"""
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return (
            f"{hours}:{minutes:02d}:{secs:02d}"
            if hours
            else f"{minutes}:{secs:02d}"
        )

    async def parse_video(self, searched: re.Match[str]):
        # 从匹配对象中获取原始URL（规范成单视频链接，去掉播放列表参数）
        url = self.canonical_url(searched.group(0))

        video_info = await self._extract_info(url)
        author = await self._fetch_author_info(video_info.channel_id)

        contents = []
        if (
            video_info.duration <= self.cfg.max_duration
            or self.ignore_duration_limit
        ):
            video = self.downloader.ytdlp_download_video(
                url,
                cookiefile=self.cookiejar.cookie_file,
                headers=self.headers,
                proxy=self.proxy,
                format="bv*[height<=720]+ba/b[height<=720]",
                ignore_duration=self.ignore_duration_limit,
            )
            contents.append(
                self.create_video_content(
                    video,
                    video_info.thumbnail,
                    video_info.duration,
                )
            )
        else:
            duration = self._fmt_duration(video_info.duration)
            limit = self._fmt_duration(self.cfg.max_duration)
            logger.info(f"[油管] 视频时长 {duration} 超过上限 {limit}，仅发送封面")
            contents.extend(self.create_image_contents([video_info.thumbnail]))
            contents.append(
                TextContent(
                    f"视频时长 {duration} 超过解析上限 {limit}，已跳过视频下载；"
                    "可在配置中调大「资源最大时长」，或 @Bot 重发链接强制解析"
                )
            )

        return self.result(
            title=video_info.title,
            author=author,
            contents=contents,
            timestamp=video_info.timestamp,
        )

    @handle(
        "ym",
        r"^ym(?P<url>https?://(?:www\.)?(youtu\.be/[A-Za-z\d_-]+|youtube\.com/(?:watch|shorts)(?:\?v=[A-Za-z\d_-]+|/[A-Za-z\d_-]+)))",
    )
    async def ym(self, searched: re.Match[str]):
        """获取油管的音频(需加ym前缀)"""
        url = self.canonical_url(searched.group("url"))
        video_info = await self._extract_info(url)
        author = await self._fetch_author_info(video_info.channel_id)

        contents = []
        contents.extend(self.create_image_contents([video_info.thumbnail]))

        if (
            video_info.duration <= self.cfg.max_duration
            or self.ignore_duration_limit
        ):
            audio_task = self.downloader.ytdlp_download_audio(
                url,
                cookiefile=self.cookiejar.cookie_file,
                headers=self.headers,
                proxy=self.proxy,
            )
            contents.append(
                self.create_audio_content(audio_task, duration=video_info.duration)
            )
        else:
            duration = self._fmt_duration(video_info.duration)
            limit = self._fmt_duration(self.cfg.max_duration)
            logger.info(f"[油管] 音频时长 {duration} 超过上限 {limit}，仅发送封面")
            contents.append(
                TextContent(
                    f"音频时长 {duration} 超过解析上限 {limit}，已跳过音频下载；"
                    "可在配置中调大「资源最大时长」，或 @Bot 重发链接强制解析"
                )
            )

        return self.result(
            title=video_info.title,
            author=author,
            contents=contents,
            timestamp=video_info.timestamp,
        )

    async def _fetch_author_info(self, channel_id: str):
        url = "https://www.youtube.com/youtubei/v1/browse?prettyPrint=false"
        payload = {
            "context": {
                "client": {
                    "hl": "zh-HK",
                    "gl": "US",
                    "deviceMake": "Apple",
                    "deviceModel": "",
                    "clientName": "WEB",
                    "clientVersion": "2.20251002.00.00",
                    "osName": "Macintosh",
                    "osVersion": "10_15_7",
                },
                "user": {"lockedSafetyMode": False},
                "request": {
                    "useSsl": True,
                    "internalExperimentFlags": [],
                    "consistencyTokenJars": [],
                },
            },
            "browseId": channel_id,
        }
        async with self.session.post(
            url,
            json=payload,
            headers=self.headers,
            proxy=self.proxy,
        ) as resp:
            if resp.status >= 400:
                raise ClientError(f"YouTube browse API {resp.status} {resp.reason}")
            browse = msgspec.json.decode(await resp.read(), type=BrowseResponse)

        return self.create_author(browse.name, browse.avatar_url, browse.description)


class Thumbnail(Struct):
    url: str


class AvatarInfo(Struct):
    thumbnails: list[Thumbnail]


class ChannelMetadataRenderer(Struct):
    title: str
    description: str
    avatar: AvatarInfo


class Metadata(Struct):
    channelMetadataRenderer: ChannelMetadataRenderer


class Avatar(Struct):
    thumbnails: list[Thumbnail]


class BrowseResponse(Struct):
    metadata: Metadata

    @property
    def name(self) -> str:
        return self.metadata.channelMetadataRenderer.title

    @property
    def avatar_url(self) -> str | None:
        thumbnails = self.metadata.channelMetadataRenderer.avatar.thumbnails
        return thumbnails[0].url if thumbnails else None

    @property
    def description(self) -> str:
        return self.metadata.channelMetadataRenderer.description
