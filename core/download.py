import os
import re
import shutil
import sysconfig
import time
from asyncio import Task, TimeoutError, create_task, gather, sleep, to_thread
from collections.abc import Callable, Coroutine
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import aiofiles
import yt_dlp
from aiohttp import ClientError, ClientSession, ClientTimeout
from msgspec import Struct, convert
from tqdm.asyncio import tqdm

from astrbot.api import logger

from .config import PluginConfig
from .constants import COMMON_HEADER
from .exception import (
    DownloadException,
    DurationLimitException,
    InvalidMediaException,
    ParseException,
    SizeLimitException,
    ZeroSizeException,
)
from .utils import LimitedSizeDict, generate_file_name, merge_av, safe_unlink

P = ParamSpec("P")
T = TypeVar("T")

# YouTube 的签名/n 挑战需要 JS 运行时才能求解。yt-dlp 默认只启用 deno，
# 这里按推荐顺序声明全部候选，由其自行跳过未安装的运行时。
_JS_RUNTIME_BINARIES: dict[str, str] = {
    "deno": "deno",
    "node": "node",
    "bun": "bun",
    "quickjs": "qjs",
}


def _runtime_path(binary: str) -> str | None:
    """查找 JS 运行时可执行文件（含当前 Python 环境的 scripts 目录）"""
    exe = binary + (sysconfig.get_config_var("EXE") or "")
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidate = Path(scripts) / exe
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(binary)


def detect_js_runtimes() -> list[str]:
    """返回当前环境可用的 JS 运行时名称"""
    return [
        name
        for name, binary in _JS_RUNTIME_BINARIES.items()
        if _runtime_path(binary)
    ]


def js_runtime_opts() -> dict[str, dict[str, Any]]:
    """yt-dlp 的 js_runtimes 选项（未安装的运行时会被自动忽略）"""
    return {name: {} for name in _JS_RUNTIME_BINARIES}


def auto_task(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Task[T]]:
    """装饰器：自动将异步函数调用转换为 Task, 完整保留类型提示"""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Task[T]:
        coro = func(*args, **kwargs)
        name = " | ".join(str(arg) for arg in args if isinstance(arg, str))
        return create_task(coro, name=func.__name__ + " | " + name)

    return wrapper


class VideoInfo(Struct):
    title: str
    """标题"""
    channel: str
    """频道名称"""
    uploader: str
    """上传者 id"""
    duration: int
    """时长"""
    timestamp: int
    """发布时间戳"""
    thumbnail: str
    """封面图片"""
    description: str
    """简介"""
    channel_id: str
    """频道 id"""

    @property
    def author_name(self) -> str:
        return f"{self.channel}@{self.uploader}"


class Downloader:
    """下载器，支持youtube-dlp 和 流式下载"""

    def __init__(self, config: PluginConfig):
        self.cfg = config
        self.max_size = self.cfg.source_max_size
        self.default_headers: dict[str, str] = COMMON_HEADER.copy()
        # 视频信息缓存
        self.info_cache: LimitedSizeDict[str, VideoInfo] = LimitedSizeDict()
        # 用于流式下载的客户端
        self.client = ClientSession(
            timeout=ClientTimeout(
                total=self.cfg.download_timeout,
                # CDN 偶发只发响应头不出数据，30 秒无数据就重试
                sock_read=30,
            )
        )

    async def close(self):
        """关闭网络客户端"""
        await self.client.close()

    def _writable_cookiefile(self, cookiefile: Path | None) -> Path | None:
        """返回可写的 cookie 文件路径。

        yt-dlp 会在下载结束时把轮换后的 Cookie 写回 cookiefile，
        如果原文件不可写（例如属主不是 AstrBot 用户），会抛 PermissionError
        干扰下载收尾。这里复制一份到缓存目录使用。
        """
        if cookiefile is None or not cookiefile.is_file():
            return cookiefile
        if os.access(cookiefile, os.W_OK):
            return cookiefile
        try:
            copy_path = self.cfg.cache_dir / f"ck_{cookiefile.stem}.txt"
            copy_path.write_bytes(cookiefile.read_bytes())
            copy_path.chmod(0o600)
            return copy_path
        except OSError as exc:
            logger.debug(f"[下载] Cookie 文件不可写且复制失败，改为只读使用: {exc}")
            return cookiefile

    def _retry_opts(self) -> dict[str, Any]:
        """yt-dlp 的 Python API 默认不重试，这里显式开启网络重试。

        CLI 的 retries / fragment_retries 默认是 10，而直接构造
        ``YoutubeDL(params)`` 时是 0，CDN 偶发 503 会直接失败。
        """
        retries = max(0, int(self.cfg.download_retry_times))
        return {
            "retries": retries,
            "fragment_retries": retries,
            "extractor_retries": 3,
            "retry_sleep_functions": {
                "http": self._retry_backoff,
                "fragment": self._retry_backoff,
            },
        }

    @staticmethod
    def _retry_backoff(n: int) -> float:
        """第 n 次重试前的等待秒数（指数退避，最多 60 秒）

        YouTube 对短时间内的重复请求很敏感，1~2 秒就重来容易触发
        风控，因此起步就是 10 秒。

        注意 yt-dlp 是以 ``sleep_func(n=...)`` 关键字形式调用的，
        参数名必须是 n。
        """
        return float(min(10 * 2**n, 60))

    @staticmethod
    def _is_format_rejected(message: str) -> bool:
        """判断是否属于“某一路媒体流被 CDN 拒绝”。

        YouTube 会对部分客户端/编码格式的 DASH 流单独返回 503，
        换一路格式往往就能成功。
        """
        lowered = message.lower()
        return "503" in lowered or "service unavailable" in lowered

    @staticmethod
    def _height_of(base_format: str) -> str | None:
        """从格式字符串里取出分辨率上限，没有则返回 None"""
        if match := re.search(r"height<=?(\d+)", base_format):
            return match.group(1)
        return None

    def _format_fallbacks(self, base_format: str) -> list[str]:
        """主格式失败后依次尝试的备用格式。

        H.264 被拒时换 AV1，再不行退回单文件渐进式格式；
        单文件格式对客户端判定最宽松。
        """
        height = self._height_of(base_format) or "720"
        if self.cfg.prefer_progressive_video:
            # 机房 IP 常被单独限流 DASH 流，直接走单文件格式
            return ["b", f"b[height<={height}]/b"]
        return [
            # 首选：同分辨率，优先 AV1（部分 IP 上 H.264 流会被单独限流）
            f"bv*[height<={height}][vcodec^=av01]+ba/bv*[height<={height}]+ba/b",
            # 次选：单文件渐进式（18 等），风控判定最宽松
            f"b[height<={height}]/b",
            # 最后：任意可用格式
            "b",
        ]

    def _initial_video_format(self, base_format: str) -> str:
        """首次下载使用的格式；开启“优先单文件”时直接选渐进式"""
        if not self.cfg.prefer_progressive_video:
            return base_format
        # 单文件格式只有 360p / 720p 等少数档位，直接让 yt-dlp 选可用的最佳单文件
        return "b"

    @staticmethod
    def _is_transient_error(message: str) -> bool:
        """判断下载错误是否值得重新解析并重试"""
        lowered = message.lower()
        return any(
            token in lowered
            for token in (
                "503",
                "502",
                "504",
                "429",
                "service unavailable",
                "temporarily unavailable",
                "too many requests",
                "connection reset",
                "connection aborted",
                "timed out",
                "timeout",
                "incomplete read",
            )
        )

    async def _download_with_retry(
        self,
        url: str,
        opts: dict[str, Any],
        *,
        label: str,
        format_fallbacks: list[str] | None = None,
    ) -> None:
        """执行 yt-dlp 下载；瞬时错误时重新解析 URL 后再试。

        format_fallbacks 不为空时，遇到媒体流被拒（503）会依次换用备用格式，
        每次换格式前都等待退避时间，避免短时间内密集请求触发风控。
        """
        attempts = max(1, int(self.cfg.download_retry_times) + 1)
        fallbacks = list(format_fallbacks or [])
        fmt_index = 0
        for attempt in range(1, attempts + 1):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore
                    await to_thread(ydl.download, [url])
                return
            except yt_dlp.utils.DownloadError as exc:
                message = str(exc)
                if attempt < attempts and self._is_transient_error(message):
                    delay = self._retry_backoff(attempt - 1)
                    # 媒体流被拒时优先换一路格式，往往比原样重来更容易成功
                    if fallbacks and self._is_format_rejected(message):
                        fmt = fallbacks.pop(0)
                        opts["format"] = fmt
                        fmt_index += 1
                        logger.warning(
                            f"[下载] {label}媒体流被拒，{delay:.0f}s 后改用格式 "
                            f"{fmt} 重试 ({attempt}/{attempts - 1})"
                        )
                    else:
                        logger.warning(
                            f"[下载] {label}遇到瞬时错误，{delay:.0f}s 后重新解析重试 "
                            f"({attempt}/{attempts - 1}): {message[:200]}"
                        )
                    await sleep(delay)
                    continue

                # 不再把 yt-dlp 的原始异常抛给上层，避免打印整段 traceback；
                # 详细原因留在日志里，发送层会按配置给出下载失败提示。
                suffix = f"，已换用 {fmt_index} 种备用格式" if fmt_index else ""
                logger.warning(
                    f"[下载] {label}最终失败（尝试 {attempt} 次{suffix}）: "
                    f"{message[:300]}"
                )
                raise DownloadException(f"{label}下载失败") from exc

    @staticmethod
    def _looks_like_html(content_type: str, head: bytes) -> bool:
        """判断响应是不是伪装成媒体的 HTML 页面。

        网易云等平台在没有版权/未登录时，会把错误页面的状态码伪装成 200，
        响应头也不带 Content-Length，这些页面不能当媒体文件使用。
        """
        if "text/html" in (content_type or "").lower():
            return True
        sample = head[:512].lstrip().lower()
        return sample.startswith((b"<!doctype html", b"<html", b"<?xml"))

    @auto_task
    async def streamd(
        self,
        url: str,
        *,
        file_name: str | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None | object = ...,
        expected_bytes: int | None = None,
    ) -> Path:
        """流式下载

        Args:
            expected_bytes: 预期的最小体积；实际明显偏小（如音频只下到几十 KB）时
                视为下载到了错误页面，直接判定失败
        """
        if not file_name:
            file_name = generate_file_name(url)
        file_path = self.cfg.cache_dir / file_name
        # 如果文件存在，则直接返回
        if file_path.exists():
            logger.info(f"[下载] 使用缓存 {file_name}")
            return file_path
        headers = headers or self.default_headers
        retries = self.cfg.download_retry_times
        logger.info(f"[下载] 开始 {file_name} <- {url[:200]}")
        for attempt in range(retries + 1):
            try:
                async with self.client.get(
                    url, headers=headers, allow_redirects=True, proxy=proxy
                ) as response:
                    if response.status >= 400:
                        raise ClientError(f"HTTP {response.status} {response.reason}")
                    content_length = response.content_length
                    logger.info(
                        f"[下载] 响应 {file_name}: HTTP {response.status}, "
                        f"size={content_length if content_length is not None else 'unknown'}"
                    )
                    max_bytes = self.max_size * 1024 * 1024

                    if content_length == 0:
                        logger.warning(f"媒体 url: {url}, 大小为 0, 取消下载")
                        raise ZeroSizeException
                    if content_length and content_length > max_bytes:
                        logger.warning(
                            f"媒体 url: {url} 大小 {content_length / 1024 / 1024:.2f} MB 超过 {self.max_size} MB, 取消下载"
                        )
                        raise SizeLimitException

                    downloaded = 0
                    head = b""
                    last_log = time.monotonic()
                    with self.get_progress_bar(file_name, content_length) as bar:
                        async with aiofiles.open(file_path, "wb") as file:
                            async for chunk in response.content.iter_chunked(
                                1024 * 1024
                            ):
                                if len(head) < 512:
                                    head += chunk[: 512 - len(head)]
                                downloaded += len(chunk)
                                if downloaded > max_bytes:
                                    raise SizeLimitException
                                await file.write(chunk)
                                bar.update(len(chunk))
                                now = time.monotonic()
                                if now - last_log >= 5:
                                    logger.info(
                                        f"[下载] 进度 {file_name}: "
                                        f"{downloaded / 1024 / 1024:.2f} MB"
                                    )
                                    last_log = now

                    if downloaded == 0:
                        logger.warning(f"媒体 url: {url}, 实际大小为 0, 取消下载")
                        raise ZeroSizeException
                    if content_length and downloaded < content_length:
                        raise ClientError(
                            f"HTTP payload incomplete {downloaded}/{content_length}"
                        )
                    content_type = response.headers.get("Content-Type", "")
                    if self._looks_like_html(content_type, head):
                        logger.warning(
                            f"媒体 url: {url} 返回的是 HTML 页面"
                            f"(Content-Type={content_type or 'unknown'})，判定为无效媒体"
                        )
                        raise InvalidMediaException(
                            "资源地址返回的是错误页面，可能无版权或需要登录"
                        )
                    if (
                        expected_bytes
                        and downloaded < expected_bytes
                        and not content_length
                    ):
                        logger.warning(
                            f"媒体 url: {url} 实际大小 {downloaded} 字节，"
                            f"明显小于预期 {expected_bytes} 字节，判定为无效媒体"
                        )
                        raise InvalidMediaException(
                            "下载到的文件明显小于预期，可能无版权或需要登录"
                        )

                size_mb = file_path.stat().st_size / 1024 / 1024
                logger.info(f"[下载] 完成 {file_name} ({size_mb:.2f} MB)")
                return file_path
            except (ZeroSizeException, SizeLimitException, InvalidMediaException):
                await safe_unlink(file_path)
                raise
            except (ClientError, TimeoutError) as exc:
                await safe_unlink(file_path)
                if attempt < retries:
                    logger.warning(
                        f"[下载] 重试 {file_name} ({attempt + 1}/{retries}): {exc}"
                    )
                    await sleep(1 + attempt)
                    continue
                logger.exception(f"下载失败 | url: {url}, file_path: {file_path}")
                raise DownloadException("媒体下载失败") from exc
        raise DownloadException("媒体下载失败")

    @staticmethod
    def get_progress_bar(desc: str, total: int | None = None) -> tqdm:
        """获取进度条 bar

        Args:
            desc (str): 描述
            total (int | None): 总大小. Defaults to None.

        Returns:
            tqdm: 进度条
        """
        return tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            colour="green",
            desc=desc,
        )

    @auto_task
    async def download_video(
        self,
        url: str,
        *,
        video_name: str | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
    ) -> Path:
        if video_name is None:
            video_name = generate_file_name(url, ".mp4")
        return await self.streamd(
            url, file_name=video_name, headers=headers, proxy=proxy
        )

    @auto_task
    async def download_audio(
        self,
        url: str,
        *,
        audio_name: str | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        expected_bytes: int | None = None,
    ) -> Path:
        if audio_name is None:
            audio_name = generate_file_name(url, ".mp3")
        return await self.streamd(
            url,
            file_name=audio_name,
            headers=headers,
            proxy=proxy,
            expected_bytes=expected_bytes,
        )

    @auto_task
    async def download_file(
        self,
        url: str,
        *,
        file_name: str | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None | object = ...,
    ) -> Path:
        if file_name is None:
            file_name = generate_file_name(url, ".zip")
        return await self.streamd(
            url, file_name=file_name, headers=headers, proxy=proxy
        )

    @auto_task
    async def download_img(
        self,
        url: str,
        *,
        img_name: str | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None | object = ...,
    ) -> Path:
        if img_name is None:
            img_name = generate_file_name(url, ".jpg")
        return await self.streamd(url, file_name=img_name, headers=headers, proxy=proxy)

    async def download_imgs_without_raise(
        self,
        urls: list[str],
        *,
        headers: dict[str, str] | None = None,
        proxy: str | None | object = ...,
    ) -> list[Path]:
        paths_or_errs = await gather(
            *[self.download_img(url, headers=headers, proxy=proxy) for url in urls],
            return_exceptions=True,
        )
        return [p for p in paths_or_errs if isinstance(p, Path)]

    @auto_task
    async def download_av_and_merge(
        self,
        v_url: str,
        a_url: str,
        *,
        output_path: Path,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
    ) -> Path:
        """
        download video and audio file by url with stream and merge
        """
        v_path, a_path = await gather(
            self.download_video(v_url, headers=headers, proxy=proxy),
            self.download_audio(a_url, headers=headers, proxy=proxy),
        )
        await merge_av(v_path=v_path, a_path=a_path, output_path=output_path)
        return output_path

    async def ytdlp_extract_info(
        self,
        url: str,
        *,
        cookiefile: Path | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        format: str | None = None,
    ) -> VideoInfo:
        if (info := self.info_cache.get(url)) is not None:
            return info
        opts = {
            "quiet": True,
            "skip_download": True,
            # 带 list=RD... 的电台/混音链接会被当成播放列表，
            # 这里强制只解析当前视频
            "noplaylist": True,
            "extract_flat": False,
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if usable_cookie := self._writable_cookiefile(cookiefile):
            opts["cookiefile"] = str(usable_cookie)
        if format:
            opts["format"] = format
        with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore
            raw = await to_thread(ydl.extract_info, url, download=False)
            if not raw:
                raise ParseException("获取视频信息失败")
        info = convert(raw, VideoInfo)
        self.info_cache[url] = info
        return info

    async def ytdlp_extract_raw(
        self,
        url: str,
        *,
        cookiefile: Path | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        opts = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if usable_cookie := self._writable_cookiefile(cookiefile):
            opts["cookiefile"] = str(usable_cookie)
        if format:
            opts["format"] = format

        with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore
            raw = await to_thread(ydl.extract_info, url, download=False)
            if not isinstance(raw, dict):
                raise ParseException("yt-dlp 返回数据异常")
            return raw  # type: ignore

    @auto_task
    async def ytdlp_download_video(
        self,
        url: str,
        *,
        cookiefile: Path | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        format: str | None = None,
        ignore_duration: bool = False,
    ) -> Path:
        info = await self.ytdlp_extract_info(
            url, cookiefile=cookiefile, headers=headers, proxy=proxy
        )
        if info.duration > self.cfg.max_duration and not ignore_duration:
            raise DurationLimitException

        video_path = self.cfg.cache_dir / generate_file_name(url, ".mp4")
        if video_path.exists():
            return video_path

        base_format = format or "best"
        video_format = self._initial_video_format(base_format)
        if video_format != base_format:
            logger.info(f"[下载] 已开启优先单文件格式，改用 {video_format}")

        opts = {
            "outtmpl": str(video_path),
            "merge_output_format": "mp4",
            # "format": f"bv[filesize<={info.duration // 10 + 10}M]+ba/b[filesize<={info.duration // 8 + 10}M]",
            # "format": "bv*[height<=720]+ba/b[height<=720]",
            "format": video_format,
            # 电台/混音链接只下载当前视频，避免把整个播放列表拖下来
            "noplaylist": True,
            "postprocessors": [
                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
            ],
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if usable_cookie := self._writable_cookiefile(cookiefile):
            opts["cookiefile"] = str(usable_cookie)

        await self._download_with_retry(
            url,
            opts,
            label="视频",
            format_fallbacks=self._format_fallbacks(base_format),
        )
        return video_path

    @auto_task
    async def ytdlp_download_video_relaxed(
        self,
        url: str,
        *,
        cookiefile: Path | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        format: str | None = None,
    ) -> Path:
        file_stem = generate_file_name(url)
        video_path = self.cfg.cache_dir / f"{file_stem}.mp4"
        if video_path.exists():
            return video_path

        opts = {
            "outtmpl": str(self.cfg.cache_dir / file_stem) + ".%(ext)s",
            "merge_output_format": "mp4",
            "format": format or None,
            "noplaylist": True,
            "postprocessors": [
                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
            ],
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
            "quiet": True,
            "no_warnings": True,
        }
        if not opts["format"]:
            opts.pop("format")
        if proxy:
            opts["proxy"] = proxy
        if usable_cookie := self._writable_cookiefile(cookiefile):
            opts["cookiefile"] = str(usable_cookie)

        await self._download_with_retry(url, opts, label="视频")
        if video_path.exists():
            return video_path

        candidates = sorted(self.cfg.cache_dir.glob(f"{file_stem}*.mp4"))
        if candidates:
            return candidates[0]
        raise DownloadException("yt-dlp 视频下载失败")

    @auto_task
    async def ytdlp_download_audio(
        self,
        url: str,
        *,
        cookiefile: Path | None,
        headers: dict[str, str] | None = None,
        proxy: str | None = None,
        format: str | None = None,
    ) -> Path:
        file_name = generate_file_name(url)
        audio_path = self.cfg.cache_dir / f"{file_name}.flac"
        if audio_path.exists():
            return audio_path

        opts = {
            "outtmpl": str(self.cfg.cache_dir / file_name) + ".%(ext)s",
            "format": format or "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "flac",
                    "preferredquality": "0",
                }
            ],
            "cookiefile": None,
            "noplaylist": True,
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if usable_cookie := self._writable_cookiefile(cookiefile):
            opts["cookiefile"] = str(usable_cookie)

        await self._download_with_retry(url, opts, label="音频")
        return audio_path
