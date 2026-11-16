import os
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
    def _retry_backoff(attempt: int) -> float:
        """第 attempt 次重试前的等待秒数（指数退避，最多 10 秒）"""
        return float(min(2**attempt, 10))

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
        self, url: str, opts: dict[str, Any], *, label: str
    ) -> None:
        """执行 yt-dlp 下载；瞬时错误时重新解析 URL 后再试。"""
        attempts = max(1, int(self.cfg.download_retry_times) + 1)
        for attempt in range(1, attempts + 1):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore
                    await to_thread(ydl.download, [url])
                return
            except yt_dlp.utils.DownloadError as exc:
                if attempt >= attempts or not self._is_transient_error(str(exc)):
                    raise
                delay = self._retry_backoff(attempt - 1)
                logger.warning(
                    f"[下载] {label}遇到瞬时错误，{delay:.0f}s 后重新解析重试 "
                    f"({attempt}/{attempts - 1}): {str(exc)[:200]}"
                )
                await sleep(delay)

    @auto_task
    async def streamd(
        self,
        url: str,
        *,
        file_name: str | None = None,
        headers: dict[str, str] | None = None,
        proxy: str | None | object = ...,
    ) -> Path:
        """流式下载"""
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
                    last_log = time.monotonic()
                    with self.get_progress_bar(file_name, content_length) as bar:
                        async with aiofiles.open(file_path, "wb") as file:
                            async for chunk in response.content.iter_chunked(
                                1024 * 1024
                            ):
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

                size_mb = file_path.stat().st_size / 1024 / 1024
                logger.info(f"[下载] 完成 {file_name} ({size_mb:.2f} MB)")
                return file_path
            except (ZeroSizeException, SizeLimitException):
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
    ) -> Path:
        if audio_name is None:
            audio_name = generate_file_name(url, ".mp3")
        return await self.streamd(
            url, file_name=audio_name, headers=headers, proxy=proxy
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
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if cookiefile and cookiefile.is_file():
            opts["cookiefile"] = str(cookiefile)
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
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if cookiefile and cookiefile.is_file():
            opts["cookiefile"] = str(cookiefile)
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
    ) -> Path:
        info = await self.ytdlp_extract_info(
            url, cookiefile=cookiefile, headers=headers, proxy=proxy
        )
        if info.duration > self.cfg.max_duration:
            raise DurationLimitException

        video_path = self.cfg.cache_dir / generate_file_name(url, ".mp4")
        if video_path.exists():
            return video_path

        opts = {
            "outtmpl": str(video_path),
            "merge_output_format": "mp4",
            # "format": f"bv[filesize<={info.duration // 10 + 10}M]+ba/b[filesize<={info.duration // 8 + 10}M]",
            # "format": "bv*[height<=720]+ba/b[height<=720]",
            "format": format or "best",
            "postprocessors": [
                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
            ],
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if cookiefile and cookiefile.is_file():
            opts["cookiefile"] = str(cookiefile)

        await self._download_with_retry(url, opts, label="视频")
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
        if cookiefile and cookiefile.is_file():
            opts["cookiefile"] = str(cookiefile)

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
            "http_headers": headers or self.default_headers,
            "js_runtimes": js_runtime_opts(),
            **self._retry_opts(),
        }
        if proxy:
            opts["proxy"] = proxy
        if cookiefile and cookiefile.is_file():
            opts["cookiefile"] = str(cookiefile)

        await self._download_with_retry(url, opts, label="音频")
        return audio_path
