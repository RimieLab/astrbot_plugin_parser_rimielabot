import asyncio
import ctypes
import gc
import hashlib
import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import unquote, urlparse

from astrbot.api import logger

K = TypeVar("K")
V = TypeVar("V")

_FFMPEG_LOCK: asyncio.Semaphore | None = None
_FFMPEG_LOCK_LOOP: asyncio.AbstractEventLoop | None = None


def _ffmpeg_lock() -> asyncio.Semaphore:
    """串行化 ffmpeg，避免多条解析同时启动导致 CPU/内存尖峰。"""
    global _FFMPEG_LOCK, _FFMPEG_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _FFMPEG_LOCK is None or _FFMPEG_LOCK_LOOP is not loop:
        _FFMPEG_LOCK = asyncio.Semaphore(1)
        _FFMPEG_LOCK_LOOP = loop
    return _FFMPEG_LOCK


class LimitedSizeDict(OrderedDict[K, V]):
    """
    定长字典
    """

    def __init__(self, *args, max_size=20, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: K, value: V):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)  # 移除最早添加的项


async def safe_unlink(path: Path):
    """
    安全删除文件
    """
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except Exception:
        logger.warning(f"删除 {path} 失败")


async def exec_ffmpeg_cmd(cmd: list[str]) -> None:
    """执行 ffmpeg 命令，串行化并确保子进程被回收。

    Args:
        cmd (list[str]): 命令序列
    """
    process = None
    async with _ffmpeg_lock():
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=300
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise RuntimeError("ffmpeg 执行超时（300 秒）")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg 未安装或无法找到可执行文件")
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

        if process.returncode != 0:
            error_msg = (stderr or b"").decode(errors="ignore").strip()
            raise RuntimeError(f"ffmpeg 执行失败: {error_msg}")
    await asyncio.to_thread(release_memory)


def has_ffmpeg() -> bool:
    """当前运行环境是否能找到 ffmpeg。"""
    return shutil.which("ffmpeg") is not None


def release_memory() -> None:
    """回收 Python 侧不再使用的内存（glibc 环境额外 malloc_trim）。"""
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


async def video_to_gif(
    video_path: Path,
    *,
    width: int = 640,
    fps: int = 15,
) -> Path | None:
    """把视频/动图 mp4 转成真正的 GIF，便于在合并转发中作为图片发送。"""
    gif_path = video_path.with_suffix(".gif")
    if gif_path.exists() and gif_path.stat().st_size > 0:
        return gif_path

    if not has_ffmpeg():
        logger.warning("未找到 ffmpeg，无法把动图转换为 GIF")
        return None

    logger.info(f"[GIF] 开始转换 {video_path.name} -> {gif_path.name}")
    # 单次解码完成取色板和编码；限制线程数，避免瞬时打满 CPU。
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
        "-i",
        str(video_path),
        "-filter_complex",
        (
            f"[0:v]fps={fps},scale={width}:-1:flags=lanczos,"
            "split[a][b];[a]palettegen[p];[b][p]paletteuse"
        ),
        "-loop",
        "0",
        "-an",
        str(gif_path),
    ]
    try:
        await exec_ffmpeg_cmd(cmd)
    except Exception as e:
        logger.warning(f"视频转 GIF 失败: {video_path.name}, {e}")
        await safe_unlink(gif_path)
        return None

    if gif_path.exists() and gif_path.stat().st_size > 0:
        logger.info(f"[GIF] 转换完成 {gif_path.name} ({gif_path.stat().st_size} bytes)")
        return gif_path
    return None


async def merge_av(
    *,
    v_path: Path,
    a_path: Path,
    output_path: Path,
) -> None:
    """合并视频和音频

    Args:
        v_path (Path): 视频文件路径
        a_path (Path): 音频文件路径
        output_path (Path): 输出文件路径
    """
    target_path = output_path
    if output_path in (v_path, a_path):
        output_path = output_path.with_name(
            f"{output_path.stem}_merged{output_path.suffix}"
        )
    logger.info(f"Merging {v_path.name} and {a_path.name} to {output_path.name}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(v_path),
        "-i",
        str(a_path),
        "-c",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(output_path),
    ]

    await exec_ffmpeg_cmd(cmd)
    if output_path != target_path:
        await safe_unlink(target_path)
        await asyncio.to_thread(output_path.replace, target_path)
        output_path = target_path
    cleanup = [p for p in (v_path, a_path) if p != output_path]
    await asyncio.gather(*(safe_unlink(p) for p in cleanup))
    logger.info(f"Merged {output_path.name}, {fmt_size(output_path)}")


async def merge_av_h264(
    *,
    v_path: Path,
    a_path: Path,
    output_path: Path,
) -> None:
    """合并视频和音频，并使用 H.264 编码

    Args:
        v_path (Path): 视频文件路径
        a_path (Path): 音频文件路径
        output_path (Path): 输出文件路径
    """
    logger.info(
        f"Merging {v_path.name} and {a_path.name} to {output_path.name} with H.264"
    )

    # 修改命令以确保视频使用 H.264 编码
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-threads",
        "2",
        "-i",
        str(v_path),
        "-i",
        str(a_path),
        "-c:v",
        "libx264",  # 明确指定使用 H.264 编码
        "-preset",
        "medium",  # 编码速度和质量的平衡
        "-crf",
        "23",  # 质量因子，值越低质量越高
        "-c:a",
        "aac",  # 音频使用 AAC 编码
        "-b:a",
        "128k",  # 音频比特率
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(output_path),
    ]

    await exec_ffmpeg_cmd(cmd)
    await asyncio.gather(safe_unlink(v_path), safe_unlink(a_path))
    logger.info(f"Merged {output_path.name} with H.264, {fmt_size(output_path)}")


async def encode_video_to_h264(video_path: Path) -> Path:
    """将视频重新编码到 h264

    Args:
        video_path (Path): 视频路径

    Returns:
        Path: 编码后的视频路径
    """
    output_path = video_path.with_name(f"{video_path.stem}_h264{video_path.suffix}")
    if output_path.exists():
        return output_path
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-threads",
        "2",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        str(output_path),
    ]
    await exec_ffmpeg_cmd(cmd)
    logger.info(f"视频重新编码为 H.264 成功: {output_path}, {fmt_size(output_path)}")
    await safe_unlink(video_path)
    return output_path


async def transcode_video_audio_to_lc(path: Path) -> Path:
    """把视频中非 AAC-LC 的音频转码为标准 AAC-LC，保持视频流 copy。

    用于解决 QQ 等平台因音频 profile（如 HE-AACv2）导致的视频无法正常渲染问题。
    """
    try:
        probe = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=profile",
            "-of",
            "default=nw=1:nk=1",
            str(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await probe.communicate()
        profile = (out or b"").decode(errors="ignore").strip()
    except Exception as e:
        logger.warning(f"[视频] ffprobe 探测失败，跳过转码: {e}")
        return path

    if not profile or profile.lower() in ("lc", "aac_lc", "aac lc"):
        return path

    out_path = path.with_name(f"{path.stem}_lc.mp4")
    proc = None
    err: bytes = b""
    async with _ffmpeg_lock():
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",
                "-nostdin",
                "-threads",
                "2",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-profile:a",
                "aac_low",
                "-movflags",
                "+faststart",
                str(out_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, err = await asyncio.wait_for(proc.communicate(), timeout=300)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("[视频] ffmpeg 转码超时，改用原文件")
                return path
        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        except Exception as e:
            logger.warning(f"[视频] ffmpeg 调用异常，跳过转码: {e}")
            return path
        finally:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()

    await asyncio.to_thread(release_memory)
    if proc.returncode != 0 or not out_path.exists():
        logger.warning(
            f"[视频] 音频转码失败(profile={profile})，改用原文件: "
            f"{(err or b'')[:200]!r}"
        )
        return path

    logger.info(
        f"[视频] 音频 {profile} -> AAC-LC 完成: {out_path.name} "
        f"({out_path.stat().st_size} bytes)"
    )
    await safe_unlink(path)
    return out_path


VIDEO_CODECS: dict[str, tuple[str, str, list[str]]] = {
    "h264": (
        "libx264",
        ".mp4",
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
        ],
    ),
    "h265": (
        "libx265",
        ".mp4",
        ["-c:v", "libx265", "-preset", "fast", "-tag:v", "hvc1"],
    ),
    "av1": (
        "libsvtav1",
        ".mp4",
        ["-c:v", "libsvtav1", "-preset", "8"],
    ),
}


def video_codec_available(codec: str) -> bool:
    """ffmpeg 是否自带该编码器。"""
    info = VIDEO_CODECS.get(codec)
    if info is None:
        return False
    try:
        import subprocess

        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return info[0] in (out.stdout or "")
    except Exception:
        return False


async def probe_video_info(path: Path) -> tuple[float, int, int]:
    """返回 (时长秒, 宽, 高)；探测失败时返回 (0, 0, 0)。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "default=nw=1",
            str(path),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
    except Exception as e:
        logger.warning(f"[压缩] ffprobe 探测失败: {e}")
        return 0.0, 0, 0
    duration = 0.0
    width = 0
    height = 0
    for line in (out or b"").decode(errors="ignore").splitlines():
        key, _, value = line.partition("=")
        value = value.strip()
        if not value:
            continue
        if key == "duration":
            try:
                duration = float(value)
            except ValueError:
                pass
        elif key == "width":
            try:
                width = int(value)
            except ValueError:
                pass
        elif key == "height":
            try:
                height = int(value)
            except ValueError:
                pass
    return duration, width, height


async def compress_video(
    video_path: Path,
    *,
    threshold_mb: float = 10,
    max_height: int = 1080,
    target_mb: float = 10,
    codec: str = "h265",
    audio_bitrate: str = "96k",
) -> Path:
    """把超过阈值的视频压到目标体积以内，优先降码率，分辨率仅封顶。

    压缩失败或压缩后反而更大时返回原文件。
    """
    try:
        size = video_path.stat().st_size
    except OSError:
        return video_path
    if size <= threshold_mb * 1024 * 1024:
        return video_path
    if not has_ffmpeg():
        logger.warning("[压缩] 未找到 ffmpeg，跳过视频压缩")
        return video_path

    codec = (codec or "h265").lower()
    if codec not in VIDEO_CODECS:
        logger.warning(f"[压缩] 未知编码 {codec}，改用 h265")
        codec = "h265"
    if not video_codec_available(codec):
        for fallback in ("h265", "h264"):
            if video_codec_available(fallback):
                logger.warning(
                    f"[压缩] ffmpeg 不支持 {codec}，改用 {fallback}"
                )
                codec = fallback
                break
        else:
            logger.warning("[压缩] ffmpeg 无可用编码器，跳过压缩")
            return video_path

    duration, width, height = await probe_video_info(video_path)
    if duration <= 0:
        logger.warning("[压缩] 无法获取视频时长，跳过压缩")
        return video_path

    target_bytes = max(int(target_mb * 1024 * 1024), 512 * 1024)
    audio_kbps = 96
    total_kbps = target_bytes * 8 / duration / 1000
    video_kbps = int(total_kbps - audio_kbps)
    if video_kbps < 150:
        # 码率太低画质崩坏，改为降低分辨率后重算
        height = min(height or max_height, 720)
        video_kbps = int(max(total_kbps * 0.85, 150))
    video_kbps = min(video_kbps, 20000)
    max_height = max(min(max_height, height or max_height), 240)
    if height and height > max_height:
        height = max_height

    _, ext, codec_args = VIDEO_CODECS[codec]
    # 等比缩放到目标高度（不放大），并保证宽高为偶数
    vf = (
        f"scale=-2:'min({height},ih)'"
        if height
        else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    output = video_path.with_name(
        f"{video_path.stem}_c{codec}{ext}"
    )
    if output.exists() and output.stat().st_size > 0:
        return output

    logger.info(
        f"[压缩] {video_path.name} {size / 1024 / 1024:.2f}MB -> "
        f"{codec} {video_kbps}kbps {height}p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        *codec_args,
        "-b:v",
        f"{video_kbps}k",
        "-maxrate",
        f"{int(video_kbps * 1.2)}k",
        "-bufsize",
        f"{video_kbps * 2}k",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        await exec_ffmpeg_cmd(cmd)
    except Exception as e:
        logger.warning(f"[压缩] 失败，改用原视频: {e}")
        await safe_unlink(output)
        return video_path

    if not output.exists() or output.stat().st_size <= 0:
        return video_path
    if output.stat().st_size >= size:
        logger.info("[压缩] 压缩后未变小，改用原视频")
        await safe_unlink(output)
        return video_path
    logger.info(
        f"[压缩] 完成 {output.name}: "
        f"{output.stat().st_size / 1024 / 1024:.2f}MB"
    )
    return output


def fmt_size(file_path: Path) -> str:
    """格式化文件大小

    Args:
        video_path (Path): 视频路径
    """
    return f"大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB"


def generate_file_name(url: str, default_suffix: str = "") -> str:
    """根据 url 生成文件名

    Args:
        url (str): url
        default_suffix (str): 默认后缀. Defaults to "".

    Returns:
        str: 文件名
    """
    # 根据 url 获取文件后缀
    path = Path(urlparse(url).path)
    suffix = path.suffix if path.suffix else default_suffix
    # 获取 url 的 md5 值
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    file_name = f"{url_hash}{suffix}"
    return file_name


URL_RE = re.compile(r"https?://[^\s\"'<>\\]+")


def _iter_json_string_values(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_json_string_values(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_json_string_values(item)
    elif isinstance(obj, str):
        yield obj


def _clean_embedded_url(value: str) -> str:
    return unquote(value.strip().replace("\\/", "/"))


def extract_json_url(data: dict | str) -> str | None:
    """处理 JSON 类型消息段，提取可交给解析器处理的 URL。"""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            return None

    if not isinstance(data, dict):
        return None

    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    # 优先读取 QQ 卡片中明确表达跳转目标的字段。
    priority_paths = (
        ("miniapp", "legacyUrl"),
        ("miniapp", "pcJumpUrl"),
        ("miniapp", "jumpUrl"),
        ("miniapp", "sourceUrl"),
        ("miniapp", "url"),
        ("detail_1", "qqdocurl"),
        ("detail_1", "jumpUrl"),
        ("detail_1", "url"),
        ("detail_1", "sourceUrl"),
        ("detail_1", "shareUrl"),
        ("detail_1", "targetUrl"),
        ("news", "jumpUrl"),
        ("news", "url"),
        ("news", "sourceUrl"),
        ("news", "shareUrl"),
        ("music", "musicUrl"),
        ("music", "jumpUrl"),
        ("music", "url"),
    )
    for key1, key2 in priority_paths:
        node = meta.get(key1)
        if not isinstance(node, dict):
            continue
        value = node.get(key2)
        if not isinstance(value, str):
            continue
        match = URL_RE.search(_clean_embedded_url(value))
        if match:
            return match.group(0)

    # 部分 OneBot/QQ 卡片会把真实 URL 藏在更深层字段中。
    candidates: list[str] = []
    for value in _iter_json_string_values(data):
        cleaned = _clean_embedded_url(value)
        candidates.extend(match.group(0) for match in URL_RE.finditer(cleaned))

    if not candidates:
        return None

    # QZone 分享卡优先，之后保持常见分享短链优先级。
    preferred_keywords = (
        "mobile.qzone.qq.com/l",
        "h5.qzone.qq.com/ugc/share",
        "m.qzone.qq.com",
        "user.qzone.qq.com",
        "qzone.qq.com",
        "b23.tv",
        "bili2233.cn",
        "bilibili.com",
        "xhslink.cn",
        "xhslink.com",
        "xiaohongshu.com",
    )
    for keyword in preferred_keywords:
        for url in candidates:
            if keyword in url:
                return url

    return candidates[0]
