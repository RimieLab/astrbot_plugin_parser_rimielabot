from asyncio import TimeoutError
from re import Match
from typing import ClassVar

from aiohttp import ClientError

from astrbot.api import logger

from ..config import PluginConfig
from ..cookie import CookieJar
from ..data import Platform
from ..download import Downloader
from ..exception import ParseException
from .base import BaseParser, handle


class NCMParser(BaseParser):
    """网易云音乐解析器"""

    platform: ClassVar[Platform] = Platform(name="ncm", display_name="网易云")

    def __init__(self, config: PluginConfig, downloader: Downloader):
        super().__init__(config, downloader)
        self.headers.update({"Referer": "https://music.163.com"})
        self.mycfg = config.parser.ncm
        self.cookiejar = CookieJar(config, self.mycfg, domain="music.163.com")
        if self.cookiejar.cookies_str:
            self.headers["cookie"] = self.cookiejar.cookies_str

    @handle("163cn.tv", r"163cn\.tv/(?P<short_key>\w+)")
    async def _parse_short(self, searched: Match[str]):
        short_url = f"https://163cn.tv/{searched.group('short_key')}"
        # 让框架跟随 302 后再走通用解析
        return await self.parse_with_redirect(short_url)

    @handle("y.music.163.com", r"y\.music\.163\.com/m/song\?.*id=(?P<song_id>\d+)")
    @handle("music.163.com", r"music\.163\.com/#/song\?.*id=(?P<song_id>\d+)")
    @handle(
        "music.163.com",
        r"music\.163\.com/(?:m/)?song\?[^#\s]*?id=(?P<song_id>\d+)",
    )
    @handle("music.163.com", r"music\.163\.com/song/(?P<song_id>\d+)")
    async def _parse_song(self, searched: Match[str]):
        song_id = searched.group("song_id")
        logger.info(f"[网易云] 开始解析 song_id={song_id}")
        detail_url = (
            f"https://music.163.com/api/song/detail/?id={song_id}&ids=[{song_id}]"
        )
        play_url = f"https://music.163.com/api/song/enhance/player/url?ids=[{song_id}]&br=320000"

        # 1. 取歌曲元数据
        try:
            async with self.session.get(detail_url, headers=self.headers) as resp:
                if resp.status >= 400:
                    raise ClientError(
                        f"[NCM] 获取歌曲信息失败 HTTP {resp.status}"
                    )
                try:
                    detail_json = await resp.json(content_type=None)
                except Exception as exc:
                    raise ParseException(f"详情接口返回非 JSON: {exc}") from exc
        except (ClientError, TimeoutError) as exc:
            logger.warning(f"[网易云] 详情请求失败: {exc}")
            raise ParseException(f"网易云详情请求失败: {exc}") from exc
        if not isinstance(detail_json, dict):
            raise ParseException("详情接口返回格式异常")
        songs = detail_json.get("songs") or []
        logger.info(
            f"[网易云] 详情接口: code={detail_json.get('code')}, "
            f"songs={len(songs)}"
        )

        song = songs[0] if songs and isinstance(songs[0], dict) else {}
        if not song:
            raise ParseException("未找到该歌曲")

        title = str(song.get("name") or "")
        aliases = song.get("alias") or []
        sub_title = str(aliases[0]) if aliases else ""
        album = song.get("album") or {}
        album_name = str(album.get("name") or "") if isinstance(album, dict) else ""
        duration_ms = int(song.get("duration") or 0)
        logger.info(
            f"[网易云] 歌曲: {title or '未知标题'} / {album_name or '未知专辑'}, "
            f"duration={duration_ms}ms"
        )

        # 作者信息
        ar_list = song.get("artists") or []
        author_name = " / ".join(
            str(ar.get("name") or "") for ar in ar_list if isinstance(ar, dict)
        )
        first_artist = ar_list[0] if ar_list and isinstance(ar_list[0], dict) else {}
        author_avatar = str(first_artist.get("img1v1Url") or "")

        # 2. 取播放地址
        try:
            async with self.session.get(play_url, headers=self.headers) as resp:
                if resp.status >= 400:
                    raise ClientError(
                        f"[NCM] 获取播放地址失败 HTTP {resp.status}"
                    )
                try:
                    play_json = await resp.json(content_type=None)
                except Exception as exc:
                    raise ParseException(
                        f"播放接口返回非 JSON: {exc}"
                    ) from exc
        except (ClientError, TimeoutError) as exc:
            logger.warning(f"[网易云] 播放请求失败: {exc}")
            raise ParseException(f"网易云播放请求失败: {exc}") from exc
        if not isinstance(play_json, dict):
            raise ParseException("播放接口返回格式异常")
        logger.info(f"[网易云] 播放接口: code={play_json.get('code')}")
        play_data = play_json.get("data") or []
        play_info = (
            play_data[0] if play_data and isinstance(play_data[0], dict) else {}
        )
        audio_url = str(play_info.get("url") or "")
        if not audio_url:
            logger.warning(
                f"[网易云] 播放接口未返回地址，尝试 outer 直链: {play_json}"
            )
            audio_url = (
                f"https://music.163.com/song/media/outer/url?id={song_id}.mp3"
            )
        logger.info(f"[网易云] 音频地址: {str(audio_url)[:160]}")

        # 3. 组装结果
        author = self.create_author(author_name, author_avatar)
        audio = self.create_audio_content(
            audio_url, duration=duration_ms // 1000
        )
        logger.info(f"[网易云] 已创建音频下载任务, 作者={author_name}")

        # 4. 返回
        return self.result(
            title=f"{title}{'（' + sub_title + '）' if sub_title else ''}",
            text=f"专辑：{album_name}",
            author=author,
            contents=[audio],
            timestamp=None,
            url=f"https://music.163.com/#/song?id={song_id}",
        )

    # 3. 直链 mp3 —— 直接下载
    @handle("music.126.net", r"https?://[^/]*music\.126\.net/.*\.mp3(?:\?.*)?$")
    async def _parse_direct_mp3(self, searched: Match[str]):
        url = searched.group(0)  # 整条 url
        audio = self.create_audio_content(url)
        return self.result(
            title="网易云音乐",
            text="直链音频",
            contents=[audio],
            url=url,
        )

    @handle(
        "music.163.com/song/media/outer/url",
        r"(https?://music\.163\.com/song/media/outer/url\?[^>\s]+)",
    )
    async def _parse_private_outer(self, searched: Match[str]):
        # 整条原始 URL 就是直链
        private_url = searched.group(0)
        audio = self.create_audio_content(private_url)
        return self.result(
            title="网易云音乐（私人直链）",
            text="直链音频",
            contents=[audio],
            url=private_url,
        )
