import re
import time
from asyncio import TimeoutError
from itertools import chain
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from aiohttp import ClientError, ClientTimeout
from bs4 import BeautifulSoup, Tag

from astrbot.api import logger

from ..config import PluginConfig
from ..cookie import CookieJar
from ..data import ParseResult, Platform
from ..download import Downloader
from ..exception import ParseException, TipException
from .base import BaseParser, handle


class TwitterParser(BaseParser):
    platform: ClassVar[Platform] = Platform(name="twitter", display_name="推特")

    def __init__(self, config: PluginConfig, downloader: Downloader):
        super().__init__(config, downloader)
        self.mycfg = config.parser.twitter
        self.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "https://xdown.app",
                "Referer": "https://xdown.app/",
            }
        )
        self.xdown_url = "https://xdown.app/api/ajaxSearch"
        self.fx_api = "https://api.fxtwitter.com/status/{tweet_id}"
        self.api_headers = {
            "Accept": "application/json",
            "User-Agent": self.headers.get("User-Agent", ""),
        }
        self.media_headers = {
            "User-Agent": self.headers.get("User-Agent", ""),
            "Referer": "https://x.com/",
        }
        self.cookiejar = CookieJar(config, self.mycfg, domain="xdown.app")
        if self.cookiejar.cookies_str:
            self.headers["cookie"] = self.cookiejar.cookies_str

        # 直链可达性缓存：避免每个链接都等待一次超时
        self._direct_media_ok: bool | None = None
        self._direct_media_checked_at: float = 0.0
        self._direct_media_ttl = 300.0

    async def _req_xdown_api(self, url: str) -> dict[str, Any]:
        async with self.session.post(
            url=self.xdown_url,
            data={"q": url, "lang": "zh-cn"},
            headers=self.headers,
        ) as resp:
            if resp.status >= 400:
                raise ClientError(f"xdown API {resp.status} {resp.reason}")
            return await resp.json()

    async def _req_fxtwitter(self, tweet_id: str) -> dict[str, Any]:
        """请求 FxTwitter 的 JSON 接口，通常比 xdown 的 HTML 抓取快。"""
        timeout = ClientTimeout(total=max(6, min(self.cfg.common_timeout, 12)))
        async with self.session.get(
            self.fx_api.format(tweet_id=tweet_id),
            headers=self.api_headers,
            timeout=timeout,
        ) as resp:
            if resp.status >= 400:
                raise ClientError(f"FxTwitter API {resp.status} {resp.reason}")
            payload = await resp.json(content_type=None)

        if not isinstance(payload, dict) or payload.get("code") != 200:
            message = payload.get("message") if isinstance(payload, dict) else None
            raise ParseException(message or "FxTwitter 解析失败")

        tweet = payload.get("tweet")
        if not isinstance(tweet, dict):
            raise ParseException("FxTwitter 返回数据为空")
        return tweet

    @staticmethod
    def _extract_status_id(url: str) -> str | None:
        if matched := re.search(r"/status/(\d+)", url):
            return matched.group(1)
        return None

    @staticmethod
    def _pick_video_url(item: dict[str, Any]) -> str | None:
        """从 FxTwitter 的 formats/variants 中选码率最高的 MP4。"""
        formats = item.get("formats") or item.get("variants") or []
        best_url: str | None = None
        best_bitrate = -1
        if isinstance(formats, list):
            for fmt in formats:
                if not isinstance(fmt, dict):
                    continue
                url = fmt.get("url")
                if not isinstance(url, str) or not url.startswith("http"):
                    continue
                container = str(fmt.get("container") or "")
                if container == "m3u8" or ".m3u8" in url:
                    continue
                if container and container != "mp4" and ".mp4" not in url:
                    continue
                bitrate = fmt.get("bitrate")
                value = int(bitrate) if isinstance(bitrate, int | float) else 0
                if value > best_bitrate:
                    best_bitrate = value
                    best_url = url

        if best_url:
            return best_url
        url = item.get("url")
        if isinstance(url, str) and url.startswith("http") and ".m3u8" not in url:
            return url
        return None

    @staticmethod
    def _normalize_photo_url(url: str) -> str:
        """推特图片统一取原图（name=orig）。"""
        if "pbs.twimg.com" not in url:
            return url
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        query["name"] = ["orig"]
        return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

    @staticmethod
    def _probe_media_url(tweet: dict[str, Any]) -> str | None:
        """取第一个可用于探测直链可达性的媒体 URL。"""
        media = tweet.get("media") or {}
        items = media.get("all") or []
        if not isinstance(items, list):
            return None

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") in ("video", "gif"):
                url = TwitterParser._pick_video_url(item)
                if url:
                    return url
        for item in items:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url
        return None

    async def _direct_media_available(self, url: str) -> bool:
        now = time.monotonic()
        if (
            self._direct_media_ok is not None
            and now - self._direct_media_checked_at < self._direct_media_ttl
        ):
            return self._direct_media_ok

        ok = False
        try:
            headers = {**self.api_headers, "Range": "bytes=0-0"}
            async with self.session.get(
                url,
                headers=headers,
                timeout=ClientTimeout(total=6),
            ) as resp:
                ok = resp.status < 400
        except (ClientError, TimeoutError) as exc:
            logger.warning(f"Twitter 直链不可达，回退 xdown: {exc}")
        except Exception as exc:
            logger.warning(f"Twitter 直链探测异常，回退 xdown: {exc}")

        self._direct_media_ok = ok
        self._direct_media_checked_at = now
        return ok

    def _build_fx_result(
        self, tweet: dict[str, Any], fallback_url: str
    ) -> ParseResult:
        contents = []
        photo_urls: list[str] = []

        media = tweet.get("media") or {}
        items = media.get("all") or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                media_type = item.get("type")
                if media_type in ("video", "gif"):
                    video_url = self._pick_video_url(item)
                    if not video_url:
                        continue
                    duration = item.get("duration")
                    duration_value = (
                        float(duration) if isinstance(duration, int | float) else 0.0
                    )
                    if media_type == "gif":
                        contents.extend(
                            self.create_dynamic_contents(
                                [video_url], headers=self.media_headers
                            )
                        )
                    else:
                        contents.append(
                            self.create_video_content(
                                video_url,
                                item.get("thumbnail_url"),
                                duration_value,
                                headers=self.media_headers,
                            )
                        )
                elif media_type == "photo":
                    url = item.get("url")
                    if isinstance(url, str) and url:
                        photo_urls.append(self._normalize_photo_url(url))

        if photo_urls:
            contents.extend(
                self.create_image_contents(photo_urls, headers=self.media_headers)
            )

        author_data = tweet.get("author") or {}
        author_name = "无用户名"
        avatar_url = None
        if isinstance(author_data, dict):
            name = author_data.get("name") or author_data.get("screen_name")
            if isinstance(name, str) and name:
                author_name = name
            avatar = author_data.get("avatar_url")
            if isinstance(avatar, str) and avatar:
                avatar_url = avatar

        text = tweet.get("text")
        title = text.strip() if isinstance(text, str) and text.strip() else None
        timestamp = tweet.get("created_timestamp")
        timestamp_value = timestamp if isinstance(timestamp, int) else None
        url = tweet.get("url") if isinstance(tweet.get("url"), str) else fallback_url

        if not contents and title is None:
            raise ParseException("未找到可下载的媒体")

        return self.result(
            title=title,
            author=self.create_author(
                author_name, avatar_url, headers=self.media_headers
            ),
            contents=contents,
            url=url,
            timestamp=timestamp_value,
        )

    async def _parse_via_xdown(self, url: str) -> ParseResult:
        resp = await self._req_xdown_api(url)
        if resp.get("status") != "ok":
            raise ParseException(resp.get("msg") or "解析失败")

        message = resp.get("msg") or ""
        status_code = resp.get("statusCode")
        if status_code == 404 or "未找到" in message:
            raise TipException("该推文无法解析：可能已删除、账号被冻结或受保护")
        if isinstance(status_code, int) and status_code >= 400:
            raise ParseException(message or "解析失败")

        html_content = resp.get("data")
        if html_content is None:
            raise ParseException("解析失败, 数据为空")

        return self.parse_twitter_html(html_content)

    @handle(
        "twitter.com",
        (
            r"(?<![A-Za-z0-9.-])(?:(?:www|mobile)\.)?twitter\.com/"
            r"(?:[A-Za-z0-9_]+/)*status/\d+"
        ),
    )
    @handle(
        "x.com",
        (
            r"(?<![A-Za-z0-9.-])(?:www\.)?x\.com/"
            r"(?:[A-Za-z0-9_]+/)*status/\d+"
        ),
    )
    async def _parse(self, searched: re.Match[str]) -> ParseResult:
        # 从匹配对象中获取原始URL
        url = f"https://{searched.group(0)}"
        tweet_id = self._extract_status_id(url)

        # 优先 FxTwitter：服务端可直连 twimg 时走官方媒体 CDN，
        # 直链不可达（如无代理的国内网络）再自动回退 xdown。
        cached_ok = (
            self._direct_media_ok is not False
            or time.monotonic() - self._direct_media_checked_at
            >= self._direct_media_ttl
        )
        if tweet_id and cached_ok:
            try:
                tweet = await self._req_fxtwitter(tweet_id)
                probe_url = self._probe_media_url(tweet)
                if probe_url is None:
                    # 纯文字推文，没有媒体可探测
                    return self._build_fx_result(tweet, url)
                if await self._direct_media_available(probe_url):
                    return self._build_fx_result(tweet, url)
                logger.info("Twitter 直链不可用，使用 xdown 解析")
            except (ClientError, TimeoutError, ParseException, ValueError) as exc:
                logger.warning(f"FxTwitter 解析失败，回退 xdown: {exc}")

        return await self._parse_via_xdown(url)

    def parse_twitter_html(self, html_content: str) -> ParseResult:
        """解析 Twitter HTML 内容

        Args:
            html_content (str): Twitter HTML 内容

        Returns:
            ParseResult: 解析结果
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # 初始化数据
        title = None
        cover_url = None
        video_url = None
        images_urls = []
        dynamic_urls = []

        # 1. 提取缩略图链接
        thumb_tag = soup.find("img")
        if isinstance(thumb_tag, Tag):
            if cover := thumb_tag.get("src"):
                cover_url = str(cover)

        # 2. 提取下载链接
        tw_button_tags = soup.find_all("a", class_="tw-button-dl")
        abutton_tags = soup.find_all("a", class_="abutton")
        for tag in chain(tw_button_tags, abutton_tags):
            if not isinstance(tag, Tag):
                continue
            href = tag.get("href")
            if href is None:
                continue

            href = str(href)
            text = tag.get_text(strip=True)
            if "下载 MP4" in text:
                video_url = href
                break
            elif "下载图片" in text:
                images_urls.append(href)
            elif "下载 gif" in text:
                dynamic_urls.append(href)

        # 3. 提取标题
        title_tag = soup.find("h3")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # 简洁的构建方式
        contents = []

        # 添加视频内容
        if video_url:
            contents.append(self.create_video_content(video_url, cover_url))

        # 添加图片内容
        if images_urls:
            contents.extend(self.create_image_contents(images_urls))

        # 添加动态内容
        if dynamic_urls:
            contents.extend(self.create_dynamic_contents(dynamic_urls))

        return self.result(
            title=title,
            author=self.create_author("无用户名"),
            contents=contents,
        )
        # # 4. 提取Twitter ID
        # twitter_id_input = soup.find("input", {"id": "TwitterId"})
        # if (
        #     twitter_id_input
        #     and isinstance(twitter_id_input, Tag)
        #     and (value := twitter_id_input.get("value"))
        #     and isinstance(value, str)
        # ):
