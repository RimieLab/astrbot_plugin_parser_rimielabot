import asyncio
import shutil

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from astrbot.api import logger

from .config import PluginConfig


class CacheCleaner:
    """
    每天固定时间自动清理插件缓存目录的调度器封装。
    """

    JOBNAME = "CacheCleaner"

    def __init__(self, config: PluginConfig):
        self.cfg = config
        self.scheduler = AsyncIOScheduler(timezone=self.cfg.timezone)
        self.scheduler.start()

        self.register_task()

        logger.info(f"{self.JOBNAME} 已启动，任务周期：{self.cfg.clean_cron}")

    def register_task(self):
        try:
            self.trigger = CronTrigger.from_crontab(self.cfg.clean_cron)
            self.scheduler.add_job(
                func=self._clean_plugin_cache,
                trigger=self.trigger,
                name=f"{self.JOBNAME}_scheduler",
                max_instances=1,
            )
        except Exception as e:
            logger.error(f"[{self.JOBNAME}] Cron 格式错误：{e}")

    async def _clean_plugin_cache(self) -> None:
        """删除并重建缓存目录（定时任务用）"""
        try:
            files, size_mb = await self.clean_now()
            logger.info(
                f"[{self.JOBNAME}] 缓存已清理：文件 {files} 个，释放 {size_mb:.2f} MB"
            )
        except Exception:
            logger.exception("Error while cleaning cache directory.")

    async def clean_now(self) -> tuple[int, float]:
        """立即清理缓存并返回清理统计。

        Returns:
            ``(文件数, 释放空间 MB)``
        """
        loop = asyncio.get_running_loop()
        cache_dir = self.cfg.cache_dir

        def _collect() -> tuple[int, int]:
            if not cache_dir.exists():
                return 0, 0
            count = total = 0
            for path in cache_dir.rglob("*"):
                if path.is_file():
                    try:
                        total += path.stat().st_size
                        count += 1
                    except OSError:
                        continue
            return count, total

        files, size = await loop.run_in_executor(None, _collect)
        await loop.run_in_executor(None, shutil.rmtree, cache_dir, True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{self.JOBNAME}] 已清理缓存目录 {cache_dir}")
        return files, size / 1024 / 1024

    async def stop(self):
        self.scheduler.remove_all_jobs()
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        logger.info(f"[{self.JOBNAME}] 已停止")
