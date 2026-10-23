import json
from pathlib import Path
from typing import Any


class ProtectedForwardStore:
    """保存用于绕过 QQ 发送拦截的“保护聊天记录”。"""

    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        return self.path.is_file()

    @staticmethod
    def _as_node(message: dict[str, Any]) -> dict[str, Any]:
        """把 get_forward_msg 返回的节点统一成 send_*_forward_msg 需要的格式。"""
        if message.get("type") == "node":
            return message

        data = message.get("data")
        if not isinstance(data, dict):
            data = {}
        sender = message.get("sender")
        if not isinstance(sender, dict):
            sender = data.get("sender")
        if not isinstance(sender, dict):
            sender = {}

        content = message.get("content")
        if content is None:
            content = data.get("content")
        if not isinstance(content, list):
            # 某些实现会直接返回内容段而不是节点，保留下来避免丢失
            if message.get("type") and message.get("type") != "node":
                content = [message]
            else:
                content = []

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

    @classmethod
    def normalize(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [cls._as_node(msg) for msg in messages if isinstance(msg, dict)]

    def count(self) -> int:
        messages = self.load()
        return len(messages) if messages else 0

    def load(self) -> list[dict[str, Any]] | None:
        if not self.path.is_file():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None
        messages = data.get("messages") if isinstance(data, dict) else None
        return self.normalize(messages) if isinstance(messages, list) else None

    def save(self, messages: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"messages": self.normalize(messages)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
