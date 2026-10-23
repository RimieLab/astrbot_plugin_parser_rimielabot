import json
from pathlib import Path
from typing import Any


class ProtectedForwardStore:
    """保存用于绕过 QQ 发送拦截的“保护聊天记录”。"""

    def __init__(self, path: Path):
        self.path = path

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _as_node(message: dict[str, Any]) -> dict[str, Any]:
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

    def exists(self) -> bool:
        return bool(self.load_id())

    def count(self) -> int:
        return 1 if self.load_id() else 0

    def load_id(self) -> str | None:
        forward_id = self._read().get("id")
        return str(forward_id) if forward_id else None

    def load_messages(self) -> list[dict[str, Any]] | None:
        messages = self._read().get("messages")
        return self.normalize(messages) if isinstance(messages, list) else None

    def save(
        self,
        forward_id: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"id": str(forward_id)}
        if messages:
            data["messages"] = self.normalize(messages)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
