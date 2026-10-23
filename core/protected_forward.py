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

    def exists(self) -> bool:
        return bool(self.load_id())

    def count(self) -> int:
        return 1 if self.load_id() else 0

    def load_id(self) -> str | None:
        forward_id = self._read().get("id")
        return str(forward_id) if forward_id else None

    def load_messages(self) -> list[dict[str, Any]] | None:
        messages = self._read().get("messages")
        return messages if isinstance(messages, list) else None

    def save(
        self,
        forward_id: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"id": str(forward_id)}
        if messages:
            data["messages"] = messages
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
