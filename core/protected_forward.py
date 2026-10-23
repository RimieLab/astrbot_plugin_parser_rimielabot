import json
from pathlib import Path
from typing import Any


class ProtectedForwardStore:
    """保存用于绕过 QQ 发送拦截的“保护聊天记录”。"""

    def __init__(self, path: Path):
        self.path = path

    def exists(self) -> bool:
        return self.path.is_file()

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
        return messages if isinstance(messages, list) else None

    def save(self, messages: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"messages": messages}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
