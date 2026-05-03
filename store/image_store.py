from __future__ import annotations

from threading import Lock


class ImageStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._images: dict[str, bytes] = {}

    def set(self, session_id: str, raw_bytes: bytes) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        if not raw_bytes:
            raise ValueError("raw_bytes is required")

        with self._lock:
            self._images[session_id] = raw_bytes

    def get(self, session_id: str) -> bytes | None:
        with self._lock:
            return self._images.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._images.pop(session_id, None)


image_store = ImageStore()
