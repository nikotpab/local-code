from __future__ import annotations

import threading
from typing import Callable, Any

from textual.app import App


class ThreadSafeConfirmationBridge:
    """Bridges synchronous confirm() calls from a worker thread to a Textual modal on the UI thread."""

    def __init__(self, app: App, show_modal_fn: Callable[[str, str, Callable[[str], None]], None]):
        self.app = app
        self._show_modal_fn = show_modal_fn

    def confirm(self, name: str, preview: str) -> str:
        event = threading.Event()
        result: list[str] = ["no"]

        def on_decision(choice: str) -> None:
            if choice in ("yes", "no", "always"):
                result[0] = choice
            else:
                result[0] = "no"
            event.set()

        def request_ui() -> None:
            try:
                self._show_modal_fn(name, preview, on_decision)
            except Exception:
                result[0] = "no"
                event.set()

        self.app.call_from_thread(request_ui)
        # Block the worker thread until user responds on UI thread
        event.wait()
        return result[0]
