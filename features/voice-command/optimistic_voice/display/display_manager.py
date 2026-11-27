"""Smart-glasses visual feedback display manager (HUD + console + callback)."""

from __future__ import annotations

import queue
import threading
from typing import Callable


class DisplayManager:
    """Displays Optimistic status and messages on the smart glasses screen."""

    def __init__(
        self,
        mode: str = "auto",
        width: int = 480,
        height: int = 240,
        on_update: Callable[[str, str], None] | None = None,
    ) -> None:
        self.requested_mode = mode.lower()
        self.width = width
        self.height = height
        self.on_update = on_update
        self.current_status: str = ""
        self.current_message: str = ""
        self.history: list[tuple[str, str]] = []

        self._ui_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._tk_thread: threading.Thread | None = None
        self.active_mode = self._resolve_mode()

        if self.active_mode == "tkinter":
            self._start_tk_window()

    def _resolve_mode(self) -> str:
        if self.requested_mode in ("headless", "console"):
            return self.requested_mode
        if self.requested_mode in ("auto", "tkinter"):
            try:
                import tkinter  # noqa: F401
                import os

                if os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                    return "tkinter"
            except Exception:
                pass
        return "console"

    def _start_tk_window(self) -> None:
        def run_tk() -> None:
            try:
                import tkinter as tk

                root = tk.Tk()
                root.title("Optimistic Smart Glasses HUD")
                root.geometry(f"{self.width}x{self.height}+40+40")
                root.configure(bg="#05070A")
                root.attributes("-topmost", True)

                header = tk.Label(
                    root,
                    text="OPTIMISTIC",
                    fg="#00E5FF",
                    bg="#05070A",
                    font=("Segoe UI", 12, "bold"),
                )
                header.pack(pady=(16, 4))

                status_var = tk.StringVar(value="Starting...")
                msg_var = tk.StringVar(value="")

                status_lbl = tk.Label(
                    root,
                    textvariable=status_var,
                    fg="#A8DADC",
                    bg="#05070A",
                    font=("Segoe UI", 14, "bold"),
                    wraplength=self.width - 32,
                )
                status_lbl.pack(pady=(8, 6))

                msg_lbl = tk.Label(
                    root,
                    textvariable=msg_var,
                    fg="#FFFFFF",
                    bg="#05070A",
                    font=("Segoe UI", 13),
                    wraplength=self.width - 32,
                    justify="center",
                )
                msg_lbl.pack(pady=(4, 16), padx=16)

                def poll_queue() -> None:
                    if self._stop_event.is_set():
                        root.destroy()
                        return
                    while not self._ui_queue.empty():
                        try:
                            status, msg = self._ui_queue.get_nowait()
                            status_var.set(status)
                            msg_var.set(msg)
                        except queue.Empty:
                            break
                    root.after(80, poll_queue)

                root.after(80, poll_queue)
                root.mainloop()
            except Exception:
                self.active_mode = "console"

        self._tk_thread = threading.Thread(target=run_tk, name="OptimisticDisplay", daemon=True)
        self._tk_thread.start()

    def show(self, status: str, message: str = "") -> None:
        """Update the smart-glasses display with a concise status and message."""
        self.current_status = status
        self.current_message = message
        self.history.append((status, message))

        if self.on_update:
            self.on_update(status, message)

        if self.active_mode == "tkinter":
            self._ui_queue.put((status, message))
        elif self.active_mode == "console":
            line = f"[OPTIMISTIC DISPLAY] {status}" + (f" | {message}" if message else "")
            print(line, flush=True)

    def close(self) -> None:
        self._stop_event.set()
        if self._tk_thread and self._tk_thread.is_alive():
            self._tk_thread.join(timeout=1.0)
