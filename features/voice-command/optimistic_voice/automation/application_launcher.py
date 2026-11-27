"""Cross-platform application launcher for Windows and Raspberry Pi OS."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import webbrowser

from optimistic_voice.automation import ExecutionResult


class ApplicationLauncher:
    """Launches installed desktop applications according to the host OS."""

    def __init__(self) -> None:
        self.os_name = platform.system()

    def launch(self, app_id: str) -> ExecutionResult:
        key = app_id.strip().lower()
        if key == "browser":
            try:
                webbrowser.open("about:blank")
                return ExecutionResult(success=True, message="Opened browser")
            except Exception as exc:
                return ExecutionResult(success=False, message=f"Failed to open browser: {exc}")

        candidates = self._get_candidates(key)
        for cmd in candidates:
            executable = cmd[0]
            if self._is_available(executable):
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=(self.os_name != "Windows"),
                    )
                    return ExecutionResult(success=True, message=f"Launched {app_id}")
                except Exception:
                    continue

        # Fallback: try launching by raw name if it's on PATH
        if shutil.which(key):
            try:
                subprocess.Popen([key], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return ExecutionResult(success=True, message=f"Launched {key}")
            except Exception as exc:
                return ExecutionResult(success=False, message=f"Failed to launch {key}: {exc}")

        return ExecutionResult(success=False, message=f"{app_id} is not installed on this system.")

    def _is_available(self, executable: str) -> bool:
        if os.path.isabs(executable):
            return os.path.exists(executable)
        return shutil.which(executable) is not None

    def _get_candidates(self, key: str) -> list[list[str]]:
        if self.os_name == "Windows":
            mapping: dict[str, list[list[str]]] = {
                "notepad": [["notepad.exe"]],
                "calculator": [["calc.exe"]],
                "file_manager": [["explorer.exe"]],
                "terminal": [["wt.exe"], ["cmd.exe"]],
            }
            return mapping.get(key, [[f"{key}.exe"], [key]])

        # Raspberry Pi OS / Debian / Linux
        mapping = {
            "notepad": [["mousepad"], ["gedit"], ["leafpad"], ["featherpad"], ["xed"]],
            "calculator": [["galculator"], ["gnome-calculator"], ["kcalc"], ["xcalc"]],
            "file_manager": [["pcmanfm"], ["thunar"], ["nautilus"], ["nemo"], ["xdg-open", os.path.expanduser("~")]],
            "terminal": [["lxterminal"], ["x-terminal-emulator"], ["gnome-terminal"], ["xfce4-terminal"]],
        }
        return mapping.get(key, [[key]])
