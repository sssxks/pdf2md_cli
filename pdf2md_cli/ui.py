from __future__ import annotations

import shutil
import sys
import threading
import time
from dataclasses import dataclass


@dataclass
class Spinner:
    """A simple spinner that writes transient progress to stderr."""

    enabled: bool = True
    _glyphs: str = "|/-\\"

    def __post_init__(self) -> None:
        self.enabled = self.enabled and sys.stderr.isatty()
        self._msg = ""
        self._running = False
        self._lock = threading.Lock()
        self._t: threading.Thread | None = None

    def start(self, msg: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._msg = msg
        self._running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def update(self, msg: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._msg = msg

    def stop(self, clear: bool = True) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._t:
            self._t.join(timeout=1.0)
        if clear:
            self._clear_line()

    def _run(self) -> None:
        i = 0
        while self._running:
            ch = self._glyphs[i % len(self._glyphs)]
            with self._lock:
                msg = self._msg
            line = f"\r{msg} {ch}"
            try:
                sys.stderr.write(line)
                sys.stderr.flush()
            except Exception:
                self.enabled = False
                return
            time.sleep(0.1)
            i += 1

    def _clear_line(self) -> None:
        try:
            cols = shutil.get_terminal_size(fallback=(120, 24)).columns
            sys.stderr.write("\r" + (" " * cols) + "\r")
            sys.stderr.flush()
        except Exception:
            pass
