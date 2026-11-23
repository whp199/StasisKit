"""Raw terminal control: alternate screen, non-blocking key input, safe restore."""

import os
import select
import signal
import sys
import termios

_CSI_KEYS = {
    "A": "up", "B": "down", "C": "right", "D": "left",
    "H": "home", "F": "end", "Z": "shift-tab",
    "P": "f1", "Q": "f2", "R": "f3", "S": "f4",
}
_TILDE_KEYS = {
    "1": "home", "2": "insert", "3": "delete", "4": "end",
    "5": "pgup", "6": "pgdn", "7": "home", "8": "end",
    "11": "f1", "12": "f2", "13": "f3", "14": "f4", "15": "f5",
}


def split_keys(text):
    """Parse decoded terminal input -> (keys, rest).

    `rest` is an escape sequence cut off at the end of the chunk (possibly a
    lone ESC); the caller may wait for more input or settle it with parse_keys.
    """
    keys = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\x1b":
            if i + 1 >= n:
                return keys, text[i:]
            nxt = text[i + 1]
            if nxt in "[O":
                if i + 2 >= n:
                    return keys, text[i:]
                if nxt == "[" and text[i + 2] == "[":   # Linux console F1-F5: ESC [ [ A..E
                    if i + 3 >= n:
                        return keys, text[i:]
                    if text[i + 3] in "ABCDE":
                        keys.append(f"f{'ABCDE'.index(text[i + 3]) + 1}")
                    i += 4
                    continue
                j = i + 2
                while j < n and " " <= text[j] <= "?":   # parameter / intermediate bytes
                    j += 1
                if j >= n:
                    return keys, text[i:]
                final = text[j]
                if not ("@" <= final <= "~"):   # aborted (e.g. by a new ESC): drop it
                    i = j
                    continue
                params = text[i + 2:j]
                if final == "~":
                    keys.append(_TILDE_KEYS.get(params.split(";")[0], "unknown"))
                elif final in _CSI_KEYS:
                    keys.append(_CSI_KEYS[final])
                i = j + 1
                continue
            if nxt == "\x1b":
                keys.append("esc")
                i += 1
                continue
            keys.append("alt-" + nxt)
            i += 2
            continue
        if c in "\r\n":
            keys.append("enter")
        elif c in "\x7f\x08":
            keys.append("backspace")
        elif c == "\t":
            keys.append("tab")
        elif c == " ":
            keys.append("space")
        elif c == "\x03":
            keys.append("ctrl-c")
        elif c == "\x04":
            keys.append("ctrl-d")
        elif c == "\x0c":
            keys.append("ctrl-l")
        elif c == "\x1a":
            keys.append("ctrl-z")
        elif ord(c) >= 32:
            keys.append(c)
        i += 1
    return keys, ""


def parse_keys(text):
    """Turn a chunk of decoded terminal input into a list of key names."""
    keys, rest = split_keys(text)
    # only a lone ESC is "esc"; a bare ESC [ / ESC O is alt-[ / alt-O, and an
    # unfinished longer sequence is dropped (never "esc", which would quit)
    if rest == "\x1b":
        keys.append("esc")
    elif len(rest) == 2:
        keys.append("alt-" + rest[1])
    return keys


class Terminal:
    """Context manager owning the tty for the lifetime of the app."""

    def __init__(self):
        self.fd_in = sys.stdin.fileno()
        self.fd_out = sys.stdout.fileno()
        self._old = None
        self._active = False

    # -- lifecycle -------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        if not os.isatty(self.fd_in) or not os.isatty(self.fd_out):
            raise RuntimeError("StasisKit needs an interactive terminal (a TTY).")
        self._old = termios.tcgetattr(self.fd_in)
        new = termios.tcgetattr(self.fd_in)
        new[0] &= ~(termios.IXON | termios.ICRNL | termios.INLCR | termios.IGNCR)
        new[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd_in, termios.TCSANOW, new)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._on_signal)
        self._active = True
        # alt screen, hide cursor, no autowrap, clear, set window title
        self.write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[0m\x1b[2J\x1b[H\x1b]2;StasisKit\x07")

    def stop(self):
        if not self._active:
            return
        self._active = False
        # the tty may already be gone (hang-up: EIO / EPIPE); restoring must never raise
        try:
            self.write("\x1b[?2026l\x1b[0m\x1b[2J\x1b[H\x1b[?7h\x1b[?25h\x1b[?1049l")
        except OSError:
            pass
        if self._old is not None:
            try:
                termios.tcsetattr(self.fd_in, termios.TCSADRAIN, self._old)
            except (OSError, termios.error):
                pass

    def _on_signal(self, signum, frame):
        raise SystemExit(128 + signum)

    # -- io ----------------------------------------------------------------
    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8", "replace")
        view = memoryview(data)
        while view:
            try:
                written = os.write(self.fd_out, view)
            except InterruptedError:
                continue
            except BlockingIOError:
                select.select([], [self.fd_out], [], 0.05)
                continue
            view = view[written:]

    def size(self):
        try:
            sz = os.get_terminal_size(self.fd_out)
            return max(20, sz.columns), max(10, sz.lines)
        except OSError:
            return 80, 24

    def _wait(self, timeout):
        try:
            return bool(select.select([self.fd_in], [], [], max(0.0, timeout))[0])
        except InterruptedError:
            return False

    def _drain(self):
        chunks = []
        while True:
            try:
                b = os.read(self.fd_in, 4096)
            except (BlockingIOError, InterruptedError):
                break
            if not b:
                break
            chunks.append(b)
            if len(b) < 4096:
                break
        return b"".join(chunks)

    def read_keys(self, timeout):
        """Wait up to `timeout` seconds for input; return list of key names."""
        if not self._wait(timeout):
            return []
        data = self._drain()
        # a sequence (or ESC) cut off by the read: give its tail a moment to arrive
        for _ in range(4):
            if not split_keys(data.decode("utf-8", "ignore"))[1] or not self._wait(0.05):
                break
            data += self._drain()
        return parse_keys(data.decode("utf-8", "ignore"))
