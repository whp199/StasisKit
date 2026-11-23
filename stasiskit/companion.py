"""Companions: the friendly attendant who readies you for hypersleep.

Companions are TOML files (built-in ones live in stasiskit/companions/, your
own go in ~/.config/stasiskit/companions/).  Each has a name, colours, a
portrait (pixel-art sprite frames, or a procedural "spark") and lines of
dialogue grouped by event.  See README.md for the format.
"""

import math
import os
import random
import re
import tomllib

import numpy as np

from .util import hexrgb

BUILTIN_DIR = os.path.join(os.path.dirname(__file__), "companions")
PW_MAX, PH_MAX = 64, 64          # sanity cap in pixels (64 px = 32 rows); ui.py hides portraits that do not fit
errors = {}                      # {path: reason} for companion files the last load_all() skipped


def user_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "stasiskit", "companions")


def load_all():
    """Return {id: data} for built-in and user companions (user wins on clash).

    A file that cannot be read or parsed (not UTF-8, not TOML, ...) is skipped
    and noted in `errors`; odd field values are repaired by Companion, so one
    broken file never stops startup, --list or C-cycling."""
    found = {}
    errors.clear()
    for d in (BUILTIN_DIR, user_dir()):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for fn in names:
            if not fn.endswith(".toml"):
                continue
            path = os.path.join(d, fn)
            try:
                with open(path, "rb") as f:
                    data = tomllib.loads(f.read().decode("utf-8-sig"))  # tolerate a BOM
                data["id"] = _text(data.get("id"), None) or _text(fn[:-5], "companion")
                comp = Companion(data)
                comp.portrait.pixels("idle", 0.0)
            except Exception as e:  # OSError, ValueError (bad TOML / UTF-8), or anything unusable
                errors[path] = f"{type(e).__name__}: {e}"
                continue
            data.update(name=comp.name, title=comp.title, _path=path)
            found[data["id"]] = data
    return found or {"attendant": {"id": "attendant", "name": "Attendant", "_path": "(built-in fallback)"}}


# =============================================================================
# Field sanitising (companion files are user-written: never trust a value)
# =============================================================================
_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")          # control chars, incl. ESC: never reach the terminal
_CTRL_ML = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")  # same but keeps "\n" (dialogue may wrap)
_HEX = re.compile(r"#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
_FIELD = re.compile(r"\{\{|\}\}|\{(\w+)\}")


def _num(v, default, lo=-math.inf, hi=math.inf):
    """A finite number (numeric strings ok) clamped to [lo, hi], else default."""
    try:
        f = math.nan if isinstance(v, bool) else float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(hi, max(lo, f)) if math.isfinite(f) else default


def _text(v, default, n=40):
    """A clean single-line string (numbers ok), at most n chars, else default."""
    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
        v = " ".join(_CTRL.sub(" ", str(v)).split())[:n]
        if v:
            return v
    return default


def _colour(v, default):
    """'#rgb', '#rrggbb' or [r, g, b] (0-1 or 0-255) -> (r, g, b) in 0..1; otherwise
    `default` (a hex string, or None)."""
    if isinstance(v, str) and _HEX.fullmatch(v.strip()):
        return hexrgb(v)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        c = [_num(x, None) for x in v]
        if None not in c:
            k = 255.0 if max(c) > 1.0 else 1.0
            return tuple(min(1.0, max(0.0, x / k)) for x in c)
    return None if default is None else hexrgb(default)


def _clean_lines(node):
    """Dialogue tree: keep tables and non-empty string entries (a bare string counts as a
    one-line list); drop everything else."""
    if isinstance(node, dict):
        return {k: v for k, v in ((k, _clean_lines(v)) for k, v in node.items()) if v}
    if isinstance(node, str):
        node = [node]
    if isinstance(node, list):
        return [s for s in (_CTRL_ML.sub(" ", o).strip() for o in node if isinstance(o, str)) if s]
    return None


# =============================================================================
# Portraits
# =============================================================================
class Portrait:
    def __init__(self, spec):
        if isinstance(spec, str):  # portrait = "spark"
            spec = {"kind": spec}
        if not isinstance(spec, dict):
            spec = {}
        kind = spec.get("kind", "sprite")
        self.kind = kind.strip().lower() if isinstance(kind, str) else None
        if self.kind == "sprite":
            self._init_sprite(spec)
        if self.kind != "sprite":  # spark, unknown kinds, and sprites with nothing drawable
            self.kind = "spark"
            self._init_spark(spec)

    def _init_sprite(self, spec):
        pal = spec.get("palette")
        self.palette = {}
        for k, v in (pal.items() if isinstance(pal, dict) else ()):
            c = _colour(v, None)
            if isinstance(k, str) and len(k) == 1 and c is not None:
                self.palette[k] = np.array(c, np.float32)
        frames = spec.get("frames")
        self.frames = {}
        for name, art in (frames.items() if isinstance(frames, dict) else ()):
            if isinstance(art, list) and all(isinstance(r, str) for r in art):
                art = "\n".join(art)  # rows given as an array
            if not isinstance(art, str):
                continue
            rows = [r.rstrip()[:PW_MAX] for r in art.replace("\r", "").strip("\n").split("\n")][:PH_MAX]
            if any(rows):
                self.frames[name] = rows
        if not self.frames or not self.palette:
            self.kind = "spark"
            return
        self.frames.setdefault("idle", next(iter(self.frames.values())))
        w = max(len(r) for rows in self.frames.values() for r in rows)
        h = max(len(rows) for rows in self.frames.values())
        self.pw, self.ph = w, h + h % 2
        self._cache = {}

    def _init_spark(self, spec):  # procedural spark
        self.pw = int(_num(spec.get("size"), 20, 4, PH_MAX))
        self.ph = self.pw + self.pw % 2
        self.rays = int(_num(spec.get("rays"), 11, 1, 48))
        self.c_core = np.array(_colour(spec.get("core"), "#ffd9c7"), np.float32)
        self.c_main = np.array(_colour(spec.get("color"), "#d97757"), np.float32)
        self.c_edge = np.array(_colour(spec.get("edge"), "#9c4a30"), np.float32)
        rs = np.random.default_rng(int(abs(_num(spec.get("seed"), 7, -1e15, 1e15))))
        self.ray_ang = (np.arange(self.rays) / self.rays * 2 * math.pi
                        + rs.uniform(-0.12, 0.12, self.rays) + _num(spec.get("rotation"), 0.3) % (2 * math.pi))
        self.ray_len = rs.uniform(0.8, 1.0, self.rays)
        self.ray_ph = rs.uniform(0, 2 * math.pi, self.rays)

    @property
    def cells(self):
        return self.pw, self.ph // 2

    def frame_names(self):
        return list(self.frames) if self.kind == "sprite" else ["idle", "blink", "talk", "happy"]

    def pixels(self, state, t):
        """RGBA float image (ph, pw, 4)."""
        if self.kind == "sprite":
            name = state if state in self.frames else ("idle" if state != "happy" else "idle")
            img = self._cache.get(name)
            if img is None:
                img = np.zeros((self.ph, self.pw, 4), np.float32)
                for y, row in enumerate(self.frames[name]):
                    for x, c in enumerate(row):
                        col = self.palette.get(c)
                        if col is not None:
                            img[y, x, :3] = col
                            img[y, x, 3] = 1.0
                self._cache[name] = img
            return img
        return self._spark(state, t)

    def _spark(self, state, t):
        n = self.pw
        yy, xx = np.mgrid[0:self.ph, 0:n].astype(np.float32)
        c = (n - 1) / 2.0
        dx, dy = xx - c, yy - (self.ph - 1) / 2.0
        r = np.hypot(dx, dy) / (n / 2.0)
        th = np.arctan2(dy, dx)
        breathe = 0.93 + 0.07 * math.sin(t * 1.1)
        if state == "blink":
            breathe *= 0.78
        talk = state in ("talk", "happy")
        inten = np.zeros_like(r)
        for a, ln, ph in zip(self.ray_ang, self.ray_len, self.ray_ph):
            L = ln * breathe * (1.0 + (0.09 * math.sin(t * 9 + ph) if talk else 0.03 * math.sin(t * 0.8 + ph)))
            d = np.abs((th - a + math.pi) % (2 * math.pi) - math.pi)
            width = 0.25 * (1.0 - np.clip(r / max(L, 1e-3), 0, 1)) + 0.07
            on = (d * np.maximum(r, 0.3) < width * 0.5) & (r < L)
            inten = np.maximum(inten, np.where(on, 1.0 - 0.45 * r / max(L, 1e-3), 0.0))
        core = r < 0.17
        inten = np.where(core, 1.0, inten)
        img = np.zeros((self.ph, n, 4), np.float32)
        k = np.clip(r, 0, 1)[..., None]
        col = np.where(k < 0.3, self.c_core * (1 - k / 0.3) + self.c_main * (k / 0.3),
                       self.c_main * (1 - (k - 0.3) / 0.7) + self.c_edge * ((k - 0.3) / 0.7))
        glow = 1.0 + (0.18 * (0.5 + 0.5 * math.sin(t * 7.0)) if talk else 0.0)
        img[..., :3] = np.clip(col * glow, 0, 1)
        img[..., 3] = (inten > 0).astype(np.float32)
        return img


def draw_portrait(cv, portrait, x, y, state, t, bg):
    px = portrait.pixels(state, t)
    top, bot = px[0::2], px[1::2]
    h, w = top.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = min(cv.w, x0 + w), min(cv.h, y0 + h)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        return
    top, bot = top[:y1 - y0, :x1 - x0], bot[:y1 - y0, :x1 - x0]
    ta, ba = top[..., 3] > 0.5, bot[..., 3] > 0.5
    bgc = np.asarray(bg, np.float32)
    ch = np.where(ta, "▀", np.where(ba, "▄", " "))
    fg = np.where(ta[..., None], top[..., :3], np.where(ba[..., None], bot[..., :3], 0.0))
    bgo = np.where((ta & ba)[..., None], bot[..., :3], bgc)
    cv.ch[y0:y1, x0:x1] = ch
    cv.fg[y0:y1, x0:x1] = fg
    cv.bg[y0:y1, x0:x1] = bgo


# =============================================================================
# Companion
# =============================================================================
class Companion:
    def __init__(self, data, human_name=None):
        self.data = data = data if isinstance(data, dict) else {}
        self.id = _text(data.get("id"), "companion")
        self.name = _text(data.get("name"), self.id, 60)
        self.title = _text(data.get("title"), "Stasis Attendant")
        self.accent = _colour(data.get("accent"), "#8ab4ff")
        self.accent2 = _colour(data.get("accent2"), "#ff9fd6")
        self.text = _colour(data.get("text"), "#e8eeff")
        self.address = _text(human_name, None) or _text(data.get("address"), "friend")
        self.cps = _num(data.get("typing_cps"), 34.0, 5.0, 200.0)
        lines = data.get("lines")
        self.lines = _clean_lines(lines) if isinstance(lines, dict) else {}
        self.portrait = Portrait(data.get("portrait", {"kind": "spark"}))
        self._recent = {}
        self.blink_at = random.uniform(2, 5)
        self.blink_until = 0.0

    def _options(self, category):
        node = self.lines
        for part in category.split("."):
            if not isinstance(node, dict):
                return []
            node = node.get(part)
            if node is None:
                return []
        return node if isinstance(node, list) else []

    def has(self, category):
        return bool(self._options(category))

    def line(self, category, **kw):
        opts = self._options(category)
        if not opts:
            return None
        recent = self._recent.setdefault(category, [])
        pool = [o for o in opts if o not in recent] or opts
        choice = random.choice(pool)
        recent.append(choice)
        del recent[:-max(1, min(4, len(opts) - 1))]
        # Only plain {name} fields from this dict are filled ("{{"/"}}" -> "{"/"}"); anything else
        # ({0}, {x.attr}, {x[i]}, {x:>99}, stray braces) stays literal text, so no line can raise.
        fields = {"human": self.address, "name": self.name, **kw}

        def sub(m):
            if m.group(1) is None:
                return m.group(0)[0]
            v = fields.get(m.group(1))
            return m.group(0) if v is None else str(v)
        try:
            return _FIELD.sub(sub, choice)
        except Exception:
            return choice

    def face(self, t, talking, happy=False):
        """Pick the portrait frame for this moment."""
        if t >= self.blink_at:
            self.blink_until = t + 0.14
            self.blink_at = t + random.uniform(2.5, 6.5)
        if t < self.blink_until:
            return "blink"
        if talking:
            return "talk" if int(t * 8) % 2 == 0 else ("happy" if happy else "idle")
        return "happy" if happy else "idle"
