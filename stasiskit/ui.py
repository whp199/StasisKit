"""HUD, companion speech bubble and overlay panels."""

import math
import time

import numpy as np

from .audio.sources import LAYER_KEYS, LAYER_LABELS, LAYER_PRESETS
from .companion import draw_portrait
from .util import fmt_duration, mix, scale, wrap_text

TEXT = (0.86, 0.9, 0.95)
DIM = (0.48, 0.53, 0.61)
FAINT = (0.26, 0.29, 0.35)
OK = (0.45, 1.0, 0.62)
PANEL_BG = (0.025, 0.03, 0.05)
PARTIAL = " ▏▎▍▌▋▊▉█"


def _a(c, alpha):
    return tuple(v * alpha for v in c)


def _fit(s, n):
    """`s` cut to `n` cells, marking the cut with an ellipsis."""
    return s if len(s) <= n else s[:n - 1] + "…" if n > 1 else ""


def panel(cv, x, y, w, h, accent, title=None, alpha=1.0, shade=0.18):
    """Translucent panel: darken what's underneath, then draw a rounded frame."""
    cv.shade(x, y, w, h, shade, tint=accent, tint_amt=0.035 * alpha)
    cv.fill(x, y, w, h, " ")
    cv.box(x, y, w, h, _a(scale(accent, 0.75), alpha), style="round",
           title=title, title_fg=_a(accent, alpha))


def meter(cv, x, y, width, value, c0, c1, empty=FAINT):
    value = max(0.0, min(1.0, value))
    full = value * width
    for i in range(width):
        t = i / max(1, width - 1)
        col = mix(c0, c1, t)
        fill = full - i
        if fill >= 1:
            cv.put(x + i, y, "█", col)
        elif fill > 0:
            cv.put(x + i, y, PARTIAL[int(fill * 8)], col)
        else:
            cv.put(x + i, y, "·", empty)


# =============================================================================
# HUD
# =============================================================================
def draw_hud(cv, app, alpha):
    if alpha <= 0.02:
        return
    w, h = cv.w, cv.h
    comp = app.companion
    acc, acc2 = comp.accent, comp.accent2
    # --- top bar ---
    cv.shade(0, 0, w, 1, 0.15, tint=acc, tint_amt=0.03)
    cv.fill(0, 0, w, 1, " ")
    x = 1
    cv.text(x, 0, "◆ STASISKIT", _a(acc, alpha))
    x += 12
    scene = app.scene
    idx = f"{app.scene_idx + 1}/{len(app.scene_classes)}"
    cv.text(x, 0, "│", _a(FAINT, alpha))
    x += 2
    rest = []
    if len(scene.variants) > 1:
        rest.append(scene.variant_name)
    if w > 100:
        rest.insert(0, scene.subtitle)
    sub = " · ".join(r for r in rest if r)
    # right side first, so the title can be cut to the room it leaves
    right = []
    if app.timer_end:
        rem = max(0, app.timer_end - time.time())
        right.append(("⏾ " + fmt_duration(rem), acc2))
    if app.autocycle:
        right.append(("⟳", DIM))
    au = app.audio
    if au.enabled:
        if au.muted:
            right.append(("♪ muted", DIM))
        elif au.active:
            right.append((f"♪ {int(au.master * 100)}%", DIM))
        elif au.status.startswith("error"):
            right.append(("♪ ✗", (1.0, 0.45, 0.4)))
    right.append((time.strftime("%H:%M"), TEXT))
    # very narrow: drop the least useful items rather than squeeze out the title
    for drop in ("⟳", "♪"):
        if sum(len(s) + 2 for s, _ in right) <= w - x - 8:
            break
        right = [r for r in right if not r[0].startswith(drop)]
    rx = w - 1
    for s, c in reversed(right):
        rx -= len(s)
        cv.text(rx, 0, s, _a(c, alpha))
        rx -= 2
    # left: view index, title, variant - the index goes first, then the title is cut
    if x + len(idx) + 1 + len(scene.title) > rx:
        idx = ""
    if idx:
        cv.text(x, 0, idx, _a(DIM, alpha))
        x += len(idx) + 1
    title = _fit(scene.title, rx - x)
    cv.text(x, 0, title, _a(TEXT, alpha))
    x += len(title) + 1
    sub = _fit(sub, rx - x - 2)
    if sub:
        cv.text(x + 1, 0, sub, _a(DIM, alpha))
    # --- bottom bar ---
    y = h - 1
    cv.shade(0, y, w, 1, 0.15, tint=acc, tint_amt=0.03)
    cv.fill(0, y, w, 1, " ")
    hints = [("←→", "view"), ("v", "variant"), ("m", "mixer"), ("t", "timer"), ("b", "breathe"),
             ("c", "talk"), ("r", "ascii/hi-res"), ("[]", "dim"), ("?", "help"), ("q", "quit")]
    x = 1
    for k, label in hints:
        need = len(k) + len(label) + 3
        if x + need > w - 1:
            break
        cv.text(x, y, k, _a(acc, alpha))
        cv.text(x + len(k) + 1, y, label, _a(DIM, alpha))
        x += need
    if app.mode == "blocks":
        tag = "HI-RES"
    else:
        tag = "ASCII"
    if x + len(tag) + 2 < w:
        cv.text(w - len(tag) - 1, y, tag, _a(FAINT, alpha))


def draw_mini_timer(cv, app):
    """Tiny unobtrusive countdown when the HUD is hidden."""
    if not app.timer_end:
        return
    rem = max(0, app.timer_end - time.time())
    s = "⏾ " + fmt_duration(rem)
    cv.text(cv.w - len(s) - 1, 0, s, scale(app.companion.accent2, 0.45))


# =============================================================================
# Companion speech
# =============================================================================
class Speech:
    def __init__(self, text, now, cps, happy=False, hold=None):
        self.text = text
        self.t0 = now
        self.cps = min(1e6, max(1.0, cps))   # a typing_cps <= 0 / nan / inf must not break the maths
        self.happy = happy
        self.type_time = len(text) / self.cps
        self.hold = hold if hold is not None else 4.0 + len(text) / 16.0
        self.fade = 1.2

    def visible_chars(self, now):
        return min(len(self.text), int((now - self.t0) * self.cps))

    def typing(self, now):
        return now - self.t0 < self.type_time

    def alpha(self, now):
        end = self.t0 + self.type_time + self.hold
        if now < end:
            return 1.0
        return max(0.0, 1.0 - (now - end) / self.fade)

    def done(self, now):
        return self.alpha(now) <= 0.0


def draw_speech(cv, app, now):
    sp = app.speech
    pin = (bool(getattr(app, "always_show_companion", False))
           and not getattr(app, "screensaver", False)
           and getattr(app, "panel", None) != "breathe")
    a = sp.alpha(now) if sp is not None else 0.0
    has_bubble = a > 0.01
    if not has_bubble and not pin:
        return
    comp = app.companion
    w, h = cv.w, cv.h
    pw, ph = comp.portrait.cells
    bottom = h - 2
    if not has_bubble:
        # Always-show idle companion portrait anchored cleanly in the bottom-left
        if min(pw, ph) > 0 and w >= pw + 6 and h >= ph + 5:
            px, py = 2, bottom - ph - 1
            panel(cv, px - 1, py - 1, pw + 2, ph + 2, comp.accent, title=comp.name, alpha=1.0)
            bg = tuple(v for v in cv.bg[py, px])
            face = comp.face(now, False, False)
            draw_portrait(cv, comp.portrait, px, py, face, now, bg)
        return
    # portraits can be any size: show one only while the bubble beside it keeps
    # a sensible width and the text still fits above the bottom bar
    show_portrait = w >= 64 and min(pw, ph) > 0 and h >= ph + 6 and w - (pw + 7) >= 24
    if show_portrait:
        bw = min(66, w - (pw + 7))
        lines = wrap_text(sp.text, bw - 4)
        show_portrait = len(lines) + 2 <= bottom
    if not show_portrait:
        bw = max(8, min(66, w - 4))
        lines = wrap_text(sp.text, bw - 4)
    shown = sp.visible_chars(now)
    bh = max(len(lines) + 2, 4 if not show_portrait else 3)
    if show_portrait:
        pa = 1.0 if pin else a
        px, py = 2, bottom - ph - 1
        panel(cv, px - 1, py - 1, pw + 2, ph + 2, comp.accent, alpha=pa)
        bg = tuple(v for v in cv.bg[py, px])
        face = comp.face(now, sp.typing(now), sp.happy)
        draw_portrait(cv, comp.portrait, px, py, face, now, bg)
        if pa < 1.0:
            cv.fg[py:py + ph, px:px + pw] *= pa
            cv.bg[py:py + ph, px:px + pw] *= pa
        bx = px + pw + 2
        by = bottom - bh + 1
    else:
        bx, by = 2, bottom - bh + 1
    panel(cv, bx, by, bw, bh, comp.accent, title=comp.name, alpha=a)
    if show_portrait:
        cv.put(bx, by + bh // 2, "◀", _a(comp.accent, a))
    count = 0
    for i, ln in enumerate(lines):
        if count >= shown:
            break
        seg = ln[:max(0, shown - count)]
        cv.text(bx + 2, by + 1 + i, seg, _a(comp.text, a))
        count += len(ln) + 1
    if sp.typing(now) and int(now * 3) % 2 == 0:
        li = min(len(lines) - 1, max(0, _line_of(lines, shown)))
        cx = bx + 2 + min(len(lines[li]), shown - sum(len(l) + 1 for l in lines[:li]))
        cv.put(min(cx, bx + bw - 2), by + 1 + li, "▌", _a(comp.accent2, a))


def _line_of(lines, n):
    c = 0
    for i, ln in enumerate(lines):
        c += len(ln) + 1
        if n < c:
            return i
    return len(lines) - 1


# =============================================================================
# Panels
# =============================================================================
def _center_box(cv, w, h):
    w = min(w, cv.w - 2)
    h = min(h, cv.h - 2)
    return (cv.w - w) // 2, (cv.h - h) // 2, w, h


def draw_mixer(cv, app):
    au = app.audio
    acc, acc2 = app.companion.accent, app.companion.accent2
    rows = ["master"] + LAYER_KEYS
    compact = cv.h < len(rows) + 9
    inner_h = len(rows) + (3 if compact else 6)
    x, y, w, h = _center_box(cv, 74, inner_h + 2)
    panel(cv, x, y, w, h, acc, title="♪ AUDIO MIXER")
    sel = app.mixer_sel
    meter_w = max(10, w - 44)
    room = w - 30 - meter_w   # cells left for the preset / ambience column
    # MASTER stays on top; on short terminals the layer rows scroll so the
    # selected one is always visible
    ly = y + (2 if compact else 4)
    nl = len(rows) - 1
    nvis = max(1, min(nl, y + h - (2 if compact else 3) - ly))
    first = max(0, min(sel - 1 - nvis // 2, nl - nvis))
    shown = [(0, y + (1 if compact else 2))] + [(i + 1, ly + i - first) for i in range(first, first + nvis)]
    for i, yy in shown:
        key = rows[i]
        selected = i == sel
        label = "MASTER" if key == "master" else LAYER_LABELS[key]
        level = au.master if key == "master" else au.levels.get(key, 0.0)
        col = TEXT if selected else DIM
        if selected:
            cv.fill(x + 1, yy, w - 2, 1, " ", None, scale(acc, 0.16))
            cv.put(x + 2, yy, "▸", acc)
        cv.text(x + 4, yy, label[:16], col)
        meter(cv, x + 21, yy, meter_w, level, scale(acc, 0.8), acc2)
        cv.text(x + 22 + meter_w, yy, f"{int(level * 100):3d}%", col)
        if room > 0 and key == "ambience":
            amb = app.scene.ambience or "none"
            cv.text(x + 28 + meter_w, yy, f"({amb})"[:room], FAINT)
        elif room > 0 and key != "master" and LAYER_PRESETS.get(key) and len(LAYER_PRESETS[key]) > 1:
            p = au.presets.get(key) or ""
            s = f"◂ {p} ▸" if selected else p
            cv.text(x + 28 + meter_w, yy, s[:room], acc2 if selected else FAINT)
    if first > 0:
        cv.put(x + w - 1, ly, "▲", acc2)
    if first + nvis < nl:
        cv.put(x + w - 1, ly + nvis - 1, "▼", acc2)
    status = au.status if au.status != "on" else f"streaming via {au.backend_name} · 48 kHz"
    if au.muted:
        status = "MUTED (shift+M to unmute)"
    hy = y + h - 2
    if not compact:
        cv.text(x + 3, hy - 1, status[:w - 6], FAINT)
    hint = "↑↓ select  ←→ level  enter variant  0 mute layer  M mute all  esc close"
    cv.text(x + 3, hy, hint[:w - 6], DIM)


TIMER_CHOICES = [0, 5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 180, 240]
END_ACTIONS = {"dark": "fade to black", "dim": "keep a dim night-light", "exit": "close StasisKit"}


def draw_timer(cv, app):
    acc, acc2 = app.companion.accent, app.companion.accent2
    x, y, w, h = _center_box(cv, 64, 13)
    panel(cv, x, y, w, h, acc, title="⏾ SLEEP TIMER")
    m = app.timer_sel
    # rows top-down as [name, blank rows above]; on short terminals squeeze out the
    # blank rows, then drop the least useful lines, so all stay above the hint row
    rows = [["strip", 1], ["big", 1]] + [["fade", 1]] * bool(m) + [["then", 0]]
    rows += [["running", 1]] * bool(app.timer_end) + [["hint", 1]]
    for key in ("hint", "strip", "running", "fade", "big"):
        if sum(1 + g for _, g in rows) > h - 2:
            for r in rows:
                if r[0] == key:
                    r[1] = 0
    for key in ("fade", "then"):
        if sum(1 + g for _, g in rows) > h - 2:
            rows = [r for r in rows if r[0] != key]
    at, yy = {}, y
    for key, g in rows[:-1]:
        yy += 1 + g
        if yy < y + h - 2:
            at[key] = yy
    # preset strip
    strip_y = at.get("strip", y + 1)
    xs = x + 3
    for c in TIMER_CHOICES:
        s = "off" if c == 0 else str(c)
        if xs + len(s) + 2 > x + w - 6:
            break
        if c == m:
            cv.text(xs - 1, strip_y, f"[{s}]", acc2)
        else:
            cv.text(xs, strip_y, s, DIM)
        xs += len(s) + 3
    cv.text(xs, strip_y, "min", FAINT)
    big = "no timer" if m == 0 else (f"{m} minutes" if m < 120 else f"{m // 60} h {m % 60:02d} min")
    if "big" in at:
        cv.text(x + (w - len(big) - 4) // 2, at["big"], "◂ " + big + " ▸", TEXT)
    if "fade" in at:
        fade = app.fade_seconds(m * 60)
        cv.text(x + 3, at["fade"], f"fade-out over the last {fmt_duration(fade)}"[:w - 6], DIM)
    if "then" in at:
        act = app.cfg.get("on_timer_end")
        act = END_ACTIONS.get(act if isinstance(act, str) else "dark", END_ACTIONS["dark"])
        cv.text(x + 3, at["then"], f"then: {act}"[:w - 6], DIM)
    if "running" in at:
        rem = app.timer_end - time.time()
        cv.text(x + 3, at["running"], f"running · {fmt_duration(rem)} remaining"[:w - 6], OK)
    hint = "←→ choose  ↑↓ ±1 min  e end-action  enter start  esc close"
    cv.text(x + 3, y + h - 2, hint[:w - 6], DIM)


HELP = [
    ("← →  1-9 0", "change view"),
    ("v / V", "next / previous variant of this view"),
    ("a", "toggle auto-cycling views"),
    ("r", "ASCII  ⇄  hi-res block rendering"),
    ("[  ]", "dim / brighten the screen"),
    ("m", "audio mixer"),
    ("-  +", "master volume"),
    ("M", "mute everything"),
    ("t", "sleep timer"),
    ("b", "guided breathing"),
    ("c / C", "talk to companion / switch companion"),
    ("s", "always show companion portrait"),
    ("i", "pin or hide the HUD"),
    ("q / esc", "quit"),
]


def draw_help(cv, app):
    acc = app.companion.accent
    x, y, w, h = _center_box(cv, 62, len(HELP) + 6)
    panel(cv, x, y, w, h, acc, title="? CONTROLS")
    for i, (k, d) in enumerate(HELP):
        if y + 2 + i >= y + h - 2:
            break
        cv.text(x + 3, y + 2 + i, k, app.companion.accent2)
        cv.text(x + 16, y + 2 + i, d[:w - 19], TEXT)
    cv.text(x + 3, y + h - 2, f"companion: {app.companion.name} · press any key", DIM)


# =============================================================================
# Guided breathing
# =============================================================================
BREATH_PATTERN = [("breathe in", 4.0), ("hold", 4.0), ("breathe out", 6.0), ("rest", 1.0)]
BREATH_CYCLES = 6


def breath_state(elapsed):
    cyc = sum(d for _, d in BREATH_PATTERN)
    n = int(elapsed // cyc)
    t = elapsed - n * cyc
    for name, d in BREATH_PATTERN:
        if t < d:
            return n, name, t, d
        t -= d
    return n, "rest", 0, 1


def breath_radius(name, t, d):
    if name == "breathe in":
        return 0.25 + 0.75 * (0.5 - 0.5 * math.cos(math.pi * t / d))
    if name == "hold":
        return 1.0
    if name == "breathe out":
        return 0.25 + 0.75 * (0.5 + 0.5 * math.cos(math.pi * t / d))
    return 0.25


def draw_breathe(cv, app, now):
    elapsed = now - app.breathe_t0
    n, name, t, d = breath_state(elapsed)
    acc, acc2 = app.companion.accent, app.companion.accent2
    cv.scale(0.25)
    w, h = cv.w, cv.h
    cx, cy = w / 2.0, h / 2.0 - 1
    Rmax = min(w * 0.22, h * 0.8)
    R = Rmax * breath_radius(name, t, d)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot((xs - cx), (ys - cy) * 2.0)
    glow = np.clip(1.0 - rr / max(R, 1e-3), 0, 1) ** 1.6
    cv.bg += (np.array(acc, np.float32) * 0.22)[None, None, :] * glow[..., None]
    vert = np.abs(ys - cy) * 2.0 / np.maximum(rr, 1e-3)
    ring = np.abs(rr - R) < 0.6 + 0.75 * vert
    cv.ch[ring] = "•"
    cv.fg[ring] = np.array(acc2, np.float32)
    halo = (np.abs(rr - R - 2.2) < 0.45 + 0.7 * vert)
    cv.ch[halo] = "·"
    cv.fg[halo] = np.array(scale(acc, 0.6), np.float32)
    label = name.upper()
    remaining = max(1, int(math.ceil(d - t))) if name != "rest" else ""
    cv.text(int(cx - len(label) / 2), int(cy), label, TEXT)
    cv.text(int(cx) - 1, int(cy) + 1, f"{remaining}", acc2)
    left = BREATH_CYCLES - n
    s = f"breath {min(n + 1, BREATH_CYCLES)} of {BREATH_CYCLES} · any key to stop"
    cv.text(int(cx - len(s) / 2), min(h - 3, int(cy + Rmax / 2 + 3)), s, DIM)
    return left <= 0


# =============================================================================
# Boot sequence
# =============================================================================
def boot_lines(app):
    au = app.audio
    if not au.enabled:
        audio = "disabled"
    elif au.active:
        audio = f"online · {au.backend_name}"
    elif au.status.startswith("error"):
        audio = "unavailable"
    else:
        audio = "warming up"
    return [
        ("Cryo-gel reservoir", "98.6 %"),
        ("Pod seal integrity", "nominal"),
        ("Vital signs monitor", "64 → 48 bpm"),
        ("Synthesis core", audio),
        ("Viewport feeds", f"{len(app.scene_classes)} channels"),
        ("Companion link", f"{app.companion.name.upper()} online"),
    ]


def draw_boot(cv, app, now):
    el = now - app.boot_t0
    acc, acc2 = app.companion.accent, app.companion.accent2
    lines = boot_lines(app)
    w = min(64, cv.w - 4)
    h = len(lines) + 7
    x, y = (cv.w - w) // 2, max(1, (cv.h - h) // 2 - 4)
    panel(cv, x, y, w, h, acc, title="HYPERSLEEP PREPARATION", shade=0.1)
    title = "S T A S I S K I T"
    cv.text(x + (w - len(title)) // 2, y + 1, title, acc)
    cv.text(x + 2, y + 2, "─" * (w - 4), FAINT)
    step = 0.42
    for i, (lbl, val) in enumerate(lines):
        t = el - 0.5 - i * step
        if t < 0:
            break
        yy = y + 3 + i
        dots = "." * max(2, w - 16 - len(lbl) - len(val))
        if t < step * 0.7:
            spin = "|/-\\"[int(now * 12) % 4]
            cv.text(x + 2, yy, f"[ {spin}  ]", DIM)
            cv.text(x + 10, yy, lbl, DIM)
        else:
            cv.text(x + 2, yy, "[ OK ]", OK)
            cv.text(x + 10, yy, lbl, TEXT)
            cv.text(x + 11 + len(lbl), yy, dots, FAINT)
            cv.text(x + w - 2 - len(val), yy, val, acc2)
    done_t = 0.5 + len(lines) * step
    if el > done_t:
        msg = "all systems nominal · press any key"
        cv.text(x + (w - len(msg)) // 2, y + h - 2, msg, DIM if int(now * 1.5) % 2 else acc)
    return done_t
