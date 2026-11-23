"""ARCANE ARCHIVE - a slow gallery of fantasy trading cards from the crew's
long shifts.  Original cards (add your own in ~/.config/stasiskit/cards/),
each with animated procedural art, a foil shimmer and a 3D flip between cards.
"""

import math
import os
import tomllib
import unicodedata

import numpy as np

from .base import Scene
from ..canvas import Canvas
from ..util import hexrgb, periodic_noise, wrap_text, hash01

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cards.toml")

FRAMES = {   # border colour, body tint
    "W": ("#e8dcb0", (0.10, 0.09, 0.065)),
    "U": ("#5fa8ff", (0.03, 0.06, 0.12)),
    "B": ("#b48cff", (0.07, 0.04, 0.09)),
    "R": ("#ff7a52", (0.12, 0.045, 0.03)),
    "G": ("#6fdc7a", (0.03, 0.09, 0.045)),
    "M": ("#ffd36e", (0.11, 0.09, 0.03)),
    "A": ("#b8c4d0", (0.07, 0.08, 0.09)),
    "L": ("#c8a070", (0.09, 0.07, 0.04)),
}
MANA = {"W": (1.0, 0.95, 0.78), "U": (0.35, 0.65, 1.0), "B": (0.62, 0.48, 0.78), "R": (1.0, 0.42, 0.3),
        "G": (0.4, 0.86, 0.45), "C": (0.75, 0.75, 0.8)}
RARITY = {"common": ((0.55, 0.55, 0.58), 0.12), "uncommon": ((0.78, 0.86, 0.95), 0.2),
          "rare": ((1.0, 0.82, 0.4), 0.32), "mythic": ((1.0, 0.5, 0.22), 0.5)}


def _clean(s, keep_nl=False):
    """Make a user string safe to draw one char per cell: control chars and odd
    spaces become spaces, combining/format marks are dropped, and double-width
    (CJK, fullwidth, emoji) or unassigned chars become '?' so rows stay aligned."""
    out = []
    for c in unicodedata.normalize("NFC", s):
        cat = unicodedata.category(c)
        if keep_nl and (c == "\n" or cat in ("Zl", "Zp")):
            out.append("\n")
        elif cat == "Cc" or cat[0] == "Z":
            out.append(" ")
        elif cat in ("Mn", "Me", "Cf"):
            continue
        elif ord(c) > 0xFFFF or cat in ("Co", "Cs", "Cn") or unicodedata.east_asian_width(c) in ("W", "F"):
            out.append("?")
        else:
            out.append(c)
    return "".join(out)


def _card(c):
    """Normalise one card table: every displayed field becomes a clean str."""
    out = {k: _clean(str(c[k])) for k in ("name", "type", "cost", "pt", "color", "rarity", "art") if k in c}
    for k in ("text", "flavor"):
        if k in c:
            v = c[k]
            out[k] = _clean("\n".join(map(str, v)) if isinstance(v, list) else str(v), keep_nl=True)
    return out


def load_cards():
    cards = []
    paths = [DATA]
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    ud = os.path.join(base, "stasiskit", "cards")
    if os.path.isdir(ud):
        paths += [os.path.join(ud, f) for f in sorted(os.listdir(ud)) if f.endswith(".toml")]
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f).get("card", [])
        except (OSError, ValueError):       # unreadable, not UTF-8, or not valid TOML
            continue
        if isinstance(data, dict):          # a single [card] table instead of [[card]]
            data = [data]
        if isinstance(data, list):
            cards += [_card(c) for c in data if isinstance(c, dict)]
    return cards or [{"name": "Blank Card", "type": "Artifact", "color": "A", "art": "space"}]


# =============================================================================
# procedural card art: f(H, W, t, cache, rng) -> (H, W, 3)
# =============================================================================
def _grid(H, W, c):
    if c.get("_g") is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        c["_g"] = (yy / max(1, H - 1), xx / max(1, W - 1))
    return c["_g"]


def _grad(y, top, bot):
    top, bot = np.array(top, np.float32), np.array(bot, np.float32)
    return top * (1 - y[..., None]) + bot * y[..., None]


def _stars(img, c, rng, t, n, ymax=1.0, key="_s"):
    H, W = img.shape[:2]
    if key not in c:
        c[key] = (rng.integers(0, W, n), rng.integers(0, max(1, int(H * ymax)), n), rng.uniform(0, 6.28, n),
                  rng.uniform(0.3, 1.0, n))
    sx, sy, ph, b = c[key]
    img[sy, sx] += ((0.5 + 0.5 * np.sin(t * 1.3 + ph)) * b * 0.8)[:, None]


def _disc(y, x, cy, cx, r, aspect=1.0):
    return np.hypot((x - cx) * aspect, y - cy) < r


def art_pod(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    asp = W / H
    img = _grad(y, (0.03, 0.05, 0.08), (0.01, 0.015, 0.03))
    d = np.hypot((x - 0.5) / 0.42, (y - 0.55) / 0.3)
    glass = d < 1.0
    glow = np.clip(1 - d, 0, 1) ** 0.7
    img = np.where(glass[..., None], np.array([0.08, 0.35, 0.42]) * glow[..., None] + 0.05, img)
    body = (np.hypot((x - 0.55) / 0.26, (y - 0.6) / 0.08) < 1) | (np.hypot((x - 0.27) / 0.06, (y - 0.56) / 0.12) < 1)
    img = np.where((body & glass)[..., None], img * 0.35 + np.array([0.02, 0.06, 0.08]), img)
    ring = np.abs(d - 1.0) < 0.07
    img = np.where(ring[..., None], np.array([0.35, 0.38, 0.42]) * (0.7 + 0.3 * (1 - y[..., None])), img)
    mist = (0.5 + 0.5 * np.sin(x * 9 + t * 0.6 + np.sin(y * 7 + t * 0.3))) * glass * 0.08
    img = img + mist[..., None] * np.array([0.6, 0.9, 1.0])
    for i, lx in enumerate((0.38, 0.46, 0.54, 0.62)):
        on = math.sin(t * 1.5 + i * 1.3) > 0
        yy, xx = int(0.9 * H), int(lx * W)
        if yy < H:
            img[yy, xx] = (0.3, 1.0, 0.5) if on else (0.1, 0.25, 0.15)
    return img


def art_jelly(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.02, 0.1, 0.18), (0.0, 0.01, 0.04))
    for i, (cx, cy, s, col) in enumerate(((0.38, 0.38, 0.2, (0.3, 0.9, 1.0)), (0.72, 0.6, 0.12, (1.0, 0.45, 0.85)))):
        p = math.sin(t * 2.0 + i * 2) * 0.5 + 0.5
        cy = (cy - 0.05 * math.sin(t * 0.3 + i)) % 1.2
        a, b = s * (1 - 0.12 * p), s * 0.9
        u, v = (x - cx) * (W / H) / a, (y - cy) / b
        r = np.sqrt(u * u + v * v)
        bell = (r < 1) & (v < 0.15)
        inten = np.where(bell, 0.25 + 0.75 * np.exp(-((1 - r) / 0.18) ** 2), 0)
        img = img + inten[..., None] * np.array(col) * 0.8
        for k in range(6):
            tx = cx + (k / 5 - 0.5) * a * 1.6 * H / W
            ty = np.linspace(cy + b * 0.15, cy + b * 2.6, 40)
            txs = tx + np.sin(ty * 18 - t * 2 + k) * 0.015
            yi, xi = (ty * (H - 1)).astype(int), (txs * (W - 1)).astype(int)
            ok = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
            img[yi[ok], xi[ok]] += np.array(col) * 0.35 * np.linspace(1, 0.1, 40)[ok, None]
    return img


def art_space(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    if "neb" not in c:
        c["neb"] = periodic_noise(H, W * 2, beta=3.0, seed=int(rng.integers(1 << 30)))
    off = int(t * 0.8) % (W * 2)
    n = np.roll(c["neb"], -off, axis=1)[:, :W]
    img = _grad(y, (0.0, 0.0, 0.02), (0.02, 0.01, 0.06)) + (np.clip(n - 0.45, 0, 1) ** 1.5)[..., None] * np.array([0.7, 0.25, 0.8])
    _stars(img, c, rng, t, W * H // 40)
    asp = W / H
    cx, cy, R = 0.62, 0.55, 0.28
    dx, dy = (x - cx) * asp, y - cy
    r = np.hypot(dx, dy)
    planet = r < R
    nz = np.sqrt(np.clip(R * R - r * r, 0, None)) / R
    lit = np.clip((-dx * 0.7 - dy * 0.5) / R + nz * 0.6, 0, 1)
    bands = 0.8 + 0.2 * np.sin(dy / R * 14 + np.sin(dx / R * 3 + t * 0.2))
    pc = np.array([0.95, 0.75, 0.5]) * (lit * bands)[..., None]
    ring = np.abs(np.hypot(dx / 1.9, dy / 0.45) - R * 0.95) < 0.03
    front = ring & (dy > 0)
    img = np.where((ring & ~planet)[..., None], np.array([0.8, 0.7, 0.55]) * 0.8, img)
    img = np.where(planet[..., None], pc, img)
    img = np.where(front[..., None], np.array([0.85, 0.75, 0.6]), img)
    return img


def art_forest(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.02, 0.03, 0.1), (0.05, 0.18, 0.22))
    _stars(img, c, rng, t, W * H // 60, 0.5)
    m = _disc(y, x, 0.22, 0.75, 0.1, W / H)
    img = img + np.exp(-np.hypot((x - 0.75) * W / H, y - 0.22) / 0.12)[..., None] * np.array([0.25, 0.25, 0.2])
    img = np.where(m[..., None], np.array([0.95, 0.95, 0.85]), img)
    hill = 0.62 + 0.06 * np.sin(x * 7 + 1)
    img = np.where((y > hill)[..., None], np.array([0.03, 0.08, 0.1]), img)
    saw = 0.72 - 0.14 * np.abs(((x * 9) % 1.0) - 0.5) * 2 + 0.05 * np.sin(x * 23)
    img = np.where((y > saw)[..., None], np.array([0.01, 0.03, 0.02]), img)
    if "ff" not in c:
        c["ff"] = rng.uniform(0, 1, (18, 2)) * np.array([1, 0.4]) + np.array([0, 0.55]), rng.uniform(0, 6.28, 18)
    pos, ph = c["ff"]
    for (fx, fy), p in zip(pos, ph):
        b = max(0.0, math.sin(t * 1.7 + p))
        xi = int(((fx + 0.02 * math.sin(t * 0.4 + p)) % 1) * (W - 1))
        yi = int((fy + 0.03 * math.sin(t * 0.3 + p * 2)) * (H - 1))
        if 0 <= yi < H:
            img[yi, xi] += np.array([0.8, 1.0, 0.3]) * b
    return img


def art_fire(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    if "n" not in c:
        c["n"] = periodic_noise(H * 2, W, beta=2.2, seed=int(rng.integers(1 << 30)))
    n = c["n"]
    off = int(t * 22) % (H * 2)
    nn = np.roll(n, off, axis=0)[:H]
    shape = np.clip(y * 1.25 - 0.1, 0, 1) * np.clip(1 - np.abs(x - 0.5) * 2.1 + (y - 0.5) * 0.9, 0, 1)
    v = np.clip(nn * 1.7 * shape - 0.22 + shape * 0.35, 0, 1)
    pal = np.array([[0, 0, 0], [0.4, 0.02, 0.02], [0.9, 0.25, 0.02], [1.0, 0.6, 0.1], [1.0, 0.95, 0.6], [1, 1, 1]], np.float32)
    k = v * (len(pal) - 1.001)
    i0 = k.astype(int)
    f = (k - i0)[..., None]
    return pal[i0] * (1 - f) + pal[np.minimum(i0 + 1, len(pal) - 1)] * f + np.array([0.03, 0.01, 0.01])


def art_tower(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.12, 0.05, 0.22), (0.85, 0.45, 0.3))
    _stars(img, c, rng, t, W * H // 70, 0.35)
    moon = np.hypot((x - 0.5) * W / H, y - 0.42)
    img = img + np.exp(-moon / 0.2)[..., None] * np.array([0.25, 0.2, 0.25])
    img = np.where((moon < 0.24)[..., None], np.array([0.98, 0.9, 0.85]) - (y[..., None] - 0.3) * 0.2, img)
    body = (np.abs(x - 0.5) < 0.07 + y * 0.03) & (y > 0.3)
    roof = (y > 0.12) & (y <= 0.3) & (np.abs(x - 0.5) < (y - 0.12) * 0.55)
    ground = y > 0.85 + 0.03 * np.sin(x * 9)
    sil = body | roof | ground
    img = np.where(sil[..., None], np.array([0.04, 0.02, 0.06]), img)
    win = (np.abs(x - 0.5) < 0.018) & (np.abs(y - 0.42) < 0.035)
    flick = 0.8 + 0.2 * math.sin(t * 7) * math.sin(t * 2.3)
    img = np.where(win[..., None], np.array([1.0, 0.8, 0.35]) * flick, img)
    return img


def art_vortex(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    dx, dy = (x - 0.5) * W / H, y - 0.5
    r = np.hypot(dx, dy) + 1e-3
    a = np.arctan2(dy, dx)
    sw = np.sin(a * 3 + np.log(r) * 7 - t * 1.2)
    v = (0.5 + 0.5 * sw) * np.clip(r * 2.2, 0, 1) * np.exp(-r * 1.2)
    img = v[..., None] * np.array([0.55, 0.2, 0.75]) + (v ** 3)[..., None] * np.array([0.6, 0.5, 0.9])
    ringv = np.exp(-((r - 0.12) / 0.02) ** 2)
    img = img + ringv[..., None] * np.array([1.0, 0.75, 0.9]) * (0.8 + 0.2 * np.sin(a * 2 + t * 3))[..., None]
    img = np.where((r < 0.1)[..., None], 0.0, img)
    return img


def art_island(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    hz = 0.58
    img = _grad(np.clip(y / hz, 0, 1), (0.25, 0.12, 0.35), (1.0, 0.6, 0.35))
    sun = np.hypot((x - 0.5) * W / H, y - hz + 0.02)
    img = img + np.exp(-sun / 0.12)[..., None] * np.array([0.4, 0.25, 0.1])
    img = np.where(((sun < 0.13) & (y < hz))[..., None], np.array([1.0, 0.9, 0.6]), img)
    sea = y >= hz
    depth = (y - hz) / (1 - hz)
    shim = 0.5 + 0.5 * np.sin(depth * 60 / (depth + 0.15) - t * 2 + np.sin(x * 20) * 0.5)
    col = _grad(np.clip(depth, 0, 1), (0.5, 0.3, 0.35), (0.05, 0.08, 0.2))
    path = np.exp(-((x - 0.5) / (0.05 + depth * 0.12)) ** 2) * shim
    col = col + path[..., None] * np.array([1.0, 0.75, 0.4]) * 0.7
    img = np.where(sea[..., None], col, img)
    isl = (y > hz - 0.1 * np.exp(-((x - 0.18) / 0.08) ** 2)) & (y < hz + 0.01) & (x < 0.35)
    img = np.where(isl[..., None], np.array([0.08, 0.04, 0.08]), img)
    return img


def art_crystal(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.04, 0.02, 0.08), (0.08, 0.03, 0.12))
    if "cr" not in c:
        c["cr"] = [(rng.uniform(0.15, 0.85), rng.uniform(0.3, 0.75), rng.uniform(0.05, 0.11), rng.uniform(-0.35, 0.35))
                   for _ in range(6)]
    for cx, top, wd, tilt in sorted(c["cr"], key=lambda q: -q[1]):
        u = (x - cx) - (y - 1.0) * tilt * 0.3
        half = wd * np.clip((y - top) / 0.12, 0, 1)
        inside = (np.abs(u) < half) & (y > top)
        facet = np.where(u < 0, 0.55, 1.0) * (0.6 + 0.4 * (1 - (y - top)))
        glow = 0.85 + 0.15 * math.sin(t * 1.2 + cx * 10)
        col = np.array([0.45, 0.75, 1.0]) * facet[..., None] * glow
        img = np.where(inside[..., None], col, img)
        img = img + np.exp(-np.abs(u) / 0.05)[..., None] * (y > top)[..., None] * np.array([0.05, 0.08, 0.15])
    _stars(img, c, rng, t * 3, 25)
    return img


def art_pool(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    dx, dy = x - 0.5, y - 0.45
    side = np.abs(dx) > np.abs(dy) * 1.3
    z = np.where(side, 0.25 / (np.abs(dx) + 1e-3), 0.18 / (np.abs(dy) + 1e-3))
    u = np.where(side, dy * z, dx * z)
    gl = (np.abs((z * 2) % 1 - 0.5) > 0.44) | (np.abs((u * 3) % 1 - 0.5) > 0.44)
    tile = np.where(gl[..., None], np.array([0.62, 0.72, 0.74]), np.array([0.86, 0.94, 0.95]))
    floor = (~side) & (dy > 0)
    water = floor & (np.abs(u) < 0.9)
    a = np.sin(u * 6 + t) + np.sin(z * 5 - t * 0.8) + np.sin((u + z) * 4 + t * 0.6)
    caus = (1 - np.abs(np.sin(a))) ** 6
    wcol = np.array([0.25, 0.72, 0.8]) + caus[..., None] * 0.35
    img = np.where(water[..., None], wcol, tile)
    ceil = (~side) & (dy < 0)
    lamp = ceil & ((np.floor(z * 1.5) % 2) == 0) & (np.abs(u) < 0.25)
    img = np.where(lamp[..., None], np.array([1.0, 1.0, 0.95]), img)
    fog = np.clip(1 - np.exp(-z * 0.25), 0, 1)
    return img * (1 - fog[..., None]) + np.array([0.85, 0.92, 0.93]) * fog[..., None]


def art_bloom(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.04, 0.03, 0.06), (0.02, 0.05, 0.03))
    img = img + np.exp(-np.hypot((x - 0.5) * W / H, y - 0.3) / 0.35)[..., None] * np.array([0.12, 0.05, 0.1])
    trunk = (np.abs(x - 0.5 - 0.03 * np.sin(y * 6)) < 0.02 + (y - 0.4) * 0.03) & (y > 0.4)
    img = np.where(trunk[..., None], np.array([0.25, 0.15, 0.1]), img)
    if "pts" not in c:
        r = np.sqrt(rng.uniform(0, 1, 260)) * 0.32
        a = rng.uniform(0, 6.28, 260)
        c["pts"] = (0.5 + np.cos(a) * r * H / W * 1.4, 0.38 + np.sin(a) * r * 0.75, rng.uniform(0.7, 1.0, 260))
    px, py, b = c["pts"]
    sway = 0.01 * math.sin(t * 0.8)
    xi = np.clip(((px + sway * (0.6 - py)) * (W - 1)).astype(int), 0, W - 1)
    yi = np.clip((py * (H - 1)).astype(int), 0, H - 1)
    img[yi, xi] = np.array([1.0, 0.7, 0.85]) * b[:, None]
    for k in range(5):
        fy = ((t * 0.05 + k * 0.21) % 1.0)
        fx = 0.3 + 0.4 * ((k * 0.37 + math.sin(t * 0.5 + k)) % 1.0)
        img[min(H - 1, int(fy * H)), int(fx * (W - 1))] = (1.0, 0.75, 0.9)
    ground = y > 0.92
    return np.where(ground[..., None], np.array([0.05, 0.08, 0.04]), img)


def art_comet(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.01, 0.01, 0.05), (0.06, 0.04, 0.14))
    _stars(img, c, rng, t, W * H // 45, 0.8)
    ph = (t * 0.03) % 1.0
    cx, cy = 0.15 + 0.7 * ph, 0.2 + 0.25 * ph
    dx, dy = (x - cx) * W / H, y - cy
    along = -(dx * 0.94 + dy * 0.34)
    perp = -dx * 0.34 + dy * 0.94
    tail = np.exp(-(perp / (0.01 + along * 0.08)) ** 2) * np.exp(-along / 0.5) * (along > 0)
    img = img + tail[..., None] * np.array([0.5, 0.8, 1.0]) * 0.9
    head = np.exp(-np.hypot(dx, dy) / 0.025)
    img = img + head[..., None] * np.array([1.0, 1.0, 0.9])
    mtn = y > 0.78 + 0.1 * np.abs(np.sin(x * 5.5 + 0.7)) * np.cos(x * 2)
    return np.where(mtn[..., None], np.array([0.02, 0.02, 0.04]), img)


def art_engine(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    dx, dy = (x - 0.5) * W / H, y - 0.5
    r = np.hypot(dx, dy)
    img = np.zeros((H, W, 3), np.float32) + np.array([0.04, 0.04, 0.05])
    grid = (np.abs((x * 12) % 1 - 0.5) > 0.47) | (np.abs((y * 8) % 1 - 0.5) > 0.46)
    img = np.where(grid[..., None], np.array([0.08, 0.09, 0.11]), img)
    for k in range(4):
        rr = ((t * 0.12 + k / 4) % 1.0) * 0.7
        img = img + np.exp(-((r - rr) / 0.02) ** 2)[..., None] * np.array([1.0, 0.55, 0.2]) * (1 - rr / 0.7)
    core = np.exp(-r / (0.08 + 0.01 * math.sin(t * 6.0)))
    img = img + core[..., None] * np.array([1.0, 0.8, 0.5]) * 1.3
    return img


def art_aurora(H, W, t, c, rng):
    y, x = _grid(H, W, c)
    img = _grad(y, (0.0, 0.02, 0.06), (0.02, 0.06, 0.12))
    _stars(img, c, rng, t, W * H // 55, 0.6)
    for k, col in enumerate(((0.2, 1.0, 0.6), (0.3, 0.8, 1.0), (0.7, 0.4, 1.0))):
        yc = 0.3 + 0.06 * k + 0.07 * np.sin(x * (5 + k) + t * (0.3 + 0.1 * k) + k)
        streak = 0.6 + 0.4 * np.sin(x * 70 + np.sin(x * 9 + t) * 3)
        band = np.exp(-((y - yc) / 0.09) ** 2) * (y < yc + 0.08) + np.exp(-((y - yc) / 0.25) ** 2) * 0.3 * (y < yc)
        img = img + (band * streak)[..., None] * np.array(col) * 0.45
    mtn = y > 0.7 + 0.12 * np.abs(np.sin(x * 4.0 + 0.3))
    snow = np.clip(1 - (y - 0.7) * 4, 0, 1)
    img = np.where(mtn[..., None], np.array([0.6, 0.7, 0.85]) * snow[..., None] * 0.5 + 0.03, img)
    return img


ARTS = {"pod": art_pod, "jelly": art_jelly, "space": art_space, "forest": art_forest, "fire": art_fire,
        "tower": art_tower, "vortex": art_vortex, "island": art_island, "crystal": art_crystal, "pool": art_pool,
        "bloom": art_bloom, "comet": art_comet, "engine": art_engine, "aurora": art_aurora}


class _CardNames:
    """`Cards.variants` read on the class (--list, --variant) gives the card names;
    an instance keeps its own tuple, loaded once in __init__."""
    def __get__(self, obj, cls=None):
        return tuple(c.get("name", "?") for c in load_cards())


# =============================================================================
class Cards(Scene):
    key = "cards"
    title = "ARCANE ARCHIVE"
    subtitle = "the crew's card collection"
    kind = "chars"
    ambience = "fire"
    variants = _CardNames()
    HOLD = 26.0
    FLIP = 2.0

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        # file order stays fixed so a saved variant index names the same card next
        # launch; a fresh visit starts on a random card unless one is restored
        self.cards = load_cards()
        self.variants = tuple(c.get("name", "?") for c in self.cards)
        self.shown = self.variant = self.flip_to = int(self.rng.integers(len(self.cards)))
        self.flip_t = None
        self.pending = None     # card asked for (v/V) while a flip was already running
        self.started = False
        self.hold = 0.0
        self.caches = {}
        self.face = None

    def on_variant(self):
        if not self.started:            # restored before the first frame: just show it
            self.shown = self.flip_to = self.variant
        elif self.flip_t is None:
            if self.variant != self.shown:
                self.flip_to = self.variant
                self.flip_t = 0.0
        elif self.flip_t < self.FLIP * 0.5:     # old face/back still showing: retarget
            self.flip_to = self.variant
        else:                                   # new face already chosen: flip again after
            self.pending = self.variant

    def update(self, dt):
        super().update(dt)
        self.started = True
        if self.flip_t is not None:
            self.flip_t += dt
            if self.flip_t >= self.FLIP:
                self.flip_t = None
                self.shown = self.flip_to
                self.hold = 0.0
                if self.pending is not None and self.pending != self.shown:
                    self.flip_to = self.pending
                    self.flip_t = 0.0
                self.pending = None
                self.variant = self.flip_to
        else:
            self.hold += dt
            if self.hold >= self.HOLD:
                self.flip_to = (self.shown + 1) % len(self.cards)
                self.variant = self.flip_to
                self.flip_t = 0.0

    # ---------------------------------------------------------------- layout
    def _dims(self):
        w, h = self.w, self.h
        ch = max(18, min(h - 4, 46))
        cw = int(round(ch / 0.698))
        if cw > w - 6:
            cw = w - 6
            ch = int(cw * 0.698)
        return cw, ch

    def _face(self, card, mode, idx):
        cw, ch = self._dims()
        cv = Canvas(cw, ch)
        color = card.get("color", "A")
        border, tint = FRAMES.get(color, FRAMES["A"])
        border = np.array(hexrgb(border), np.float32)
        tint = np.array(tint, np.float32)
        rar_col, foil = RARITY.get(card.get("rarity", "common"), RARITY["common"])
        cv.bg[:] = tint * 1.4
        cv.ch[:] = " "
        cv.box(0, 0, cw, ch, border, style="round")
        text = (0.92, 0.9, 0.86)
        dim = tuple(min(1.0, v * 0.9 + 0.08) for v in border * 0.75)
        # name bar
        cv.fill(1, 1, cw - 2, 1, " ", None, tint * 2.6)
        name = card.get("name", "Untitled")
        cv.text(2, 1, name[:cw - 14], text)
        cost = card.get("cost", "")
        xx = cw - 2
        for sym in reversed(cost):
            col = MANA.get(sym)
            if col is None:
                cv.text(xx - 2, 1, f"{sym} ", (0.85, 0.85, 0.85), (0.22, 0.22, 0.25))
            else:
                cv.text(xx - 2, 1, "● ", col)
            xx -= 2
        # rules text first, so the art box can take whatever space is left
        lines = []
        for para in card.get("text", "").split("\n"):
            lines += [(ln, (0.92, 0.9, 0.86)) for ln in wrap_text(para, cw - 8)]
        flavor = card.get("flavor")
        if flavor:
            lines.append(("", text))
            lines += [(ln, (0.62, 0.6, 0.58)) for ln in wrap_text(flavor, cw - 8)]
        need = len(lines) + 3
        art_h = int(np.clip(ch - 8 - need, ch * 0.40, ch * 0.6))
        # art box
        ax, ay, aw = 2, 3, cw - 4
        cv.box(ax - 1, ay - 1, aw + 2, art_h + 2, border * 0.6, style="square")
        art = ARTS.get(card.get("art", "space"), art_space)
        cache = self.caches.setdefault(idx, {"rng": np.random.default_rng(int(hash01(idx, len(name)) * 1e9))})
        if cache.get("_shape") != (art_h, aw):
            cache.clear()
            cache["rng"] = np.random.default_rng(idx * 7919 + 13)
            cache["_shape"] = (art_h, aw)
        img = np.clip(art(art_h * 2, aw, self.t, cache, cache["rng"]), 0, 1.5).astype(np.float32)
        cv.blit_pixels(img, mode, x=ax, y=ay, bgfill=0.3, floor=0.04, gamma=0.8)
        # type line
        ty = ay + art_h + 1
        cv.fill(1, ty, cw - 2, 1, " ", None, tint * 2.6)
        cv.text(2, ty, card.get("type", "")[:cw - 7], text)
        cv.text(cw - 4, ty, "◆", rar_col)
        # text box
        by = ty + 1
        bh = ch - by - 2
        cv.fill(2, by, cw - 4, bh, " ", None, tint * 0.6 + 0.03)
        for i, (ln, col) in enumerate(lines[:max(0, bh - 1)]):
            cv.text(4, by + 1 + i, ln, col)
        # footer
        fy = ch - 2
        num = f"SKT · {idx + 1:03d}/{len(self.cards):03d} · ✦ art: procedural"
        cv.text(2, fy, num[:cw - 12], dim)
        pt = card.get("pt")
        if pt:
            s = f" {pt} "
            cv.text(cw - len(s) - 2, fy, s, text, tint * 3.0 + 0.05)
        self._foil(cv, foil)
        return cv

    def _foil(self, cv, strength):
        h, w = cv.h, cv.w
        t = self.t
        period = 10.0
        s = ((t % period) / period) * (w + h * 2 + 40) - h * 2 - 20
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = (xx + (h - yy) * 2.0) - s
        band = np.exp(-(d / 7.0) ** 2) * strength
        hue = (xx * 0.015 + yy * 0.03 + t * 0.05) % 1.0
        rb = 0.5 + 0.5 * np.cos(6.283 * (hue[..., None] + np.array([0.0, 0.33, 0.67], np.float32)))
        cv.fg[:] = cv.fg * (1 - band[..., None] * 0.5) + rb * band[..., None] * 0.9
        cv.bg[:] = cv.bg + rb * band[..., None] * 0.12

    def _back(self, mode):
        cw, ch = self._dims()
        cv = Canvas(cw, ch)
        cv.clear((0.06, 0.035, 0.02))
        cv.box(0, 0, cw, ch, (0.55, 0.38, 0.2), style="double")
        H, W = (ch - 2) * 2, cw - 2
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = (xx / W - 0.5) * 1.4, (yy / H - 0.5)
        r = np.hypot(dx, dy) + 1e-3
        a = np.arctan2(dy, dx)
        sw = 0.5 + 0.5 * np.sin(a * 5 + np.log(r) * 8 - self.t * 0.8)
        oval = np.hypot(dx / 0.62, dy / 0.46) < 1
        img = np.where(oval[..., None], (sw * np.exp(-r * 2))[..., None] * np.array([0.6, 0.35, 0.8]) + 0.03,
                       np.array([0.25, 0.14, 0.06]) * (0.6 + 0.4 * sw[..., None]))
        rim = np.abs(np.hypot(dx / 0.62, dy / 0.46) - 1) < 0.04
        img = np.where(rim[..., None], np.array([0.9, 0.7, 0.35]), img)
        cv.blit_pixels(img.astype(np.float32), mode, x=1, y=1, bgfill=0.3, floor=0.03)
        label = "S T A S I S"
        cv.text((cw - len(label)) // 2, ch // 2, label, (1.0, 0.85, 0.55))
        return cv

    # ---------------------------------------------------------------- draw
    def draw_chars(self, cv, mode):
        w, h = self.w, self.h
        t = self.t
        cw, ch = self._dims()
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        # playmat background with a glow in the card's colour
        idx = self.shown if self.flip_t is None or self.flip_t < self.FLIP * 0.5 else self.flip_to
        card = self.cards[idx]
        border = np.array(hexrgb(FRAMES.get(card.get("color", "A"), FRAMES["A"])[0]), np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w * 0.5), (yy - h / 2) / (h * 0.5))
        cv.ch[:] = " "
        cv.fg[:] = 0
        cv.bg[:] = np.array([0.02, 0.035, 0.045], np.float32) * (1.4 - 0.6 * np.clip(r, 0, 1))[..., None]
        cv.bg[:] += border * (0.06 * np.exp(-r * 2.4))[..., None]
        # drifting motes
        for i in range(24):
            mx = int((hash01(i, 1) * w + t * (0.4 + hash01(i, 2))) % w)
            my = int((hash01(i, 3) * h - t * 0.2 * (0.5 + hash01(i, 4))) % h)
            b = 0.25 + 0.2 * math.sin(t + i)
            cv.put(mx, my, "·", (b, b * 0.9, b * 0.7))
        # card (with flip)
        if self.flip_t is None:
            face = self._face(card, mode, idx)
            k, show = 1.0, face
        else:
            ph = self.flip_t / self.FLIP
            th = ph * 2 * math.pi
            k = abs(math.cos(th))
            if ph < 0.25:
                show = self._face(self.cards[self.shown], mode, self.shown)
            elif ph < 0.75:
                show = self._back(mode)
            else:
                show = self._face(self.cards[self.flip_to], mode, self.flip_to)
        # shadow, under the (possibly narrowed, centred) card
        dw = max(1, int(round(cw * k)))
        xs = x0 + (cw - dw) // 2
        cv.shade(xs + 2, y0 + 1, dw, ch, 0.5)
        if dw >= cw:
            cv.blit_canvas(show, x0, y0)
        else:
            cols = np.minimum(((np.arange(dw) + 0.5) / dw * cw).astype(int), cw - 1)
            x1 = min(w, xs + dw)
            ys0, ys1 = max(0, y0), min(h, y0 + ch)
            sc = cols[:x1 - xs]
            light = 0.55 + 0.45 * k
            cv.ch[ys0:ys1, xs:x1] = show.ch[ys0 - y0:ys1 - y0][:, sc]
            cv.fg[ys0:ys1, xs:x1] = show.fg[ys0 - y0:ys1 - y0][:, sc] * light
            cv.bg[ys0:ys1, xs:x1] = show.bg[ys0 - y0:ys1 - y0][:, sc] * light
