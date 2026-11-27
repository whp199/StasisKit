"""Animated layers of the CATHEDRAL view, cheap enough to redraw every frame.

* Flicker: gentle candle-light modulation - a slow waver of about +-10 % and a
  faint fast shimmer, never a strobe.
* Flames: crisp teardrops (white-gold core, orange edge, a breath of blue at
  the wick) that sway and stretch, each with a soft halo; flames too small to
  resolve become a bright point with a glow.  Each flame is z-tested against
  the building once, when the view is baked.
* Haze: tileable noise drifting slowly upward; it modulates how much light the
  incense haze scatters (the air light baked per light group) and the
  moonbeams.
* Plume: incense smoke rising from the censer - a narrow column that widens,
  sways and curls, its texture advected upward, lit by the air light around it.
* Motes: dust drifting through the moonbeams, visible only where a beam is.
* glint(): a small soft sparkle (rosary bead, monstrance rays).
"""

import math

import numpy as np

from ...util import periodic_noise
from . import kit

F32 = np.float32
FLAME_EDGE = np.array((1.0, 0.50, 0.14), F32)
FLAME_CORE = np.array((1.0, 0.93, 0.72), F32)
FLAME_BLUE = np.array((0.30, 0.38, 1.0), F32)
HALO = np.array((1.0, 0.58, 0.24), F32)


class Flicker:
    """n independent gentle flicker signals (deterministic for a given rng)."""

    def __init__(self, rng, n, slow=0.055, mid=0.035, fast=0.012):
        # phases and rates in double precision: om * t stays smooth after days of scene time
        self.ph = rng.uniform(0, 2 * math.pi, (n, 4)).astype(F32).astype(np.float64)
        self.om = np.stack([rng.uniform(0.35, 0.7, n), rng.uniform(1.0, 1.7, n),
                            rng.uniform(9.0, 13.0, n), rng.uniform(15.0, 21.0, n)], 1).astype(F32).astype(np.float64)
        self.amp = np.array((slow, mid, fast * 0.6, fast * 0.4), F32)

    def __call__(self, t):
        return (1.0 + np.sin(self.om * float(t) + self.ph) @ self.amp).astype(F32)

    def sway(self, t):
        """A second, slower signal in -1..1 (flame lean)."""
        return (np.sin(self.om[:, 0] * 0.7 * t + self.ph[:, 2]) * 0.7
                + np.sin(self.om[:, 1] * 0.9 * t + self.ph[:, 3]) * 0.3).astype(F32)


# =============================================================================
# flames
# =============================================================================
class Flames:
    """Every visible flame of one baked view, ready to draw."""

    def __init__(self, cam, gb, flames, group_index):
        H, W = gb.H, gb.W
        rows = []
        for wi, (pos, size, grp, cup) in enumerate(flames):
            sx, sy, dep = cam.project(np.array(pos, F32))
            sx, sy, dep = float(sx), float(sy), float(dep)
            if dep < 0.2:
                continue
            h = size * cam.f / dep
            if not (-4 < sx < W + 4 and -4 < sy < H + 4 + h):
                continue
            # z-test just above the wick: is anything nearer than the flame there?
            ix = int(np.clip(round(sx - 0.5), 0, W - 1))
            iy = int(np.clip(round(sy - 0.5 - max(1.0, h * 0.4)), 0, H - 1))
            if gb.depth[iy, ix] < dep - 0.15 and gb.mat[iy, ix] != 0:
                continue
            rows.append((sx, sy, h, dep, group_index.get(grp, -1), cup, 1.0 if cup < 0 else 0.75, wi))
        self.n = len(rows)
        a = np.array(rows, F32).reshape(-1, 8)
        self.ids = a[:, 7].astype(int)                     # index into the world's flame list
        self.sx, self.sy, self.h, self.dep = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        self.grp = a[:, 4].astype(int)
        self.cup = a[:, 5].astype(int)
        self.gain = a[:, 6]
        self.H, self.W = H, W
        self.depth = gb.depth

    def draw(self, img, flick, sway, mode):
        for i in range(self.n):
            h = float(self.h[i]) * (1.0 + 1.8 * (float(flick[i]) - 1.0))
            sx, sy = float(self.sx[i]), float(self.sy[i])
            g = float(self.gain[i]) * float(flick[i])
            # the style follows the flame's own size, not the flicker: a flame near the
            # threshold would otherwise pop between a point and a teardrop
            if float(self.h[i]) < 2.6:
                self._point(img, sx, sy - max(0.5, h * 0.45), h, g, mode)
            else:
                self._tear(img, sx, sy, h, float(sway[i]), g, float(self.dep[i]), mode)

    def _patch(self, x0, y0, x1, y1):
        x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
        x1, y1 = min(self.W, int(math.ceil(x1))), min(self.H, int(math.ceil(y1)))
        if x0 >= x1 or y0 >= y1:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(F32)
        return (slice(y0, y1), slice(x0, x1)), xs + 0.5, ys + 0.5

    def _point(self, img, cx, cy, h, g, mode):
        """A flame smaller than a few pixels: bright core and a soft glow."""
        r = 2.5 + h
        p = self._patch(cx - r, cy - r, cx + r, cy + r)
        if p is None:
            return
        sl, xs, ys = p
        d2 = (xs - cx) ** 2 + ((ys - cy) * 1.25) ** 2
        core = np.exp(-d2 / (0.35 + 0.25 * h)) * (1.1 + 0.25 * h)
        halo = np.exp(-d2 / (1.2 + 2.0 * h)) * (0.16 if mode == "ascii" else 0.24)
        img[sl] += (core[..., None] * FLAME_CORE + halo[..., None] * HALO) * g

    def _tear(self, img, cx, cy, h, sway, g, dep, mode):
        """A resolved flame: teardrop with a white-gold core, swaying at the tip."""
        w = h * 0.2
        lean = sway * h * 0.09
        hr = h * 0.85 + 2.0
        p = self._patch(cx - hr, cy - h - hr * 0.6, cx + hr, cy + hr * 0.5)
        if p is None:
            return
        sl, xs, ys = p
        v = (cy - ys) / h                                   # 0 at the wick, 1 at the tip
        u = xs - cx - lean * np.clip(v, 0, 1) ** 2
        vb = np.clip(v, 0.0, 1.0)
        prof = np.where(vb < 0.3, np.sin(np.pi * vb / 0.6), (1.0 - (vb - 0.3) / 0.7) ** 1.25)
        r = w * prof + 0.15
        inside = (v > -0.04) & (v < 1.0)
        cov = np.clip((r - np.abs(u)) + 0.5, 0.0, 1.0) * inside
        core = np.clip(1.0 - np.abs(u) / (r * 0.55 + 0.2), 0, 1) * np.clip(1.0 - np.abs(v - 0.33) / 0.33, 0, 1)
        blue = np.clip(1.0 - v / 0.12, 0, 1) * (v > -0.04)
        col = (FLAME_EDGE[None, None] * 1.25 + (FLAME_CORE * 2.0 - FLAME_EDGE * 1.25)[None, None] * core[..., None] ** 0.7
               + FLAME_BLUE * 0.35 * blue[..., None])
        # do not draw over anything that stands in front of the flame
        vis = (self.depth[sl] > dep - 0.12)[..., None]
        fl = col * cov[..., None]
        d2 = (xs - cx - lean * 0.4) ** 2 + ((ys - (cy - h * 0.42)) * 0.8) ** 2
        s2 = (h * 0.55 + 1.0) ** 2
        halo = np.exp(-d2 / s2)[..., None] * HALO * (0.22 if mode == "ascii" else 0.32)
        img[sl] = img[sl] * (1.0 - cov[..., None] * vis * 0.85) + (fl * g + halo * g) * vis


# =============================================================================
# haze, incense, motes
# =============================================================================
class Haze:
    """Slowly rising incense haze: a density field around 1 (0.55 .. 1.5)."""

    def __init__(self, seed):
        self.n1 = periodic_noise(128, 128, beta=2.4, seed=seed)
        self.n2 = periodic_noise(128, 128, beta=1.8, seed=seed + 1)
        self.key = None

    def density(self, H, W, t):
        hh, ww = max(1, (H + 1) // 2), max(1, (W + 1) // 2)
        if self.key != (hh, ww):
            self.key = (hh, ww)
            self.gy, self.gx = kit.grid(hh, ww)
        sc = 128.0 / max(60.0, min(ww, hh * 1.6) * 1.1)
        # the drift wraps with the 128-texel tiles, in double precision (smooth for days)
        a = kit.bilinear(self.n1, self.gx * sc + (t * 0.9) % 128, self.gy * sc + (t * 1.6) % 128)
        b = kit.bilinear(self.n2, self.gx * sc * 1.9 + (-t * 0.6) % 128, self.gy * sc * 1.9 + (t * 2.4) % 128)
        d = 0.55 + 0.95 * np.clip(a * 0.75 + b * 0.5 - 0.15, 0, 1.0) ** 1.2
        d = np.repeat(np.repeat(d.astype(F32), 2, 0), 2, 1)
        return d[:H, :W]


class Plume:
    """Incense rising from the censer, drawn in a box above it."""

    def __init__(self, sx, sy, height_px, H, W, seed, scale=1.0):
        self.sx, self.sy, self.hp = sx, sy, max(4.0, height_px)
        self.H, self.W = H, W
        self.tex = periodic_noise(96, 96, beta=1.7, seed=seed + 7)
        self.scale = scale
        x0 = int(max(0, sx - self.hp * 0.45))
        x1 = int(min(W, sx + self.hp * 0.45 + 1))
        y0 = int(max(0, sy - self.hp))
        y1 = int(min(H, sy + 1))
        self.sl = None
        if x0 < x1 and y0 < y1:
            self.sl = (slice(y0, y1), slice(x0, x1))
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(F32)
            self.xs, self.ys = xs + 0.5, ys + 0.5

    def density(self, t):
        if self.sl is None:
            return None
        hp = self.hp
        v = (self.sy - self.ys) / hp                       # 0 at the censer, 1 at the top
        vc = np.clip(v, 0, 1)
        tau = 2 * math.pi                                  # phases wrapped in double precision
        sway = hp * (0.05 * vc + 0.10 * vc * vc) * np.sin((0.45 * t) % tau + 2.6 * vc) \
            + hp * 0.035 * vc * np.sin((1.1 * t) % tau + 7.0 * vc + 1.0)
        xc = self.sx + sway
        width = hp * (0.018 + 0.16 * vc ** 1.15)
        dx = (self.xs - xc) / width
        env = np.exp(-dx * dx) * np.clip(v / 0.04, 0, 1) * (1.0 - vc) ** 1.3 * (v > 0)
        # wisps: texture advected upward, sheared sideways with height (curl)
        tu = (self.xs - xc) / hp * 7.0 + vc * 2.0 * np.sin(0.3 * t)
        tv = vc * 5.0 - ((t * 0.55 * 9.0) % 96.0) / 9.0          # the texture's 96 texels, at 9 per unit
        n = kit.bilinear(self.tex, tu * 9.0, tv * 9.0)
        dens = env * (0.25 + 1.6 * np.clip(n - 0.25, 0, 1) ** 1.2)
        return dens.astype(F32)


class Motes:
    """Dust drifting through the moonbeams (screen space; brightness from the beam map)."""

    def __init__(self, rng, H, W, beam_lum):
        self.H, self.W = H, W
        n = int(np.clip(W * H / 90, 10, 900))
        self.x = rng.uniform(0, W, n).astype(F32)
        self.y = rng.uniform(0, H, n).astype(F32)
        self.ph = rng.uniform(0, 2 * math.pi, n).astype(F32)
        self.sp = rng.uniform(0.4, 1.0, n).astype(F32)
        self.beam = beam_lum

    def update(self, dt, t):
        self.y += (0.35 + 0.5 * self.sp) * dt * (self.H / 90.0)
        self.x += (0.6 * np.sin((0.25 * t) % (2 * math.pi) + self.ph) + 0.25) * dt * self.sp
        self.y %= self.H                    # (however long the step)
        self.x %= self.W

    def draw(self, img, t, mode):
        xi = np.clip(self.x.astype(np.int32), 0, self.W - 1)
        yi = np.clip(self.y.astype(np.int32), 0, self.H - 1)
        b = np.clip(self.beam[yi, xi] - 0.12, 0, None)       # only inside a real shaft of light
        tw = 0.55 + 0.45 * np.sin(2.3 * float(t) * self.sp.astype(np.float64) + self.ph)
        a = b * tw * (0.5 if mode == "blocks" else 0.4)
        keep = a > 0.004
        if keep.any():
            np.add.at(img, (yi[keep], xi[keep]), a[keep, None] * np.array((0.80, 0.86, 1.0), F32))


def glint(img, x, y, r, amp, col, star=False):
    """A small soft sparkle at (x, y) (pixels), optionally with a faint cross."""
    H, W = img.shape[:2]
    R = int(math.ceil(r * 3 + 1 + (r * 3 if star else 0)))
    x0, x1 = max(0, int(x) - R), min(W, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(H, int(y) + R + 1)
    if x0 >= x1 or y0 >= y1 or amp <= 0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(F32)
    dx, dy = xs + 0.5 - x, ys + 0.5 - y
    k = np.exp(-(dx * dx + dy * dy) / (2 * r * r))
    if star:
        k = k + 0.35 * (np.exp(-dy * dy / 0.5) * np.exp(-np.abs(dx) / (r * 2.5))
                        + np.exp(-dx * dx / 0.5) * np.exp(-np.abs(dy) / (r * 2.5)))
    img[y0:y1, x0:x1] += k[..., None] * (np.asarray(col, F32) * amp)
