"""ABYSSAL TANK - bioluminescent jellyfish drifting through a deep aquarium.

Bells pulse (fast contraction, slow relaxation) and that thrust is what moves
them; tentacles are verlet chains that trail and sway in the current.  Marine
snow falls through faint god rays; everything glows with a soft bloom.
"""

import math

import numpy as np

from .base import Scene
from ..util import hexrgb

PALETTES = {
    "bioluminescence": ["#4ff0ff", "#ff6ad5", "#9d7bff", "#6affc8", "#7ab8ff"],
    "moon jellies": ["#dfe9ff", "#ffd6ec", "#cfe0ff", "#e6dcff", "#d8fff4"],
    "deep ember": ["#ff7a3c", "#ff4f6e", "#ffb347", "#ff9466", "#e8466a"],
}
WATER = {
    "bioluminescence": ((0.015, 0.075, 0.14), (0.0, 0.008, 0.025)),
    "moon jellies": ((0.03, 0.10, 0.17), (0.0, 0.015, 0.04)),
    "deep ember": ((0.04, 0.03, 0.08), (0.006, 0.0, 0.015)),
}


def _blur(img, r):
    """Fast box blur along both axes (cumulative sums), applied twice ~ gaussian."""
    for _ in range(2):
        for ax in (0, 1):
            n = img.shape[ax]
            c = np.cumsum(img, axis=ax)
            pad = [(0, 0)] * img.ndim
            pad[ax] = (r + 1, r)
            c = np.pad(c, pad, mode="edge")
            if ax == 0:
                c[:r + 1] = 0
                img = (c[2 * r + 1:2 * r + 1 + n] - c[:n]) / (2 * r + 1)
            else:
                c[:, :r + 1] = 0
                img = (c[:, 2 * r + 1:2 * r + 1 + n] - c[:, :n]) / (2 * r + 1)
    return img


class Jelly:
    def __init__(self, rng, W, H, palette, initial=False):
        self.rng = rng
        self.depth = rng.uniform(0.35, 1.0)
        self.s = (5.0 + 13.0 * self.depth ** 1.5) * min(1.6, max(0.7, H / 90))
        self.x = rng.uniform(0.08, 0.92) * W
        # replacements start with the bell (top at y - ~0.85 s) just below the edge;
        # tentacles and arms are built from self.y below, so they hang out of view too
        self.y = rng.uniform(0.1, 1.0) * H if initial else H + self.s
        self.vx = rng.uniform(-0.6, 0.6)
        self.vy = 0.0
        self.phase = rng.uniform(0, 1)
        self.freq = rng.uniform(0.45, 0.75)
        self.col = np.array(hexrgb(palette[rng.integers(len(palette))]), np.float32)
        self.col2 = np.array(hexrgb(palette[rng.integers(len(palette))]), np.float32)
        self.tilt = 0.0
        self.nt = int(rng.integers(7, 12))
        self.L = int(rng.integers(14, 22))
        self.seg = self.s * 3.0 / self.L * rng.uniform(0.8, 1.25)
        self.anchor_u = np.linspace(-0.85, 0.85, self.nt) + rng.uniform(-0.05, 0.05, self.nt)
        ax, ay = self._anchors(0.0)
        self.P = np.zeros((self.nt, self.L, 2))
        for k in range(self.L):
            self.P[:, k, 0] = ax
            self.P[:, k, 1] = ay + k * self.seg
        self.Pp = self.P.copy()
        # oral arms
        self.na = 4
        self.La = 9
        self.A = np.zeros((self.na, self.La, 2))
        for k in range(self.La):
            self.A[:, k, 0] = self.x + np.linspace(-0.25, 0.25, self.na) * self.s
            self.A[:, k, 1] = self.y + k * self.seg * 0.8
        self.Ap = self.A.copy()
        self.pulse = 0.0

    def _anchors(self, pulse):
        a = self.s * (1.0 - 0.17 * pulse)
        b = self.s * 0.82 * (1.0 + 0.08 * pulse)
        u = self.anchor_u
        lx = u * a
        ly = b * 0.16 * np.ones_like(u)
        c, s = math.cos(self.tilt), math.sin(self.tilt)
        return self.x + lx * c - ly * s, self.y + lx * s + ly * c

    def update(self, dt, t, current):
        self.phase += self.freq * dt
        ph = self.phase % 1.0
        # quick contraction then slow relaxation
        self.pulse = math.exp(-((ph - 0.14) / 0.09) ** 2) if ph < 0.5 else 0.0
        self.pulse = max(self.pulse, math.exp(-((ph - 0.14) / 0.09) ** 2))
        thrust = 26.0 * self.pulse * (0.55 + 0.45 * self.depth)
        self.vy += (-thrust - self.vy * 1.6 + 1.2) * dt
        self.vx += (current * 0.4 - self.vx * 0.3) * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        target_tilt = max(-0.35, min(0.35, self.vx * 0.12))
        self.tilt += (target_tilt + 0.06 * math.sin(t * 0.7 + self.freq * 9) - self.tilt) * min(1, dt * 1.5)
        ax, ay = self._anchors(self.pulse)
        self._chain(self.P, self.Pp, ax, ay, self.seg, dt, t, current, 1.0)
        armx = self.x + np.linspace(-0.22, 0.22, self.na) * self.s
        army = np.full(self.na, self.y + self.s * 0.1)
        self._chain(self.A, self.Ap, armx, army, self.seg * 0.75, dt, t, current, 0.6)

    def _chain(self, P, Pp, ax, ay, seg, dt, t, current, sway):
        vel = (P - Pp) * 0.9
        Pp[:] = P
        n = P.shape[1]
        k = np.arange(n)[None, :]
        wave = np.sin(t * 1.3 + k * 0.45 + self.anchor_u[:P.shape[0], None] * 3 if P.shape[0] == self.nt
                      else t * 1.3 + k * 0.45)
        P[..., 0] += vel[..., 0] + (current * 0.6 + wave * 0.9 * sway) * dt * 2.0
        P[..., 1] += vel[..., 1] + 3.0 * dt * 2.0 * dt * 10
        P[:, 0, 0] = ax
        P[:, 0, 1] = ay
        for j in range(1, n):
            d = P[:, j] - P[:, j - 1]
            ln = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
            P[:, j] = P[:, j - 1] + d / ln * seg

    def offscreen(self, H):
        return self.y + self.s < -self.s * 0.5 and (self.P[:, -1, 1] < -2).all()


class Abyss(Scene):
    key = "abyss"
    title = "ABYSSAL TANK"
    subtitle = "bioluminescent fauna, aquarium deck"
    kind = "pixels"
    variants = ("bioluminescence", "moon jellies", "deep ember")
    ambience = "deepsea"
    bgfill = 0.3
    ramp = "classic"
    floor = 0.09
    edges = 0.9

    def resize(self, w, h):
        W, H = w, h * 2
        self.W, self.H = W, H
        pal = PALETTES[self.variant_name]
        n = max(4, min(9, int(W * H / 2300)))
        self.jellies = [Jelly(self.rng, W, H, pal, initial=True) for _ in range(n)]
        self.jellies.sort(key=lambda j: j.depth)
        ns = int(W * H / 150)
        self.sx = self.rng.uniform(0, W, ns)
        self.sy = self.rng.uniform(0, H, ns)
        self.sd = self.rng.uniform(0.2, 1.0, ns)
        self.sph = self.rng.uniform(0, 6.28, ns)
        top, bot = WATER[self.variant_name]
        g = np.linspace(0, 1, H)[:, None, None] ** 0.8
        self.water = (np.array(top) * (1 - g) + np.array(bot) * g).astype(np.float32) * np.ones((1, W, 1), np.float32)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.ray_coord = xx - yy * 0.42
        self.ray_fall = np.clip(1.0 - yy / (H * 0.95), 0, 1) ** 1.8
        self.current = 0.0

    def on_variant(self):
        if self.w:
            self.resize(self.w, self.h)

    def update(self, dt):
        super().update(dt)
        if not self.w:
            return
        t = self.t
        self.current = 1.2 * math.sin(t * 0.05) + 0.6 * math.sin(t * 0.13 + 1.0)
        pal = PALETTES[self.variant_name]
        for i, j in enumerate(self.jellies):
            j.update(dt, t, self.current)
            if j.offscreen(self.H) or j.x < -j.s * 4 or j.x > self.W + j.s * 4:
                self.jellies[i] = Jelly(self.rng, self.W, self.H, pal)
        self.jellies.sort(key=lambda j: j.depth)
        self.sy += (2.0 + 2.5 * self.sd) * dt
        self.sx += (self.current * 0.8 * self.sd + 0.6 * np.sin(t * 0.8 + self.sph)) * dt
        wrap = self.sy > self.H
        if wrap.any():
            self.sy[wrap] -= self.H
            self.sx[wrap] = self.rng.uniform(0, self.W, int(wrap.sum()))
        self.sx %= self.W

    def draw_pixels(self, mode):
        W, H = self.W, self.H
        t = self.t
        img = self.image()
        img[:] = self.water
        # god rays
        rc = self.ray_coord
        rays = np.zeros((H, W), np.float32)
        for i, (sp, wd, a) in enumerate(((0.07, 7.0, 1.0), (0.045, 11.0, 0.7), (0.11, 5.0, 0.6), (0.03, 15.0, 0.5))):
            c = (math.sin(t * sp + i * 2.1) * 0.5 + 0.5) * (W * 1.3) - W * 0.1
            rays += a * np.exp(-((rc - c) / wd) ** 2)
        rays *= self.ray_fall * (0.75 + 0.25 * math.sin(t * 0.3))
        img += rays[..., None] * np.array([0.05, 0.11, 0.13], np.float32)
        # marine snow
        xi = self.sx.astype(np.int32) % W
        yi = np.clip(self.sy.astype(np.int32), 0, H - 1)
        b = (0.05 + 0.12 * self.sd) * (0.7 + 0.3 * np.sin(t * 1.5 + self.sph))
        np.add.at(img, (yi, xi), (b[:, None] * np.array([0.75, 0.9, 1.0], np.float32)))
        # jellies into a glow layer
        glow = np.zeros((H, W, 3), np.float32)
        for j in self.jellies:
            self._draw_jelly(glow, j, t)
        img += glow + _blur(glow, max(1, int(min(W, H) / 40))) * (0.9 if mode == "blocks" else 0.45)
        return img

    def _draw_jelly(self, g, j, t):
        W, H = self.W, self.H
        bright = 0.35 + 0.65 * j.depth
        tint = np.array([0.55, 0.75, 1.0], np.float32)
        col = j.col * (0.75 + 0.25 * j.depth) + tint * (1 - j.depth) * 0.25
        col2 = j.col2
        # tentacles
        self._lines(g, j.P, col * (0.55 * bright), fade=True)
        self._lines(g, j.A, (col * 0.6 + col2 * 0.4) * (0.8 * bright), fade=True, thick=j.s > 9)
        # bell
        a = j.s * (1.0 - 0.17 * j.pulse)
        b = j.s * 0.82 * (1.0 + 0.08 * j.pulse)
        x0, x1 = int(max(0, j.x - a * 1.3 - 2)), int(min(W, j.x + a * 1.3 + 3))
        y0, y1 = int(max(0, j.y - b * 1.3 - 2)), int(min(H, j.y + b * 0.6 + 3))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx + 0.5 - j.x, yy + 0.5 - j.y
        c, s = math.cos(-j.tilt), math.sin(-j.tilt)
        lx, ly = dx * c - dy * s, dx * s + dy * c
        u, v = lx / a, ly / b
        rim = 0.16 + 0.07 * np.sin(u * 9.0 + t * 3.0 + j.freq * 10)
        r = np.sqrt(u * u + v * v)
        inside = (r < 1.0) & (v < rim)
        if not inside.any():
            return
        edge = np.exp(-((1.0 - r) / 0.13) ** 2)
        skirt = np.exp(-((v - rim) / 0.09) ** 2) * (np.abs(u) < 1.0)
        body = 0.16 + 0.1 * (1 - r)
        canal = 0.08 * np.maximum(0, np.cos(np.arctan2(u, v + 1.1) * 16)) ** 6 * (r < 0.95)
        gon = sum(np.exp(-((u - gu) ** 2 + (v - gv) ** 2) / 0.018) for gu, gv in
                  ((-0.32, -0.42), (0.32, -0.42), (-0.12, -0.6), (0.12, -0.6)))
        inten = np.where(inside, body + edge * 0.75 + canal + gon * 0.55, 0.0) + skirt * 0.5 * (v < rim + 0.15)
        pulse_glow = 1.0 + 0.35 * j.pulse
        colmap = col[None, None, :] * (1 - np.clip(gon, 0, 1)[..., None] * 0.6) + \
            col2[None, None, :] * np.clip(gon, 0, 1)[..., None] * 0.6
        g[y0:y1, x0:x1] += colmap * (inten * bright * pulse_glow)[..., None] * 0.85

    def _lines(self, g, P, col, fade=True, thick=False):
        W, H = self.W, self.H
        n, L = P.shape[:2]
        sub = 3
        f = np.linspace(0, 1, sub, endpoint=False)
        a = P[:, :-1, None, :]
        b = P[:, 1:, None, :]
        pts = a + (b - a) * f[None, None, :, None]
        pts = pts.reshape(n, -1, 2)
        m = pts.shape[1]
        wgt = np.linspace(1.0, 0.08, m) ** 1.2 if fade else np.ones(m)
        wgt = np.broadcast_to(wgt[None, :], (n, m)).ravel() * 0.45
        xs = pts[..., 0].ravel()
        ys = pts[..., 1].ravel()
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xi, yi, wv = xs[ok].astype(np.int32), ys[ok].astype(np.int32), wgt[ok]
        np.add.at(g, (yi, xi), wv[:, None] * col[None, :])
        if thick:
            xi2 = np.minimum(xi + 1, W - 1)
            np.add.at(g, (yi, xi2), (wv * 0.6)[:, None] * col[None, :])
