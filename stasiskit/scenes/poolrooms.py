"""POOL ROOMS - an endless, empty, softly humming tiled pool complex.

A vectorised grid raycaster drifts through an infinite lattice of 12x12
rooms.  Each room's layout (pools, pillars) is picked by hashing its
coordinates, so the place never repeats in an obvious way.  Water floors get
animated caustics; at night, underwater lights throw caustics onto the walls.
It arrives on the ship's feed like a channel from somewhere else.
"""

import math

import numpy as np

from .base import Scene

P = 12              # room size in cells
DOOR = (5, 6, 7)    # doorway cells along each wall
WALL, FLOOR, WATER, COPING, PILLAR = 1, 0, 2, 3, 4

LOOKS = {
    #             haze               wall              trim              water               fog   light
    "daylight": ((0.80, 0.89, 0.91), (0.90, 0.95, 0.96), (0.30, 0.62, 0.70), (0.25, 0.72, 0.80), 0.060, 1.0),
    "night swim": ((0.01, 0.03, 0.05), (0.13, 0.17, 0.20), (0.06, 0.18, 0.24), (0.15, 0.85, 0.95), 0.10, 0.25),
    "vapor dusk": ((0.86, 0.70, 0.82), (0.95, 0.86, 0.92), (0.55, 0.42, 0.78), (0.35, 0.80, 0.92), 0.07, 0.9),
}


def _templates(rng, n=10):
    temps = np.zeros((n, P, P), np.int8)
    for k in range(n):
        t = np.full((P, P), FLOOR, np.int8)
        t[0, :] = WALL
        t[:, 0] = WALL
        for d in DOOR:
            t[0, d] = FLOOR
            t[d, 0] = FLOOR
        if k % 5 == 4:          # some rooms are open halls with no walls at all
            t[0, :] = FLOOR
            t[:, 0] = FLOOR
            t[0, 0] = PILLAR
        for qx, qy in ((1, 1), (8, 1), (1, 8), (8, 8)):
            kind = rng.choice(["pool", "pool", "pool", "deep", "pillars", "island", "empty"])
            if kind == "pool":
                t[qy:qy + 4, qx:qx + 4] = WATER
            elif kind == "deep":
                y0 = qy if qy == 1 else qy - 1
                x0 = qx if qx == 1 else qx - 1
                t[y0:y0 + 5, x0:x0 + 5] = WATER
            elif kind == "pillars":
                for dx in (0, 3):
                    for dy in (0, 3):
                        t[qy + dy, qx + dx] = PILLAR
            elif kind == "island":
                t[qy:qy + 4, qx:qx + 4] = WATER
                t[qy + 1:qy + 3, qx + 1:qx + 3] = FLOOR
                t[qy + 1, qx + 1] = PILLAR
        # sometimes a long lap pool across one side
        if rng.random() < 0.3:
            t[1:4, 2:11] = WATER
        # walkways along both centre lines stay dry (sometimes just a narrow causeway)
        narrow = rng.random() < 0.45
        r0, r1 = (6, 7) if narrow else (5, 8)
        t[r0:r1, 1:] = np.where(t[r0:r1, 1:] == WALL, WALL, FLOOR)
        t[1:, 5:8] = np.where(t[1:, 5:8] == WALL, WALL, FLOOR)
        if narrow:
            t[5, 1:5] = WATER
            t[7, 1:5] = WATER
            t[5, 8:12] = WATER
            t[7, 8:12] = WATER
        for d in DOOR:
            t[d, 0] = FLOOR if t[d, 0] != PILLAR else PILLAR
            t[0, d] = FLOOR if t[0, d] != PILLAR else PILLAR
        # coping: dry floor touching water
        wet = t == WATER
        near = np.zeros_like(wet)
        near[1:] |= wet[:-1]
        near[:-1] |= wet[1:]
        near[:, 1:] |= wet[:, :-1]
        near[:, :-1] |= wet[:, 1:]
        t[(t == FLOOR) & near] = COPING
        temps[k] = t
    return temps


def _hash2(a, b):
    h = (a.astype(np.int64) * 73856093) ^ (b.astype(np.int64) * 19349663)
    h ^= h >> 13
    return np.abs(h)


class PoolRooms(Scene):
    key = "poolrooms"
    title = "POOL ROOMS"
    subtitle = "interdimensional cable, channel 3"
    kind = "pixels"
    variants = ("daylight", "night swim", "vapor dusk", "channel surf")
    ambience = "pool"
    bgfill = 0.3
    floor = 0.05
    edges = 1.1

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        self.temps = _templates(self.rng)
        self.px, self.py = 6.5, 6.5
        self.heading = 0.0
        self.target_heading = 0.0
        self.last_cell = (0, 0)
        self.look = "daylight"
        self.surf_next = 75.0
        self.static_t = -10.0
        self.osd_t = 0.0
        self.channel = 3
        self.on_variant()

    def on_enter(self):
        self.osd_t = self.t

    def on_variant(self):
        v = self.variant_name
        if v != "channel surf":
            self.look = v
        self.osd_t = self.t
        self.static_t = self.t

    # -------------------------------------------------------------- world
    def cell(self, ix, iy):
        rx, ry = np.floor_divide(ix, P), np.floor_divide(iy, P)
        lx, ly = ix - rx * P, iy - ry * P
        rt = _hash2(rx, ry) % len(self.temps)
        return self.temps[rt, ly, lx]

    def update(self, dt):
        super().update(dt)
        t = self.t
        if self.variant_name == "channel surf":
            self.surf_next -= dt
            if self.surf_next <= 0:
                self.surf_next = self.rng.uniform(60, 120)
                names = [n for n in LOOKS if n != self.look]
                self.look = names[self.rng.integers(len(names))]
                self.static_t = t
                self.osd_t = t
                self.channel = int(self.rng.integers(2, 99))
        # turning at room centres
        turn_err = (self.target_heading - self.heading + math.pi) % (2 * math.pi) - math.pi
        self.heading += turn_err * min(1.0, dt * 0.55)
        speed = 0.85 * max(0.0, 1.0 - abs(turn_err) / 0.6)
        hx, hy = math.cos(self.target_heading), math.sin(self.target_heading)
        self.px += hx * speed * dt
        self.py += hy * speed * dt
        # keep on the walkway centre line, with a slight drift
        lat = 0.35 * math.sin(t * 0.11)
        if abs(hx) > 0.5:
            cy = math.floor(self.py / P) * P + 6.5 + lat
            self.py += (cy - self.py) * min(1.0, dt * 0.8)
        else:
            cx = math.floor(self.px / P) * P + 6.5 + lat
            self.px += (cx - self.px) * min(1.0, dt * 0.8)
        cell = (math.floor((self.px - 6.5) / P + 0.5), math.floor((self.py - 6.5) / P + 0.5))
        lx = (self.px - 6.5) - cell[0] * P
        ly = (self.py - 6.5) - cell[1] * P
        if cell != self.last_cell and abs(lx) < 0.3 and abs(ly) < 0.3:
            self.last_cell = cell
            r = self.rng.random()
            if r < 0.22:
                self.target_heading += math.pi / 2
            elif r < 0.44:
                self.target_heading -= math.pi / 2

    # -------------------------------------------------------------- render
    def draw_pixels(self, mode):
        H, W = self.h * 2, self.w
        t = self.t
        haze, wallc, trimc, waterc, fogk, light = LOOKS[self.look]
        haze = np.array(haze, np.float32)
        night = self.look == "night swim"
        ang = self.heading + 0.42 * math.sin(t * 0.09) + 0.1 * math.sin(t * 0.23 + 1)
        # one focal length for both axes (pixels are square), so nothing is squashed at any
        # terminal shape.  tan(hfov/2) = sqrt(W/H)/2 gives ~66 deg at 80x24 and keeps the
        # view area constant; clamped to ~29..115 deg for extreme shapes.
        tf = min(1.557, max(0.255, 0.5 * math.sqrt(W / H)))
        foc = 0.5 * W / tf
        dx, dy = math.cos(ang), math.sin(ang)
        plx, ply = -dy * tf, dx * tf
        camx = 2.0 * (np.arange(W) + 0.5) / W - 1.0
        rdx = dx + plx * camx
        rdy = dy + ply * camx
        # ---- DDA (all columns at once)
        mx = np.full(W, math.floor(self.px), np.int64)
        my = np.full(W, math.floor(self.py), np.int64)
        ddx = np.abs(1.0 / np.where(rdx == 0, 1e-9, rdx))
        ddy = np.abs(1.0 / np.where(rdy == 0, 1e-9, rdy))
        stx = np.where(rdx < 0, -1, 1)
        sty = np.where(rdy < 0, -1, 1)
        sdx = np.where(rdx < 0, (self.px - mx) * ddx, (mx + 1.0 - self.px) * ddx)
        sdy = np.where(rdy < 0, (self.py - my) * ddy, (my + 1.0 - self.py) * ddy)
        hit = np.zeros(W, bool)
        side = np.zeros(W, np.int8)
        kind = np.zeros(W, np.int8)
        for _ in range(64):
            act = ~hit
            if not act.any():
                break
            mxs = act & (sdx < sdy)
            mys = act & ~mxs
            sdx[mxs] += ddx[mxs]
            mx[mxs] += stx[mxs]
            side[mxs] = 0
            sdy[mys] += ddy[mys]
            my[mys] += sty[mys]
            side[mys] = 1
            c = self.cell(mx[act], my[act])
            solid = (c == WALL) | (c == PILLAR)
            idx = np.flatnonzero(act)
            hit[idx[solid]] = True
            kind[idx[solid]] = c[solid]
        perp = np.where(side == 0, sdx - ddx, sdy - ddy)
        perp = np.where(hit, np.maximum(perp, 1e-3), 60.0)
        wx = np.where(side == 0, self.py + perp * rdy, self.px + perp * rdx)
        wx -= np.floor(wx)
        # ---- per-pixel geometry
        bob = 0.6 * math.sin(t * 0.8)
        hor = H / 2.0 + bob
        ys = (np.arange(H, dtype=np.float32) + 0.5)[:, None]
        lineH = (foc / perp)[None, :]
        top = hor - lineH * 0.5
        v = (ys - top) / lineH
        is_wall = (v >= 0) & (v < 1)
        img = self.image()
        # floor & ceiling
        below = ys > hor
        rowd = (0.5 * foc) / np.maximum(np.abs(ys - hor), 0.5)
        fx = self.px + rowd * rdx[None, :]
        fy = self.py + rowd * rdy[None, :]
        ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        ctype = self.cell(ix, iy)
        gx, gy = fx - ix, fy - iy
        lamp = np.array([1.0, 0.98, 0.92], np.float32) * (1.15 if not night else 0.45)
        if self.look == "vapor dusk":
            lamp = np.array([1.0, 0.85, 0.95], np.float32)
        # caustics shared by water and night walls
        a = (np.sin(fx * 4.1 + t * 0.9) + np.sin(fy * 4.7 - t * 0.7) + np.sin((fx + fy) * 3.1 + t * 0.55)
             + np.sin((fx - fy) * 3.6 - t * 0.83))
        caus = (1.0 - np.abs(np.sin(a * 0.9))) ** 7
        gfade = np.exp(-rowd * 0.22)
        tile3x = np.abs((fx * 3) % 1.0 - 0.5) > 0.46
        tile3y = np.abs((fy * 3) % 1.0 - 0.5) > 0.46
        grout_f = (tile3x | tile3y)
        gk = 1.0 - 0.13 * grout_f * gfade
        floor_col = np.array(wallc, np.float32) * (0.97 * gk)[..., None]
        cop = (ctype == COPING)[..., None]
        floor_col = np.where(cop, floor_col * 1.04 + 0.02, floor_col)
        wcol = np.array(waterc, np.float32)
        ripple = 0.85 + 0.15 * np.sin(fx * 7 + t * 1.3) * np.sin(fy * 6 - t * 1.1)
        wgrid = (np.abs(((fx + 0.05 * np.sin(fy * 5 + t)) * 3) % 1.0 - 0.5) > 0.46) | \
                (np.abs(((fy + 0.05 * np.sin(fx * 5 + t)) * 3) % 1.0 - 0.5) > 0.46)
        water = wcol * ripple[..., None] * np.where(wgrid[..., None], 0.88, 1.0) + \
            caus[..., None] * np.array([0.55, 0.65, 0.6], np.float32) * (0.75 if night else 0.45)
        if night:
            water = water * 1.5 + np.array([0.02, 0.12, 0.14], np.float32)
        floor_col = np.where((ctype == WATER)[..., None], water, floor_col)
        if night:
            # light spilling from the pools onto the floor
            floor_col = floor_col + np.where(cop, 0.12, 0.0) * wcol
        # ceiling: tiles + square light panels
        lpan = ((ix % 3) == 1) & ((iy % 3) == 1) & (np.abs(gx - 0.5) < 0.32) & (np.abs(gy - 0.5) < 0.32)
        ceil_col = np.array(wallc, np.float32) * (0.92 * gk)[..., None]
        ceil_col = ceil_col * (0.75 + 0.25 * light)
        ceil_col = np.where(lpan[..., None], lamp, ceil_col)
        # wet, glossy floor: mirror image of the lamp grid (ceiling at h=1, eye at 0.5)
        mfx = self.px + 3.0 * (fx - self.px)
        mfy = self.py + 3.0 * (fy - self.py)
        mix_, miy = np.floor(mfx).astype(np.int64), np.floor(mfy).astype(np.int64)
        refl = ((mix_ % 3) == 1) & ((miy % 3) == 1) & (np.abs(mfx - mix_ - 0.5) < 0.3) & (np.abs(mfy - miy - 0.5) < 0.3)
        gloss = np.where(ctype == WATER, 0.0, 0.22) * refl * np.exp(-rowd * 0.06)
        floor_col = floor_col + gloss[..., None] * lamp
        img[:] = np.where(below[..., None], floor_col, ceil_col)
        dist_fc = rowd
        # walls
        u6 = (wx[None, :] * 6.0) % 1.0
        v6 = (v * 6.0) % 1.0
        grout_w = (u6 < 0.07) | (v6 < 0.07)
        trim = (v > 0.84) | ((v > 0.30) & (v < 0.345))
        wall_rgb = np.where(trim[..., None], np.array(trimc), np.array(wallc))
        wall_rgb = np.where(grout_w[..., None], wall_rgb * 0.8, wall_rgb)
        shade = np.where(side == 0, 1.0, 0.86)[None, :] * (1.0 - 0.12 * v * light)
        wall_rgb = wall_rgb * shade[..., None]
        if night:
            wc = (np.sin(wx[None, :] * 9 + t * 0.8) + np.sin(v * 7 - t * 0.6) + np.sin((wx[None, :] + v) * 5 + t * 0.4))
            wcaus = (1 - np.abs(np.sin(wc))) ** 6 * np.clip((v - 0.35) / 0.65, 0, 1)
            wall_rgb = wall_rgb + wcaus[..., None] * wcol * 0.7
        dist = np.where(is_wall, perp[None, :], dist_fc)
        img[:] = np.where(is_wall[..., None], wall_rgb, img)
        fog = 1.0 - np.exp(-dist * fogk)
        img[:] = img * (1 - fog[..., None]) + haze * fog[..., None]
        if night:
            img *= 0.85
        # VHS-ish touches
        st = t - self.static_t
        if st < 0.7:
            k = (1.0 - st / 0.7) ** 1.5
            noise = self.rng.random((H, W)).astype(np.float32)
            img[:] = img * (1 - k) + noise[..., None] * k * 0.9
        return img

    def draw_overlay(self, cv, mode):
        el = self.t - self.osd_t
        if el < 5.0:
            g = (0.45, 1.0, 0.55) if el < 4.0 or int(el * 6) % 2 else (0.2, 0.5, 0.25)
            cv.text(3, 2, f"CH {self.channel:02d}", g)
            cv.text(3, 3, "▶ PLAY", g)
            s = self.look.upper()
            cv.text(cv.w - len(s) - 3, 2, s, g)
