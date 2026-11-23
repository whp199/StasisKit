"""HYDROPONICS BAY - plants that sprout, grow, bloom and reseed under grow-lights.

Each plant is a stochastic branching structure (fractal trees, uncurling
fern fronds, luminous coral).  Every branch has a birth time and grows at its
own pace, so you can literally watch them grow; the circulation fans keep
them swaying.  Rendering paints batches of discs (vectorised), with a soft
bloom on flowers and glowing species.
"""

import math

import numpy as np

from .base import Scene
from .abyss import _blur
from ..util import hexrgb

SPECIES = {
    "blossom": dict(kind="tree", depth=7, branches=(2, 3), spread=(16, 34), decay=(0.70, 0.82), trunk=0.15,
                    thick=3.2, wood=("#5a3a2a", "#8a5a3c"), leaf="#5f9e4f", leaf_p=0.35,
                    flower=("#ffd1e6", "#ff9cc8", "#fff0f6"), flower_p=0.9, glow=0.25, speed=1.0),
    "fern": dict(kind="fern", fronds=(4, 6), segs=(12, 16), wood=("#2f6a34", "#6bbf59"), leaf="#79d36a",
                 glow=0.0, speed=1.0),
    "lumen coral": dict(kind="tree", depth=6, branches=(2, 2), spread=(14, 28), decay=(0.74, 0.86), trunk=0.13,
                        thick=2.6, wood=("#1d5f7a", "#5ff0ff"), leaf="#9d7bff", leaf_p=0.0,
                        flower=("#7affe0", "#c6a8ff", "#ffffff"), flower_p=0.55, glow=0.7, speed=1.0, fsize=0.6),
    "moon lily": dict(kind="lily", wood=("#2e6b3a", "#5fae5a"), leaf="#4f9e58",
                      flower=("#f4f0ff", "#c9b8ff", "#fff3b0"), glow=0.45, speed=1.0),
}
VARIANT_SPECIES = {
    "mixed garden": ["blossom", "fern", "moon lily", "blossom", "fern"],
    "blossom trees": ["blossom"],
    "fern grotto": ["fern"],
    "lumen coral": ["lumen coral", "lumen coral", "moon lily"],
}


def _rgb(c):
    return np.array(hexrgb(c), np.float32)


class Plant:
    GROW_RATE = 1.0 / 26.0     # px of branch per second, scaled by size

    def __init__(self, rng, species, x, ground, H, age=0.0):
        self.rng = rng
        self.sp = SPECIES[species]
        self.name = species
        self.x, self.ground, self.H = x, ground, H
        self.scale = H / 96.0
        self.segs = []        # parent, angle, length, thick, birth, dur, depth, curl, phase
        self.leaves = []      # seg, pos, side, size, birth
        self.flowers = []     # seg, size, birth, petals
        self.age = age
        k = self.sp["kind"]
        if k == "tree":
            self._tree()
        elif k == "fern":
            self._fern()
        else:
            self._lily()
        self.grown_at = max(s[4] + s[5] for s in self.segs) + 6.0
        self.bloom_end = self.grown_at + rng.uniform(70, 130)
        self.die_end = self.bloom_end + 22.0
        self._arrays()
        self.petals = []
        # compute geometry now, so a freshly (re)seeded plant can be drawn
        # before its first update() - e.g. when it replaces one that just died
        self.update(0.0, 0.0, 0.0)

    # --- generators --------------------------------------------------------------
    def _add(self, parent, angle, length, thick, birth, depth, curl=0.0):
        dur = length / (self.GROW_RATE * 60 * self.scale) * self.rng.uniform(0.8, 1.2)
        self.segs.append([parent, angle, length, thick, birth, dur, depth, curl, self.rng.uniform(0, 6.28)])
        return len(self.segs) - 1, birth + dur

    def _tree(self):
        sp, rng = self.sp, self.rng
        L0 = sp["trunk"] * self.H * rng.uniform(0.85, 1.15)
        maxd = sp["depth"]

        budget = 150

        def rec(parent, depth, ang, length, thick, birth):
            idx, end = self._add(parent, ang, length, thick, birth, depth)
            if depth >= maxd or length < 1.2 * self.scale or len(self.segs) > budget:
                if rng.random() < sp.get("flower_p", 0):
                    self.flowers.append([idx, rng.uniform(1.4, 2.4) * self.scale * sp.get("fsize", 1.0),
                                         end + rng.uniform(4, 30), 5])
                if rng.random() < sp.get("leaf_p", 0) + 0.3:
                    self.leaves.append([idx, 1.0, rng.choice([-1, 1]), rng.uniform(1.5, 2.6) * self.scale,
                                        end + rng.uniform(0, 6)])
                return
            n = int(rng.integers(sp["branches"][0], sp["branches"][1] + 1))
            if depth > 3 and n > 2 and rng.random() < 0.6:
                n = 2
            spread = math.radians(rng.uniform(*sp["spread"]))
            for i in range(n):
                a = (i - (n - 1) / 2) * spread * (1.2 if n == 2 else 1.0) + rng.normal(0, 0.12)
                rec(idx, depth + 1, a, length * rng.uniform(*sp["decay"]), thick * 0.72,
                    end - rng.uniform(0, 0.15) * (end - birth))
            if depth >= 2 and rng.random() < sp.get("leaf_p", 0):
                self.leaves.append([idx, rng.uniform(0.4, 0.9), rng.choice([-1, 1]),
                                    rng.uniform(1.6, 2.6) * self.scale, end])

        rec(-1, 1, rng.normal(0, 0.06), L0, sp["thick"] * self.scale, 0.0)

    def _fern(self):
        rng = self.rng
        nf = int(rng.integers(*self.sp["fronds"]))
        for f in range(nf):
            base_ang = (f - (nf - 1) / 2) * 0.42 + rng.normal(0, 0.08)
            birth = f * rng.uniform(6, 14)
            nseg = int(rng.integers(*self.sp["segs"]))
            seglen = self.H * rng.uniform(0.032, 0.046)
            parent = -1
            side_curl = -1 if base_ang < 0 else 1
            for k in range(nseg):
                curl = 0.55 * side_curl * (0.4 + k / nseg)
                bend = side_curl * 0.06 * (k / nseg) ** 1.5
                ang = base_ang if k == 0 else bend
                parent, end = self._add(parent, ang, seglen * (1 - 0.45 * k / nseg), 1.4 * self.scale * (1 - k / nseg) + 0.5,
                                        birth, 1 + k, curl=curl)
                birth = end
                if 1 < k < nseg - 1:
                    for side in (-1, 1):
                        size = (1.0 - k / nseg) * 3.4 * self.scale + 0.8
                        self.leaves.append([parent, 0.7, side, size, end + 2.0])

    def _lily(self):
        rng = self.rng
        h = self.H * rng.uniform(0.32, 0.45)
        n = 7
        parent = -1
        birth = 0.0
        for k in range(n):
            ang = rng.normal(0, 0.07) if k else rng.normal(0, 0.1)
            parent, birth = self._add(parent, ang, h / n, 1.6 * self.scale, birth, 1 + k, curl=0.0)
            if k in (1, 3):
                for side in (-1, 1):
                    self.leaves.append([parent, 0.5, side, 4.5 * self.scale, birth])
        self.flowers.append([parent, 4.2 * self.scale, birth + 8, 6])

    def _arrays(self):
        s = np.array([x[:9] for x in self.segs], np.float64)
        self.parent = s[:, 0].astype(int)
        self.ang = s[:, 1]
        self.length = s[:, 2]
        self.thick = s[:, 3]
        self.birth = s[:, 4]
        self.dur = s[:, 5]
        self.depth = s[:, 6]
        self.curl = s[:, 7]
        self.phase = s[:, 8]
        self.maxdepth = max(1.0, self.depth.max())
        lv = np.array(self.leaves, np.float64).reshape(-1, 5)
        self.lv_seg, self.lv_pos, self.lv_side, self.lv_size, self.lv_birth = (
            lv[:, 0].astype(int), lv[:, 1], lv[:, 2], lv[:, 3], lv[:, 4])
        fl = np.array(self.flowers, np.float64).reshape(-1, 4)
        self.fl_seg, self.fl_size, self.fl_birth = fl[:, 0].astype(int), fl[:, 1], fl[:, 2]
        self.fl_npet = int(fl[0, 3]) if len(fl) else 5

    # --- per frame -------------------------------------------------------------------
    def update(self, dt, t, breeze):
        self.age += dt
        a = self.age
        g = np.clip((a - self.birth) / self.dur, 0.0, 1.0)
        g = g * g * (3 - 2 * g)
        self.g = g
        flex = (self.depth / self.maxdepth) ** 1.5
        sway = breeze * 0.11 * flex + 0.035 * flex * np.sin(t * 1.3 + self.phase)
        rel = self.ang + self.curl * (1.0 - g) ** 2 * 2.2 + sway
        n = len(self.segs)
        x0 = np.empty(n)
        y0 = np.empty(n)
        x1 = np.empty(n)
        y1 = np.empty(n)
        absang = np.empty(n)
        par = self.parent
        L = self.length * g
        for i in range(n):
            p = par[i]
            if p < 0:
                absang[i] = rel[i]
                x0[i], y0[i] = self.x, self.ground
            else:
                absang[i] = absang[p] + rel[i]
                x0[i], y0[i] = x1[p], y1[p]
            x1[i] = x0[i] + math.sin(absang[i]) * L[i]
            y1[i] = y0[i] - math.cos(absang[i]) * L[i]
        self.x0, self.y0, self.x1, self.y1, self.absang = x0, y0, x1, y1, absang
        # petals fall during the late bloom
        if self.bloom_end - 30 < a < self.bloom_end + 10 and self.flowers and self.rng.random() < dt * 1.2:
            f = self.flowers[self.rng.integers(len(self.flowers))]
            i = f[0]
            self.petals.append([x1[i], y1[i], self.rng.uniform(-3, 3), self.rng.uniform(2, 5),
                                self.rng.uniform(0, 6.28), _rgb(self.sp["flower"][0])])
        for p in self.petals:
            p[0] += (p[2] + math.sin(t * 2 + p[4]) * 3) * dt
            p[1] += p[3] * dt
        self.petals = [p for p in self.petals if p[1] < self.ground + 2]

    @property
    def alpha(self):
        if self.age < self.bloom_end:
            return 1.0
        return max(0.0, 1.0 - (self.age - self.bloom_end) / (self.die_end - self.bloom_end))

    @property
    def dead(self):
        return self.age >= self.die_end and not self.petals


class Painter:
    """Accumulates discs then paints them with a small stencil, vectorised."""

    def __init__(self):
        self.x, self.y, self.r, self.c = [], [], [], []

    def add(self, x, y, r, c):
        x = np.atleast_1d(np.asarray(x, np.float64))
        n = len(x)
        self.x.append(x)
        self.y.append(np.broadcast_to(np.asarray(y, np.float64), (n,)))
        self.r.append(np.broadcast_to(np.asarray(r, np.float64), (n,)))
        self.c.append(np.broadcast_to(np.asarray(c, np.float32), (n, 3)))

    def paint(self, img, additive=False):
        if not self.x:
            return
        H, W = img.shape[:2]
        x = np.concatenate(self.x)
        y = np.concatenate(self.y)
        r = np.concatenate(self.r)
        c = np.concatenate(self.c)
        xi, yi = np.round(x).astype(np.int64), np.round(y).astype(np.int64)
        R = int(min(4, math.ceil(r.max()))) if len(r) else 0
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                d2 = dx * dx + dy * dy
                m = (d2 <= r * r + 0.3) & (xi + dx >= 0) & (xi + dx < W) & (yi + dy >= 0) & (yi + dy < H)
                if not m.any():
                    continue
                if additive:
                    np.add.at(img, (yi[m] + dy, xi[m] + dx), c[m])
                else:
                    img[yi[m] + dy, xi[m] + dx] = c[m]
        self.x, self.y, self.r, self.c = [], [], [], []


class Garden(Scene):
    key = "garden"
    title = "HYDROPONICS BAY"
    subtitle = "growing things, slowly"
    kind = "pixels"
    variants = ("mixed garden", "blossom trees", "fern grotto", "lumen coral")
    ambience = "stream"
    bgfill = 0.28
    floor = 0.06
    edges = 1.0

    def resize(self, w, h):
        self.W, self.H = w, h * 2
        self.ground = self.H - max(6, int(self.H * 0.1))
        n = max(2, min(5, w // 34))
        self.slots = [(i + 0.5) / n for i in range(n)]
        self.plants = []
        for i, s in enumerate(self.slots):
            self.plants.append(self._new_plant(s, age=self.rng.uniform(40, 160)))
        ys = np.linspace(0, 1, self.H)[:, None, None]
        self.bgimg = np.zeros((self.H, self.W, 3), np.float32)
        self.motes = np.column_stack([self.rng.uniform(0, self.W, 40), self.rng.uniform(0, self.H, 40),
                                      self.rng.uniform(0, 6.28, 40)])
        self._bg()

    def _bg(self):
        H, W = self.H, self.W
        y = np.linspace(0, 1, H)[:, None]
        x = np.linspace(-1, 1, W)[None, :]
        lum = self.variant_name == "lumen coral"
        top = np.array([0.20, 0.05, 0.22] if not lum else [0.03, 0.06, 0.16], np.float32)
        base = np.array([0.012, 0.01, 0.022], np.float32)
        glow = np.exp(-y * 3.2) * (0.75 + 0.25 * np.cos(x * 1.4))
        self.bgimg = (base + top * glow[..., None] * 0.55).astype(np.float32)

    def _new_plant(self, slot, age=0.0):
        sp = VARIANT_SPECIES[self.variant_name]
        name = sp[self.rng.integers(len(sp))]
        x = slot * self.W + self.rng.uniform(-4, 4)
        return Plant(self.rng, name, x, self.ground, self.H, age)

    def on_variant(self):
        if self.w:
            self.resize(self.w, self.h)

    def update(self, dt):
        super().update(dt)
        if not self.w:
            return
        t = self.t
        breeze = 0.6 * math.sin(t * 0.37) + 0.4 * math.sin(t * 0.83 + 1.3)
        for i, p in enumerate(self.plants):
            p.update(dt, t, breeze)
            if p.dead:
                self.plants[i] = self._new_plant(self.slots[i])
        m = self.motes
        m[:, 1] -= dt * 1.5
        m[:, 0] += np.sin(t * 0.5 + m[:, 2]) * dt * 1.2
        wrap = m[:, 1] < 0
        m[wrap, 1] = self.ground
        m[wrap, 0] = self.rng.uniform(0, self.W, int(wrap.sum()))

    def draw_pixels(self, mode):
        H, W = self.H, self.W
        t = self.t
        img = self.image()
        img[:] = self.bgimg
        lum = self.variant_name == "lumen coral"
        # light fixture
        img[0:2] = np.array([0.08, 0.06, 0.1], np.float32)
        leds = (np.arange(W) % 6 == 3)
        ledc = np.array([1.0, 0.35, 0.8] if not lum else [0.4, 0.8, 1.0], np.float32)
        img[1, leds] = ledc
        glow = np.zeros((H, W, 3), np.float32)
        pt = Painter()
        for p in self.plants:
            self._draw_plant(img, glow, pt, p, t)
        # tray
        g = self.ground
        img[g:g + 1] = np.array([0.15, 0.35, 0.45] if not lum else [0.1, 0.45, 0.6], np.float32) * (0.85 + 0.15 * math.sin(t))
        img[g + 1:] = np.array([0.07, 0.07, 0.09], np.float32)
        img[g + 2, ::9] = np.array([0.2, 0.9, 0.5], np.float32) * (0.6 + 0.4 * math.sin(t * 2))
        # motes
        mx = self.motes[:, 0].astype(int) % W
        my = np.clip(self.motes[:, 1].astype(int), 0, H - 1)
        mc = np.array([0.5, 0.35, 0.6] if not lum else [0.3, 0.7, 0.9], np.float32)
        np.add.at(img, (my, mx), mc * (0.25 + 0.15 * np.sin(t * 2 + self.motes[:, 2]))[:, None])
        img += glow + _blur(glow, max(1, int(min(W, H) / 45))) * (1.0 if mode == "blocks" else 0.5)
        return img

    def _draw_plant(self, img, glow, pt, p, t):
        a = p.alpha
        if a > 0 and (p.g > 0.001).any():
            if a < 1:   # dying: paint onto a copy, then blend it in so the plant fades into the bay
                layer = img.copy()
                self._draw_body(layer, glow, pt, p, t, a)
                img += (layer - img) * a
            else:
                self._draw_body(img, glow, pt, p, t, a)
        for px, py, _, _, ph, c in p.petals:
            pt.add(px, py, 0.6, c * 0.9)
        pt.paint(img)

    def _draw_body(self, img, glow, pt, p, t, a):
        sp = p.sp
        wood0, wood1 = _rgb(sp["wood"][0]), _rgb(sp["wood"][1])
        live = p.g > 0.001
        # branches: sample discs along each grown segment
        idx = np.flatnonzero(live)
        x0, y0, x1, y1 = p.x0[idx], p.y0[idx], p.x1[idx], p.y1[idx]
        L = np.hypot(x1 - x0, y1 - y0)
        k = np.maximum(1, np.ceil(L / 0.7)).astype(int)
        rep = np.repeat(np.arange(len(idx)), k)
        offs = np.concatenate([np.arange(n) / n for n in k]) if len(k) else np.zeros(0)
        xs = x0[rep] + (x1 - x0)[rep] * offs
        ys = y0[rep] + (y1 - y0)[rep] * offs
        dep = p.depth[idx][rep] / p.maxdepth
        thick = p.thick[idx][rep] * 0.5 * (1.0 - 0.25 * offs)
        col = wood0[None, :] * (1 - dep[:, None]) + wood1[None, :] * dep[:, None]
        # shade: darker on one side
        pt.add(xs, ys, np.maximum(thick, 0.45), col)
        pt.paint(img)
        if sp["glow"] > 0.5:
            pt.add(xs, ys, 0.0, col * 0.25 * sp["glow"] * a)
            pt.paint(glow, additive=True)
        # leaves (vectorised: every leaf is 5 discs along its midrib)
        if len(p.lv_seg):
            si = p.lv_seg
            lg = np.clip((p.age - p.lv_birth) / 10.0, 0, 1) * (p.g[si] >= 0.3)
            ok = lg > 0
            if ok.any():
                si, lg = si[ok], lg[ok]
                pos, side, size = p.lv_pos[ok], p.lv_side[ok], p.lv_size[ok]
                bx = p.x0[si] + (p.x1[si] - p.x0[si]) * pos
                by = p.y0[si] + (p.y1[si] - p.y0[si]) * pos
                fern = sp["kind"] == "fern"
                ang = p.absang[si] + side * (1.35 if fern else 0.9) + 0.1 * np.sin(t * 1.7 + si)
                f = np.linspace(0.15, 1.0, 5)[None, :]
                sz = (size * lg)[:, None]
                lx = bx[:, None] + np.sin(ang)[:, None] * sz * f
                ly = by[:, None] - np.cos(ang)[:, None] * sz * f
                lr = np.sin(np.pi * f) * sz * (0.22 if fern else 0.32) + 0.35
                shade = np.broadcast_to(0.75 + 0.35 * f, lx.shape)[..., None]
                lc = _rgb(sp["leaf"])
                autumn = 1.0 - a
                cc = (lc * (1 - autumn) + np.array([0.6, 0.35, 0.1], np.float32) * autumn) * shade
                pt.add(lx.ravel(), ly.ravel(), lr.ravel(), cc.reshape(-1, 3))
                pt.paint(img)
        # flowers (vectorised: petals are two discs each, plus a centre)
        if len(p.fl_seg):
            si = p.fl_seg
            bl = np.clip((p.age - p.fl_birth) / 14.0, 0, 1) * (p.g[si] >= 0.95)
            if p.age > p.bloom_end:
                bl = bl * a
            ok = bl > 0
            if ok.any():
                fc0, fc1, fcc = (_rgb(c) for c in sp["flower"])
                si, bl, size = si[ok], bl[ok], p.fl_size[ok]
                cx, cy = p.x1[si], p.y1[si]
                npet = p.fl_npet
                pa = (t * 0.05 + si)[:, None] + np.arange(npet)[None, :] * (2 * math.pi / npet)
                for q, rr in ((0.45, 0.55), (0.9, 0.42)):
                    d = (size * bl * q)[:, None]
                    c = fc0 * (1 - q * 0.5) + fc1 * (q * 0.5)
                    pt.add((cx[:, None] + np.cos(pa) * d).ravel(), (cy[:, None] + np.sin(pa) * d * 0.9).ravel(),
                           np.repeat(size * rr * bl, npet), c)
                pt.add(cx, cy, np.maximum(0.5, size * 0.28 * bl), fcc)
                pt.paint(img)
                gk = 0.35 + sp["glow"]
                pt.add(cx, cy, size * 0.6 * bl, fc1 * 0.3 * gk * a)
                pt.paint(glow, additive=True)
