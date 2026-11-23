"""TAP TAP - an elegant hand drums long stiletto nails on a glossy table.

The hand is built from shaded, z-buffered tapered capsules (skin, glossy
nails, a gold bracelet, a velvet sleeve).  Fingers lift and drop in rolling
"gallops", lazy single taps and the odd slow scratch; every impact fires a
synchronised, binaurally panned tap sound in the audio engine and a glint
on the lacquered surface, which mirrors the whole hand.
"""

import math

import numpy as np

from .base import Scene
from .abyss import _blur
from ..util import hexrgb, periodic_noise

NAILS = {
    #                 albedo     spec  env tint   iridescent  french
    "oxblood gloss": ("#7a0a1e", 1.0, "#ffd0d8", 0.0, False),
    "midnight chrome": ("#141a2e", 1.4, "#9fd0ff", 0.25, False),
    "pearl": ("#e9dce8", 0.8, "#ffffff", 0.6, False),
    "french tip": ("#f3c9c2", 0.7, "#ffffff", 0.0, True),
    "neon cyan": ("#11c9d6", 1.0, "#e0ffff", 0.1, False),
}
SKIN = np.array(hexrgb("#e9b49a"), np.float32)
SKIN_SSS = np.array([0.22, 0.05, 0.03], np.float32)
SLEEVE = np.array(hexrgb("#2a1030"), np.float32)
GOLD = np.array(hexrgb("#e8b04a"), np.float32)
LIGHT = np.array([-0.45, -0.65, 0.62], np.float32)
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0, 0, 1.0], np.float32)
HALF /= np.linalg.norm(HALF)

MAT_SKIN, MAT_NAIL, MAT_SLEEVE, MAT_GOLD = 1, 2, 3, 4


class Nails(Scene):
    key = "nails"
    title = "TAP TAP"
    subtitle = "nail-tap ASMR channel"
    kind = "pixels"
    variants = tuple(NAILS)
    ambience = "room"
    bgfill = 0.3
    floor = 0.1
    edges = 0.9

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        self.events = []         # (time, finger, strength)
        self.next_pattern = 0.5
        self.lift = np.zeros(5)
        self.slide = np.zeros(5)
        self.glints = []         # (x, y, t0, strength)
        self.scratch = None

    # -------------------------------------------------------------- geometry
    def resize(self, w, h):
        W, H = w, h * 2
        self.W, self.H = W, H
        u = H
        self.u = u
        self.table_back = int(H * 0.60)
        # rest contact points (index .. pinky; pinky nearest the viewer)
        cx0 = min(W * 0.70, W * 0.5 + u * 0.55)
        self.contact = np.array([[cx0 - i * u * 0.075, H * (0.73 + i * 0.035)] for i in range(4)])
        self.flen = np.array([1.0, 1.08, 1.02, 0.84])
        self.base_r = np.array([0.043, 0.046, 0.044, 0.038]) * u
        # background: dark room with bokeh lights
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        y = yy / H
        bg = np.zeros((H, W, 3), np.float32)
        bg[:] = np.array([0.03, 0.02, 0.035], np.float32)
        bg += (np.exp(-((xx / W - 0.65) ** 2 + (y - 0.2) ** 2) / 0.12)[..., None]
               * np.array([0.09, 0.05, 0.06], np.float32))
        self.bokeh = []
        cols = [hexrgb(c) for c in ("#ffcf8a", "#ff9ec4", "#b89cff", "#ffd9a8", "#8fd6ff")]
        for _ in range(11):
            bx, by = rng.uniform(0, W), rng.uniform(0, self.table_back * 0.9)
            r = rng.uniform(0.04, 0.11) * u
            d = np.hypot(xx - bx, yy - by)
            m = np.clip((r - d) / 1.5 + 0.5, 0, 1) * (0.55 + 0.45 * np.clip(d / r, 0, 1) ** 2)
            col = np.array(cols[rng.integers(len(cols))], np.float32)
            self.bokeh.append((m[..., None] * col * rng.uniform(0.08, 0.2), rng.uniform(0, 6.28), rng.uniform(0.1, 0.4)))
        tb = self.table_back
        bg[tb:] = 0.0
        self.bg = bg
        veins = periodic_noise(H, W, beta=2.4, seed=int(rng.integers(1 << 30)))
        veins = np.exp(-((veins - 0.5) / 0.018) ** 2) * 0.05
        tbl = np.zeros((H, W, 3), np.float32)
        depth = np.clip((yy - tb) / max(1, H - tb), 0, 1)
        tbl[:] = (np.array([0.035, 0.028, 0.04], np.float32) * (0.6 + 0.6 * depth[..., None])
                  + veins[..., None] * np.array([0.8, 0.75, 0.9], np.float32))
        tbl[:tb] = 0.0
        self.table = tbl
        self.yy, self.xx = yy, xx

    # -------------------------------------------------------------- rhythm
    def _schedule(self, now):
        rng = self.rng
        r = rng.random()
        t = now + rng.uniform(0.05, 0.25)
        if r < 0.45:   # gallop: pinky -> index rolls
            reps = int(rng.integers(2, 6))
            gap = rng.uniform(0.075, 0.11)
            period = rng.uniform(0.62, 0.95)
            for k in range(reps):
                order = [3, 2, 1, 0] if rng.random() < 0.85 else [0, 1, 2, 3]
                for j, f in enumerate(order):
                    self.events.append((t + k * period + j * gap, f, rng.uniform(0.6, 1.0)))
            self.next_pattern = t + reps * period + rng.uniform(0.6, 2.0)
        elif r < 0.8:  # lazy single taps
            n = int(rng.integers(4, 10))
            for k in range(n):
                t += rng.uniform(0.28, 0.6)
                self.events.append((t, int(rng.integers(0, 4)), rng.uniform(0.45, 0.9)))
            self.next_pattern = t + rng.uniform(0.8, 2.2)
        elif r < 0.92:  # one finger, metronome
            f = int(rng.integers(0, 3))
            per = rng.uniform(0.32, 0.5)
            n = int(rng.integers(5, 12))
            for k in range(n):
                self.events.append((t + k * per, f, rng.uniform(0.55, 0.85)))
            self.next_pattern = t + n * per + rng.uniform(0.8, 2.0)
        else:           # slow scratch
            f = int(rng.integers(0, 3))
            self.scratch = (t, f, rng.uniform(0.8, 1.4))
            self.next_pattern = t + 2.2

    def update(self, dt):
        prev = self.t
        super().update(dt)
        if not self.w:
            return
        t = self.t
        if t >= self.next_pattern and not self.events:
            self._schedule(t)
        # fire impacts
        keep = []
        for ev in self.events:
            et, f, s = ev
            if prev < et <= t:
                cx, cy = self._tip_rest(f)
                pan = float(np.clip((cx / self.W) * 2 - 1, -1, 1))
                self.emit("tap", pan=pan, vel=s)
                self.glints.append([cx, cy, t, s])
            if et > t - 0.3:
                keep.append(ev)
        self.events = keep
        self.glints = [g for g in self.glints if t - g[2] < 0.5]
        # finger lift angles from upcoming/recent events
        lift = np.zeros(4)
        for et, f, s in self.events:
            d = t - et
            if -0.2 <= d <= 0.0:
                lift[f] = max(lift[f], 0.32 * s * math.sin(math.pi * (d + 0.2) / 0.2) ** 0.8)
            elif 0 < d < 0.12:
                lift[f] = max(lift[f], 0.03 * s * math.sin(math.pi * d / 0.12))
        idle = 0.015 * np.sin(t * np.array([0.9, 1.1, 0.8, 1.3]) + np.array([0, 1, 2, 3]))
        self.lift = lift + idle
        self.slide = np.zeros(4)
        if self.scratch:
            st, f, dur = self.scratch
            d = t - st
            if 0 <= d <= dur:
                if prev < st:
                    cx, _ = self._tip_rest(f)
                    self.emit("scratch", pan=float(cx / self.W * 2 - 1), vel=0.8, dur=dur)
                self.slide[f] = math.sin(math.pi * d / dur) * 0.05 * self.u
                self.lift[f] = 0.0
            elif d > dur:
                self.scratch = None

    def _tip_rest(self, f):
        return self.contact[f]

    # -------------------------------------------------------------- render helpers
    def _capsule(self, buf, a, b, ra, rb, z, mat, extra=None):
        zb, nrm, matb, ub = buf
        H, W = zb.shape
        r = max(ra, rb)
        x0, x1 = int(max(0, min(a[0], b[0]) - r - 1)), int(min(W, max(a[0], b[0]) + r + 2))
        y0, y1 = int(max(0, min(a[1], b[1]) - r - 1)), int(min(H, max(a[1], b[1]) + r + 2))
        if x0 >= x1 or y0 >= y1:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        px, py = xs + 0.5 - a[0], ys + 0.5 - a[1]
        bx, by = b[0] - a[0], b[1] - a[1]
        L2 = bx * bx + by * by + 1e-9
        u = np.clip((px * bx + py * by) / L2, 0, 1)
        dx, dy = px - bx * u, py - by * u
        rr = ra + (rb - ra) * u
        d2 = dx * dx + dy * dy
        inside = d2 < rr * rr
        if not inside.any():
            return
        hgt = np.sqrt(np.maximum(rr * rr - d2, 0))
        depth = z + hgt
        sub = zb[y0:y1, x0:x1]
        win = inside & (depth > sub)
        if not win.any():
            return
        sub[win] = depth[win]
        inv = 1.0 / np.maximum(rr, 1e-3)
        n = nrm[y0:y1, x0:x1]
        n[..., 0][win] = (dx * inv)[win]
        n[..., 1][win] = (dy * inv)[win]
        n[..., 2][win] = (hgt * inv)[win]
        matb[y0:y1, x0:x1][win] = mat
        ub[y0:y1, x0:x1][win] = u[win] if extra is None else extra

    def _hand(self, buf):
        u = self.u
        H, W = self.H, self.W
        t = self.t
        breathe = 0.004 * u * math.sin(t * 0.7)
        dirs = []
        tips = []
        # fingers (back to front so z increases toward the viewer)
        prox_d = np.array([math.cos(0.33), math.sin(0.33)])
        dist_d = np.array([math.cos(1.02), math.sin(1.02)])
        nail_d = np.array([math.cos(1.16), math.sin(1.16)])
        knuckles = []
        for f in range(4):
            C = self.contact[f].copy()
            C[0] += self.slide[f]
            s = self.flen[f]
            L1, L2, Ln = 0.165 * u * s, 0.15 * u * s, 0.105 * u
            K = C - nail_d * Ln - dist_d * L2 - prox_d * L1
            K[1] += breathe
            th = -self.lift[f]
            c, sn = math.cos(th), math.sin(th)
            rot = lambda v: np.array([v[0] * c - v[1] * sn, v[0] * sn + v[1] * c])
            pd, dd, nd = rot(prox_d), rot(dist_d * 1.0), rot(nail_d)
            # a little extra curl at the middle joint while lifting
            M = K + pd * L1
            T = M + dd * L2
            z = 20.0 + f * 9.0
            r0 = self.base_r[f]
            self._capsule(buf, K, M, r0, r0 * 0.9, z, MAT_SKIN)
            self._capsule(buf, M, T, r0 * 0.9, r0 * 0.78, z + 1, MAT_SKIN)
            # nail: starts mid-distal on the back of the finger, tapers to a point
            perp = np.array([dd[1], -dd[0]])
            B = M + dd * (L2 * 0.42) + perp * r0 * 0.28
            N = T + nd * Ln
            self._capsule(buf, B, T + perp * r0 * 0.2, r0 * 0.62, r0 * 0.58, z + 2.2, MAT_NAIL, None)
            self._capsule(buf, T + perp * r0 * 0.2, N, r0 * 0.58, 0.35, z + 2.2, MAT_NAIL, None)
            knuckles.append((K, r0, z))
            tips.append(N)
        # back of the hand: capsules from the wrist to each knuckle
        wrist = np.array([W * 0.5 - u * 0.62, H * 0.40 + breathe])
        kc = np.mean([k[0] for k in knuckles], axis=0)
        for K, r0, z in knuckles:
            self._capsule(buf, wrist + (kc - wrist) * 0.35, K + np.array([-0.01 * u, -0.012 * u]),
                          0.07 * u, r0 * 1.12, 44.0, MAT_SKIN)
        # broad, gently domed back of the hand over the knuckle fan
        self._capsule(buf, wrist, wrist + (kc - wrist) * 0.82 + np.array([0, -0.02 * u]),
                      0.084 * u, 0.074 * u, 50.0, MAT_SKIN)
        # thumb, tucked under toward the viewer
        th0 = wrist + np.array([0.13 * u, 0.075 * u])
        th1 = th0 + np.array([0.1 * u, 0.07 * u])
        th2 = th1 + np.array([0.045 * u, 0.055 * u])
        self._capsule(buf, th0, th1, 0.05 * u, 0.042 * u, 64.0, MAT_SKIN)
        self._capsule(buf, th1, th2, 0.042 * u, 0.036 * u, 66.0, MAT_SKIN)
        tn = th2 + np.array([0.035 * u, 0.04 * u])
        self._capsule(buf, th1 + (th2 - th1) * 0.6, tn, 0.026 * u, 0.3, 69.0, MAT_NAIL)
        # forearm + sleeve + bracelet
        arm0 = np.array([-0.4 * u, H * 0.36])
        self._capsule(buf, arm0, wrist, 0.1 * u, 0.082 * u, 50.0, MAT_SKIN)
        cuff = wrist + (arm0 - wrist) * 0.42
        self._capsule(buf, arm0 + np.array([-0.2 * u, 0]), cuff, 0.13 * u, 0.115 * u, 58.0, MAT_SLEEVE)
        bdir = (arm0 - wrist) / np.linalg.norm(arm0 - wrist)
        bc = wrist + bdir * 0.13 * u
        perp = np.array([bdir[1], -bdir[0]])
        for k in range(-5, 6):
            p = bc + perp * (k / 5.0) * 0.09 * u + bdir * 0.01 * u * math.sin(k * 1.3 + t * 0.5)
            self._capsule(buf, p, p + 0.001, 0.016 * u, 0.016 * u, 58.0 + 6 * math.cos(k / 5.0 * 1.4), MAT_GOLD)
        return tips

    # -------------------------------------------------------------- render
    def draw_pixels(self, mode):
        H, W = self.H, self.W
        t = self.t
        u = self.u
        img = self.image()
        img[:] = self.bg + self.table
        for m, ph, sp in self.bokeh:
            img += m * (0.75 + 0.25 * math.sin(t * sp + ph))
        zb = np.full((H, W), -1e9, np.float32)
        nrm = np.zeros((H, W, 3), np.float32)
        mat = np.zeros((H, W), np.int8)
        ub = np.zeros((H, W), np.float32)
        buf = (zb, nrm, mat, ub)
        self._hand(buf)
        hand = mat > 0
        if hand.any():
            col = self._shade(nrm, mat, ub, t)
            # reflection in the lacquered table
            mirror_y = int(np.mean(self.contact[:, 1]))
            refl = np.zeros_like(img)
            ys = np.arange(mirror_y, H)
            src = 2 * mirror_y - ys
            ok = (src >= 0) & (src < H)
            ys, src = ys[ok], src[ok]
            m = hand[src][..., None]
            fade = np.clip(1.0 - (ys - mirror_y) / (0.35 * H), 0, 1)[:, None, None] ** 1.5
            refl[ys] = np.where(m, col[src], 0) * fade * 0.28
            refl = _blur(refl, 1)
            tabmask = (np.arange(H) >= self.table_back)[:, None, None]
            img += refl * tabmask
            img[hand] = col[hand]
        # glints at impacts
        for gx, gy, g0, s in self.glints:
            age = t - g0
            k = max(0.0, 1.0 - age / 0.45) ** 2 * s
            gx, gy = int(gx), int(gy)
            r = int(2 + 3 * k)
            for d in range(-r, r + 1):
                a = k * (1 - abs(d) / (r + 1)) * 0.9
                if 0 <= gx + d < W and 0 <= gy < H:
                    img[gy, gx + d] += a
                if 0 <= gy + d // 2 < H and 0 <= gx < W:
                    img[gy + d // 2, gx] += a * 0.7
        return img

    def _shade(self, nrm, mat, ub, t):
        name = self.variant_name
        alb, spec, env, irid, french = NAILS[name]
        alb = np.array(hexrgb(alb), np.float32)
        env = np.array(hexrgb(env), np.float32)
        N = nrm
        ndl = np.clip(N @ LIGHT, 0, 1)
        ndh = np.clip(N @ HALF, 0, 1)
        rim = np.clip(1.0 - N[..., 2], 0, 1)
        out = np.zeros(N.shape, np.float32)
        # skin: diffuse + warm subsurface rim + faint sheen
        sk = mat == MAT_SKIN
        if sk.any():
            c = SKIN * (0.28 + 0.78 * ndl[..., None]) + SKIN_SSS * rim[..., None] ** 1.5 + 0.07 * ndh[..., None] ** 30
            out[sk] = c[sk]
        sl = mat == MAT_SLEEVE
        if sl.any():
            c = SLEEVE * (0.35 + 0.6 * ndl[..., None]) + np.array([0.25, 0.12, 0.3]) * rim[..., None] ** 3 * 0.6
            out[sl] = c[sl]
        gd = mat == MAT_GOLD
        if gd.any():
            c = GOLD * (0.3 + 0.7 * ndl[..., None]) + 0.9 * ndh[..., None] ** 40
            out[gd] = c[gd]
        nl = mat == MAT_NAIL
        if nl.any():
            base = np.broadcast_to(alb, N.shape).copy()
            if irid:
                hue = (N[..., 0] * 0.6 + N[..., 1] * 0.4 + t * 0.03) % 1.0
                rainbow = 0.5 + 0.5 * np.cos(6.283 * (hue[..., None] + np.array([0.0, 0.33, 0.67])))
                base = base * (1 - irid) + rainbow * irid * base.max()
            if french:
                tip = ub > 0.55
                base[tip] = np.array([0.98, 0.97, 0.95], np.float32)
            # glossy: diffuse + sharp highlight + soft "softbox" reflection band
            band = np.exp(-((N[..., 1] + 0.45) / 0.18) ** 2) * np.clip(N[..., 2], 0, 1)
            c = (base * (0.25 + 0.65 * ndl[..., None]) + spec * 1.1 * ndh[..., None] ** 60 * env
                 + spec * 0.35 * band[..., None] * env + 0.12 * rim[..., None] ** 2 * env)
            out[nl] = c[nl]
        return np.clip(out, 0, 1.6)
