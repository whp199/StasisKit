"""REEF - a pair of clownfish living in a bubble-tip anemone.

The fish are 3D point clouds built from a body profile (laterally compressed
elliptical sections) with fins.  A travelling wave bends the spine and drives
the tail; the pectoral fins scull.  Each fish is rotated in full 3D (so turns
pass through a head-on view), projected with perspective and z-buffered
together with the anemone, whose tentacles are chains of sphere splats.  The
tentacles sway in a current that rolls across the crown, and they lean away
from the fish (and glow a little) when it nestles in among them.

Inspired by "Clownfish & anemone" from ASCII Lab (@ascii.lab); an independent
implementation.
"""

import math

import numpy as np

from .base import Scene
from .abyss import _blur
from ..util import periodic_noise

LIGHT = np.array([-0.35, 0.8, -0.48])
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0.0, 0.0, -1.0])
HALF /= np.linalg.norm(HALF)
F_CAM = 24.0         # camera distance behind the z=0 plane
Y_CAM = 3.55         # camera height

LOOKS = {
    #                water top            water deep           tentacle            tip                 column
    "bubble-tip": ((0.07, 0.36, 0.46), (0.01, 0.07, 0.13), (0.50, 0.62, 0.30), (0.96, 0.45, 0.66), (0.55, 0.28, 0.32)),
    "magnificent": ((0.06, 0.26, 0.52), (0.01, 0.05, 0.15), (0.86, 0.80, 0.56), (0.98, 0.93, 0.74), (0.55, 0.2, 0.62)),
    "night reef": ((0.01, 0.06, 0.12), (0.0, 0.01, 0.03), (0.10, 0.20, 0.12), (0.35, 1.0, 0.45), (0.12, 0.08, 0.14)),
}
ORANGE = np.array([1.0, 0.43, 0.05])
WHITE = np.array([0.97, 0.97, 0.95])
BLACK = np.array([0.05, 0.03, 0.03])


# =============================================================================
# fish model (local units: body length 1, nose at +x, up +y, side +z)
# =============================================================================
def _h(s):
    s = np.asarray(s, np.float64)
    front = 0.2 * np.sqrt(np.clip(1 - ((0.3 - s) / 0.3) ** 2, 0, 1))
    back = 0.2 * (1 - 0.7 * np.clip((s - 0.3) / 0.7, 0, 1) ** 1.35)
    return np.where(s < 0.3, front, back)


def _x(s):
    return 0.5 - 0.78 * np.asarray(s, np.float64)


BANDS = [(0.19, 0.045, 0.02), (0.5, 0.06, -0.045), (0.86, 0.032, 0.0)]   # centre, half width, curve


def _skin(s, th):
    """Clownfish colouring for body surface params (s along, th around from the top)."""
    c = np.cos(th)
    col = np.empty(s.shape + (3,))
    col[:] = ORANGE * (0.9 + 0.1 * (-c))[..., None]           # belly a touch lighter
    col = col + np.array([0.0, 0.12, 0.08]) * np.clip(-c, 0, 1)[..., None]
    edge = 0.016
    for cen, hw, curve in BANDS:
        sc = cen + curve * (1 - np.abs(c))
        d = np.abs(s - sc)
        col = np.where((d < hw)[..., None], WHITE, col)
        col = np.where(((d >= hw) & (d < hw + edge))[..., None], BLACK, col)
    # eye (both sides), near the top of the head
    for side in (1, -1):
        ds = (s - 0.105) * 6.5
        dt = (th - side * 0.95)
        r = np.hypot(ds, dt)
        col = np.where((r < 0.42)[..., None], np.array([0.85, 0.5, 0.1]), col)
        col = np.where((r < 0.28)[..., None], np.array([0.02, 0.02, 0.03]), col)
        glint = np.hypot(ds - 0.08, dt - side * -0.1) < 0.09
        col = np.where(glint[..., None], np.array([1.0, 1.0, 1.0]), col)
    return col


def _fin_colour(v, edge_at=0.82):
    """Orange fin with a black then thin white margin (v = 0 base .. 1 edge)."""
    col = np.empty(v.shape + (3,))
    col[:] = ORANGE * 0.95
    col = np.where((v > edge_at)[..., None], BLACK, col)
    col = np.where((v > edge_at + (1 - edge_at) * 0.65)[..., None], WHITE * 0.92, col)
    return col


class FishModel:
    """Static point cloud; FishModel.pose() bends, rotates and places it."""

    def __init__(self, px_len):
        n_s = int(np.clip(px_len * 1.7, 40, 260))
        n_t = int(np.clip(px_len * 1.0, 24, 160))
        s, th = np.meshgrid(np.linspace(0.0, 1.0, n_s), np.linspace(-math.pi, math.pi, n_t, endpoint=False), indexing="ij")
        s, th = s.ravel(), th.ravel()
        h = _h(s)
        w = 0.55 * h + 0.006
        x = _x(s)
        y = h * np.cos(th)
        z = w * np.sin(th)
        eps = 1e-3
        dhdx = (_h(np.clip(s + eps, 0, 1)) - _h(np.clip(s - eps, 0, 1))) / (_x(np.clip(s + eps, 0, 1)) - _x(np.clip(s - eps, 0, 1)) - 1e-9)
        nx = -dhdx * 0.9
        ny = np.cos(th)
        nz = np.sin(th) * (h / np.maximum(w, 1e-4))
        body = dict(p=np.stack([x, y, z], 1), n=self._unit(np.stack([nx, ny, nz], 1)), c=_skin(s, th), s=s, kind=0)
        parts = [body]
        # fins: (s range, base side, height profile, rake)
        def fin(s0, s1, prof, top, rake, ns, nv):
            fs, fv = np.meshgrid(np.linspace(s0, s1, ns), np.linspace(0.0, 1.0, nv), indexing="ij")
            fs, fv = fs.ravel(), fv.ravel()
            fh = prof(fs)
            ok = fh > 0.004
            fs, fv, fh = fs[ok], fv[ok], fh[ok]
            base = _h(fs) * (0.92 if top else -0.92)
            y = base + (fv * fh if top else -fv * fh)
            x = _x(fs) - fv * fh * rake
            p = np.stack([x, y, np.zeros_like(x)], 1)
            n = np.tile(np.array([0.0, 0.0, 1.0]), (len(x), 1))
            return dict(p=p, n=n, c=_fin_colour(fv), s=fs, kind=1)

        k = max(8, int(px_len * 0.9))
        dorsal = lambda fs: np.maximum(0.075 * np.clip(1 - ((fs - 0.42) / 0.13) ** 2, 0, 1),
                                       0.12 * np.clip(1 - ((fs - 0.68) / 0.19) ** 2, 0, 1))
        anal = lambda fs: 0.095 * np.clip(1 - ((fs - 0.74) / 0.12) ** 2, 0, 1)
        pelvic = lambda fs: 0.085 * np.clip(1 - ((fs - 0.36) / 0.06) ** 2, 0, 1)
        parts.append(fin(0.28, 0.88, dorsal, True, 0.5, k, max(6, k // 5)))
        parts.append(fin(0.6, 0.87, anal, False, 0.55, k // 2, max(5, k // 6)))
        parts.append(fin(0.29, 0.43, pelvic, False, 0.9, k // 4, max(5, k // 6)))
        # caudal fin: a rounded fan behind the peduncle
        u, v = np.meshgrid(np.linspace(0, 1, max(10, k // 3)), np.linspace(-1, 1, max(12, k // 2)), indexing="ij")
        u, v = u.ravel(), v.ravel()
        keep = u <= 1.0 - 0.18 * v * v
        u, v = u[keep], v[keep]
        hh = 0.055 + 0.15 * u ** 0.8
        cp = np.stack([-0.28 - u * 0.22, v * hh, np.zeros_like(u)], 1)
        edge_v = u / (1.0 - 0.18 * v * v)
        parts.append(dict(p=cp, n=np.tile(np.array([0.0, 0.0, 1.0]), (len(u), 1)), c=_fin_colour(edge_v, 0.8),
                          s=np.ones_like(u), kind=2, u=u))
        self.parts = parts
        self.pect = self._pectoral(k)

    @staticmethod
    def _unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _pectoral(self, k):
        a, b = np.meshgrid(np.linspace(0, 1, max(6, k // 5)), np.linspace(-1, 1, max(6, k // 6)), indexing="ij")
        a, b = a.ravel(), b.ravel()
        keep = np.abs(b) <= np.sqrt(np.clip(1 - (a - 0.45) ** 2 / 0.3, 0, 1)) + 0.1
        a, b = a[keep], b[keep]
        col = ORANGE * 0.9 + 0.08
        c = np.tile(col, (len(a), 1))
        c[a > 0.85] = BLACK
        return a, b, c

    def pose(self, t_wave, amp, tail_amp, pect_ang, yaw, pitch, scale, pos):
        """World-space points, normals and colours for one frame."""
        P, N, C = [], [], []
        for part in self.parts:
            p = part["p"].copy()
            n = part["n"].copy()
            s = part["s"]
            ph = t_wave - 4.2 * s
            zoff = amp * s * s * np.sin(ph)
            dz = amp * (2 * s * np.sin(ph) - 4.2 * s * s * np.cos(ph)) / 0.78
            if part["kind"] == 2:
                # tail fin rotates about the peduncle, following the spine's angle
                alpha = math.atan(amp * (2 * math.sin(t_wave - 4.2) - 4.2 * math.cos(t_wave - 4.2)) / 0.78) * tail_amp
                x0 = p[:, 0] + 0.28
                ca, sa = math.cos(alpha), math.sin(alpha)
                p[:, 0] = -0.28 + x0 * ca
                p[:, 2] = -x0 * sa + amp * math.sin(t_wave - 4.2)
                n = np.tile(np.array([sa, 0.0, ca]), (len(p), 1))
            else:
                p[:, 2] += zoff
                bend = np.arctan(dz)
                cb, sb = np.cos(bend), np.sin(bend)
                nx, nz = n[:, 0].copy(), n[:, 2].copy()
                n[:, 0] = nx * cb + nz * sb
                n[:, 2] = -nx * sb + nz * cb
            P.append(p)
            N.append(n)
            C.append(part["c"])
        # pectoral fins: sculling paddles on both sides behind the gill
        a, b, c = self.pect
        for side in (1.0, -1.0):
            ang = pect_ang
            bx = _x(0.3)
            px = bx - a * 0.14 * math.cos(ang)
            pz = side * (0.06 + a * 0.14 * math.sin(ang))
            py = -0.035 + b * 0.04 * (0.4 + a)
            P.append(np.stack([px, py, pz], 1))
            N.append(np.tile(np.array([math.sin(ang) * side, 0.0, math.cos(ang)]), (len(a), 1)))
            C.append(c)
        p = np.concatenate(P)
        n = np.concatenate(N)
        c = np.concatenate(C)
        # pitch (about local z), then yaw (about world y), scale, translate
        cp, sp = math.cos(pitch), math.sin(pitch)
        x, y = p[:, 0] * cp - p[:, 1] * sp, p[:, 0] * sp + p[:, 1] * cp
        nx, ny = n[:, 0] * cp - n[:, 1] * sp, n[:, 0] * sp + n[:, 1] * cp
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        X = x * cy_ + p[:, 2] * sy_
        Z = -x * sy_ + p[:, 2] * cy_
        NX = nx * cy_ + n[:, 2] * sy_
        NZ = -nx * sy_ + n[:, 2] * cy_
        world = np.stack([X, y, Z], 1) * scale + pos
        nrm = np.stack([NX, ny, NZ], 1)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        return world, nrm, c


class Fish:
    def __init__(self, rng, size, region, pos):
        self.rng = rng
        self.size = size
        self.region = region          # (xmin, xmax, ymin, ymax, zmin, zmax)
        self.pos = np.array(pos, np.float64)
        self.vel = np.zeros(3)
        self.yaw = 0.0 if rng.random() < 0.5 else math.pi
        self.pitch = 0.0
        self.wave = rng.uniform(0, 6.28)
        self.pect = 0.0
        self.mode = "cruise"
        self.timer = 0.0
        self.target = self._pick()
        self.model = None

    def _pick(self):
        x0, x1, y0, y1, z0, z1 = self.region
        r = self.rng
        return np.array([r.uniform(x0, x1), r.uniform(y0, y1), r.uniform(z0, z1)])

    def update(self, dt, crown):
        r = self.rng
        self.timer -= dt
        to = self.target - self.pos
        dist = float(np.linalg.norm(to))
        if self.mode == "nestle":
            speed_want = 0.25
            self.target = crown + np.array([math.sin(self.wave * 0.05) * 0.6, 0.15 * math.sin(self.wave * 0.11), 0])
            if self.timer <= 0:
                self.mode = "cruise"
                self.target = self._pick()
                self.timer = r.uniform(6, 14)
        else:
            speed_want = 0.55 + 0.6 * min(1.0, dist / 3.0)
            if dist < 0.5 or self.timer <= 0:
                if r.random() < 0.35:
                    self.mode = "nestle"
                    self.timer = r.uniform(5, 10)
                    self.target = crown.copy()
                else:
                    self.target = self._pick()
                    self.timer = r.uniform(5, 12)
        desired = to / max(dist, 1e-6) * speed_want * min(1.0, dist / 1.2 + 0.15)
        self.vel += (desired - self.vel) * min(1.0, dt * 1.1)
        self.pos += self.vel * dt
        sp = float(np.linalg.norm(self.vel))
        hv = math.hypot(self.vel[0], self.vel[2])
        if hv > 0.12:
            want = math.atan2(-self.vel[2], self.vel[0])
            # prefer near-side views: keep the heading within ~45 degrees of side-on
            base = 0.0 if math.cos(want) >= 0 else math.pi
            dev = (want - base + math.pi) % (2 * math.pi) - math.pi
            want = base + max(-0.75, min(0.75, dev))
            d = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += d * min(1.0, dt * 1.6)
        self.pitch += (max(-0.35, min(0.35, math.atan2(self.vel[1], hv + 0.2))) - self.pitch) * min(1.0, dt * 2)
        self.speed = sp
        beat = 1.4 + 2.2 * min(1.2, sp) + (0.6 if self.mode == "nestle" else 0.0)
        self.wave += dt * beat * 2 * math.pi
        self.pect += dt * (2.6 + (2.0 if self.mode == "nestle" else 0.0)) * 2 * math.pi

    def spine(self):
        """A few world-space points along the body for tentacle interaction."""
        f = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        return np.array([self.pos + f * self.size * (0.45 - 0.22 * k) for k in range(5)])


# =============================================================================
class Reef(Scene):
    key = "reef"
    title = "REEF"
    subtitle = "a clownfish pair at home in their anemone"
    kind = "pixels"
    variants = ("bubble-tip", "magnificent", "night reef")
    ambience = "reef"
    bgfill = 0.3
    floor = 0.1
    edges = 0.9

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        self.caus = periodic_noise(128, 128, beta=2.0, seed=int(self.rng.integers(1 << 30)))
        self.fish = [Fish(self.rng, 3.0, (-4.6, 4.6, 3.1, 5.6, -2.0, 2.0), (-2.5, 4.4, 0.0)),
                     Fish(self.rng, 2.2, (-5.2, 5.2, 3.0, 5.4, 0.6, 4.0), (3.0, 4.0, 2.5))]
        self.crown = np.array([0.0, 3.65, 1.2])

    # ------------------------------------------------------------------ setup
    def resize(self, w, h):
        W, H = w, h * 2
        self.W, self.H = W, H
        self.S = min(W / 10.5, H / 6.2)
        self.cy = H * 0.5
        self._build_anemone()
        self._build_background()
        for f in self.fish:
            f.model = FishModel(f.size * self.S)
        ns = int(W * H / 160)
        self.snow = np.column_stack([self.rng.uniform(0, W, ns), self.rng.uniform(0, H, ns), self.rng.uniform(0.2, 1, ns),
                                     self.rng.uniform(0, 6.28, ns)])
        self.bubbles = np.zeros((0, 4))
        self.zbuf_shape = None

    def on_variant(self):
        if self.w:
            self._build_background()

    def project(self, P):
        p = F_CAM / (F_CAM + P[..., 2])
        sx = self.W / 2 + P[..., 0] * self.S * p
        sy = self.cy - (P[..., 1] - Y_CAM) * self.S * p
        return sx, sy, p

    def _build_anemone(self):
        rng = self.rng
        n = int(np.clip(70 + self.S * 3.0, 70, 150))
        r = np.sqrt(rng.uniform(0, 1, n)) * 1.75
        a = rng.uniform(0, 2 * math.pi, n)
        bx, bz = r * np.cos(a), r * np.sin(a) * 0.8
        by = 1.95 + 0.25 * (1 - (r / 1.75) ** 2)
        self.t_base = np.stack([bx, by, bz + 1.2], 1)
        out = np.stack([np.cos(a), np.zeros(n), np.sin(a)], 1)
        tilt = 0.25 + 0.75 * (r / 1.75)
        rest = out * tilt[:, None] + np.array([0.0, 1.0, 0.0])
        self.t_rest = rest / np.linalg.norm(rest, axis=1, keepdims=True)
        self.t_len = rng.uniform(1.3, 2.0, n)
        self.t_phase = rng.uniform(0, 6.28, n)
        self.t_push = np.zeros((n, 3))
        self.t_glow = np.zeros(n)
        self.nseg = 9

    def _build_background(self):
        W, H = self.W, self.H
        top, deep, _, _, colc = LOOKS[self.variant_name]
        y = np.linspace(0, 1, H)[:, None, None]
        bg = np.array(top) * (1 - y) ** 1.6 + np.array(deep) * (1 - (1 - y) ** 1.6)
        bg = np.broadcast_to(bg, (H, W, 3)).astype(np.float32).copy()
        # seabed region: per-pixel world coordinates on the plane y = 0
        ys = np.arange(H, dtype=np.float64)[:, None] + 0.5
        xs = np.arange(W, dtype=np.float64)[None, :] + 0.5
        denom = (ys - self.cy) / (Y_CAM * self.S)
        with np.errstate(divide="ignore", invalid="ignore"):
            pz = np.where(denom > 0, F_CAM / np.maximum(denom, 1e-6) - F_CAM, np.inf)
        pz = np.broadcast_to(pz, (H, W))
        self.sea_z = pz
        self.sea_x = (xs - W / 2) / (self.S * F_CAM / (F_CAM + np.minimum(pz, 1e4)))
        floor = np.isfinite(pz) & (pz < 60)
        self.floor_mask = floor
        fog = np.clip((np.minimum(pz, 60) + 4) / 40.0, 0, 1)
        sand = np.array([0.62, 0.56, 0.42]) * (0.32 if self.variant_name != "night reef" else 0.07)
        bgf = sand * (1 - fog[..., None]) + np.array(deep) * fog[..., None]
        bg = np.where(floor[..., None], bgf, bg).astype(np.float32)
        self.sand_fog = fog
        # distant reef silhouettes on the horizon of the seabed
        rng = np.random.default_rng(7)
        for _ in range(9):
            cx = rng.uniform(-14, 14)
            cz = rng.uniform(12, 30)
            r = rng.uniform(1.0, 3.2)
            sx, sy, p = self.project(np.array([cx, 0.0, cz]))
            rr = r * self.S * p
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            blob = (np.hypot((xx - sx) / 1.3, (yy - sy)) < rr) & (yy < sy + 1)
            shade = np.array(deep) * 0.6 + np.array(colc) * 0.05
            bg = np.where(blob[..., None], bg * 0.55 + shade * 0.45, bg).astype(np.float32)
        # the rock the anemone sits on
        rock = np.zeros((H, W), bool)
        rockz = np.full((H, W), np.inf)
        rocks = [(-0.6, 0.9, 1.4, 1.5), (0.9, 0.8, 1.0, 1.25), (0.1, 1.3, 1.6, 1.0), (-1.6, 0.5, 1.1, 0.9),
                 (1.8, 0.45, 1.3, 0.8)]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        tex = periodic_noise(H, W, beta=1.4, seed=11)
        for (cx, cyw, cz, rad) in rocks:
            sx, sy, p = self.project(np.array([cx, cyw, cz]))
            rr = rad * self.S * p
            dx, dy = (xx - sx) / rr, (yy - sy) / rr
            d2 = dx * dx + dy * dy
            inside = d2 < 1
            nz = np.sqrt(np.clip(1 - d2, 0, 1))
            lam = np.clip(-dx * LIGHT[0] + -dy * -LIGHT[1] * -1 + nz * 0.5, 0, 1)
            lam = np.clip(0.35 + 0.65 * (dx * LIGHT[0] - dy * LIGHT[1] + nz * -LIGHT[2] * -1) * 0.8, 0.25, 1)
            col = np.array([0.40, 0.30, 0.33]) * lam[..., None] * (0.7 + 0.6 * tex[..., None])
            if self.variant_name == "night reef":
                col = col * 0.3
            closer = inside & (cz < rockz)
            bg = np.where(closer[..., None], col, bg).astype(np.float32)
            rockz = np.where(closer, cz, rockz)
            rock |= inside
        self.bg = bg

    # ------------------------------------------------------------------ simulation
    def update(self, dt):
        super().update(dt)
        if not self.w:
            return
        t = self.t
        for f in self.fish:
            f.update(dt, self.crown)
        # tentacles: push away from fish bodies, glow when brushed
        tips = self.t_base + self.t_rest * (self.t_len * 0.75)[:, None]
        push = np.zeros_like(self.t_push)
        touch = np.zeros(len(tips))
        for f in self.fish:
            for c in f.spine():
                d = tips - c
                dist = np.linalg.norm(d, axis=1)
                infl = f.size * 0.42
                k = np.clip(1 - dist / infl, 0, 1)
                push += d / np.maximum(dist, 1e-3)[:, None] * (k * 1.4)[:, None]
                touch = np.maximum(touch, k)
        self.t_push += (push - self.t_push) * min(1.0, dt * 3.0)
        self.t_glow = np.maximum(self.t_glow - dt * 0.6, touch)
        # bubbles from the rock, now and then
        if self.rng.random() < dt * 0.25:
            bx = self.rng.uniform(-1.2, 1.2)
            n = int(self.rng.integers(3, 8))
            new = np.column_stack([np.full(n, bx) + self.rng.normal(0, 0.05, n), 1.2 - np.arange(n) * 0.35,
                                   np.full(n, 0.3), self.rng.uniform(0.04, 0.09, n)])
            self.bubbles = np.concatenate([self.bubbles, new])[-60:]
        if len(self.bubbles):
            b = self.bubbles
            b[:, 1] += dt * (0.9 + b[:, 3] * 6)
            b[:, 0] += np.sin(t * 3 + b[:, 1] * 2) * dt * 0.15
            self.bubbles = b[b[:, 1] < 12]
        sn = self.snow
        sn[:, 1] += dt * (1.2 + 1.8 * sn[:, 2])
        sn[:, 0] += np.sin(t * 0.6 + sn[:, 3]) * dt * 0.8
        wrap = sn[:, 1] > self.H
        sn[wrap, 1] = 0
        sn[wrap, 0] = self.rng.uniform(0, self.W, int(wrap.sum()))
        sn[:, 0] %= self.W

    # ------------------------------------------------------------------ render
    def _tentacles(self, t):
        n, k = len(self.t_base), self.nseg
        j = (np.arange(1, k + 1) / k)[None, :, None]
        wave = t * 1.1 - self.t_base[:, 0] * 0.55 + self.t_phase * 0.15
        sway = np.stack([np.sin(wave), np.zeros(n), 0.5 * np.cos(wave * 0.8 + 1.0)], 1) * 0.55
        sway += np.stack([np.sin(t * 2.3 + self.t_phase), np.zeros(n), np.cos(t * 1.9 + self.t_phase)], 1) * 0.08
        d = self.t_rest[:, None, :] + sway[:, None, :] * j ** 1.3 + self.t_push[:, None, :] * j ** 1.5
        d /= np.linalg.norm(d, axis=2, keepdims=True)
        seg = (self.t_len / k)[:, None, None]
        pts = self.t_base[:, None, :] + np.cumsum(d * seg, axis=1)
        pts = np.concatenate([self.t_base[:, None, :], pts], 1)          # (n, k+1, 3)
        # two samples per segment so the chain has no gaps
        mid = (pts[:, :-1] + pts[:, 1:]) * 0.5
        allp = np.concatenate([pts, mid], 1)
        frac = np.concatenate([np.arange(k + 1) / k, (np.arange(k) + 0.5) / k])
        frac = np.broadcast_to(frac[None, :], allp.shape[:2])
        rad = np.where(frac > 0.88, 0.085 + 0.035 * np.sin(np.pi * (frac - 0.88) / 0.12), 0.075 * (1 - 0.25 * frac))
        return allp.reshape(-1, 3), rad.reshape(-1), frac.reshape(-1), np.repeat(np.arange(n), allp.shape[1])

    def draw_pixels(self, mode):
        W, H = self.W, self.H
        t = self.t
        look = self.variant_name
        top, deep, tcol, tip, colc = LOOKS[look]
        night = look == "night reef"
        img = self.image()
        # in ASCII mode keep open water dark, so glyphs land on the creatures
        dimbg = 0.5 if mode == "ascii" else 1.0
        img[:] = self.bg * dimbg
        # light shafts from the surface
        yy, xx = self.grid()
        rc = xx - yy * 0.35
        rays = sum(a * np.exp(-((rc - ((math.sin(t * sp + i * 2.0) * 0.5 + 0.5) * W * 1.2 - W * 0.1)) / wd) ** 2)
                   for i, (sp, wd, a) in enumerate(((0.06, W * 0.05, 1.0), (0.04, W * 0.08, 0.7), (0.09, W * 0.03, 0.6))))
        img += (rays * np.clip(1 - yy / H, 0, 1) ** 1.5 * (0.05 if night else 0.12) * dimbg)[..., None] * np.array([0.6, 0.9, 1.0], np.float32)
        # caustics on the sand
        fm = self.floor_mask
        if fm.any():
            sx, sz = self.sea_x[fm], self.sea_z[fm]
            a = (np.sin(sx * 0.9 + t * 0.7) + np.sin(sz * 1.1 - t * 0.55) + np.sin((sx + sz) * 0.7 + t * 0.4)
                 + np.sin((sx - sz) * 0.8 - t * 0.6))
            caus = (1.0 - np.abs(np.sin(a * 0.9))) ** 6 * (1 - self.sand_fog[fm])
            img[fm] += (caus[:, None] * np.array([0.35, 0.45, 0.4]) * (0.15 if night else 1.0)).astype(np.float32)
        # ---- z-buffered splats: tentacles + fish
        xs_all, ys_all, zs_all, cs_all = [], [], [], []
        P, rad, frac, tid = self._tentacles(t)
        sx, sy, p = self.project(P)
        rpx = rad * self.S * p
        glow = self.t_glow[tid]
        base = np.array(tcol) * (0.55 + 0.45 * frac[:, None])
        base = np.where((frac > 0.88)[:, None], np.array(tip), base)
        if night:
            base = base + (frac > 0.88)[:, None] * np.array(tip) * 0.6
        base = base + glow[:, None] * np.array(tip) * 0.35 * (frac[:, None] ** 2)
        self._splat_spheres(sx, sy, P[:, 2], rpx, base, xs_all, ys_all, zs_all, cs_all, emissive=night)
        for f in self.fish:
            pts, nrm, col = f.model.pose(f.wave, 0.13 + 0.05 * min(1.0, f.speed), 1.0, 0.6 + 0.55 * math.sin(f.pect),
                                         f.yaw, f.pitch, f.size, f.pos)
            fx, fy, fp = self.project(pts)
            shade = self._shade(nrm, col, pts, t, night)
            xs_all.append(fx)
            ys_all.append(fy)
            zs_all.append(pts[:, 2])
            cs_all.append(shade)
        X = np.round(np.concatenate(xs_all)).astype(np.int64)
        Y = np.round(np.concatenate(ys_all)).astype(np.int64)
        Z = np.concatenate(zs_all)
        C = np.concatenate(cs_all)
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        X, Y, Z, C = X[ok], Y[ok], Z[ok], C[ok]
        if len(X):
            order = np.argsort(Z, kind="stable")
            lin = (Y * W + X)[order]
            _, first = np.unique(lin, return_index=True)
            sel = order[first]
            # underwater haze grows with distance
            fog = np.clip((Z[sel] + 3.0) / 26.0, 0, 0.55)[:, None]
            col = C[sel] * (1 - fog) + np.array(deep) * fog
            img[Y[sel], X[sel]] = col.astype(np.float32)
        # bubbles and marine snow
        if len(self.bubbles):
            b = self.bubbles
            bp = np.column_stack([b[:, 0], b[:, 1], b[:, 2]])
            bx, by, bpp = self.project(bp)
            bxi, byi = np.round(bx).astype(int), np.round(by).astype(int)
            okb = (bxi >= 0) & (bxi < W) & (byi >= 0) & (byi < H)
            np.add.at(img, (byi[okb], bxi[okb]), np.array([0.35, 0.45, 0.5], np.float32))
        sn = self.snow
        sxi, syi = sn[:, 0].astype(int) % W, np.clip(sn[:, 1].astype(int), 0, H - 1)
        np.add.at(img, (syi, sxi), ((0.05 + 0.1 * sn[:, 2])[:, None] * np.array([0.8, 0.9, 1.0])).astype(np.float32))
        if night:
            g = np.zeros_like(img)
            tipm = frac > 0.88
            gx, gy = np.round(sx[tipm]).astype(int), np.round(sy[tipm]).astype(int)
            okg = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            np.add.at(g, (gy[okg], gx[okg]), (np.array(tip) * 0.5).astype(np.float32))
            img += _blur(g, max(1, int(min(W, H) / 45))) * (1.2 if mode == "blocks" else 0.6)
        np.clip(img, 0, 1.5, out=img)
        return img

    def _shade(self, nrm, col, pts, t, night):
        n = nrm.copy()
        flip = n[:, 2] > 0                      # make two-sided fins face the camera
        n[flip] *= -1
        dif = np.clip(n @ LIGHT, 0, 1)
        spec = np.clip(n @ HALF, 0, 1) ** 28
        rim = (1 - np.abs(n[:, 2])) ** 2
        cx, cz = pts[:, 0] + pts[:, 1] * 0.3, pts[:, 2]
        a = np.sin(cx * 2.1 + t * 0.9) + np.sin(cz * 2.4 - t * 0.7) + np.sin((cx + cz) * 1.6 + t * 0.5)
        dapple = 1.0 + 0.35 * (1.0 - np.abs(np.sin(a * 0.9))) ** 6 * np.clip(n[:, 1], 0, 1)
        light = (0.32 + 0.78 * dif * dapple)[:, None]
        out = col * light + spec[:, None] * 0.45 + rim[:, None] * np.array([0.05, 0.12, 0.16])
        if night:
            out = out * 0.45 + rim[:, None] * np.array([0.02, 0.08, 0.14])
        return out

    def _splat_spheres(self, sx, sy, z, rpx, base, xs, ys, zs, cs, emissive=False):
        """Shaded sphere splats, bucketed by on-screen radius."""
        rb = np.clip(np.round(rpx).astype(int), 0, 6)
        for r in np.unique(rb):
            m = rb == r
            if not m.any():
                continue
            rr = max(0.6, float(r) + 0.25)
            R = int(math.ceil(rr))
            oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
            d2 = (ox * ox + oy * oy).astype(np.float64)
            keep = d2 <= rr * rr
            ox, oy, d2 = ox[keep], oy[keep], d2[keep]
            nz = np.sqrt(np.clip(1 - d2 / (rr * rr), 0, 1))
            nx, ny = ox / rr, -oy / rr
            dif = np.clip(nx * LIGHT[0] + ny * LIGHT[1] - nz * LIGHT[2], 0, 1)
            lit = 0.35 + 0.75 * dif
            cx = np.round(sx[m]).astype(np.int64)[:, None] + ox[None, :]
            cy = np.round(sy[m]).astype(np.int64)[:, None] + oy[None, :]
            cz = z[m][:, None] - nz[None, :] * 0.05
            col = base[m][:, None, :] * lit[None, :, None]
            if emissive:
                col = col + base[m][:, None, :] * 0.25
            xs.append(cx.ravel())
            ys.append(cy.ravel())
            zs.append(cz.ravel())
            cs.append(col.reshape(-1, 3))
