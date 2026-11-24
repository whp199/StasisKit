"""TIDELINE - the sea at night, seen from the beach.

Each pixel below the horizon is traced to a point on the water (eye 1.8 m
above the sea).  The open sea is a sum of Gerstner swells, faded out where
they are too small to resolve at that distance.  The water mirrors the
night sky, with Fresnel making the far sea brighter, and the moon or a
lighthouse is caught as glitter.

Breakers are separate events.  Each one shoals toward the shore at
shallow-water speed and breaks in deeper water the taller it is.  It turns
into a band of white water that frays into lace, then runs up the sand as a
thin scalloped sheet, leaving a fading foam line.  Every breaker also tells
the audio engine when it will crash, so the sound lands with the picture.

Inspired by "NIGHT TIDE" by JIN2 (@jin_22ee); an independent implementation.
"""

import math

import numpy as np

from .base import Scene
from .abyss import _blur
from ..util import periodic_noise

G = 9.81
EYE = 1.8            # eye height above the water, m
Z_SHORE = 3.6        # resting water line, m from the eye
SLOPE = 0.05         # beach / seabed slope

# open-sea swells: wavelength m, amplitude m, direction (deg from straight-in)
SWELLS = [(38, 0.20, 4), (24, 0.14, -13), (17, 0.09, 21), (11, 0.065, -26), (7.5, 0.045, 33),
          (5.0, 0.032, -41), (3.3, 0.021, 9), (2.2, 0.014, -57), (1.45, 0.009, 63)]

LOOKS = {
    #               sky horizon            sky zenith            deep water             foam
    "moon path": ((0.06, 0.085, 0.15), (0.008, 0.012, 0.03), (0.004, 0.012, 0.03), (0.85, 0.9, 0.97)),
    "new moon": ((0.025, 0.04, 0.08), (0.004, 0.006, 0.016), (0.002, 0.008, 0.02), (0.25, 0.95, 1.0)),
    "lighthouse": ((0.05, 0.07, 0.12), (0.007, 0.01, 0.025), (0.003, 0.01, 0.026), (0.8, 0.86, 0.94)),
}
MOON_DIR = np.array([-math.sin(0.28) * math.cos(0.15), math.sin(0.15), math.cos(0.28) * math.cos(0.15)])
LAMP_POS = np.array([38.0, 9.0, 150.0])     # lighthouse lamp: x, height, distance


class Breaker:
    def __init__(self, rng, t):
        self.t0 = t
        self.H = rng.uniform(0.35, 1.25)
        self.z = rng.uniform(46, 60)
        self.zb = float(np.clip(7.0 + self.H * 13.0 + rng.normal(0, 1.5), 7.5, 26.0))
        self.xc = rng.uniform(-12, 12)
        self.Lx = rng.uniform(10, 28)
        self.angle = rng.uniform(-0.12, 0.12)
        self.curv = rng.uniform(-0.004, 0.008)
        self.broke = None         # time it broke
        self.shore_t = None       # time it reached the shore
        self.runup = rng.uniform(0.5, 1.1) * (0.6 + 0.4 * self.H)
        self.seed = rng.uniform(0, 255, 2)
        self.spray = None

    @staticmethod
    def speed(z):
        return max(1.3, math.sqrt(G * SLOPE * max(0.2, z - Z_SHORE + 0.4)))

    def eta_crash(self):
        """Seconds from now until it breaks (numeric shoaling)."""
        z, t = self.z, 0.0
        while z > self.zb and t < 60:
            z -= self.speed(z) * 0.1
            t += 0.1
        return t


class Tideline(Scene):
    key = "surf"
    title = "TIDELINE"
    subtitle = "night surf from the shore"
    kind = "pixels"
    variants = ("moon path", "new moon", "lighthouse")
    ambience = "surf"
    ramp = " .,:;+=ixtzjnaeuo0O8%B$#"
    bgfill = 0.32
    floor = 0.11
    edges = 0.0

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        self.breakers = []
        self.next_break = 0.5
        self.foamlines = []       # (x0 world center, z, born, strength, Lx, seed)
        self.lace = periodic_noise(256, 256, beta=1.7, seed=int(self.rng.integers(1 << 30)))
        # a few breakers already on their way, so the view never opens empty
        for k in range(3):
            b = Breaker(self.rng, -k * 3.0)
            b.z = b.zb + 6 + k * 10
            self.breakers.append(b)

    def _lace_at(self, u, v):
        """Bilinear sample of the tileable lace texture at texel coords (u, v)."""
        u0, v0 = np.floor(u), np.floor(v)
        fu, fv = u - u0, v - v0
        i0, j0 = u0.astype(np.int64) % 256, v0.astype(np.int64) % 256
        i1, j1 = (i0 + 1) % 256, (j0 + 1) % 256
        L = self.lace
        return ((L[j0, i0] * (1 - fu) + L[j0, i1] * fu) * (1 - fv)
                + (L[j1, i0] * (1 - fu) + L[j1, i1] * fu) * fv)

    def _rows(self, zmin, zmax):
        """Water-row slice [r0, r1) whose distance lies in [zmin, zmax] (zrow falls with row)."""
        nz = -self.zrow
        return int(np.searchsorted(nz, -zmax, "left")), int(np.searchsorted(nz, -zmin, "right"))

    @staticmethod
    def _front_span(b):
        """Nearest and farthest distance of a breaker's front over its visible length."""
        u = np.linspace(-1.5 * b.Lx, 1.5 * b.Lx, 41)
        zf = b.z + b.angle * u + b.curv * u * u
        return float(zf.min()), float(zf.max())

    # ------------------------------------------------------------------ setup
    def resize(self, w, h):
        W, H = w, h * 2
        self.W, self.H = W, H
        self.yh = int(round(H * 0.30))
        self.f = H * 1.0
        ys = np.arange(self.yh + 1, H, dtype=np.float64) + 0.5
        self.rows = ys
        self.zrow = EYE * self.f / (ys - self.yh)                   # distance of each water row
        xs = np.arange(W, dtype=np.float64) + 0.5 - W / 2
        self.xw = self.zrow[:, None] * xs[None, :] / self.f          # lateral metres per pixel
        self.zw = np.broadcast_to(self.zrow[:, None], self.xw.shape)
        fp = np.maximum(self.zrow / self.f, self.zrow ** 2 / (EYE * self.f))   # metres per pixel
        self.sw = []
        for lam, amp, deg in SWELLS:
            k = 2 * math.pi / lam
            th = math.radians(deg)
            lod = np.clip((lam / fp - 2.5) / 3.0, 0.0, 1.0)[:, None]
            self.sw.append((k, amp, math.sin(th), math.cos(th), math.sqrt(G * k), self.rng.uniform(0, 6.283), lod))
        vx, vy, vz = self.xw, np.full_like(self.xw, -EYE), self.zw.copy()
        n = np.sqrt(vx * vx + vy * vy + vz * vz)
        self.view = np.stack([vx / n, vy / n, vz / n], -1)
        # sky
        sy = (np.arange(self.yh + 1, dtype=np.float64) + 0.5)
        self.sky_el = (self.yh - sy) / self.f                          # approx elevation (rad)
        k = max(1, int(W * self.yh / 60))
        self.stars = (self.rng.integers(0, W, k), self.rng.integers(0, max(1, self.yh - 2), k),
                      self.rng.uniform(0, 6.28, k), self.rng.uniform(0.2, 1.0, k) ** 2)
        self.spray = np.zeros((0, 5))
        self.sandc = (np.array([0.035, 0.03, 0.028]) + np.array([0.025, 0.02, 0.016])
                      * self._lace_at(self.xw * 25.0, self.zw * 25.0)[..., None])

    def on_variant(self):
        pass

    # ------------------------------------------------------------------ simulation
    def update(self, dt):
        super().update(dt)
        t = self.t
        self.next_break -= dt
        if self.next_break <= 0:
            self.next_break = self.rng.uniform(2.2, 3.6)
            b = Breaker(self.rng, t)
            self.breakers.append(b)
            # tell the audio engine when this one will crash
            pan = float(np.clip(b.xc / 18.0, -0.9, 0.9))
            self.emit("wave", layer="ambience", amp=float(0.45 + 0.5 * b.H), pan=pan,
                      rise=float(b.eta_crash()), fall=float(4.5 + 3.0 * b.H))
        keep = []
        for b in self.breakers:
            if b.shore_t is None:
                b.z -= Breaker.speed(b.z) * dt
                if b.broke is None and b.z <= b.zb:
                    b.broke = t
                    if self.w:
                        self._spawn_spray(b)
                if b.z <= Z_SHORE:
                    b.z = Z_SHORE
                    b.shore_t = t
            elif t - b.shore_t > 4.2:
                self.foamlines.append([b.xc, Z_SHORE - b.runup * 0.92, t, b.H, b.Lx, b.seed.copy()])
                continue
            keep.append(b)
        self.breakers = keep[-7:]
        self.foamlines = [f for f in self.foamlines if t - f[2] < 9.0][-6:]
        if self.w and len(self.spray):
            sp = self.spray
            sp[:, 0] += sp[:, 2] * dt
            sp[:, 1] += sp[:, 3] * dt
            sp[:, 3] += 9.0 * dt
            sp[:, 4] -= dt
            self.spray = sp[sp[:, 4] > 0]

    def _spawn_spray(self, b):
        n = int(18 + 30 * b.H)
        rng = self.rng
        xw = b.xc + rng.uniform(-b.Lx, b.Lx, n) * 0.8
        zf = b.z + b.angle * (xw - b.xc) + b.curv * (xw - b.xc) ** 2
        zf = np.maximum(zf, Z_SHORE + 0.5)
        px = xw / zf * self.f + self.W / 2
        py = self.yh + EYE * self.f / zf - b.H * 0.9 * self.f / zf
        vx = rng.uniform(-3, 3, n)
        vy = -rng.uniform(6, 16, n) * (0.4 + b.H) * (12.0 / zf)
        self.spray = np.concatenate([self.spray, np.column_stack([px, py, vx, vy, rng.uniform(0.5, 1.1, n)])])[-260:]

    # ------------------------------------------------------------------ render
    def _sky(self, rx, ry, rz, look, t):
        hor, zen, _, _ = LOOKS[look]
        el = np.clip(ry, -0.2, 1.0)
        L = (np.array(hor) * np.exp(-np.maximum(el, 0) / 0.11)[..., None] + np.array(zen))
        if look == "moon path":
            c = np.clip(rx * MOON_DIR[0] + ry * MOON_DIR[1] + rz * MOON_DIR[2], -1, 1)
            ang = np.arccos(c)
            L = L + (np.exp(-ang / 0.10) * 0.45 + np.exp(-ang / 0.03) * 1.2 + (c > 0.99985) * 9.0)[..., None] \
                * np.array([0.95, 0.92, 0.82])
        elif look == "lighthouse":
            pass
        return L

    def draw_pixels(self, mode):
        W, H, yh = self.W, self.H, self.yh
        t = self.t
        look = self.variant_name
        hor, zen, deep, foamc = LOOKS[look]
        img = self.image()
        # ---------------- sky
        el = self.sky_el
        sky = np.array(hor) * np.exp(-np.maximum(el, 0) / 0.11)[:, None] + np.array(zen)
        img[:yh + 1] = sky[:, None, :].astype(np.float32)
        sx, sy, sph, sb = self.stars
        tw = sb * (0.55 + 0.45 * np.sin(t * 1.3 + sph))
        np.add.at(img, (sy, sx), (tw[:, None] * np.array([0.8, 0.85, 1.0]) * 0.9).astype(np.float32))
        if look == "moon path":
            mx = W / 2 + MOON_DIR[0] / MOON_DIR[2] * self.f
            my = yh - MOON_DIR[1] / MOON_DIR[2] * self.f
            r = max(1.6, self.f * 0.035)
            yy, xx = np.mgrid[0:yh + 1, 0:W].astype(np.float32)
            d = np.hypot(xx - mx, yy - my)
            disc = np.clip(r - d + 0.5, 0, 1)
            shadow = np.clip(r - np.hypot(xx - mx - r * 0.55, yy - my + r * 0.25) + 0.5, 0, 1)
            img[:yh + 1] += ((disc * (1 - 0.92 * shadow))[..., None] * np.array([1.0, 0.96, 0.85])
                             + np.exp(-d / (r * 3.5))[..., None] * np.array([0.12, 0.12, 0.14])).astype(np.float32)
        if look == "lighthouse":
            self._lighthouse_sky(img, t)
        # ---------------- open sea (Gerstner swells)
        xw, zw = self.xw, self.zw
        sx_ = np.zeros_like(xw)
        sz_ = np.zeros_like(xw)
        ny = np.ones_like(xw)
        eta = np.zeros_like(xw)
        for k, A, dx, dz, om, ph, lod in self.sw:
            a = A * lod
            phi = k * (dx * xw + dz * zw) + om * t + ph
            s, c = np.sin(phi), np.cos(phi)
            eta += a * c
            sx_ += dx * k * a * s
            sz_ += dz * k * a * s
            ny -= 0.55 * k * a * c
        # ---------------- breakers: shoaling humps and white water (each only on its own row band)
        foam = np.zeros_like(xw)
        for b in self.breakers:
            if b.broke is None:
                lo, hi = self._front_span(b)
                r0, r1 = self._rows(lo - 6.0, hi + 12.0)
            else:
                lo, hi = self._front_span(b)
                r0, r1 = self._rows(lo - 1.5, hi + max(0.0, b.zb - b.z) + 2.0)
            if r1 <= r0:
                continue
            xs_, zs_ = xw[r0:r1], zw[r0:r1]
            dxl = xs_ - b.xc
            zf = b.z + b.angle * dxl + b.curv * dxl * dxl
            g = np.exp(-(dxl / b.Lx) ** 4)
            d = zs_ - zf                                  # > 0 seaward of the front
            if b.broke is None:
                closeness = float(np.clip(1.0 - (b.z - b.zb) / 18.0, 0.15, 1.0))
                wf = 2.4 - 1.9 * closeness
                w = np.where(d < 0, wf, 4.0)
                S = np.exp(-(d / w) ** 2)
                h = b.H * closeness * g
                eta[r0:r1] += h * S
                dS = -2 * d / (w * w) * S
                sz_[r0:r1] -= h * dS                           # n = (-d eta/dx, 1, -d eta/dz)
                sx_[r0:r1] += h * dS * (b.angle + 2 * b.curv * dxl)
                ny[r0:r1] -= h * S * 0.35
                # the lip starts to feather just before it breaks
                if closeness > 0.75:
                    foam[r0:r1] += np.exp(-((d + 0.15) / 0.35) ** 2) * g * (closeness - 0.75) * 1.8
            else:
                age_b = t - b.broke
                travelled = max(0.0, b.zb - b.z)
                band = np.clip(d, 0, None)
                lace = self._lace_at(xs_ * 7.0 + b.seed[0], zs_ * 5.0 + b.seed[1] - 1.5 * t)
                front = np.exp(-(d / 0.25) ** 2)                # the white wall at the bore
                trail = (d >= 0) & (d < travelled + 0.5)
                tau = band / 2.2                               # seconds since the bore passed here
                dense = np.exp(-tau / 0.45)                    # churning white water just behind it
                frayed = np.clip((lace - 0.58 + 0.4 * dense) * 4.0, 0, 1)
                residue = np.exp(-tau / 4.0) * np.clip((lace - 0.68) * 7.0, 0, 1) * 0.35
                wh = (front * 1.1 + trail * (dense * frayed + residue)) * g * min(1.0, 0.35 + b.H)
                if b.shore_t is not None:
                    wh *= max(0.0, 1.0 - (t - b.shore_t) / 3.2)
                foam[r0:r1] = np.maximum(foam[r0:r1], wh)
                # a rolling white wall right at the front
                if age_b < 3.0:
                    eta[r0:r1] += 0.35 * b.H * g * np.exp(-(d / 0.7) ** 2) * (1 - age_b / 3.0)
        # ---------------- light on the water
        n = np.stack([-sx_ * -1.0, ny, -sz_ * -1.0], -1)
        n[..., 0] = sx_
        n[..., 2] = sz_
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        v = self.view
        vn = np.sum(v * n, -1)
        r = v - 2 * vn[..., None] * n
        fres = 0.02 + 0.98 * (1.0 - np.clip(-vn, 0, 1)) ** 5
        L = self._sky(r[..., 0], r[..., 1], r[..., 2], look, t)
        water = np.array(deep) * (1 - fres[..., None]) + L * fres[..., None]
        if look == "lighthouse":
            water += self._lighthouse_glint(r, t)
        # moonlit crests: a little forward scatter on steep lit faces
        water += (np.clip(eta, 0, None) * 0.06)[..., None] * np.array(hor) * 4
        # ---------------- the beach and run-up (rows near the water line only)
        zrow = zw
        sheet = np.zeros_like(xw)
        sand_wet = np.zeros_like(xw)
        for b in self.breakers:
            if b.shore_t is None:
                continue
            r0, r1 = self._rows(Z_SHORE - b.runup - 0.3, Z_SHORE + 0.05)
            if r1 <= r0:
                continue
            xs_, zs_ = xw[r0:r1], zrow[r0:r1]
            tau = t - b.shore_t
            reach = b.runup * math.sin(math.pi * min(1.0, tau / 4.2))
            g = np.exp(-((xs_ - b.xc) / (b.Lx * 1.2)) ** 4)
            edge = Z_SHORE - reach * g + 0.07 * np.sin(xs_ * 2.1 + b.seed[0]) + 0.05 * np.sin(xs_ * 5.3 + b.seed[1])
            inside = (zs_ < Z_SHORE) & (zs_ > edge)
            sheet[r0:r1] = np.maximum(sheet[r0:r1], inside * (0.35 + 0.65 * np.exp(-(zs_ - edge) / 0.25)) * g)
            sand_wet[r0:r1] = np.maximum(sand_wet[r0:r1], ((zs_ < Z_SHORE) & (zs_ > Z_SHORE - b.runup * g)) * 1.0)
        for (xc, zl, born, hh, lx, sd) in self.foamlines:
            r0, r1 = self._rows(zl - 0.25, Z_SHORE + 0.05)
            if r1 <= r0:
                continue
            xs_, zs_ = xw[r0:r1], zrow[r0:r1]
            age = t - born
            g = np.exp(-((xs_ - xc) / (lx * 1.2)) ** 4)
            zl2 = zl + 0.07 * np.sin(xs_ * 2.1 + sd[0]) + 0.05 * np.sin(xs_ * 5.3 + sd[1])
            line = np.exp(-((zs_ - zl2) / 0.06) ** 2) * g * max(0.0, 1 - age / 9.0) * 0.55
            dots = self._lace_at(xs_ * 9.0, zs_ * 9.0 + sd[1]) > 0.55
            sheet[r0:r1] = np.maximum(sheet[r0:r1], line * dots)
            sand_wet[r0:r1] = np.maximum(sand_wet[r0:r1], ((zs_ < Z_SHORE) & (zs_ > zl2)) * g * max(0.0, 1 - age / 12.0))
        on_sand = zrow < Z_SHORE
        sandc = self.sandc
        wet_mirror = L * fres[..., None] * 0.35 + np.array(deep)
        sandpix = sandc * (1 - sand_wet[..., None] * 0.55) + wet_mirror * sand_wet[..., None]
        water = np.where(on_sand[..., None], sandpix, water)
        # ---------------- white water
        fc = np.array(foamc)
        if look == "new moon":
            fc = fc * (0.7 + 0.3 * math.sin(t * 2.0))
        white = np.clip(foam + sheet, 0, 1.2)
        water = water * (1 - np.clip(white, 0, 1)[..., None] * 0.8) + white[..., None] * fc * 0.85
        img[yh + 1:] = water.astype(np.float32)
        # spray particles
        if len(self.spray):
            sp = self.spray
            px, py = sp[:, 0].astype(np.int64), sp[:, 1].astype(np.int64)
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            np.add.at(img, (py[ok], px[ok]), (np.clip(sp[ok, 4], 0, 1)[:, None] * fc * 0.9).astype(np.float32))
        if look == "new moon":
            glow = np.zeros_like(img)
            glow[yh + 1:] = (white[..., None] * fc * 0.6).astype(np.float32)
            img += _blur(glow, max(1, int(min(W, H) / 40))) * (1.0 if mode == "blocks" else 0.5)
        np.clip(img, 0, 1.5, out=img)
        return img

    # ------------------------------------------------------------------ lighthouse
    def _lamp_screen(self):
        x = self.W / 2 + LAMP_POS[0] / LAMP_POS[2] * self.f
        y = self.yh - (LAMP_POS[1] - EYE) / LAMP_POS[2] * self.f
        return x, y

    def _beam(self, t):
        a = t * 0.75                                  # one revolution every ~8 s
        return math.sin(a), math.cos(a)              # beam heading: (sideways, away from us)

    def _lighthouse_sky(self, img, t):
        W, yh = self.W, self.yh
        lx, ly = self._lamp_screen()
        # headland silhouette under the lamp
        xs = np.arange(W, dtype=np.float64)
        hill = yh - np.clip((xs - (lx - self.f * 0.35)) / (self.f * 0.35), 0, 1) ** 0.6 * (yh - ly) * 0.55
        for x in range(W):
            top = int(max(0, hill[x]))
            if top <= yh:
                img[top:yh + 1, x] = np.array([0.006, 0.007, 0.012], np.float32)
        tw = int(round(lx))
        if 0 <= tw < W:
            img[int(ly):yh + 1, max(0, tw - 1):tw + 1] = np.array([0.02, 0.02, 0.03], np.float32)
        bs, bc = self._beam(t)
        toward = max(0.0, -bc)                         # how much it faces us
        flash = math.exp(-((math.atan2(bs, -bc)) / 0.22) ** 2)
        yy, xx = np.mgrid[0:yh + 1, 0:W].astype(np.float32)
        # the beam as a sweeping shaft of lit haze
        dirx = bs * (1.0 - 0.6 * toward)
        ln = max(1e-3, abs(dirx))
        ux, uy = dirx / ln, -0.06
        px, py = xx - lx, yy - ly
        along = px * ux + py * uy
        perp = np.abs(px * uy - py * ux)
        shaft = np.exp(-(perp / (1.2 + along * 0.05)) ** 2) * np.exp(-np.clip(along, 0, None) / (W * 0.45 * ln + 1)) * (along > 0)
        img[:yh + 1] += (shaft * (0.10 + 0.25 * (1 - toward)))[..., None].astype(np.float32) * np.array([0.9, 0.85, 0.6], np.float32)
        lamp = np.exp(-np.hypot(px, py) / (1.0 + 4.0 * flash)) * (0.6 + 2.5 * flash)
        img[:yh + 1] += lamp[..., None].astype(np.float32) * np.array([1.0, 0.92, 0.65], np.float32)

    def _lighthouse_glint(self, r, t):
        bs, bc = self._beam(t)
        flash = math.exp(-((math.atan2(bs, -bc)) / 0.22) ** 2)
        lx, ly, lz = LAMP_POS
        ld = np.array([lx, ly, lz]) / np.linalg.norm(LAMP_POS)
        c = np.clip(r[..., 0] * ld[0] + r[..., 1] * ld[1] + r[..., 2] * ld[2], -1, 1)
        spec = (np.clip(c, 0, 1) ** 900) * (0.4 + 6.0 * flash) + (np.clip(c, 0, 1) ** 60) * 0.05 * (0.3 + flash)
        return spec[..., None] * np.array([1.0, 0.9, 0.6])
