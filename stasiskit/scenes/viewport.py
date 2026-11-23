"""FORWARD VIEWPORT - the view out the front of the ship.

3D starfield streaming toward the viewer over a slowly drifting painted
nebula, framed by a dim cockpit console (stasis vitals, course, ETA).
"""

import math
from collections import deque

import numpy as np

from .base import Scene
from ..util import Drift, hexrgb, periodic_noise, smoothstep

STAR_COLORS = np.array([hexrgb(c) for c in (
    "#9bb0ff", "#aabfff", "#cad7ff", "#f8f7ff", "#fff4ea", "#ffe2b8", "#ffcc6f", "#ff9a6a",
)], np.float32)
STAR_WEIGHTS = np.array([0.07, 0.12, 0.18, 0.24, 0.18, 0.11, 0.07, 0.03])
STAR_WEIGHTS /= STAR_WEIGHTS.sum()

HEAD_GLYPHS = np.array(list(" .·:+*✦"), dtype="<U1")
HEAD_LEVELS = np.array([0.0, 0.07, 0.17, 0.3, 0.45, 0.62, 0.86])
TRAIL_GLYPHS = np.array(list("-\\|/"), dtype="<U1")

DESTINATIONS = ("TRAPPIST-1e", "KEPLER-442b", "TAU CETI f", "ROSS 128b", "LHS 1140b", "GLIESE 667Cc")

NEBULA_PALETTES = {
    "cockpit": ["#160a33", "#5a1450", "#0b4a5c", "#1a2a6e", "#6e3a18"],
    "open space": ["#0a1838", "#2b1a5e", "#0e5560", "#3d1240", "#14304a"],
    "warp dream": ["#2a0a44", "#7a1a6a", "#0a5a7a", "#3a1a8a", "#8a4a1a"],
}


class Viewport(Scene):
    key = "viewport"
    title = "FORWARD VIEWPORT"
    subtitle = "cruising the long dark between stars"
    kind = "chars"
    variants = ("cockpit", "open space", "warp dream")
    ambience = "shiphum"

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        self.dx = Drift(26.0, self.rng.integers(1 << 30))
        self.dy = Drift(34.0, self.rng.integers(1 << 30))
        self.sd = Drift(45.0, self.rng.integers(1 << 30))
        self.surge = 0.0
        self.next_surge = self.rng.uniform(70, 140)
        self.v = 0.06
        self.c_frac = 0.21 + self.v * 0.55     # telemetry; render may run before update()
        self.dist_ly = 4.2207 + self.rng.uniform(0, 3)
        self.dest = DESTINATIONS[self.rng.integers(len(DESTINATIONS))]
        self.eta_s = self.rng.uniform(30, 60) * 365.25 * 86400
        self.heart = 0.0
        self.ecg = deque([0.0] * 64, maxlen=64)
        self.ecg_acc = 0.0
        self._neb_variant = None

    # ------------------------------------------------------------------
    def resize(self, w, h):
        rng = self.rng
        n = int(w * h / 6.5)
        self.n = n
        self.aspect = (2.0 * h) / w
        self.X = rng.uniform(-1.5, 1.5, n)
        self.Y = rng.uniform(-1.5, 1.5, n) * self.aspect
        self.Z = rng.uniform(0.03, 1.0, n)
        self.col = STAR_COLORS[rng.choice(len(STAR_COLORS), n, p=STAR_WEIGHTS)]
        self.mag = rng.uniform(0.25, 1.0, n) ** 1.6
        self.tw_ph = rng.uniform(0, 2 * math.pi, n)
        self.tw_f = rng.uniform(0.6, 3.2, n)
        self._build_nebula()

    def on_variant(self):
        if self.w:
            self._build_nebula()

    def _build_nebula(self):
        w, h = self.w, self.h
        H2, W2 = h * 2, w * 2
        seed = int(self.rng.integers(1 << 30))
        n1 = periodic_noise(H2, W2, beta=3.2, seed=seed)
        n2 = periodic_noise(H2, W2, beta=2.1, seed=seed + 1)
        n3 = periodic_noise(H2, W2, beta=3.6, seed=seed + 2)
        n4 = periodic_noise(H2, W2, beta=2.6, seed=seed + 3)
        dens = smoothstep(0.30, 0.95, n1) ** 1.35 * (0.35 + 0.65 * n2 ** 1.3)
        dens *= 1.0 - 0.65 * smoothstep(0.62, 0.88, n4)
        pal = np.array([hexrgb(c) for c in NEBULA_PALETTES[self.variant_name]], np.float32)
        hpos = np.clip(n3 * (len(pal) - 1.001), 0, len(pal) - 1.001)
        i0 = hpos.astype(np.int32)
        fr = (hpos - i0)[..., None]
        col = pal[i0] * (1 - fr) + pal[i0 + 1] * fr
        strength = 1.55 if self.variant_name != "warp dream" else 1.8
        tex = col * dens[..., None] * strength + np.array([0.004, 0.006, 0.016], np.float32)
        # glass glare: a faint diagonal sheen (screen space, not scrolled)
        self.nebula = (tex[0::2] + tex[1::2]) * 0.5     # (h, 2w, 3)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        band = (xx / w - yy / h * 0.55 - 0.18)
        self.glare = (np.exp(-(band / 0.05) ** 2) * 0.018 + np.exp(-((band - 0.11) / 0.02) ** 2) * 0.01)
        # static far stars embedded in the nebula
        k = int(w * h / 14)
        self.fx = self.rng.uniform(0, W2 / 2 * 2, k) % (2 * w)
        self.fy = self.rng.integers(0, h, k)
        self.fb = self.rng.uniform(0.15, 0.55, k) ** 1.4
        self.fc = STAR_COLORS[self.rng.choice(len(STAR_COLORS), k, p=STAR_WEIGHTS)]
        self._neb_variant = self.variant_name

    # ------------------------------------------------------------------
    def _target_speed(self):
        if self.variant_name == "warp dream":
            return 0.34 + 0.12 * self.sd.value() + 0.35 * self.surge
        return 0.055 + 0.018 * self.sd.value() + 0.24 * self.surge

    def update(self, dt):
        super().update(dt)
        self.dx.step(dt)
        self.dy.step(dt)
        self.sd.step(dt)
        # occasional gentle surge (a slow thrust burn)
        self.next_surge -= dt
        if self.next_surge <= 0:
            self.next_surge = self.rng.uniform(80, 160)
            self.surge_t = 0.0
        if hasattr(self, "surge_t"):
            self.surge_t += dt
            st = self.surge_t
            dur = 16.0
            self.surge = math.sin(math.pi * min(1.0, st / dur)) ** 2 if st < dur else 0.0
            if st >= dur:
                del self.surge_t
        self.v += (self._target_speed() - self.v) * min(1.0, dt * 0.6)
        if not self.w:
            return
        self.Z -= self.v * dt
        f = self.w * 0.5
        px = self.X / self.Z * f
        py = self.Y / self.Z * f * 0.5
        dead = (self.Z < 0.018) | (np.abs(px) > self.w * 0.62 + 2) | (np.abs(py) > self.h * 0.62 + 2)
        k = int(dead.sum())
        if k:
            rng = self.rng
            self.X[dead] = rng.uniform(-1.5, 1.5, k)
            self.Y[dead] = rng.uniform(-1.5, 1.5, k) * self.aspect
            self.Z[dead] = rng.uniform(0.82, 1.0, k)
            self.col[dead] = STAR_COLORS[rng.choice(len(STAR_COLORS), k, p=STAR_WEIGHTS)]
            self.mag[dead] = rng.uniform(0.25, 1.0, k) ** 1.6
        # ship telemetry
        c_frac = 0.21 + self.v * 0.55
        self.c_frac = c_frac
        self.dist_ly += c_frac * dt / (365.25 * 86400) * 3.0e4  # time-compressed for show
        self.eta_s = max(0.0, self.eta_s - dt * 3.0e4)
        # slow sleeping heartbeat ~48 bpm
        self.heart += dt * 48 / 60.0
        self.ecg_acc += dt
        while self.ecg_acc > 0.08:
            self.ecg_acc -= 0.08
            ph = self.heart % 1.0
            v = 0.12
            v += 0.9 * math.exp(-((ph - 0.20) / 0.018) ** 2)
            v -= 0.25 * math.exp(-((ph - 0.24) / 0.02) ** 2)
            v += 0.22 * math.exp(-((ph - 0.48) / 0.06) ** 2)
            v += 0.06 * math.exp(-((ph - 0.08) / 0.03) ** 2)
            self.ecg.append(v)

    # ------------------------------------------------------------------
    def draw_chars(self, cv, mode):
        w, h = self.w, self.h
        if self._neb_variant != self.variant_name:
            self._build_nebula()
        t = self.t
        cx = w * 0.5 + self.dx.value() * w * 0.05
        cy = h * 0.5 + self.dy.value() * h * 0.07
        # --- nebula background (slow drift tied to the wandering heading) ---
        off = (t * 0.22 + (cx - w * 0.5) * 0.6) % (2 * w)
        i0 = int(off)
        fr = off - i0
        cols = (np.arange(w) + i0) % (2 * w)
        cols1 = (cols + 1) % (2 * w)
        neb = self.nebula[:, cols] * (1 - fr) + self.nebula[:, cols1] * fr
        if self.variant_name == "warp dream":
            neb = neb * (0.85 + 0.15 * math.sin(t * 0.4))
        cv.ch[:] = " "
        cv.bg[:] = neb + self.glare[..., None] * np.array([0.7, 0.85, 1.0], np.float32)
        cv.fg[:] = 0
        # --- far static stars ---
        fx = ((self.fx - off) % (2 * w)).astype(np.int32)
        vis = fx < w
        if vis.any():
            fxv, fyv = fx[vis], self.fy[vis]
            tw = 0.75 + 0.25 * np.sin(t * 0.7 + fxv * 1.3 + fyv * 2.1)
            b = self.fb[vis] * tw
            cv.ch[fyv, fxv] = np.where(b > 0.42, "·", ".")
            cv.fg[fyv, fxv] = self.fc[vis] * (0.25 + 0.75 * b)[:, None]

        # --- moving stars ---
        f = w * 0.5
        Z = self.Z
        px = cx + self.X / Z * f
        py = cy + self.Y / Z * f * 0.5
        tw = 0.82 + 0.18 * np.sin(t * self.tw_f + self.tw_ph)
        b = self.mag * np.clip((1.05 - Z) ** 1.35, 0, 1) * tw * 1.55
        # trails
        trail_t = 0.02 + self.v * 2.2
        Zp = Z + self.v * trail_t
        ppx = cx + self.X / Zp * f
        ppy = cy + self.Y / Zp * f * 0.5
        ddx = px - ppx
        ddy = (py - ppy) * 2.0
        tlen = np.hypot(ddx, ddy)
        tm = (tlen > 2.2) & (b > 0.14)
        if tm.any():
            K = 10
            idx = np.flatnonzero(tm)
            ang = np.arctan2(ddy[idx], ddx[idx])
            octant = (np.round(ang / (math.pi / 4)).astype(np.int32)) % 4
            glyph = TRAIL_GLYPHS[np.array([0, 1, 2, 3])[octant]]
            ts = np.linspace(0.12, 1.0, K)[None, :]
            tx = (px[idx, None] - (px[idx, None] - ppx[idx, None]) * ts)
            ty = (py[idx, None] - (py[idx, None] - ppy[idx, None]) * ts)
            tb = b[idx, None] * (1 - ts) ** 1.4 * 0.85
            txi = np.round(tx).astype(np.int32).ravel()
            tyi = np.round(ty).astype(np.int32).ravel()
            tbf = tb.ravel()
            gl = np.repeat(glyph, K)
            colr = np.repeat(self.col[idx], K, axis=0)
            ok = (txi >= 0) & (txi < w) & (tyi >= 0) & (tyi < h) & (tbf > 0.05)
            order = np.argsort(tbf[ok])
            txi, tyi, tbf, gl, colr = txi[ok][order], tyi[ok][order], tbf[ok][order], gl[ok][order], colr[ok][order]
            cv.ch[tyi, txi] = gl
            cv.fg[tyi, txi] = colr * (0.3 + 0.7 * tbf)[:, None]
        # heads
        pxi = np.round(px).astype(np.int32)
        pyi = np.round(py).astype(np.int32)
        ok = (pxi >= 0) & (pxi < w) & (pyi >= 0) & (pyi < h) & (b > HEAD_LEVELS[1])
        if ok.any():
            bo = b[ok]
            order = np.argsort(bo)
            xs, ys, bo, co = pxi[ok][order], pyi[ok][order], bo[order], self.col[ok][order]
            gi = np.searchsorted(HEAD_LEVELS, bo, side="right") - 1
            cv.ch[ys, xs] = HEAD_GLYPHS[gi]
            cv.fg[ys, xs] = np.clip(co * (0.35 + 0.75 * bo)[:, None], 0, 1)
            glow = bo > 0.55
            if glow.any():
                cv.bg[ys[glow], xs[glow]] += co[glow] * (0.12 * bo[glow])[:, None]

        if self.variant_name == "cockpit":
            self._draw_cockpit(cv)

    # ------------------------------------------------------------------
    def _draw_cockpit(self, cv):
        w, h = self.w, self.h
        t = self.t
        frame = (0.10, 0.11, 0.135)
        frame_hi = (0.17, 0.18, 0.21)
        dim = (0.36, 0.42, 0.48)
        cyan = (0.35, 0.85, 0.95)
        amber = (1.0, 0.72, 0.3)
        green = (0.4, 1.0, 0.6)
        # window bezel
        cv.text(0, 0, "▀" * w, frame)
        for y in range(1, h):
            cv.put(0, y, "▌", frame)
            cv.put(w - 1, y, "▐", frame)
        cv.put(1, 1, "◤", frame)
        cv.put(w - 2, 1, "◥", frame)
        # console
        ch_h = max(4, min(6, h // 7))
        y0 = h - ch_h
        cv.text(0, y0, "▄" * w, frame)
        cv.put(1, y0 - 1, "◣", frame)
        cv.put(w - 2, y0 - 1, "◢", frame)
        grad = np.linspace(0.9, 0.55, ch_h - 1)[:, None, None]
        cv.ch[y0 + 1:h] = " "
        cv.bg[y0 + 1:h] = np.array(frame, np.float32) * grad
        cv.text(2, y0 + 1, "─" * (w - 4), frame_hi)
        if w < 60:
            return
        # tiny blinking indicator lights along the top rail
        for i, x in enumerate(range(4, w - 4, 9)):
            on = (math.sin(t * (0.3 + 0.07 * (i % 5)) + i * 1.7) > 0.6)
            c = (0.25, 0.6, 1.0) if i % 3 else (1.0, 0.5, 0.25)
            cv.put(x, y0 + 1, "•", c if on else tuple(v * 0.25 for v in c))
        r1, r2 = y0 + 2, y0 + 3
        # left telemetry
        vel, dist = f"{self.c_frac:0.4f} c", f"{self.dist_ly:0.5f} ly"
        cv.text(3, r1, "VELOCITY", dim)
        cv.text(13, r1, vel, cyan)
        cv.text(3, r2, "DISTANCE", dim)
        cv.text(13, r2, dist, cyan)
        # right: course & eta
        days = self.eta_s / 86400
        years = int(days // 365.25)
        d = int(days - years * 365.25)
        rx = w - 30
        lo = 15 + max(len(vel), len(dist))      # first free column after the left block
        if rx >= lo:
            cv.text(rx, r1, "COURSE", dim)
            cv.text(rx + 8, r1, self.dest[:20], amber)
            cv.text(rx, r2, "ETA", dim)
            cv.text(rx + 8, r2, f"{years}y {d:03d}d", amber)
        # center: stasis vitals with ECG sparkline, fitted between the side blocks
        hi = rx - 2
        span = min(40, w // 3, hi - lo)
        if span < 18:
            return
        cx0 = min(max(lo, w // 2 - span // 2), hi - span)
        bars = " ▁▂▃▄▅▆▇█"
        vals = list(self.ecg)[-span:]
        line = "".join(bars[max(0, min(8, int(v * 8.5 + 0.5)))] for v in vals)
        cv.text(cx0, r1, line, (0.35, 0.95, 0.55))
        beat = (self.heart % 1.0)
        heart_col = (1.0, 0.35, 0.45) if 0.17 < beat < 0.34 else (0.55, 0.2, 0.25)
        cv.text(cx0, r2, "♥", heart_col)
        cv.text(cx0 + 2, r2, "48 BPM", dim)
        if span >= 33:
            cv.text(cx0 + 10, r2, "CRYO -196.2°C", dim)
        led = green if (t % 2.4) < 1.9 else (0.1, 0.25, 0.15)
        cv.text(cx0 + span - 9, r2, "●", led)
        cv.text(cx0 + span - 7, r2, "STASIS", (0.6, 0.9, 0.7))
