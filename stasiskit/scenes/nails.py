"""TAP TAP - an elegant hand drums long stiletto nails on a glossy table.

The hand is sculpted as a 3D signed-distance field (smoothly blended skin,
curved stiletto nails, a beaded gold bracelet, a velvet sleeve) and raymarched
within tight projected bounds.  Fingers lift and drop in rolling "gallops",
lazy single taps and the odd slow scratch; every impact fires a synchronised,
binaurally panned tap sound in the audio engine and a glint on the lacquered
surface, which mirrors the whole hand in 3D.
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

MAT_SKIN, MAT_NAIL, MAT_SLEEVE, MAT_GOLD = 1, 2, 3, 4

# Right-hand finger proportions (Index, Middle, Ring, Pinky)
_FLEN = (0.92, 1.00, 0.94, 0.80)
_R_BASE = (0.043, 0.045, 0.042, 0.036)
_NAIL_LEN = (0.168, 0.178, 0.174, 0.152)
_SPLAYS = (math.radians(9.0), math.radians(1.0), math.radians(-7.5), math.radians(-16.5))
_PIP_BENDS = (math.radians(32.0), math.radians(38.0), math.radians(33.0), math.radians(22.0))
_DIP_BENDS = (math.radians(18.0), math.radians(21.0), math.radians(18.0), math.radians(12.0))
_NAIL_BENDS = (math.radians(5.0), math.radians(5.0), math.radians(5.0), math.radians(4.0))
_MCP_BASE = (
    np.array([0.455, 0.294, 0.152], np.float32),   # index (nearest)
    np.array([0.482, 0.310, 0.048], np.float32),   # middle (longest / arch peak)
    np.array([0.458, 0.294, -0.058], np.float32),  # ring
    np.array([0.422, 0.268, -0.165], np.float32),  # pinky (farthest)
)


def _smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.minimum(a, b) - h * h * (0.25 * k)


def _pack_capsules(caps):
    a = np.array([c[0] for c in caps], np.float32)
    b = np.array([c[1] for c in caps], np.float32)
    ba = b - a
    inv_ba2 = (1.0 / (np.sum(ba * ba, axis=1) + 1e-8)).astype(np.float32)[:, None]
    ax, ay, az = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    bax, bay, baz = ba[:, 0:1], ba[:, 1:2], ba[:, 2:3]
    ra = np.array([c[2] for c in caps], np.float32)[:, None]
    dr = (np.array([c[3] for c in caps], np.float32) - np.array([c[2] for c in caps], np.float32))[:, None]
    return ax, ay, az, bax, bay, baz, bax * inv_ba2, bay * inv_ba2, baz * inv_ba2, ra, dr


def _eval_capsules(px, py, pz, packed):
    ax, ay, az, bax, bay, baz, bax_s, bay_s, baz_s, ra, dr = packed
    pax = px - ax
    pay = py - ay
    paz = pz - az
    h = np.clip(pax * bax_s + pay * bay_s + paz * baz_s, 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    dz = paz - baz * h
    return np.sqrt(dx * dx + dy * dy + dz * dz + 1e-10) - (ra + dr * h), h


def _eval_flat_capsules(px, py, pz, packed, nrm_x, nrm_y, nrm_z, flat_k2=1.45, inv_flat=0.64):
    ax, ay, az, bax, bay, baz, bax_s, bay_s, baz_s, ra, dr = packed
    pax = px - ax
    pay = py - ay
    paz = pz - az
    h = np.clip(pax * bax_s + pay * bay_s + paz * baz_s, 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    dz = paz - baz * h
    dn = dx * nrm_x + dy * nrm_y + dz * nrm_z
    d2 = dx * dx + dy * dy + dz * dz + flat_k2 * dn * dn
    return (np.sqrt(np.maximum(d2, 1e-10)) - (ra + dr * h)) * inv_flat, h


def _finger_chain(K, L1, L2, L3, Ln, splay, th1, b2, b3, bn):
    cs, ss = math.cos(splay), math.sin(splay)
    c1, s1 = math.cos(th1), math.sin(th1)
    c2, s2 = math.cos(th1 + b2), math.sin(th1 + b2)
    c3, s3 = math.cos(th1 + b2 + b3), math.sin(th1 + b2 + b3)
    cn, sn = math.cos(th1 + b2 + b3 + bn), math.sin(th1 + b2 + b3 + bn)
    d1 = np.array([c1 * cs, -s1, c1 * ss], np.float32)
    d2 = np.array([c2 * cs, -s2, c2 * ss], np.float32)
    d3 = np.array([c3 * cs, -s3, c3 * ss], np.float32)
    dn = np.array([cn * cs, -sn, cn * ss], np.float32)
    P = K + d1 * L1
    D = P + d2 * L2
    T = D + d3 * L3
    N = T + dn * Ln
    return P, D, T, N, d1, d2, d3, dn


def _solve_rest_pitch(K, L1, L2, L3, Ln, splay, b2, b3, bn, target_y=0.006):
    lo, hi = math.radians(-15.0), math.radians(58.0)
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        _, _, _, N, *_ = _finger_chain(K, L1, L2, L3, Ln, splay, mid, b2, b3, bn)
        if N[1] > target_y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
        self.lift = np.zeros(4)
        self.slide = np.zeros(4)
        self.glints = []         # (x, y, t0, strength)
        self.scratch = None
        # 3D camera basis (3/4 front-radial view of a right hand reaching from the left)
        yaw = math.radians(48.0)
        pitch = math.radians(23.0)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        self._cam_fwd = np.array([sy * cp, sp, cy * cp], np.float32)
        self._cam_right = np.array([cy, 0.0, -sy], np.float32)
        cam_up = np.cross(self._cam_fwd, self._cam_right)
        cam_up /= np.linalg.norm(cam_up)
        self._cam_up = cam_up.astype(np.float32)
        self._cam_down = -self._cam_up
        self._target = np.array([0.40, 0.16, 0.00], np.float32)
        L = (-0.20 * self._cam_right + 0.74 * self._cam_up + 0.64 * self._cam_fwd).astype(np.float32)
        self._light = L / np.linalg.norm(L)
        Hf = self._light + self._cam_fwd
        self._half = (Hf / np.linalg.norm(Hf)).astype(np.float32)
        # Pre-solve rest finger angles so every stiletto nail kisses the table plane Y=0
        self._th1_rest = np.zeros(4, np.float32)
        self._rest_tips3 = []
        for f in range(4):
            L_f = 0.465 * _FLEN[f]
            L1, L2, L3 = 0.47 * L_f, 0.31 * L_f, 0.22 * L_f
            th = _solve_rest_pitch(_MCP_BASE[f], L1, L2, L3, _NAIL_LEN[f],
                                   _SPLAYS[f], _PIP_BENDS[f], _DIP_BENDS[f], _NAIL_BENDS[f])
            self._th1_rest[f] = th
            _, _, _, N, *_ = _finger_chain(_MCP_BASE[f], L1, L2, L3, _NAIL_LEN[f],
                                           _SPLAYS[f], th, _PIP_BENDS[f], _DIP_BENDS[f], _NAIL_BENDS[f])
            self._rest_tips3.append(N)

    # -------------------------------------------------------------- geometry
    def _project(self, p3):
        rel = p3 - self._target
        sx = self._screen_origin[0] + float(np.dot(rel, self._cam_right)) * self._scale
        sy = self._screen_origin[1] + float(np.dot(rel, self._cam_down)) * self._scale
        sz = float(np.dot(rel, self._cam_fwd))
        return np.array([sx, sy, sz], np.float32)

    def resize(self, w, h):
        W, H = w, h * 2
        self.W, self.H = W, H
        u = min(H, W * 0.72)
        self.u = u
        self.table_back = int(H * 0.59)
        self._scale = 0.94 * u
        self._screen_origin = np.array([min(W * 0.51, W * 0.5 + u * 0.03), H * 0.55], np.float32)
        # Rest contact points on screen (index .. pinky; index nearest the viewer)
        self.contact = np.array([self._project(pt)[:2] for pt in self._rest_tips3], np.float32)
        # Background: dark room with drifting warm bokeh lights
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
        idle = 0.012 * (0.5 + 0.5 * np.sin(t * np.array([0.9, 1.1, 0.8, 1.3]) + np.array([0, 1, 2, 3])))
        self.lift = lift + idle
        self.slide = np.zeros(4)
        if self.scratch:
            st, f, dur = self.scratch
            d = t - st
            if 0 <= d <= dur:
                if prev < st:
                    cx, _ = self._tip_rest(f)
                    self.emit("scratch", pan=float(np.clip(cx / self.W * 2 - 1, -1, 1)), vel=0.8, dur=dur)
                self.slide[f] = math.sin(math.pi * d / dur) * 0.45
                self.lift[f] = 0.0
            elif d > dur:
                self.scratch = None

    def _tip_rest(self, f):
        return self.contact[f]

    # -------------------------------------------------------------- 3D rig & SDF
    def _build_rig(self):
        t = self.t
        breathe = 0.0045 * math.sin(t * 0.7)
        wy = 0.258 + breathe
        # Extend forearm past the left viewport edge, sloping gently toward elbow so it enters near-level
        need_x = (self._screen_origin[0] + 0.18 * self.W) / max(self._scale * float(self._cam_right[0]), 1e-3) + 0.5
        arm_x = -max(1.1, need_x)
        arm_slope = 0.20
        arm_y = wy + arm_slope * arm_x
        cuff_x = -0.25
        cuff_y = wy + arm_slope * cuff_x

        # Oval forearm tapering smoothly into a slender wrist (radius +Z, ulna -Z)
        wr_rad = np.array([0.0, wy, 0.048], np.float32)
        wr_uln = np.array([0.0, wy - 0.008, -0.048], np.float32)
        fa_rad = np.array([arm_x, arm_y, 0.058], np.float32)
        fa_uln = np.array([arm_x, arm_y - 0.008, -0.058], np.float32)

        mcps = [m + np.array([0.0, breathe, 0.0], np.float32) for m in _MCP_BASE]
        m_orig = (
            np.array([0.02, wy - 0.002, 0.068], np.float32),
            np.array([0.02, wy + 0.005, 0.022], np.float32),
            np.array([0.02, wy + 0.001, -0.024], np.float32),
            np.array([0.02, wy - 0.008, -0.068], np.float32),
        )

        # Solid arched palm core (keeps dorsum smooth and seamless)
        self._palm_center = np.array([0.245, 0.273 + breathe, -0.002], np.float32)
        pa = math.radians(7.5)
        self._palm_ax = (math.cos(pa), math.sin(pa))
        self._palm_ay = (-math.sin(pa), math.cos(pa))

        body_caps = [
            (fa_rad, wr_rad, 0.078, 0.053),
            (fa_uln, wr_uln, 0.076, 0.051),
            (m_orig[0], mcps[0], 0.054, 0.047),
            (m_orig[1], mcps[1], 0.056, 0.049),
            (m_orig[2], mcps[2], 0.054, 0.046),
            (m_orig[3], mcps[3], 0.050, 0.041),
            (mcps[0] + np.array([-0.015, -0.006, 0.0], np.float32),
             mcps[3] + np.array([-0.015, -0.006, 0.0], np.float32), 0.046, 0.040),
        ]

        # Right thumb: slender wrist waist -> fleshy thenar eminence along lower palm ->
        # thumb emerges past mid-palm and curves gently forward under the index arch
        t_cmc = np.array([0.145, wy - 0.016, 0.090], np.float32)
        t_mcp = np.array([0.330, 0.190 + breathe, 0.194], np.float32)
        t_ip = np.array([0.468, 0.132 + breathe * 0.5, 0.232], np.float32)
        t_tip = np.array([0.576, 0.102 + breathe * 0.25, 0.222], np.float32)
        t_pad_a = np.array([0.195, 0.230 + breathe, 0.112], np.float32)
        t_pad_b = np.array([0.360, 0.229 + breathe, 0.162], np.float32)

        thumb_caps = [
            (t_cmc, t_mcp, 0.056, 0.048),
            (t_pad_a, t_pad_b, 0.059, 0.046),
            (t_mcp, t_ip, 0.044, 0.037),
            (t_ip, t_tip, 0.036, 0.029),
        ]

        # 4 articulated 3-phalanx fingers + curved stiletto nails
        finger_caps = []
        nail_caps = []
        nail_nrms = []
        nail_seg_u = []

        for f in range(4):
            K = mcps[f]
            L_f = 0.465 * _FLEN[f]
            L1, L2, L3 = 0.47 * L_f, 0.31 * L_f, 0.22 * L_f
            Ln = _NAIL_LEN[f]
            sp = _SPLAYS[f]
            b2, b3, bn = _PIP_BENDS[f], _DIP_BENDS[f], _NAIL_BENDS[f]
            lf = float(self.lift[f])
            sl = float(self.slide[f])

            b2_eff = b2 + lf * 0.14 + sl * 0.38
            b3_eff = b3 + lf * 0.06 + sl * 0.22
            if sl > 1e-4:
                # Keep scratching tip planted smoothly on the table surface Y=0.006
                th1 = _solve_rest_pitch(K, L1, L2, L3, Ln, sp, b2_eff, b3_eff, bn)
            else:
                # Compensate gentle breathing so resting tips stay kissed to the lacquer
                th1 = float(self._th1_rest[f]) + breathe * 1.4 - lf * 1.0

            P, D, T, N, _, _, d3, dn = _finger_chain(K, L1, L2, L3, Ln, sp, th1, b2_eff, b3_eff, bn)
            r0 = _R_BASE[f]
            finger_caps.append((K, P, r0, r0 * 0.88))
            finger_caps.append((P, D, r0 * 0.88, r0 * 0.76))
            finger_caps.append((D, T, r0 * 0.76, r0 * 0.64))

            cs, ss = math.cos(sp), math.sin(sp)
            lat = np.array([-ss, 0.0, cs], np.float32)
            n_dors = np.cross(lat, d3)
            n_dors /= np.linalg.norm(n_dors)
            if n_dors[1] < 0:
                n_dors = -n_dors
            n_dors2 = np.cross(lat, dn)
            n_dors2 /= np.linalg.norm(n_dors2)
            if n_dors2[1] < 0:
                n_dors2 = -n_dors2

            n_base = D + d3 * (L3 * 0.32) + n_dors * (r0 * 0.42)
            n_mid = T + n_dors * (r0 * 0.32) + dn * (Ln * 0.14)
            n_tip = N + n_dors * (r0 * 0.04)
            rn = r0 * 0.64
            nail_caps.append((n_base, n_mid, rn * 0.72, rn * 0.92))
            nail_caps.append((n_mid, n_tip, rn * 0.92, 0.004))
            nail_nrms.append(n_dors)
            nail_nrms.append(n_dors2)
            nail_seg_u.append((0.0, 0.45))
            nail_seg_u.append((0.45, 1.0))

        # Thumb stiletto nail (slightly shorter almond-stiletto matching relaxed thumb)
        td = t_tip - t_ip
        td /= np.linalg.norm(td)
        tn_dors = np.array([0.10, 0.86, 0.50], np.float32)
        tn_dors -= td * float(np.dot(tn_dors, td))
        tn_dors /= np.linalg.norm(tn_dors)
        tn_dir = td * 0.98 - tn_dors * 0.06
        tn_dir /= np.linalg.norm(tn_dir)
        tn_base = t_ip + td * 0.042 + tn_dors * 0.019
        tn_mid = t_tip + tn_dors * 0.014 + tn_dir * 0.016
        tn_tip = t_tip + tn_dir * 0.095
        nail_caps.append((tn_base, tn_mid, 0.020, 0.024))
        nail_caps.append((tn_mid, tn_tip, 0.024, 0.005))
        nail_nrms.append(tn_dors)
        nail_nrms.append(tn_dors)
        nail_seg_u.append((0.0, 0.45))
        nail_seg_u.append((0.45, 1.0))

        # Velvet sleeve ending cleanly left of wrist
        sleeve_caps = [
            (np.array([arm_x - 0.3, arm_y - 0.06, 0.048], np.float32),
             np.array([cuff_x, cuff_y, 0.048], np.float32), 0.108, 0.094),
            (np.array([arm_x - 0.3, arm_y - 0.068, -0.048], np.float32),
             np.array([cuff_x, cuff_y - 0.008, -0.048], np.float32), 0.106, 0.092),
        ]

        # Gold beaded bracelet wrapping in 3D around the slender oval wrist
        br_x = -0.10
        self._br_c = np.array([br_x, wy + arm_slope * br_x - 0.005, 0.0], np.float32)

        all_body = body_caps + thumb_caps
        self._body_packed = _pack_capsules(all_body)
        self._finger_packed = _pack_capsules(finger_caps)
        self._nail_packed = _pack_capsules(nail_caps)
        nrm_arr = np.array(nail_nrms, np.float32)
        self._nail_nx = nrm_arr[:, 0:1]
        self._nail_ny = nrm_arr[:, 1:2]
        self._nail_nz = nrm_arr[:, 2:3]
        self._nail_u0 = np.array([s[0] for s in nail_seg_u], np.float32)
        self._nail_u1 = np.array([s[1] for s in nail_seg_u], np.float32)
        self._sleeve_packed = _pack_capsules(sleeve_caps)
        self._bounds = (
            [(a, b, ra, rb, 0.032) for a, b, ra, rb in all_body]
            + [(a, b, ra, rb, 0.018) for a, b, ra, rb in finger_caps]
            + [(a, b, ra, rb, 0.014) for a, b, ra, rb in nail_caps]
            + [(a, b, ra, rb, 0.022) for a, b, ra, rb in sleeve_caps]
        )

    def _eval_body(self, px, py, pz):
        db, _ = _eval_capsules(px, py, pz, self._body_packed)
        d_arm = _smin(db[0], db[1], 0.052)
        rx = px - self._palm_center[0]
        ry = py - self._palm_center[1]
        rz = pz - self._palm_center[2]
        plx = self._palm_ax[0] * rx + self._palm_ax[1] * ry
        ply = self._palm_ay[0] * rx + self._palm_ay[1] * ry
        taper = np.clip((plx + 0.18) * 2.78, 0.0, 1.0)
        half_z = 0.068 + 0.062 * taper
        ply_arch = ply + 0.018 * (rz * 6.5) ** 2
        qx = np.abs(plx) - 0.175
        qy = np.abs(ply_arch) - 0.020
        qz = np.abs(rz) - half_z
        d_slab = (np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2 + 1e-10)
                  + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0) - 0.039)
        d_m01 = _smin(db[2], db[3], 0.054)
        d_m23 = _smin(db[4], db[5], 0.054)
        d_meta = _smin(_smin(d_m01, d_m23, 0.058), d_slab, 0.048)
        d_palm = _smin(_smin(d_arm, d_meta, 0.062), db[6], 0.045)
        d_th_digits = _smin(db[9], db[10], 0.022)
        d_thenar = _smin(db[7], db[8], 0.048)
        d_thumb = _smin(d_thenar, d_th_digits, 0.036)
        return _smin(d_palm, d_thumb, 0.048)

    def _eval_sdf(self, px, py, pz, need_mat=False):
        M = px.shape[0]
        # Partition body (arm/palm/thumb, px < 0.72) and fingers (px > 0.28) along X
        m_body = px < 0.72
        if m_body.all():
            d_body = self._eval_body(px, py, pz)
        else:
            d_body = np.full(M, 1e6, np.float32)
            if m_body.any():
                d_body[m_body] = self._eval_body(px[m_body], py[m_body], pz[m_body])

        m_fin = px > 0.28
        if m_fin.any():
            df, _ = _eval_capsules(px[m_fin], py[m_fin], pz[m_fin], self._finger_packed)
            d_f0 = _smin(_smin(df[0], df[1], 0.024), df[2], 0.020)
            d_f1 = _smin(_smin(df[3], df[4], 0.024), df[5], 0.020)
            d_f2 = _smin(_smin(df[6], df[7], 0.024), df[8], 0.020)
            d_f3 = _smin(_smin(df[9], df[10], 0.022), df[11], 0.018)
            d_fin = np.minimum(np.minimum(d_f0, d_f1), np.minimum(d_f2, d_f3))
            d_skin = d_body.copy()
            d_skin[m_fin] = _smin(d_body[m_fin], d_fin, 0.030)
        else:
            d_skin = d_body
        d_min = d_skin

        # Partitioned secondary materials along X for speed
        m_nail = px > 0.38
        d_nail = None
        if m_nail.any():
            dn_all, hn_all = _eval_flat_capsules(
                px[m_nail], py[m_nail], pz[m_nail],
                self._nail_packed, self._nail_nx, self._nail_ny, self._nail_nz)
            dn = _smin(dn_all[0], dn_all[1], 0.008)
            for i in range(2, dn_all.shape[0], 2):
                dn = np.minimum(dn, _smin(dn_all[i], dn_all[i + 1], 0.008))
            d_nail = np.full(M, 1e6, np.float32)
            d_nail[m_nail] = dn
            d_min = np.minimum(d_min, d_nail)

        m_sl = px < -0.05
        d_sleeve = None
        if m_sl.any():
            dsl, _ = _eval_capsules(px[m_sl], py[m_sl], pz[m_sl], self._sleeve_packed)
            d_sleeve = np.full(M, 1e6, np.float32)
            d_sleeve[m_sl] = _smin(dsl[0], dsl[1], 0.052)
            d_min = np.minimum(d_min, d_sleeve)

        bc = self._br_c
        m_gd = (px > bc[0] - 0.16) & (px < bc[0] + 0.16)
        d_gold = None
        if m_gd.any():
            lx = px[m_gd] - bc[0]
            ly = py[m_gd] - bc[1]
            lz = pz[m_gd] - bc[2]
            lx_p = lx * 0.98 + ly * 0.20
            ly_p = -lx * 0.20 + ly * 0.98
            ry, rz = 0.064, 0.110
            ang = np.arctan2(ly_p, lz)
            lx_t = lx_p - 0.010 * np.cos(ang) + 0.005 * np.sin(ang * 2.0 + self.t * 0.5)
            r_ov = np.hypot(ly_p / ry, lz / rz)
            d_ring = (r_ov - 1.0) * 0.082
            bead_r = 0.0155 * (0.82 + 0.18 * np.cos(14.0 * ang))
            d_gold = np.full(M, 1e6, np.float32)
            d_gold[m_gd] = np.hypot(lx_t, d_ring) - bead_r
            d_min = np.minimum(d_min, d_gold)

        if not need_mat:
            return d_min

        mat = np.full(M, MAT_SKIN, np.int8)
        u_val = np.zeros(M, np.float32)
        if d_nail is not None:
            is_n = d_nail <= d_min + 1e-5
            mat[is_n] = MAT_NAIL
            if is_n.any():
                sub_is_n = is_n[m_nail]
                if sub_is_n.any():
                    dn_sub = dn_all[:, sub_is_n]
                    hn_sub = hn_all[:, sub_is_n]
                    bi = np.argmin(dn_sub, axis=0)
                    bh = hn_sub[bi, np.arange(bi.size)]
                    u_val[is_n] = self._nail_u0[bi] + (self._nail_u1[bi] - self._nail_u0[bi]) * bh
        if d_sleeve is not None:
            mat[d_sleeve <= d_min + 1e-5] = MAT_SLEEVE
        if d_gold is not None:
            mat[d_gold <= d_min + 1e-5] = MAT_GOLD
        return d_min, mat, u_val

    # -------------------------------------------------------------- render
    def draw_pixels(self, mode):
        H, W = self.H, self.W
        t = self.t
        img = self.image()
        img[:] = self.bg + self.table
        for m, ph, sp in self.bokeh:
            img += m * (0.75 + 0.25 * math.sin(t * sp + ph))

        self._build_rig()

        # Tight projected capsule envelopes -> per-pixel start/end ray depth
        z_front = np.full((H, W), -1e9, np.float32)
        z_back = np.full((H, W), 1e9, np.float32)
        scale = self._scale
        inv_s = 1.0 / max(scale, 1e-3)

        yy_grid, xx_grid = self.yy, self.xx

        def stamp_capsule(a3, b3, ra, rb, pad):
            pa = self._project(a3)
            pb = self._project(b3)
            rmax_px = (max(ra, rb) + pad) * scale + 1.5
            x0 = max(0, int(min(pa[0], pb[0]) - rmax_px))
            x1 = min(W, int(max(pa[0], pb[0]) + rmax_px) + 2)
            y0 = max(0, int(min(pa[1], pb[1]) - rmax_px))
            y1 = min(H, int(max(pa[1], pb[1]) + rmax_px) + 2)
            if x0 >= x1 or y0 >= y1:
                return
            px = xx_grid[y0:y1, x0:x1] + (0.5 - pa[0])
            py = yy_grid[y0:y1, x0:x1] + (0.5 - pa[1])
            bx = pb[0] - pa[0]
            by = pb[1] - pa[1]
            h = np.clip((px * bx + py * by) / (bx * bx + by * by + 1e-6), 0.0, 1.0)
            dx = px - bx * h
            dy = py - by * h
            rw = ra + (rb - ra) * h + pad
            r_px = rw * scale + 1.0
            d2_px = dx * dx + dy * dy
            hit = d2_px <= (r_px * r_px)
            if not hit.any():
                return
            zc = pa[2] + (pb[2] - pa[2]) * h
            hz = np.sqrt(np.maximum(rw * rw - d2_px * (inv_s * inv_s), 0.0)) + 0.015
            zf_sub = z_front[y0:y1, x0:x1]
            zb_sub = z_back[y0:y1, x0:x1]
            zf_sub[hit] = np.maximum(zf_sub[hit], (zc + hz)[hit])
            zb_sub[hit] = np.minimum(zb_sub[hit], (zc - hz)[hit])

        for a, b, ra, rb, pad in self._bounds:
            stamp_capsule(a, b, ra, rb, pad)
        stamp_capsule(self._br_c, self._br_c, 0.125, 0.125, 0.025)

        ys, xs = np.nonzero(z_front > -1e8)
        M = len(xs)
        if M > 0:
            rx = (xs.astype(np.float32) + 0.5 - self._screen_origin[0]) * inv_s
            ry = (ys.astype(np.float32) + 0.5 - self._screen_origin[1]) * inv_s
            zf = z_front[ys, xs]
            zb = z_back[ys, xs]

            R, D, F, T = self._cam_right, self._cam_down, self._cam_fwd, self._target
            rox = T[0] + R[0] * rx + D[0] * ry + F[0] * zf
            roy = T[1] + R[1] * rx + D[1] * ry + F[1] * zf
            roz = T[2] + R[2] * rx + D[2] * ry + F[2] * zf
            rdx, rdy, rdz = -float(F[0]), -float(F[1]), -float(F[2])
            t_max_all = np.maximum(zf - zb, 0.05)

            idx = np.arange(M, dtype=np.int32)
            t_cur = np.zeros(M, np.float32)
            hit_idx, hit_x, hit_y, hit_z = [], [], [], []
            eps = max(0.0014, 0.36 * inv_s)

            for _ in range(20):
                if idx.size == 0:
                    break
                cx = rox[idx] + rdx * t_cur
                cy = roy[idx] + rdy * t_cur
                cz = roz[idx] + rdz * t_cur
                d = self._eval_sdf(cx, cy, cz)
                hit = d < eps
                if hit.any():
                    hit_idx.append(idx[hit])
                    hit_x.append(cx[hit])
                    hit_y.append(cy[hit])
                    hit_z.append(cz[hit])
                t_cur = t_cur + np.maximum(d, eps * 0.5)
                alive = (~hit) & (t_cur < t_max_all[idx])
                idx = idx[alive]
                t_cur = t_cur[alive]

            if hit_idx:
                h_idx = np.concatenate(hit_idx)
                hx_w = np.concatenate(hit_x)
                hy_w = np.concatenate(hit_y)
                hz_w = np.concatenate(hit_z)
                d0, mat, u_val = self._eval_sdf(hx_w, hy_w, hz_w, need_mat=True)

                e = max(0.0025, 0.50 * inv_s)
                nx = self._eval_sdf(hx_w + e, hy_w, hz_w) - d0
                ny = self._eval_sdf(hx_w, hy_w + e, hz_w) - d0
                nz = self._eval_sdf(hx_w, hy_w, hz_w + e) - d0
                inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-12)
                nx *= inv_len
                ny *= inv_len
                nz *= inv_len

                ao = np.ones(hx_w.shape[0], np.float32)
                need_ao = (mat == MAT_SKIN) | (mat == MAT_SLEEVE)
                if need_ao.any():
                    ao1 = np.clip(
                        self._eval_sdf(hx_w[need_ao] + nx[need_ao] * 0.048,
                                       hy_w[need_ao] + ny[need_ao] * 0.048,
                                       hz_w[need_ao] + nz[need_ao] * 0.048) * (1.0 / 0.048),
                        0.0, 1.0)
                    ao[need_ao] = 0.46 + 0.54 * ao1
                col = self._shade(nx, ny, nz, mat, u_val, ao, t)

                hy = ys[h_idx]
                hx = xs[h_idx]

                # 3D reflection across the lacquered table plane Y=0
                refl = np.zeros_like(img)
                above = hy_w > -0.01
                if above.any():
                    wy_a = np.maximum(hy_w[above], 0.0)
                    dy_refl = -2.0 * wy_a * float(D[1]) * scale
                    ry_p = np.round(hy[above] + dy_refl).astype(np.int32)
                    rx_p = hx[above]
                    ok = (ry_p >= self.table_back) & (ry_p < H)
                    if ok.any():
                        ry_ok = ry_p[ok]
                        rx_ok = rx_p[ok]
                        fade = np.clip(1.0 - wy_a[ok] / 0.48, 0.0, 1.0) ** 1.35
                        rc = col[above][ok] * (fade[:, None] * 0.32)
                        refl[ry_ok, rx_ok] = np.maximum(refl[ry_ok, rx_ok], rc)
                refl = _blur(refl, 1)
                tabmask = (np.arange(H) >= self.table_back)[:, None, None]
                img += refl * tabmask
                img[hy, hx] = col

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

    def _shade(self, nx, ny, nz, mat, ub, ao, t):
        name = self.variant_name
        alb, spec, env, irid, french = NAILS[name]
        alb = np.array(hexrgb(alb), np.float32)
        env = np.array(hexrgb(env), np.float32)
        L, Hf, F = self._light, self._half, self._cam_fwd
        ndl = np.clip(L[0] * nx + L[1] * ny + L[2] * nz, 0.0, 1.0)
        ndh = np.clip(Hf[0] * nx + Hf[1] * ny + Hf[2] * nz, 0.0, 1.0)
        ndv = np.clip(F[0] * nx + F[1] * ny + F[2] * nz, 0.0, 1.0)
        rim = np.clip(1.0 - ndv, 0.0, 1.0)
        fill = np.clip(-ny * 0.35 + 0.5, 0.0, 1.0)

        out = np.zeros((nx.shape[0], 3), np.float32)

        # skin: diffuse + ambient occlusion + warm subsurface rim + subtle sheen
        sk = mat == MAT_SKIN
        if sk.any():
            a = ao[sk, None]
            c = (SKIN * (0.25 * a + 0.77 * ndl[sk, None] * (0.38 + 0.62 * a) + 0.06 * fill[sk, None])
                 + SKIN_SSS * (rim[sk, None] ** 1.6) * a
                 + 0.07 * (ndh[sk, None] ** 28) * a)
            out[sk] = c

        sl = mat == MAT_SLEEVE
        if sl.any():
            a = ao[sl, None]
            c = (SLEEVE * (0.32 * a + 0.62 * ndl[sl, None])
                 + np.array([0.26, 0.12, 0.32], np.float32) * (rim[sl, None] ** 2.8) * 0.65)
            out[sl] = c

        gd = mat == MAT_GOLD
        if gd.any():
            c = GOLD * (0.30 + 0.72 * ndl[gd, None]) + 0.95 * (ndh[gd, None] ** 36)
            out[gd] = c

        nl = mat == MAT_NAIL
        if nl.any():
            nnx, nny, nnz = nx[nl], ny[nl], nz[nl]
            R, U = self._cam_right, self._cam_up
            nx_cam = R[0] * nnx + R[1] * nny + R[2] * nnz
            ny_cam = U[0] * nnx + U[1] * nny + U[2] * nnz
            base = np.broadcast_to(alb, (nnx.shape[0], 3)).copy()
            if irid:
                hue = (nx_cam * 0.6 - ny_cam * 0.4 + t * 0.03) % 1.0
                rainbow = 0.5 + 0.5 * np.cos(6.283 * (hue[:, None] + np.array([0.0, 0.33, 0.67], np.float32)))
                base = base * (1 - irid) + rainbow * irid * base.max()
            if french:
                tip = ub[nl] > 0.52
                base[tip] = np.array([0.98, 0.97, 0.95], np.float32)
            # glossy: diffuse + sharp specular highlight + soft "softbox" reflection band
            band = np.exp(-((ny_cam - 0.35) / 0.24) ** 2) * np.clip(ndv[nl], 0, 1)
            c = (base * (0.28 + 0.68 * ndl[nl, None])
                 + spec * 1.15 * (ndh[nl, None] ** 45) * env
                 + spec * 0.42 * band[:, None] * env
                 + 0.14 * (rim[nl, None] ** 2) * env)
            out[nl] = c
        return np.clip(out, 0.0, 1.6)
