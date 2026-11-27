"""DIVINE MERCY - the image of the merciful Jesus, in its carved gilded frame.

The picture follows the classic paintings of St. Faustina Kowalska's vision
(Kazimirowski 1934, Hyla 1943): the risen Jesus, in a long white robe, steps
toward the viewer out of a dark ground.  His right hand is raised in blessing
(on the viewer's left); his left hand rests at his breast, drawing the garment
slightly aside, and from his heart two broad translucent rays fall to the foot
of the picture: the red ray on the viewer's left, the pale ray on the viewer's
right.  A plain dark band is left at the bottom for the inscription
"JESUS, I TRUST IN YOU", which the scene writes as terminal text.

The figure is drawn in its own units (1 = its height) with 2D signed distance
fields, so one description serves every size; small sprites are rendered
supersampled and box-filtered.  The canvas is flat, so the light and shade of
the drapery (a fold height field over the rounded body), of the hands and of
the face (a softly sculpted relief) are *painted* into the albedo with the
picture's own gentle light from the upper left; the scene's candles then light
the painting as a whole.  Only the frame has relief: a profiled moulding
(beads, a hollow, a twisted gadroon) with corner rosettes, given as normals so
the candle flames glint along it.  The rays are glazes plus a little emission,
and are also returned as masks for the scene's shimmer and for the coloured
light they spill.
"""

import math

import numpy as np

from . import kit

F32 = np.float32

# -- palette (albedo; the painting's own light and shade is baked in) ---------
ROBE_HI = kit.rgb((0.76, 0.76, 0.75))       # white robe in the light (warm under candles)
ROBE_LO = kit.rgb((0.27, 0.29, 0.36))       # cool grey in the folds
SKIN_HI = kit.rgb((0.66, 0.47, 0.36))
SKIN_LO = kit.rgb((0.27, 0.16, 0.115))
HAIR_HI = kit.rgb((0.27, 0.165, 0.085))     # brown
HAIR_LO = kit.rgb((0.050, 0.030, 0.019))
WOUND = kit.rgb((0.42, 0.07, 0.05))
WALL_TOP = kit.rgb((0.008, 0.012, 0.034))   # deep blue-black
WALL_LOW = kit.rgb((0.018, 0.018, 0.030))
FLOOR_FAR = kit.rgb((0.036, 0.028, 0.022))  # umber floor
FLOOR_NEAR = kit.rgb((0.052, 0.040, 0.029))
BAND = kit.rgb((0.008, 0.007, 0.008))
GOLD = kit.rgb((0.98, 0.74, 0.31))
BOLE = kit.rgb((0.40, 0.17, 0.08))           # red clay under the gilding, seen in the hollows
RED_TINT = kit.rgb((1.0, 0.12, 0.10))        # how the red glaze filters the robe beneath
RED_GLAZE = kit.rgb((0.62, 0.05, 0.04))
RED_EMIT = kit.rgb((0.50, 0.035, 0.025))
PALE_TINT = kit.rgb((0.84, 0.93, 1.0))
PALE_GLAZE = kit.rgb((0.55, 0.62, 0.74))
PALE_EMIT = kit.rgb((0.30, 0.36, 0.46))
PAINT_LIGHT = kit.normalize((-0.40, 0.46, 0.79))   # the picture's own light: soft, upper left

# -- the figure, in figure units: u to the viewer's right, v DOWN from the top of the head
FACE = (0.0, 0.074)
FACE_R = (0.0305, 0.046)
HEART = (0.012, 0.290)
HAND_R = (-0.188, 0.180)        # raised right hand (palm centre), viewer's left
R_SHOULDER, R_ELBOW, R_WRIST = (-0.090, 0.196), (-0.148, 0.318), (-0.184, 0.226)
L_SHOULDER, L_ELBOW, L_WRIST = (0.092, 0.196), (0.140, 0.322), (0.082, 0.290)
BODY = [(-0.026, 0.134), (-0.066, 0.146), (-0.094, 0.166), (-0.108, 0.196), (-0.108, 0.300),
        (-0.106, 0.420), (-0.120, 0.600), (-0.138, 0.800), (-0.150, 0.946), (-0.090, 0.954),
        (-0.020, 0.958), (0.050, 0.966), (0.110, 0.960), (0.156, 0.948), (0.144, 0.800),
        (0.124, 0.600), (0.108, 0.420), (0.110, 0.300), (0.110, 0.196), (0.096, 0.166),
        (0.068, 0.146), (0.028, 0.134), (0.0, 0.131)]
R_DRAPE = [(-0.214, 0.236), (-0.218, 0.300), (-0.212, 0.352), (-0.198, 0.394), (-0.186, 0.404),
           (-0.174, 0.390), (-0.156, 0.346), (-0.150, 0.300)]
L_DRAPE = [(0.082, 0.304), (0.160, 0.318), (0.166, 0.360), (0.158, 0.398), (0.142, 0.418),
           (0.124, 0.410), (0.104, 0.384), (0.092, 0.350)]
FEET = [(-0.050, 0.966, 0.024, 0.0145, -0.25), (0.058, 0.981, 0.026, 0.0165, 0.18)]   # the left foot steps forward
TOES = ((0.62, 0.0058), (0.30, 0.0047), (0.03, 0.0043), (-0.24, 0.0040), (-0.50, 0.0036))   # big toe first
SKIRT_FOLDS = ((1.15, 0.022, 0.22), (2.3, 0.014, 0.30), (3.6, 0.009, 0.42), (5.8, 0.005, 0.56))  # freq, depth, start

# -- the rays (angles from straight down, radians)
RAY_AXIS = 0.39         # each ray's axis leans this far out from the vertical
RAY_HALF = 0.23         # half the angle each ray subtends
RAY_W0 = 0.011          # half width where it issues from the breast (figure units)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# At thumbnail sizes the raised hand and the head are drawn a little larger (about the
# wrist and the neck), so that the blessing and the face still read.
HAND_PIVOT = (R_WRIST[0], R_WRIST[1] - 0.004)
HEAD_PIVOT = (0.0, 0.15)


def _hand_k(Fpx):
    return 1.0 + 0.5 * (1.0 - float(_ss(45.0, 100.0, Fpx)))


def _head_k(Fpx):
    return 1.0 + 0.18 * (1.0 - float(_ss(40.0, 90.0, Fpx)))


def _enlarged(pt, pivot, k):
    return pivot[0] + (pt[0] - pivot[0]) * k, pivot[1] + (pt[1] - pivot[1]) * k


class _Layout:
    """Where everything goes, in final sprite pixels (x right, y down)."""

    def __init__(self, h):
        self.h = h
        m = 1 if h >= 12 else 0                          # ~1 px of margin around the frame
        self.W = W = max(2 * m + 2, int(round(0.59 * h)))
        self.outer = (m, m, W - m, h - m)
        self.fw = fw = max(1, int(round(0.085 * (W - 2 * m))))
        self.canvas = cx0, cy0, cx1, cy1 = (m + fw, m + fw, W - m - fw, h - m - fw)
        bh = 0 if h < 14 else int(np.clip(round(0.055 * h), 2 if h < 40 else 3, 10))
        self.band = (cx0, cy1 - bh, cx1, cy1)
        pic = cy1 - bh - cy0
        self.F = 0.885 * pic                             # figure height
        self.top = cy0 + 0.06 * pic                      # top of the head
        self.axis = 0.5 * (cx0 + cx1) + 0.012 * self.F   # the figure's centre line
        self.ss = int(np.clip(math.ceil(150.0 / h), 1, 8))   # supersampling factor

    def to_px(self, u, v):
        return self.axis + u * self.F, self.top + v * self.F


# =============================================================================
# the figure (painted)
# =============================================================================
class _Painter:
    """Paints layers back to front on figure-unit grids (U right, V down).

    px: one sample in figure units; Fpx: figure height in final pixels, which
    decides how much painted detail can survive at this size."""

    def __init__(self, U, V, px, Fpx):
        self.U, self.V, self.px, self.Fpx = U, V, px, Fpx
        self.col = np.zeros(U.shape + (3,), F32)
        self.cov = np.zeros(U.shape, F32)

    def scaled(self, ou, ov, k):
        """A view of this painter (same canvas) whose drawing is enlarged k times about (ou, ov)."""
        q = _Painter.__new__(_Painter)
        q.U, q.V = ou + (self.U - ou) / k, ov + (self.V - ov) / k
        q.px, q.Fpx = self.px / k, self.Fpx * k
        q.col, q.cov = self.col, self.cov
        return q

    def cv(self, d):
        return np.clip(0.5 - d / self.px, 0.0, 1.0).astype(F32)

    def put(self, c, rgb):
        c3 = c[..., None]
        self.col[:] = self.col * (1.0 - c3) + np.asarray(rgb, F32) * c3
        self.cov[:] = self.cov + c * (1.0 - self.cov)

    def rim(self, d, dist, k):
        """Soft occlusion on what is already painted, just outside a layer about to go on top."""
        o = k * np.clip(1.0 - np.maximum(d, 0.0) / dist, 0.0, 1.0) ** 2
        self.col[:] = self.col * (1.0 - o)[..., None]

    def lam(self, hgt):
        """Painted lambert term of a height field given in figure units."""
        n = kit.normal_from_height(hgt / self.px)
        return n @ PAINT_LIGHT

    def detail(self, size, lo=2.0, hi=4.5):
        """0..1: how much of a painted detail of `size` (figure units) shows at this scale."""
        return float(_ss(lo, hi, size * self.Fpx))


def _tone(lo, hi, l, a=0.10, b=0.72, g=0.9):
    k = np.clip((l - a) / b, 0.0, 1.0) ** g
    return lo + (hi - lo) * k[..., None]


def _robe(P, ph):
    """The bare feet, then the long robe (torso and skirt) falling over them."""
    U, V = P.U, P.V
    # feet, under the hem
    gt = P.detail(0.004, 1.6, 3.0)                        # toes, where there is room
    for fu, fv, rx, ry, ang in FEET:
        c, s = math.cos(ang), math.sin(ang)
        a, b = (U - fu) * c + (V - fv) * s, -(U - fu) * s + (V - fv) * c
        d = kit.sd_ellipse(a, b, 0.0, -0.002, rx * 0.92, ry * 0.85)
        inward = -math.copysign(1.0, fu)                  # the big toe is on the inner side
        toes = []
        for k, (ox, r) in enumerate(TOES):
            toes.append(kit.sd_circle(a, b, inward * ox * rx, ry * 0.42 - 0.0012 * k, r))
        d_t = toes[0]
        for t in toes[1:]:
            d_t = np.minimum(d_t, t)
        d = kit.smin(d, d_t, 0.003) if gt > 0 else d
        sep = np.zeros_like(U)
        if gt > 0:
            for i in range(4):
                sep = np.maximum(sep, np.exp(-(np.abs(toes[i] - toes[i + 1]) / 0.0009) ** 2))
            sep *= _ss(0.0, -0.002, d_t) * gt
        foot = _tone(SKIN_LO, SKIN_HI, P.lam(kit.pillow(d, ry) * 0.8), a=0.2) * (1.0 - 0.4 * sep)[..., None]
        P.put(P.cv(d), foot)
        _wound(P, fu - inward * 0.004, fv + 0.001, 0.0030)
    # torso and skirt: the hem follows the folds a little
    hw = np.interp(V, [0.13, 0.20, 0.42, 0.96], [0.06, 0.110, 0.110, 0.153]).astype(F32)
    phi = (U - 0.004) / hw + 0.05 * np.sin(V * 7.0 + ph[3]) * _ss(0.3, 0.9, V)
    folds = np.zeros_like(U)
    for i, (f, a, v0) in enumerate(SKIRT_FOLDS):
        # |cos| makes rounded ridges with sharp valleys, hw / f apart; finer folds start lower
        ridge = np.abs(np.cos(math.pi * f * phi + ph[i]))
        folds += a * P.detail(0.13 / f, 1.6, 3.2) * ridge * _ss(v0, v0 + 0.25, V)
    # the forward (left) leg: the robe falls over the knee and the shin
    folds += 0.030 * np.exp(-((U - 0.050) / 0.045) ** 2 - ((V - 0.650) / 0.060) ** 2)
    folds += 0.016 * np.exp(-((U - 0.056) / 0.032) ** 2) * _ss(0.62, 0.74, V)
    # where the left hand draws the garment aside, folds fan out from it
    ru, rv = U - 0.040, V - 0.272
    r = np.hypot(ru, rv)
    fan = np.abs(np.cos(3.0 * np.arctan2(rv, ru) + ph[5])) * np.exp(-r / 0.07) * _ss(0.01, 0.035, r)
    folds += 0.014 * P.detail(0.04) * fan
    # gentle folds from the shoulders down over the chest
    chest = np.abs(np.cos((U + 0.01) / 0.05 * math.pi + ph[4])) * _ss(0.15, 0.25, V) * _ss(0.45, 0.3, V)
    folds += 0.006 * P.detail(0.05) * chest
    d_body = kit.sd_poly(U, V, BODY) - 0.004
    d_body = d_body + 0.004 * P.detail(0.03) * np.cos(math.pi * 3.6 * phi + ph[2]) * _ss(0.90, 0.95, V)
    P.rim(d_body, 0.010, 0.55)                            # the hem shades the feet
    P.put(P.cv(d_body), _tone(ROBE_LO, ROBE_HI, P.lam(kit.pillow(d_body, 0.10) + folds)))
    return d_body


def _sleeve(P, sh, el, wr, drape, ph):
    """A wide sleeve: upper arm, forearm and the fabric hanging below the forearm."""
    U, V = P.U, P.V
    d_up = kit.sd_capsule(U, V, *sh, *el, 0.033, 0.038)
    d_fo = kit.sd_capsule(U, V, *el, *wr, 0.038, 0.044)
    d_dr = kit.sd_poly(U, V, drape) - 0.008
    d = kit.smin(kit.smin(d_up, d_fo, 0.02), d_dr, 0.03)
    P.rim(d, 0.020, 0.55)
    # folds run along the forearm and hang down the drape
    ax, ay = wr[0] - el[0], wr[1] - el[1]
    ln = math.hypot(ax, ay)
    across = ((U - el[0]) * ay - (V - el[1]) * ax) / ln
    f1 = np.abs(np.cos(across / 0.045 * math.pi * 1.3 + ph[0]))
    f2 = np.abs(np.cos((U - drape[0][0]) / 0.032 * math.pi * 0.95 + ph[1]))
    w_dr = _ss(0.0, -0.03, d_dr) * _ss(-0.01, 0.01, d_fo)
    fh = P.detail(0.05) * (0.012 * f1 * (1 - w_dr) + 0.020 * f2 * w_dr)
    # the upper arm is a separate tube from the forearm: a crease at the elbow
    crease = np.exp(-(np.maximum(d_up, d_fo) / 0.008) ** 2) * 0.012 * P.detail(0.03)
    tube = kit.pillow(d, 0.045) * (1 - w_dr) + kit.pillow(d, 0.022) * w_dr
    P.put(P.cv(d), _tone(ROBE_LO, ROBE_HI, P.lam(tube + fh - crease)))
    return d


def _skin(P, d, round_w, extra=None):
    hgt = kit.pillow(d, round_w)
    if extra is not None:
        hgt = hgt + extra
    return _tone(SKIN_LO, SKIN_HI, P.lam(hgt), a=0.15, b=0.72)


def _wound(P, u, v, r):
    """A small soft mark of the nail wounds (only where there is room to show it)."""
    w = np.exp(-((P.U - u) ** 2 + (P.V - v) ** 2) / (r * r)) * P.detail(2.5 * r, 1.1, 1.8)
    P.col[:] = P.col * (1.0 - 0.55 * w)[..., None] + WOUND * (0.55 * w)[..., None]


def _arms(P, ph):
    U, V = P.U, P.V
    # left arm (viewer's right): the hand rests at the breast
    _sleeve(P, L_SHOULDER, L_ELBOW, L_WRIST, L_DRAPE, ph[0:2])
    d_back = kit.sd_capsule(U, V, 0.078, 0.287, 0.050, 0.274, 0.0140, 0.0148)
    d_fing = kit.sd_capsule(U, V, 0.050, 0.274, 0.022, 0.265, 0.0120, 0.0082)
    d_thumb = kit.sd_capsule(U, V, 0.062, 0.273, 0.046, 0.259, 0.0060, 0.0048)
    d_lh = kit.smin(kit.smin(d_back, d_fing, 0.008), d_thumb, 0.004)
    P.rim(d_lh, 0.012, 0.45)
    # the garment drawn a little aside under the fingertips, where the rays issue
    d_open, _ = kit.sd_segment(U, V, 0.022, 0.272, 0.012, 0.302)
    P.put(P.cv(d_open - 0.0025) * P.detail(0.02) * 0.55, ROBE_LO * 0.8)
    knuck = 0.004 * P.detail(0.015) * np.abs(np.cos((U - 0.05) / 0.0085 * math.pi)) * _ss(0.0, -0.01, d_fing)
    P.put(P.cv(d_lh), _skin(P, d_lh, 0.012, knuck))
    _wound(P, 0.062, 0.280, 0.0032)

    # right arm (viewer's left): raised in blessing, palm toward us, fingers up
    _sleeve(P, R_SHOULDER, R_ELBOW, R_WRIST, R_DRAPE, ph[2:4])
    wu, wv = HAND_PIVOT
    H = P.scaled(wu, wv, _hand_k(P.Fpx))
    U, V = H.U, H.V
    pu, pv = HAND_R
    tilt = 0.10                                          # the fingers lean a little outward
    palm = kit.sd_box(U + (V - pv) * tilt, V, pu, pv, 0.0165, 0.0195, 0.0095)
    fingers = []
    for ox, ln, rr in ((-0.0118, 0.027, 0.0045), (-0.0040, 0.034, 0.0050),
                       (0.0040, 0.039, 0.0052), (0.0118, 0.036, 0.0050)):
        x0, y0 = pu + ox, pv - 0.012
        fingers.append(kit.sd_capsule(U, V, x0, y0, x0 - tilt * ln, y0 - ln, rr, rr * 0.82))
    d_f = fingers[0]
    for f in fingers[1:]:
        d_f = np.minimum(d_f, f)
    thumb = kit.sd_capsule(U, V, pu + 0.013, pv + 0.007, pu + 0.028, pv - 0.010, 0.0062, 0.0050)
    wrist = kit.sd_capsule(U, V, pu + 0.002, pv + 0.016, wu, wv, 0.0115, 0.012)
    d_rh = kit.smin(kit.smin(kit.smin(palm, d_f, 0.003), thumb, 0.004), wrist, 0.004)
    # the fingers are separate: a fine crease between them where there is room
    sep = np.zeros_like(U)
    gs = H.detail(0.008)
    if gs > 0:
        for i in range(3):
            sep = np.maximum(sep, np.exp(-(np.abs(fingers[i] - fingers[i + 1]) / 0.0013) ** 2))
        sep *= _ss(0.0, -0.004, d_f) * _ss(pv - 0.011, pv - 0.017, V) * gs
    hand = _skin(H, d_rh, 0.011, -0.003 * sep)
    H.put(H.cv(d_rh), hand * (1.0 - 0.35 * sep)[..., None])
    _wound(H, pu - 0.001, pv + 0.002, 0.0034)
    return d_rh


def _head(P, ph):
    P = P.scaled(*HEAD_PIVOT, _head_k(P.Fpx))
    U, V = P.U, P.V
    fu, fv = FACE
    rx, ry = FACE_R
    a, b = (U - fu) / rx, (V - fv) / ry                  # face coordinates: the oval is a^2 + b^2 < 1
    aa = np.abs(a)
    d_face = kit.sd_ellipse(U, V, fu, fv, rx, ry)
    fwp = 2 * rx * P.Fpx                                 # face width in final pixels
    g1 = float(_ss(3.0, 6.0, fwp))                       # broad modelling
    g2 = float(_ss(5.0, 9.0, fwp))                       # eyes, brows, nose
    g3 = float(_ss(8.0, 13.0, fwp))                      # mouth, lips, finer things

    # hair: parted in the middle, falling in soft waves to the shoulders
    wave = 0.002 * np.sin(V * 90.0 + ph[0]) * P.detail(0.02)
    crown = kit.sd_ellipse(U, V, 0.0, 0.055, 0.0405, 0.057)
    curt = []
    for sgn in (-1.0, 1.0):
        pts = [(sgn * x, y) for x, y in ((0.022, 0.044), (0.040, 0.052), (0.047, 0.085), (0.052, 0.120),
                                         (0.058, 0.160), (0.059, 0.184), (0.050, 0.195), (0.039, 0.190),
                                         (0.034, 0.160), (0.029, 0.130), (0.025, 0.100))]
        curt.append(kit.sd_poly(U, V, pts) - 0.003 + wave)
    d_hair = kit.smin(crown, np.minimum(curt[0], curt[1]), 0.016)
    P.rim(d_hair, 0.008, 0.25)
    # strands wrap round the face from the parting; locks are divided by darker partings
    flow = d_face + 0.3 * np.abs(U) * _ss(0.05, 0.0, V)
    wob = 1.5 * np.sin(V * 38.0 + ph[1])
    fine = np.cos(flow / 0.0085 * 2 * math.pi + wob) * P.detail(0.0085, 2.2, 4.0)
    lk = flow / 0.020 * 2 * math.pi + 0.6 * wob + ph[2] + 0.8 * np.sin(flow / 0.031 + V * 23.0)
    locks = (np.cos(lk) * 0.7 + 0.3 * np.cos(2.3 * lk + 1.0)) * P.detail(0.02, 2.5, 5.0)
    hl = P.lam(kit.pillow(d_hair, 0.03) + 0.0025 * locks + 0.0012 * fine)
    inner = _ss(0.010, 0.0, d_face) * _ss(0.03, 0.08, V)          # shadow next to the face
    part = np.exp(-(U / 0.0016) ** 2) * _ss(0.036, 0.026, V) * g2  # the parting
    shade = 0.92 + 0.12 * fine + 0.12 * locks - 0.5 * inner + 0.6 * part
    P.put(P.cv(d_hair), _tone(HAIR_LO, HAIR_HI, hl, a=0.2, b=0.68) * shade[..., None])

    # the face: a gently sculpted relief, lit by the picture's light
    hgt = rx * 0.9 * np.sqrt(np.clip(1.0 - a * a - b * b, 0.0, 1.0))
    nose = np.exp(-(a / 0.13) ** 2) * _ss(-0.28, -0.08, b) * _ss(0.40, 0.30, b)
    tip = np.exp(-(a / 0.19) ** 2 - ((b - 0.30) / 0.09) ** 2)
    sockets = np.exp(-((aa - 0.42) / 0.24) ** 2 - ((b + 0.09) / 0.15) ** 2)
    brow = np.exp(-((b + 0.27) / 0.10) ** 2) * _ss(0.85, 0.5, aa)
    cheek = np.exp(-((aa - 0.48) / 0.25) ** 2 - ((b - 0.18) / 0.18) ** 2)
    hgt = hgt + rx * g1 * (0.32 * nose + 0.18 * tip + 0.10 * brow - 0.22 * sockets + 0.06 * cheek)
    face = _tone(SKIN_LO, SKIN_HI, P.lam(hgt), a=0.12, b=0.74)
    face = face * (1.0 + 0.10 * g1 * cheek[..., None] * np.array([1.0, -0.1, -0.2], F32))   # warmth in the cheeks
    face = face * (1.0 - 0.30 * _ss(0.6, 1.0, aa) * g1)[..., None]          # the hair shades the temples
    under_nose = np.exp(-(a / 0.16) ** 2 - ((b - 0.39) / 0.045) ** 2)
    face = face * (1.0 - 0.35 * g2 * under_nose)[..., None]
    # eyes: lids lowered, looking gently down toward us
    ea, eb = (aa - 0.42) / 0.21, b + 0.08
    lid = np.exp(-((eb + 0.02 * (1 - ea ** 2)) / 0.045) ** 2) * _ss(1.15, 0.7, np.abs(ea))
    iris = np.exp(-(ea / 0.45) ** 2 - ((eb - 0.035) / 0.05) ** 2)
    eye = np.clip(lid * 0.8 + iris * 0.5 * g3, 0.0, 1.0) * g2
    face = face * (1.0 - 0.62 * eye)[..., None] + kit.rgb((0.10, 0.06, 0.045)) * (0.25 * eye)[..., None]
    bb = b + 0.27 + 0.07 * (1.0 - ((aa - 0.45) / 0.36) ** 2)
    brow_m = np.exp(-(bb / 0.05) ** 2) * _ss(0.85, 0.65, aa) * _ss(0.08, 0.18, aa) * g2
    face = face * (1.0 - 0.45 * brow_m)[..., None]
    hairline = 0.034 + 0.034 * (a ** 2) - V              # > 0 above the hairline
    P.put(P.cv(np.maximum(d_face, hairline)), face)

    # a short soft beard round the jaw and chin, and the moustache
    b_line = np.maximum(0.48 - 0.98 * (aa - 0.38), 0.66 - 0.474 * aa)     # upper edge of the beard
    d_jaw = np.maximum(d_face, (b_line - b) * ry)
    d_chin = kit.sd_ellipse(U, V, fu, fv + 0.92 * ry, 0.66 * rx, 0.40 * ry)
    d_must = np.maximum((np.abs(b - 0.455 - 0.32 * a * a) - 0.065) * ry, (aa - 0.56) * rx)
    d_beard = np.minimum(kit.smin(d_jaw, d_chin, 0.004), d_must)
    curl = np.cos(U / 0.0045 * math.pi + 2.0 * np.sin(V / 0.006)) * P.detail(0.009)
    bl = P.lam(kit.pillow(d_beard, 0.008) + 0.0010 * curl)
    beard = _tone(HAIR_LO * 1.6, HAIR_HI, bl, a=0.1, b=0.8) * 1.12
    mouth = np.exp(-(a / 0.22) ** 2 - ((b - 0.545) / 0.03) ** 2) * g3
    lip = np.exp(-(a / 0.19) ** 2 - ((b - 0.60) / 0.045) ** 2) * g3
    beard = beard * (1.0 - 0.5 * mouth)[..., None]
    beard = beard * (1 - lip[..., None]) + (SKIN_HI * kit.rgb((0.92, 0.72, 0.70))) * lip[..., None]
    P.put(np.clip(0.5 - d_beard / max(P.px, 0.004), 0.0, 1.0), beard)    # a beard has soft edges


def _figure(U, V, px, Fpx, rng):
    """Paint the whole figure; returns (coverage, albedo)."""
    P = _Painter(U, V, px, Fpx)
    ph = rng.uniform(0, 2 * math.pi, 16).astype(F32)
    _robe(P, ph[0:6])
    _arms(P, ph[6:12])
    _head(P, ph[12:16])
    return P.cov, P.col


# =============================================================================
# the rays
# =============================================================================
def _rays(X, Y, lay, rng):
    """Masks (0..1) of the red and pale rays on final-pixel grids X, Y."""
    F = lay.F
    hx, hy = lay.to_px(*HEART)
    by0 = lay.band[1]
    dx, dy = X - hx, Y - hy
    ang = np.arctan2(dx, dy)
    out = []
    for sgn in (-1.0, 1.0):
        th = sgn * RAY_AXIS
        ax, ay = math.sin(th), math.cos(th)
        s = dx * ax + dy * ay                         # along the ray
        q = dx * ay - dy * ax                         # across it
        sp = np.maximum(s, 0.0)
        half = RAY_W0 * F + sp * math.tan(RAY_HALF)
        soft = 0.5 + 0.05 * sp
        m = np.clip(0.5 - (np.abs(q) - half) / soft, 0.0, 1.0)
        m *= 1.0 - 0.3 * np.clip(np.abs(q) / np.maximum(half, 1e-3), 0.0, 1.0) ** 2   # fuller in the middle
        m *= _ss(-0.01 * F, 0.04 * F, s)              # issues from under the garment
        m *= 0.40 + 0.60 * np.exp(-sp / (0.30 * F))   # strongest at the heart
        p = rng.uniform(0, 2 * math.pi, 3)
        st = 0.16 * np.sin(ang * 47.0 + p[0]) + 0.10 * np.sin(ang * 83.0 + p[1]) + 0.06 * np.sin(ang * 131.0 + p[2])
        m *= 1.0 + st * float(_ss(60.0, 130.0, F))    # faint streaks of light, where there is room
        m *= _ss(by0, by0 - max(1.5, 0.035 * F), Y)   # fades out above the inscription band
        out.append(np.clip(m, 0.0, 1.0).astype(F32))
    return out


# =============================================================================
# the frame
# =============================================================================
def _bump(t, c, r, a):
    return a * np.sqrt(np.clip(1.0 - ((t - c) / r) ** 2, 0.0, 1.0))


def _frame(X, Y, lay, ss):
    """G-buffer layers of the carved gilded frame: (coverage, albedo, normal, spec, gloss, ao)."""
    ox0, oy0, ox1, oy1 = lay.outer
    fw = float(lay.fw)
    dl, dr, dt, db = X - ox0, ox1 - X, Y - oy0, oy1 - Y
    dmin = np.minimum(np.minimum(dl, dr), np.minimum(dt, db))
    t = dmin / fw                                     # 0 outer edge .. 1 sight edge
    cov = ((dmin > 0) & (t < 1.0)).astype(F32)
    orn = float(_ss(2.5, 5.0, fw))                    # carving shows only on a wide enough moulding
    # profile: outer bead, a hollow, the big twisted torus, a bevel down to the sight-edge bead
    base = np.interp(t, [0.0, 0.14, 0.40, 0.70, 0.90, 1.0], [0.30, 0.26, 0.40, 0.36, 0.14, 0.10])
    hollow = 0.12 * np.sin(np.pi * np.clip((t - 0.14) / 0.26, 0.0, 1.0))
    torus = _bump(t, 0.55, 0.16, 0.28)
    outer_bead = _bump(t, 0.07, 0.07, 0.12)
    inner_bead = _bump(t, 0.94, 0.06, 0.08)
    p = base - hollow + torus + outer_bead + inner_bead
    # twisted gadroon along the torus, continuous around the frame
    Wf, Hf = ox1 - ox0, oy1 - oy0
    s = np.where(dmin == dt, X - ox0,
                 np.where(dmin == dr, Wf + Y - oy0,
                          np.where(dmin == db, Wf + Hf + ox1 - X, 2 * Wf + Hf + oy1 - Y)))
    perim = 2 * (Wf + Hf)
    n_tw = max(4, int(round(perim / max(2.0, 0.85 * fw))))
    tw = np.cos(2 * math.pi * (s * n_tw / perim + 1.3 * (t - 0.55) / 0.32))
    p = p + 0.09 * tw * (torus / 0.28) * orn
    # rosettes over the mitred corners
    rose = np.zeros_like(t)
    rr = 0.46 * fw
    for cx, cy in ((ox0, oy0), (ox1, oy0), (ox0, oy1), (ox1, oy1)):
        ux = cx + (0.5 * fw if cx == ox0 else -0.5 * fw)
        uy = cy + (0.5 * fw if cy == oy0 else -0.5 * fw)
        rx, ry = X - ux, Y - uy
        rd = np.hypot(rx, ry)
        petals = 1.0 + 0.2 * np.cos(8 * np.arctan2(ry, rx)) * orn
        dome = np.sqrt(np.clip(1.0 - (rd / (rr * petals)) ** 2, 0.0, 1.0))
        one = 0.36 + 0.42 * dome - 0.08 * np.exp(-(rd / (0.2 * rr)) ** 2) * orn
        rose = np.maximum(rose, np.where(rd < rr * petals, one, 0.0))
    p = np.maximum(p, rose)
    hgt = p * fw * (0.16 + 0.12 * orn)           # relief: flatter on a narrow moulding
    nrm = kit.normal_from_height(hgt * ss)
    burn = np.clip(torus / 0.28 + outer_bead / 0.12 + inner_bead / 0.08 + (rose > 0) * 1.0, 0.0, 1.0)
    spec = (0.82 + 0.18 * burn).astype(F32)
    gloss = (8.0 + 26.0 * burn).astype(F32)
    hol = hollow / 0.12
    ao = (1.0 - 0.30 * hol - 0.25 * _ss(0.86, 1.0, t)).astype(F32)
    alb = GOLD + (BOLE - GOLD) * (0.30 * hol * orn)[..., None]
    return cov, alb.astype(F32), nrm, spec, gloss, ao


# =============================================================================
def _down(a, ss):
    if ss == 1:
        return a.astype(F32)
    H, W = a.shape[:2]
    return a.reshape(H // ss, ss, W // ss, ss, *a.shape[2:]).mean(axis=(1, 3)).astype(F32)


def render(h, seed=0):
    """The Divine Mercy image in its frame, as a kit.Sprite of height h (width ~0.59 h)."""
    h = max(6, int(h))
    lay = _Layout(h)
    W, ss = lay.W, lay.ss
    rng = np.random.default_rng((int(seed) & 0xFFFFFFFF) * 7919 + 17)
    Y, X = kit.grid(h * ss, W * ss)
    X, Y = X / ss, Y / ss                              # final-pixel units at sample centres
    cx0, cy0, cx1, cy1 = lay.canvas
    bx0, by0, bx1, by1 = lay.band
    F = lay.F

    # -- the painted canvas -------------------------------------------------------
    img = np.empty(X.shape + (3,), F32)
    tv = np.clip((Y - cy0) / max(1.0, by0 - cy0), 0.0, 1.0)[..., None]
    img[:] = WALL_TOP + (WALL_LOW - WALL_TOP) * tv
    yh = lay.top + 0.84 * F                            # the floor's horizon behind the figure
    fl = _ss(yh - 0.02 * F - 0.5, yh + 0.06 * F + 0.5, Y)[..., None]
    fn = np.clip((Y - yh) / max(1.0, by0 - yh), 0.0, 1.0)[..., None]
    img[:] = img * (1 - fl) + (FLOOR_FAR + (FLOOR_NEAR - FLOOR_FAR) * fn) * fl
    U = (X - lay.axis) / F
    V = (Y - lay.top) / F
    sh = np.exp(-((U - 0.004) / 0.13) ** 2 - ((V - 0.99) / 0.022) ** 2)     # contact shadow
    img *= (1.0 - 0.6 * sh)[..., None]
    # the figure, painted in a window around it
    fx0, fy0 = lay.to_px(-0.25, -0.045)
    fx1, fy1 = lay.to_px(0.20, 1.01)
    j0, j1 = max(0, int(fy0 * ss)), min(X.shape[0], int(math.ceil(fy1 * ss)))
    i0, i1 = max(0, int(fx0 * ss)), min(X.shape[1], int(math.ceil(fx1 * ss)))
    fig = np.zeros(X.shape, F32)
    if j1 > j0 + 2 and i1 > i0 + 2:
        fc, fcol = _figure(U[j0:j1, i0:i1], V[j0:j1, i0:i1], 1.0 / (F * ss), F, rng)
        fig[j0:j1, i0:i1] = fc
        # a faint radiance in the dark around him
        glow = kit.blur(fig, max(1, int(0.035 * F * ss)))
        img += (glow * (1.0 - fig))[..., None] * kit.rgb((0.040, 0.036, 0.030))
        sub = img[j0:j1, i0:i1]
        sub[:] = sub * (1.0 - fc[..., None]) + fcol * fc[..., None]
    # the rays: translucent glazes over everything below the heart, plus a little light
    m_red, m_pale = _rays(X, Y, lay, rng)
    r3, p3 = m_red[..., None], m_pale[..., None]
    img[:] = img * (1.0 - 0.80 * r3 * (1.0 - RED_TINT)) + RED_GLAZE * r3 * (0.30 * (1.0 - fig))[..., None]
    img[:] = img * (1.0 - 0.60 * p3 * (1.0 - PALE_TINT)) + PALE_GLAZE * p3 * (0.30 * (1.0 - fig))[..., None]
    # they glow most against the dark ground; over the robe they read mostly as tint
    emi = (r3 * RED_EMIT * (1.0 - 0.55 * fig)[..., None] + p3 * PALE_EMIT * (1.0 - 0.25 * fig)[..., None]).astype(F32)
    # the inscription band, plain and dark
    band = ((Y > by0) & (Y < by1)).astype(F32)[..., None]
    img[:] = img * (1.0 - band) + BAND * band
    emi *= 1.0 - band
    # a little shadow in the rabbet, just inside the frame
    dsight = np.minimum(np.minimum(X - cx0, cx1 - X), np.minimum(Y - cy0, cy1 - Y))
    img *= (1.0 - 0.45 * np.exp(-np.maximum(dsight, 0.0) / 0.6))[..., None]

    spr = kit.Sprite.blank(h * ss, W * ss)
    canvas_cov = ((X > cx0) & (X < cx1) & (Y > cy0) & (Y < cy1)).astype(F32)
    m_red *= canvas_cov                                # the masks stop at the frame
    m_pale *= canvas_cov
    spr.paint(canvas_cov, img, spec=0.012, gloss=12.0, emissive=emi)    # a faint varnish sheen
    fcov, falb, fnrm, fspec, fgloss, fao = _frame(X, Y, lay, ss)
    spr.paint(fcov, falb, fnrm, spec=fspec, gloss=fgloss, metal=1.0, ao=fao)

    # -- box-filter down to the final size ---------------------------------------
    if ss > 1:
        nrm = _down(spr.normal, ss)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        spr = kit.Sprite(alpha=_down(spr.alpha, ss), albedo=_down(spr.albedo, ss), normal=nrm.astype(F32),
                         spec=_down(spr.spec, ss), gloss=_down(spr.gloss, ss), metal=_down(spr.metal, ss),
                         emissive=_down(spr.emissive, ss), ao=_down(spr.ao, ss))
        m_red, m_pale = _down(m_red, ss), _down(m_pale, ss)
    spr.anchors = {
        "base": (W / 2.0, float(lay.outer[3])),
        "heart": lay.to_px(*HEART),
        "hand": lay.to_px(*_enlarged(HAND_R, HAND_PIVOT, _hand_k(F))),
        "face": lay.to_px(*_enlarged(FACE, HEAD_PIVOT, _head_k(F))),
        "ray_red": m_red,
        "ray_pale": m_pale,
        "inscription": (float(bx0), float(by0), float(bx1), float(by1)),
    }
    return spr.check()
