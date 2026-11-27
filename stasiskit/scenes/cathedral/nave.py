"""The church of the CATHEDRAL view: geometry, furnishings, cameras and G-buffer.

One Gothic church, modelled as a signed-distance field in metres: x across
the nave (+x is the south side, on the right when you face the altar), y up,
z toward the east end.  The chancel arch stands at z = 0, the aisles end in
flat east walls at z = ZE (Our Lady's altar on the left, the Divine Mercy
shrine on the right) and the sanctuary ends in a five-sided apse.

The free space is a union of "air" volumes under pointed-arch profiles
(nave, aisles, arcade openings, chancel arch, sanctuary and apse, window
embrasures, Our Lady's niche); the masonry is everything else.  Solid detail
is added on top: clustered piers with bases and capitals, rib vaults, string
courses, the sanctuary steps, pews and the furnishings.  Each detail group has
a bounding box and is evaluated only for rays that come near it, which keeps
the raymarch cheap enough to run once per (camera, size).

`build(cam, W, H)` traces every pixel once and returns a `GBuffer` (world
position, depth, view-space normal, material, albedo / specular / emission,
ambient occlusion and per-pixel object ids for votive cups and windows).
Primary rays start with a coarse cone-marching pass (one cone per 4..8 pixel
block gives every pixel in the block a safe starting distance), and rays that
reach the floor plane stop on it analytically.  The scene lights the G-buffer
(light.py) and pastes the devotional subjects into it (subjects.py).

Where things are: the dimensions and furniture positions are the constants
below; `flames()` lists every flame (and which light group it belongs to),
`votive_cups()` the votive glasses; `CAMERAS` holds the four fixed cameras,
each with a closer framing for small screens.
"""

import math
from dataclasses import dataclass

import numpy as np

F32 = np.float32

# =============================================================================
# dimensions (metres)
# =============================================================================
XP = 3.3                  # arcade line (pier centres), |x|
WT = 0.32                 # half thickness of the arcade walls
XN = XP - WT              # nave half width
XW = 6.45                 # aisle outer walls (inner face)
XA0 = XP + WT
XAC = (XA0 + XW) / 2      # aisle centre line = axis of the side altars
XAH = (XW - XA0) / 2
BAY = 4.6
NBAY = 7
ZW = -NBAY * BAY - 0.3    # west end of the modelled nave (always behind the cameras)
ZE = 0.75                 # aisle east walls (just behind the crossing piers)
ZA = 7.0                  # where the apse polygon starts
YV, KV = 10.0, 0.75       # nave / sanctuary vault: springing height, pointedness
YA, KA = 5.0, 0.8         # aisle vaults
YAR, KAR = 4.2, 0.7       # arcade arches
YC, KC = 5.8, 0.75        # chancel arch
YS = 0.51                 # sanctuary floor (three steps of 0.17 m)
YP = 0.68                 # altar foot-pace

# high altar
ALT_Z0, ALT_Z1 = 2.55, 3.25          # altar block front / back
ALT_TOP = 1.68
GRAD_TOP = 1.86                      # gradine (candle shelf)
THRONE_Y = 2.33                      # top of the thabor: the monstrance stands here
MON_Z = 3.05
CANDLE_X = (0.55, 0.85, 1.12)        # altar candlesticks (both sides), on the gradine
CANDLE_Z = 3.13
CANDLE_STICK = (0.44, 0.38, 0.32)    # stick heights, tallest nearest the monstrance
CANDLE_LEN = 0.30
CANDELABRA = (1.72, 2.12)            # floor candelabra (|x|, z) on the foot-pace, before the altar
CENSER = (1.3, 1.65)                 # censer stand (x, z) on the sanctuary floor
LAMP = (-2.05, 3.05, 1.75)           # sanctuary lamp (red glass), hanging

# Our Lady's altar (left aisle) and the Divine Mercy shrine (right aisle)
XM = -XAC
XR = XAC
NICHE_Y0, NICHE_SPRING, NICHE_HW = 1.24, 3.12, 0.6
MARY_BASE = (XM, 1.44, ZE + 0.27)
MERCY_BASE = (XR, 1.62, ZE - 0.035)              # hung above the shrine's candles
MERCY_CANDLE_Z = ZE - 0.3                        # the shrine's candlesticks, clear of the wall
VOTIVE_Z = ZE - 1.02                            # one tiered bank before Our Lady's altar
VOTIVE_TIERS, VOTIVE_PER = 3, 9
VOTIVE_SP, VOTIVE_Y0, VOTIVE_DY, VOTIVE_DZ = 0.112, 0.60, 0.13, 0.12
KNEELER_Z = ZE - 1.55
KNEELER_X = XAC - 0.62                     # the prie-dieu stands to the left, clear of the floor

# =============================================================================
# materials
# =============================================================================
(M_NONE, M_STONE, M_FLOOR, M_CHECK, M_MARBLE, M_WOOD, M_GLASS, M_GOLD, M_LINEN, M_FRONTAL, M_NICHE,
 M_WAX, M_IRON, M_VOTIVE, M_LAMP, M_BRASS, M_VELVET, M_PETAL_W, M_PETAL_R, M_LEAF, M_BLUE, M_VAULT) = range(22)

#            albedo                spec   gloss  metal
MATS = {
    M_STONE: ((0.44, 0.40, 0.345), 0.04, 8.0, 0.0),
    M_FLOOR: ((0.30, 0.285, 0.26), 0.30, 46.0, 0.0),
    M_CHECK: ((0.60, 0.58, 0.54), 0.45, 70.0, 0.0),
    M_MARBLE: ((0.70, 0.67, 0.62), 0.30, 40.0, 0.0),
    M_WOOD: ((0.12, 0.068, 0.036), 0.3, 34.0, 0.0),
    M_GLASS: ((0.02, 0.02, 0.025), 0.10, 30.0, 0.0),
    M_GOLD: ((1.0, 0.74, 0.30), 1.0, 55.0, 1.0),
    M_LINEN: ((0.86, 0.84, 0.78), 0.04, 10.0, 0.0),
    M_FRONTAL: ((0.84, 0.80, 0.68), 0.12, 20.0, 0.0),
    M_NICHE: ((0.02, 0.055, 0.30), 0.05, 10.0, 0.0),
    M_WAX: ((0.93, 0.89, 0.76), 0.18, 22.0, 0.0),
    M_IRON: ((0.05, 0.045, 0.04), 0.35, 30.0, 0.5),
    M_VOTIVE: ((0.30, 0.10, 0.08), 0.6, 60.0, 0.0),
    M_LAMP: ((0.50, 0.03, 0.03), 0.6, 60.0, 0.0),
    M_BRASS: ((0.88, 0.63, 0.28), 0.9, 42.0, 1.0),
    M_VELVET: ((0.11, 0.009, 0.016), 0.06, 6.0, 0.0),
    M_PETAL_W: ((0.92, 0.90, 0.84), 0.05, 10.0, 0.0),
    M_PETAL_R: ((0.66, 0.05, 0.09), 0.08, 12.0, 0.0),
    M_LEAF: ((0.05, 0.15, 0.06), 0.10, 16.0, 0.0),
    M_BLUE: ((0.05, 0.10, 0.34), 0.10, 18.0, 0.0),
    M_VAULT: ((0.025, 0.05, 0.16), 0.05, 10.0, 0.0),
}


# =============================================================================
# SDF building blocks (flat float32 arrays, y UP, metres)
# =============================================================================
def _arch(u, y, half, spring, k):
    """2D SDF of a pointed-arch opening (negative inside): straight sides |u| < half
    below `spring`, two arcs above it (k = 0.5 round, 1 equilateral, > 1 lancet).
    Below the springing the max(dy, 0) collapses the arc term into the side term."""
    r = 2.0 * half * k
    off = r - half
    au = np.abs(u) + off
    dy = np.maximum(y - spring, 0.0)
    return np.sqrt(au * au + dy * dy) - r


def _cbox(x, y, z, cx, cy, cz, hx, hy, hz):
    """Box lower-bound SDF (Chebyshev): exact on the faces, sharp edges, cheap."""
    return np.maximum(np.maximum(np.abs(x - cx) - hx, np.abs(y - cy) - hy), np.abs(z - cz) - hz)


def _rbox(x, y, z, cx, cy, cz, hx, hy, hz, r):
    """Exact rounded box."""
    qx = np.abs(x - cx) - hx + r
    qy = np.abs(y - cy) - hy + r
    qz = np.abs(z - cz) - hz + r
    ox, oy, oz = np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)
    return np.sqrt(ox * ox + oy * oy + oz * oz) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r


def _cyl(x, y, z, cx, cz, r, y0, y1):
    dx, dz = x - cx, z - cz
    return np.maximum(np.sqrt(dx * dx + dz * dz) - r, np.maximum(y0 - y, y - y1))


def _sph(x, y, z, cx, cy, cz, r):
    dx, dy, dz = x - cx, y - cy, z - cz
    return np.sqrt(dx * dx + dy * dy + dz * dz) - r


def _lathe(x, y, z, cx, cz, ys, rs):
    """Solid of revolution about a vertical axis; radius rs at heights ys (piecewise linear)."""
    dx, dz = x - cx, z - cz
    r = np.sqrt(dx * dx + dz * dz)
    prof = np.interp(y, ys, rs).astype(F32)
    return np.maximum((r - prof) * 0.75, np.maximum(ys[0] - y, y - ys[-1]))


def _rep(v, period):
    """v folded into [-period/2, period/2) (cheap np.mod)."""
    return v - np.floor(v / period + 0.5) * period


class _Acc:
    """min() accumulator that also remembers which material won."""

    def __init__(self, n, want_mat):
        self.d = np.full(n, 1e9, F32)
        self.m = np.zeros(n, np.int8) if want_mat else None

    def add(self, d, mat):
        if self.m is not None:
            self.m[d < self.d] = mat
        self.d = np.minimum(self.d, d)

    def out(self):
        return (self.d, self.m) if self.m is not None else self.d


def _hash2(i, j, s=0.0):
    """Cheap deterministic per-cell random in [0, 1) for integer-valued float arrays."""
    v = np.sin(i * 127.1 + j * 311.7 + s * 74.7) * 43758.5453
    return (v - np.floor(v)).astype(F32)


# =============================================================================
# the building
# =============================================================================
def _rooms(x, y, z):
    """Union of the main air volumes (negative inside), and helpers reused below."""
    ax = np.abs(x)
    zb = _rep(z + BAY / 2, BAY)                       # 0 at bay centres, +-BAY/2 at the piers
    d_n = np.maximum(_arch(x, y, XN, YV, KV), z + 0.45)
    d_a = np.maximum(_arch(ax - XAC, y, XAH, YA, KA), z - ZE)
    d_o = np.maximum(np.maximum(_arch(zb, y, BAY / 2 - 0.42, YAR, KAR), np.abs(ax - XP) - 0.5), z + 0.4)
    d_c = np.maximum(_arch(x, y, XN - 0.15, YC, KC), np.abs(z) - 0.6)
    dz = z - ZA
    rho = np.maximum(np.maximum(ax, (ax + dz) * 0.70711), dz)
    d_s = np.maximum(_arch(rho, y, XN, YV, KV), 0.45 - z)
    air = np.minimum(np.minimum(np.minimum(d_n, d_a), np.minimum(d_o, d_c)), d_s)
    return np.maximum(air, np.maximum(-y, ZW - z)), ax, zb, dz, rho


def _recesses(x, y, z, ax, zb, dz):
    """Window embrasures (glass at the back of each) and Our Lady's niche, as air."""
    w_a = np.maximum(np.maximum(_arch(zb, y, 0.62, 3.75, 1.0), 2.1 - y),
                     np.maximum(np.maximum(ax - (XW + 0.38), (XW - 0.3) - ax), z - (ZE - 0.9)))
    w_c = np.maximum(np.maximum(_arch(zb, y, 0.62, 9.0, 1.0), 7.6 - y),
                     np.maximum(np.maximum(ax - (XN + 0.38), (XN - 0.3) - ax), z + 0.7))
    zs = z - 2.75 - 2.5 * np.clip(np.floor((z - 2.75) / 2.5 + 0.5), 0, 1)
    w_s = np.maximum(np.maximum(_arch(zs, y, 0.5, 7.8, 1.1), 3.0 - y),
                     np.maximum(np.maximum(ax - (XN + 0.38), (XN - 0.3) - ax), np.maximum(0.9 - z, z - (ZA - 0.2))))
    w0 = dz - XN
    w_p0 = np.maximum(np.maximum(_arch(x, y, 0.58, 7.9, 1.1), 3.0 - y), np.maximum(w0 - 0.38, -0.3 - w0))
    w1 = (ax + dz) * 0.70711 - XN
    u1 = (ax - dz) * 0.70711
    w_p1 = np.maximum(np.maximum(_arch(u1, y, 0.5, 7.8, 1.1), 3.0 - y),
                      np.maximum(np.maximum(w1 - 0.38, -0.3 - w1), -dz))
    w_m = np.maximum(np.maximum(_arch(x - XM, y, NICHE_HW, NICHE_SPRING, 1.0), NICHE_Y0 - y),
                     np.maximum(z - (ZE + 0.55), (ZE - 0.3) - z))
    return np.minimum(np.minimum(np.minimum(w_a, w_c), np.minimum(w_s, w_p0)), np.minimum(w_p1, w_m))


def _courses(ax, y, z, rho):
    """String courses: horizontal mouldings that catch the light."""
    sc_n = np.maximum(np.sqrt((ax - XN) ** 2 + (y - 7.0) ** 2) - 0.09, z + 0.6)
    sc_a = np.maximum(np.sqrt((ax - XW) ** 2 + (y - 2.05) ** 2) - 0.08, z - ZE)
    sc_s = np.maximum(np.sqrt((rho - XN) ** 2 + (y - 2.9) ** 2) - 0.09, 0.6 - z)
    return np.minimum(np.minimum(sc_n, sc_a), sc_s)


def _shell(x, y, z, want_mat=False):
    """Masonry: everything that is not inside one of the air volumes.

    The recesses only remove masonry and the string courses stand at most 0.1 m
    proud of a wall, so for points well away from every wall the bare rooms give
    a safe (slightly short) distance and the details are skipped."""
    air, ax, zb, dz, rho = _rooms(x, y, z)
    d = -air
    near = d < 0.6
    if want_mat or near.all():
        air = np.minimum(air, _recesses(x, y, z, ax, zb, dz))
        d = np.minimum(-air, _courses(ax, y, z, rho))
    else:
        d = d - 0.1
        i = np.flatnonzero(near)
        if i.size:
            a2 = np.minimum(air[i], _recesses(x[i], y[i], z[i], ax[i], zb[i], dz[i]))
            d[i] = np.minimum(-a2, _courses(ax[i], y[i], z[i], rho[i]))
    if not want_mat:
        return d
    return d, np.full(d.shape, M_STONE, np.int8)


def _piers(x, y, z, want_mat=False):
    """Clustered piers: a core with four shafts and four slimmer diagonal shafts, a
    moulded base and a flared capital; the shaft facing the nave runs up to the vault."""
    ax = np.abs(x)
    k = np.clip(np.floor(-z / BAY + 0.5), 0, NBAY)
    zl = z + k * BAY
    xl = ax - XP
    cr = (0.40 + 0.04 * (k == 0)).astype(F32)                  # the crossing piers are a little stouter
    c = np.sqrt(xl * xl + zl * zl) - cr
    axl, azl = np.abs(xl), np.abs(zl)
    s1 = np.sqrt((axl - cr) ** 2 + zl * zl) - 0.12
    s2 = np.sqrt(xl * xl + (azl - cr) ** 2) - 0.12
    a = np.abs(xl + zl) * 0.70711
    b = np.abs(xl - zl) * 0.70711
    s3 = np.sqrt((a - cr) ** 2 + b * b) - 0.085
    s4 = np.sqrt(a * a + (b - cr) ** 2) - 0.085
    d2 = np.minimum(np.minimum(np.minimum(c, s1), np.minimum(s2, s3)), s4)
    col = np.maximum(d2, y - 4.3)
    plinth = np.maximum(np.maximum(axl, azl) - (cr + 0.26), y - 0.30)
    torus = np.sqrt((d2 - 0.03) ** 2 + (y - 0.37) ** 2) - 0.07
    cap = np.maximum(d2 - 0.05 - 0.11 * np.clip((y - 3.85) / 0.3, 0, 1), np.abs(y - 4.05) - 0.2)
    abacus = np.maximum(np.maximum(axl, azl) - (cr + 0.24), np.abs(y - 4.31) - 0.06)
    shaft = np.maximum(np.sqrt((xl + cr) ** 2 + zl * zl) - 0.13, y - YV)
    d = np.minimum(np.minimum(np.minimum(col, plinth), np.minimum(torus, cap)), np.minimum(abacus, shaft))
    if not want_mat:
        return d
    return d, np.full(d.shape, M_STONE, np.int8)


_DG_N = (BAY / 2) / XN
_DG_A = (BAY / 2) / XAH


def _ribs(x, y, z, want_mat=False):
    """Rib vaults: transverse ribs over the piers, diagonal ribs across each bay, a ridge
    rib along the nave, and ribs fanning from the apse corners to the keystone."""
    ax = np.abs(x)
    zb = _rep(z + BAY / 2, BAY)
    zp = z + np.clip(np.floor(-z / BAY + 0.5), 0, NBAY) * BAY
    vn = np.abs(_arch(x, y, XN, YV, KV))
    t_n = np.maximum(vn - 0.22, np.abs(zp) - 0.14)
    g_n = np.abs(ax * _DG_N - np.abs(zb)) / math.sqrt(_DG_N ** 2 + 1)
    d_n = np.minimum(np.minimum(t_n, np.maximum(vn - 0.16, g_n - 0.09)), np.maximum(vn - 0.15, ax - 0.09))
    d_n = np.maximum(d_n, z + 0.45)
    va = np.abs(_arch(ax - XAC, y, XAH, YA, KA))
    t_a = np.maximum(va - 0.18, np.abs(zp) - 0.12)
    g_a = np.abs(np.abs(ax - XAC) * _DG_A - np.abs(zb)) / math.sqrt(_DG_A ** 2 + 1)
    d_a = np.maximum(np.minimum(t_a, np.maximum(va - 0.13, g_a - 0.07)), np.maximum(z - ZE, XA0 - 0.3 - ax))
    dz = z - ZA
    rho = np.maximum(np.maximum(ax, (ax + dz) * 0.70711), dz)
    vs = np.abs(_arch(rho, y, XN, YV, KV))
    t_s = np.maximum(vs - 0.2, np.minimum(np.abs(z - 0.55), np.minimum(np.abs(z - 3.8), np.abs(dz))) - 0.13)
    c1 = np.abs(ax * 0.92388 - dz * 0.38268)
    c2 = np.abs(ax * 0.38268 - dz * 0.92388)
    r_s = np.maximum(np.maximum(vs - 0.16, np.minimum(c1, c2) - 0.09), -dz)
    ridge = np.maximum(vs - 0.15, ax - 0.09)
    d_s = np.maximum(np.minimum(np.minimum(t_s, r_s), ridge), 0.45 - z)
    d = np.minimum(np.minimum(d_n, d_a), d_s)
    if not want_mat:
        return d
    return d, np.full(d.shape, M_STONE, np.int8)


def _steps(x, y, z, want_mat=False):
    """Three steps up into the sanctuary across the chancel arch, and the altar foot-pace."""
    ax = np.abs(x)
    s1 = _cbox(ax, y, z, 0, 0.085, 4.0, XN, 0.085, 4.95)
    s2 = _cbox(ax, y, z, 0, 0.17, 4.15, XN, 0.17, 4.8)
    s3 = _cbox(ax, y, z, 0, 0.255, 4.3, XN, 0.255, 4.65)
    fp = _cbox(ax, y, z, 0, 0.34, 2.95, 2.0, 0.34, 1.05)
    d = np.minimum(np.minimum(s1, s2), np.minimum(s3, fp))
    if not want_mat:
        return d
    return d, np.full(d.shape, M_MARBLE, np.int8)


PEW_Z0, PEW_Z1, PEW_P = -31.0, -2.35, 0.96


def _pews(x, y, z, want_mat=False):
    """Rows of dark oak pews either side of the central aisle (backs face the west door)."""
    ax = np.abs(x)
    zr = _rep(z - PEW_Z1 + PEW_P / 2, PEW_P)            # row-local, rows end at PEW_Z1
    xc = 1.76
    seat = _cbox(ax, y, zr, xc, 0.445, 0.02, 0.96, 0.022, 0.2)
    back = _cbox(ax, y + zr * 0.12, zr, xc, 0.69, -0.215, 0.96, 0.235, 0.022)
    rail = _rbox(ax, y, zr, xc, 0.935, -0.245, 0.96, 0.026, 0.042, 0.015)
    ends = _cbox(np.abs(ax - xc), y, zr, 0.995, 0.47, -0.02, 0.03, 0.47, 0.27)
    knob = _sph(np.abs(ax - xc), y, zr, 0.995, 0.97, -0.17, 0.065)
    kneel = _cbox(ax, y, zr, xc, 0.13, 0.3, 0.93, 0.045, 0.07)
    d = np.minimum(np.minimum(np.minimum(seat, back), np.minimum(rail, ends)), np.minimum(knob, kneel))
    d = np.maximum(d, np.maximum(PEW_Z0 - z, z - (PEW_Z1 + 0.05)))
    if not want_mat:
        return d
    return d, np.full(d.shape, M_WOOD, np.int8)


def _candle_x(ax):
    """Nearest altar candlestick (on the gradine) to |x|: centre and stick height."""
    cx = np.where(ax < 0.7, CANDLE_X[0], np.where(ax < 0.985, CANDLE_X[1], CANDLE_X[2])).astype(F32)
    h = np.where(ax < 0.7, CANDLE_STICK[0], np.where(ax < 0.985, CANDLE_STICK[1], CANDLE_STICK[2])).astype(F32)
    return cx, h


_STICK_Y = np.array([0.0, 0.03, 0.06, 0.10, 0.16, 0.40, 0.46, 0.86, 0.92, 0.97, 1.0], np.float32)
_STICK_R = np.array([0.10, 0.10, 0.06, 0.035, 0.03, 0.022, 0.04, 0.02, 0.025, 0.055, 0.03], np.float32)


def _altar(x, y, z, want_mat=False):
    """High altar: block and mensa, linen cloth, white-and-gold frontal, gradine,
    tabernacle and the thabor the monstrance stands on."""
    ax = np.abs(x)
    acc = _Acc(x.shape[0], want_mat)
    acc.add(_cbox(ax, y, z, 0, (YP + 1.58) / 2, (ALT_Z0 + ALT_Z1) / 2, 1.28, (1.58 - YP) / 2, 0.35), M_MARBLE)
    acc.add(_rbox(ax, y, z, 0, 1.63, 2.9, 1.36, 0.05, 0.4, 0.015), M_MARBLE)
    cloth = np.minimum(_cbox(ax, y, z, 0, 1.69, 2.9, 1.41, 0.012, 0.41),
                       _cbox(ax, y, z, 1.395, 1.42, 2.9, 0.015, 0.27, 0.41))
    acc.add(cloth, M_LINEN)
    acc.add(_cbox(ax, y, z, 0, 1.15, ALT_Z0 - 0.012, 1.2, 0.4, 0.012), M_FRONTAL)
    acc.add(_cbox(ax, y, z, 0, (ALT_TOP + GRAD_TOP) / 2, 3.13, 1.25, (GRAD_TOP - ALT_TOP) / 2, 0.12), M_MARBLE)
    acc.add(_rbox(ax, y, z, 0, 1.97, 3.12, 0.27, 0.29, 0.15, 0.02), M_GOLD)
    acc.add(_cyl(x, y, z, 0, MON_Z, 0.17, 2.26, THRONE_Y), M_GOLD)
    return acc.out() if want_mat else acc.d


def _throne(x, y, z, want_mat=False):
    """Exposition throne: a velvet backdrop, two slim gilded columns, a pointed canopy."""
    ax = np.abs(x)
    acc = _Acc(x.shape[0], want_mat)
    acc.add(_cbox(ax, y, z, 0, 2.98, 3.27, 0.36, 0.7, 0.015), M_VELVET)
    acc.add(_cyl(ax, y, z, 0.41, 3.2, 0.024, 2.26, 3.7), M_GOLD)
    can = np.maximum(np.abs(_arch(x, y, 0.41, 3.7, 0.9)) - 0.03, np.abs(z - 3.22) - 0.04)
    acc.add(np.maximum(can, 3.62 - y), M_GOLD)
    acc.add(_sph(x, y, z, 0, 4.37, 3.22, 0.045), M_GOLD)
    return acc.out() if want_mat else acc.d


def _altar_candles(x, y, z, want_mat=False):
    """Six graded brass candlesticks on the gradine (lathe profile scaled to each
    stick's height) with their candles."""
    ax = np.abs(x)
    acc = _Acc(x.shape[0], want_mat)
    cx, h = _candle_x(ax)
    yl = (y - GRAD_TOP) / h
    dx, dz = ax - cx, z - CANDLE_Z
    r = np.sqrt(dx * dx + dz * dz)
    prof = np.interp(yl, _STICK_Y, _STICK_R).astype(F32) * (0.75 + 0.6 * h)
    acc.add(np.maximum((r - prof) * 0.7, np.maximum(-yl, yl - 1.0) * h), M_BRASS)
    acc.add(np.maximum(r - 0.026, np.maximum(GRAD_TOP + h - y, y - (GRAD_TOP + h + CANDLE_LEN))), M_WAX)
    return acc.out() if want_mat else acc.d


# the flowers stand on the gradine, either side of the tabernacle and inside the candlesticks:
# never on the mensa itself, which is kept for what the celebration needs (GIRM 305-306)
VASE_X, VASE_Z = 0.37, CANDLE_Z
VASE_R = 0.1                         # bouquet radius (m)


def _altar_flowers(x, y, z, want_mat=False):
    """Two slim gilded vases of lilies and roses on the gradine, beside the tabernacle."""
    ax = np.abs(x)
    acc = _Acc(x.shape[0], want_mat)
    vase = _lathe(ax, y, z, VASE_X, VASE_Z, np.array([GRAD_TOP, GRAD_TOP + 0.03, GRAD_TOP + 0.14, GRAD_TOP + 0.18], F32),
                  np.array([0.04, 0.05, 0.03, 0.042], F32))
    acc.add(vase, M_GOLD)
    _flowers(ax, y, z, VASE_X, GRAD_TOP + 0.28, VASE_Z, VASE_R, acc, seed=1)
    return acc.out() if want_mat else acc.d


_FLOWER_OFFS = None


def _flower_offsets():
    global _FLOWER_OFFS
    if _FLOWER_OFFS is None:
        rng = np.random.default_rng(21)
        pts = []
        for _ in range(14):
            th = rng.uniform(0, 2 * math.pi)
            ph = math.acos(rng.uniform(-0.3, 1.0))
            rr = rng.uniform(0.6, 0.85)
            pts.append((math.sin(ph) * math.cos(th) * rr, math.cos(ph) * rr * 0.9, math.sin(ph) * math.sin(th) * rr,
                        rng.uniform(0.24, 0.32), 0 if rng.random() < 0.55 else 1))
        _FLOWER_OFFS = pts
    return _FLOWER_OFFS


def _flowers(x, y, z, cx, cy, cz, rad, acc, seed=0, palette=(M_PETAL_W, M_PETAL_R)):
    """A rounded bouquet: dark leaves with white and red blossoms on its surface."""
    leaves = _sph(x, y, z, cx, cy - rad * 0.2, cz, rad * 0.66)
    acc.add(leaves, M_LEAF)
    near = np.flatnonzero(leaves < rad * 0.8)          # blossoms reach at most ~0.71 rad beyond the leaves
    if near.size == 0:
        return acc
    xs, ys, zs = x[near], y[near], z[near]
    best_w = np.full(near.size, 1e9, F32)
    best_r = np.full(near.size, 1e9, F32)
    for (ox, oy, oz, s, kind) in _flower_offsets()[seed % 3:][:10]:
        d = _sph(xs, ys, zs, cx + ox * rad, cy + oy * rad, cz + oz * rad, s * rad)
        if kind == 0:
            best_w = np.minimum(best_w, d)
        else:
            best_r = np.minimum(best_r, d)
    bw = np.full(x.shape, 1e9, F32)
    br = np.full(x.shape, 1e9, F32)
    bw[near] = best_w
    br[near] = best_r
    acc.add(bw, palette[0])
    acc.add(br, palette[1])
    return acc


def _candelabra(x, y, z, want_mat=False):
    """Two tall floor candlesticks on the foot-pace either side of the altar."""
    ax = np.abs(x)
    acc = _Acc(x.shape[0], want_mat)
    ys = np.array([YP, YP + 0.04, YP + 0.12, YP + 0.2, YP + 0.9, YP + 0.96, YP + 1.32, YP + 1.38, YP + 1.42], F32)
    rs = np.array([0.17, 0.16, 0.06, 0.045, 0.035, 0.06, 0.03, 0.07, 0.035], F32)
    acc.add(_lathe(ax, y, z, CANDELABRA[0], CANDELABRA[1], ys, rs), M_BRASS)
    acc.add(_cyl(ax, y, z, CANDELABRA[0], CANDELABRA[1], 0.034, YP + 1.42, YP + 1.42 + 0.34), M_WAX)
    return acc.out() if want_mat else acc.d


def _censer(x, y, z, want_mat=False):
    """A brass thurible hanging from a tall stand, beside the altar steps."""
    acc = _Acc(x.shape[0], want_mat)
    cx, cz = CENSER
    ys = np.array([YS, YS + 0.03, YS + 0.12, YS + 1.15, YS + 1.2], F32)
    rs = np.array([0.16, 0.14, 0.03, 0.018, 0.03], F32)
    acc.add(_lathe(x, y, z, cx, cz, ys, rs), M_BRASS)
    arm = _cbox(x, y, z, cx - 0.12, YS + 1.19, cz, 0.12, 0.015, 0.015)
    acc.add(arm, M_BRASS)
    hx = cx - 0.24
    acc.add(_cyl(x, y, z, hx, cz, 0.006, YS + 0.98, YS + 1.19), M_BRASS)
    bowl = np.maximum(_sph(x, y, z, hx, YS + 0.86, cz, 0.1), y - (YS + 0.88))
    lid = np.maximum(_sph(x, y, z, hx, YS + 0.84, cz, 0.095), (YS + 0.88) - y)
    acc.add(np.minimum(bowl, lid), M_BRASS)
    acc.add(_sph(x, y, z, hx, YS + 0.965, cz, 0.022), M_BRASS)
    return acc.out() if want_mat else acc.d


def _lamp(x, y, z, want_mat=False):
    """The sanctuary lamp: red glass in a brass holder, hung from the vault on a chain."""
    acc = _Acc(x.shape[0], want_mat)
    lx, ly, lz = LAMP
    acc.add(_cyl(x, y, z, lx, lz, 0.062, ly - 0.07, ly + 0.08), M_LAMP)
    cup = np.maximum(_sph(x, y, z, lx, ly - 0.02, lz, 0.115), y - (ly - 0.05))
    acc.add(cup, M_BRASS)
    acc.add(_cyl(x, y, z, lx, lz, 0.08, ly + 0.08, ly + 0.11), M_BRASS)
    acc.add(_cyl(x, y, z, lx, lz, 0.008, ly + 0.11, 12.0), M_IRON)
    return acc.out() if want_mat else acc.d


def _mary_altar(x, y, z, want_mat=False):
    """Our Lady's altar: marble altar with a blue frontal, linen, gradine and the
    pedestal inside the niche."""
    u = x - XM
    au = np.abs(u)
    acc = _Acc(x.shape[0], want_mat)
    acc.add(_cbox(au, y, z, 0, 0.07, ZE - 0.52, 0.95, 0.07, 0.52), M_MARBLE)
    acc.add(_cbox(au, y, z, 0, 0.55, ZE - 0.3, 0.7, 0.41, 0.3), M_MARBLE)
    acc.add(_rbox(au, y, z, 0, 0.97, ZE - 0.31, 0.75, 0.035, 0.32, 0.01), M_MARBLE)
    acc.add(np.minimum(_cbox(au, y, z, 0, 1.012, ZE - 0.31, 0.78, 0.008, 0.33),
                       _cbox(au, y, z, 0, 0.9, ZE - 0.637, 0.78, 0.11, 0.008)), M_LINEN)
    acc.add(_cbox(au, y, z, 0, 0.47, ZE - 0.606, 0.6, 0.3, 0.008), M_BLUE)
    acc.add(_cbox(au, y, z, 0, 1.08, ZE - 0.12, 0.72, 0.07, 0.12), M_MARBLE)
    ped = _rbox(au, y, z, 0, (NICHE_Y0 + MARY_BASE[1]) / 2, ZE + 0.25, 0.3, (MARY_BASE[1] - NICHE_Y0) / 2, 0.22, 0.02)
    acc.add(ped, M_MARBLE)
    return acc.out() if want_mat else acc.d


def _mary_frame(x, y, z, want_mat=False):
    """The niche's frame: an arch moulding standing proud of the wall, a gable with a
    finial, and two pinnacles."""
    u = x - XM
    au = np.abs(u)
    acc = _Acc(x.shape[0], want_mat)
    ring = np.abs(_arch(u, y, NICHE_HW + 0.07, NICHE_SPRING, 1.0)) - 0.075
    acc.add(np.maximum(np.maximum(ring, NICHE_Y0 - 0.02 - y), np.abs(z - (ZE - 0.04)) - 0.05), M_STONE)
    gy = NICHE_SPRING + 1.25                                    # gable apex above the arch
    gab = np.abs(au * 1.35 + (y - gy)) / 1.68 - 0.05             # two sloping copings
    gab = np.maximum(np.maximum(gab, np.maximum(au - 0.95, (NICHE_SPRING + 0.35) - y)), np.abs(z - (ZE - 0.04)) - 0.05)
    acc.add(gab, M_STONE)
    pin_x = 0.92
    top = NICHE_SPRING + 1.1
    pin = _cbox(au, y, z, pin_x, (1.14 + top) / 2, ZE - 0.07, 0.07, (top - 1.14) / 2, 0.07)
    spire = np.maximum((np.maximum(np.abs(au - pin_x), np.abs(z - (ZE - 0.07))) - 0.08 + (y - top) * 0.18),
                       np.maximum(top - y, y - (top + 0.46)))
    acc.add(np.minimum(pin, spire), M_STONE)
    acc.add(_sph(au, y, z, 0, gy + 0.08, ZE - 0.05, 0.07), M_STONE)
    return acc.out() if want_mat else acc.d


def _mary_props(x, y, z, want_mat=False):
    """Candlesticks on the gradine, with roses and lilies between them and her plinth."""
    u = x - XM
    au = np.abs(u)
    acc = _Acc(x.shape[0], want_mat)
    sx, sz = 0.52, ZE - 0.1
    stick = _lathe(au, y, z, sx, sz, np.array([1.15, 1.18, 1.22, 1.42, 1.45], F32),
                   np.array([0.06, 0.05, 0.02, 0.017, 0.035], F32))
    acc.add(stick, M_BRASS)
    acc.add(_cyl(au, y, z, sx, sz, 0.02, 1.45, 1.66), M_WAX)
    _flowers(au, y, z, 0.3, 1.27, ZE - 0.05, 0.12, acc, seed=2)
    return acc.out() if want_mat else acc.d


def votive_cups():
    """World positions (n, 3) of the votive cups on the bank before Our Lady's altar (cup
    centre), which half of the bank each is on, and its glass (0 red, 1 blue, 2 amber)."""
    rng = np.random.default_rng(5)
    pts, side, col = [], [], []
    half = (VOTIVE_PER - 1) / 2
    for k in range(VOTIVE_TIERS):
        for i in range(VOTIVE_PER):
            pts.append((XM + (i - half) * VOTIVE_SP, VOTIVE_Y0 + k * VOTIVE_DY + 0.05, VOTIVE_Z + k * VOTIVE_DZ))
            side.append(0 if i <= half else 1)
            col.append(int(rng.choice(3, p=(0.5, 0.28, 0.22))))
    return np.array(pts, F32), np.array(side), np.array(col)


VOTIVE_HW = VOTIVE_SP * VOTIVE_PER / 2 + 0.04


def _votives(x, y, z, want_mat=False):
    """A tiered wrought-iron votive bank with red, blue and amber glass cups."""
    u = x - XM
    au = np.abs(u)
    acc = _Acc(x.shape[0], want_mat)
    k = np.clip(np.floor((y - VOTIVE_Y0) / VOTIVE_DY + 0.5), 0, VOTIVE_TIERS - 1)
    zt = VOTIVE_Z + k * VOTIVE_DZ
    yt = VOTIVE_Y0 + k * VOTIVE_DY
    acc.add(_cbox(au, y, z, 0, yt - 0.012, zt, VOTIVE_HW, 0.012, 0.055), M_IRON)
    ztop = VOTIVE_Z + VOTIVE_DZ * (VOTIVE_TIERS - 1)
    ytop = VOTIVE_Y0 + VOTIVE_DY * (VOTIVE_TIERS - 1)
    legs = np.minimum(_cyl(au, y, z, VOTIVE_HW - 0.01, VOTIVE_Z - 0.05, 0.013, 0.0, VOTIVE_Y0 + 0.05),
                      _cyl(au, y, z, VOTIVE_HW - 0.01, ztop + 0.05, 0.013, 0.0, ytop + 0.24))
    acc.add(legs, M_IRON)
    acc.add(_cbox(au, y, z, 0, 0.05, VOTIVE_Z + 0.1, VOTIVE_HW, 0.012, 0.2), M_IRON)
    ci = np.clip(np.floor(u / VOTIVE_SP + 0.5), -(VOTIVE_PER // 2), VOTIVE_PER // 2)
    acc.add(_cyl(u, y, z, ci * VOTIVE_SP, zt, 0.036, yt, yt + 0.078), M_VOTIVE)
    return acc.out() if want_mat else acc.d


def _mercy(x, y, z, want_mat=False):
    """The Divine Mercy shrine: a small altar with a white cloth, two candlesticks, a row
    of votive lights, and a prie-dieu in front."""
    u = x - XR
    au = np.abs(u)
    acc = _Acc(x.shape[0], want_mat)
    acc.add(_cbox(au, y, z, 0, 0.07, ZE - 0.5, 0.9, 0.07, 0.5), M_MARBLE)
    acc.add(_cbox(au, y, z, 0, 0.53, ZE - 0.28, 0.68, 0.4, 0.28), M_MARBLE)
    acc.add(np.minimum(_cbox(au, y, z, 0, 0.94, ZE - 0.29, 0.74, 0.012, 0.3),
                       _cbox(au, y, z, 0, 0.82, ZE - 0.595, 0.74, 0.13, 0.008)), M_LINEN)
    sx, sz = 0.5, MERCY_CANDLE_Z
    stick = _lathe(au, y, z, sx, sz, np.array([0.95, 0.98, 1.02, 1.24, 1.27], F32),
                   np.array([0.06, 0.05, 0.02, 0.017, 0.034], F32))
    acc.add(stick, M_BRASS)
    acc.add(_cyl(au, y, z, sx, sz, 0.02, 1.27, 1.5), M_WAX)
    ti = np.clip(np.floor(u / 0.13 + 0.5), -2, 2)
    acc.add(_cyl(u, y, z, ti * 0.13, ZE - 0.47, 0.03, 0.952, 1.012), M_VOTIVE)
    # prie-dieu, seen from behind: a kneeling board with a red cushion, two side posts,
    # a low front panel and the sloping book rest on top
    kz = KNEELER_Z
    ak = np.abs(x - KNEELER_X)
    acc.add(_cbox(ak, y, z, 0, 0.06, kz - 0.1, 0.3, 0.06, 0.14), M_WOOD)
    acc.add(_rbox(ak, y, z, 0, 0.15, kz - 0.1, 0.27, 0.035, 0.12, 0.03), M_VELVET)
    acc.add(_cbox(ak, y, z, 0.28, 0.42, kz + 0.04, 0.025, 0.42, 0.025), M_WOOD)
    acc.add(_cbox(ak, y, z, 0, 0.36, kz + 0.06, 0.28, 0.2, 0.018), M_WOOD)
    acc.add(_rbox(ak, y + (z - kz) * 0.4, z, 0, 0.86, kz + 0.02, 0.33, 0.018, 0.12, 0.008), M_WOOD)
    return acc.out() if want_mat else acc.d


def mercy_tealights():
    return np.array([(XR + i * 0.13, 1.03, ZE - 0.47) for i in range(-2, 3)], F32)


def flames():
    """Every flame in the church, as (wick-top position (3,), flame height m, light group,
    cup id or -1).  The light groups follow the candle banks: the scene flickers each
    group as a whole and draws every flame on its own."""
    out = []
    for sgn, g in ((-1, "altar_l"), (1, "altar_r")):
        for cx, h in zip(CANDLE_X, CANDLE_STICK):
            out.append(((sgn * cx, GRAD_TOP + h + CANDLE_LEN, CANDLE_Z), 0.05, g, -1))
        out.append(((sgn * CANDELABRA[0], YP + 1.42 + 0.34, CANDELABRA[1]), 0.055, g, -1))
    cups, side, _ = votive_cups()
    for i, (p, sd) in enumerate(zip(cups, side)):
        out.append(((p[0], p[1] + 0.012, p[2]), 0.026, "votive_l" if sd == 0 else "votive_r", i))
    for sgn in (-1, 1):
        out.append(((XM + sgn * 0.52, 1.66, ZE - 0.1), 0.045, "mary", -1))
    for sgn in (-1, 1):
        out.append(((XR + sgn * 0.5, 1.5, MERCY_CANDLE_Z), 0.045, "mercy", -1))
    for i, p in enumerate(mercy_tealights()):
        out.append(((p[0], p[1] - 0.008, p[2]), 0.02, "mercy", len(cups) + i))
    return out


# -----------------------------------------------------------------------------
# detail groups: (name, sdf, bounding box lo, hi, mirrored in x, casts shadows)
# -----------------------------------------------------------------------------
@dataclass
class Group:
    name: str
    fn: object
    lo: tuple
    hi: tuple
    mirror: bool = False
    occluder: bool = True

    def bound(self, x, y, z):
        """Chebyshev distance to the box: a cheap lower bound of the group's SDF."""
        xx = np.abs(x) if self.mirror else x
        cx, cy, cz = ((a + b) * 0.5 for a, b in zip(self.lo, self.hi))
        hx, hy, hz = ((b - a) * 0.5 for a, b in zip(self.lo, self.hi))
        return np.maximum(np.maximum(np.abs(xx - cx) - hx, np.abs(y - cy) - hy), np.abs(z - cz) - hz)


GROUPS = [
    Group("piers", _piers, (XP - 1.0, -0.1, ZW), (XP + 1.0, YV + 0.1, 0.9), mirror=True),
    Group("ribs", _ribs, (-XW, YA - 0.4, ZW), (XW, 16.0, ZA + XN + 0.5), occluder=False),
    Group("steps", _steps, (-XN, -0.1, -1.0), (XN, YP + 0.05, 9.0)),
    Group("pews", _pews, (0.68, -0.1, PEW_Z0), (2.84, 1.1, PEW_Z1 + 0.1), mirror=True),
    Group("altar", _altar, (-1.45, YP - 0.05, ALT_Z0 - 0.06), (1.45, THRONE_Y + 0.02, ALT_Z1 + 0.05)),
    Group("throne", _throne, (-0.46, 2.24, 3.15), (0.46, 4.45, 3.3)),
    Group("candles", _altar_candles, (CANDLE_X[0] - 0.12, GRAD_TOP - 0.01, CANDLE_Z - 0.12),
          (CANDLE_X[2] + 0.12, GRAD_TOP + CANDLE_STICK[0] + CANDLE_LEN + 0.02, CANDLE_Z + 0.12), mirror=True,
          occluder=False),
    Group("flowers", _altar_flowers, (VASE_X - 0.16, GRAD_TOP - 0.01, VASE_Z - 0.16),
          (VASE_X + 0.16, GRAD_TOP + 0.28 + 0.16, VASE_Z + 0.16), mirror=True, occluder=False),
    Group("candelabra", _candelabra, (CANDELABRA[0] - 0.2, YP - 0.05, CANDELABRA[1] - 0.2),
          (CANDELABRA[0] + 0.2, YP + 1.8, CANDELABRA[1] + 0.2), mirror=True, occluder=False),
    Group("censer", _censer, (CENSER[0] - 0.4, YS - 0.05, CENSER[1] - 0.2), (CENSER[0] + 0.2, YS + 1.3, CENSER[1] + 0.2),
          occluder=False),
    Group("lamp", _lamp, (LAMP[0] - 0.14, LAMP[1] - 0.16, LAMP[2] - 0.14), (LAMP[0] + 0.14, 12.0, LAMP[2] + 0.14),
          occluder=False),
    Group("mary", _mary_altar, (XM - 0.96, -0.05, ZE - 1.05), (XM + 0.96, MARY_BASE[1] + 0.02, ZE + 0.48)),
    Group("frame", _mary_frame, (XM - 1.0, NICHE_Y0 - 0.12, ZE - 0.15), (XM + 1.0, NICHE_SPRING + 1.62, ZE + 0.02)),
    Group("mary_props", _mary_props, (XM - 0.65, 1.12, ZE - 0.25), (XM + 0.65, 1.68, ZE + 0.17), occluder=False),
    Group("votives", _votives, (XM - VOTIVE_HW - 0.05, -0.05, VOTIVE_Z - 0.12),
          (XM + VOTIVE_HW + 0.05, VOTIVE_Y0 + VOTIVE_DY * VOTIVE_TIERS + 0.2, VOTIVE_Z + VOTIVE_DZ * VOTIVE_TIERS + 0.1),
          occluder=False),
    Group("mercy", _mercy, (XR - 0.95, -0.05, KNEELER_Z - 0.35), (XR + 0.95, 1.55, ZE + 0.02)),  # incl. prie-dieu
]
# clusters: one bound test for a whole corner of the church before testing its members
CLUSTERS = [
    ("sanctuary", (-2.2, 0.4, 0.9), (2.2, 12.0, 3.4), ("altar", "throne", "candles", "flowers", "candelabra", "censer", "lamp")),
    ("lady", (XM - 1.05, -0.1, ZE - 1.2), (XM + 1.05, 5.0, ZE + 0.5), ("mary", "frame", "mary_props", "votives")),
]


class World:
    """The whole building as one SDF, with bounding-box guarded detail groups."""

    def __init__(self):
        self.groups = GROUPS
        self.active = GROUPS             # groups visible to the current camera (see cull())
        self.plan = self._plan(GROUPS)

    def cull(self, cam):
        """Keep only the detail groups whose bounding box can appear on screen."""
        keep = []
        for g in GROUPS:
            lo, hi = np.array(g.lo, F32), np.array(g.hi, F32)
            xs = [(-hi[0], -lo[0]), (lo[0], hi[0])] if g.mirror else [(lo[0], hi[0])]
            vis = False
            for x0, x1 in xs:
                c = np.array([[x, y, z] for x in (x0, x1) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], F32)
                sx, sy, dep = cam.project(c)
                if (dep <= 0.05).any():
                    vis = True
                    break
                if sx.max() >= -2 and sx.min() <= cam.W + 2 and sy.max() >= -2 and sy.min() <= cam.H + 2:
                    vis = True
                    break
            if vis:
                keep.append(g)
        self.active = keep
        self.plan = self._plan(keep)
        return keep

    @staticmethod
    def _plan(groups):
        """Visible groups, with the clustered ones gathered under their cluster box."""
        byname = {g.name: g for g in groups}
        plan, used = [], set()
        for _name, lo, hi, members in CLUSTERS:
            ms = [byname[m] for m in members if m in byname]
            if len(ms) > 1:
                plan.append((np.array(lo, F32), np.array(hi, F32), ms))
                used.update(m.name for m in ms)
        plan += [g for g in groups if g.name not in used]
        return plan

    def sdf(self, x, y, z, occluders_only=False):
        d = _shell(x, y, z)
        for item in self.plan:
            if isinstance(item, Group):
                self._eval(item, x, y, z, d, None, occluders_only)
                continue
            lo, hi, members = item
            lb = np.maximum(np.maximum(np.maximum(lo[0] - x, x - hi[0]), np.maximum(lo[1] - y, y - hi[1])),
                            np.maximum(lo[2] - z, z - hi[2]))
            i = np.flatnonzero(lb < d)
            if i.size == 0:
                continue
            xs, ys, zs, ds = x[i], y[i], z[i], d[i]
            for g in members:
                self._eval(g, xs, ys, zs, ds, None, occluders_only)
            d[i] = ds
        return d

    @staticmethod
    def _eval(g, x, y, z, d, _unused, occluders_only):
        """d = min(d, group) in place, only where the group's box is closer than d."""
        if occluders_only and not g.occluder:
            return
        i = np.flatnonzero(g.bound(x, y, z) < d)
        if i.size == 0:
            return
        if i.size == d.size:
            np.minimum(d, g.fn(x, y, z), out=d)
        else:
            d[i] = np.minimum(d[i], g.fn(x[i], y[i], z[i]))

    def material(self, x, y, z):
        d, m = _shell(x, y, z, True)
        for g in self.active:
            lb = g.bound(x, y, z)
            i = np.flatnonzero(lb < d + 0.01)
            if i.size == 0:
                continue
            dg, mg = g.fn(x[i], y[i], z[i], True)
            better = dg < d[i]
            d[i[better]] = dg[better]
            m[i[better]] = mg[better]
        return d, m

    @staticmethod
    def shadow(x, y, z, shell=True):
        """Simplified occluders for shadow rays: the rooms, plain piers, pews as solid
        rows, the altars and the prie-dieu as blocks (no windows, mouldings or props).
        shell=False leaves out the masonry (the moon's way out through the windows is
        tested analytically)."""
        ax = np.abs(x)
        if not shell:
            d = np.full(x.shape, 1e9, F32)
            return World._shadow_props(x, y, z, ax, d)
        zb = _rep(z + BAY / 2, BAY)
        d_n = np.maximum(_arch(x, y, XN, YV, KV), z + 0.45)
        d_a = np.maximum(_arch(ax - XAC, y, XAH, YA, KA), z - ZE)
        d_o = np.maximum(np.maximum(_arch(zb, y, BAY / 2 - 0.42, YAR, KAR), np.abs(ax - XP) - 0.5), z + 0.4)
        d_c = np.maximum(_arch(x, y, XN - 0.15, YC, KC), np.abs(z) - 0.6)
        dz = z - ZA
        rho = np.maximum(np.maximum(ax, (ax + dz) * 0.70711), dz)
        d_s = np.maximum(_arch(rho, y, XN, YV, KV), 0.45 - z)
        w_m = np.maximum(np.maximum(_arch(x - XM, y, NICHE_HW, NICHE_SPRING, 1.0), NICHE_Y0 - y),
                         np.maximum(z - (ZE + 0.55), (ZE - 0.3) - z))
        air = np.minimum(np.minimum(np.minimum(d_n, d_a), np.minimum(d_o, d_c)), np.minimum(d_s, w_m))
        d = -np.maximum(air, -y)
        return World._shadow_props(x, y, z, ax, d)

    @staticmethod
    def _shadow_props(x, y, z, ax, d):
        k = np.clip(np.floor(-z / BAY + 0.5), 0, NBAY)
        zl = z + k * BAY
        xl = ax - XP
        d = np.minimum(d, np.maximum(np.sqrt(xl * xl + zl * zl) - 0.55, y - 4.4))
        zr = _rep(z - PEW_Z1 + PEW_P / 2, PEW_P)
        pew = np.maximum(_cbox(ax, y, zr, 1.76, 0.48, -0.1, 0.99, 0.48, 0.13),
                         np.maximum(PEW_Z0 - z, z - (PEW_Z1 + 0.05)))
        d = np.minimum(d, pew)
        alt = _cbox(ax, y, z, 0, 1.2, (ALT_Z0 + ALT_Z1) / 2, 1.36, 0.5, 0.36)
        tab = _cbox(ax, y, z, 0, 2.0, 3.12, 0.27, 0.32, 0.15)
        ma = _cbox(np.abs(x - XM), y, z, 0, 0.5, ZE - 0.3, 0.75, 0.5, 0.3)
        me = _cbox(np.abs(x - XR), y, z, 0, 0.48, ZE - 0.28, 0.7, 0.48, 0.28)
        kn = _cbox(np.abs(x - KNEELER_X), y, z, 0, 0.3, KNEELER_Z, 0.3, 0.3, 0.16)
        return np.minimum(np.minimum(np.minimum(d, alt), np.minimum(tab, ma)), np.minimum(me, kn))


WORLD = World()


# =============================================================================
# cameras
# =============================================================================
class Camera:
    """Pinhole camera with a horizontal axis (verticals stay vertical) and lens shift.

    The frame - a world rectangle around `look_at`, half_w by half_h metres at its
    depth - should fill the screen: `fit(W, H)` picks the focal length so the whole
    frame is visible, without ever going wider than `max_fov` degrees across the
    larger screen axis, and shifts the lens so the frame is centred.  On a small
    screen the optional `small` frame (look_at, half_w, half_h, used below this many
    pixel rows) is used instead: closer on the image, which would otherwise be only a
    few cells tall.  `min_f` also takes the small frame when the frame would leave the
    focal length below that many pixels (a narrow screen).  `side` = (x, z, half_h)
    keeps the small frame from cutting into what stands beside it: when its view would
    reach past |x| at depth z, the frame is kept instead, fitted to its width and to a
    band of half_h metres (the vault and the floor are cropped, never the images).
    After fit(), `frame_px` is the chosen frame's half size on screen (pixels)."""

    def __init__(self, eye, look_at, half_w, half_h, max_fov=118.0, small=None, min_f=0.0, side=None):
        self.eye = np.array(eye, F32)
        self.look = np.array(look_at, F32)
        d = self.look - self.eye
        yaw = math.atan2(float(d[0]), float(d[2]))
        self.F = np.array([math.sin(yaw), 0.0, math.cos(yaw)], F32)
        self.R = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], F32)
        self.U = np.array([0.0, 1.0, 0.0], F32)
        self.half_w, self.half_h = half_w, half_h
        self.max_fov = max_fov
        self.small = small
        self.min_f = min_f
        self.side = side
        self.W = self.H = 0
        self.f = self.cx = self.cy = 1.0
        self.frame_px = (1.0, 1.0)

    def _focal(self, W, H, look, hw, hh):
        """Focal length (pixels) that fits a frame on a W x H screen, and the frame's depth."""
        D = float((look - self.eye) @ self.F)
        f = min(W / 2 * D / hw, H / 2 * D / hh)
        return max(f, max(W, H) / 2 / math.tan(math.radians(self.max_fov / 2))), D

    def fit(self, W, H):
        look, hw, hh = self.look, self.half_w, self.half_h
        if self.small is not None and (H < self.small[3] or self._focal(W, H, look, hw, hh)[0] < self.min_f):
            s_look = np.array(self.small[0], F32)
            fs, _ = self._focal(W, H, s_look, self.small[1], self.small[2])
            sx, sz, sh = self.side if self.side is not None else (math.inf, 0.0, hh)
            reach = W / 2 * float((np.array((0.0, 0.0, sz), F32) - self.eye) @ self.F) / fs
            if reach <= sx:
                look, hw, hh = s_look, self.small[1], self.small[2]
            else:
                hh = min(hh, sh)
        f, D = self._focal(W, H, look, hw, hh)
        q = look - self.eye
        self.W, self.H, self.f = W, H, f
        self.cx = W / 2 - f * float(q @ self.R) / D
        self.cy = H / 2 + f * float(q[1]) / D
        self.frame_px = (f * hw / D, f * hh / D)
        return self

    def rays(self):
        """Unit world directions (H*W, 3) through every pixel centre, plus the depth
        factor (forward component of each unit direction)."""
        W, H = self.W, self.H
        ys, xs = np.mgrid[0:H, 0:W].astype(F32)
        u = (xs + 0.5 - self.cx) / self.f
        v = -(ys + 0.5 - self.cy) / self.f
        d = u[..., None] * self.R + v[..., None] * self.U + self.F
        n = np.sqrt(np.sum(d * d, -1))
        d /= n[..., None]
        return d.reshape(-1, 3).astype(F32), (1.0 / n).reshape(-1).astype(F32)

    def project(self, P):
        """World points (..., 3) -> screen x, y (pixels) and camera depth."""
        q = np.asarray(P, F32) - self.eye
        cz = q @ self.F
        cxx = q @ self.R
        cyy = q[..., 1]
        czs = np.maximum(cz, 1e-3)
        return self.cx + self.f * cxx / czs, self.cy - self.f * cyy / czs, cz

    def to_view(self, v):
        """World vectors (..., 3) -> view space (+x right, +y up, +z toward the viewer)."""
        v = np.asarray(v, F32)
        return np.stack([v @ self.R, v[..., 1], -(v @ self.F)], -1).astype(F32)


CAMERAS = {
    # from the nave, looking up the church to the sanctuary, wide enough for the whole votive
    # bank before Our Lady's altar (on a small or narrow screen: the high altar, unless that
    # frame would cut into the side altars - then all three images in a band across the church)
    "sanctuary": dict(eye=(0.0, 1.65, -8.0), look_at=(0.0, 2.65, ZE), half_w=XAC + 1.4, half_h=3.0,
                      small=((0.0, 2.6, 2.4), 2.4, 1.75, 84), min_f=100.0, side=(XA0 + 0.3, ZE, 1.45)),
    # close on the high altar, as if kneeling at the chancel step
    "adoration": dict(eye=(0.0, 1.62, -1.55), look_at=(0.0, 2.5, MON_Z), half_w=1.95, half_h=1.45,
                      small=((0.0, 2.72, MON_Z), 1.45, 1.08, 64)),
    # in the north aisle before Our Lady's altar
    "our lady": dict(eye=(XM + 0.1, 1.55, ZE - 3.6), look_at=(XM, 2.55, ZE), half_w=1.35, half_h=1.72,
                     small=((XM, 2.45, ZE), 1.0, 1.35, 64)),
    # in the south aisle before the Divine Mercy image
    "divine mercy": dict(eye=(XR - 0.45, 1.6, ZE - 4.0), look_at=(XR, 2.55, ZE), half_w=1.3, half_h=1.7,
                         small=((XR, 2.7, ZE), 1.0, 1.55, 64)),
}


def camera(name):
    return Camera(**CAMERAS[name])


# =============================================================================
# G-buffer
# =============================================================================
@dataclass
class GBuffer:
    H: int
    W: int
    pos: np.ndarray        # (H, W, 3) world position
    depth: np.ndarray      # (H, W) camera depth (along the view axis)
    dist: np.ndarray       # (H, W) distance from the eye
    nworld: np.ndarray     # (H, W, 3) world normals
    normal: np.ndarray     # (H, W, 3) view-space normals (kit convention)
    view: np.ndarray       # (H, W, 3) view-space unit vectors toward the eye
    mat: np.ndarray        # (H, W) int8 material
    albedo: np.ndarray
    spec: np.ndarray
    gloss: np.ndarray
    metal: np.ndarray
    emissive: np.ndarray
    ao: np.ndarray
    obj: np.ndarray        # (H, W) int16: votive cup index, -1 elsewhere
    cam: Camera = None


def _trace(dirs, eye, t0, t_lim, eps_k, eps_min, steps):
    """Sphere-trace rays from distances t0 until the SDF drops below the threshold
    eps_k * t (a cone of that half-angle touches a surface) or t reaches t_lim."""
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    ox, oy, oz = (float(v) for v in eye)
    t = t0.astype(F32).copy()
    act = np.flatnonzero(t < t_lim)
    for _ in range(steps):              # rays still going after `steps` are grazing a surface
        if act.size == 0:
            break
        ta = t[act]
        d = WORLD.sdf(ox + dx[act] * ta, oy + dy[act] * ta, oz + dz[act] * ta)
        h = d < np.maximum(ta * eps_k, eps_min)
        ta = ta + d
        t[act] = ta
        act = act[~(h | (ta >= t_lim[act]))]
    return t


def march(dirs, eye, f, W, H, t_max=60.0, block=None):
    """Primary rays.  A coarse pass sends one cone per block x block pixels: where the
    cone first touches a surface is a safe start for every pixel ray in that block
    (cone marching), which skips most of the empty nave.  Rays that reach the floor
    plane stop there exactly."""
    dy = dirs[:, 1]
    with np.errstate(divide="ignore"):
        t_floor = np.where(dy < -1e-4, float(eye[1]) / np.maximum(-dy, 1e-6), t_max).astype(F32)
    t_lim = np.minimum(t_floor, t_max)
    start = np.full(dirs.shape[0], 0.02, F32)
    if block is None:
        block = int(np.clip(round(math.sqrt(W * H) / 40), 4, 8))
    if W >= 3 * block and H >= 3 * block:
        bw, bh = (W + block - 1) // block, (H + block - 1) // block
        cy = np.minimum(np.arange(bh) * block + block // 2, H - 1)
        cx = np.minimum(np.arange(bw) * block + block // 2, W - 1)
        ci = (cy[:, None] * W + cx[None, :]).ravel()
        cone = block * 0.75 / f
        tc = _trace(dirs[ci], eye, np.full(ci.size, 0.02, F32), t_lim[ci], cone, 0.01, 48)
        # every pixel starts where its block's cone stopped, a little short of it
        tb = (tc * 0.92 - 0.02).reshape(bh, bw)
        tb = np.minimum(tb, np.minimum(np.roll(tb, 1, 0), np.roll(tb, -1, 0)))
        tb = np.minimum(tb, np.minimum(np.roll(tb, 1, 1), np.roll(tb, -1, 1)))
        full = np.repeat(np.repeat(tb, block, 0), block, 1)[:H, :W].reshape(-1)
        start = np.maximum(start, full)
    return np.minimum(_trace(dirs, eye, start, t_lim, 0.5 / f, 0.002, 64), t_lim)


def _normals(P, t, f):
    """Tetrahedral SDF gradient, step scaled with the pixel footprint."""
    e = np.maximum(t / f * 0.6, 0.004).astype(F32)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    k = ((1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, 1, 1))
    g = np.zeros_like(P)
    for kx, ky, kz in k:
        d = WORLD.sdf(x + kx * e, y + ky * e, z + kz * e)
        g[:, 0] += kx * d
        g[:, 1] += ky * d
        g[:, 2] += kz * d
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return (g / np.maximum(n, 1e-9)).astype(F32)


def _ao(P, N):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    occ = np.zeros(P.shape[0], F32)
    for h, w in ((0.06, 0.45), (0.2, 0.32), (0.55, 0.23)):
        d = WORLD.sdf(x + N[:, 0] * h, y + N[:, 1] * h, z + N[:, 2] * h)
        occ += w * np.clip((h - d) / h, 0.0, 1.0)
    return np.clip(1.0 - occ, 0.0, 1.0).astype(F32)


def build(cam, W, H):
    """Trace the camera's view at W x H pixels and fill a G-buffer."""
    cam.fit(W, H)
    WORLD.cull(cam)
    dirs, fwd = cam.rays()
    t = march(dirs, cam.eye, cam.f, W, H)
    P = (cam.eye[None, :] + dirs * t[:, None]).astype(F32)
    N = _normals(P, t, cam.f)
    # rays that stopped on the floor plane: exact floor normal
    flo = P[:, 1] < 0.004
    N[flo] = (0.0, 1.0, 0.0)
    _, mat = WORLD.material(P[:, 0], P[:, 1], P[:, 2])
    ao = _ao(P, N)
    albedo, spec, gloss, metal, emis, obj = shade_materials(P, N, mat)
    gb = GBuffer(
        H=H, W=W,
        pos=P.reshape(H, W, 3), depth=(t * fwd).reshape(H, W).astype(F32), dist=t.reshape(H, W).astype(F32),
        nworld=N.reshape(H, W, 3), normal=cam.to_view(N).reshape(H, W, 3),
        view=cam.to_view(-dirs).reshape(H, W, 3),
        mat=mat.reshape(H, W), albedo=albedo.reshape(H, W, 3), spec=spec.reshape(H, W),
        gloss=gloss.reshape(H, W), metal=metal.reshape(H, W), emissive=emis.reshape(H, W, 3),
        ao=ao.reshape(H, W), obj=obj.reshape(H, W), cam=cam)
    return gb


# =============================================================================
# surface detail: masonry joints, floors, stained glass, the starry niche
# =============================================================================
GLASS_COLS = np.array([
    (0.10, 0.20, 0.85),     # deep blue
    (0.85, 0.08, 0.10),     # ruby
    (0.95, 0.70, 0.18),     # gold
    (0.12, 0.55, 0.25),     # green
    (0.80, 0.85, 0.95),     # white / grisaille
    (0.45, 0.15, 0.65),     # violet
], F32)


def glass_colour(u, v, wid):
    """Stained glass at window coordinates u (across, -1..1) and v (metres up from the
    sill): a mosaic field (mostly deep blue, with ruby, gold, green and white), a column
    of plain roundels - ruby, gold, ruby, violet up the light - a ruby and blue border,
    and the lead lines.  Returns (colour (n, 3), lead mask (n,))."""
    cell = 0.15
    i = np.floor(u * 0.6 / cell)
    j = np.floor(v / cell)
    r = _hash2(i + wid * 17.0, j, 3.0)
    pal = np.searchsorted(np.array([0.46, 0.68, 0.78, 0.86, 0.94], F32), r)
    col = GLASS_COLS[pal]
    # medallions: roundels stacked up the light, one colour each
    k = np.floor(v / 0.74)
    mv = v - (k + 0.5) * 0.74
    mr = np.sqrt((u * 0.6) ** 2 + mv * mv)
    med = mr < 0.27
    medc = GLASS_COLS[np.array([1, 2, 1, 5])[((k + wid) % 4).astype(np.int64)]]
    col = np.where(med[:, None], medc, col)
    border = np.abs(u) > 0.78
    bc = np.where((np.floor(v / 0.09) % 2 == 0)[:, None], GLASS_COLS[1], GLASS_COLS[0])
    col = np.where(border[:, None], bc, col)
    # leading: the mosaic grid (not across a roundel: a few dark dots in a small disc
    # read as a face), each roundel's rim and the border's inner edge
    fu = np.abs(u * 0.6 / cell - np.floor(u * 0.6 / cell) - 0.5)
    fv = np.abs(v / cell - np.floor(v / cell) - 0.5)
    lead = ((fu > 0.44) | (fv > 0.44)) & ~med | (np.abs(mr - 0.27) < 0.022) | (np.abs(np.abs(u) - 0.78) < 0.05)
    return col.astype(F32), lead


def glass_tint(u, v, wid):
    """The colour (n, 3) of moonlight after passing through a window: the same glass as
    glass_colour() seen through the blur of the moon's disc - big soft patches of the
    window's main hues, no lead lines."""
    cell = 0.42
    i = np.floor(u * 0.6 / cell)
    j = np.floor(v / cell)
    r = _hash2(i + wid * 17.0, j, 3.0)
    pal = np.searchsorted(np.array([0.5, 0.72, 0.82, 0.9, 0.97], F32), r)
    col = GLASS_COLS[pal] * 0.6 + np.array((0.32, 0.36, 0.62), F32) * 0.4
    return col.astype(F32)


def window_coords(P, N):
    """For points on window glass: (is_glass, u across -1..1, v metres up from the sill,
    window id)."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)
    n = x.shape[0]
    isg = np.zeros(n, bool)
    u = np.zeros(n, F32)
    v = np.zeros(n, F32)
    wid = np.zeros(n, F32)
    zb = _rep(z + BAY / 2, BAY)
    bay = np.floor((z + BAY / 2) / BAY + 0.5)
    nx, nz = np.abs(N[:, 0]), N[:, 2]
    # aisle windows
    m = (ax > XW + 0.3) & (z < ZE - 0.8) & (nx > 0.7)
    isg |= m
    u = np.where(m, zb / 0.62, u)
    v = np.where(m, y - 2.1, v)
    wid = np.where(m, bay * 2 + (x > 0), wid)
    # clerestory
    m = (ax > XN + 0.3) & (ax < XN + 0.5) & (y > 7.55) & (z < -0.6) & (nx > 0.7)
    isg |= m
    u = np.where(m, zb / 0.62, u)
    v = np.where(m, y - 7.6, v)
    wid = np.where(m, 40 + bay * 2 + (x > 0), wid)
    # sanctuary side windows
    zs = z - 2.75 - 2.5 * np.clip(np.floor((z - 2.75) / 2.5 + 0.5), 0, 1)
    m = (ax > XN + 0.3) & (ax < XN + 0.5) & (z > 0.9) & (z < ZA - 0.2) & (y > 2.95) & (nx > 0.7)
    isg |= m
    u = np.where(m, zs / 0.5, u)
    v = np.where(m, y - 3.0, v)
    wid = np.where(m, 80 + (z > 4.0) * 2 + (x > 0), wid)
    # apse
    dz = z - ZA
    w0 = dz - XN
    m = (w0 > 0.3) & (y > 2.95) & (nz < -0.7)
    isg |= m
    u = np.where(m, x / 0.58, u)
    v = np.where(m, y - 3.0, v)
    wid = np.where(m, 90, wid)
    w1 = (ax + dz) * 0.70711 - XN
    m = (w1 > 0.3) & (dz > 0) & (y > 2.95) & ~(w0 > 0.3) & (nz < -0.5) & (nx > 0.5)
    isg |= m
    u = np.where(m, (ax - dz) * 0.70711 / 0.5, u)
    v = np.where(m, y - 3.0, v)
    wid = np.where(m, 91 + (x > 0), wid)
    return isg, u, v, wid


def shade_materials(P, N, mat):
    """Per-pixel albedo, specular, gloss, metalness, emission and object ids."""
    n = P.shape[0]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    alb = np.zeros((n, 3), F32)
    spec = np.zeros(n, F32)
    gloss = np.full(n, 16.0, F32)
    metal = np.zeros(n, F32)
    emis = np.zeros((n, 3), F32)
    obj = np.full(n, -1, np.int16)
    for mid, (a, s, g, mt) in MATS.items():
        m = mat == mid
        if m.any():
            alb[m] = a
            spec[m] = s
            gloss[m] = g
            metal[m] = mt
    # ---- floors: flagstones in the nave and aisles, marble chequer in the sanctuary
    fl = (mat == M_STONE) & (N[:, 1] > 0.9) & (y < 0.02)
    if fl.any():
        fx, fz = x[fl] / 0.92, z[fl] / 0.92
        row = np.floor(fz)
        fx = fx + 0.5 * (row % 2)
        ix = np.floor(fx)
        jx, jz = np.abs(fx - ix - 0.5), np.abs(fz - row - 0.5)
        var = 0.82 + 0.3 * _hash2(ix, row, 1.0)
        joint = (np.maximum(jx, jz) > 0.485)
        a = np.array(MATS[M_FLOOR][0], F32)[None, :] * var[:, None]
        a[joint] *= 0.45
        alb[fl] = a
        spec[fl] = np.where(joint, 0.05, MATS[M_FLOOR][1])
        gloss[fl] = MATS[M_FLOOR][2]
        mat[fl] = M_FLOOR
    top = (mat == M_MARBLE) & (N[:, 1] > 0.9) & (np.abs(y - YS) < 0.02) & (np.abs(x) < XN) & (z > -0.4)
    if top.any():
        cu = (x[top] + z[top]) / 0.78
        cv = (x[top] - z[top]) / 0.78
        par = (np.floor(cu) + np.floor(cv)) % 2
        a = np.where(par[:, None] > 0, np.array((0.62, 0.60, 0.56), F32), np.array((0.07, 0.065, 0.07), F32))
        alb[top] = a
        spec[top] = MATS[M_CHECK][1]
        gloss[top] = MATS[M_CHECK][2]
        mat[top] = M_CHECK
    # ---- masonry: ashlar courses with staggered joints, a little colour variation
    st = mat == M_STONE
    if st.any():
        ns = N[st]
        hcoord = np.where(np.abs(ns[:, 0]) > np.abs(ns[:, 2]), z[st], x[st])
        cy = y[st] / 0.36
        row = np.floor(cy)
        hc = hcoord / 0.72 + 0.5 * (row % 2)
        col = np.floor(hc)
        var = 0.88 + 0.2 * _hash2(col, row, 2.0)
        joint = (np.abs(cy - row - 0.5) > 0.47) | (np.abs(hc - col - 0.5) > 0.485)
        k = var * np.where(joint, 0.7, 1.0)
        alb[st] = alb[st] * k[:, None]
    # ---- the apse vault: deep blue sown with gold stars
    va = (mat == M_STONE) & (z > 0.6) & (y > YV + 0.3) & (N[:, 1] < -0.2)
    if va.any():
        alb[va] = MATS[M_VAULT][0]
        mat[va] = M_VAULT
        _stars(alb, spec, gloss, metal, va, x, z, 0.42, 0.06)
    # ---- Our Lady's niche: blue, sown with small gold stars
    nm = (mat == M_STONE) & (z > ZE - 0.02) & (y > NICHE_Y0 - 0.01) & (np.abs(x - XM) < NICHE_HW + 0.01)
    nm[nm] = _arch(x[nm] - XM, y[nm], NICHE_HW + 0.01, NICHE_SPRING, 1.0) < 0.01     # not the wall above it
    if nm.any():
        alb[nm] = MATS[M_NICHE][0]
        spec[nm] = MATS[M_NICHE][1]
        mat[nm] = M_NICHE
        _stars(alb, spec, gloss, metal, nm, x - XM, y, 0.19, 0.03)
    # ---- frontals: gold orphreys and a central motif
    fr = mat == M_FRONTAL
    if fr.any():
        u, v = np.abs(x[fr]), y[fr] - 1.15
        gold = (u > 1.08) | (np.abs(v) > 0.33) | (np.abs(u - 0.0) < 0.035) & (np.abs(v) < 0.24) \
            | (np.abs(v - 0.06) < 0.035) & (u < 0.16)
        ring = np.abs(np.sqrt(u * u + (v - 0.02) ** 2) - 0.25) < 0.025
        gold |= ring
        _gild(alb, spec, gloss, metal, np.flatnonzero(fr)[gold])
    bl = mat == M_BLUE
    if bl.any():
        u, v = np.abs(x[bl] - XM), y[bl] - 0.47
        gold = (u > 0.54) | (np.abs(v) > 0.24)
        # Marian monogram: an M (two uprights and a V) over a small cross bar
        mm = (np.abs(u - 0.11) < 0.022) & (np.abs(v) < 0.12) | \
             (np.abs(u - (0.11 - (0.12 - v) * 0.45)) < 0.022) & (v > -0.02) & (v < 0.12) & (u < 0.11)
        _gild(alb, spec, gloss, metal, np.flatnonzero(bl)[gold | mm])
    # ---- stained glass: emissive, coloured; lead lines dark
    gl = mat == M_STONE
    isg, u, v, wid = window_coords(P[gl], N[gl])
    if isg.any():
        idx = np.flatnonzero(gl)[isg]
        gris = np.where(wid[isg] < 40, 0.55, 0.0)               # the aisle lights are mostly grisaille
        c, lead = glass_colour(u[isg], v[isg], wid[isg])
        c = c * (1 - gris[:, None]) + (GLASS_COLS[4] * 0.8 * (0.85 + 0.3 * _hash2(np.floor(u[isg] * 4), np.floor(v[isg] * 8), 5.0))[:, None]) * gris[:, None]
        mat[idx] = M_GLASS
        alb[idx] = MATS[M_GLASS][0]
        spec[idx] = MATS[M_GLASS][1]
        emis[idx] = c * np.where(lead, 0.08, 1.0)[:, None]
        obj[idx] = (1000 + wid[isg]).astype(np.int16)
    # ---- votive cups: remember which cup each pixel shows (they flicker one by one)
    vm = mat == M_VOTIVE
    if vm.any():
        cups, _, _ = votive_cups()
        tl = mercy_tealights()
        allc = np.concatenate([cups, tl])
        pv = P[vm]
        d2 = ((pv[:, None, :] - allc[None, :, :]) ** 2).sum(-1)
        obj[vm] = np.argmin(d2, axis=1).astype(np.int16)
    return alb, spec, gloss, metal, emis, obj


def _gild(alb, spec, gloss, metal, idx):
    alb[idx] = MATS[M_GOLD][0]
    spec[idx] = 0.8
    gloss[idx] = 40.0
    metal[idx] = 1.0


def _stars(alb, spec, gloss, metal, m, h, w, cell, rad):
    """Gold stars painted on a blue ground in a regular diamond lattice, as on a
    Gothic starry vault (h, w: the surface's own 2D coordinates in metres)."""
    u, v = h[m] / cell, w[m] / cell
    j = np.floor(v)
    u = u + 0.5 * (j % 2)                             # every other row shifted: a diamond lattice
    i = np.floor(u)
    du, dv = (u - i - 0.5) * cell, (v - j - 0.5) * cell
    r = np.sqrt(du * du + dv * dv)
    ang = np.arctan2(dv, du) + np.pi / 2
    star = r < rad * (0.5 + 0.5 * np.abs(np.cos(2.5 * ang)))      # five points
    _gild(alb, spec, gloss, metal, np.flatnonzero(m)[star])
