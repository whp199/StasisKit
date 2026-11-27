"""OUR LADY OF THE ROSARY - the painted statue on the Lady altar.

Our Lady in the Lourdes / Immaculate Conception tradition: a white veil over
her gently bowed head, a blue mantle with a gilt border open over an ivory
gown and a gilt sash, hands joined in prayer at the breast with a long rosary
of pale beads hanging from them, a golden rose on each bare foot, a crown of
twelve stars (Revelation 12:1), standing on a carved stone plinth.

The statue is a relief of closed-form front surfaces.  Every part gives, per
sample, the depth of its nearest surface toward the viewer: rotated
ellipsoids (head, jaw, hands, toes), ray-capsule hits (arms, cuffs, neck),
elliptic cloth shells whose radius carries hanging folds (mantle, gown,
veil), faceted stars and sphere caps (beads).  The face is two blended
ellipsoids with soft gaussian features (brow, lowered lids, nose, lips) in
the bowed head's own frame.  The parts are z-buffered on a supersampled grid;
normals come from the depth gradients, occlusion from depth cavities (where
the blurred depth stands above a sample it sits in a hollow), and everything
is box-filtered down to the sprite's G-buffer.  Small sprites keep what
identifies her - white veil, blue mantle, pale joined hands, gold stars - by
giving the stars a minimum size and a little glow and keeping a band of blue
under the hands; painted facial features and the plinth's monogram only
appear where there are pixels to carry them.

The rosary has the real structure (crucifix, five pendant beads, centre
medal, five decades separated by four Our Father beads), laid out by arc
length along a loop held in the joined hands, and its bead centres are
exported in prayer order so the scene can glint the bead being prayed.
"""

import math

import numpy as np

from . import kit

F32 = np.float32
ZBG = F32(-0.8)                 # depth of empty space (behind everything)

# ---- layout, in head heights: y down from the crown, x from the axis ----------
TOP, BOT = -0.50, 8.22          # sprite extent: top of the star crown .. plinth foot
HALF_W = 1.43                   # sprite half width (plinth base)
SOLE = 7.40                     # soles of the feet = top of the plinth
PL_BOT = 8.21                   # bottom of the plinth

PIVOT = np.array([0.0, 0.98, -0.06])   # the head bows about the top of the neck
BOW = math.radians(15.0)               # forward bow
ROLL = math.radians(-5.0)              # lean of the head toward her right shoulder

HALO_C = (0.0, 0.47)            # star crown: centre, radius, star size, arc span
HALO_R = 0.80
STAR_R = 0.13
STAR_SPAN = math.radians(128.0)

# ---- palette: unlit albedo of the painted statue ------------------------------
VEIL = kit.rgb((0.82, 0.85, 0.90))
GOWN = kit.rgb((0.58, 0.54, 0.46))
MANTLE = kit.rgb((0.11, 0.27, 0.80))
SKIN = kit.rgb((0.86, 0.63, 0.52))
LIPS = kit.rgb((0.74, 0.38, 0.36))
LASH = kit.rgb((0.30, 0.19, 0.16))
BROW = kit.rgb((0.48, 0.34, 0.26))
GOLD = kit.rgb((1.00, 0.76, 0.34))
STONE = kit.rgb((0.30, 0.28, 0.26))
BEAD = kit.rgb((0.96, 0.95, 0.93))

#           spec  gloss metal
M_VEIL = (0.16, 12.0, 0.0)
M_GOWN = (0.12, 10.0, 0.0)
M_MANTLE = (0.10, 9.0, 0.0)
M_SKIN = (0.14, 16.0, 0.0)
M_GOLD = (0.90, 22.0, 0.55)
M_STONE = (0.05, 6.0, 0.0)
M_BEAD = (0.65, 50.0, 0.0)


# =============================================================================
# small math helpers
# =============================================================================
def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _interp(y, ys, vs):
    return np.interp(y, ys, vs).astype(F32)


def _rot(bow=0.0, roll=0.0):
    """Local -> world rotation (y down, z toward the viewer): `bow` tips the top
    toward the viewer, `roll` leans the top toward +x."""
    cb, sb = math.cos(bow), math.sin(bow)
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cb, sb], [0, -sb, cb]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx


def _normals(z, step, gmax=3.5):
    """View-space normals of a depth field (units) sampled every `step` units.
    Slopes are clamped so silhouettes and depth steps roll over softly."""
    if z.shape[0] < 2 or z.shape[1] < 2:
        n = np.zeros(z.shape + (3,), F32)
        n[..., 2] = 1.0
        return n
    gy, gx = np.gradient(z.astype(F32), step)
    g = np.hypot(gx, gy)
    k = np.minimum(1.0, gmax / np.maximum(g, 1e-6))
    n = np.stack([-gx * k, gy * k, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.astype(F32)


def _ellipsoid(X, Y, c, r, R=None, zf=2.0):
    """Front surface of an ellipsoid (centre c, radii r, local->world rotation R)
    seen along -z.  Returns (mask, z, (lx, ly, lz) local hit coordinates).
    Outside the silhouette z continues as the depth of closest approach."""
    Rt = np.eye(3) if R is None else np.asarray(R).T
    dx, dy, dz = X - F32(c[0]), Y - F32(c[1]), F32(zf - c[2])
    o = [(Rt[i, 0] * dx + Rt[i, 1] * dy + Rt[i, 2] * dz) / r[i] for i in range(3)]
    v = [-Rt[i, 2] / r[i] for i in range(3)]
    A = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    B = o[0] * v[0] + o[1] * v[1] + o[2] * v[2]
    C = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - 1.0
    disc = B * B - A * C
    t = (-B - np.sqrt(np.maximum(disc, 0.0))) / A
    loc = tuple(((o[i] + t * v[i]) * r[i]).astype(F32) for i in range(3))
    return disc > 0, (zf - t).astype(F32), loc


def _capsule(X, Y, a, b, r, zf=2.0):
    """Front hit of a capsule (segment a-b, radius r) along -z (Inigo Quilez's
    ray-capsule intersection with rd = (0, 0, -1)).  Returns (mask, z)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ba = b - a
    oax, oay, oaz = X - F32(a[0]), Y - F32(a[1]), F32(zf - a[2])
    baba = float(ba @ ba)
    bard = -float(ba[2])
    baoa = F32(ba[0]) * oax + F32(ba[1]) * oay + F32(ba[2]) * oaz
    rdoa = -oaz
    oaoa = oax * oax + oay * oay + oaz * oaz
    A = max(baba - bard * bard, 1e-9)
    Bq = baba * rdoa - baoa * bard
    Cq = baba * oaoa - baoa * baoa - r * r * baba
    hq = Bq * Bq - A * Cq
    t = (-Bq - np.sqrt(np.maximum(hq, 0.0))) / A
    yq = baoa + t * bard
    body = (hq >= 0) & (yq > 0) & (yq < baba)
    use_a = yq <= 0
    ocx = np.where(use_a, oax, X - F32(b[0]))
    ocy = np.where(use_a, oay, Y - F32(b[1]))
    ocz = np.where(use_a, oaz, F32(zf - b[2]))
    b2 = -ocz
    h2 = b2 * b2 - (ocx * ocx + ocy * ocy + ocz * ocz - r * r)
    t2 = -b2 - np.sqrt(np.maximum(h2, 0.0))
    tt = np.where(body, t, t2)
    mask = (hq >= 0) & (body | (h2 > 0))
    z = np.where(mask, zf - tt, ZBG)
    return mask, z.astype(F32)


def _sd_star5(px, py, r, rf):
    """Signed distance to a five-pointed star pointing up (+y up), outer radius r
    (Inigo Quilez's sdStar5; rf sets how slender the points are)."""
    k1x, k1y = 0.809016994375, -0.587785252292
    k2x, k2y = -k1x, k1y
    px = np.abs(px)
    d = np.maximum(k1x * px + k1y * py, 0.0)
    px, py = px - 2 * d * k1x, py - 2 * d * k1y
    d = np.maximum(k2x * px + k2y * py, 0.0)
    px, py = px - 2 * d * k2x, py - 2 * d * k2y
    px = np.abs(px)
    py = py - r
    bax, bay = rf * -k1y, rf * k1x - 1.0
    h = np.clip((px * bax + py * bay) / (bax * bax + bay * bay), 0.0, r)
    return np.hypot(px - bax * h, py - bay * h) * np.sign(py * bax - px * bay)


def _down(a, ss):
    if ss == 1:
        return a
    H, W = a.shape[0] // ss, a.shape[1] // ss
    return a.reshape((H, ss, W, ss) + a.shape[2:]).mean(axis=(1, 3))


# =============================================================================
# the supersampled relief buffer
# =============================================================================
class _Relief:
    """Z-buffered G-buffer on a supersampled grid, addressed in statue units."""

    def __init__(self, h, w, s, ss):
        self.h, self.w, self.s, self.ss = h, w, s, ss
        self.step = 1.0 / (s * ss)
        self.ux0 = -0.5 * w / s            # unit x of the sprite's left edge
        self.uy0 = TOP - 1.0 / s           # unit y of the sprite's top edge (1 px margin)
        H, W = h * ss, w * ss
        self.z = np.full((H, W), ZBG, F32)
        self.alb = np.zeros((H, W, 3), F32)
        self.nrm = np.zeros((H, W, 3), F32)
        self.nrm[..., 2] = 1.0
        self.spec = np.zeros((H, W), F32)
        self.gloss = np.full((H, W), 16.0, F32)
        self.metal = np.zeros((H, W), F32)
        self.emi = np.zeros((H, W, 3), F32)
        self.aom = np.ones((H, W), F32)
        self.cov = np.zeros((H, W), bool)
        self.body = np.zeros((H, W), bool)

    def px(self, x, y):
        """Statue units -> sprite pixel coordinates."""
        return (np.asarray(x) - self.ux0) * self.s, (np.asarray(y) - self.uy0) * self.s

    def region(self, x0, x1, y0, y1):
        """(slices, X, Y) of the samples covering a box in statue units."""
        st = self.step
        j0 = max(0, int(math.floor((x0 - self.ux0) / st)))
        j1 = min(self.cov.shape[1], int(math.ceil((x1 - self.ux0) / st)) + 1)
        i0 = max(0, int(math.floor((y0 - self.uy0) / st)))
        i1 = min(self.cov.shape[0], int(math.ceil((y1 - self.uy0) / st)) + 1)
        if j1 <= j0 or i1 <= i0:
            return None
        xs = (self.ux0 + (np.arange(j0, j1) + 0.5) * st).astype(F32)
        ys = (self.uy0 + (np.arange(i0, i1) + 0.5) * st).astype(F32)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        return (slice(i0, i1), slice(j0, j1)), X, Y

    def put(self, sl, mask, z, alb, mat, nrm=None, emi=None, ao=None, ztest=True, body=True):
        """Write one part where it covers and (with ztest) lies in front."""
        if nrm is None:
            nrm = _normals(z, self.step)
        zb = self.z[sl]
        m = mask & (z > zb) if ztest else mask
        if not m.any():
            return
        shp = m.shape
        zb[m] = z[m]
        self.alb[sl][m] = np.broadcast_to(np.asarray(alb, F32), shp + (3,))[m]
        self.nrm[sl][m] = nrm[m]
        sp, gl, mt = mat
        self.spec[sl][m] = np.broadcast_to(np.asarray(sp, F32), shp)[m]
        self.gloss[sl][m] = np.broadcast_to(np.asarray(gl, F32), shp)[m]
        self.metal[sl][m] = np.broadcast_to(np.asarray(mt, F32), shp)[m]
        self.emi[sl][m] = 0.0 if emi is None else np.broadcast_to(np.asarray(emi, F32), shp + (3,))[m]
        self.aom[sl][m] = 1.0 if ao is None else np.broadcast_to(np.asarray(ao, F32), shp)[m]
        self.cov[sl][m] = True
        self.body[sl][m] = body

    def resolve(self):
        """Cavity occlusion, then box-filter everything down to a Sprite."""
        ss, s = self.ss, self.s
        z = np.where(self.body, self.z, F32(-0.6))
        ao = np.ones_like(z)
        for r_u, k, lo in ((0.06, 4.0, 0.45), (0.20, 3.0, 0.4)):
            r = int(round(r_u * s * ss))
            if r >= 1:
                ao *= np.clip(1.0 - k * (kit.blur(z, r) - z), lo, 1.0)
        ao = np.where(self.body, ao, 1.0) * self.aom
        cov = self.cov.astype(F32)
        a = _down(cov, ss)
        inv = 1.0 / np.maximum(a, 1e-6)

        def avg(v):
            if v.ndim == 3:
                return (_down(v * cov[..., None], ss) * inv[..., None]).astype(F32)
            return (_down(v * cov, ss) * inv).astype(F32)

        spr = kit.Sprite.blank(self.h, self.w)
        n = _down(self.nrm * cov[..., None], ss)
        nl = np.linalg.norm(n, axis=-1, keepdims=True)
        flat = np.zeros_like(n)
        flat[..., 2] = 1.0
        spr.normal = np.where(nl > 1e-6, n / np.maximum(nl, 1e-6), flat).astype(F32)
        spr.alpha = a.astype(F32)
        spr.albedo = avg(self.alb)
        spr.spec = avg(self.spec)
        spr.gloss = np.where(a > 0, avg(self.gloss), 16.0).astype(F32)
        spr.metal = avg(self.metal)
        spr.emissive = avg(self.emi)
        spr.ao = np.where(a > 0, avg(ao), 1.0).astype(F32)
        return spr


# =============================================================================
# cloth shells
# =============================================================================
# mantle: half width W, depth D (by height), front opening half width O
M_Y = [1.12, 1.20, 1.32, 1.50, 1.80, 2.30, 2.80, 3.20, 4.50, 6.00, 7.40]
M_W = [0.00, 0.30, 0.54, 0.73, 0.83, 0.88, 0.95, 0.98, 1.02, 1.07, 1.13]
M_D = [0.20, 0.25, 0.30, 0.34, 0.37, 0.42, 0.50, 0.52, 0.52, 0.52, 0.53]
O_Y = [1.12, 2.68, 2.92, 3.25, 4.00, 5.00, 7.40]    # closed behind the hands, opening below
O_X = [0.00, 0.00, 0.10, 0.155, 0.20, 0.245, 0.32]
# top edge of the mantle where it lies over the veil (collar), by |x|
C_X = [0.00, 0.10, 0.25, 0.45, 0.60, 1.30]
C_Y = [1.36, 1.345, 1.31, 1.29, 1.30, 1.30]
# gown under it
G_Y = [1.22, 1.45, 2.00, 2.50, 2.95, 3.60, 5.00, 6.50, 7.40]
G_W = [0.16, 0.42, 0.52, 0.50, 0.46, 0.57, 0.60, 0.63, 0.68]
G_D = [0.20, 0.28, 0.34, 0.35, 0.35, 0.40, 0.40, 0.42, 0.47]
SASH_Y = (2.90, 3.00)


class _Folds:
    """Seeded phases of the hanging folds (small variations only)."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.m = (rng.uniform(-0.6, 0.6, 3) + np.array([0.3, 1.9, 4.1])).tolist()
        self.g = (rng.uniform(-0.6, 0.6, 3) + np.array([1.2, 0.4, 2.6])).tolist()
        self.v = (rng.uniform(-0.5, 0.5, 2) + np.array([0.7, 2.2])).tolist()

    def mantle(self, phi, y):
        a = 0.012 + 0.085 * _ss(2.6, 5.0, y)
        p = self.m
        return a * (0.55 * np.cos(7.0 * phi + p[0] + 0.25 * y) + 0.30 * np.cos(12.0 * phi + p[1] - 0.2 * y)
                    + 0.15 * np.cos(21.0 * phi + p[2] + 0.1 * y))

    def gown(self, phi, y):
        a = 0.008 + 0.040 * _ss(2.9, 6.0, y)
        p = self.g
        return a * (0.65 * np.cos(15.0 * phi + p[0] + 0.3 * y) + 0.35 * np.cos(26.0 * phi + p[1]))

    def veil(self, phi, y):
        a = 0.004 + 0.018 * _ss(0.7, 1.4, y)
        p = self.v
        return a * (0.7 * np.cos(7.0 * phi + p[0] + 0.8 * y) + 0.3 * np.cos(12.0 * phi + p[1]))


def _shell(X, Y, W, D, z0, fold):
    """Elliptic cloth shell x = (W+f) sin(phi), z = z0 + (D+f) cos(phi), where the
    fold offset f(phi, y) moves the cloth along its radius.  phi is found from x
    by a short fixed-point iteration.  Returns (mask, z, phi)."""
    Wd = np.maximum(W, 1e-3)
    phi = np.arcsin(np.clip(X / Wd, -1.0, 1.0))
    for _ in range(2):
        phi = np.arcsin(np.clip(X / np.maximum(Wd + fold(phi, Y), 1e-3), -1.0, 1.0))
    f = fold(phi, Y)
    q = 1.0 - (X / np.maximum(Wd + f, 1e-3)) ** 2
    z = z0 + (D + f) * np.sqrt(np.maximum(q, 0.0))
    return (q > 0) & (W > 1e-3), z.astype(F32), phi


def _gown_z(X, Y, folds):
    W = _interp(Y, G_Y, G_W)
    D = _interp(Y, G_Y, G_D)
    m, z, _ = _shell(X, Y, W, D, -0.02, folds.gown)
    return m, z


def _mantle(rl, folds):
    """Blue mantle: one elliptic shell from the shoulders to the hem, open in front,
    over the arms (upper arm and forearm capsules blended in), its collar lying
    over the veil.  Gilt border along the collar, the front edges and the hem."""
    reg = rl.region(-1.3, 1.3, 1.05, SOLE)
    if reg is None:
        return
    sl, X, Y = reg
    ax = np.abs(X)
    W = _interp(Y, M_Y, M_W)
    D = _interp(Y, M_Y, M_D)
    m, z, phi = _shell(X, Y, W, D, -0.02, folds.mantle)
    # the opening starts below the forearms; on small sprites a little lower, so the
    # joined hands keep a band of blue under them
    y_open = 2.68 + 0.32 * (1.0 - float(_ss(8.0, 14.0, rl.s)))
    o = _interp(Y, O_Y, O_X) * _ss(y_open, y_open + 0.3, Y)
    e = ax - o                               # distance in from the front edge
    col = _interp(ax, C_X, C_Y)
    ec = Y - col                             # distance down from the collar
    hem = SOLE - 0.014 - 0.012 * np.cos(7.0 * phi + 1.3) - 0.006 * np.cos(19.0 * phi)
    m &= (e > 0) & (ec > 0) & (Y < hem)
    z = z - 0.035 * (1.0 - np.clip(e / 0.09, 0.0, 1.0)) ** 2 * _ss(0.0, 0.04, o)   # front edge rolls back
    # the collar lies on the veil: lift it over the veil's surface, with a thick lip
    mv, zv = _veil_z(X, Y, folds)
    lift = np.where(mv, zv + 0.03 + 0.02 * np.exp(-(ec / 0.05) ** 2), ZBG)
    z = np.maximum(z, lift * _ss(0.30, 0.0, ec) + z * (1.0 - _ss(0.30, 0.0, ec)))
    zs = np.where(m, z, ZBG)
    zm = zs
    for sx in (-1.0, 1.0):                  # arms under the cloth: upper arm, forearm
        mu, zu = _capsule(X, Y, (sx * 0.64, 1.45, -0.03), (sx * 0.80, 2.78, 0.02), 0.18)
        mf, zf = _capsule(X, Y, (sx * 0.80, 2.84, 0.06), (sx * 0.27, 2.46, 0.36), 0.165)
        zm = kit.smax(zm, np.where(mu & (Y > col), zu, ZBG), 0.08)
        zm = kit.smax(zm, np.where(mf, zf, ZBG), 0.05)
    mask = zm > ZBG + 0.1
    # gilt border where the shell is on top
    bw = 0.045
    border = m & (zs >= zm - 0.01) & (((e < bw) & (o > 0.02)) | (ec < bw) | (hem - Y < bw))
    bs = border.astype(F32)
    alb = MANTLE * (1 - bs[..., None]) + GOLD * bs[..., None]
    mat = (M_MANTLE[0] + (M_GOLD[0] - M_MANTLE[0]) * bs, M_MANTLE[1] + (M_GOLD[1] - M_MANTLE[1]) * bs,
           M_GOLD[2] * bs)
    rl.put(sl, mask, zm, alb, mat, ztest=False)


def _gown(rl, folds):
    reg = rl.region(-0.75, 0.75, 1.15, SOLE)
    if reg is None:
        return
    sl, X, Y = reg
    m, z = _gown_z(X, Y, folds)
    neck = 1.26 + 0.12 * (X / 0.16) ** 2
    hem = SOLE - 0.13 * (np.exp(-((X - 0.16) / 0.11) ** 2) + np.exp(-((X + 0.16) / 0.11) ** 2))
    m &= (Y > neck) & (Y < hem)
    rl.put(sl, m, z, GOWN, M_GOWN)
    # gilt sash at the waist (where the mantle opens)
    reg = rl.region(-0.5, 0.5, SASH_Y[0] - 0.05, SASH_Y[1] + 0.05)
    if reg is None:
        return
    sl, X, Y = reg
    m, z = _gown_z(X, Y, folds)
    yc, hh = 0.5 * (SASH_Y[0] + SASH_Y[1]), 0.5 * (SASH_Y[1] - SASH_Y[0])
    t = np.clip(1.0 - ((Y - yc) / hh) ** 2, 0.0, 1.0)
    m &= t > 0
    zz = z + 0.012 + 0.014 * np.sqrt(t)
    rl.put(sl, m, zz, GOLD, M_GOLD)


# =============================================================================
# head, veil, hands
# =============================================================================
def _head(rl):
    """Face (two blended ellipsoids + soft sculpted features) and neck."""
    reg = rl.region(-0.45, 0.45, 0.1, 1.45)
    if reg is None:
        return None
    sl, X, Y = reg
    R = _rot(BOW, ROLL)
    P = PIVOT

    def world(c):
        return P + R @ (np.asarray(c, np.float64) - P)

    m1, z1, _ = _ellipsoid(X, Y, world((0.0, 0.50, -0.03)), (0.30, 0.47, 0.37), R)
    m2, z2, _ = _ellipsoid(X, Y, world((0.0, 0.78, 0.04)), (0.20, 0.22, 0.27), R)
    z = kit.smax(np.where(m1, z1, ZBG), np.where(m2, z2, ZBG), 0.08)
    m = m1 | m2
    # local (upright) coordinates of the surface point, for the features
    dx, dy, dz = X - F32(P[0]), Y - F32(P[1]), z - F32(P[2])
    Rt = R.T
    qx = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz + P[0]
    qy = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz + P[1]
    qz = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz + P[2]
    ax = np.abs(qx)
    front = np.clip(qz / 0.15, 0.0, 1.0)
    g = np.exp
    d = (0.045 * g(-(qx / 0.04) ** 2) * _ss(0.50, 0.73, qy) * (1.0 - _ss(0.735, 0.775, qy))   # nose ridge
         + 0.020 * g(-(qx / 0.06) ** 2 - ((qy - 0.745) / 0.03) ** 2)                            # nose tip
         + 0.018 * g(-((ax - 0.10) / 0.09) ** 2 - ((qy - 0.47) / 0.03) ** 2)                     # brow ridge
         - 0.030 * g(-((ax - 0.105) / 0.065) ** 2 - ((qy - 0.535) / 0.04) ** 2)                  # eye socket
         + 0.016 * g(-((ax - 0.105) / 0.05) ** 2 - ((qy - 0.545) / 0.026) ** 2)                  # lowered lid
         + 0.012 * g(-(qx / 0.055) ** 2 - ((qy - 0.815) / 0.018) ** 2)                           # upper lip
         + 0.014 * g(-(qx / 0.05) ** 2 - ((qy - 0.85) / 0.022) ** 2)                             # lower lip
         - 0.010 * g(-(qx / 0.05) ** 2 - ((qy - 0.832) / 0.008) ** 2)                            # mouth line
         + 0.012 * g(-(qx / 0.08) ** 2 - ((qy - 0.94) / 0.04) ** 2)                              # chin
         + 0.015 * g(-((ax - 0.14) / 0.08) ** 2 - ((qy - 0.67) / 0.07) ** 2))                    # cheeks
    z = np.where(m, z + d * front, z).astype(F32)
    # neck (not bowed)
    mn, zn = _capsule(X, Y, (0.0, 0.90, -0.07), (0.0, 1.40, -0.05), 0.115)
    zz = kit.smax(np.where(m, z, ZBG), np.where(mn, zn, ZBG), 0.05)
    mm = m | mn
    # painted features, only where the face is large enough to carry them
    fv = float(np.clip((rl.s - 14.0) / 12.0, 0.0, 1.0))
    alb = np.broadcast_to(SKIN, X.shape + (3,)).copy()
    if fv > 0:
        lips = g(-(qx / 0.05) ** 2 - ((qy - 0.835) / 0.022) ** 2)
        ylash = 0.556 + 2.2 * (ax - 0.105) ** 2
        lash = g(-((qy - ylash) / 0.007) ** 2) * _ss(0.055, 0.035, np.abs(ax - 0.105))
        ybrow = 0.468 - 2.0 * (ax - 0.11) ** 2
        brow = g(-((qy - ybrow) / 0.008) ** 2) * _ss(0.07, 0.04, np.abs(ax - 0.11))
        cheek = g(-((ax - 0.15) / 0.07) ** 2 - ((qy - 0.69) / 0.06) ** 2)
        for wgt, col in ((lips * 0.8, LIPS), (lash * 0.7, LASH), (brow * 0.45, BROW), (cheek * 0.18, LIPS)):
            k = (wgt * fv * front)[..., None]
            alb = alb * (1 - k) + col * k
    # the neck sits back in the chin's shadow (baked: the scene does not self-shadow sprites)
    neck = ~(m & (zz > np.where(mn, zn, ZBG) + 0.01))
    shade = np.where(neck, 0.2 + 0.5 * _ss(1.0, 1.4, Y), 1.0)
    alb = np.where(neck[..., None], alb * (0.55 + 0.35 * _ss(1.0, 1.4, Y))[..., None], alb)
    rl.put(sl, mm, zz, alb, M_SKIN, ao=shade)
    fx, fy = world((0.0, 0.62, 0.30))[:2]
    return fx, fy


FACE_O = (0.715, 0.25, 0.355)               # face opening in the veil: centre y, half width, half height


def _veil_z(X, Y, folds):
    """Depth of the white veil: a hood over the head (leaning with it), falling
    beside the face and neck to the shoulders, where it goes under the mantle."""
    lean = ROLL * (1.0 - _ss(0.8, 1.3, Y))  # leans with the head on top, upright below
    c, s_ = np.cos(lean), np.sin(lean)
    dx, dy = X - F32(PIVOT[0]), Y - F32(PIVOT[1])
    xr = c * dx + s_ * dy + PIVOT[0]
    yr = -s_ * dx + c * dy + PIVOT[1]
    top = 0.47
    cap = np.sqrt(np.clip(1.0 - ((yr - top) / 0.56) ** 2, 0.0, 1.0))
    wv = np.where(yr < top, 0.43 * cap, 0.43 + 0.15 * _ss(0.9, 1.45, yr))
    dv = np.where(yr < top, 0.40 * cap, 0.40 - 0.05 * _ss(0.9, 1.5, yr))
    m, z, _ = _shell(xr, yr, wv, dv, 0.02, folds.veil)
    fy, fw, fh = FACE_O
    ef = np.sqrt((xr / fw) ** 2 + ((yr - fy) / fh) ** 2) - 1.0
    neck = (np.abs(xr) < 0.125) & (yr > fy)
    m &= (ef > 0) & ~neck & (yr < 1.65)
    z = z + 0.025 * np.exp(-((ef - 0.07) / 0.07) ** 2)   # rolled hem around the face
    return m, z.astype(F32)


def _veil(rl, folds):
    reg = rl.region(-0.7, 0.7, -0.12, 1.66)
    if reg is None:
        return
    sl, X, Y = reg
    m, z = _veil_z(X, Y, folds)
    rl.put(sl, m, z, VEIL, M_VEIL, ztest=False)


def _hands(rl):
    """Hands joined palm to palm, fingertips up and leaning a little forward,
    thumbs side by side in front, wrists into ivory cuffs."""
    reg = rl.region(-0.42, 0.42, 1.45, 2.55)
    if reg is None:
        return None
    sl, X, Y = reg
    zc = np.full(X.shape, ZBG, F32)
    for sx in (-1.0, 1.0):                  # sleeve cuffs at the wrists
        mc, zcf = _capsule(X, Y, (sx * 0.13, 2.36, 0.46), (sx * 0.28, 2.47, 0.38), 0.09)
        zc = np.maximum(zc, np.where(mc, zcf, ZBG))
    rl.put(sl, zc > ZBG + 0.1, zc, GOWN, M_GOWN)
    zh = np.full(X.shape, ZBG, F32)
    for sx in (-1.0, 1.0):
        R = _rot(math.radians(20.0), -sx * math.radians(6.0))
        mp, zp, _ = _ellipsoid(X, Y, (sx * 0.075, 2.02, 0.55), (0.08, 0.28, 0.14), R)
        mf, zf, _ = _ellipsoid(X, Y, (sx * 0.05, 1.80, 0.60), (0.055, 0.18, 0.11), R)
        mt, zt = _capsule(X, Y, (sx * 0.055, 2.20, 0.66), (sx * 0.035, 1.94, 0.71), 0.044)
        mw, zw = _capsule(X, Y, (sx * 0.07, 2.24, 0.52), (sx * 0.17, 2.38, 0.47), 0.07)
        one = kit.smax(np.where(mp, zp, ZBG), np.where(mf, zf, ZBG), 0.04)
        one = kit.smax(one, np.where(mt, zt, ZBG), 0.02)
        one = kit.smax(one, np.where(mw, zw, ZBG), 0.03)
        zh = np.maximum(zh, one)
    rl.put(sl, zh > ZBG + 0.1, zh, SKIN, M_SKIN)
    return 0.0, 1.95


def _feet(rl):
    """Bare toes under the hem, a golden rose on each foot."""
    reg = rl.region(-0.4, 0.4, SOLE - 0.3, SOLE)
    if reg is None:
        return
    sl, X, Y = reg
    zt = np.full(X.shape, ZBG, F32)
    zr = np.full(X.shape, ZBG, F32)
    for sx in (-1.0, 1.0):
        mt, z1, _ = _ellipsoid(X, Y, (sx * 0.16, SOLE - 0.065, 0.40), (0.092, 0.08, 0.17))
        zt = np.maximum(zt, np.where(mt & (Y < SOLE), z1, ZBG))
        mr, z2, loc = _ellipsoid(X, Y, (sx * 0.16, SOLE - 0.15, 0.47), (0.085, 0.07, 0.07))
        # petals: a little swirl of ridges
        ang = np.arctan2(loc[1], loc[0])
        rr = np.hypot(loc[0] / 0.085, loc[1] / 0.07)
        z2 = z2 + 0.006 * np.cos(5.0 * ang + 9.0 * rr) * (rr < 1)
        zr = np.maximum(zr, np.where(mr, z2, ZBG))
    rl.put(sl, zt > ZBG + 0.1, zt, SKIN, M_SKIN)
    rl.put(sl, zr > ZBG + 0.1, zr, GOLD, M_GOLD)


def _monogram(X, Y, cx, cy, h):
    """Distance (units) to the strokes of the Marian monogram: an M beneath a
    small cross, h tall overall, centred on (cx, cy)."""
    m = h * 0.62                            # height of the M
    top, bot = cy - 0.5 * h + h - m, cy + 0.5 * h
    hw = m * 0.6
    pts = [(cx - hw, bot), (cx - hw * 0.8, top), (cx, bot - m * 0.38), (cx + hw * 0.8, top), (cx + hw, bot)]
    d = np.full(X.shape, 9.0, F32)
    for (ax_, ay_), (bx_, by_) in zip(pts[:-1], pts[1:]):
        d = np.minimum(d, kit.sd_segment(X, Y, ax_, ay_, bx_, by_)[0])
    ct = cy - 0.5 * h
    d = np.minimum(d, kit.sd_segment(X, Y, cx, ct, cx, top - 0.18 * m)[0])
    arm = ct + 0.3 * (top - 0.18 * m - ct)
    d = np.minimum(d, kit.sd_segment(X, Y, cx - 0.22 * m, arm, cx + 0.22 * m, arm)[0])
    return d


def _plinth(rl):
    """Octagonal stone plinth: cornice and base mouldings, gilt fillet, a sunk
    front panel, and on large sprites a gilt Marian monogram in it."""
    reg = rl.region(-HALF_W, HALF_W, SOLE, PL_BOT)
    if reg is None:
        return
    sl, X, Y = reg
    yc1, h1 = SOLE + 0.07, 0.07             # cornice roll
    yc2, h2 = PL_BOT - 0.10, 0.10           # base torus
    mould = (0.08 * np.sqrt(np.clip(1.0 - ((Y - yc1) / h1) ** 2, 0.0, 1.0))
             + 0.12 * np.sqrt(np.clip(1.0 - ((Y - yc2) / h2) ** 2, 0.0, 1.0)))
    hw = 1.30 + mould
    ch = 0.24
    e = np.abs(X) - (hw - ch)
    z = 0.90 + mould - np.maximum(e, 0.0) * 1.1
    m = (np.abs(X) < hw) & (Y >= SOLE) & (Y < PL_BOT)
    # sunk panel with chamfered edges
    py0, py1 = SOLE + 0.22, PL_BOT - 0.23
    dp = kit.sd_box(X, Y, 0.0, 0.5 * (py0 + py1), 0.98, 0.5 * (py1 - py0))
    z = z - 0.035 * np.clip(-dp / 0.035, 0.0, 1.0)
    fil = (Y > SOLE + 0.145) & (Y < SOLE + 0.175)
    f = fil.astype(F32)
    k = float(_ss(19.0, 28.0, rl.s))        # monogram only where it can be read
    if k > 0:
        rs = 0.022
        dm = _monogram(X, Y, 0.0, 0.5 * (py0 + py1), 0.80 * (py1 - py0)) - rs
        relief = np.clip(-dm / rs, 0.0, 1.0)
        z = z + 0.02 * np.sqrt(relief)
        f = np.maximum(f, k * np.clip(0.5 - dm * rl.s, 0.0, 1.0))
    alb = STONE * (1 - f[..., None]) + GOLD * f[..., None]
    mat = (M_STONE[0] + (M_GOLD[0] - M_STONE[0]) * f, M_STONE[1] + (M_GOLD[1] - M_STONE[1]) * f, M_GOLD[2] * f)
    rl.put(sl, m, z.astype(F32), alb, mat)


# =============================================================================
# the rosary
# =============================================================================
MEDAL = (0.0, 5.25)
# the loop, from the medal up her right strand (viewer's left), through the
# joined hands (hidden behind them) and down the other strand to the medal
LOOP = [MEDAL, (-0.04, 5.04), (-0.075, 4.50), (-0.095, 3.85), (-0.105, 3.15), (-0.105, 2.60), (-0.09, 2.32),
        (-0.06, 2.18), (0.0, 2.12), (0.06, 2.18),
        (0.09, 2.32), (0.105, 2.60), (0.105, 3.15), (0.095, 3.85), (0.075, 4.50), (0.04, 5.04), MEDAL]
HELD_Y = 2.31                    # beads above this are inside the hands
# centre spacing in units of the small-bead pitch p
GAP_SS, GAP_SL, GAP_MS = 1.0, 1.65, 1.5          # small-small, small-large, medal-small
GAP_ML, GAP_LC = 1.8, 2.35                       # medal-large (pendant), large-crucifix centre
R_SMALL, R_LARGE = 0.46, 0.70                    # bead radii (in p)
KIND_CROSS, KIND_LARGE, KIND_SMALL, KIND_MEDAL = 0, 1, 2, 3


def _catmull_rom(pts, n=24):
    """Dense polyline through the points (uniform Catmull-Rom)."""
    P = np.asarray(pts, np.float64)
    P = np.vstack([P[:1], P, P[-1:]])
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    out.append(P[-2:-1])
    return np.vstack(out)


def _rosary_layout():
    """Bead centres (61, 2) in prayer order, kinds, radii (units) and the pitch p.

    Prayer order: crucifix, large, 3 small, large (the pendant, going up), then
    around the loop from the medal: 5 decades of 10 small beads with a large
    bead between decades, and the medal itself last."""
    path = _catmull_rom(LOOP)
    seg = np.hypot(*np.diff(path, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    # loop elements by arc length (in p): decades separated by large beads
    pos, kinds, sp = [], [], GAP_MS
    for dec in range(5):
        for k in range(10):
            pos.append(sp)
            kinds.append(KIND_SMALL)
            sp += GAP_SS if k < 9 else GAP_SL
        if dec < 4:
            pos.append(sp)
            kinds.append(KIND_LARGE)
            sp += GAP_SL
    p = arc[-1] / (pos[-1] + GAP_MS)        # the loop closes at the medal
    sl = np.asarray(pos) * p
    loop = np.stack([np.interp(sl, arc, path[:, 0]), np.interp(sl, arc, path[:, 1])], -1)
    # the pendant hangs straight down from the medal
    mx, my = MEDAL
    drops = np.cumsum([GAP_ML, GAP_SL, GAP_SS, GAP_SS, GAP_SL, GAP_LC]) * p
    pend = np.stack([np.full(6, mx), my + drops], -1)[::-1]          # crucifix first
    pend_k = [KIND_CROSS, KIND_LARGE, KIND_SMALL, KIND_SMALL, KIND_SMALL, KIND_LARGE]
    xy = np.vstack([pend, loop, [MEDAL]])
    kind = np.array(pend_k + kinds + [KIND_MEDAL], np.int32)
    rad = np.where(kind == KIND_LARGE, R_LARGE, R_SMALL) * p
    rad = np.where(kind == KIND_MEDAL, 0.75 * p, np.where(kind == KIND_CROSS, 1.35 * p, rad))
    return xy, kind, rad, p


def _rosary(rl):
    """Pale beads on a thin gilt chain, gilt medal and crucifix.  Beads sit on
    whatever is behind them (read back from the depth buffer); the ones held in
    the hands sit behind the fingers.  Returns the layout (units)."""
    xy, kind, rad, p = _rosary_layout()
    s = rl.s
    held = xy[:, 1] < HELD_Y
    # depth of each element: just in front of the deepest thing under its footprint
    st = rl.step
    bz = np.empty(len(xy))
    for i, ((x, y), r) in enumerate(zip(xy, rad)):
        j0 = int((x - r - rl.ux0) / st)
        j1 = int((x + r - rl.ux0) / st) + 1
        i0 = int((y - r - rl.uy0) / st)
        i1 = int((y + r - rl.uy0) / st) + 1
        win = rl.z[max(0, i0):max(1, i1), max(0, j0):max(1, j1)]
        bz[i] = (float(win.max()) if win.size else 0.4) + 0.01
    bz = np.where(held, 0.45, bz)
    rb = np.maximum(rad, 0.3 / s * _ss(6.0, 11.0, s))    # >= ~0.6 px wide unless the statue is tiny
    pad = float(rb.max()) * 1.5
    reg = rl.region(xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    if reg is None:
        return xy, kind, rad
    sl, X, Y = reg
    zr = np.full(X.shape, ZBG, F32)
    em = np.zeros(X.shape, F32)
    gold = np.zeros(X.shape, bool)
    # chain: thin segments between consecutive elements (loop closed through the medal)
    order = list(range(6, 61)) + [6]
    chain = [(order[i], order[i + 1]) for i in range(len(order) - 1)] + [(i, i + 1) for i in range(5)] + [(5, 60)]
    rc = max(0.1 * p, 0.22 / s * float(_ss(6.0, 11.0, s)))
    for a_, b_ in chain:
        (ax_, ay_), (bx_, by_) = xy[a_], xy[b_]
        d, t = kit.sd_segment(X, Y, ax_, ay_, bx_, by_)
        zc = bz[a_] + (bz[b_] - bz[a_]) * t - 0.01 + np.sqrt(np.maximum(rc * rc - d * d, 0.0))
        hit = (d < rc) & (zc > zr)
        zr = np.where(hit, zc, zr)
        gold |= hit
    # beads: sphere caps
    for i in np.flatnonzero((kind == KIND_SMALL) | (kind == KIND_LARGE)):
        x0, y0 = xy[i]
        r = rb[i]
        d2 = (X - x0) ** 2 + (Y - y0) ** 2
        cap = np.sqrt(np.maximum(r * r - d2, 0.0))
        zc = bz[i] + cap
        hit = (d2 < r * r) & (zc > zr)
        zr = np.where(hit, zc, zr)
        gold &= ~hit
        em = np.where(hit, (cap / r) ** 6, em)
    # medal: domed oval with a raised rim
    i = int(np.flatnonzero(kind == KIND_MEDAL)[0])
    x0, y0 = xy[i]
    rx, ry = max(0.55 * p, 0.35 / s), max(0.75 * p, 0.45 / s)
    q = ((X - x0) / rx) ** 2 + ((Y - y0) / ry) ** 2
    zc = bz[i] + 0.25 * rx * np.sqrt(np.maximum(1.0 - q, 0.0)) + 0.1 * rx * np.exp(-((np.sqrt(q) - 0.85) / 0.12) ** 2)
    hit = (q < 1.0) & (zc > zr)
    zr = np.where(hit, zc, zr)
    gold |= hit
    # crucifix: a slender Latin cross, gently domed
    i = int(np.flatnonzero(kind == KIND_CROSS)[0])
    x0, y0 = xy[i]
    hl, tw = max(1.35 * p, 0.9 / s), max(0.2 * p, 0.3 / s)
    d = np.minimum(kit.sd_box(X, Y, x0, y0, tw, hl), kit.sd_box(X, Y, x0, y0 - 0.38 * hl, 0.62 * hl, tw))
    zc = bz[i] + 0.6 * tw * np.clip(-d / tw, 0.0, 1.0)
    hit = (d < 0) & (zc > zr)
    zr = np.where(hit, zc, zr)
    gold |= hit
    m = zr > ZBG + 0.1
    # contact shadow: a soft dark rim on the cloth around every bead, the medal and
    # the cross, so the beads stay distinct on the pale gown
    ring = 0.55 + 0.45 * _ss(0.0, max(0.5 * tw, 0.6 / s), d)                 # the cross (d: its SDF)
    ring = np.minimum(ring, 0.55 + 0.45 * _ss(1.0, 1.6, np.sqrt(q)))         # the medal
    for i in np.flatnonzero(~held & ((kind == KIND_SMALL) | (kind == KIND_LARGE))):
        x0, y0 = xy[i]
        r = rb[i]
        dd = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        ring = np.minimum(ring, 0.55 + 0.45 * _ss(r, r + max(0.7 * r, 0.6 / s), dd))
    under = rl.aom[sl]
    under *= np.where(m, 1.0, ring)
    g = gold.astype(F32)
    alb = BEAD * (1 - g[..., None]) + GOLD * g[..., None]
    mat = (M_BEAD[0] + (M_GOLD[0] - M_BEAD[0]) * g, M_BEAD[1] + (M_GOLD[1] - M_BEAD[1]) * g, M_GOLD[2] * g)
    rl.put(sl, m, zr, alb, mat, emi=(em * (1 - g))[..., None] * (BEAD * 0.10))
    return xy, kind, rad


def _stars(rl):
    """Twelve faceted gilt stars on an arc around the head, points outward."""
    ang = np.linspace(-STAR_SPAN, STAR_SPAN, 12)
    cx = HALO_C[0] + HALO_R * np.sin(ang)
    cy = HALO_C[1] - HALO_R * np.cos(ang)
    sr = max(STAR_R, 0.85 / rl.s)           # small sprites: stars at least ~1.7 px across
    glow = 0.12 + 0.45 * float(np.clip((10.0 - rl.s) / 6.0, 0.0, 1.0))
    pad = sr * 1.2
    reg = rl.region(cx.min() - pad, cx.max() + pad, cy.min() - pad, cy.max() + pad)
    if reg is not None:
        sl, X, Y = reg
        z = np.full(X.shape, ZBG, F32)
        for x0, y0, a in zip(cx, cy, ang):
            dx, dy = X - F32(x0), -(Y - F32(y0))
            ca, sa = math.cos(a), math.sin(a)
            u, v = dx * ca - dy * sa, dx * sa + dy * ca
            d = _sd_star5(u, v, sr, 0.42)
            hz = -0.32 + 0.04 * np.clip(-d / (sr * 0.4), 0.0, 1.0)
            z = np.where(d < 0, np.maximum(z, hz), z)
        rl.put(sl, z > ZBG + 0.1, z, GOLD, M_GOLD, emi=GOLD * glow, body=False)
    return np.stack([cx, cy], -1)


# =============================================================================
# entry point
# =============================================================================
def render(h, seed=0):
    """Our Lady praying the rosary, `h` pixels tall (width about 0.34 h).

    Anchors (sprite pixels, x right, y down): base (plinth foot, centre), face,
    hands, halo (x, y, r of the star crown), stars (12, 2), beads (61, 2) in
    prayer order with bead_kind (61,) 0 crucifix / 1 large / 2 small / 3 medal
    and bead_r (61,) radii.  The seed only varies the folds a little."""
    h = max(6, int(h))
    s = (h - 2) / (BOT - TOP)               # pixels per head height
    w = max(3, int(math.ceil(2 * HALF_W * s)) + 2)
    ss = 4 if h < 48 else 3 if h < 120 else 2
    rl = _Relief(h, w, s, ss)
    folds = _Folds(seed)
    stars = _stars(rl)
    face = _head(rl)
    _veil(rl, folds)
    _mantle(rl, folds)
    _gown(rl, folds)
    hands = _hands(rl)
    _feet(rl)
    beads, kind, rad = _rosary(rl)
    _plinth(rl)
    spr = rl.resolve()

    def px(p):
        x, y = rl.px(p[0], p[1])
        return float(x), float(y)

    spr.anchors["base"] = px((0.0, PL_BOT))
    spr.anchors["face"] = px(face if face else (0.0, 0.7))
    spr.anchors["hands"] = px(hands if hands else (0.0, 1.95))
    hx, hy = px(HALO_C)
    spr.anchors["halo"] = (hx, hy, float(HALO_R * s))
    sx, sy = rl.px(stars[:, 0], stars[:, 1])
    spr.anchors["stars"] = np.stack([sx, sy], -1).astype(F32)
    bx, by = rl.px(beads[:, 0], beads[:, 1])
    spr.anchors["beads"] = np.stack([bx, by], -1).astype(F32)
    spr.anchors["bead_kind"] = kind.astype(np.int32)
    spr.anchors["bead_r"] = (rad * s).astype(F32)
    return spr.check()
