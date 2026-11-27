"""MONSTRANCE - the Blessed Sacrament exposed for adoration (CATHEDRAL view).

A gold sunburst monstrance (ostensorium) built straight into a `kit.Sprite`
G-buffer, so the scene's candles relight it every frame.  Everything is
analytic 2D with anti-aliased coverage, composited back to front with
straight alpha:

* The sun: straight blades alternating long and short, with slim wavy flame
  rays between them when there is room.  Domain repetition in angle means each
  pixel only measures the nearest blade of each family.  A blade's normal is a
  bevelled ridge across it plus a slight cone tilt of the whole sunburst, so
  rays at different angles catch different candles and glint in turn; where
  the roots crowd together under the frame the bevels fade into one collar.
* The cross: a small Latin cross pattee on an orb, carried by a rod that takes
  the place of the topmost ray.
* Neck, stem, knop and stepped foot: one solid of revolution seen from the
  front.  Its profile R(y) is sampled on sub-rows (smooth horizontal edges);
  each pixel's normal follows from the profile slope and the pixel's angle
  around the axis, and the gadroons and the chased bead row of the base band
  are ripples of the normal in that angle.
* The luna: burnished rims (flattened torus normals), a ring of gold beads
  (sphere normals) set with rubies and sapphires, a slightly domed glass, and
  the Host - a matte, faintly ivory wafer with a soft glow of its own and an
  impressed cross.  The scene adds the breathing halo and bloom.

Detail follows the size: beads, stones, flame rays, gadroons and the bead row
only appear once they span enough pixels to read, rays keep a minimum width,
and small sprites get a broader gloss (a pixel there averages more of the
bevels under it), so a 24 px monstrance still reads as a gold starburst with a
white Host on a stem.  The seed only varies the hand-chased bevels of the rays.

`render(h, seed)` returns the sprite with anchors for the scene: base, center,
host (x, y, r), cross (top of the cross) and rays (the tip of every visible
straight ray, for twinkling glints).
"""

import math

import numpy as np

from . import kit

F32 = np.float32

GOLD = kit.rgb((1.0, 0.84, 0.50))          # polished 18k yellow gold (its reflectance)
HOST = kit.rgb((0.95, 0.92, 0.84))         # unleavened wafer: matte, faintly ivory
HOST_GLOW = kit.rgb((0.58, 0.55, 0.47))    # its own soft light (the scene adds the halo)
GLASS = kit.rgb((0.02, 0.025, 0.04))       # the luna's glass, dark against the night
RUBY = kit.rgb((0.55, 0.03, 0.05))
SAPPHIRE = kit.rgb((0.07, 0.16, 0.62))

SPEC = 1.0
GLOSS_RAY = 10.0
GLOSS_CAST = 12.0          # turned and cast parts: foot, stem, knop, cross
GLOSS_RIM = 24.0           # the burnished rim of the luna
RAY_AO = 0.35

RAY_SPACING = 6.5          # px between neighbouring ray tips (long to short)
W_LONG = 0.016             # half width of a long ray at its root, as a fraction of the height
W_SHORT = 0.0135
W_MIN = 1.2                # (px) rays never get thinner than this at the root
GEMS_S = 120.0             # (px) from this height on the luna is set with stones
WAVY_R = 40.0              # (px) sun radius from which flame rays appear between the straight ones
T_NECK = 0.085             # heights below the sun, as fractions of the span from the
T_KNOP = 0.445             # bottom of the luna frame (0) to the base (1)
T_FOOT = 0.665
GADROONS = 22
LATHE_DEPTH = 0.6          # depth of the turned parts' section relative to their width

BEVEL = math.tan(math.radians(18.0))     # facet slope across a ray
TORUS_TILT = math.tan(math.radians(50.0))  # how far the luna's rings round over at their edges
CONE = math.tan(math.radians(4.0))       # the sunburst leans back toward its tips


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _unit(nx, ny, nz):
    n = np.stack([nx, ny, nz], -1).astype(F32)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    return n


def _over(spr, win, cov, albedo, normal=None, spec=0.0, gloss=16.0, metal=0.0, emissive=None, ao=None):
    """Composite one layer over the window spr[win], in place.

    Like `Sprite.paint`, but with straight (unpremultiplied) alpha: a layer
    over empty pixels keeps its own colour and normal, and only `alpha`
    records the coverage.  (Blending with the blank sprite's black albedo
    would darken anti-aliased edges and thin rays twice once the scene
    composites the sprite by its alpha.)"""
    c = np.clip(np.asarray(cov, F32), 0.0, 1.0)
    P = spr.alpha[win]
    under = P * (1.0 - c)                   # weight of what is already there
    a_new = c + under
    k = np.where(a_new > 1e-6, 1.0 / np.maximum(a_new, 1e-6), 0.0).astype(F32)
    wl, wu = c * k, under * k               # normalised weights: layer, underneath
    keep = a_new <= 1e-6                    # nothing at all here: leave the blank values
    wl3, wu3, keep3 = wl[..., None], wu[..., None], keep[..., None]
    A = spr.albedo[win]
    A[:] = np.where(keep3, A, np.asarray(albedo, F32) * wl3 + A * wu3)
    N = spr.normal[win]
    nn = (np.array([0.0, 0.0, 1.0], F32) if normal is None else np.asarray(normal, F32)) * wl3 + N * wu3
    nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
    N[:] = np.where(keep3, N, nn)
    for name, val in (("spec", spec), ("gloss", gloss), ("metal", metal)):
        B = getattr(spr, name)[win]
        B[:] = np.where(keep, B, np.asarray(val, F32) * wl + B * wu)
    E = spr.emissive[win]
    em = np.zeros(3, F32) if emissive is None else np.asarray(emissive, F32)
    E[:] = np.where(keep3, E, em * wl3 + E * wu3)
    O = spr.ao[win]
    O[:] = np.where(keep, O, (np.ones(1, F32) if ao is None else np.asarray(ao, F32)) * wl + O * wu)
    P[:] = a_new


def _window(lay, y0, y1, x0, x1):
    i0, i1 = max(0, int(math.floor(y0))), min(lay.h, int(math.ceil(y1)))
    j0, j1 = max(0, int(math.floor(x0))), min(lay.W, int(math.ceil(x1)))
    if i1 <= i0 or j1 <= j0:
        return None
    return (slice(i0, i1), slice(j0, j1))


# =============================================================================
# layout: every dimension in sprite pixels for a sprite h pixels tall
# =============================================================================
class _Layout:
    def __init__(self, h):
        self.h = h
        m = 1.0 if h >= 12 else 0.5
        s = max(h - 2.0 * m, 1.0)                   # height of the monstrance itself
        pf = min(1.0, max(0.3, s / 30.0))           # pixel floors shrink on tiny sprites
        self.m, self.s, self.pf = m, s, pf
        # a smaller pixel averages more of the chasing and bevels under it, so the
        # highlights it shows are broader: lower the effective gloss on small sprites
        gs = min(1.0, 0.35 + s / 150.0)
        self.g_ray, self.g_cast, self.g_rim = GLOSS_RAY * gs, GLOSS_CAST * gs, GLOSS_RIM * gs

        # the sun: long rays reach R, about half the height across
        self.R = R = 0.275 * s
        W = int(math.ceil(2.0 * R + 2.0 * m - 0.15))
        W += (W + 1) % 2              # odd: the axis runs down the middle of a pixel column
        self.W = max(3, W)
        self.cx = self.W / 2.0

        # cross on an orb, the orb sitting on the circle of the long ray tips
        ct = max(0.095 * s, min(4.2, 0.17 * s))
        self.r_orb = min(max(0.75 * pf, 0.021 * s), 0.25 * ct)
        self.arm = max(0.3, (ct - 0.8 * self.r_orb) / 3.0)
        self.bar = max(0.5 * pf, 0.0085 * s)
        self.y_top = m
        self.y_bar = m + self.arm
        self.y_orb = m + ct
        self.cy = self.y_orb + R

        # the luna: Host, glass, rim, beads, frame
        self.r_host = max(1.2 * pf, 0.062 * s)
        self.r_glass = self.r_host + max(0.35 * pf, 0.011 * s)
        self.r_rim = self.r_glass + max(0.7 * pf, 0.022 * s)
        self.br = 0.0105 * s
        self.beads = self.br >= 1.25            # beads (and stones) only where they resolve
        self.gems = s >= GEMS_S
        if self.beads:
            self.r_bead = self.r_rim + 0.9 * self.br
            self.r_frame_in = self.r_bead + 0.55 * self.br
            self.r_frame = self.r_frame_in + 0.017 * s
            self.r_gems, self.rg = self.r_bead, 1.45 * self.br
        else:
            self.r_bead = self.r_rim
            self.r_frame_in = self.r_rim - 0.15
            self.r_frame = self.r_frame_in + max(0.6 * pf, 0.024 * s)
            self.r_gems = 0.5 * (self.r_frame_in + self.r_frame)
            self.rg = max(1.0, 0.45 * (self.r_frame - self.r_frame_in))

        # rays
        # long rays: a multiple of 4 (one points straight up), about RAY_SPACING px
        # apart at the tips.  Big sprites get fewer, slender straight rays with
        # wavy flame rays in the gaps between them.
        self.wavy = R >= WAVY_R
        if self.wavy:
            self.n_long = 16 if R < 72 else 20
        else:
            self.n_long = int(min(24, max(8, 4 * round(math.pi * R / (4.0 * RAY_SPACING)))))
        self.r0 = 0.82 * self.r_frame
        L = R - self.r0
        self.R_short = self.r0 + 0.64 * L
        self.w_long = max(W_MIN * pf, min(W_LONG * s, 1.75 + 0.005 * s))
        self.w_short = max(0.83 * W_MIN * pf, W_SHORT / W_LONG * self.w_long)
        self.R_wavy = self.r0 + 0.56 * L
        self.w_wavy = max(0.6, 0.42 * self.w_long)
        gap = math.pi / self.n_long * (self.r0 + 0.25 * L)      # between straight rays, px
        self.wave = (0.2 * gap, (self.R_wavy - self.r0) / 2.2)    # amplitude, wavelength
        self.w_rod = max(0.5 * pf, min(0.6 * self.w_long, 0.55 * self.r_orb))

        # turned parts below the sun: heights are fractions t of the span from
        # the bottom of the luna frame (t = 0) to the base (t = 1), radii in px
        self.y_fb = self.cy + self.r_frame
        self.y_base = h - m
        self.span = max(self.y_base - self.y_fb, 1.0)
        self.r_stem = max(0.5 * min(1.0, s / 14.0), 0.016 * s)
        self.r_neck = max(self.r_stem + 0.6 * pf, 0.042 * s)
        self.r_collar = max(self.r_stem + 0.5 * pf, 0.027 * s)
        self.r_knop = max(self.r_stem + 0.9 * pf, 0.056 * s)
        self.r_foot_top = max(self.r_stem + 0.8 * pf, 0.05 * s)
        self.r_drum = max(self.r_foot_top + 0.6 * pf, 0.104 * s)      # upper step of the foot
        self.r_step = max(self.r_drum + 0.6 * pf, 0.148 * s)          # middle step (gadrooned)
        self.r_foot = 0.175 * s                                        # base band (engraved)
        self.hh_collar = max(0.5 * pf, 0.0075 * s) / self.span
        self.hh_knop = max(0.9 * pf, 0.04 * s) / self.span


# =============================================================================
# the sun's rays
# =============================================================================
def _ray_family(dx, dyv, rho, theta, n, phase, r0, R, w0, jitter, fade, crest=0.35, wave=None):
    """Signed distance (px), view-space normals and index of n tapered blades.

    Blade k points along angle phase + k*2pi/n (0 = up, clockwise), from r0
    (hidden under the frame) to its tip at R, w0 px half wide at the root.
    Each blade is a ridge: two bevelled facets meeting along its axis, rounded
    over the inner `crest` of its half width.  jitter (per blade) and fade (per
    pixel) scale the bevel.  wave = (amplitude, wavelength) bends the axis into
    a flame, pinned at both ends."""
    step = 2.0 * math.pi / n
    k = np.round((theta - phase) / step)
    tk = phase + k * step
    a = theta - tk
    along = rho * np.cos(a)
    across = rho * np.sin(a)
    L = max(R - r0, 1e-3)
    side = 1.0 / math.sqrt(1.0 + (w0 / L) ** 2)       # distance factor of the slanted sides
    slope = np.zeros_like(across)
    if wave is not None:
        amp, lam = wave
        u = np.clip((along - r0) / L, 0.0, 1.0)
        env = amp * np.maximum(np.sin(np.pi * u), 0.0) ** 0.8     # pinned at the base and the tip
        ph = (along - r0) * F32(2.0 * math.pi / lam)
        across = across - env * np.sin(ph)
        slope = env * np.cos(ph) * F32(2.0 * math.pi / lam)
    w = w0 * (R - along) / L
    corr = 1.0 / np.sqrt(1.0 + slope * slope)
    d = np.maximum((np.abs(across) - w) * side * corr, along - R)
    d = np.maximum(d, r0 - along)
    # ridge: height falls away from the axis (facets), rounded near the crest
    ki = k.astype(np.int64) % n
    b = BEVEL * (1.0 + jitter[ki]) * fade         # roots that crowd together stay flat
    u = np.abs(across) / np.maximum(w, 0.3)
    hc = -np.sign(across) * b * np.clip(u / crest, 0.0, 1.0)
    # gradient in (along, across), with the across direction bent by the wave
    g_al = -CONE - hc * slope * corr
    g_ac = hc * corr
    st, ct = np.sin(tk), np.cos(tk)
    gx = g_al * st + g_ac * ct
    gy = g_al * ct - g_ac * st
    return d.astype(F32), _unit(-gx, -gy, np.ones_like(gx)), k


def _sun_rays(spr, lay, dx, dyv, rho, theta, rng):
    nl = lay.n_long
    step = 2.0 * math.pi / nl
    jit = rng.uniform(-0.18, 0.18, (3, 2 * nl)).astype(F32)
    # where the roots of the rays crowd together under the frame they merge
    # into one smooth collar: the bevels only develop further out
    fade = _smooth((rho - lay.r_frame) / (0.045 * lay.s + 1.0))
    # the rays sit in the shadow of the frame near their roots
    ao = (1.0 - RAY_AO * np.exp(-np.maximum(rho - lay.r_frame, 0.0) / (0.03 * lay.s + 0.6))).astype(F32)
    full = (slice(0, lay.h), slice(0, lay.W))
    if lay.wavy:
        d, n, _ = _ray_family(dx, dyv, rho, theta, 2 * nl, step / 4.0, lay.r0, lay.R_wavy, lay.w_wavy,
                              jit[2], fade, wave=lay.wave)
        _over(spr, full, kit.cover(d), GOLD, n, SPEC, lay.g_ray, 1.0, None, ao)
    d, n, _ = _ray_family(dx, dyv, rho, theta, nl, step / 2.0, lay.r0, lay.R_short, lay.w_short, jit[1], fade)
    _over(spr, full, kit.cover(d), GOLD, n, SPEC, lay.g_ray, 1.0, None, ao)
    d, n, k = _ray_family(dx, dyv, rho, theta, nl, 0.0, lay.r0, lay.R, lay.w_long, jit[0], fade)
    cov = kit.cover(d) * (k != 0)          # the topmost ray is the rod of the cross
    _over(spr, full, cov, GOLD, n, SPEC, lay.g_ray, 1.0, None, ao)


def _ray_tips(lay):
    """Tips of the straight rays that are not hidden by the cross or the stem."""
    nl = lay.n_long
    step = 2.0 * math.pi / nl
    tips = []
    for k in range(nl):
        for ang, R in ((k * step, lay.R), ((k + 0.5) * step, lay.R_short)):
            x = lay.cx + R * math.sin(ang)
            y = lay.cy - R * math.cos(ang)
            if k == 0 and R == lay.R:
                continue                                  # under the orb
            if y > lay.y_fb - 1.0:
                half = float(_profile(lay, np.array([y], F32))[0])
                if abs(x - lay.cx) < half + 0.6:
                    continue                              # behind the stem
            tips.append((x, y))
    return np.array(tips, F32).reshape(-1, 2)


# =============================================================================
# cross, orb and rod
# =============================================================================
def _ridge_x(X, w, b=BEVEL, crest=0.35):
    """x part (before normalising, z = 1) of the normal of a bevelled bar of half
    width w running along y, at offset X from its axis."""
    hc = -np.sign(X) * b * np.clip(np.abs(X) / np.maximum(w, 0.3) / crest, 0.0, 1.0)
    return -hc


def _cross(spr, lay, ys, xs):
    half = max(lay.arm * 1.6, lay.r_orb) + 1.5
    y_end = lay.cy - 0.9 * lay.r_frame              # the rod ends under the frame
    win = _window(lay, lay.y_top - 1.0, y_end + 1.0, lay.cx - half, lay.cx + half)
    if win is None:
        return
    X = xs[win] - F32(lay.cx)
    Y = ys[win]
    one = np.ones_like(X)

    # rod from inside the frame up to the orb (it replaces the topmost ray)
    d = np.maximum(np.abs(X) - lay.w_rod, np.maximum(lay.y_orb - Y, Y - y_end))
    n = _unit(_ridge_x(X, lay.w_rod), CONE * one, one)
    _over(spr, win, kit.cover(d), GOLD, n, SPEC, lay.g_ray, 1.0)

    # the cross: a Latin cross whose arms flare a little toward their ends
    a, t = lay.arm, lay.bar
    Yb = Y - F32(lay.y_bar)
    flare = 0.55 if lay.s >= 40 else 0.0
    tv = t * (1.0 + flare * np.clip(-Yb / a, 0.0, 1.0) ** 2)
    d_v = np.maximum(np.abs(X) - tv, np.maximum(-a - Yb, Yb - (lay.y_orb - lay.y_bar)))
    th = t * (1.0 + flare * np.clip(np.abs(X) / a, 0.0, 1.0) ** 2)
    d_h = np.maximum(np.abs(Yb) - th, np.abs(X) - a)
    d = np.minimum(d_v, d_h)
    vert = d_v <= d_h
    nx = np.where(vert, _ridge_x(X, tv), 0.0)
    ny = np.where(vert, 0.0, -_ridge_x(Yb, th))
    _over(spr, win, kit.cover(d), GOLD, _unit(nx, ny, one), SPEC, lay.g_cast, 1.0)

    # the orb the cross stands on
    oy = Y - F32(lay.y_orb)
    r = lay.r_orb
    q = np.clip((X * X + oy * oy) / (r * r), 0.0, 1.0)
    d = np.hypot(X, oy) - r
    _over(spr, win, kit.cover(d), GOLD, _unit(X / r, -oy / r, np.sqrt(1.0 - q)), SPEC, lay.g_cast, 1.0)


# =============================================================================
# neck, stem, knop and foot: one solid of revolution
# =============================================================================
def _profile(lay, y):
    """Half width (px) of the turned parts at rows y (px)."""
    t = (y - lay.y_fb) / lay.span          # 0 under the luna frame, 1 at the base
    st = lay.r_stem
    R = np.zeros_like(y)

    def seg(t0, t1, r):
        return np.where((t >= t0) & (t <= t1), r, 0.0)

    def ell(tc, hh, r):
        v = (t - tc) / hh
        return r * np.sqrt(np.clip(1.0 - v * v, 0.0, 1.0))

    hk, hc = lay.hh_knop, lay.hh_collar
    # neck: a small round-shouldered cup that the luna sits in, then a collar
    u = np.clip(t / T_NECK, 0.0, 1.0)
    R = np.maximum(R, seg(-1.0, T_NECK, st + (lay.r_neck - st) * np.maximum(np.cos(0.5 * np.pi * u), 0.0) ** 0.8))
    R = np.maximum(R, ell(T_NECK + 0.6 * hc, hc, lay.r_collar))
    # stem with its knop (an apple-shaped node with a raised girdle) at mid height
    R = np.maximum(R, seg(T_NECK, T_FOOT, st))
    R = np.maximum(R, ell(T_KNOP - hk - 0.8 * hc, hc, lay.r_collar))
    v = (t - T_KNOP) / hk
    R = np.maximum(R, lay.r_knop * np.clip(1.0 - v * v, 0.0, 1.0) ** 0.5 * (1.0 - 0.12 * np.clip(-v, 0.0, 1.0)))
    R = np.maximum(R, ell(T_KNOP, 0.8 * hc, lay.r_knop * 1.04))
    R = np.maximum(R, ell(T_KNOP + hk + 0.8 * hc, hc, lay.r_collar))
    # the foot, from the top: trumpet, collar, a low dome (ovolo) on a drum, a
    # cavetto down to the gadrooned step, another cavetto, then the base band
    def ramp(t0, t1):
        return np.clip((t - t0) / (t1 - t0), 0.0, 1.0)

    def ovolo(u):               # convex quarter round: steep at the top, vertical below
        return np.sqrt(np.clip(1.0 - (1.0 - u) ** 2, 0.0, 1.0))

    def cavetto(u):             # concave quarter round: vertical at the top, flat below
        return 1.0 - np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))

    u = ramp(T_FOOT - 0.05, T_FOOT + 0.015)
    R = np.maximum(R, seg(T_FOOT - 0.05, T_FOOT + 0.015, st + (lay.r_foot_top - st) * u * u))
    R = np.maximum(R, ell(T_FOOT + 0.018, hc, lay.r_foot_top * 1.12))
    u = ramp(T_FOOT + 0.018, T_FOOT + 0.113)
    R = np.maximum(R, seg(T_FOOT + 0.018, 0.82, lay.r_foot_top + (lay.r_drum - lay.r_foot_top) * ovolo(u)))
    R = np.maximum(R, seg(0.815, 0.85, lay.r_drum + (lay.r_step - lay.r_drum) * cavetto(ramp(0.815, 0.845))))
    R = np.maximum(R, ell(0.852, hc, lay.r_step * 1.025))
    R = np.maximum(R, seg(0.85, 0.91, lay.r_step))
    R = np.maximum(R, seg(0.905, 0.935, lay.r_step + (lay.r_foot - lay.r_step) * cavetto(ramp(0.905, 0.93))))
    R = np.maximum(R, ell(0.936, hc, lay.r_foot))
    u = ramp(0.975, 1.0)
    R = np.maximum(R, seg(0.935, 1.0, lay.r_foot * (1.0 - 0.02 * u * u)))     # eased bottom edge
    return R.astype(F32)


def _turned_parts(spr, lay):
    win = _window(lay, lay.cy + 0.6 * lay.r_frame, lay.h, lay.cx - lay.r_foot - 1.0, lay.cx + lay.r_foot + 1.0)
    if win is None:
        return
    (ri, cj) = win
    ns = 4
    rows = np.arange(ri.start, ri.stop, dtype=F32)
    yy = (rows[:, None] + (np.arange(ns, dtype=F32)[None, :] + 0.5) / ns).reshape(-1)
    R = _profile(lay, yy)
    eps = 0.08
    dR = (_profile(lay, yy + eps) - _profile(lay, yy - eps)) / (2 * eps)        # dR/dy (y down)
    inv = 1.0 / np.sqrt(1.0 + dR * dR)
    X = np.arange(cj.start, cj.stop, dtype=F32) + 0.5 - F32(lay.cx)
    d = (np.abs(X)[None, :] - R[:, None]) * inv[:, None]
    cov = kit.cover(d) * (R > 1e-3)[:, None]        # (an empty profile must not cover the axis)
    sphi = np.clip(X[None, :] / np.maximum(R[:, None], 1e-3), -1.0, 1.0)
    cphi = np.sqrt(1.0 - sphi * sphi)
    # profile normal (c_r, c_y) spun around the axis by the pixel's angle phi; the
    # section is drawn a little shallower than round (LATHE_DEPTH), so the normals
    # only swing sideways close to the silhouette and the bands stay lit to their ends
    nx = sphi * inv[:, None]
    ny = np.broadcast_to((dR * inv)[:, None], d.shape)
    nz = cphi * inv[:, None] / LATHE_DEPTH
    t = ((yy - lay.y_fb) / lay.span)[:, None]
    phi = np.arcsin(sphi)
    # occlusion: narrow parts next to wide ones (under the knop, the collars, the frame)
    reach = max(2, int(round(0.035 * lay.s * ns)))
    Rp = np.pad(R, reach, mode="edge")
    Rmax = R.copy()
    for o in range(-reach, reach + 1):
        Rmax = np.maximum(Rmax, Rp[reach + o:reach + o + R.size])
    ao = 1.0 - 0.45 * np.clip((Rmax - R) / (0.025 * lay.s + 0.5), 0.0, 1.0)
    ao = ao * (1.0 - 0.3 * np.exp(-np.maximum(yy - lay.y_fb, 0.0) / (0.015 * lay.s + 0.4)))
    ao = np.broadcast_to(ao[:, None], d.shape).copy()
    # chased decoration, as ripples of the normal along the tangent around the
    # axis (tx, 0, tz) and up the profile (0, 1, 0)
    tx, tz = cphi, -sphi
    if lay.s >= 140:
        # gadroons (convex flutes) round the middle step
        step = ((t > 0.86) & (t < 0.905)).astype(F32)
        rip = 0.3 * step * np.sin(GADROONS * phi)
        nx, nz = nx + rip * tx, nz + rip * tz
        ao *= 1.0 - 0.15 * step * (0.5 - 0.5 * np.cos(GADROONS * phi)) ** 4
    if 60 <= lay.s < 140:
        # one engraved line round the middle of the base band
        ao *= 1.0 - 0.55 * np.exp(-(((t - 0.967) * lay.span) / 0.45) ** 2)
    if lay.s >= 140:
        # the base band: a row of chased beads between two engraved lines,
        # with a polished strip left below them
        yc = 0.96
        rb = 0.0085 * lay.s
        nbead = max(8, int(round(2.0 * math.pi * lay.r_foot / (2.3 * rb))))
        per = 2.0 * math.pi / nbead
        du = (np.mod(phi + 0.5 * per, per) - 0.5 * per) * lay.r_foot      # arc length to the nearest bead
        dv = (t - yc) * lay.span
        q = (du * du + dv * dv) / (rb * rb)
        inb = q < 1.0
        k = np.where(inb, 0.9, 0.0)
        nx = np.where(inb, nx * np.sqrt(np.clip(1.0 - q, 0.0, 1.0)) + k * du / rb * tx, nx)
        nz = np.where(inb, nz * np.sqrt(np.clip(1.0 - q, 0.0, 1.0)) + k * du / rb * tz, nz)
        ny = np.where(inb, ny - k * dv / rb, ny)
        band = np.abs(dv) < rb
        ao *= np.where(band & ~inb, 0.75, 1.0)
        for off in (-1.0, 1.0):
            ao *= 1.0 - 0.45 * np.exp(-((dv - off * (rb + 0.7)) / 0.45) ** 2)
    # average the sub-rows (coverage weighted)
    nr, nc = rows.size, X.size
    cw = cov.reshape(nr, ns, nc)
    csum = cw.sum(1)
    w = cw / np.maximum(csum, 1e-6)[:, None, :]
    n = np.stack([nx, ny, nz], -1).reshape(nr, ns, nc, 3)
    n = (n * w[..., None]).sum(1)
    n[csum < 1e-6] = (0.0, 0.0, 1.0)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    aop = (ao.reshape(nr, ns, nc) * w).sum(1)
    aop[csum < 1e-6] = 1.0
    _over(spr, win, csum / ns, GOLD, n, SPEC, lay.g_cast, 1.0, None, aop)


# =============================================================================
# the luna: frame, beads and stones, rim, glass and the Host
# =============================================================================
def _torus(rho, ex, ey, r_in, r_out, tilt=TORUS_TILT):
    """A burnished ring between radii r_in and r_out: rounded across, its
    normals leaning out (or in) by up to `tilt` at its edges."""
    mid, hw = 0.5 * (r_in + r_out), max(0.5 * (r_out - r_in), 0.2)
    t = np.clip((rho - mid) / hw, -1.0, 1.0)
    k = t * abs(t) ** 0.3 * tilt                     # flat-topped, rounding over at the edges
    return np.abs(rho - mid) - hw, _unit(k * ex, k * ey, np.ones_like(k))


def _luna(spr, lay, dx, dyv, rho, theta):
    rr = lay.r_frame + 1.5
    win = _window(lay, lay.cy - rr, lay.cy + rr, lay.cx - rr, lay.cx + rr)
    if win is None:
        return
    DX, DY, P, TH = dx[win], dyv[win], rho[win], theta[win]
    ex, ey = DX / np.maximum(P, 1e-4), DY / np.maximum(P, 1e-4)

    # recessed backing plate, then the outer frame
    _over(spr, win, kit.cover(P - lay.r_frame_in - 0.3), GOLD, None, SPEC, lay.g_rim, 1.0, None, 0.45)
    d, n = _torus(P, ex, ey, lay.r_frame_in, lay.r_frame)
    _over(spr, win, kit.cover(d), GOLD, n, SPEC, lay.g_rim, 1.0)

    if lay.beads:
        br = lay.br
        nb = 8 * max(1, int(round(2.0 * math.pi * lay.r_bead / (2.15 * br) / 8.0)))
        step = 2.0 * math.pi / nb
        k = np.round(TH / step)
        tk = k * step
        ox = DX - lay.r_bead * np.sin(tk)
        oy = DY - lay.r_bead * np.cos(tk)
        q = np.clip((ox * ox + oy * oy) / (br * br), 0.0, 1.0)
        cov = kit.cover(np.hypot(ox, oy) - br)
        if lay.gems:
            cov = cov * ((k.astype(np.int64) % (nb // 8)) != 0)
        _over(spr, win, cov, GOLD, _unit(ox / br, oy / br, np.sqrt(1.0 - q)), SPEC, lay.g_rim, 1.0)

    if lay.gems:
        # eight cabochons in gold collets: rubies at the cardinal points, sapphires between
        rg = lay.rg
        gk = np.round(TH / (math.pi / 4.0))
        gt = gk * (math.pi / 4.0)
        ox = DX - lay.r_gems * np.sin(gt)
        oy = DY - lay.r_gems * np.cos(gt)
        dg = np.hypot(ox, oy)
        q = np.clip((dg / rg) ** 2, 0.0, 1.0)
        _over(spr, win, kit.cover(dg - rg - 0.45), GOLD, _unit(ox / rg, oy / rg, np.sqrt(1.0 - q) + 0.3),
              SPEC, lay.g_rim, 1.0)
        col = np.where((gk.astype(np.int64) % 2 == 0)[..., None], RUBY, SAPPHIRE)
        _over(spr, win, kit.cover(dg - rg), col, _unit(ox / rg * 0.8, oy / rg * 0.8, np.sqrt(1.0 - q) + 0.2),
              0.9, 90.0, 0.0)

    # burnished inner rim
    d, n = _torus(P, ex, ey, lay.r_glass - 0.1, lay.r_rim)
    _over(spr, win, kit.cover(d), GOLD, n, SPEC, lay.g_rim, 1.0)

    # the glass: very slightly domed, so it catches one candle as a glint
    kdome = 0.32 / max(lay.r_glass, 1.0)
    _over(spr, win, kit.cover(P - lay.r_glass), GLASS, _unit(DX * kdome, DY * kdome, np.ones_like(DX)),
          0.7, 70.0, 0.0)

    # the Host
    r = lay.r_host
    bev = 0.35 * np.clip(1.0 - (r - P) / 0.9, 0.0, 1.0)
    hx, hy = ex * bev, ey * bev
    alb = np.broadcast_to(HOST, DX.shape + (3,)).copy()
    glow = np.broadcast_to(HOST_GLOW, DX.shape + (3,)).copy()
    if lay.s >= 64:
        # an impressed cross: a shallow groove, a little darker and less luminous
        lw = max(0.45, 0.075 * r)
        Yd = -DY
        dv = np.maximum(np.abs(DX) - lw, np.maximum(-0.62 * r - Yd, Yd - 0.66 * r))
        dh = np.maximum(np.abs(Yd + 0.14 * r) - lw, np.abs(DX) - 0.46 * r)
        g = kit.cover(np.minimum(dv, dh), 0.9)
        alb *= (1.0 - 0.10 * g)[..., None]
        glow *= (1.0 - 0.30 * g)[..., None]
        gn = kit.normal_from_height(-0.8 * g)
        hx, hy = hx + gn[..., 0], hy + gn[..., 1]
    _over(spr, win, kit.cover(P - r), alb, _unit(hx, hy, np.ones_like(hx)), 0.06, 8.0, 0.0, glow)


# =============================================================================
def render(h, seed=0):
    """The monstrance as a Sprite h pixels tall (any h >= 1; width about 0.56 h).

    Deterministic for (h, seed).  Anchors (sprite pixels, x right, y down):
    base (x, y) bottom centre of the foot, center (x, y) of the sun, host
    (x, y, r), cross (x, y) top of the cross, rays (K, 2) visible ray tips."""
    h = max(1, int(h))
    lay = _Layout(h)
    rng = np.random.default_rng(seed)
    spr = kit.Sprite.blank(h, lay.W)
    ys, xs = kit.grid(h, lay.W)
    dx = xs - F32(lay.cx)
    dyv = F32(lay.cy) - ys
    rho = np.hypot(dx, dyv)
    theta = np.arctan2(dx, dyv)                # 0 = straight up, clockwise
    _sun_rays(spr, lay, dx, dyv, rho, theta, rng)
    _cross(spr, lay, ys, xs)
    _turned_parts(spr, lay)
    _luna(spr, lay, dx, dyv, rho, theta)
    spr.anchors = {
        "base": (lay.cx, lay.y_base),
        "center": (lay.cx, lay.cy),
        "host": (lay.cx, lay.cy, lay.r_host),
        "cross": (lay.cx, lay.y_top),
        "rays": _ray_tips(lay),
    }
    return spr.check()
