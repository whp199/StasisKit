"""Light for the CATHEDRAL view: candle banks, the sanctuary lamp, the Host,
moonlight through the stained glass, and the incense haze that glows around them.

Candles are gathered into light groups (the altar's left and right candles,
each half of the votive bank, Our Lady's candles, the Divine Mercy shrine, the
lamp, the Host...).  For each group `bake()` computes, once per camera and
size, two maps with the group at full strength:

* surface light: kit.shade() summed over the group's flames with inverse-square
  falloff, a soft shadow (marched once per shadow "site" - the altar, Our
  Lady's altar, the Mercy shrine - and shared by the groups around it), and a
  little unshadowed "bounce" fill;
* air light: the light the incense haze scatters toward the eye along each view
  ray, integrated in closed form for every point light, with a falloff that
  gives each light a soft glow about a metre and a half wide instead of tinting
  the whole church (see airlight()).

Every frame the scene just weights these maps by the groups' flicker.

Moonlight is static: a surface or a point of air is moonlit when the ray toward
the moon leaves the building through a window opening (tested analytically,
through the arcade openings and the chancel arch on the way) - the light takes
the colour of the glass it passed through, so the floor gets coloured pools and
the air gets shafts.  Surfaces also get a soft shadow toward the moon, and the
haze everywhere takes a faint blue that deepens with distance (moon_haze()), so
the far reaches of the church fade into blue-black.
"""

from dataclasses import dataclass

import numpy as np

from . import kit, nave

F32 = np.float32

CANDLE = np.array((1.0, 0.66, 0.36), F32)          # candle flame colour
CUP_TINT = np.array([(1.0, 0.22, 0.13), (0.32, 0.42, 1.0), (1.0, 0.62, 0.22)], F32)   # red, blue, amber glass
MOON_DIR = kit.normalize((0.62, 0.50, 0.60))         # toward the moon (south-east, high)
MOON_COL = np.array((0.46, 0.56, 0.90), F32)
AMBIENT = np.array((0.030, 0.040, 0.085), F32)     # sky glow from all the windows
HAZE_COL = np.array((0.020, 0.028, 0.060), F32)    # the moonlit haze, seen through a long way of it


@dataclass
class PointLight:
    pos: tuple
    col: np.ndarray            # colour * power (irradiance at 1 m)
    r0: float = 0.25           # softening radius of the inverse-square falloff


@dataclass
class LightGroup:
    name: str
    lights: list
    site: str = None           # shadow site (see SITES); None: unshadowed
    fill: float = 0.12         # unshadowed bounce fill, fraction of the direct light
    haze: float = 1.0          # how strongly the group lights the incense haze
    wrap: float = 0.2
    boost: float = 1.0         # extra light on the devotional subjects (a painter's key light)
    falls: float = 0.0         # > 0: the light only lands on upward-facing surfaces (it "falls")
    only: int = None           # light only the subject with this G-buffer object id (a key light)...
    also: tuple = ()           # ...and surfaces of these materials (Our Lady's niche behind her)


# light power of each kind of flame (CANDLE x this = irradiance at 1 m), tuned by eye
POWER = dict(altar=0.4, candelabra=0.45, votive=0.018, mary=0.17, mercy=0.16, tealight=0.04, lamp=0.26, host=0.3)


def layout(host=None):
    """The church's light groups (subject-derived ones - the Mercy rays, the key lights on
    the images - are added by the scene).  host: world position of the Host, if the
    monstrance is placed."""
    fl = nave.flames()
    by = {}
    for pos, size, grp, cup in fl:
        by.setdefault(grp, []).append((pos, size, cup))
    groups = []
    for name in ("altar_l", "altar_r"):
        lights = [PointLight(p, CANDLE * POWER["altar" if s < 0.052 else "candelabra"]) for p, s, _ in by[name]]
        groups.append(LightGroup(name, lights, site="altar", boost=1.8))
    cups, side, col = nave.votive_cups()
    for s, name in ((0, "votive_l"), (1, "votive_r")):
        lights = []
        for k in range(nave.VOTIVE_TIERS):
            sel = [i for i in range(len(cups)) if side[i] == s and int(round((cups[i][1] - nave.VOTIVE_Y0 - 0.05) / nave.VOTIVE_DY)) == k]
            if not sel:
                continue
            p = cups[sel].mean(0) + (0, 0.03, 0)
            # light escapes through the open tops (warm) and through the coloured glass;
            # a wide softening radius, as the cups of a tier spread over half a metre
            c = sum(0.55 + 0.45 * CUP_TINT[col[i]] for i in sel) * CANDLE * POWER["votive"]
            lights.append(PointLight(tuple(p), c.astype(F32), 0.45))
        groups.append(LightGroup(name, lights, site="mary", fill=0.14))
    groups.append(LightGroup("mary", [PointLight(p, CANDLE * POWER["mary"]) for p, _, _ in by["mary"]], site="mary",
                             boost=1.2))
    mer = [PointLight(p, CANDLE * POWER["mercy"]) for p, s, _ in by["mercy"] if s > 0.03]
    tl = [p for p, s, _ in by["mercy"] if s <= 0.03]
    if tl:                                            # the row of small votive lights, as one
        mer.append(PointLight(tuple(np.mean(np.array(tl, F32), 0)), CANDLE * POWER["tealight"] * len(tl), 0.3))
    groups.append(LightGroup("mercy", mer, site="mercy", boost=1.2))
    lp = (nave.LAMP[0], nave.LAMP[1] + 0.02, nave.LAMP[2])
    groups.append(LightGroup("lamp", [PointLight(lp, np.array((1.0, 0.06, 0.035), F32) * POWER["lamp"], 0.2)], haze=1.4))
    if host is not None:
        groups.append(LightGroup("host", [PointLight(tuple(host), np.array((1.0, 0.93, 0.78), F32) * POWER["host"], 0.3)],
                                 site="altar", fill=0.05, haze=0.8))
    return groups


# =============================================================================
# shadows
# =============================================================================
def soft_shadow(P, N, target, radius, steps=18):
    """Soft visibility (n,) from points P (n, 3) of a light `radius` metres across centred
    on `target` (a whole bank of candles), marched through the building's occluders.

    Penumbra estimate in the spirit of IQ's soft shadows, scaled to the light's size: at
    distance t along a ray of length D the light's cone is radius * t / D wide, so an
    occluder passing at clearance h hides about clamp(h * D / (radius * t)) of it.  The
    march stops `radius` short of the centre: what lies inside the bank is not an
    obstacle to all of its candles at once."""
    n = P.shape[0]
    res = np.ones(n, F32)
    if n == 0:
        return res
    tgt = np.asarray(target, F32)
    v = tgt[None, :] - P
    dist = np.sqrt(np.sum(v * v, 1))
    d = v / np.maximum(dist, 1e-6)[:, None]
    k = dist / radius
    o = P + N * 0.03
    tmax = dist - radius
    t = np.full(n, 0.04, F32)
    act = np.flatnonzero(tmax > 0.1)
    for _ in range(steps):
        if act.size == 0:
            break
        ta = t[act]
        q = o[act] + d[act] * ta[:, None]
        h = nave.WORLD.shadow(q[:, 0], q[:, 1], q[:, 2])
        res[act] = np.minimum(res[act], k[act] * h / np.maximum(ta, 1e-3))
        ta = ta + np.clip(h, 0.05, 0.9)
        t[act] = ta
        alive = (ta < tmax[act]) & (res[act] > 0.02)
        act = act[alive]
    r = np.clip(res, 0.0, 1.0)
    return (r * r * (3 - 2 * r)).astype(F32)


# shadow sites: one soft-shadow march serves every group around it - (centre, radius of
# the cluster of flames it stands for)
SITES = {
    "altar": ((0.0, 2.4, 3.0), 0.7),
    "mary": ((nave.XM, 1.15, nave.ZE - 0.6), 0.55),
    "mercy": ((nave.XR, 1.4, nave.MERCY_CANDLE_Z), 0.5),
}


def site_visibility(gb, site, half=None):
    """Soft visibility (H*W,) of a shadow site, marched at half resolution on big screens."""
    H, W = gb.H, gb.W
    centre, radius = SITES[site]
    if half is None:
        half = H * W > 8000
    P = gb.pos
    N = gb.nworld
    if half:
        Ps, Ns = P[::2, ::2], N[::2, ::2]
        v = soft_shadow(Ps.reshape(-1, 3), Ns.reshape(-1, 3), centre, radius).reshape(Ps.shape[:2])
        v = np.repeat(np.repeat(v, 2, 0), 2, 1)[:H, :W]
        return v.reshape(-1)
    return soft_shadow(P.reshape(-1, 3), N.reshape(-1, 3), centre, radius)


# =============================================================================
# baking one group
# =============================================================================
# Flames are tiny, very bright sources: on polished surfaces their reflection is a
# glint far brighter than a point light's irradiance suggests.  Varnished oak, worn
# flagstones and marble get their highlights scaled up accordingly.
POLISH = {nave.M_WOOD: 7.0, nave.M_FLOOR: 3.0, nave.M_CHECK: 2.0, nave.M_MARBLE: 1.6}
WAX_OTHER = 0.35          # how much of the other flames' light a candle's wax shows (see bake)


class _Flat:
    """The bits of a G-buffer kit.shade() needs, flattened to (n, ...)."""

    def __init__(self, gb, idx=None):
        def f(a, c=None):
            a = a.reshape(-1, c) if c else a.reshape(-1)
            return a if idx is None else a[idx]
        self.normal = f(gb.normal, 3)
        self.albedo = f(gb.albedo, 3)
        self.spec = f(gb.spec).copy()
        self.gloss = f(gb.gloss)
        self.metal = f(gb.metal)
        mat = f(gb.mat)
        for m, k in POLISH.items():
            self.spec[mat == m] *= k


def view_rays(cam):
    dirs, _ = cam.rays()
    return dirs


def airlight(eye, dirs, tmax, lights, r_air=0.35, reach=1.4):
    """Haze in-scatter (n, 3) along rays eye + s*dirs, s in [0, tmax], from point lights.

    The light scattered toward the eye falls off as reach / (d^2 + r^2)^(3/2) with the
    distance d from the light (as if the haze it can light were thickest around it), so
    each light wears a soft glow about `reach` metres wide instead of tinting the whole
    church.  Along a ray that is (with u = s - the ray's closest approach to the light,
    and b its distance there) the integral of reach / (u^2 + b^2)^(3/2), in closed form:
    reach * [u / (b^2 sqrt(u^2 + b^2))]; pi/2 scales it to 1/d^2 light at d = reach."""
    out = np.zeros((dirs.shape[0], 3), F32)
    k = F32(0.5 * np.pi * reach)
    for L in lights:
        lv = np.asarray(L.pos, F32) - eye
        sc = dirs @ lv
        b2 = np.maximum(float(lv @ lv) - sc * sc, 0.0) + F32(r_air * r_air)
        u1 = tmax - sc
        a = (u1 / np.sqrt(u1 * u1 + b2) + sc / np.sqrt(sc * sc + b2)) / b2 * k
        out += a[:, None] * np.asarray(L.col, F32)[None, :]
    return out


def bake(gb, grp, dirs, vis=None, cutoff=0.007):
    """Surface light and air light (each (H, W, 3)) of one group at full strength."""
    H, W = gb.H, gb.W
    cam = gb.cam
    P = gb.pos.reshape(-1, 3)
    Nw = gb.nworld.reshape(-1, 3)
    n = P.shape[0]
    # pixels this group can noticeably light (skip the rest: cheaper shadows)
    reach = np.zeros(n, F32)
    for L in grp.lights:
        v = np.asarray(L.pos, F32) - P
        reach = np.maximum(reach, float(np.max(L.col)) / (np.sum(v * v, 1) + L.r0 ** 2))
    if grp.only is not None:
        reach = reach * ((gb.obj.reshape(-1) == grp.only) | np.isin(gb.mat.reshape(-1), grp.also))
    idx = np.flatnonzero(reach > cutoff)
    surf = np.zeros((n, 3), F32)
    if idx.size:
        fg = _Flat(gb, idx)
        Pi = P[idx]
        view = gb.view.reshape(-1, 3)[idx]
        direct = np.zeros((idx.size, 3), F32)
        fill = np.zeros((idx.size, 3), F32)
        glow = np.zeros((idx.size, 3), F32)
        wax = gb.mat.reshape(-1)[idx] == nave.M_WAX
        for L in grp.lights:
            v = np.asarray(L.pos, F32) - Pi
            d2 = np.sum(v * v, 1)
            e = np.asarray(L.col, F32)[None, :] / (d2 + L.r0 ** 2)[:, None]
            # a flame does not floodlight its own candle: the wax just below it glows
            # (light scattered inside the wax), fading a few centimetres down, and a
            # fainter warmth carries on down the candle
            own = wax & (v[:, 0] ** 2 + v[:, 2] ** 2 < 0.05 ** 2) & (v[:, 1] > -0.02)
            if own.any():
                e[own] = 0.0
                dy = np.maximum(v[own, 1], 0.0)
                glow[own] += np.asarray(L.col, F32) * (1.5 * np.exp(-dy / 0.04) + 0.22 * np.exp(-dy / 0.25))[:, None]
            # ...and the neighbouring flames, a hand's breadth away, would light the candles
            # into white columns as bright as the Host: translucent wax takes their light in
            # rather than reflecting it, so it reaches the surface damped
            e[wax & ~own] *= WAX_OTHER
            Lv = cam.to_view(v / np.sqrt(np.maximum(d2, 1e-8))[:, None])
            direct += kit.shade(fg, Lv, e, V=view, wrap=grp.wrap)
            fill += e
        if vis is not None:
            direct *= vis[idx][:, None]
        if grp.falls > 0:
            up = np.clip(Nw[idx][:, 1], 0.0, 1.0) ** grp.falls
            direct *= up[:, None]
            fill *= up[:, None]
        ao = gb.ao.reshape(-1)[idx]
        surf[idx] = direct * (0.55 + 0.45 * ao)[:, None] + fill * fg.albedo * (grp.fill * ao)[:, None] + glow
        if grp.boost != 1.0:
            subj = gb.mat.reshape(-1)[idx] >= 40              # subjects.M_SUBJECT: a pasted image
            surf[idx[subj]] *= grp.boost
    air = airlight(cam.eye, dirs, np.minimum(gb.dist.reshape(-1), 30.0), grp.lights) * grp.haze
    return surf.reshape(H, W, 3), air.reshape(H, W, 3)


# =============================================================================
# moonlight through the windows
# =============================================================================
def _in_arch(u, y, half, y0, spring, k):
    return (_arch_d(u, y, half, spring, k) < 0) & (y > y0)


def _arch_d(u, y, half, spring, k):
    return nave._arch(u, y, half, spring, k)


def moon_aperture(P):
    """Colour (n, 3) of the moonlight reaching points P (n, 3) through the south and
    east windows (0 where the way to the moon is blocked by masonry)."""
    M = MOON_DIR
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    n = P.shape[0]
    out = np.zeros((n, 3), F32)
    sd = np.zeros(n, F32)
    hit = np.zeros(n, bool)

    def cross(plane_x):
        s = (plane_x - x) / M[0]
        return s, y + s * M[1], z + s * M[2]

    def through_arcade(side_x, s_target):
        """True where the segment to the target either never crosses the arcade line at
        side_x or crosses it inside an arcade opening (clear of the piers)."""
        s, cy, cz = cross(side_x)
        crosses = (s > 0) & (s < s_target)
        zb = nave._rep(cz + nave.BAY / 2, nave.BAY)
        ok = (_arch_d(zb, cy, nave.BAY / 2 - 0.62, nave.YAR, nave.KAR) < 0) & (cz < -0.5) & (cy > 0)
        return ~crosses | ok

    def through_chancel(s_target):
        s = (0.0 - z) / M[2]
        cx = x + s * M[0]
        cy = y + s * M[1]
        crosses = (s > 0) & (s < s_target)
        ok = _arch_d(cx, cy, nave.XN - 0.15, nave.YC, nave.KC) < 0
        return ~crosses | ok

    # south aisle windows (one per bay)
    s, cy, cz = cross(nave.XW + 0.38)
    zb = nave._rep(cz + nave.BAY / 2, nave.BAY)
    win = _in_arch(zb, cy, 0.62, 2.1, 3.75, 1.0) & (cz < nave.ZE - 0.9) & (s > 0)
    win &= through_arcade(-nave.XP, s) & through_arcade(nave.XP, s) & (x < nave.XW)
    if win.any():
        c = nave.glass_tint(zb[win] / 0.62, cy[win] - 2.1, np.floor((cz[win] + nave.BAY / 2) / nave.BAY + 0.5) * 2 + 1)
        out[win] = c * 0.45 + nave.GLASS_COLS[4] * 0.55
        sd[win] = s[win]
        hit |= win
    # south clerestory
    s, cy, cz = cross(nave.XN + 0.38)
    zb = nave._rep(cz + nave.BAY / 2, nave.BAY)
    win = _in_arch(zb, cy, 0.62, 7.6, 9.0, 1.0) & (cz < -0.7) & (s > 0) & ~hit
    win &= through_arcade(-nave.XP, s) & (x < nave.XN + 0.3)
    if win.any():
        c = nave.glass_tint(zb[win] / 0.62, cy[win] - 7.6, 40 + np.floor((cz[win] + nave.BAY / 2) / nave.BAY + 0.5) * 2 + 1)
        out[win] = c * 0.8
        sd[win] = s[win]
        hit |= win
    # south windows of the sanctuary
    zs = cz - 2.75 - 2.5 * np.clip(np.floor((cz - 2.75) / 2.5 + 0.5), 0, 1)
    win = _in_arch(zs, cy, 0.5, 3.0, 7.8, 1.1) & (cz > 0.9) & (cz < nave.ZA - 0.2) & (s > 0) & ~hit
    win &= through_chancel(s) & (x < nave.XN + 0.3)
    if win.any():
        c = nave.glass_tint(zs[win] / 0.5, cy[win] - 3.0, 80 + (cz[win] > 4.0) * 2 + 1)
        out[win] = c * 0.8
        sd[win] = s[win]
        hit |= win
    # east (apse) windows: the central light and the south-east one
    s = (nave.ZA + nave.XN + 0.38 - z) / M[2]
    cx, cy = x + s * M[0], y + s * M[1]
    win = _in_arch(cx, cy, 0.58, 3.0, 7.9, 1.1) & (s > 0) & ~hit & through_chancel(s)
    if win.any():
        c = nave.glass_tint(cx[win] / 0.58, cy[win] - 3.0, np.full(int(win.sum()), 90.0))
        out[win] = c * 0.8
        sd[win] = s[win]
        hit |= win
    # the south-east face: n = (0.7071, 0.7071) in (x, z - ZA)
    nx, nz = 0.70711, 0.70711
    den = M[0] * nx + M[2] * nz
    s = ((nave.XN + 0.38) - (x * nx + (z - nave.ZA) * nz)) / den
    qx, qy, qz = x + s * M[0], y + s * M[1], z + s * M[2]
    u = (qx - (qz - nave.ZA)) * 0.70711
    win = _in_arch(u, qy, 0.5, 3.0, 7.8, 1.1) & (s > 0) & (qz > nave.ZA) & ~hit & through_chancel(s)
    if win.any():
        c = nave.glass_tint(u[win] / 0.5, qy[win] - 3.0, np.full(int(win.sum()), 92.0))
        out[win] = c * 0.8
        sd[win] = s[win]
    return out, sd


def bake_moon(gb, dirs, power=0.9, beam_samples=18, beam=0.065):
    """(surface light, air light) of the moon, both (H, W, 3), static."""
    H, W = gb.H, gb.W
    cam = gb.cam
    P = gb.pos.reshape(-1, 3)
    Nw = gb.nworld.reshape(-1, 3)
    n = P.shape[0]
    ap, sd = moon_aperture(P)
    facing = (Nw @ MOON_DIR) > 0.02
    idx = np.flatnonzero(facing & (ap.max(1) > 0.01))
    surf = np.zeros((n, 3), F32)
    if idx.size:
        fg = _Flat(gb, idx)
        Lv = np.broadcast_to(cam.to_view(MOON_DIR), (idx.size, 3))
        # march toward the moon, stopping just inside the window it shines through
        vis = soft_shadow_dir(P[idx], Nw[idx], MOON_DIR, np.maximum(sd[idx] - 0.5, 0.1))
        col = ap[idx] * MOON_COL[None, :] * power * vis[:, None]
        view = gb.view.reshape(-1, 3)[idx]
        surf[idx] = kit.shade(fg, Lv, col, V=view, wrap=0.0) * (0.6 + 0.4 * gb.ao.reshape(-1)[idx])[:, None]
    # shafts of moonlight in the haze: sample the aperture along each view ray
    # (at half resolution on big screens - the shafts are soft)
    half = H * W > 8000
    dd = dirs.reshape(H, W, 3)
    tm = np.minimum(gb.dist, 26.0)
    if half:
        dd, tm = dd[::2, ::2], tm[::2, ::2]
    hh, ww = tm.shape
    dd = dd.reshape(-1, 3)
    tm = tm.reshape(-1)
    air = np.zeros((dd.shape[0], 3), F32)
    eye = cam.eye
    # per-pixel jittered sample offsets turn banding into fine noise, which the blur removes
    jit = np.random.default_rng(17).random(dd.shape[0]).astype(F32)
    for k in range(beam_samples):
        sk = tm * ((k + jit) / beam_samples)
        q = eye[None, :] + dd * sk[:, None]
        air += moon_aperture(q)[0]
    air *= (tm / beam_samples)[:, None] * MOON_COL[None, :] * (power * beam)
    air = kit.blur(air.reshape(hh, ww, 3), max(1, int(min(hh, ww) / 45)))
    if half:
        air = np.repeat(np.repeat(air, 2, 0), 2, 1)[:H, :W]
    return surf.reshape(H, W, 3), air.astype(F32)


def soft_shadow_dir(P, N, direction, maxd, soft=8.0, steps=20):
    """Soft visibility along a fixed direction (the moon) for at most maxd metres,
    against the piers, pews and altars only (walls are the aperture test's job)."""
    n = P.shape[0]
    res = np.ones(n, F32)
    d = np.asarray(direction, F32)
    o = P + N * 0.03
    t = np.full(n, 0.05, F32)
    act = np.arange(n)
    for _ in range(steps):
        if act.size == 0:
            break
        ta = t[act]
        q = o[act] + d[None, :] * ta[:, None]
        h = nave.WORLD.shadow(q[:, 0], q[:, 1], q[:, 2], shell=False)
        res[act] = np.minimum(res[act], soft * h / np.maximum(ta, 1e-3))
        ta = ta + np.clip(h, 0.05, 1.0)
        t[act] = ta
        act = act[(ta < maxd[act]) & (res[act] > 0.02)]
    r = np.clip(res, 0, 1)
    return (r * r * (3 - 2 * r)).astype(F32)


def moon_haze(gb, depth=16.0):
    """The faint blue of the moonlit incense haze (H, W, 3): it thickens with the length of
    air each pixel looks through, so the far end of the church fades into blue-black."""
    return ((1.0 - np.exp(-np.minimum(gb.dist, 40.0) / depth))[..., None] * HAZE_COL).astype(F32)


def ambient(gb):
    """Sky glow from all the windows: dim, blue, occluded."""
    return (gb.albedo * AMBIENT[None, None, :] * (0.35 + 0.65 * gb.ao[..., None])
            * (0.75 + 0.25 * np.clip(gb.nworld[..., 1:2], 0, 1))).astype(F32)
