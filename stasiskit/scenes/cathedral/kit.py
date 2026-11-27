"""Shared toolkit for the CATHEDRAL view.

The devotional subjects (Our Lady praying the rosary, the Divine Mercy image and
the monstrance) are each rendered into a `Sprite`: a small G-buffer (coverage,
albedo, normals, specular, emission, occlusion) plus named anchor points.  The
scene pastes the sprites into its own G-buffer and relights everything every
frame with flickering candles, so the subjects and the architecture share one
lighting model (`shade`).

Conventions (everyone relies on these):

* Pixel grids: x to the right, y DOWN (row index).  One terminal cell is two
  pixel rows, so pixels are roughly square.  Pixel centres are at index + 0.5.
* Normals: unit vectors in VIEW space, +x right, +y UP, +z toward the viewer.
  A flat surface facing the viewer is (0, 0, 1).
* Colours are linear-ish floats in 0..1; the canvas applies its own gamma/ramp.
* Distances returned by the 2D SDF helpers are in pixels, negative inside.

`preview(sprite, path)` lights a sprite with a standard candle rig, renders it
through the real ASCII and half-block renderers and writes one PNG, so you can
see exactly how it will look in the terminal.
"""

from dataclasses import dataclass, field

import numpy as np

from ..abyss import _blur

# tone mapping used by the cathedral view (and by the preview, so they match)
RAMP = " .,:;-=+*oO#%@"
BGFILL = 0.32
GAMMA = 0.85
FLOOR = 0.13          # dim haze stays a blank, coloured cell; glyphs mark candlelit surfaces
EDGES = 0.0

F32 = np.float32


def rgb(c):
    """'#rrggbb' / '#rgb' / (r, g, b) -> float32 array (3,)."""
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)
    return np.asarray(c, F32)


def grid(h, w, x0=0.0, y0=0.0):
    """Pixel-centre coordinate grids (ys, xs), each (h, w) float32."""
    ys, xs = np.mgrid[0:h, 0:w].astype(F32)
    return ys + F32(0.5 + y0), xs + F32(0.5 + x0)


def blur(img, r):
    """Approximately gaussian blur of radius ~r pixels (r < 1 returns img)."""
    r = int(r)
    if r < 1:
        return img
    return _blur(img, r)


def bilinear(tex, u, v):
    """Wrap-around bilinear sample of a 2D texture at texel coordinates (u, v)."""
    h, w = tex.shape[:2]
    u0, v0 = np.floor(u), np.floor(v)
    fu, fv = u - u0, v - v0
    i0, j0 = u0.astype(np.int64) % w, v0.astype(np.int64) % h
    i1, j1 = (i0 + 1) % w, (j0 + 1) % h
    if tex.ndim == 3:
        fu, fv = fu[..., None], fv[..., None]
    return ((tex[j0, i0] * (1 - fu) + tex[j0, i1] * fu) * (1 - fv)
            + (tex[j1, i0] * (1 - fu) + tex[j1, i1] * fu) * fv)


# =============================================================================
# 2D signed distance functions (pixels, negative inside)
# =============================================================================
def sd_circle(x, y, cx, cy, r):
    return np.hypot(x - cx, y - cy) - r


def sd_ellipse(x, y, cx, cy, rx, ry):
    """Approximate ellipse distance (accurate near the outline, fine for AA and shading)."""
    rx, ry = max(rx, 1e-3), max(ry, 1e-3)
    px, py = x - cx, y - cy
    k0 = np.hypot(px / rx, py / ry)
    k1 = np.hypot(px / (rx * rx), py / (ry * ry))
    return (k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)).astype(F32)


def sd_box(x, y, cx, cy, hw, hh, r=0.0):
    """Axis-aligned box with half sizes (hw, hh) and corner radius r."""
    qx = np.abs(x - cx) - hw + r
    qy = np.abs(y - cy) - hh + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_segment(x, y, ax, ay, bx, by):
    """Distance to segment AB, and the parameter h in 0..1 of the closest point."""
    pax, pay = x - ax, y - ay
    bax, bay = bx - ax, by - ay
    h = np.clip((pax * bax + pay * bay) / max(bax * bax + bay * bay, 1e-9), 0.0, 1.0)
    return np.hypot(pax - bax * h, pay - bay * h), h


def sd_capsule(x, y, ax, ay, bx, by, ra, rb=None):
    """Tapered capsule from A (radius ra) to B (radius rb)."""
    rb = ra if rb is None else rb
    d, h = sd_segment(x, y, ax, ay, bx, by)
    return d - (ra + (rb - ra) * h)


def sd_poly(x, y, pts):
    """Signed distance to a simple polygon given as [(x, y), ...] (any winding)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    d = np.full(np.shape(x), np.inf, F32)
    s = np.ones(np.shape(x), F32)
    for i in range(n):
        vix, viy = pts[i]
        vjx, vjy = pts[i - 1]
        dd, _ = sd_segment(x, y, vix, viy, vjx, vjy)
        d = np.minimum(d, dd)
        ex, ey = vjx - vix, vjy - viy
        c1 = y >= viy
        c2 = y < vjy
        c3 = ex * (y - viy) > ey * (x - vix)
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return (s * d).astype(F32)


def sd_arch(x, y, cx, spring_y, half_w, k=1.0, base_y=None):
    """Pointed (gothic) arch opening, axis-aligned, opening downward from its apex.

    half_w: half width; spring_y: row where the curve starts (y grows down);
    k: 1.0 = equilateral arch, 0.5 = round arch, >1 = lancet.  The sides run
    straight down from spring_y to base_y (default: infinitely)."""
    k = max(0.5, k)
    r = 2.0 * half_w * k
    off = r - half_w                      # centre of each arc, measured from the opposite side
    dx = x - cx
    top = np.maximum(np.hypot(dx - off, y - spring_y) - r, np.hypot(dx + off, y - spring_y) - r)
    sides = np.abs(dx) - half_w
    if base_y is not None:
        sides = np.maximum(sides, y - base_y)
    return np.where(y > spring_y, sides, top).astype(F32)


def smin(a, b, k):
    """Smooth union (polynomial), blend radius k."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def cover(d, aa=1.0):
    """Anti-aliased coverage (0..1) of an SDF in pixel units."""
    return np.clip(0.5 - d / max(aa, 1e-3), 0.0, 1.0).astype(F32)


def pillow(d, width):
    """Height profile of a rounded (pillowed) shape from its SDF: 0 at the edge,
    rising like a quarter circle to `width` at `width` pixels inside."""
    width = max(width, 1e-3)
    s = np.clip(-d / width, 0.0, 1.0)
    return (width * np.sqrt(np.clip(1.0 - (1.0 - s) ** 2, 0.0, 1.0))).astype(F32)


def normal_from_height(hgt, strength=1.0):
    """View-space unit normals (h, w, 3) of a height field (height toward the viewer,
    in pixels; y down in the array, +y UP in the normal)."""
    gy, gx = np.gradient(hgt.astype(F32) * F32(strength))
    n = np.stack([-gx, gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.astype(F32)


def normalize(v):
    v = np.asarray(v, F32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


# =============================================================================
# 3D helpers (for subjects that want true relief: raymarch an SDF orthographically)
# =============================================================================
def sd3_capsule(px, py, pz, a, b, ra, rb=None):
    rb = ra if rb is None else rb
    bax, bay, baz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    pax, pay, paz = px - a[0], py - a[1], pz - a[2]
    h = np.clip((pax * bax + pay * bay + paz * baz) / max(bax * bax + bay * bay + baz * baz, 1e-9), 0, 1)
    return np.sqrt((pax - bax * h) ** 2 + (pay - bay * h) ** 2 + (paz - baz * h) ** 2) - (ra + (rb - ra) * h)


def sd3_ellipsoid(px, py, pz, c, r):
    """Approximate ellipsoid distance (IQ's bound), centre c, radii r."""
    x, y, z = (px - c[0]) / r[0], (py - c[1]) / r[1], (pz - c[2]) / r[2]
    k0 = np.sqrt(x * x + y * y + z * z)
    k1 = np.sqrt((x / r[0]) ** 2 + (y / r[1]) ** 2 + (z / r[2]) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd3_box(px, py, pz, c, half, r=0.0):
    qx = np.abs(px - c[0]) - half[0] + r
    qy = np.abs(py - c[1]) - half[1] + r
    qz = np.abs(pz - c[2]) - half[2] + r
    return (np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
            + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r)


def march_ortho(sdf, xs, ys, z_front, z_back, steps=64, eps=0.002, relax=0.9):
    """Orthographic sphere tracing toward -z (the viewer looks along -z from +z).

    sdf(px, py, pz) -> distance (flat arrays).  xs, ys: flat arrays of ray
    positions in the SDF's units.  Returns (hit bool, z of the hit, normals (n, 3))
    with +z toward the viewer.  Rays that miss get z = z_back and normal (0, 0, 1).
    Normals come back in the SDF's own axes: if your SDF uses image rows (y DOWN),
    negate normals[:, 1] to get view space (+y up)."""
    n = xs.size
    z = np.full(n, z_front, F32)
    hit = np.zeros(n, bool)
    act = np.arange(n)
    for _ in range(steps):
        if act.size == 0:
            break
        d = sdf(xs[act], ys[act], z[act])
        z[act] = z[act] - d * relax
        h = d < eps
        hit[act[h]] = True
        act = act[~h & (z[act] > z_back)]
    nrm = np.zeros((n, 3), F32)
    nrm[:, 2] = 1.0
    if hit.any():
        i = np.flatnonzero(hit)
        e = eps * 2.0
        px, py, pz = xs[i], ys[i], z[i]
        gx = sdf(px + e, py, pz) - sdf(px - e, py, pz)
        gy = sdf(px, py + e, pz) - sdf(px, py - e, pz)
        gz = sdf(px, py, pz + e) - sdf(px, py, pz - e)
        g = np.stack([gx, gy, gz], -1)
        g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
        nrm[i] = g
    z[~hit] = z_back
    return hit, z, nrm


# =============================================================================
# Sprite: a small G-buffer for one subject
# =============================================================================
@dataclass
class Sprite:
    alpha: np.ndarray            # (h, w) coverage 0..1
    albedo: np.ndarray           # (h, w, 3) base colour
    normal: np.ndarray           # (h, w, 3) view-space unit normals
    spec: np.ndarray             # (h, w) specular strength: cloth 0, glaze 0.2, polished gold 1
    gloss: np.ndarray            # (h, w) Blinn-Phong exponent: 6 matte .. 150 polished metal
    metal: np.ndarray            # (h, w) 0 dielectric (white highlights) .. 1 metal (tinted, no diffuse)
    emissive: np.ndarray         # (h, w, 3) self-lit colour, added after lighting
    ao: np.ndarray               # (h, w) ambient occlusion, 1 = open
    anchors: dict = field(default_factory=dict)

    @property
    def h(self):
        return self.alpha.shape[0]

    @property
    def w(self):
        return self.alpha.shape[1]

    @classmethod
    def blank(cls, h, w):
        h, w = max(1, int(h)), max(1, int(w))
        nrm = np.zeros((h, w, 3), F32)
        nrm[..., 2] = 1.0
        return cls(alpha=np.zeros((h, w), F32), albedo=np.zeros((h, w, 3), F32), normal=nrm,
                   spec=np.zeros((h, w), F32), gloss=np.full((h, w), 16.0, F32), metal=np.zeros((h, w), F32),
                   emissive=np.zeros((h, w, 3), F32), ao=np.ones((h, w), F32), anchors={})

    def paint(self, cov, albedo, normal=None, spec=0.0, gloss=16.0, metal=0.0, emissive=None, ao=None):
        """Composite one layer OVER the sprite (painter's algorithm, back to front).

        cov: (h, w) coverage 0..1.  Every other argument is a scalar, a colour (3,)
        or a per-pixel array of the matching shape.  normal=None keeps (0, 0, 1)."""
        c = np.clip(np.asarray(cov, F32), 0.0, 1.0)
        c3 = c[..., None]
        ic, ic3 = 1.0 - c, 1.0 - c3
        self.albedo = (np.asarray(albedo, F32) * c3 + self.albedo * ic3).astype(F32)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0], F32)
        nn = np.asarray(normal, F32) * c3 + self.normal * ic3
        nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
        self.normal = nn.astype(F32)
        self.spec = (np.asarray(spec, F32) * c + self.spec * ic).astype(F32)
        self.gloss = (np.asarray(gloss, F32) * c + self.gloss * ic).astype(F32)
        self.metal = (np.asarray(metal, F32) * c + self.metal * ic).astype(F32)
        em = np.zeros(3, F32) if emissive is None else np.asarray(emissive, F32)
        self.emissive = (em * c3 + self.emissive * ic3).astype(F32)
        a = np.ones(1, F32) if ao is None else np.asarray(ao, F32)
        self.ao = (a * c + self.ao * ic).astype(F32)
        self.alpha = (c + self.alpha * ic).astype(F32)
        return self

    def add_emissive(self, cov, colour):
        """Additive glow (e.g. a halo of light) that does not occlude anything."""
        self.emissive = (self.emissive + np.asarray(cov, F32)[..., None] * np.asarray(colour, F32)).astype(F32)
        return self

    def check(self):
        """Raise if any buffer has the wrong shape or non-finite values."""
        h, w = self.h, self.w
        for name, shp in (("albedo", (h, w, 3)), ("normal", (h, w, 3)), ("spec", (h, w)), ("gloss", (h, w)),
                          ("metal", (h, w)), ("emissive", (h, w, 3)), ("ao", (h, w))):
            a = getattr(self, name)
            if a.shape != shp:
                raise ValueError(f"sprite.{name} has shape {a.shape}, expected {shp}")
            if not np.isfinite(a).all():
                raise ValueError(f"sprite.{name} has non-finite values")
        if not np.isfinite(self.alpha).all():
            raise ValueError("sprite.alpha has non-finite values")
        return self


# =============================================================================
# Lighting (the scene uses exactly this, so previews match the final look)
# =============================================================================
def shade(gb, L, col, V=(0.0, 0.0, 1.0), wrap=0.0):
    """Light contribution (h, w, 3) of one light on a G-buffer-like object.

    gb: anything with albedo/normal/spec/gloss/metal (a Sprite, or the scene's
    G-buffer).  L: unit direction(s) from the surface toward the light, (3,) or
    (h, w, 3).  col: light colour * intensity (3,) or (h, w, 3) (fold distance
    falloff and shadowing into it).  wrap > 0 softens the terminator (cloth)."""
    n = gb.normal
    L = np.asarray(L, F32)
    ndl = np.sum(n * L, -1)
    diff = np.clip((ndl + wrap) / (1.0 + wrap), 0.0, None)
    V = np.asarray(V, F32)
    Hh = L + V
    Hh = Hh / np.maximum(np.linalg.norm(Hh, axis=-1, keepdims=True), 1e-6)
    ndh = np.clip(np.sum(n * Hh, -1), 0.0, 1.0)
    sp = gb.spec * ndh ** gb.gloss * (ndl > 0)
    metal = gb.metal[..., None]
    base = gb.albedo * (1.0 - metal) * diff[..., None]
    spec_col = 1.0 + (gb.albedo - 1.0) * metal
    return ((base + spec_col * sp[..., None]) * np.asarray(col, F32)).astype(F32)


# standard "chapel" preview rig: (direction toward light, colour)
PREVIEW_RIG = [
    (normalize((-0.55, -0.20, 0.81)), rgb((1.25, 0.82, 0.45))),   # candle bank, front-left, low
    (normalize((0.60, -0.10, 0.79)), rgb((0.55, 0.36, 0.20))),    # candles, front-right
    (normalize((0.25, 0.85, -0.45)), rgb((0.10, 0.13, 0.22))),    # moonlit window, high behind
]
PREVIEW_AMBIENT = rgb((0.035, 0.032, 0.04))


def light_preview(spr, rig=None, ambient=None):
    """Lit RGB (h, w, 3) of a sprite under the preview rig, unpremultiplied."""
    rig = PREVIEW_RIG if rig is None else rig
    amb = PREVIEW_AMBIENT if ambient is None else np.asarray(ambient, F32)
    out = spr.albedo * amb * spr.ao[..., None]
    for L, col in rig:
        out = out + shade(spr, L, col, wrap=0.15) * (0.55 + 0.45 * spr.ao)[..., None]
    return (out + spr.emissive).astype(F32)


def preview(spr, path, bg=(0.012, 0.011, 0.014), rig=None, glow=0.0, pad_cells=2, scale=4):
    """Write one PNG: [ASCII render | half-block render | lit pixels x`scale`].

    The sprite is centred on a dark background with `pad_cells` cells of margin
    and rendered through the real Canvas, so this is what the terminal will show
    (glyph colours and density included).  glow > 0 adds bloom around emission."""
    from PIL import Image
    from ...canvas import Canvas, save_png
    import os
    import tempfile

    spr.check()
    lit = light_preview(spr, rig)
    em = spr.emissive
    a = spr.alpha[..., None]
    pw = spr.w + 2 * pad_cells
    ph = spr.h + 4 * pad_cells
    ph += ph % 2
    img = np.empty((ph, pw, 3), F32)
    img[:] = rgb(bg)
    y0, x0 = 2 * pad_cells, pad_cells
    sub = img[y0:y0 + spr.h, x0:x0 + spr.w]
    sub[:] = lit * a + sub * (1.0 - a)
    if glow > 0:
        gl = np.zeros_like(img)
        gl[y0:y0 + spr.h, x0:x0 + spr.w] = em * a
        img += blur(gl, max(1, int(min(ph, pw) / 30))) * glow
    np.clip(img, 0.0, 1.5, out=img)
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for mode in ("ascii", "blocks"):
            cv = Canvas(pw, ph // 2)
            cv.clear()
            cv.blit_pixels(img, mode, RAMP, BGFILL, gamma=GAMMA, floor=FLOOR, edges=EDGES)
            p = os.path.join(td, f"{mode}.png")
            save_png(cv, p)
            tiles.append(Image.open(p).convert("RGB"))
    raw = (np.clip(img, 0, 1) ** (1 / 1.6) * 255).astype(np.uint8)
    big = Image.fromarray(raw, "RGB").resize((pw * scale, ph * scale), Image.NEAREST)
    tiles.append(big)
    H = max(t.height for t in tiles)
    out = Image.new("RGB", (sum(t.width for t in tiles) + 16 * (len(tiles) - 1), H), (40, 40, 48))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 16
    out.save(path)
    return path
