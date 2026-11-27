"""The three devotional subjects as camera-facing cards in the church.

`sprite(name, h)` renders Our Lady ("mary"), the Divine Mercy image ("mercy")
or the monstrance ("monstrance") at h pixels tall with its own module, and
falls back to a simple stand-in drawn here if that module is missing, fails,
or returns something malformed - so the view always works.

`Card` places a sprite in the scene: the world point its "base" anchor stands
on is projected to the screen, the world height gives its size in pixels, and
`paste()` writes it into the G-buffer (z-tested against the building) with
world positions on a camera-facing plane, so the same lights shine on it.
Anchors (sprite pixels) are mapped to screen pixels by `to_screen()`.
"""

import importlib
import math
import traceback
from collections import OrderedDict

import numpy as np

from . import kit, nave

F32 = np.float32
M_SUBJECT = 40                      # material id used for subject pixels in the G-buffer
SUBJECT_IDS = {"mary": 2001, "mercy": 2002, "monstrance": 2003}

# subjects hung flat on a wall: their card keeps just clear of the wall even when the
# camera sees the wall obliquely (a camera-facing card would cut into it)
FLAT = {"mercy"}
# where each subject stands (world base point) and how tall it is (metres)
PLACES = {
    "mary": (nave.MARY_BASE, 1.78),
    "mercy": (nave.MERCY_BASE, 2.36),
    "monstrance": ((0.0, nave.THRONE_Y, nave.MON_Z), 1.18),
}
REQUIRED = {
    "mary": ("base", "beads", "bead_kind", "halo", "face"),
    "mercy": ("base", "heart", "ray_red", "ray_pale", "inscription", "face"),
    "monstrance": ("base", "host", "rays"),
}

_cache = OrderedDict()
ERRORS = OrderedDict()              # (name, h) -> traceback of a subject module that failed (a stand-in is shown)


def sprite(name, h, seed=0):
    """(Sprite, real?) - the subject module's sprite, or a stand-in (the reason is kept
    in ERRORS, for the previews and the tests)."""
    key = (name, int(h), seed)
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    real = True
    try:
        mod = importlib.import_module(f"{__package__}.{name}")
        spr = mod.render(int(h), seed=seed)
        spr.check()
        _validate(name, spr)
    except Exception:
        ERRORS[(name, int(h))] = traceback.format_exc()
        while len(ERRORS) > 12:
            ERRORS.popitem(last=False)
        spr = STAND_INS[name](int(h))
        real = False
    _cache[key] = (spr, real)
    while len(_cache) > 12:
        _cache.popitem(last=False)
    return spr, real


def _validate(name, spr):
    """Raise unless the anchors the scene relies on are present and well formed; turn
    list-valued anchors into arrays."""
    a = spr.anchors
    if spr.h < 2 or spr.w < 2:
        raise ValueError("empty sprite")
    missing = [k for k in REQUIRED[name] if k not in a]
    if missing:
        raise ValueError(f"missing anchors {missing}")
    for k in ("beads", "stars", "rays"):
        if k in a:
            a[k] = np.asarray(a[k], F32).reshape(-1, 2)
    for k in ("bead_kind", "bead_r"):
        if k in a:
            a[k] = np.asarray(a[k]).reshape(-1)
    for k in ("ray_red", "ray_pale"):
        if k in a:
            a[k] = np.asarray(a[k], F32)
            if a[k].shape != (spr.h, spr.w) or not np.isfinite(a[k]).all():
                raise ValueError(f"{k} mask must be (h, w)")
    if name == "mary":
        if len(a["beads"]) < 7 or len(a["bead_kind"]) != len(a["beads"]):
            raise ValueError("beads / bead_kind mismatch")
        if "bead_r" in a and len(a["bead_r"]) != len(a["beads"]):
            a.pop("bead_r")
    if name == "monstrance" and (len(a["host"]) < 3 or len(a["rays"]) == 0):
        raise ValueError("host must be (x, y, r) and rays non-empty")
    if name == "mercy" and len(a["inscription"]) != 4:
        raise ValueError("inscription must be (x0, y0, x1, y1)")
    pts = [a["base"]] + ([a["face"]] if "face" in a else [])
    if not np.isfinite(np.asarray(pts, F32)).all():
        raise ValueError("non-finite anchors")


# =============================================================================
# stand-ins (used until / unless the real subject modules work)
# =============================================================================
def _rosary_points(cx, hands_y, loop_bottom, half_w, pend_len):
    """61 bead positions in prayer order, as the real statue gives them: crucifix,
    the pendant up to the medal, the loop of five decades round from the medal, and
    the medal last."""
    kinds = []
    pts = []
    # pendant (from the crucifix upward): crucifix, large, 3 small, large
    ys = np.linspace(loop_bottom + pend_len, loop_bottom + pend_len * 0.12, 6)
    for i, k in enumerate((0, 1, 2, 2, 2, 1)):
        pts.append((cx, ys[i]))
        kinds.append(k)
    # loop: up the left side, over the hands, down the right side (54 beads)
    seq = []
    for d in range(5):
        seq += [2] * 10
        if d < 4:
            seq.append(1)
    n = len(seq)
    for i, k in enumerate(seq):
        a = (i + 1) / (n + 1)                      # 0..1 round the loop, starting at the medal
        th = math.pi * 0.5 + a * 2 * math.pi       # from the bottom, clockwise as seen
        x = cx + half_w * math.cos(th)
        y = (loop_bottom + hands_y) / 2 + (loop_bottom - hands_y) / 2 * math.sin(th)
        pts.append((x, y))
        kinds.append(k)
    pts.append((cx, loop_bottom))                    # the medal, where the loop closes
    kinds.append(3)
    return np.array(pts, F32), np.array(kinds, np.int8)


def stand_in_mary(h):
    """Our Lady, white-robed with a blue mantle, hands joined, a crown of twelve stars,
    the rosary hanging from her hands."""
    h = max(8, h)
    w = max(6, int(round(h * 0.42)))
    s = kit.Sprite.blank(h, w)
    ys, xs = kit.grid(h, w)
    cx = w / 2
    head_y, head_r = h * 0.15, h * 0.058
    # plinth (a small cloud-like base)
    d_base = kit.sd_ellipse(xs, ys, cx, h * 0.955, w * 0.34, h * 0.045)
    s.paint(kit.cover(d_base), kit.rgb("#d8d2c4"), kit.normal_from_height(kit.pillow(d_base, 3)), spec=0.15, gloss=20)
    # white robe: a tapered column
    robe = kit.sd_poly(xs, ys, [(cx - w * 0.12, h * 0.22), (cx + w * 0.12, h * 0.22),
                                (cx + w * 0.30, h * 0.93), (cx - w * 0.30, h * 0.93)])
    s.paint(kit.cover(robe), kit.rgb("#ece6da"), kit.normal_from_height(kit.pillow(robe, w * 0.16)), spec=0.2, gloss=24,
            ao=np.clip(0.75 + 0.25 * (1 - ys / h), 0, 1))
    # blue mantle over the head and shoulders, open at the front
    mant = kit.sd_poly(xs, ys, [(cx, h * 0.07), (cx + w * 0.22, h * 0.2), (cx + w * 0.36, h * 0.9),
                                (cx + w * 0.16, h * 0.92), (cx + w * 0.08, h * 0.32), (cx - w * 0.08, h * 0.32),
                                (cx - w * 0.16, h * 0.92), (cx - w * 0.36, h * 0.9), (cx - w * 0.22, h * 0.2)])
    s.paint(kit.cover(mant), kit.rgb("#2c4f9e"), kit.normal_from_height(kit.pillow(mant, w * 0.12)), spec=0.18, gloss=20)
    # face
    face = kit.sd_ellipse(xs, ys, cx, head_y, head_r * 0.82, head_r)
    s.paint(kit.cover(face), kit.rgb("#f0dcc8"), kit.normal_from_height(kit.pillow(face, head_r)), spec=0.15, gloss=22)
    # hands joined in prayer at the breast
    hy = h * 0.37
    hands = kit.sd_ellipse(xs, ys, cx, hy, w * 0.05, h * 0.045)
    s.paint(kit.cover(hands), kit.rgb("#f0dcc8"), kit.normal_from_height(kit.pillow(hands, 2)), spec=0.15)
    # rosary hanging from the hands
    beads, kinds = _rosary_points(cx, hy + h * 0.01, h * 0.58, w * 0.11, h * 0.16)
    br = np.where(kinds == 1, 1.0, 0.7) * max(0.45, h / 150)
    br = np.where(kinds == 0, max(0.6, h / 110), br)
    br = np.where(kinds == 3, max(0.6, h / 120), br)
    d_b = np.full((h, w), 1e9, F32)
    for (bx, by), r in zip(beads, br):
        d_b = np.minimum(d_b, kit.sd_circle(xs, ys, bx, by, r))
    if h >= 40:
        s.paint(kit.cover(d_b, 0.8), kit.rgb("#d8c8a8"), spec=0.5, gloss=40)
    # crown of twelve stars around the head
    stars = []
    for k in range(12):
        a = math.pi * (0.1 + 0.8 * k / 11)
        stars.append((cx - math.cos(a) * head_r * 2.0, head_y - math.sin(a) * head_r * 1.7))
    stars = np.array(stars, F32)
    if h >= 30:
        d_s = np.full((h, w), 1e9, F32)
        for sx, sy in stars:
            d_s = np.minimum(d_s, kit.sd_circle(xs, ys, sx, sy, max(0.5, h / 160)))
        s.add_emissive(kit.cover(d_s, 0.8), kit.rgb("#ffe7a0") * 0.5)
        s.paint(kit.cover(d_s, 0.8), kit.rgb("#ffd060"), spec=1.0, gloss=60, metal=1.0)
    s.anchors.update(base=(cx, h - 0.5), beads=beads, bead_kind=kinds, bead_r=br.astype(F32), stars=stars,
                     halo=(cx, head_y - head_r * 0.5, head_r * 2.2), face=(cx, head_y), hands=(cx, hy))
    return s.check()


def stand_in_mercy(h):
    """The Divine Mercy image in a gilded frame: the risen Christ in white, his right
    hand raised in blessing, the pale and red rays from his heart, and the
    inscription band below."""
    h = max(10, h)
    w = max(6, int(round(h * 0.52)))
    s = kit.Sprite.blank(h, w)
    ys, xs = kit.grid(h, w)
    fr = max(1.0, w * 0.07)
    outer = kit.sd_box(xs, ys, w / 2, h / 2, w / 2 - 0.3, h / 2 - 0.3)
    inner = kit.sd_box(xs, ys, w / 2, h / 2, w / 2 - fr, h / 2 - fr)
    frame = np.maximum(outer, -inner)
    hf = kit.pillow(frame, fr * 0.6) + 0.0
    s.paint(kit.cover(outer), kit.rgb("#141414"))
    s.paint(kit.cover(frame), kit.rgb("#d4a24a"), kit.normal_from_height(hf, 1.0), spec=1.0, gloss=50, metal=1.0)
    cx = w / 2
    hy = h * 0.46                       # heart
    # rays fanning down from the heart: red on the viewer's left, pale on the right (as
    # in the Vilnius image)
    ang = np.arctan2(xs - cx, ys - hy)
    dist = np.hypot(xs - cx, ys - hy)
    below = (ys > hy) & (kit.cover(inner) > 0.5)
    red = below & (ang < -0.05) & (ang > -0.62)
    pale = below & (ang > 0.05) & (ang < 0.62)
    fade = np.clip(1.15 - dist / (h * 0.55), 0, 1)
    s.paint(pale * fade * 0.9, kit.rgb("#dfe8ff"), emissive=kit.rgb("#cfe0ff") * 0.25)
    s.paint(red * fade * 0.9, kit.rgb("#d2202a"), emissive=kit.rgb("#ff3030") * 0.2)
    # figure in a white robe
    robe = kit.sd_poly(xs, ys, [(cx - w * 0.1, h * 0.22), (cx + w * 0.1, h * 0.22), (cx + w * 0.2, h * 0.84),
                                (cx - w * 0.2, h * 0.84)])
    s.paint(kit.cover(robe) * 0.95, kit.rgb("#ece8e0"), kit.normal_from_height(kit.pillow(robe, w * 0.08)), spec=0.05)
    head = kit.sd_ellipse(xs, ys, cx, h * 0.17, w * 0.06, h * 0.045)
    s.paint(kit.cover(head), kit.rgb("#d8b898"), kit.normal_from_height(kit.pillow(head, 2)))
    # his right hand raised in blessing (on the viewer's left)
    hand = kit.sd_capsule(xs, ys, cx - w * 0.09, h * 0.32, cx - w * 0.18, h * 0.22, max(0.6, w * 0.025))
    s.paint(kit.cover(hand), kit.rgb("#e0c0a0"))
    band = (int(h * 0.86), int(h - fr - 1))
    if band[1] > band[0]:
        bm = (ys > band[0]) & (ys < band[1]) & (kit.cover(inner) > 0.5)
        s.paint(bm.astype(F32), kit.rgb("#e8dcc0"))
    s.anchors.update(base=(cx, h - 0.5), heart=(cx, hy), ray_red=(red * fade).astype(F32),
                     ray_pale=(pale * fade).astype(F32),
                     inscription=(fr + 1, band[0], w - fr - 1, band[1]), face=(cx, h * 0.17),
                     hand=(cx - w * 0.18, h * 0.22))
    return s.check()


def stand_in_monstrance(h):
    """A gold sunburst monstrance: rays round a glass lunette with the white Host, on a
    knopped stem and a broad foot, a small cross on top."""
    h = max(8, h)
    w = max(6, int(round(h * 0.62)))
    s = kit.Sprite.blank(h, w)
    ys, xs = kit.grid(h, w)
    cx, cy = w / 2, h * 0.36
    R = w * 0.46
    ang = np.arctan2(ys - cy, xs - cx)
    r = np.hypot(xs - cx, ys - cy)
    K = 24
    ray_len = R * (0.78 + 0.22 * (np.cos(ang * K) > 0))
    spikes = (r < ray_len) & (np.abs(np.sin(ang * K / 2)) > 0.35) & (r > R * 0.36)
    gold = kit.rgb("#e8b452")
    s.paint(spikes.astype(F32), gold, normal=None, spec=1.0, gloss=60, metal=1.0)
    rim = np.abs(r - R * 0.34) < max(0.7, R * 0.06)
    s.paint(rim.astype(F32), gold, spec=1.0, gloss=70, metal=1.0)
    stem = kit.sd_box(xs, ys, cx, h * 0.7, max(0.6, w * 0.04), h * 0.2)
    knop = kit.sd_ellipse(xs, ys, cx, h * 0.72, w * 0.08, h * 0.035)
    foot = kit.sd_ellipse(xs, ys, cx, h * 0.95, w * 0.26, h * 0.05)
    body = np.minimum(np.minimum(stem, knop), foot)
    s.paint(kit.cover(body), gold, kit.normal_from_height(kit.pillow(body, 2)), spec=1.0, gloss=55, metal=1.0)
    cross = np.minimum(kit.sd_box(xs, ys, cx, cy - R * 1.08, max(0.5, w * 0.02), R * 0.12),
                       kit.sd_box(xs, ys, cx, cy - R * 1.1, R * 0.08, max(0.5, w * 0.02)))
    s.paint(kit.cover(cross), gold, spec=1.0, gloss=60, metal=1.0)
    host_r = R * 0.27
    host = kit.sd_circle(xs, ys, cx, cy, host_r)
    s.paint(kit.cover(host), kit.rgb("#f6f0e2"), emissive=kit.rgb("#fff4dc") * 0.55)
    tips = np.array([(cx + math.cos(a) * R, cy + math.sin(a) * R) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)],
                    F32)
    s.anchors.update(base=(cx, h - 0.5), host=(cx, cy, host_r), center=(cx, cy), rays=tips, cross=(cx, cy - R * 1.1))
    return s.check()


STAND_INS = {"mary": stand_in_mary, "mercy": stand_in_mercy, "monstrance": stand_in_monstrance}


# =============================================================================
# placing a sprite in the scene
# =============================================================================
class Card:
    """A subject placed in the church for one camera: its screen size, position, depth."""

    def __init__(self, name, cam, min_px=6, seed=0):
        self.name = name
        base, height = PLACES[name]
        self.base_w = np.array(base, F32)
        sx, sy, dep = cam.project(self.base_w)
        top = self.base_w + np.array((0, height, 0), F32)
        _, ty, _ = cam.project(top)
        self.depth = float(dep)
        self.h_px = float(sy - ty)
        self.cam = cam
        self.spr = None
        self.real = False
        self.ok = self.depth > 0.3 and self.h_px >= min_px
        self.bx, self.by = float(sx), float(sy)
        # a subject wholly off screen is not rendered at all (every sprite is at most ~0.42 of
        # its height either side of its base anchor, which is at its foot)
        hw = 0.45 * self.h_px + 3
        if self.ok and (self.bx + hw < 0 or self.bx - hw > cam.W or self.by + 3 < 0
                        or self.by - 1.05 * self.h_px - 2 > cam.H):
            self.ok = False
        if self.ok:
            self.spr, self.real = sprite(name, int(round(self.h_px)), seed)
            ax, ay = self.spr.anchors["base"]
            # whole-pixel placement keeps the sprite crisp
            self.ox = int(round(self.bx - ax))
            self.oy = int(round(self.by - ay))
            self.on_screen = (self.ox < cam.W and self.oy < cam.H and self.ox + self.spr.w > 0
                              and self.oy + self.spr.h > 0)
            self.ok = self.on_screen

    def to_screen(self, xy):
        """Sprite pixel coordinates (..., 2) -> screen pixel coordinates."""
        xy = np.asarray(xy, F32)
        return xy + np.array((self.ox, self.oy), F32)

    def world_at(self, sx, sy):
        """World points on the card plane under screen pixel positions."""
        cam = self.cam
        u = (np.asarray(sx, F32) - cam.cx) / cam.f
        v = -(np.asarray(sy, F32) - cam.cy) / cam.f
        d = self.depth
        return (cam.eye + cam.F * d + cam.R * (u * d)[..., None] + cam.U * (v * d)[..., None]).astype(F32)

    def window(self, H, W):
        """(screen slices, sprite slices) of the visible part, or None."""
        if not self.ok:
            return None
        s = self.spr
        y0, x0 = max(0, self.oy), max(0, self.ox)
        y1, x1 = min(H, self.oy + s.h), min(W, self.ox + s.w)
        if y0 >= y1 or x0 >= x1:
            return None
        return ((slice(y0, y1), slice(x0, x1)),
                (slice(y0 - self.oy, y1 - self.oy), slice(x0 - self.ox, x1 - self.ox)))

    def paste(self, gb):
        """Composite the sprite into the G-buffer where it is in front of the building.
        Returns the visible coverage (H, W) (0 elsewhere)."""
        cov = np.zeros((gb.H, gb.W), F32)
        win = self.window(gb.H, gb.W)
        if win is None:
            return cov
        (sy, sx), (ty, tx) = win
        s = self.spr
        if self.name in FLAT:
            under = gb.depth[sy, sx][s.alpha[ty, tx] > 0.5]
            if under.size:
                self.depth = min(self.depth, float(np.percentile(under, 3)) - 0.025)
        a = s.alpha[ty, tx] * (gb.depth[sy, sx] > self.depth + 0.02)
        a3 = a[..., None]
        ia, ia3 = 1.0 - a, 1.0 - a3
        cam = self.cam
        gb.albedo[sy, sx] = s.albedo[ty, tx] * a3 + gb.albedo[sy, sx] * ia3
        n = s.normal[ty, tx] * a3 + gb.normal[sy, sx] * ia3
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        gb.normal[sy, sx] = n
        nw = n[..., 0:1] * cam.R + n[..., 1:2] * cam.U - n[..., 2:3] * cam.F
        gb.nworld[sy, sx] = nw / np.maximum(np.linalg.norm(nw, axis=-1, keepdims=True), 1e-6)
        for k in ("spec", "gloss", "metal", "ao"):
            getattr(gb, k)[sy, sx] = getattr(s, k)[ty, tx] * a + getattr(gb, k)[sy, sx] * ia
        gb.emissive[sy, sx] = s.emissive[ty, tx] * a3 + gb.emissive[sy, sx] * ia3
        ys, xs = np.mgrid[sy, sx]
        P = self.world_at(xs + 0.5, ys + 0.5)
        front = a > 0.5
        gb.pos[sy, sx] = np.where(front[..., None], P, gb.pos[sy, sx])
        gb.depth[sy, sx] = np.where(front, self.depth, gb.depth[sy, sx])
        gb.dist[sy, sx] = np.where(front, np.linalg.norm(P - cam.eye, axis=-1), gb.dist[sy, sx])
        gb.mat[sy, sx] = np.where(front, M_SUBJECT, gb.mat[sy, sx])
        gb.obj[sy, sx] = np.where(front, SUBJECT_IDS[self.name], gb.obj[sy, sx])
        cov[sy, sx] = a
        return cov

    def layer(self, arr, H, W, cov=None):
        """A per-pixel sprite array (h, w) or (h, w, 3) placed on a screen-sized buffer
        (multiplied by the visible coverage if given)."""
        shape = (H, W) + arr.shape[2:]
        out = np.zeros(shape, F32)
        win = self.window(H, W)
        if win is None:
            return out
        (sy, sx), (ty, tx) = win
        out[sy, sx] = arr[ty, tx]
        if cov is not None:
            out *= cov[..., None] if out.ndim == 3 else cov
        return out

    def drop_shadow(self, L, bg_depth, H, W, spread=0.22, opacity=0.85):
        """Soft shadow (H, W) the card casts, from a light at world point L, onto what
        stands behind it (bg_depth: the building's depth before the card was pasted).
        None when the light is not in front of the card."""
        if not self.ok:
            return None
        cam = self.cam
        _, _, lz = cam.project(np.asarray(L, F32))
        lz = float(lz)
        if lz >= self.depth - 0.05:
            return None
        win = self.window(H, W)
        if win is None:
            return None
        (sy, sx), (ty, tx) = win
        a = self.spr.alpha[ty, tx]
        behind = bg_depth[sy, sx][a > 0.5]
        if behind.size == 0:
            return None
        zb = float(np.median(behind))
        if zb <= self.depth + 0.02:
            return None
        s = self.spr
        cxs, cys = self.ox + s.w / 2, self.oy + s.h / 2
        C = self.world_at(cxs, cys)
        lam = (zb - lz) / (self.depth - lz)              # along L -> C until the depth of the wall
        S = np.asarray(L, F32) + (C - np.asarray(L, F32)) * lam
        px, py, _ = cam.project(S)
        dx, dy = float(px) - cxs, float(py) - cys
        mag = lam * self.depth / zb                      # the shadow is a little larger than the card
        blur = spread * (zb - self.depth) / (self.depth - lz) * cam.f / zb
        # gather: screen pixel -> card pixel (nearest), inside an enlarged box
        pad = int(abs(dx) + abs(dy) + blur * 3 + s.h * (mag - 1) + 3)
        y0, y1 = max(0, self.oy - pad), min(H, self.oy + s.h + pad)
        x0, x1 = max(0, self.ox - pad), min(W, self.ox + s.w + pad)
        if y0 >= y1 or x0 >= x1:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(F32)
        u = ((xs + 0.5 - cxs - dx) / mag + cxs - self.ox).astype(np.int32)
        v = ((ys + 0.5 - cys - dy) / mag + cys - self.oy).astype(np.int32)
        ok = (u >= 0) & (u < s.w) & (v >= 0) & (v < s.h)
        sh = np.zeros((y1 - y0, x1 - x0), F32)
        sh[ok] = s.alpha[v[ok], u[ok]]
        sh = kit.blur(sh, max(0, int(round(blur)))) if blur >= 1 else sh
        out = np.zeros((H, W), F32)
        out[y0:y1, x0:x1] = np.clip(sh, 0, 1) * opacity
        own = self.layer(s.alpha, H, W)                  # the card does not shadow itself
        return out * (bg_depth > self.depth + 0.04) * (1.0 - own)
