"""CATHEDRAL - a night vigil in a Gothic church, before the Blessed Sacrament.

Four fixed cameras in one church (nave.py): the whole sanctuary seen from the
nave, the monstrance exposed on the high altar, Our Lady's altar in the north
aisle, and the Divine Mercy shrine in the south aisle.

Because the camera never moves, each (camera, size) is baked once into a
`View`: the building is raymarched into a G-buffer, the three subjects are
pasted in as camera-facing sprite cards, and every light group (the altar
candles, each half of the votive bank, Our Lady's candles, the Mercy shrine,
the sanctuary lamp, the Host, the Mercy rays, and a soft key light on each
image, as a shrine's image is kept lit) is lit into its own surface map, haze
("air light") map and bloom map; moonlight through the stained glass, the
blue of the moonlit haze and the sky glow are baked into a static base.  A
small cache (within a memory budget) keeps recent views while the cathedral is
on screen, so switching back to a variant is instant; leaving the view keeps
only the one on screen.  While the window is being resized, the last frame is
stretched instead of baking a view for every size on the way.

Each frame is then cheap: the light maps are weighted by gentle per-group
flicker and summed, and the living details are drawn on top - candle flames,
the glowing votive glass, incense haze and the censer's smoke, moonbeams
with drifting dust, the sanctuary lamp, the Host's slow breathing glow with
an occasional glint running out along the monstrance's rays, the shimmer of
the Divine Mercy rays, and a soft glint on Our Lady's rosary, which the
scene prays in real time (rosary.py), with a bead click on each new bead and
a quiet caption of the mysteries.  A gentle vignette keeps each view on its
image, and in ASCII mode the images get a little extra exposure (a glyph
covers only part of its cell, so mid-tones read darker there than in blocks).
"""

import functools
import math
import textwrap
import time
from collections import OrderedDict

import numpy as np

from ..base import Scene
from . import fx, kit, light, nave, rosary, subjects

F32 = np.float32
VARIANTS = ("sanctuary", "adoration", "our lady", "divine mercy")
THEMES = {"sanctuary": "eucharistic", "adoration": "eucharistic", "our lady": "marian", "divine mercy": "mercy"}
WORLD_FLAMES = nave.flames()
_CUPS, _SIDE, _ = nave.votive_cups()
CUP_GROUP = np.array(["votive_l" if sd == 0 else "votive_r" for sd in _SIDE] + ["mercy"] * 5)
GROUP_FLAMES = {}
for _i, _f in enumerate(WORLD_FLAMES):
    GROUP_FLAMES.setdefault(_f[2], []).append(_i)
GROUP_SLOT = {n: i for i, n in enumerate(("altar_l", "altar_r", "votive_l", "votive_r", "mary", "mercy"))}
# extra exposure on the subjects in ASCII mode (see View); the gold and the Host already shine
ASCII_LIFT = {"mary": 1.45, "mercy": 1.45, "monstrance": 1.0}
# ...and on the gilded stars of the niche and the vault, which only glints light: each is lifted
# toward a light glyph (STAR_RAW, in the canvas's ASCII brightness), at most STAR_LIFT times
STAR_RAW, STAR_LIFT = 0.36, 2.4
LUMA = np.array((0.2126, 0.7152, 0.0722), F32)     # the canvas's luma weights
HAZE_BLOCKS, HAZE_ASCII = 0.009, 0.005     # how much of the baked air light the incense haze shows
VIGNETTE = {"sanctuary": 0.3, "adoration": 0.45, "our lady": 0.45, "divine mercy": 0.5}
NICHE_FRAME = (1.0, 1.14, 4.68)            # Our Lady's niche frame: half width, bottom, top (m; nave._mary_frame)
CACHE_MAX, CACHE_BYTES = 8, 128 << 20      # baked views kept (LRU): at most this many, and this much memory
SETTLE = 0.3                               # s: while a resize goes on, stretch the last frame instead of baking
HYMN_EVERY = 600.0                         # s: the drone's theme hint is renewed while the view is shown
BLOCK_MIN = 20                             # cells: the narrowest room the caption block beside the niche takes

# key lights on the images: (offset x, fraction of the height, offset z) from the base, power
KEY_LIGHTS = (("mary", (-0.3, 1.05, -1.2), 1.2), ("mercy", (-0.2, 0.62, -2.0), 3.2))
KEY_COL = np.array((1.0, 0.87, 0.72), F32)         # a warm white: the veil and the robe stay white

GOLD_TXT = (0.66, 0.53, 0.30)
IVORY_TXT = (0.70, 0.66, 0.56)
DIM_TXT = (0.62, 0.58, 0.49)
INSCRIPTION = "JESUS, I TRUST IN YOU"


def _half(img):
    """2x2 box downsample (odd sizes keep their last row / column)."""
    H, W = img.shape[:2]
    p = np.pad(img, ((0, H % 2), (0, W % 2), (0, 0)), mode="edge")
    return ((p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) * 0.25).astype(F32)


def _shoulder(img, knee=0.7):
    """Roll highlights off above the knee instead of clipping them flat (in place):
    candlelit marble and linen stay golden, flame cores still reach white."""
    np.clip(img, 0.0, 4.0, out=img)
    over = np.maximum(img - knee, 0.0)
    img -= over
    img += over / (1.0 + over * 2.0)
    return img


def _weighted(M, w):
    """sum_k w[k] * M[k] - a plain loop beats numpy's vector @ matrix here by 10x."""
    out = M[0] * w[0]
    for k in range(1, M.shape[0]):
        out += M[k] * w[k]
    return out


@functools.lru_cache(maxsize=1024)
def _wrap(text, width):
    """`text` in lines of at most `width` cells, as few lines as will do and as even as can
    be (a right-aligned block should not end in a stub); a word longer than `width` stays
    whole, so check the result's widths."""
    if len(text) <= width:
        return (text,)
    best = textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False)
    for w in range(width - 1, 0, -1):
        t = textwrap.wrap(text, w, break_long_words=False, break_on_hyphens=False)
        if len(t) > len(best) or max(map(len, t)) > w:
            break
        best = t
    return tuple(best)


class View:
    """One camera at one size, baked: light maps, static base and the per-frame layers."""

    def __init__(self, variant, W, H, seed=0):
        self.variant, self.W, self.H = variant, W, H
        cam = nave.camera(variant)
        gb = nave.build(cam, W, H)
        self.cam = cam
        depth0 = gb.depth.copy()
        # ---- the subjects, as cards in the G-buffer
        self.cards = {}
        covs = {}
        for name in ("monstrance", "mary", "mercy"):
            card = subjects.Card(name, cam, min_px=6)
            if card.ok:
                cov = card.paste(gb)
                if cov.max() > 0.5:                # not hidden behind a pier
                    self.cards[name] = card
                    covs[name] = cov
        # in ASCII a glyph covers only part of its cell, so mid-tones read darker than in
        # blocks: there the subjects get a little more exposure to stay the focal points
        lift = np.ones((H, W), F32)
        for name, c in covs.items():
            np.maximum(lift, 1.0 + (ASCII_LIFT[name] - 1.0) * c, out=lift)
        self.ascii_lift = lift[..., None]
        # the gilded stars of Our Lady's niche and of the apse vault (see below), not next to
        # a subject: her own crown of stars stays as it is
        stars = ((gb.mat == nave.M_NICHE) | (gb.mat == nave.M_VAULT)) & (gb.metal > 0.5)
        if covs:
            close = sum(covs.values()) > 0.02
            for _ in range(2):
                close = close | np.roll(close, 1, 0) | np.roll(close, -1, 0) | np.roll(close, 1, 1) | np.roll(close, -1, 1)
            stars &= ~close
        # Our Lady's niche and its frame (moulding, gable, pinnacles) on screen, in cells:
        # the rosary caption keeps clear of it
        self.niche_box = None
        if "mary" in self.cards:
            fx0, _, fd0 = cam.project(np.array((nave.XM - NICHE_FRAME[0], 0.0, nave.ZE - 0.09), F32))
            fx1, _, fd1 = cam.project(np.array((nave.XM + NICHE_FRAME[0], 0.0, nave.ZE - 0.09), F32))
            _, fy0, _ = cam.project(np.array((nave.XM, NICHE_FRAME[2], nave.ZE - 0.09), F32))
            _, fy1, _ = cam.project(np.array((nave.XM, NICHE_FRAME[1], nave.ZE - 0.09), F32))
            if fd0 > 0.3 and fd1 > 0.3:
                self.niche_box = (float(min(fx0, fx1)), float(fy0) / 2.0, float(max(fx0, fx1)), float(fy1) / 2.0)
        # ...and, row by row, the starry opening with its arch moulding: the one-line caption
        # near the top may cross the stone of the gable and the pinnacles, never the stars
        self.niche_rows = self._niche_rows(cam) if self.niche_box is not None else None
        mon = self.cards.get("monstrance")
        if mon is not None:
            hx, hy, hr = mon.spr.anchors["host"]
            host = mon.world_at(*mon.to_screen((hx, hy)))
            host = host + cam.F * -0.03
        else:
            host = np.array((0.0, nave.THRONE_Y + 0.6, nave.MON_Z - 0.03), F32)
        groups = light.layout(host=host)
        rays = self._ray_lights(cam)
        if rays is not None:
            groups.append(rays)
        groups += self._key_lights()
        dirs = light.view_rays(cam)
        sites = {g.site for g in groups if g.site}
        vis = {s: light.site_visibility(gb, s) for s in sites}
        N3 = H * W * 3
        S = np.zeros((len(groups), N3), F32)
        A = np.zeros((len(groups), N3), F32)
        for k, g in enumerate(groups):
            s, a = light.bake(gb, g, dirs, vis.get(g.site))
            s = self._drop_shadows(s, g, depth0)
            S[k] = s.reshape(-1)
            A[k] = a.reshape(-1)
        # groups that light nothing in this view (e.g. the far shrine) are dropped
        keep = [k for k in range(len(groups)) if S[k].max() > 2e-3 or A[k].max() > 0.05 or groups[k].name in
                ("host", "rays", "lamp") or groups[k].only is not None]
        groups = [groups[k] for k in keep]
        self.names = [g.name for g in groups]
        # air light only for the groups that light the haze (not the key lights): a_idx maps
        # each row of A to its group, for the per-frame weights
        self.a_idx = np.array([i for i, k in enumerate(keep) if groups[i].haze > 0], int)
        self.S, self.A = S[keep], A[[keep[i] for i in self.a_idx]]
        del S, A
        # ---- static base: sky glow, moonlight, the windows' glass, the subjects' own glow
        ms, ma = light.bake_moon(gb, dirs)
        glass = gb.mat == nave.M_GLASS
        facing = np.clip(-(gb.nworld @ light.MOON_DIR), 0, 1)
        win_k = np.where(glass, 0.1 + 0.3 * facing, 0.0)[..., None]
        em = gb.emissive.copy()
        self.host_layer = None
        self.ray_layer = None
        self.ray_phase = None
        if mon is not None:
            hx, hy, hr = mon.spr.anchors["host"]
            ys, xs = kit.grid(mon.spr.h, mon.spr.w)
            near = np.clip(1.6 - np.hypot(xs - hx, ys - hy) / max(hr, 0.5) * 0.6, 0, 1)
            lay = mon.layer(mon.spr.emissive * near[..., None], H, W, covs["monstrance"])
            em -= lay
            self.host_layer = lay
        mer = self.cards.get("mercy")
        if mer is not None:
            msk = np.clip(mer.spr.anchors["ray_red"] + mer.spr.anchors["ray_pale"], 0, 1)
            lay = mer.layer(mer.spr.emissive * msk[..., None], H, W, covs["mercy"])
            em -= lay
            self.ray_layer = lay
            hx, hy = mer.to_screen(mer.spr.anchors["heart"])
            gy, gx = kit.grid(H, W)
            self.ray_phase = (np.hypot(gx - hx, gy - hy) / max(2.0, mer.h_px * 0.05)).astype(F32)
        em = np.clip(em, 0, None)
        self.static = (light.ambient(gb) + ms + gb.emissive * win_k + em * (1 - glass[..., None])).astype(F32)
        self.beams = ma.astype(F32)
        # in ASCII the stars, lit only by glints, would mostly fall below the black point and
        # read as blank brownish cells: each is lifted toward a light glyph, never further
        if stars.any():
            est = (self.static + _weighted(self.S, np.ones(len(self.names), F32)).reshape(H, W, 3))[stars]
            raw = 0.55 * (est @ LUMA) + 0.45 * est.max(-1)
            k = np.clip(STAR_RAW / np.maximum(raw, 1e-4), 1.0, STAR_LIFT)
            self.ascii_lift[stars, 0] = np.maximum(self.ascii_lift[stars, 0], k)
        lum = ma @ np.array((0.3, 0.5, 0.2), F32)
        self.beam_lum = np.clip(lum / max(1e-4, float(lum.max()) * 0.55), 0, 1).astype(F32) * 0.3
        self.beams += light.moon_haze(gb)
        # ---- votive glass and the sanctuary lamp (they flicker one by one)
        flat_obj = gb.obj.reshape(-1)
        cups, _, cols = nave.votive_cups()
        ncup = len(cups)
        cup_col = np.concatenate([light.CUP_TINT[cols] * 0.2, np.tile(np.array((1.0, 0.82, 0.55), F32) * 0.18, (5, 1))])
        ci = np.flatnonzero((flat_obj >= 0) & (flat_obj < ncup + 5) & (gb.mat.reshape(-1) == nave.M_VOTIVE))
        self.cup_idx = ci
        self.cup_id = flat_obj[ci].astype(int)
        self.cup_col = cup_col[self.cup_id].astype(F32)
        li = np.flatnonzero(gb.mat.reshape(-1) == nave.M_LAMP)
        self.lamp_idx = li
        # ---- flames (index map into the world list for consistent flicker)
        gidx = {n: i for i, n in enumerate(self.names)}
        self.flames = fx.Flames(cam, gb, WORLD_FLAMES, gidx)
        self.flame_ids = self.flames.ids
        cup_flame = {cup: i for i, (_, _, _, cup) in enumerate(WORLD_FLAMES) if cup >= 0}
        self.cup_fidx = np.array([cup_flame.get(int(c), 0) for c in self.cup_id], int)
        # ---- bloom: what glows, per group, blurred once (kept at half resolution)
        r = max(1, int(min(W, H) / 26))
        self.bh, self.bw = (H + 1) // 2, (W + 1) // 2
        B = np.zeros((len(self.names), self.bh * self.bw * 3), F32)
        for k, name in enumerate(self.names):
            e = np.zeros((H, W, 3), F32)
            for i in range(self.flames.n):
                if self.flames.grp[i] == k:
                    fx.glint(e, self.flames.sx[i], self.flames.sy[i] - self.flames.h[i] * 0.45,
                             max(0.7, self.flames.h[i] * 0.4), 1.4 * self.flames.gain[i], fx.HALO)
            if name.startswith("votive") or name == "mercy":
                on = self.cup_idx[CUP_GROUP[self.cup_id] == name]
                e.reshape(-1, 3)[on] += cup_col[flat_obj[on]] * 0.6
            if name == "lamp":
                e.reshape(-1, 3)[li] += np.array((1.0, 0.1, 0.05), F32) * 1.4
            if name == "host" and self.host_layer is not None:
                e += self.host_layer * 1.5
            if name == "rays" and self.ray_layer is not None:
                e += self.ray_layer * 0.9
            B[k] = kit.blur(_half(e), max(1, r // 2)).reshape(-1)
        self.B = B
        self.bloom_static = kit.blur(_half(np.clip(gb.emissive * win_k, 0, 1) * glass[..., None]), max(1, r // 2)) * 0.7
        # ---- incense from the censer, dust in the moonbeams
        cx, cz = nave.CENSER
        top = np.array((cx - 0.24, nave.YS + 0.99, cz), F32)
        px, py, dep = cam.project(top)
        self.plume = None
        if dep > 0.3 and -20 < px < W + 20 and -10 < py < H + 40:
            z_at = gb.depth[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))]
            if z_at > dep - 0.3:
                self.plume = fx.Plume(float(px), float(py), 1.9 * cam.f / float(dep), H, W, seed + 3)
        # ---- rosary beads on screen
        self.beads = None
        mary = self.cards.get("mary")
        if mary is not None:
            a = mary.spr.anchors
            self.beads = mary.to_screen(a["beads"]).astype(F32)
            br = a.get("bead_r")
            self.bead_r = (np.asarray(br, F32) if br is not None else np.full(len(self.beads), 1.0, F32))
            self.bead_kind = np.asarray(a["bead_kind"]).astype(int)
            bx = np.clip(self.beads[:, 0].astype(int), 0, W - 1)
            by = np.clip(self.beads[:, 1].astype(int), 0, H - 1)
            self.bead_vis = covs["mary"][by, bx] > 0.5
        # a soft vignette keeps the eye on the image at the centre of each view; on a screen
        # much wider than the camera's frame, what lies far beyond the frame's sides (the
        # high altar's candles seen from a side altar...) fades further, so nothing cut by
        # the screen edge outshines the image
        gy, gx = kit.grid(H, W)
        r = np.hypot((gx / W - 0.5) * 2.0, (gy / H - 0.5) * 2.0) / math.sqrt(2.0)
        vig = 1.0 - VIGNETTE[variant] * np.clip((r - 0.35) / 0.65, 0, 1) ** 1.5
        rx = np.abs(gx[:1] - W / 2.0) / max(1.0, cam.frame_px[0])
        vig = vig * (1.0 - 0.7 * np.clip((rx - 1.25) / 1.25, 0, 1) ** 1.5)
        self.vignette = vig[..., None].astype(F32)
        self.depth = gb.depth                  # the rest of the G-buffer is not needed any more
        self.nbytes = sum(a.nbytes for a in vars(self).values() if isinstance(a, np.ndarray))

    # ------------------------------------------------------------------
    def _niche_rows(self, cam):
        """(H/2, 2) left and right screen columns of Our Lady's niche opening with its arch
        moulding (nave._mary_frame) on each cell row; NaN on rows above or below it."""
        H = self.H
        zf = nave.ZE - 0.09                                     # the moulding's front
        D = float((np.array((nave.XM, float(cam.eye[1]), zf), F32) - cam.eye) @ cam.F)
        out = np.full(((H + 1) // 2, 2), np.nan)
        if D <= 0.3:
            return out
        yw = float(cam.eye[1]) + (cam.cy - (np.arange(H) + 0.5)) * D / cam.f
        half = nave.NICHE_HW + 0.07                             # the ring's centre line
        dy = np.maximum(yw - nave.NICHE_SPRING, 0.0)
        hw = np.sqrt(np.maximum((2 * half + 0.075) ** 2 - dy * dy, 0.0)) - half
        hw = np.where(yw < nave.NICHE_Y0 - 0.02, 0.0, hw)
        on = hw > 0
        xs = np.stack([np.full(H, nave.XM), yw, np.full(H, zf)], -1).astype(F32)
        left, _, _ = cam.project(xs - np.stack([hw, 0 * hw, 0 * hw], -1).astype(F32))
        right, _, _ = cam.project(xs + np.stack([hw, 0 * hw, 0 * hw], -1).astype(F32))
        for r in range(out.shape[0]):
            k = [j for j in (2 * r, 2 * r + 1) if j < H and on[j]]
            if k:
                out[r] = (float(min(left[j] for j in k)), float(max(right[j] for j in k)))
        return out

    def _ray_lights(self, cam):
        """The Divine Mercy rays as soft coloured lights, sampled from the ray masks, held a
        little in front of the painting so they spill onto the floor and prie-dieu."""
        mer = self.cards.get("mercy")
        if mer is None:
            return None
        lights = []
        for key, col in (("ray_pale", (0.62, 0.74, 1.0)), ("ray_red", (1.0, 0.10, 0.12))):
            m = np.asarray(mer.spr.anchors[key], F32)
            if m.sum() < 1:
                continue
            ys, xs = np.nonzero(m > 0.2)
            wts = m[ys, xs]
            order = np.argsort(ys)
            # two lights where the fan of the ray reaches the foot of the painting, held
            # out in front of it, so the colour spills down onto the floor and prie-dieu
            low = order[len(order) // 2:]
            px = float(np.average(xs[low], weights=wts[low])) + 0.5
            py = float(np.average(ys[low], weights=wts[low])) + 0.5
            sx, sy = mer.to_screen((px, py))
            P = mer.world_at(sx, sy) - cam.F * 0.45 - cam.U * 0.25
            lights.append(light.PointLight(tuple(P), np.array(col, F32) * 0.18, 0.4))
            # ...and a low one before the altar on the same side, where the ray would reach
            # the floor: a soft pool of its colour on the stones and the prie-dieu
            side = float(np.sign(P[0] - mer.base_w[0])) or (1.0 if key == "ray_red" else -1.0)
            Q = (float(mer.base_w[0]) + side * 0.45, 1.05, float(mer.base_w[2]) - 1.05)
            lights.append(light.PointLight(Q, np.array(col, F32) * 0.3, 0.45))
        if not lights:
            return None
        # boost 0: the rays do not wash the painting they come from
        return light.LightGroup("rays", lights, site=None, fill=0.25, haze=0.5, wrap=0.4, boost=0.0, falls=0.7)

    def _key_lights(self):
        """A soft warm light on each image itself (not on the stones around it), as a
        shrine's image is kept lit: candlelight alone from below leaves the faces dark.
        Each key light breathes with its own candles (see Cathedral._weights)."""
        out = []
        for name, (dx, fy, dz), power in KEY_LIGHTS:
            card = self.cards.get(name)
            if card is None:
                continue
            base, height = subjects.PLACES[name]
            P = (base[0] + dx, base[1] + height * fy, base[2] + dz)
            out.append(light.LightGroup(f"{name}_key", [light.PointLight(P, KEY_COL * power, 0.4)],
                                        fill=0.3, haze=0.0, wrap=0.35, only=subjects.SUBJECT_IDS[name],
                                        also=(nave.M_NICHE,) if name == "mary" else ()))
        return out

    def _drop_shadows(self, s, g, depth0):
        """Darken a group's light on the walls behind each subject it shines on (a key
        light only lights its own image - and Our Lady's niche, which does get her shadow)."""
        if not self.cards or not g.lights or (g.only is not None and not g.also):
            return s
        w = np.array([float(np.max(L.col)) for L in g.lights], F32)
        P = np.array([L.pos for L in g.lights], F32)
        Lc = (P * w[:, None]).sum(0) / max(1e-6, float(w.sum()))
        for name, card in self.cards.items():
            if name == "monstrance" and g.name == "host":
                continue
            # only lights that are near enough to matter for this card
            if np.linalg.norm(Lc - card.base_w) > 6.0:
                continue
            sh = card.drop_shadow(Lc, depth0, self.H, self.W, spread=0.18 if len(g.lights) < 3 else 0.3)
            if sh is not None:
                s = s * (1.0 - sh[..., None])
        return s


class Cathedral(Scene):
    key = "cathedral"
    title = "CATHEDRAL"
    subtitle = "night vigil before the Blessed Sacrament"
    kind = "pixels"
    variants = VARIANTS
    ambience = "cathedral"
    ramp = kit.RAMP
    bgfill = kit.BGFILL
    gamma = kit.GAMMA
    floor = kit.FLOOR
    edges = kit.EDGES

    def __init__(self, app=None, seed=None):
        super().__init__(app, seed)
        self.seed = int(self.rng.integers(1 << 30))
        frng = np.random.default_rng(self.seed)
        self.flame_flick = fx.Flicker(frng, len(WORLD_FLAMES))
        self.group_flick = fx.Flicker(frng, 16, slow=0.05, mid=0.02, fast=0.006)
        self.lamp_flick = fx.Flicker(frng, 1, slow=0.06, mid=0.03, fast=0.01)
        self.haze = fx.Haze(self.seed % 100000)
        self.cache = OrderedDict()
        self.view = None
        self.motes = None
        self.rosary = rosary.Rosary()
        self.glint_pos = None
        self.captions = {}             # line -> [shown text, target text, alpha, fading out?]
        self._hymn = None              # (theme, scene time) of the last "hymn" hint to the drone
        self._layout = None            # cheap per-(variant, size) facts for update()
        self._resized_t = -1e9         # scene time of the last resize (see draw_pixels)
        self._last = None              # the last frame drawn from a baked view
        self._rows = (None, None)      # (room, rows of the caption block's lines) - see _block_rows

    # ------------------------------------------------------------------ lifecycle
    def resize(self, w, h):
        self.view = None
        self.motes = None
        self._layout = None
        self.glint_pos = None          # the beads are elsewhere now: no gliding there from the old spot
        self._resized_t = self.t

    def on_variant(self):
        self.view = None
        self.motes = None
        self._layout = None
        self.glint_pos = None
        self._last = None
        if self._hymn is None or THEMES[self.variant_name] != self._hymn[0]:
            self._send_hymn()

    def on_enter(self):
        # always: a hint sent while the scene was being built (before the audio process
        # was there to hear it) is lost, and the drone takes a repeated hint in its stride
        self._send_hymn()

    def on_exit(self):
        # the organ goes back to all its chants, and only the view on screen stays baked
        if self._hymn is not None:
            self.emit("hymn", layer="drone", theme=None)
        self._hymn = None
        keep = (self.variant_name, self.w, self.h * 2)
        for k in [k for k in self.cache if k != keep]:
            del self.cache[k]

    def _send_hymn(self):
        self._hymn = (THEMES[self.variant_name], self.t, time.monotonic())
        self.emit("hymn", layer="drone", theme=self._hymn[0])

    def _get_view(self):
        if self.view is not None:
            return self.view
        key = (self.variant_name, self.w, self.h * 2)
        v = self.cache.get(key)
        if v is None:
            v = View(self.variant_name, self.w, self.h * 2, self.seed)
            self.cache[key] = v
            # recent views stay baked within a memory budget (the new one always stays)
            while len(self.cache) > 1 and (len(self.cache) > CACHE_MAX
                                           or sum(c.nbytes for c in self.cache.values()) > CACHE_BYTES):
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        self.view = v
        mary = v.cards.get("mary")
        if mary is not None:
            self.rosary.set_kinds(v.bead_kind.tolist())
        self.motes = fx.Motes(np.random.default_rng(self.seed + 11), v.H, v.W, v.beam_lum)
        return v

    def _get_layout(self):
        """Where Our Lady is on screen for this camera (cheap: no rendering)."""
        if self._layout is None:
            cam = nave.camera(self.variant_name).fit(max(1, self.w), max(1, self.h * 2))
            base, height = subjects.PLACES["mary"]
            sx, sy, dep = cam.project(np.array(base, F32))
            _, ty, _ = cam.project(np.array(base, F32) + np.array((0, height, 0), F32))
            hpx = float(sy - ty)
            vis = dep > 0.3 and hpx >= 6 and 0 <= sx < self.w
            self._layout = dict(x=float(sx), h=hpx, visible=bool(vis))
        return self._layout

    # ------------------------------------------------------------------ update
    def update(self, dt):
        super().update(dt)
        # the drone follows a hint for a while only: renew it while we are here (by the wall
        # clock too, as scene time stands still while the screen sleeps and the organ plays on)
        if self._hymn is not None and (self.t - self._hymn[1] > HYMN_EVERY
                                       or time.monotonic() - self._hymn[2] > HYMN_EVERY):
            self._send_hymn()
        bead = self.rosary.update(dt)
        if bead is not None and self.w:
            lay = self._get_layout()
            if lay["visible"]:
                kind = self.rosary.kind_of(bead)
                pan = float(np.clip(lay["x"] / max(1, self.w) * 2 - 1, -1, 1))
                vel = float(np.clip(0.22 + lay["h"] / 170.0, 0.2, 0.9)) * {"cross": 1.0, "medal": 0.95,
                                                                             "large": 0.9}.get(kind, 0.75)
                self.emit("bead", layer="ambience", kind=kind, pan=pan, vel=vel)
        if self.motes is not None:
            self.motes.update(dt, self.t)
        self._ease_glint(dt)
        self._update_captions(dt)

    def _ease_glint(self, dt):
        """Glide the rosary glint toward the bead (or chain) in Our Lady's fingers."""
        v = self.view
        if v is None or v.beads is None:
            return
        a, b, f = self.rosary.glint()
        n = len(v.beads)
        target = v.beads[min(a, n - 1)] * (1 - f) + v.beads[min(b, n - 1)] * f
        if self.glint_pos is None:
            self.glint_pos = target.copy()
        else:
            self.glint_pos += (target - self.glint_pos) * (1.0 - math.exp(-dt / 0.28))

    # ------------------------------------------------------------------ light weights
    def _weights(self, v, t):
        fl = self.flame_flick(t)
        out = np.ones(len(v.names), F32)
        gf = self.group_flick(t)
        for k, name in enumerate(v.names):
            if name == "host":
                out[k] = 1.0 + 0.10 * math.sin(2 * math.pi * t / 7.0)
            elif name == "rays":
                out[k] = 1.0 + 0.07 * math.sin(2 * math.pi * t / 9.5)
            elif name == "lamp":
                out[k] = float(self.lamp_flick(t)[0])
            elif name.endswith("_key"):              # follows its candles (computed below)
                continue
            else:
                ids = GROUP_FLAMES.get(name)
                out[k] = float(np.mean(fl[ids])) * float(gf[GROUP_SLOT.get(name, 15)]) if ids else 1.0
        for k, name in enumerate(v.names):
            if name.endswith("_key"):                # breathes with its candles, but only half as much
                parent = name[:-4]
                out[k] = 0.5 + 0.5 * next((out[j] for j, n in enumerate(v.names) if n == parent), 1.0)
        return out, fl

    # ------------------------------------------------------------------ render
    def draw_pixels(self, mode):
        img = self.image()
        if self.w < 1 or self.h < 1:
            return img
        if (self.view is None and self._last is not None and self.t - self._resized_t < SETTLE
                and (self.variant_name, self.w, self.h * 2) not in self.cache):
            # the size is still changing (a window being dragged): show the last frame
            # stretched instead of baking a view for every size on the way
            src = self._last
            H, W = img.shape[:2]
            img[:] = src[(np.arange(H) * src.shape[0]) // H][:, (np.arange(W) * src.shape[1]) // W]
            return img
        v = self._get_view()
        t = self.t
        H, W = v.H, v.W
        w, fl = self._weights(v, t)
        lit = _weighted(v.S, w).reshape(H, W, 3)
        air = _weighted(v.A, w[v.a_idx]).reshape(H, W, 3) if len(v.a_idx) else np.zeros((H, W, 3), F32)
        dens = self.haze.density(H, W, t)[..., None]
        blocks = mode == "blocks"
        img[:] = v.static
        img += lit
        img += air * ((HAZE_BLOCKS if blocks else HAZE_ASCII) * dens)
        img += v.beams * ((0.55 + 0.45 * dens) * (1.0 if blocks else 0.7))
        flat = img.reshape(-1, 3)
        # votive glass, each cup with its own flame
        if v.cup_idx.size:
            flat[v.cup_idx] += v.cup_col * fl[v.cup_fidx][:, None]
        if v.lamp_idx.size:
            k = next((w[i] for i, n in enumerate(v.names) if n == "lamp"), 1.0)
            flat[v.lamp_idx] += np.array((0.95, 0.06, 0.03), F32) * (0.75 * k)
        # the Host breathes; the Mercy rays shimmer outward
        if v.host_layer is not None:
            img += v.host_layer * (1.0 + 0.12 * math.sin(2 * math.pi * t / 7.0))
        if v.ray_layer is not None:
            img += v.ray_layer * (1.0 + 0.22 * np.sin(v.ray_phase - (t * 1.1) % (2 * math.pi))[..., None])
        if not blocks:
            img *= v.ascii_lift
        # incense
        if v.plume is not None:
            d = v.plume.density(t)
            if d is not None:
                sl = v.plume.sl
                a = air[sl]
                lum = (a @ np.array((0.3, 0.5, 0.2), F32))[..., None]
                smoke = (a * 0.02 + lum * 0.007 + np.array((0.010, 0.010, 0.013), F32)) * d[..., None]
                img[sl] = img[sl] * (1.0 - 0.12 * d[..., None]) + smoke * 1.6
        # flames
        if v.flames.n:
            ids = v.flame_ids
            v.flames.draw(img, fl[ids], self.flame_flick.sway(t)[ids], mode)
        if self.motes is not None:
            self.motes.draw(img, t, mode)
        self._draw_glints(img, v, t)
        # bloom
        # on a small screen the glow would swallow the little subjects: less of it there
        k = (0.75 if blocks else 0.3) * min(1.0, 0.35 + H / 150.0)
        bloom = (_weighted(v.B, w).reshape(v.bh, v.bw, 3) + v.bloom_static) * k
        img += np.repeat(np.repeat(bloom, 2, 0), 2, 1)[:H, :W]
        img *= v.vignette
        _shoulder(img)
        self._last = img
        return img

    def _draw_glints(self, img, v, t):
        # rosary: a soft glint on the bead in her fingers, gliding to the next one
        if v.beads is not None:
            a, _, _ = self.rosary.glint()
            a = min(a, len(v.beads) - 1)
            if self.glint_pos is None:
                self._ease_glint(0.0)
            if v.bead_vis[a]:
                r = max(0.55, float(v.bead_r[a]) * 0.6)
                gx, gy = float(self.glint_pos[0]), float(self.glint_pos[1])
                pulse = 0.45 + 0.1 * math.sin(t * 2.1)
                fx.glint(img, gx, gy, r, pulse, (1.0, 0.9, 0.7), star=r > 1.2)
                fx.glint(img, gx, gy, r * 2.4, pulse * 0.1, (1.0, 0.78, 0.5))
        # monstrance: now and then a glint runs out along one of the rays
        mon = v.cards.get("monstrance")
        if mon is not None:
            period = 11.0
            k = math.floor(t / period)
            s = (t - k * period - 2.0) / 1.9
            if 0.0 <= s <= 1.0:
                rays = np.asarray(mon.spr.anchors["rays"], F32)
                if len(rays):
                    j = int((k * 7919) % len(rays))
                    c = mon.to_screen(mon.spr.anchors["host"][:2])
                    tip = mon.to_screen(rays[j])
                    e = s * s * (3 - 2 * s)
                    p = c + (tip - c) * (0.3 + 0.7 * e)
                    amp = math.sin(math.pi * s) * 0.55
                    fx.glint(img, float(p[0]), float(p[1]), max(0.6, mon.h_px / 60), amp, (1.0, 0.92, 0.7),
                             star=mon.h_px > 40)

    # ------------------------------------------------------------------ captions
    def _update_captions(self, dt):
        title, mystery, prayer = self.rosary.caption()
        for key, txt in (("title", title), ("mystery", mystery), ("prayer", prayer)):
            c = self.captions.get(key)
            if c is None:
                self.captions[key] = [txt, txt, 0.0, False]
                continue
            if c[1] != txt:
                c[1] = txt
                c[3] = True
            if c[3]:                       # fading out the old text
                c[2] -= dt / 0.7
                if c[2] <= 0:
                    c[2] = 0.0
                    c[0] = c[1]
                    c[3] = False
            else:
                c[2] = min(1.0, c[2] + dt / 1.4)

    def draw_overlay(self, cv, mode):
        v = self.view
        if v is None:
            return
        name = self.variant_name
        if name in ("our lady", "sanctuary"):
            self._draw_rosary_caption(cv, v, compact=(name == "sanctuary"))
        if "mercy" in v.cards:
            self._draw_inscription(cv, v)

    def _caption_text(self, key):
        c = self.captions.get(key)
        return (c[0], c[2]) if c else ("", 0.0)

    def _block_rows(self, room):
        """Rows for the title, the mystery and the prayer in the caption block at this room
        (cells): the most each line needs, wrapped, over every set of mysteries, so the block
        keeps its shape all through the rosary and from day to day.  None: too narrow."""
        if self._rows[0] != room:
            rows = None
            if room >= BLOCK_MIN:
                groups = ([], [], [])
                for name in rosary.MYSTERIES:
                    for g, texts in zip(groups, self.rosary.caption_lines(name)):
                        g.extend(texts)
                wraps = [[_wrap(t.upper() if i == 0 else t, room) for t in set(g)] for i, g in enumerate(groups)]
                if all(len(ln) <= room for w in wraps for lines in w for ln in lines):
                    rows = tuple(max(len(lines) for lines in w) for w in wraps)
            self._rows = (room, rows)
        return self._rows[1]

    def _draw_rosary_caption(self, cv, v, compact):
        """The rosary caption: three quiet lines beside Our Lady's niche on a wide screen in
        "our lady" (clear of the niche's whole frame; a line too long for the room wraps),
        otherwise one short line near the top (clear of her and of the niche's starry
        opening, in a shorter form where the whole one does not fit).  Which of the two is
        decided by the layout alone, so the caption never jumps between them as the prayers
        change.  Never on the HUD rows 0 and h-1; each line fades out and in when its text
        changes."""
        mary = v.cards.get("mary")
        if mary is None or cv.h < 12 or cv.w < 40:
            return
        title, ta = self._caption_text("title")
        mystery, ma = self._caption_text("mystery")
        prayer, pa = self._caption_text("prayer")
        s = mary.spr
        # Our Lady's extent in cells (with a margin for her glow), and with her niche's frame
        y0, y1 = mary.oy / 2.0, (mary.oy + s.h) / 2.0
        mx0, mx1 = mary.ox - 1, mary.ox + s.w + 1
        bx0, bx1 = mx0, mx1
        if v.niche_box is not None:
            bx0, bx1 = min(bx0, int(math.floor(v.niche_box[0]))), max(bx1, int(math.ceil(v.niche_box[2])))
        if not compact and cv.w >= 96:
            gap = 3
            left_room, right_room = bx0 - gap - 1, cv.w - bx1 - gap - 1
            # the side does not depend on the text (no jumping across her as the lines change):
            # left, where the longest line of today's rosary fits there or it has the more room
            on_left = left_room >= self.rosary.caption_width() or left_room >= right_room
            room = left_room if on_left else right_room
            rows = self._block_rows(room)
            if rows is not None:
                total = rows[0] + rows[1] + 1 + rows[2]          # a blank row before the prayer
                row0 = int(max(2, min(cv.h - 1 - total, y0 + (y1 - y0) * 0.18)))
                if row0 + total <= cv.h - 1:
                    tops = (row0, row0 + rows[0], row0 + rows[0] + rows[1] + 1)
                    lines = ((title.upper(), GOLD_TXT, ta), (mystery, IVORY_TXT, ma), (prayer, DIM_TXT, pa))
                    for (txt, col, a), top, n in zip(lines, tops, rows):
                        if txt and a > 0.01:
                            for k, ln in enumerate(_wrap(txt, room)[:n]):
                                x = bx0 - gap - len(ln) if on_left else bx1 + gap
                                cv.text(x, top + k, ln, col, alpha=0.82 * a)
                    return
        # one line near the top (row 2, else 1): the mystery and the prayer, or less, whichever
        # fits beside Our Lady and her niche's opening; on a narrow screen, a shorter form
        st = self.rosary.step
        num = rosary.ROMAN[st.decade - 1] + " · " if 1 <= st.decade <= 5 else ""
        a = min(ma, pa)
        gap = 2

        def keep_out(row):
            """The columns the caption keeps clear of on this row (or None)."""
            spans = [(mx0, mx1)] if row + 1 > y0 and row < y1 else []
            nr = v.niche_rows
            if nr is not None and 0 <= row < len(nr) and np.isfinite(nr[row, 0]):
                spans.append((int(math.floor(nr[row, 0])), int(math.ceil(nr[row, 1]))))
            return (min(p for p, _ in spans), max(q for _, q in spans)) if spans else None

        options = [f"{mystery}  ·  {prayer}", f"{num}{prayer}", prayer] if prayer else [mystery]
        for short in rosary.short_forms(prayer):
            options += [f"{num}{short}", short]
        for line in options:
            n = len(line)
            if not line or n > max(24, int(cv.w * 0.5)) or n + 4 > cv.w:
                continue
            for row in (2, 1):
                ko = keep_out(row)
                for col in ((2,) if ko is None else (2, ko[1] + gap)):
                    if 2 <= col and col + n <= cv.w - 2 and (ko is None or col + n + gap <= ko[0] or col >= ko[1] + gap):
                        cv.text(col, row, line, IVORY_TXT, alpha=0.75 * a)
                        return

    def _draw_inscription(self, cv, v):
        mer = v.cards["mercy"]
        x0, y0, x1, y1 = mer.spr.anchors["inscription"]
        sx0, sy0 = mer.to_screen((x0, y0))
        sx1, sy1 = mer.to_screen((x1, y1))
        c0, c1 = int(math.ceil(sx0)), int(math.floor(sx1))
        r0, r1 = int(math.ceil(sy0 / 2.0)), int(math.floor(sy1 / 2.0)) - 1
        text = INSCRIPTION
        if c1 - c0 < len(text) + 2 or r1 < r0:          # too small to letter (with a cell clear of the
            return                                      # frame on each side): the band stays plain
        row = (r0 + r1) // 2
        if not (1 <= row < cv.h - 1):
            return
        col = c0 + (c1 - c0 - len(text)) // 2
        if col < 0 or col + len(text) > cv.w:
            return
        bg = cv.bg[row, col:col + len(text)].mean(0)
        fgc = cv.fg[row, col:col + len(text)].mean(0)
        band = np.maximum(bg, fgc * 0.8)
        lum = float(band @ np.array((0.3, 0.55, 0.15)))
        if lum > 0.3:                    # a lit cartouche: dark lettering, as painted
            ink = np.clip(band * 0.25 + np.array((0.06, 0.015, 0.01)), 0, 1)
        else:                            # in the dark the letters catch the candlelight instead
            ink = np.clip(band * 0.5 + np.array((0.42, 0.34, 0.2)), 0, 1)
        cv.text(col, row, text, tuple(float(c) for c in ink), bg=tuple(float(c) for c in band))
