"""Cell framebuffer (numpy) + diffing truecolor ANSI encoder + PNG snapshots.

A Canvas holds three arrays:
    ch  (h, w)     unicode code points (one per cell)
    fg  (h, w, 3)  float32 0..1 foreground colour
    bg  (h, w, 3)  float32 0..1 background colour

Pixel scenes render an RGB float image of shape (2*h, w, 3) and blit it with
either the "ascii" renderer (luminance -> glyph ramp, coloured) or the
"blocks" renderer (upper-half-block glyphs: 2 pixels per cell).
"""

import numpy as np

RAMPS = {
    "classic": " .:-=+*#%@",
    "soft": " .,:;~=+*x#",
    "dense": " .'`^\":;Il!i><~+_-?][}{1)(|tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@",
    "dots": " .·:+*oO0@",
    "water": " .-~=≈*%#@",
    "stars": " ..·:+*✦#@",
}
DEFAULT_RAMP = "classic"

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_ramp_cache = {}


def _ramp_array(ramp):
    arr = _ramp_cache.get(ramp)
    if arr is None:
        arr = np.array(list(RAMPS.get(ramp, ramp)), dtype="<U1")
        _ramp_cache[ramp] = arr
    return arr


class Canvas:
    def __init__(self, w, h):
        self.resize(w, h)

    def resize(self, w, h):
        self.w, self.h = int(w), int(h)
        self.ch = np.full((self.h, self.w), " ", dtype="<U1")
        self.fg = np.zeros((self.h, self.w, 3), np.float32)
        self.bg = np.zeros((self.h, self.w, 3), np.float32)

    def clear(self, bg=(0.0, 0.0, 0.0)):
        self.ch[:] = " "
        self.fg[:] = 0.0
        self.bg[:] = bg

    # -- primitives -------------------------------------------------------
    def _clip_span(self, x, y, n):
        if y < 0 or y >= self.h or n <= 0:
            return None
        x0, x1 = max(0, x), min(self.w, x + n)
        if x0 >= x1:
            return None
        return x0, x1, x0 - x

    def text(self, x, y, s, fg=(1, 1, 1), bg=None, alpha=1.0):
        """Write a string. `alpha` < 1 blends colours toward what is underneath."""
        x, y = int(x), int(y)
        span = self._clip_span(x, y, len(s))
        if span is None:
            return
        x0, x1, off = span
        chars = s[off:off + (x1 - x0)]
        self.ch[y, x0:x1] = list(chars)
        fg = np.asarray(fg, np.float32)
        if alpha >= 1.0:
            self.fg[y, x0:x1] = fg
            if bg is not None:
                self.bg[y, x0:x1] = bg
        else:
            a = max(0.0, alpha)
            self.fg[y, x0:x1] = fg * a + self.bg[y, x0:x1] * (1 - a)
            if bg is not None:
                self.bg[y, x0:x1] = np.asarray(bg, np.float32) * a + self.bg[y, x0:x1] * (1 - a)

    def put(self, x, y, c, fg, bg=None):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            self.ch[y, x] = c
            self.fg[y, x] = fg
            if bg is not None:
                self.bg[y, x] = bg

    def fill(self, x, y, w, h, ch=" ", fg=None, bg=None):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.w, int(x + w)), min(self.h, int(y + h))
        if x0 >= x1 or y0 >= y1:
            return
        if ch is not None:
            self.ch[y0:y1, x0:x1] = ch
        if fg is not None:
            self.fg[y0:y1, x0:x1] = fg
        if bg is not None:
            self.bg[y0:y1, x0:x1] = bg

    def shade(self, x, y, w, h, k=0.35, tint=None, tint_amt=0.0):
        """Darken (and optionally tint) a region - used for translucent panels."""
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.w, int(x + w)), min(self.h, int(y + h))
        if x0 >= x1 or y0 >= y1:
            return
        self.fg[y0:y1, x0:x1] *= k
        self.bg[y0:y1, x0:x1] *= k
        if tint is not None and tint_amt > 0:
            t = np.asarray(tint, np.float32) * tint_amt
            self.bg[y0:y1, x0:x1] += t

    def scale(self, k):
        self.fg *= k
        self.bg *= k

    def blit_canvas(self, other, x, y, skip_blank=False):
        """Copy another canvas onto this one at (x, y)."""
        x, y = int(x), int(y)
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        w = min(other.w - sx0, self.w - dx0)
        h = min(other.h - sy0, self.h - dy0)
        if w <= 0 or h <= 0:
            return
        src_ch = other.ch[sy0:sy0 + h, sx0:sx0 + w]
        if skip_blank:
            m = (src_ch != " ") | (other.bg[sy0:sy0 + h, sx0:sx0 + w].max(-1) > 0.004)
            dch = self.ch[dy0:dy0 + h, dx0:dx0 + w]
            dch[m] = src_ch[m]
            self.fg[dy0:dy0 + h, dx0:dx0 + w][m] = other.fg[sy0:sy0 + h, sx0:sx0 + w][m]
            self.bg[dy0:dy0 + h, dx0:dx0 + w][m] = other.bg[sy0:sy0 + h, sx0:sx0 + w][m]
        else:
            self.ch[dy0:dy0 + h, dx0:dx0 + w] = src_ch
            self.fg[dy0:dy0 + h, dx0:dx0 + w] = other.fg[sy0:sy0 + h, sx0:sx0 + w]
            self.bg[dy0:dy0 + h, dx0:dx0 + w] = other.bg[sy0:sy0 + h, sx0:sx0 + w]

    def box(self, x, y, w, h, fg, bg=None, style="round", title=None, title_fg=None):
        chars = {
            "round": "╭╮╰╯─│",
            "square": "┌┐└┘─│",
            "heavy": "┏┓┗┛━┃",
            "double": "╔╗╚╝═║",
            "thin": "┌┐└┘╌╎",
        }[style]
        tl, tr, bl, br, hz, vt = chars
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < 2 or h < 2:
            return
        if bg is not None:
            self.fill(x, y, w, h, " ", None, bg)
        self.text(x, y, tl + hz * (w - 2) + tr, fg)
        self.text(x, y + h - 1, bl + hz * (w - 2) + br, fg)
        for yy in range(y + 1, y + h - 1):
            self.put(x, yy, vt, fg)
            self.put(x + w - 1, yy, vt, fg)
        if title and w > 6:
            t = f" {title} "
            if len(t) > w - 4:
                t = f" {str(title)[:max(1, w - 7)]}… "
            self.text(x + 2, y, t, title_fg or fg)

    # -- pixel blitting ------------------------------------------------------
    def blit_pixels(self, img, mode="ascii", ramp=DEFAULT_RAMP, bgfill=0.22, x=0, y=0,
                    gamma=0.8, floor=0.07, edges=0.0):
        """Blit an RGB float image of shape (2*rows, cols, 3) at cell (x, y)."""
        rows, cols = img.shape[0] // 2, img.shape[1]
        x, y = int(x), int(y)
        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        w = min(cols - sx0, self.w - dx0)
        h = min(rows - sy0, self.h - dy0)
        if w <= 0 or h <= 0:
            return
        sub = img[2 * sy0:2 * (sy0 + h), sx0:sx0 + w]
        top = sub[0::2]
        bot = sub[1::2]
        dch = self.ch[dy0:dy0 + h, dx0:dx0 + w]
        dfg = self.fg[dy0:dy0 + h, dx0:dx0 + w]
        dbg = self.bg[dy0:dy0 + h, dx0:dx0 + w]
        if mode == "blocks":
            same = np.abs(top - bot).max(-1) < 0.012
            dch[:] = "▀"
            dch[same] = " "
            dfg[:] = top
            dbg[:] = bot
            return
        c = np.clip((top + bot) * 0.5, 0.0, 4.0)
        m = c.max(-1)
        lum = c @ _LUMA
        raw = np.clip(0.55 * lum + 0.45 * m, 0.0, 1.0)
        # black point: dark areas become blank cells that keep their true colour
        # as background, so gradients stay smooth and glyphs mark only the light
        v = np.clip((raw - floor) / (1.0 - floor), 0.0, 1.0) ** gamma
        ra = _ramp_array(ramp)
        idx = np.minimum((v * (len(ra) - 1) + 0.5).astype(np.int32), len(ra) - 1)
        glyphs = ra[idx]
        if edges and h > 2 and w > 2:
            # edge-aware glyphs: where luminance changes sharply, draw the
            # outline direction (/ \ | -) instead of a density character
            L = np.pad(v, 1, mode="edge")
            gx = (L[1:-1, 2:] - L[1:-1, :-2]) * 2 + (L[:-2, 2:] - L[:-2, :-2]) + (L[2:, 2:] - L[2:, :-2])
            gy = ((L[2:, 1:-1] - L[:-2, 1:-1]) * 2 + (L[2:, :-2] - L[:-2, :-2]) + (L[2:, 2:] - L[:-2, 2:])) * 0.5
            mag = np.hypot(gx, gy)
            edge = (mag > edges) & (v > 0.12) & (v < 0.85)
            if edge.any():
                ax, ay = np.abs(gx), np.abs(gy)
                eg = np.where(ay > ax * 1.9, "-", np.where(ax > ay * 1.9, "|",
                              np.where(gx * gy > 0, "/", "\\")))
                glyphs = np.where(edge, eg, glyphs)
        dch[:] = glyphs
        boost = np.where(m > 1e-4, np.power(np.maximum(m, 1e-4), -0.38), 0.0)[..., None]
        dfg[:] = np.clip(c * boost, 0.0, 1.0)
        keep = 1.0 - np.clip((raw - floor * 0.6) / (floor * 1.6 + 1e-6), 0.0, 1.0)
        dbg[:] = np.clip(c * (bgfill + (1.0 - bgfill) * keep)[..., None], 0.0, 1.0)


class AnsiEncoder:
    """Turns a Canvas into the minimal escape stream since the last frame."""

    def __init__(self, heal=True):
        self.prev = None
        self._fgc = {}
        self._bgc = {}
        self.heal = heal       # repaint one row per frame so glitches self-correct
        self._heal_row = 0

    def invalidate(self):
        self.prev = None

    def encode(self, cv, gain=1.0):
        fg, bg = cv.fg, cv.bg
        if gain != 1.0:
            fg = fg * gain
            bg = bg * gain
        f8 = np.clip(fg * 255.0 + 0.5, 0, 255).astype(np.uint32)
        b8 = np.clip(bg * 255.0 + 0.5, 0, 255).astype(np.uint32)
        f8 &= 0xFE
        b8 &= 0xFE
        fgp = (f8[..., 0] << 16) | (f8[..., 1] << 8) | f8[..., 2]
        bgp = (b8[..., 0] << 16) | (b8[..., 1] << 8) | b8[..., 2]
        ch = cv.ch
        prev = self.prev
        parts = ["\x1b[?2026h"]
        if prev is None or prev[0].shape != ch.shape:
            changed = np.ones(ch.shape, bool)
            parts.append("\x1b[0m\x1b[2J")
        else:
            pch, pfg, pbg = prev
            changed = (ch != pch) | (bgp != pbg) | ((fgp != pfg) & (ch != " "))
            if self.heal and ch.shape[0]:
                self._heal_row = (self._heal_row + 1) % ch.shape[0]
                changed[self._heal_row] = True
        self.prev = (ch.copy(), fgp, bgp)
        rows = np.flatnonzero(changed.any(axis=1))
        if len(rows) == 0:
            return ""
        if len(self._fgc) > 60000:
            self._fgc.clear()
        if len(self._bgc) > 60000:
            self._bgc.clear()
        fgc, bgc = self._fgc, self._bgc
        ap = parts.append
        cur_f = cur_b = -1
        for y in rows.tolist():
            cols = np.flatnonzero(changed[y]).tolist()
            chs = ch[y].tolist()
            fr = fgp[y].tolist()
            br = bgp[y].tolist()
            nextx = -100
            for x in cols:
                if x != nextx:
                    gap = x - nextx
                    if 0 < gap <= 3:
                        xs = range(nextx, x + 1)
                    else:
                        ap(f"\x1b[{y + 1};{x + 1}H")
                        xs = (x,)
                else:
                    xs = (x,)
                for gx in xs:
                    b = br[gx]
                    if b != cur_b:
                        s = bgc.get(b)
                        if s is None:
                            s = bgc[b] = f"\x1b[48;2;{b >> 16};{(b >> 8) & 255};{b & 255}m"
                        ap(s)
                        cur_b = b
                    c = chs[gx]
                    if c != " ":
                        f = fr[gx]
                        if f != cur_f:
                            s = fgc.get(f)
                            if s is None:
                                s = fgc[f] = f"\x1b[38;2;{f >> 16};{(f >> 8) & 255};{f & 255}m"
                            ap(s)
                            cur_f = f
                    ap(c)
                nextx = x + 1
        ap("\x1b[?2026l")
        return "".join(parts)


# -- PNG snapshots (development / sharing) ---------------------------------
_FONT_CANDIDATES = [
    "/usr/share/fonts/hack/Hack-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/TTF/Hack-Regular.ttf",
    "/usr/share/fonts/truetype/hack/Hack-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
]
_FALLBACK_CANDIDATES = [
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf",
]


def save_png(cv, path, font_size=16, gain=1.0):
    import os
    from PIL import Image, ImageDraw, ImageFont

    fonts = [ImageFont.truetype(p, font_size) for p in _FONT_CANDIDATES if os.path.exists(p)][:1]
    fonts += [ImageFont.truetype(p, font_size) for p in _FALLBACK_CANDIDATES if os.path.exists(p)]
    if not fonts:
        fonts = [ImageFont.load_default()]
    cw = max(1, round(fonts[0].getlength("M")))
    chh = cw * 2
    f8 = np.clip(cv.fg * gain * 255 + 0.5, 0, 255).astype(np.uint8)
    b8 = np.clip(cv.bg * gain * 255 + 0.5, 0, 255).astype(np.uint8)
    pix = b8.repeat(chh, 0).repeat(cw, 1).copy()
    half = chh // 2
    for (yy, xx) in zip(*np.nonzero(cv.ch == "▀")):
        pix[yy * chh:yy * chh + half, xx * cw:(xx + 1) * cw] = f8[yy, xx]
    for (yy, xx) in zip(*np.nonzero(cv.ch == "▄")):
        pix[yy * chh + half:(yy + 1) * chh, xx * cw:(xx + 1) * cw] = f8[yy, xx]
    for (yy, xx) in zip(*np.nonzero(cv.ch == "█")):
        pix[yy * chh:(yy + 1) * chh, xx * cw:(xx + 1) * cw] = f8[yy, xx]
    im = Image.fromarray(pix, "RGB")
    dr = ImageDraw.Draw(im)
    glyph_ok = {}
    skip = set(" ▀▄█")
    for yy in range(cv.h):
        row = cv.ch[yy]
        for xx in range(cv.w):
            c = row[xx]
            if c in skip:
                continue
            font = glyph_ok.get(c)
            if font is None:
                font = fonts[0]
                if ord(c) > 0x2400 and len(fonts) > 1:
                    font = fonts[1] if ord(c) < 0x2600 else fonts[-1]
                glyph_ok[c] = font
            dr.text((xx * cw, yy * chh + (chh - font_size) // 2 - 1), c,
                    fill=tuple(int(v) for v in f8[yy, xx]), font=font)
    im.save(path)
    return path
