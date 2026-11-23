"""Small shared helpers: colours, easing, smooth randomness, noise fields."""

import colorsys
import math

import numpy as np


def hexrgb(s):
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb(c):
    """Accept '#rrggbb', (r,g,b) floats, or (r,g,b) 0-255 ints."""
    if isinstance(c, str):
        return hexrgb(c)
    c = tuple(c)
    if any(v > 1.0 for v in c):
        return tuple(v / 255.0 for v in c)
    return c


def hsv(h, s, v):
    return colorsys.hsv_to_rgb(h % 1.0, max(0, min(1, s)), max(0, min(1, v)))


def mix(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def scale(c, k):
    return tuple(min(1.0, v * k) for v in c)


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def cosine_palette(t, a, b, c, d):
    """Inigo Quilez style palette; t is an array, returns (..., 3)."""
    t = np.asarray(t, np.float32)[..., None]
    return np.asarray(a, np.float32) + np.asarray(b, np.float32) * np.cos(
        6.2831853 * (np.asarray(c, np.float32) * t + np.asarray(d, np.float32)))


def hash01(*vals):
    """Deterministic pseudo-random float in [0, 1) from integers."""
    h = 2166136261
    for v in vals:
        h ^= (int(v) * 0x9E3779B1) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
        h ^= h >> 12
    return (h & 0xFFFFFF) / float(0x1000000)


class Drift:
    """Smooth 1D random signal in [-1, 1] (cubic-interpolated random knots)."""

    def __init__(self, period=8.0, seed=None):
        self.period = period
        self.rng = np.random.default_rng(seed)
        self.knots = list(self.rng.uniform(-1, 1, 4))
        self.pos = 0.0

    def step(self, dt):
        self.pos += dt / self.period
        while self.pos >= 1.0:
            self.pos -= 1.0
            self.knots.pop(0)
            self.knots.append(float(self.rng.uniform(-1, 1)))
        return self.value()

    def value(self):
        p0, p1, p2, p3 = self.knots
        t = self.pos
        # Catmull-Rom
        return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                      + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def periodic_noise(h, w, beta=2.0, seed=None, aspect=1.0):
    """Seamless (tileable) fractal noise via FFT spectral shaping, scaled 0..1."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None] * aspect
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spec = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) / f ** (beta / 2.0)
    spec[0, 0] = 0
    n = np.fft.irfft2(spec, s=(h, w))
    n -= n.min()
    n /= max(1e-9, n.max())
    return n.astype(np.float32)


def fmt_duration(seconds):
    seconds = max(0, int(seconds + 0.5))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def wrap_text(text, width):
    """Word-wrap preserving explicit newlines."""
    width = max(1, int(width))   # width <= 0 would never shorten a long word
    out = []
    for para in str(text).split("\n"):
        words = para.split(" ")
        line = ""
        for wd in words:
            if not line:
                line = wd
            elif len(line) + 1 + len(wd) <= width:
                line += " " + wd
            else:
                out.append(line)
                line = wd
            while len(line) > width:
                out.append(line[:width])
                line = line[width:]
        out.append(line)
    return out


def lerp_angle(a, b, t):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * t
