"""JULIA DRIFT - a slowly morphing Julia set: one quiet equation, endlessly.

z -> z^2 + c, with c tracing a slow loop just inside the edge of the
Mandelbrot set, gently rotating and breathing in scale.  Smooth (continuous)
iteration counts are coloured with soft cosine palettes.
"""

import math

import numpy as np

from .base import Scene
from ..util import cosine_palette

PALETTES = {
    "moonlit": ((0.5, 0.5, 0.6), (0.5, 0.45, 0.4), (1.0, 1.0, 1.0), (0.62, 0.68, 0.78)),
    "ember": ((0.5, 0.35, 0.3), (0.5, 0.4, 0.3), (1.0, 0.9, 0.7), (0.0, 0.12, 0.22)),
    "aurora": ((0.4, 0.5, 0.5), (0.45, 0.5, 0.45), (1.0, 1.0, 1.0), (0.3, 0.45, 0.62)),
    "nebula": ((0.5, 0.4, 0.55), (0.5, 0.4, 0.45), (0.8, 0.9, 1.0), (0.85, 0.6, 0.45)),
}


class Julia(Scene):
    key = "julia"
    title = "JULIA DRIFT"
    subtitle = "the math garden on deck nine"
    kind = "pixels"
    variants = tuple(PALETTES)
    ambience = "wind"
    bgfill = 0.3
    floor = 0.06
    edges = 0.0
    ITER = 72

    def c_of(self, t):
        # a slow walk along the main cardioid boundary, pulled slightly inward
        a = t * 0.021 + 0.9
        r = 1.012 + 0.035 * math.sin(t * 0.047)
        w = r * complex(math.cos(a), math.sin(a)) / 2
        return w - w * w

    def draw_pixels(self, mode):
        H, W = self.h * 2, self.w
        t = self.t
        gy, gx = self.grid()
        scale = 1.3 * (1.0 + 0.12 * math.sin(t * 0.031))
        rot = t * 0.012
        aspect = W / H
        u = (gx / W - 0.5) * 2 * aspect * scale / 1.0
        v = (gy / H - 0.5) * 2 * scale
        cr, sr = math.cos(rot), math.sin(rot)
        z = (u * cr - v * sr) + 1j * (u * sr + v * cr)
        z = z.astype(np.complex64)
        c = np.complex64(self.c_of(t))
        n = np.zeros(z.shape, np.float32)
        alive = np.ones(z.shape, bool)
        zz = z.ravel()
        nn = n.ravel()
        idx = np.arange(zz.size)
        for i in range(self.ITER):
            zi = zz[idx]
            zi = zi * zi + c
            zz[idx] = zi
            m = (zi.real * zi.real + zi.imag * zi.imag) < 64.0
            esc = idx[~m]
            if len(esc):
                mag = np.abs(zz[esc])
                nn[esc] = i + 1 - np.log2(np.maximum(np.log(np.maximum(mag, 1.0001)), 1e-6))
            idx = idx[m]
            if len(idx) == 0:
                break
        inside = np.zeros(zz.size, bool)
        inside[idx] = True
        inside = inside.reshape(z.shape)
        k = np.sqrt(np.maximum(n, 0) / self.ITER)
        pal = PALETTES[self.variant_name]
        col = cosine_palette(k * 2.6 + t * 0.008, *pal).astype(np.float32)
        glow = np.clip(k * 1.9, 0, 1) ** 1.15
        img = self.image()
        img[:] = np.clip(col * glow[..., None], 0, 1)
        img[inside] = np.array([0.015, 0.015, 0.035], np.float32)
        return img
