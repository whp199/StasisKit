"""Scene base class.

A scene either draws straight into the cell Canvas (kind = "chars") or renders
an RGB float image of shape (2*h, w, 3) that the canvas converts into ASCII or
half-block art (kind = "pixels").  Scenes get `update(dt)` then `render(cv, mode)`
every frame and may emit audio events through `self.emit(...)`.
"""

import numpy as np


class Scene:
    key = "base"
    title = "UNTITLED"
    subtitle = ""
    kind = "pixels"            # "pixels" or "chars"
    ramp = "classic"
    bgfill = 0.22
    gamma = 0.8
    floor = 0.07               # ascii black point (see Canvas.blit_pixels)
    edges = 0.0                # ascii edge-glyph threshold (0 = off)
    variants = ("default",)
    ambience = None            # key of the scene-ambience sound generator

    def __init__(self, app=None, seed=None):
        self.app = app
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.w = self.h = 0
        self.variant = 0
        self._img = None

    # -- lifecycle ---------------------------------------------------------
    def ensure_size(self, w, h):
        if (w, h) != (self.w, self.h):
            self.w, self.h = w, h
            self.resize(w, h)

    def resize(self, w, h):
        pass

    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def update(self, dt):
        self.t += dt

    # -- variants ----------------------------------------------------------
    def next_variant(self, step=1):
        self.variant = (self.variant + step) % len(self.variants)
        self.on_variant()

    def set_variant(self, name_or_idx):
        if isinstance(name_or_idx, str):
            names = [v.lower() for v in self.variants]
            if name_or_idx.lower() in names:
                self.variant = names.index(name_or_idx.lower())
        else:
            self.variant = int(name_or_idx) % len(self.variants)
        self.on_variant()

    def on_variant(self):
        pass

    @property
    def variant_name(self):
        return self.variants[self.variant]

    # -- rendering ----------------------------------------------------------
    def image(self):
        """Reusable float image buffer of shape (2h, w, 3)."""
        shape = (self.h * 2, self.w, 3)
        if self._img is None or self._img.shape != shape:
            self._img = np.zeros(shape, np.float32)
        return self._img

    def render(self, cv, mode):
        if self.kind == "pixels":
            img = self.draw_pixels(mode)
            cv.blit_pixels(img, mode, self.ramp, self.bgfill, gamma=self.gamma, floor=self.floor,
                            edges=self.edges)
            self.draw_overlay(cv, mode)
        else:
            self.draw_chars(cv, mode)

    def draw_pixels(self, mode):
        return self.image()

    def draw_overlay(self, cv, mode):
        pass

    def draw_chars(self, cv, mode):
        cv.clear()

    # -- helpers ------------------------------------------------------------
    def emit(self, event, **kw):
        if self.app is not None:
            self.app.audio_event(event, **kw)

    def grid(self):
        """Pixel coordinate grids (ys, xs) for the current image size."""
        H, W = self.h * 2, self.w
        key = (H, W)
        if getattr(self, "_grid_key", None) != key:
            self._grid_key = key
            self._gy, self._gx = np.mgrid[0:H, 0:W].astype(np.float32)
        return self._gy, self._gx
