"""Sound generators.  Every source renders one stereo block (2, BLOCK) per call.

All synthesis is procedural - nothing is sampled - so StasisKit runs fully
offline and the sounds never loop.
"""

import math

import numpy as np

from .dsp import (BLOCK, F_SAFE, FREQS, SR, TAU, GrainBus, Param, Partials, RandomWalks,
                  SpectralNoise, bandpass, highpass, lowpass, normalize, pan_gains, peak, tilt)

DT = BLOCK / SR


def _exp_env(n, tau):
    return np.exp(-np.arange(n) / SR / tau)


def _chirp(f0, f1, n):
    return np.sin(TAU * np.cumsum(np.linspace(f0, f1, n)) / SR)


def _attack(n, secs):
    k = max(1, int(secs * SR))
    env = np.ones(n)
    env[:k] = np.linspace(0, 1, k) ** 2
    return env


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


class Source:
    key = "base"
    label = "BASE"
    presets = None
    wet = 0.25           # reverb send
    gain = 1.0

    def __init__(self, rng):
        self.rng = rng
        self.preset = self.presets[0] if self.presets else None

    def set_preset(self, name, instant=False):
        """Switch preset; instant=True skips any crossfade (used for saved state at start-up)."""
        if self.presets and name in self.presets:
            self.preset = name

    def render(self):
        return np.zeros((2, BLOCK))

    def event(self, name, **kw):
        pass


# =============================================================================
# DRONES
# =============================================================================
class _Stasis:
    """Slow modal pad: detuned saw-ish voices through drifting low-pass filters,
    gliding between chords every minute or so."""

    ROOT = 36.71   # D1
    PROG = [0, -2, 3, -4, 5, 0, -7, 2]
    CHORDS = [[0, 7, 12, 15, 19, 26], [0, 7, 12, 17, 19, 24], [0, 7, 14, 15, 19, 22]]

    def __init__(self, rng):
        self.rng = rng
        self.V, self.H = 6, 9
        self.step = 0
        self.vf = self._chord_freqs(0, self.CHORDS[0])
        self.target = self.vf.copy()
        self.timer = rng.uniform(40, 70)
        self.part = Partials(self.V * self.H * 2, rng)
        self.cut = RandomWalks(self.V, rng, rate=0.025, smooth=9.0)
        self.swell = RandomWalks(self.V, rng, rate=0.03, smooth=7.0)
        k = np.arange(1, self.H + 1)
        self.k = k[None, :, None].astype(float)
        self.det = (np.array([-1.0, 1.0])[None, None, :] * (0.07 + 0.035 * self.k)
                    * rng.uniform(0.7, 1.3, (self.V, 1, 1)))
        vp = rng.uniform(-0.45, 0.45, self.V)
        self.pan = np.clip(vp[:, None, None] + np.array([-0.35, 0.35])[None, None, :], -1, 1)
        self.pan = np.broadcast_to(self.pan, (self.V, self.H, 2)).ravel()
        self.vgain = np.array([1.0, 0.72, 0.6, 0.42, 0.36, 0.22])[:, None, None]

    def _chord_freqs(self, semis, chord):
        return np.array([self.ROOT * 2 ** ((semis + c) / 12.0) for c in chord])

    def render(self):
        self.timer -= DT
        if self.timer <= 0:
            self.timer = self.rng.uniform(55, 95)
            self.step = (self.step + 1) % len(self.PROG)
            chord = self.CHORDS[self.rng.integers(len(self.CHORDS))]
            self.target = self._chord_freqs(self.PROG[self.step], chord)
        self.vf = np.exp(np.log(self.vf) + (np.log(self.target) - np.log(self.vf)) * (DT / 9.0))
        cut = 160 + 760 * self.cut.step()[:, None, None] ** 1.6
        sw = 0.45 + 0.55 * self.swell.step()[:, None, None]
        f = self.vf[:, None, None] * self.k + self.det
        amp = (1.0 / self.k ** 1.05) / np.sqrt(1.0 + (f / cut) ** 4) * self.vgain * sw
        amp = np.broadcast_to(amp, f.shape)
        return self.part.render(f.ravel(), amp.ravel() * 0.11, self.pan)


class _Lumen:
    """Harmonic-series shimmer: individual partials bloom and fade while the
    spectral focus drifts up and down the series."""

    def __init__(self, rng):
        self.rng = rng
        self.K = 40
        self.f0 = 41.2
        self.f0_target = self.f0
        self.k = np.arange(1, self.K + 1, dtype=float)
        self.det = np.stack([np.zeros(self.K), rng.uniform(0.04, 0.22, self.K) * rng.choice([-1, 1], self.K)])
        self.part = Partials(self.K * 2, rng)
        self.walk = RandomWalks(self.K * 2, rng, rate=0.09, smooth=2.6)
        self.focus = RandomWalks(1, rng, rate=0.015, smooth=14.0)
        self.panw = RandomWalks(self.K * 2, rng, rate=0.02, smooth=10.0)
        self.timer = rng.uniform(150, 240)
        self.roots = [41.2, 55.0, 36.71, 49.0, 43.65]

    def render(self):
        self.timer -= DT
        if self.timer <= 0:
            self.timer = self.rng.uniform(150, 260)
            self.f0_target = self.roots[self.rng.integers(len(self.roots))]
        self.f0 += (self.f0_target - self.f0) * (DT / 20.0)
        c = 3.0 + 20.0 * self.focus.step()[0]
        w = self.walk.step().reshape(2, self.K) ** 3.0
        focus = np.exp(-0.5 * ((self.k - c) / (3.0 + c * 0.35)) ** 2)
        amp = w * focus / np.sqrt(self.k)
        amp[:, :2] += 0.18 / np.array([1.0, 1.6])
        f = self.f0 * self.k[None, :] + self.det
        amp = np.where(f < 9000, amp, 0.0)
        pan = (self.panw.step() * 2 - 1) * 0.85
        return self.part.render(f.ravel(), amp.ravel() * 0.24, pan)


class _Abyss:
    """Sub-bass drone with distant inharmonic hull resonances swelling in and out."""

    RATIOS = np.array([1.0, 2.756, 5.404, 8.933])
    BASES = [110.0, 146.83, 164.81, 98.0, 123.47, 82.41]

    def __init__(self, rng):
        self.rng = rng
        self.subf = np.array([32.70, 49.0, 65.41, 98.0, 32.78, 49.07])
        self.suba = np.array([1.0, 0.55, 0.42, 0.16, 0.8, 0.4])
        self.subpan = np.array([0.0, -0.3, 0.3, 0.0, 0.15, -0.15])
        self.subw = RandomWalks(6, rng, rate=0.03, smooth=8.0)
        self.S = 3
        self.res_f = np.array([self.BASES[i] for i in rng.integers(len(self.BASES), size=self.S)])
        self.res_t = rng.uniform(0, 30, self.S)
        self.res_len = rng.uniform(22, 38, self.S)
        self.res_pan = rng.uniform(-0.7, 0.7, self.S)
        self.res_amp = rng.uniform(0.4, 1.0, self.S)
        self.part = Partials(6 + self.S * len(self.RATIOS), rng)
        self.noise = SpectralNoise(rng)
        self.nmag = normalize(lowpass(120, 3) * highpass(22, 2) * tilt(-0.5), 0.06)
        self.nw = RandomWalks(1, rng, rate=0.05, smooth=6.0)

    def render(self):
        sw = 0.5 + 0.5 * self.subw.step()
        self.res_t += DT
        done = self.res_t > self.res_len
        if done.any():
            for i in np.flatnonzero(done):
                self.res_t[i] = -self.rng.uniform(4, 20)
                self.res_len[i] = self.rng.uniform(22, 40)
                self.res_f[i] = self.BASES[self.rng.integers(len(self.BASES))] * self.rng.choice([0.5, 1.0, 1.0, 2.0])
                self.res_pan[i] = self.rng.uniform(-0.75, 0.75)
                self.res_amp[i] = self.rng.uniform(0.35, 1.0)
        ph = np.clip(self.res_t / self.res_len, 0, 1)
        env = np.where(self.res_t < 0, 0.0, np.sin(np.pi * ph) ** 2.2) * self.res_amp
        rf = (self.res_f[:, None] * self.RATIOS[None, :]).ravel()
        ra = (env[:, None] * np.array([1.0, 0.45, 0.22, 0.1])[None, :]).ravel() * 0.35
        rp = np.repeat(self.res_pan, len(self.RATIOS))
        f = np.concatenate([self.subf, rf])
        a = np.concatenate([self.suba * sw, ra])
        p = np.concatenate([self.subpan, rp])
        out = self.part.render(f, a * 0.16, p)
        out += self.noise.render(self.nmag) * (0.5 + 0.5 * self.nw.step()[0])
        return out


class _Halo:
    """Choir-like pad: harmonic voices shaped by slowly morphing vowel formants."""

    VOWELS = [  # (F1, F2, F3), gains
        ((800, 1150, 2900), (1.0, 0.5, 0.12)),   # a
        ((450, 800, 2830), (1.0, 0.35, 0.08)),   # o
        ((325, 700, 2530), (1.0, 0.25, 0.05)),   # u
        ((400, 1600, 2700), (1.0, 0.3, 0.1)),    # e
    ]
    CHORDS = [[45, 52, 56, 59, 61, 64], [42, 49, 54, 57, 61, 64], [38, 45, 50, 54, 57, 64],
              [40, 47, 52, 56, 59, 66]]

    def __init__(self, rng):
        self.rng = rng
        self.V = 6
        self.H = 24
        self.ci = 0
        self.vf = np.array([midi(m) for m in self.CHORDS[0]])
        self.target = self.vf.copy()
        self.timer = rng.uniform(40, 70)
        self.k = np.arange(1, self.H + 1, dtype=float)[None, :, None]
        self.det = np.array([-1.0, 1.0])[None, None, :] * rng.uniform(0.12, 0.3, (self.V, 1, 1))
        self.part = Partials(self.V * self.H * 2, rng)
        self.vowel = RandomWalks(1, rng, rate=0.03, smooth=8.0)
        self.swell = RandomWalks(self.V, rng, rate=0.04, smooth=6.0)
        vp = np.linspace(-0.6, 0.6, self.V)
        rng.shuffle(vp)
        self.pan = np.broadcast_to(np.clip(vp[:, None, None] + np.array([-0.2, 0.2]), -1, 1),
                                   (self.V, self.H, 2)).ravel()
        self.vgain = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.45])[:, None, None]

    def render(self):
        self.timer -= DT
        if self.timer <= 0:
            self.timer = self.rng.uniform(45, 80)
            self.ci = (self.ci + 1) % len(self.CHORDS)
            self.target = np.array([midi(m) for m in self.CHORDS[self.ci]])
        self.vf = np.exp(np.log(self.vf) + (np.log(self.target) - np.log(self.vf)) * (DT / 6.0))
        pos = self.vowel.step()[0] * (len(self.VOWELS) - 1.001)
        i0 = int(pos)
        fr = pos - i0
        (fa, ga), (fb, gb) = self.VOWELS[i0], self.VOWELS[i0 + 1]
        F = [a + (b - a) * fr for a, b in zip(fa, fb)]
        G = [a + (b - a) * fr for a, b in zip(ga, gb)]
        f = self.vf[:, None, None] * self.k + self.det
        env = sum(g * np.exp(-0.5 * ((f - fc) / (fc * 0.11 + 40)) ** 2) for fc, g in zip(F, G))
        env = env + 0.015
        sw = 0.5 + 0.5 * self.swell.step()[:, None, None]
        amp = env / self.k ** 0.4 * self.vgain * sw
        amp = np.where(f < 4000, amp, 0.0)
        return self.part.render(f.ravel(), amp.ravel() * 0.145, self.pan)


class _Orbit:
    """Sparse soft bells over a hushed pad - an airport for sleeping astronauts."""

    SCALE = [62, 64, 66, 69, 71]   # D major pentatonic
    RATIOS = np.array([1.0, 2.0, 3.01, 4.17, 5.43])
    AMPS = np.array([1.0, 0.36, 0.16, 0.08, 0.04])
    TAUS = np.array([3.6, 2.1, 1.2, 0.7, 0.4])

    def __init__(self, rng):
        self.rng = rng
        self.bus = GrainBus(12.0)
        self.next = rng.uniform(1, 4)
        notes = [38, 45, 54, 57, 64]
        self.pf = np.array([midi(n) for n in notes])
        self.H = 6
        k = np.arange(1, self.H + 1, dtype=float)
        self.f = (self.pf[:, None, None] * k[None, :, None]
                  + np.array([-0.1, 0.1])[None, None, :] * (1 + k[None, :, None] * 0.2))
        self.a = (np.array([1.0, 0.6, 0.45, 0.3, 0.2])[:, None, None] / k[None, :, None] ** 1.6
                  * np.ones((1, 1, 2)))
        self.part = Partials(self.f.size, rng)
        self.pan = np.broadcast_to(np.array([-0.4, 0.4])[None, None, :], self.f.shape).ravel()
        self.sw = RandomWalks(len(notes), rng, rate=0.03, smooth=8.0)

    def _bell(self, freq, vel):
        n = int(7.0 * SR)
        t = np.arange(n) / SR
        g = np.zeros(n)
        for r, a, tau in zip(self.RATIOS, self.AMPS, self.TAUS):
            g += a * np.sin(TAU * freq * r * t + self.rng.uniform(0, TAU)) * np.exp(-t / tau)
        g *= _attack(n, 0.008)
        return g * vel

    def render(self):
        self.next -= DT
        if self.next <= 0:
            self.next = self.rng.uniform(3.5, 9.5)
            m = self.SCALE[self.rng.integers(len(self.SCALE))] + 12 * int(self.rng.integers(0, 2))
            off = int(self.rng.uniform(0, BLOCK))
            self.bus.add(self._bell(midi(m), self.rng.uniform(0.25, 0.6)), off, 0.35, self.rng.uniform(-0.7, 0.7))
            if self.rng.random() < 0.3:
                m2 = self.SCALE[self.rng.integers(len(self.SCALE))] + 12
                self.bus.add(self._bell(midi(m2), self.rng.uniform(0.2, 0.4)),
                             off + int(self.rng.uniform(0.3, 1.2) * SR), 0.2, self.rng.uniform(-0.7, 0.7))
        sw = (0.4 + 0.6 * self.sw.step())[:, None, None]
        amp = (self.a * sw).ravel() * 0.12
        return self.part.render(self.f.ravel(), amp, self.pan) + self.bus.pull()


DRONES = {"stasis": _Stasis, "lumen": _Lumen, "abyss": _Abyss, "halo": _Halo, "orbit": _Orbit}


class Drone(Source):
    key = "drone"
    label = "DRONE"
    presets = tuple(DRONES)
    wet = 0.5

    def __init__(self, rng):
        super().__init__(rng)
        self.voices = [[self.preset, DRONES[self.preset](rng), Param(1.0, 3.0)]]

    def set_preset(self, name, instant=False):
        if name not in DRONES:
            return
        if instant:   # this voice alone, already at full level
            v = next((v for v in self.voices if v[0] == name), None)
            self.voices = [v or [name, DRONES[name](self.rng), Param(1.0, 3.0)]]
            self.voices[0][2].jump(1.0)
            self.preset = name
            return
        if name == self.preset:
            return
        self.preset = name
        for v in self.voices:
            v[2].set(0.0, 2.5)
        for v in self.voices:
            if v[0] == name:
                v[2].set(1.0, 2.5)
                return
        self.voices.append([name, DRONES[name](self.rng), Param(0.0, 2.5)])
        self.voices[-1][2].set(1.0)

    def render(self):
        out = np.zeros((2, BLOCK))
        keep = []
        for v in self.voices:
            g = v[2].ramp()
            if v[2].v < 1e-4 and v[2].target == 0.0:
                continue
            out += v[1].render() * g
            keep.append(v)
        self.voices = keep
        return out


# =============================================================================
# NOISE COLOURS
# =============================================================================
def _a_weight(f):
    f2 = f * f
    ra = (12194.0 ** 2 * f2 * f2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                     * (f2 + 12194.0 ** 2))
    return ra * 10 ** (2.0 / 20)


def _noise_spectra():
    hp = highpass(22, 2)
    spectra = {
        "brown": normalize(tilt(-1.0) * hp * lowpass(14000, 1), 0.2),
        "pink": normalize(tilt(-0.5) * hp * lowpass(16000, 1), 0.15),
        "white": normalize(np.ones_like(F_SAFE) * hp * lowpass(17000, 2), 0.09),
        "grey": normalize(np.clip(1.0 / np.maximum(_a_weight(F_SAFE), 0.1), 0, 10) * hp * lowpass(15000, 1), 0.14),
        "blue": normalize(tilt(0.5) * hp * lowpass(17000, 2), 0.06),
        "violet": normalize(tilt(1.0) * hp * lowpass(17000, 2), 0.045),
        "deep": normalize(tilt(-1.5) * highpass(25, 2) * lowpass(500, 2), 0.24),
    }
    return spectra


class Noise(Source):
    key = "noise"
    label = "NOISE"
    presets = ("brown", "pink", "white", "grey", "deep", "blue", "violet")
    wet = 0.0

    def __init__(self, rng):
        super().__init__(rng)
        self.spectra = _noise_spectra()
        self.mag = self.spectra[self.preset].copy()
        self.gen = SpectralNoise(rng)

    def set_preset(self, name, instant=False):
        super().set_preset(name)
        if instant:
            self.mag = self.spectra[self.preset].copy()

    def render(self):
        tgt = self.spectra[self.preset]
        self.mag += (tgt - self.mag) * 0.08
        return self.gen.render(self.mag)


# =============================================================================
# NATURE
# =============================================================================
class Rain(Source):
    key = "rain"
    label = "RAIN"
    presets = ("steady", "downpour", "drizzle", "on a tent")
    wet = 0.15

    def __init__(self, rng):
        super().__init__(rng)
        self.bed = SpectralNoise(rng)
        self.low = SpectralNoise(rng)
        self.bus = GrainBus(4.0)
        self.gust = RandomWalks(2, rng, rate=0.06, smooth=5.0)
        self.bed_mag = normalize(bandpass(280, 9000, 2) * tilt(-0.25) * (1 + peak(2500, 1.2, 0.6)), 0.1)
        self.low_mag = normalize(lowpass(320, 2) * highpass(30, 2) * tilt(-0.6), 0.06)
        self.drops = [self._drop() for _ in range(64)]
        self.tent = [self._tent_drop() for _ in range(24)]

    def _drop(self):
        rng = self.rng
        n = int(0.045 * SR)
        f = rng.uniform(1600, 6500)
        tone = _chirp(f, f * rng.uniform(1.05, 1.6), n) * _exp_env(n, rng.uniform(0.002, 0.011))
        tick = np.diff(rng.standard_normal(n + 1)) * _exp_env(n, 0.0006) * 0.35
        g = tone * rng.uniform(0.15, 0.7) + tick
        return g / (np.abs(g).max() + 1e-9)

    def _tent_drop(self):
        rng = self.rng
        n = int(0.06 * SR)
        f = rng.uniform(180, 420)
        body = np.sin(TAU * f * np.arange(n) / SR) * _exp_env(n, rng.uniform(0.006, 0.014))
        tick = np.diff(rng.standard_normal(n + 1)) * _exp_env(n, 0.0012) * 0.25
        g = body + tick
        return g / (np.abs(g).max() + 1e-9)

    def render(self):
        g = self.gust.step()
        inten = {"steady": 0.7, "downpour": 1.0, "drizzle": 0.35, "on a tent": 0.6}[self.preset]
        level = inten * (0.75 + 0.35 * g[0])
        out = self.bed.render(self.bed_mag) * level
        out += self.low.render(self.low_mag) * (0.4 + 0.8 * inten) * (0.7 + 0.3 * g[1])
        rate = {"steady": 55, "downpour": 140, "drizzle": 18, "on a tent": 30}[self.preset] * (0.6 + 0.6 * g[0])
        n = self.rng.poisson(rate * DT)
        bank = self.tent if self.preset == "on a tent" else self.drops
        for _ in range(n):
            amp = min(1.0, self.rng.exponential(0.12)) * 0.5
            self.bus.add(bank[self.rng.integers(len(bank))], self.rng.integers(BLOCK), amp,
                         self.rng.uniform(-0.9, 0.9))
        return out + self.bus.pull()


class Ocean(Source):
    key = "ocean"
    label = "OCEAN"
    presets = ("shore", "calm bay", "storm surf")
    wet = 0.12

    def __init__(self, rng):
        super().__init__(rng)
        self.low = SpectralNoise(rng)
        self.hiss = SpectralNoise(rng)
        self.bed = SpectralNoise(rng)
        self.bus = GrainBus(4.0)
        self.low_mag = normalize(lowpass(700, 2) * highpass(30, 2) * tilt(-0.8), 0.3)
        self.hiss_mag = normalize(bandpass(500, 11000, 2) * tilt(-0.3) * (1 + peak(3500, 1.0, 0.5)), 0.22)
        self.bed_mag = normalize(lowpass(400, 2) * highpass(25, 2) * tilt(-1.0), 0.05)
        self.waves = []
        self.next = 0.5
        self.t = 0.0
        self.fizz = [np.diff(rng.standard_normal(int(0.003 * SR) + 1)) * _exp_env(int(0.003 * SR), 0.0005)
                     for _ in range(16)]

    def render(self):
        rng = self.rng
        cfg = {"shore": (7.5, 12.0, 1.0), "calm bay": (9.0, 15.0, 0.55), "storm surf": (5.0, 8.5, 1.1)}[self.preset]
        self.next -= DT
        if self.next <= 0:
            self.next = rng.uniform(cfg[0], cfg[1])
            self.waves.append(dict(t0=self.t, rise=rng.uniform(2.5, 4.5), fall=rng.uniform(5.0, 8.5),
                                   amp=rng.uniform(0.55, 1.0) * cfg[2], pan=rng.uniform(-0.6, 0.6)))
        tt = self.t + np.arange(BLOCK) / SR
        lowenv = np.zeros((2, BLOCK))
        hissenv = np.zeros((2, BLOCK))
        keep = []
        for wv in self.waves:
            x = tt - wv["t0"]
            r, f = wv["rise"], wv["fall"]
            if x[0] > r + f * 1.6:
                continue
            keep.append(wv)
            up = np.clip(x / r, 0, 1) ** 2.2
            down = np.exp(-np.clip(x - r, 0, None) / (f * 0.45))
            le = up * down * wv["amp"]
            crash = np.clip(x - r * 0.85, 0, None)
            he = (1 - np.exp(-crash / 0.35)) * np.exp(-crash / (f * 0.38)) * wv["amp"]
            gl, gr = pan_gains(wv["pan"])
            lowenv += np.stack([le * gl, le * gr])
            hissenv += np.stack([he * gl, he * gr])
        self.waves = keep
        out = self.low.render(self.low_mag) * lowenv
        out += self.hiss.render(self.hiss_mag) * hissenv
        out += self.bed.render(self.bed_mag)
        dens = float(hissenv.mean()) * 900
        for _ in range(rng.poisson(dens * DT)):
            self.bus.add(self.fizz[rng.integers(16)], rng.integers(BLOCK),
                         rng.uniform(0.02, 0.09), rng.uniform(-0.8, 0.8))
        self.t += DT
        return out + self.bus.pull()


class Wind(Source):
    key = "wind"
    label = "WIND"
    presets = ("breeze", "high desert", "solar wind")
    wet = 0.2

    def __init__(self, rng):
        super().__init__(rng)
        self.gen = SpectralNoise(rng)
        self.gust = RandomWalks(4, rng, rate=0.07, smooth=3.5)
        self.base = lowpass(350, 1) * highpass(30, 2) * tilt(-0.7)

    def render(self):
        g = self.gust.step()
        k = {"breeze": 0.7, "high desert": 1.0, "solar wind": 0.85}[self.preset]
        mags = []
        for c in range(2):
            gc = 0.65 * g[0] + 0.35 * g[1 + c]
            fc = 220 + 1100 * gc ** 1.5
            m = self.base * (0.35 + 0.5 * gc) + peak(fc, 0.9, 1.2 * (0.25 + gc))
            if self.preset != "breeze":
                wf = 520 + 900 * g[3] ** 1.3
                m = m + peak(wf, 0.035, 2.5 * gc ** 2.5)
            if self.preset == "solar wind":
                m = m + peak(wf * 2.01, 0.03, 1.4 * gc ** 2) + peak(wf * 3.0, 0.03, 0.7 * gc ** 2)
            mags.append(normalize(m, 0.18 * k * (0.25 + 0.95 * gc)))
        return self.gen.render(np.stack(mags))


class Stream(Source):
    key = "stream"
    label = "STREAM"
    presets = ("brook", "trickle", "river")
    wet = 0.15

    def __init__(self, rng):
        super().__init__(rng)
        self.gen = SpectralNoise(rng)
        self.bus = GrainBus(3.0)
        self.base = bandpass(260, 6500, 2) * tilt(-0.35)
        self.walk = RandomWalks(3, rng, rate=0.4, smooth=0.25)

    def _bubble(self, f0, amp, pan):
        rng = self.rng
        tau = rng.uniform(0.006, 0.03) * (900 / f0) ** 0.5
        n = int(min(0.12, tau * 6) * SR)
        g = _chirp(f0, f0 * rng.uniform(1.15, 1.9), n) * _exp_env(n, tau) * _attack(n, 0.0015)
        self.bus.add(g, rng.integers(BLOCK), amp, pan)

    def render(self):
        rng = self.rng
        cfg = {"brook": (0.11, 70, 500, 2400), "trickle": (0.05, 45, 700, 3200), "river": (0.13, 110, 300, 1500)}[self.preset]
        w = self.walk.step()
        m = self.base.copy()
        for _ in range(3):
            m += peak(rng.uniform(cfg[2], cfg[3] * 1.4), 0.25, rng.uniform(0.0, 1.8))
        out = self.gen.render(normalize(m, cfg[0] * (0.7 + 0.5 * w[0])))
        for _ in range(rng.poisson(cfg[1] * (0.6 + 0.8 * w[1]) * DT)):
            f0 = math.exp(rng.uniform(math.log(cfg[2]), math.log(cfg[3])))
            self._bubble(f0, min(1.0, rng.exponential(0.2)) * 0.32, rng.uniform(-0.8, 0.8))
        return out + self.bus.pull()


class Crickets(Source):
    key = "crickets"
    label = "CRICKETS"
    presets = ("summer night", "meadow", "distant")
    wet = 0.3

    def __init__(self, rng):
        super().__init__(rng)
        self.bus = GrainBus(3.0)
        self.bed = SpectralNoise(rng)
        self.bed_mag = normalize(bandpass(80, 1800, 1) * tilt(-0.8), 0.012)
        self.crickets = []
        for i in range(6):
            f = rng.uniform(4100, 5300)
            pulses = int(rng.integers(3, 5))
            pd, gap = rng.uniform(0.013, 0.02), rng.uniform(0.012, 0.018)
            n = int((pulses * (pd + gap) + 0.01) * SR)
            t = np.arange(n) / SR
            env = np.zeros(n)
            for p in range(pulses):
                a = p * (pd + gap)
                x = (t - a) / pd
                env += np.where((x >= 0) & (x <= 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 1.5, 0)
            tone = np.sin(TAU * f * t) + 0.18 * np.sin(TAU * 2 * f * t)
            self.crickets.append(dict(grain=tone * env, period=rng.uniform(0.42, 1.1), next=rng.uniform(0, 1),
                                      amp=rng.uniform(0.15, 0.6), pan=rng.uniform(-0.9, 0.9),
                                      active=True, mood=rng.uniform(20, 60)))
        n = int(1.0 * SR)
        t = np.arange(n) / SR
        trill = np.sin(TAU * 2900 * t) * (0.5 + 0.5 * np.sign(np.sin(TAU * 42 * t))) * 0.6
        self.trill = trill * np.minimum(1, np.minimum(t, 1 - t) / 0.01)
        self.trill_amp = RandomWalks(1, rng, rate=0.05, smooth=6.0)
        self.trill_t = 0.0

    def render(self):
        rng = self.rng
        dist = {"summer night": 1.0, "meadow": 1.3, "distant": 0.35}[self.preset]
        nact = {"summer night": 4, "meadow": 6, "distant": 5}[self.preset]
        for i, c in enumerate(self.crickets[:nact]):
            c["mood"] -= DT
            if c["mood"] <= 0:
                c["active"] = rng.random() < 0.8
                c["mood"] = rng.uniform(15, 60)
            c["next"] -= DT
            while c["next"] < 0:
                if c["active"]:
                    off = int((c["next"] + DT) * SR)
                    self.bus.add(c["grain"], max(0, off), c["amp"] * dist * 0.45, c["pan"])
                c["next"] += c["period"] * rng.uniform(0.92, 1.08)
        self.trill_t -= DT
        if self.trill_t <= 0:
            self.trill_t = 1.0
            ta = self.trill_amp.step()[0] ** 2 * 0.05 * dist
            self.bus.add(self.trill, 0, ta, 0.35)
        else:
            self.trill_amp.step()
        return self.bus.pull() + self.bed.render(self.bed_mag)


class Fire(Source):
    key = "fire"
    label = "FIRE"
    presets = ("hearth", "campfire", "embers")
    wet = 0.1

    def __init__(self, rng):
        super().__init__(rng)
        self.roar = SpectralNoise(rng)
        self.hiss = SpectralNoise(rng)
        self.bus = GrainBus(2.0)
        self.roar_mag = normalize(bandpass(40, 420, 2) * tilt(-0.6), 0.12)
        self.hiss_mag = normalize(bandpass(2500, 12000, 2), 0.012)
        self.flick = RandomWalks(2, rng, rate=1.5, smooth=0.12)
        self.clicks = []
        for _ in range(40):
            n = int(rng.uniform(0.0008, 0.004) * SR) + 2
            c = np.diff(rng.standard_normal(n + 1)) * _exp_env(n, n / SR * 0.3)
            self.clicks.append(c / (np.abs(c).max() + 1e-9))
        self.pops = []
        for _ in range(8):
            n = int(0.03 * SR)
            f = rng.uniform(350, 900)
            p = np.sin(TAU * f * np.arange(n) / SR) * _exp_env(n, 0.004) + \
                np.diff(rng.standard_normal(n + 1)) * _exp_env(n, 0.001)
            self.pops.append(p / (np.abs(p).max() + 1e-9))

    def render(self):
        rng = self.rng
        cfg = {"hearth": (1.0, 7.0), "campfire": (1.2, 11.0), "embers": (0.45, 2.5)}[self.preset]
        fl = self.flick.step()
        out = self.roar.render(self.roar_mag) * cfg[0] * (0.55 + 0.6 * fl[0])
        out += self.hiss.render(self.hiss_mag) * (0.5 + 0.8 * fl[1])
        for _ in range(rng.poisson(cfg[1] * DT)):
            base = rng.integers(BLOCK)
            pan = rng.uniform(-0.6, 0.6)
            for _ in range(int(rng.integers(1, 7))):
                self.bus.add(self.clicks[rng.integers(40)], base + int(rng.exponential(0.008) * SR),
                             min(1.0, rng.exponential(0.25)) * 0.45, pan + rng.uniform(-0.15, 0.15))
        if rng.random() < 0.25 * DT * cfg[0]:
            self.bus.add(self.pops[rng.integers(8)], rng.integers(BLOCK), rng.uniform(0.25, 0.6),
                         rng.uniform(-0.5, 0.5))
        return out + self.bus.pull()


class DeepSea(Source):
    key = "deepsea"
    label = "DEEP SEA"
    presets = ("whale song", "trench", "kelp")
    wet = 0.6

    def __init__(self, rng):
        super().__init__(rng)
        self.rumble = SpectralNoise(rng)
        self.rmag = normalize(lowpass(180, 3) * highpass(20, 2) * tilt(-0.7), 0.12)
        self.walk = RandomWalks(1, rng, rate=0.05, smooth=5.0)
        self.bus = GrainBus(16.0)
        self.next = rng.uniform(3, 8)
        self.bub_next = rng.uniform(5, 15)

    def _call(self):
        rng = self.rng
        dur = rng.uniform(2.0, 5.5)
        n = int(dur * SR)
        x = np.linspace(0, 1, n)
        f0 = rng.uniform(95, 360)
        contour = np.zeros(n)
        for _ in range(int(rng.integers(1, 4))):
            c, w, a = rng.uniform(0, 1), rng.uniform(0.15, 0.6), rng.uniform(-0.9, 0.9)
            contour += a * np.exp(-0.5 * ((x - c) / w) ** 2)
        contour += rng.uniform(-0.4, 0.4) * x
        f = f0 * 2 ** contour
        f *= 1 + 0.006 * np.sin(TAU * rng.uniform(4, 7) * x * dur)
        ph = TAU * np.cumsum(f) / SR
        g = sum(a * np.sin(k * ph) for k, a in zip((1, 2, 3, 4), (1.0, 0.45, 0.22, 0.1)))
        env = np.sin(np.pi * x) ** 1.4 * (0.8 + 0.2 * np.sin(TAU * 1.3 * x * dur))
        return g * env

    def render(self):
        rng = self.rng
        w = self.walk.step()[0]
        out = self.rumble.render(self.rmag) * (0.6 + 0.5 * w)
        self.next -= DT
        if self.next <= 0 and self.preset != "trench":
            self.next = rng.uniform(10, 28)
            pan = rng.uniform(-0.7, 0.7)
            amp = rng.uniform(0.08, 0.2) * (0.6 if self.preset == "kelp" else 1.0)
            off = 0
            for _ in range(int(rng.integers(1, 4))):
                call = self._call()
                self.bus.add(call, off, amp, pan)
                off += len(call) + int(rng.uniform(0.3, 1.5) * SR)
        self.bub_next -= DT
        if self.bub_next <= 0:
            self.bub_next = rng.uniform(4, 14) if self.preset != "kelp" else rng.uniform(1, 4)
            base = 0
            for _ in range(int(rng.integers(3, 12))):
                f0 = rng.uniform(120, 420)
                n = int(0.12 * SR)
                g = _chirp(f0, f0 * 1.7, n) * _exp_env(n, 0.025)
                base += int(rng.exponential(0.07) * SR)
                self.bus.add(g, base, rng.uniform(0.03, 0.09), rng.uniform(-0.5, 0.5))
        return out + self.bus.pull()


class ShipHum(Source):
    key = "shiphum"
    label = "SHIP HUM"
    presets = ("engine room", "cryo bay", "bridge")
    wet = 0.15

    def __init__(self, rng):
        super().__init__(rng)
        self.K = 12
        k = np.arange(1, self.K + 1, dtype=float)
        self.k = k
        self.amp = (1.0 / k ** 1.25) * np.where(k % 2 == 1, 1.0, 0.55)
        self.part = Partials(self.K * 2, rng)
        self.air = SpectralNoise(rng)
        self.throb = 0.0
        self.walk = RandomWalks(self.K, rng, rate=0.1, smooth=3.0)

    def render(self):
        cfg = {"engine room": (46.25, 1.0, 0.07, 900), "cryo bay": (55.0, 0.5, 0.05, 2200),
               "bridge": (61.7, 0.35, 0.04, 1400)}[self.preset]
        f0 = cfg[0]
        self.throb += DT
        th = 0.85 + 0.15 * math.sin(TAU * 0.27 * self.throb) * math.sin(TAU * 0.041 * self.throb)
        w = 0.6 + 0.4 * self.walk.step()
        f = np.concatenate([f0 * self.k, f0 * self.k + 0.11 * self.k])
        a = np.concatenate([self.amp * w, self.amp * w * 0.7]) * th * cfg[1] * 0.06
        a = np.where(f < 1200, a, 0)
        pan = np.concatenate([np.full(self.K, -0.3), np.full(self.K, 0.3)])
        out = self.part.render(f, a, pan)
        out += self.air.render(normalize(lowpass(cfg[3], 2) * highpass(40, 2) * tilt(-0.5), cfg[2]))
        return out


class Pool(Source):
    """Liminal indoor pool: lapping water, drips in a tiled hall, fluorescent hum."""
    key = "pool"
    label = "POOL HALL"
    presets = ("pool hall",)
    wet = 0.55

    def __init__(self, rng):
        super().__init__(rng)
        self.lap = SpectralNoise(rng)
        self.hvac = SpectralNoise(rng)
        self.bus = GrainBus(4.0)
        self.lap_mag = normalize(bandpass(90, 1400, 2) * tilt(-0.6), 0.09)
        self.hvac_mag = normalize(lowpass(500, 2) * highpass(40, 2) * tilt(-0.5), 0.03)
        self.lw = RandomWalks(2, rng, rate=0.6, smooth=0.5)
        self.part = Partials(10, rng)
        self.drip_next = rng.uniform(1, 4)

    def render(self):
        rng = self.rng
        lw = self.lw.step()
        out = self.lap.render(self.lap_mag) * (0.25 + 0.9 * lw[0] ** 2)
        out += self.hvac.render(self.hvac_mag)
        k = np.arange(1, 11, dtype=float)
        out += self.part.render(120.0 * k, 0.004 / k * (0.8 + 0.2 * lw[1]), np.zeros(10))
        self.drip_next -= DT
        if self.drip_next <= 0:
            self.drip_next = rng.uniform(1.5, 6.0)
            f = rng.uniform(800, 1600)
            n = int(0.06 * SR)
            g = _chirp(f, f * rng.uniform(1.3, 1.8), n) * _exp_env(n, 0.012) * _attack(n, 0.001)
            self.bus.add(g, rng.integers(BLOCK), rng.uniform(0.06, 0.16), rng.uniform(-0.8, 0.8))
        if rng.random() < 3 * DT:
            f = rng.uniform(150, 400)
            n = int(0.08 * SR)
            g = _chirp(f, f * 1.5, n) * _exp_env(n, 0.02)
            self.bus.add(g, rng.integers(BLOCK), rng.uniform(0.02, 0.06), rng.uniform(-0.6, 0.6))
        return out + self.bus.pull()


class RoomTone(Source):
    key = "room"
    label = "ROOM"
    presets = ("quiet room",)
    wet = 0.1

    def __init__(self, rng):
        super().__init__(rng)
        self.gen = SpectralNoise(rng)
        self.mag = normalize(lowpass(700, 2) * highpass(35, 2) * tilt(-0.6), 0.035)

    def render(self):
        return self.gen.render(self.mag)


class Taps(Source):
    """ASMR nail taps, triggered by the TAP TAP scene."""
    key = "taps"
    label = "NAIL TAPS"
    presets = ("acrylic", "glass", "wood")
    wet = 0.12

    def __init__(self, rng):
        super().__init__(rng)
        self.bus = GrainBus(3.0)

    def event(self, name, pan=0.0, vel=0.8, **kw):
        rng = self.rng
        if name == "scratch":
            self._scratch(pan, vel, kw.get("dur", 0.4))
            return
        if name != "tap":
            return
        n = int(0.09 * SR)
        t = np.arange(n) / SR
        mat = self.preset
        click = np.diff(rng.standard_normal(n + 1)) * np.exp(-t / rng.uniform(0.0005, 0.0011))
        pings = np.zeros(n)
        if mat == "glass":
            fs, taus, body_f, body_tau = (rng.uniform(3200, 4200), rng.uniform(6200, 7600)), (0.05, 0.03), 1900, 0.02
        elif mat == "wood":
            fs, taus, body_f, body_tau = (rng.uniform(1700, 2300), rng.uniform(3500, 4500)), (0.004, 0.003), rng.uniform(380, 560), 0.012
        else:
            fs, taus, body_f, body_tau = (rng.uniform(2600, 3600), rng.uniform(5200, 6900)), (0.007, 0.004), rng.uniform(600, 900), 0.008
        for f, tau in zip(fs, taus):
            pings += np.sin(TAU * f * t + rng.uniform(0, TAU)) * np.exp(-t / tau)
        body = np.sin(TAU * body_f * t) * np.exp(-t / body_tau)
        g = click * 0.55 + pings * 0.3 + body * 0.35
        g *= _attack(n, 0.0003)
        g /= np.abs(g).max() + 1e-9
        self.bus.add(g, int(rng.uniform(0, 64)), 0.45 * vel, pan * 0.85, itd=0.0006 * abs(pan))

    def _scratch(self, pan, vel, dur):
        rng = self.rng
        n = dur * SR
        if not n >= 2:          # too short to shape (also rejects NaN)
            return
        n = int(min(n, self.bus.size - BLOCK))
        x = rng.standard_normal(n)
        spec = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / SR)
        spec *= np.exp(-0.5 * (np.log2(np.maximum(f, 1) / 4200) / 0.9) ** 2)
        s = np.fft.irfft(spec, n)
        grit = np.repeat(rng.uniform(0.3, 1.0, n // 96 + 1), 96)[:n]
        env = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.7 * grit
        s = s * env
        s /= np.abs(s).max() + 1e-9
        self.bus.add(s, 0, 0.12 * vel, pan, itd=0.0005 * abs(pan))

    def render(self):
        return self.bus.pull()


AMBIENCE_MAP = {
    "shiphum": (ShipHum, "cryo bay"),
    "pool": (Pool, None),
    "fire": (Fire, "hearth"),
    "room": (RoomTone, None),
    "stream": (Stream, "trickle"),
    "deepsea": (DeepSea, "whale song"),
    "wind": (Wind, "solar wind"),
}


class Ambience(Source):
    """Follows the current scene, crossfading between per-scene soundscapes."""
    key = "ambience"
    label = "SCENE AMBIENCE"
    presets = None
    wet = 0.3

    def __init__(self, rng):
        super().__init__(rng)
        self.gens = {}
        self.active = {}
        self.current = None

    def set_scene(self, amb_key):
        if amb_key == self.current:
            return
        self.current = amb_key
        for k, p in self.active.items():
            p.set(0.0, 1.5)
        if amb_key in AMBIENCE_MAP:
            if amb_key not in self.gens:
                cls, preset = AMBIENCE_MAP[amb_key]
                g = cls(self.rng)
                if preset:
                    g.set_preset(preset)
                self.gens[amb_key] = g
            p = self.active.get(amb_key) or Param(0.0, 1.5)
            p.set(1.0, 1.5)
            self.active[amb_key] = p

    def render(self):
        out = np.zeros((2, BLOCK))
        for k in list(self.active):
            p = self.active[k]
            g = p.ramp()
            if p.v < 1e-4 and p.target == 0.0:
                del self.active[k]
                continue
            out += self.gens[k].render() * g
        return out


LAYERS = [Drone, Noise, Ambience, Rain, Ocean, Wind, Stream, Crickets, Fire, DeepSea, ShipHum, Taps]
LAYER_KEYS = [c.key for c in LAYERS]
LAYER_LABELS = {c.key: c.label for c in LAYERS}
LAYER_PRESETS = {c.key: c.presets for c in LAYERS}
