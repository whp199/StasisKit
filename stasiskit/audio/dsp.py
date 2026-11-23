"""Vectorised DSP building blocks (numpy only - no scipy needed).

Everything runs in fixed blocks of BLOCK samples at SR Hz, stereo arrays are
shaped (2, BLOCK).
"""

import math

import numpy as np

SR = 48000
BLOCK = 2048
NFFT = BLOCK * 2
FREQS = np.fft.rfftfreq(NFFT, 1.0 / SR)          # (BLOCK + 1,)
F_SAFE = np.maximum(FREQS, 1.0)
TAU = 2.0 * math.pi
_T_BLOCK = np.arange(BLOCK) / SR


def pan_gains(pan):
    """Constant-power pan, pan in [-1, 1] (scalar or array) -> (gl, gr)."""
    a = (np.clip(pan, -1, 1) + 1.0) * (math.pi / 4)
    return np.cos(a), np.sin(a)


# -- spectral shapes ---------------------------------------------------------
def tilt(alpha):
    """Magnitude ~ f^alpha (alpha=-0.5 pink, -1 brown...)."""
    return F_SAFE ** alpha


def highpass(fc, order=2):
    r = (F_SAFE / fc) ** (2 * order)
    return np.sqrt(r / (1.0 + r))


def lowpass(fc, order=2):
    r = (F_SAFE / fc) ** (2 * order)
    return np.sqrt(1.0 / (1.0 + r))


def bandpass(lo, hi, order=2):
    return highpass(lo, order) * lowpass(hi, order)


def peak(fc, width_oct=0.3, gain=1.0):
    """Gaussian bump on a log-frequency axis."""
    d = np.log2(F_SAFE / fc) / width_oct
    return gain * np.exp(-0.5 * d * d)


def normalize(mag, rms):
    """Scale a magnitude spectrum so SpectralNoise output has roughly this RMS."""
    e = float(np.sum(mag * mag))
    if e <= 0:
        return mag * 0
    return mag * (rms * NFFT / (2.0 * math.sqrt(e)))


class SpectralNoise:
    """Noise with an arbitrary, per-block magnitude spectrum.

    Random-phase STFT synthesis with a sine window at 50% overlap: the squared
    windows sum to one, so independent frames overlap into stationary noise.
    Changing the spectrum every block gives smooth time-varying filtering.
    """

    _win = np.sin(np.pi * (np.arange(NFFT) + 0.5) / NFFT)

    def __init__(self, rng, channels=2):
        self.rng = rng
        self.c = channels
        self.tail = np.zeros((channels, BLOCK))

    def render(self, mag):
        rng = self.rng
        shape = (self.c, BLOCK + 1)
        spec = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * mag
        frame = np.fft.irfft(spec, n=NFFT, axis=-1) * self._win
        out = self.tail + frame[:, :BLOCK]
        self.tail = frame[:, BLOCK:]
        return out


class Param:
    """Parameter with exponential smoothing and per-block linear ramps."""

    def __init__(self, value=0.0, tau=0.25):
        self.v = float(value)
        self.target = float(value)
        self.tau = tau

    def set(self, target, tau=None):
        self.target = float(target)
        if tau is not None:
            self.tau = tau

    def jump(self, value):
        self.v = self.target = float(value)

    @property
    def settled(self):
        return abs(self.v - self.target) < 1e-5

    def ramp(self):
        old = self.v
        if self.v != self.target:
            a = 1.0 - math.exp(-BLOCK / SR / max(1e-4, self.tau))
            self.v += (self.target - self.v) * a
            if abs(self.v - self.target) < 1e-5:
                self.v = self.target
        if old == self.v:
            return old
        return old + (self.v - old) * (np.arange(BLOCK) / BLOCK)


class Partials:
    """Bank of sine oscillators; frequency fixed per block, amplitude ramped."""

    def __init__(self, n, rng):
        self.n = n
        self.phase = rng.uniform(0, TAU, n)
        self.prev_amp = np.zeros(n)
        self._ramp = (np.arange(BLOCK) / BLOCK)[None, :]
        self._idx = np.arange(BLOCK)[None, :]

    def render(self, freq, amp, pan):
        freq = np.asarray(freq, np.float64)
        amp = np.asarray(amp, np.float64)
        inc = TAU * freq / SR
        live = (amp > 1e-6) | (self.prev_amp > 1e-6)
        out = np.zeros((2, BLOCK))
        if live.any():
            idx = np.flatnonzero(live)
            ph = self.phase[idx, None] + inc[idx, None] * self._idx
            a0 = self.prev_amp[idx, None]
            a = a0 + (amp[idx, None] - a0) * self._ramp
            sa = np.sin(ph) * a
            gl, gr = pan_gains(np.asarray(pan)[idx] if np.ndim(pan) else np.full(len(idx), pan))
            out[0] = gl @ sa
            out[1] = gr @ sa
        self.phase = (self.phase + inc * BLOCK) % TAU
        self.prev_amp = amp.copy()
        return out


class RandomWalks:
    """Vectorised smooth random signals in [0, 1] (doubly smoothed jumps)."""

    def __init__(self, n, rng, rate=0.05, smooth=4.0):
        self.rng = rng
        self.rate = rate          # target changes per second (per signal)
        self.target = rng.random(n)
        self.v1 = self.target.copy()
        self.v2 = self.target.copy()
        self.a = 1.0 - math.exp(-BLOCK / SR / smooth)

    def step(self):
        p = self.rate * BLOCK / SR
        flip = self.rng.random(len(self.target)) < p
        if flip.any():
            self.target[flip] = self.rng.random(int(flip.sum()))
        self.v1 += (self.target - self.v1) * self.a
        self.v2 += (self.v1 - self.v2) * self.a
        return self.v2


class GrainBus:
    """Ring buffer that one-shot sounds are mixed into ahead of time."""

    def __init__(self, seconds=16.0):
        size = 1
        while size < seconds * SR + BLOCK:
            size <<= 1
        self.size = size
        self.buf = np.zeros((2, size))
        self.pos = 0

    def add(self, grain, offset=0, gain=1.0, pan=0.0, itd=0.0):
        """Mix a mono (L,) or stereo (2, L) grain starting `offset` samples into
        the current block. `itd` delays the far ear in seconds (binaural cue)."""
        g = np.asarray(grain, np.float64)
        if g.ndim == 1:
            gl, gr = pan_gains(pan)
            dl = dr = 0
            if itd:
                d = int(abs(itd) * SR)
                if pan > 0:
                    dl = d
                else:
                    dr = d
            self._mix(0, g * (gain * gl), offset + dl)
            self._mix(1, g * (gain * gr), offset + dr)
        else:
            self._mix(0, g[0] * gain, offset)
            self._mix(1, g[1] * gain, offset)

    def _mix(self, ch, sig, offset):
        n = min(len(sig), self.size - BLOCK)
        start = (self.pos + int(offset)) % self.size
        end = start + n
        if end <= self.size:
            self.buf[ch, start:end] += sig[:n]
        else:
            k = self.size - start
            self.buf[ch, start:] += sig[:k]
            self.buf[ch, :n - k] += sig[k:n]

    def pull(self):
        p = self.pos
        out = self.buf[:, p:p + BLOCK].copy()
        self.buf[:, p:p + BLOCK] = 0.0
        self.pos = (p + BLOCK) % self.size
        return out


def make_ir(seconds, rng, bands=None, predelay=0.018):
    """Synthetic stereo hall impulse response: band-wise exponentially decaying
    noise (highs die first) plus a few early reflections."""
    L = int(seconds * SR)
    t = np.arange(L) / SR
    if bands is None:
        bands = [(0, 250, seconds * 0.62), (250, 1000, seconds * 0.52), (1000, 3500, seconds * 0.36),
                 (3500, 8000, seconds * 0.2), (8000, SR / 2, seconds * 0.1)]
    f = np.fft.rfftfreq(L, 1.0 / SR)
    ir = np.zeros((2, L))
    for c in range(2):
        spec = np.fft.rfft(rng.standard_normal(L))
        for lo, hi, t60 in bands:
            m = (f >= lo) & (f < hi)
            sub = np.zeros_like(spec)
            sub[m] = spec[m]
            ir[c] += np.fft.irfft(sub, n=L) * np.exp(-6.9 * t / max(0.05, t60))
        # soft onset
        ir[c] *= np.minimum(1.0, t / 0.04) ** 1.5
        for _ in range(6):
            d = int(rng.uniform(0.008, 0.07) * SR)
            ir[c, d] += rng.uniform(-0.6, 0.6) * 4.0
    pd = int(predelay * SR)
    ir = np.concatenate([np.zeros((2, pd)), ir[:, :L - pd]], axis=1)
    ir /= math.sqrt(float(np.sum(ir * ir)) / 2.0) + 1e-12
    return ir


class Reverb:
    """Uniformly partitioned overlap-save FFT convolution (mono in, stereo out)."""

    def __init__(self, rng, seconds=4.5):
        ir = make_ir(seconds, rng)
        K = int(math.ceil(ir.shape[1] / BLOCK))
        pad = K * BLOCK - ir.shape[1]
        ir = np.pad(ir, ((0, 0), (0, pad))).reshape(2, K, BLOCK)
        self.H = np.fft.rfft(np.concatenate([ir, np.zeros_like(ir)], axis=2), axis=2).astype(np.complex64)
        self.K = K
        self.fdl = np.zeros((K, BLOCK + 1), np.complex64)
        self.p = 0
        self.prev = np.zeros(BLOCK)
        self._order = np.arange(K)

    def process(self, x):
        buf = np.concatenate([self.prev, x])
        self.prev = x.copy()
        self.p = (self.p + 1) % self.K
        self.fdl[self.p] = np.fft.rfft(buf).astype(np.complex64)
        idx = (self.p - self._order) % self.K
        Y = np.einsum("kf,ckf->cf", self.fdl[idx], self.H)
        return np.fft.irfft(Y, n=NFFT, axis=1)[:, BLOCK:]


def soft_clip(x):
    return np.tanh(x)
