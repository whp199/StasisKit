"""Sacred sound for the CATHEDRAL view: the "organ" drone and the "cathedral" ambience.

Organ: a quiet pipe organ improvising during night adoration in a stone church.
Every pipe is a phase-continuous wavetable oscillator.  Its single-cycle table
comes from the harmonic recipe of its stop: a stopped Gedackt (strong
fundamental, weak odd harmonics), a Salicional and a Voix celeste (many gentle
string harmonics; the celeste is tuned a few cents sharp so the pair beats
slowly), a soft Principal, a 16' Subbass with an 8' Bourdon in the pedal, and an
open flute for the chant.  A pipe speaks with a smoothstep attack (bass and
string pipes slower), a faint chiff grain (the twelfth of a stopped pipe, the
octave of an open one) and a breath of wind noise around its harmonics, and it
stops with a short exponential release.  The Swell division sits behind a swell
box: an STFT shelving low-pass whose opening and level follow slow phrase arcs.
Pipes on the C and C# chests are panned to opposite sides, as in a real case.

The harmony is a slow modal improvisation in the Gregorian / French-organ manner
(Dorian, Phrygian, Mixolydian, Aeolian, Ionian, and F with B-flat).  A phrase of
three to six chords starts and ends on the final, often over a pedal point:
open fifths, sus2 and sus4 (often resolving within the chord), added ninths,
sevenths and quartal chords.  Only tritone-free sonorities are used, so nothing
ever sounds like a dominant cadence.  Each change is voice-led by a small search
over four voices (common tones held, the others moving by step where they can,
leaping when no nearby voicing fits), every 12-25 s.  No voice sits under the
pedal or a step above it, and low seconds and minor ninths are avoided.

Every few minutes the texture thins to an open fifth on the mode's final (in
Phrygian the final alone: its fifth B would grate on the reciting tone C) and
a soft flute plays a real Gregorian chant (see CHANTS), slowly: dotted notes
and phrase ends are lengthened, there are breaths at the bars, and each
syllable is lightly articulated while a neume is slurred.  The cathedral view
hints the theme with Drone events ("hymn", theme="marian" | "eucharistic" |
"mercy"), so the next chant fits what is on screen.  Without hints it goes
through them all.

Besides a light send to the mixer's hall reverb the organ has its own long,
dark cathedral tail.  It is a ~9 s low-band (< 3.3 kHz) partitioned
convolution, and its impulse response is synthesised a few partitions per
block, so creating the organ never stalls the audio.  As at a listener's two
ears, its channels share most of the bass, so the held pipes stay centred.  A
slow leveller (like an organist easing the swell when the chords pile up) and
a soft knee on the rare coincident peaks keep its loudness and peaks in line
with the other drones.

Cathedral air: the room tone of a vast stone nave that slowly breathes, distant
wind sighing against the windows, a faint flutter of candle flames when a draft
passes, and rosary beads (event "bead") touching close by.  Each bead click has
early reflections and a long statistical reverb tail: an energy spectrum that
decays band by band (highs first) and is resynthesised as noise.
"""

import heapq
import itertools
import math
import re

import numpy as np

from .dsp import (BLOCK, F_SAFE, NFFT, SR, TAU, GrainBus, Param, RandomWalks, SpectralNoise, bandpass, highpass,
                  lowpass, normalize, pan_gains, peak, tilt)

DT = BLOCK / SR
_IDX = np.arange(BLOCK)
_TT = _IDX / SR                                                 # sample times within a block
_WIN = np.sin(np.pi * (np.arange(NFFT) + 0.5) / NFFT)           # sin^2 overlap-adds to 1 at 50 %


def _hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def _num(v, lo, hi, default):
    """Event argument -> float in [lo, hi]; anything odd (None, text, NaN, inf, arrays) gives `default`."""
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(x):
        return default
    return min(max(x, lo), hi)


# =============================================================================
# GREGORIAN CHANT
# =============================================================================
# Public-domain melodies as GABC neume groups, copied from GregoBase
# (https://gregobase.selapa.net/chant.php?id=N, transcriptions released as CC0)
# with the syllable text left out: one (...) group per syllable, bars as (,) (;)
# (:) (::), clef first.  Each was checked against the Latin text (one group per
# syllable), and Adoro te and Tantum ergo also against a second GregoBase
# edition.  With the c4 clef, j = C5: the melodies lie between C4 and D5.
CHANTS = {
    "salve": dict(
        theme="marian", mode="ionian", repeat=1,
        src="Salve Regina, simple tone, mode V (on do) - GregoBase #8127, Chants of the Church (1956) p. 138",
        gabc="(c4) (c) (e) (g') (h) (g.) (,) (h) (j') (i) (h) (g) (h') (g) (g.) (;) (j) (g') (h) (ff) (d.) (,) "
             "(e) (f) (g') (e) (ed) (c.) (::) (g) (h) (i') (j) (g.) (,) (h) (i) (j) (i') (h) (g') (h) (g.) (::) "
             "(j) (g') (h) (f') (d) (e'_) (,) (e) (g') (h) (j') (h) (g'_) (,) (h) (g') (f) (e') (d) (e') (d) "
             "(c.) (::) (g) (h') (i) (j'_) (,) (g) (h') (j) (i') (h) (g.) (;) (j) (g') (h) (f'_) (d) (e') (f) "
             "(g') (f) (h') (g) (g'_) (`) (f) (e') (d) (ed) (c.) (::) (g) (hi) (jj) (,) (i) (g') (h) (g') (g) "
             "(h') (j) (i') (h) (g.) (;) (c) (g'_) (h) (j') (i) (h') (g) (e'_) (f) (ed) (c.) (::) (e!fwg) (ee) "
             "(c.) (::) (g!hwi/ji) (hg) (g.) (::) (jjj//ghf'fd) (ef) (g'_) (,) (c) (f') (e) (dc) (c.) (::)"),
    "ave": dict(
        theme="marian", mode="dorian", repeat=1,
        src="Ave Maria (Salutatio Angelica), mode I - GregoBase #1860, Liber Usualis (1961) p. 1861",
        gabc="(c4) (f) (c') (d) (ixdh'!iv) (h.) (,) (h') (g) (f) (g_[uh:l]h) (h.) (,) (h') (g) (f') (g) (d.) (;) "
             "(g) (g) (h') (g) (f.) (,) (ef) (g) (fe) (d) (cd) (d.) (;) (e) (c) (d') (f) (f') (e) (f') (g) (e') "
             "(g) (fe) (cd) (df..) (::) (h) (g') (h) (ji) (g.) (g) (f) (gh) (h.) (;) (h) (g') (f) (gf) (d) (f) "
             "(fe) (de) (d) (c.) (;) (d) (c) (df) (ffg) (f) (e') (f) (gh) (fe) (d.) (d.) (::)"),
    "adoro": dict(
        theme="eucharistic", mode="ionian", repeat=1,
        src="Adoro te devote, mode V, first stanza - GregoBase #7479, Cantus selecti (1957) p. 11*; "
            "agrees with #3020, Liber Usualis (1961) p. 1855",
        gabc="(c4) (c) (e') (g) (g') (g) (gh) (g.) (,) (f') (e) (d') (c) (c.) (;) (c') (e) (g') (g) (gh) (g.) (,) "
             "(f') (e) (d') (c) (c.) (:) (g') (g) (hi) (j) (ji) (g.) (,) (h') (g) (f') (e) (d.) (;) (e') (g) (f') "
             "(e) (dc) (d.) (,) (e') (c) (d') (c) (c.) (::)"),
    "tantum": dict(
        theme="eucharistic", mode="phrygian", repeat=1,
        src="Tantum ergo (Pange lingua tone), mode III - GregoBase #2176, Graduale Romanum (1961) p. 158* and "
            "Liber Usualis p. 954; agrees with #8206, Graduale simplex (1975) p. 463",
        gabc="(c4) (e) (e') (f) (e[ll:1]d) (g') (g) (hj) (j.) (,) (jk) (j) (j') (i) (h') (j) (ivHG.) (:) (g) (h') "
             "(j) (i') (h) (g') (h) (g.) (,) (h') (i) (g') (g) (e') (h) (hd..) (:) (e) (g') (g) (e') (g) (h') (h) "
             "(g.) (,) (h) (i') (g) (hg) (fe) (d) (e.) (::)"),
    "salutaris": dict(
        theme="eucharistic", mode="mixolydian", repeat=1,
        src="O salutaris Hostia I, mode VIII - GregoBase #3034, Chants of the Church (1956) p. 123",
        gabc="(c4) (g) (gh!jvIH) (gf) (hg) (hih) (g') (f) (g.) (;) (g) (i') (j) (k) (ji) (hg) (h) (j.) (:) (j) (h') "
             "(j) (g) (gh) (hg) (f) (g.) (;) (f) (hg) (hj) (ji) (hg) (e') (f) (g.) (::)"),
    "parce": dict(
        theme="mercy", mode="dorian", repeat=2,
        src="Parce Domine (refrain), mode I - GregoBase #3038, Liber Usualis (1961) p. 1868",
        gabc="(c4) (h') (h) (h') (g) (gf..) (,) (h') (h) (h') (g) (gf) (gf) (e[ll:1]d..) (;) (f') (e) (f') (g) (h.) "
             "(e') (f) (gf) (e) (fe) (d.) (::)"),
    "attende": dict(
        theme="mercy", mode="ionian", repeat=2,
        src="Attende Domine (refrain), mode V (on do) - GregoBase #3021, Liber Usualis (1961) p. 1871",
        gabc="(c4) (c) (e') (g) (g') (g) (g.) (j') (i) (g') (h) (g.) (;) (j) (g') (d) (f') (g) (e) (e[ll:1]d) "
             "(c.) (::)"),
}
THEMES = ("marian", "eucharistic", "mercy")
HINT_SECS = 3600.0           # a theme hint is followed for up to an hour

_PITCH = "abcdefghijklm"                    # GABC staff positions, a = lowest
_STEP = (0, 2, 4, 5, 7, 9, 11)              # semitones of C D E F G A B
_BREATH = {"`": 0.3, ",": 0.45, ";": 0.75, ":": 1.1, "::": 1.6}    # rest after each bar, in note units


def _clef(spec):
    """'c4', 'cb3', 'f3'... -> (staff position of the clef line, its degree from do, B-flat positions) or None."""
    m = re.fullmatch(r"([cf])(b?)([1-4])", spec)
    if not m:
        return None
    line = 1 + 2 * int(m.group(3))            # staff position of the clef line (lines are d f h j)
    deg0 = 0 if m.group(1) == "c" else -4     # the line is do (C), or fa (F) a fifth below it
    key = {p for p in range(13) if (p - line + deg0) % 7 == 6} if m.group(2) else set()
    return line, deg0, key


def parse_gabc(gabc, do=72):
    """GABC neume groups -> [("syl", [[midi, units], ...]) | ("bar", kind), ...].

    Clef c1-c4 (cb = with B-flat) or f1-f4, also changing mid-chant; the C of a
    c-clef line is MIDI `do`, the F of an f-clef line the F below it.  Solesmes
    rhythm: every note is one unit, a punctum mora (.) doubles it ('..' dots the
    last two notes), a horizontal episema (_) lengthens a little, the note before
    a quilisma (w, W) is lengthened, and repeated pitches inside one neume
    (strophae) are tied.  Accidentals (x flat, y natural) last to the next bar.
    A custos (g+) only shows the next pitch, so it is not sung.  Bracketed
    annotations, ictus marks, virgae and spacing are ignored."""
    groups = re.findall(r"\(([^)]*)\)", gabc)
    clef = _clef(groups[0].strip()) if groups else None
    if not clef:
        raise ValueError("GABC must start with a clef group such as (c4)")
    line, deg0, key = clef
    flats = set(key)
    out = []
    for g in groups[1:]:
        g = re.sub(r"\[[^\]]*\]", "", g).strip()
        g = re.sub(r"[a-m]\+", "", g)                                     # custos
        m = re.fullmatch(r"(.*?)([cf]b?[1-4])", g)
        if m and m.group(1).rstrip("zZ0123456789+-") in ("", *_BREATH):
            line, deg0, key = _clef(m.group(2))                           # a new clef (after a bar, maybe)
            flats = set(key)
            g = m.group(1)
        bar = g.rstrip("zZ0123456789+-")
        if bar in _BREATH:
            out.append(("bar", bar))
            flats = set(key)
            continue
        notes = []
        i = 0
        while i < len(g):
            c = g[i]
            p = _PITCH.find(c.lower())
            if p >= 0:
                if i + 1 < len(g) and g[i + 1] in "xy#":          # an accidental, not a note
                    (flats.add if g[i + 1] == "x" else flats.discard)(p)
                    i += 2
                    continue
                o, d = divmod(p - line + deg0, 7)
                notes.append([do + 12 * o + _STEP[d] - (p in flats), 1.0])
            elif c == "." and notes:
                if g[i:i + 2] == "..":
                    for nt in notes[-2:]:
                        nt[1] = 2.0
                    i += 1
                else:
                    notes[-1][1] = 2.0
            elif c == "_" and notes:
                notes[-1][1] *= 1.33
            elif c in "wW" and len(notes) >= 2:
                notes[-2][1] *= 1.5
            i += 1
        tied = []
        for nt in notes:
            if tied and tied[-1][0] == nt[0]:
                tied[-1][1] += nt[1]
            else:
                tied.append(nt)
        if tied:
            out.append(("syl", tied))
    return out


def perform(events, unit, rng, t0=0.0, repeat=1):
    """Timed flute notes [(t_on, t_off, midi, articulated)] for a parsed chant, and its end time.

    `unit` is the length of a plain note in seconds.  Phrase-final notes are
    stretched, the note before a bar is released early for the breath, notes
    inside a neume are slurred, and a repeated pitch is re-struck."""
    seq = []                                   # [t_on, dur, midi, articulated, before_bar]
    t = t0
    for r in range(repeat):
        if r:
            t += 1.4 * unit
        for kind, val in events:
            if kind == "bar":
                if seq and not seq[-1][4]:
                    last = seq[-1]
                    extra = last[1] * (0.35 if last[1] < 1.9 * unit else 0.12)
                    last[1] += extra
                    last[4] = True
                    t += extra
                t += _BREATH[val] * unit
                continue
            for j, (midi, units) in enumerate(val):
                dur = unit * units * rng.uniform(0.97, 1.03)
                seq.append([t, dur, midi, j == 0, False])
                t += dur
    if seq:
        seq[-1][1] *= 1.6                        # let the last note ring
        t = max(t, seq[-1][0] + seq[-1][1])
    notes = []
    for k, (ton, dur, midi, art, bar) in enumerate(seq):
        nxt = seq[k + 1] if k + 1 < len(seq) else None
        if bar or nxt is None:
            toff = ton + dur * 0.93
        elif nxt[2] == midi:
            toff = nxt[0] - 0.09
        else:
            toff = nxt[0] + 0.03                 # legato overlap
        notes.append((ton, toff, midi, art))
    return notes, t


# =============================================================================
# PIPES
# =============================================================================
TN = 2048                                       # samples per single-cycle table
SWELL, PEDAL, SOLO = 0, 1, 2


def _recipe_pedal():
    """16' Subbass (stopped) + 8' Bourdon (stopped, an octave up) + a little Salicional coupled at 8'."""
    h = {1: 0.62, 2: 0.025, 3: 0.06, 5: 0.02, 7: 0.006}
    for k, a in {1: 0.6, 2: 0.014, 3: 0.055, 5: 0.014}.items():
        h[2 * k] = h.get(2 * k, 0.0) + a
    for k, a in enumerate(_SALICIONAL[:10], 1):
        h[2 * k] = h.get(2 * k, 0.0) + 0.07 * a
    return h


_SALICIONAL = [0.5, 0.56, 0.5, 0.43, 0.37, 0.31, 0.27, 0.23, 0.19, 0.16, 0.14, 0.12, 0.10, 0.085, 0.072, 0.061,
               0.052, 0.044, 0.037, 0.031, 0.026, 0.022, 0.018, 0.015]


class _Rank:
    def __init__(self, name, harmonics, level, attack, release, detune, div, pan_k, chiff, breath):
        self.name, self.level, self.attack, self.release = name, level, attack, release
        self.detune, self.div, self.pan_k = detune, div, pan_k
        self.chiff = chiff            # (harmonic, tone amp, tone decay, noise amp, noise decay) or None
        self.breath = breath          # (((harmonic, weight), ...), broadband weight, rms per unit level)
        self.harmonics = harmonics


RANKS = [
    _Rank("gedackt 8'", {1: 1.0, 2: 0.025, 3: 0.19, 4: 0.012, 5: 0.06, 6: 0.006, 7: 0.024, 9: 0.009, 11: 0.004},
          1.0, 0.05, 0.030, 0.0, SWELL, 1.0, (3, 0.05, 0.022, 0.03, 0.012), (((1, 1.0), (3, 0.5)), 0.0, 0.006)),
    _Rank("salicional 8'", dict(enumerate(_SALICIONAL, 1)),
          0.48, 0.13, 0.035, 0.0, SWELL, 1.0, None, (((2, 0.4), (4, 0.3)), 0.25, 0.0035)),
    _Rank("voix celeste 8'", {k: a * 0.94 ** k for k, a in enumerate(_SALICIONAL, 1)},
          0.44, 0.18, 0.035, 3.6, SWELL, -0.6, None, (((2, 0.4), (4, 0.3)), 0.25, 0.0035)),
    _Rank("principal 8'", {1: 1.0, 2: 0.42, 3: 0.22, 4: 0.14, 5: 0.09, 6: 0.06, 7: 0.04, 8: 0.03, 9: 0.02, 10: 0.015},
          0.32, 0.07, 0.030, 0.0, SWELL, 0.5, (2, 0.04, 0.018, 0.02, 0.01), (((1, 1.0), (2, 0.6)), 0.1, 0.004)),
    _Rank("subbass 16' + bourdon 8'", _recipe_pedal(),
          0.44, 0.16, 0.060, 0.0, PEDAL, 1.0, (-1, 0.0, 0.0, 0.05, 0.045), (((2, 1.0), (4, 0.4)), 0.0, 0.004)),
    _Rank("flute 4' (8' octave)", {1: 1.0, 2: 0.2, 3: 0.075, 4: 0.03, 5: 0.014, 6: 0.006},
          1.0, 0.045, 0.035, 0.0, SOLO, 0.0, (2, 0.09, 0.020, 0.06, 0.010), (((1, 1.0), (2, 0.5), (3, 0.2)), 0.07, 0.009)),
]
GEDACKT, SALICIONAL, CELESTE, PRINCIPAL, PEDALR, FLUTE = range(6)


def _tables():
    """All rank tables stacked into one flat float32 array (each TN + 1 long, wrap sample at the end).
    Harmonic phases are the least peaky of a few fixed random draws, so chords keep a low crest factor."""
    rng = np.random.default_rng(7)               # fixed: the organ always has the same pipes
    out = []
    for rk in RANKS:
        ks = np.array(sorted(rk.harmonics))
        amp = np.array([rk.harmonics[k] for k in ks])
        best = None
        for _ in range(24):
            spec = np.zeros(TN // 2 + 1, complex)
            spec[ks] = amp * np.exp(1j * rng.uniform(0, TAU, len(ks)))
            tab = np.fft.irfft(spec, TN)
            crest = np.abs(tab).max() / tab.std()
            if best is None or crest < best[0]:
                best = (crest, tab)
        tab = best[1] * (0.5 / best[1].std())      # every table has RMS 0.5
        out.append(np.append(tab, tab[0]))
    return np.concatenate(out).astype(np.float32)


_TABLES = _tables()


class _Note:
    __slots__ = ("rank", "midi", "f", "ph", "lvl", "ton", "toff", "ta", "tr", "gl", "gr", "chiff")


# =============================================================================
# HARMONY
# =============================================================================
# chord types as stacks of scale degrees above the root
_TYPES = {"5": (0, 4), "sus2": (0, 1, 4), "sus4": (0, 3, 4), "triad": (0, 2, 4), "add9": (0, 2, 4, 1),
          "7": (0, 2, 4, 6), "7sus4": (0, 3, 4, 6), "9": (0, 2, 4, 6, 1), "quartal": (0, 3, 6)}
_RESOLVE = {"sus4": "triad", "7sus4": "7", "sus2": "triad"}
_TONIC = {"add9": 1.0, "sus2": 0.7, "5": 0.55, "7": 0.7, "sus4": 0.6, "triad": 0.35, "9": 0.35}
_COLOUR = {"triad": 1.0, "7": 0.9, "add9": 0.6, "sus2": 0.5, "sus4": 0.45, "7sus4": 0.35, "quartal": 0.35, "9": 0.3}

# name: (final pitch class, scale above the final, chord roots {degree: weight}, approach chords before the final)
MODES = {
    "dorian":     (2, (0, 2, 3, 5, 7, 9, 10), {1: .5, 2: 1.0, 3: 1.1, 4: .8, 6: 1.2}, {6: 1.0, 3: .8, 4: .5, 1: .4}),
    "phrygian":   (4, (0, 1, 3, 5, 7, 8, 10), {1: .3, 2: .8, 3: 1.1, 5: .9, 6: 1.0}, {6: 1.0, 3: .9, 1: .4}),
    "lydian":     (5, (0, 2, 4, 5, 7, 9, 11), {1: .9, 2: .5, 3: 1.2, 4: .6, 5: .9}, {3: 1.0, 1: .8, 4: .4, 5: .4}),
    "mixolydian": (7, (0, 2, 4, 5, 7, 9, 10), {1: .7, 3: 1.1, 4: .9, 5: .6, 6: 1.2}, {6: 1.0, 3: .9, 4: .6, 1: .4}),
    "aeolian":    (9, (0, 2, 3, 5, 7, 8, 10), {2: 1.0, 3: 1.0, 4: .7, 5: 1.1, 6: 1.0}, {6: 1.0, 3: .8, 5: .7}),
    "ionian":     (0, (0, 2, 4, 5, 7, 9, 11), {1: 1.0, 2: .6, 3: 1.2, 4: .7, 5: 1.0}, {3: 1.0, 1: .8, 4: .5, 5: .4}),
}
_CENTRES = {"dorian": 1.2, "mixolydian": 1.0, "lydian": 1.0, "aeolian": 0.8, "ionian": 0.8, "phrygian": 0.45}

_RANGES = ((47, 59), (48, 63), (53, 67), (57, 70))     # the four Swell voices (MIDI, sounding 8')
_IDEAL = (50, 55, 60, 64)
_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


class _Chord:
    __slots__ = ("deg", "kind", "pcs", "bass", "fifth", "root", "resolve")

    def __init__(self, mode, deg, kind, bass):
        final, steps = MODES[mode][0], MODES[mode][1]
        self.deg, self.kind, self.bass = deg, kind, bass
        self.pcs = tuple(sorted({(final + steps[(deg + d) % 7]) % 12 for d in _TYPES[kind]}))
        self.root = (final + steps[deg]) % 12
        self.fifth = (final + steps[(deg + 4) % 7]) % 12
        self.resolve = None

    def ok(self, mode):
        steps = MODES[mode][1]
        if (steps[(self.deg + 4) % 7] - steps[self.deg]) % 12 != 7:     # diminished root
            return False
        s = set(self.pcs) | {self.bass}
        if any((a - b) % 12 == 6 for a in s for b in s):                # no tritone: nothing dominant
            return False
        return all((p - self.bass) % 12 != 1 for p in self.pcs)         # no minor ninth over the bass

    def name(self):
        return f"{_NAMES[self.root]}{self.kind}/{_NAMES[self.bass]}"


def _wchoice(rng, weights):
    keys = list(weights)
    w = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


# =============================================================================
# THE ORGAN (drone preset "organ")
# =============================================================================
class _DarkTail:
    """Long, dark cathedral tail: a low-band (< fmax) uniformly partitioned
    convolution, mono in, stereo out.  Its impulse response is decaying noise
    (low frequencies ring longest) synthesised a few partitions per block, so
    building it never stalls the audio; until it is complete the tail is just
    shorter.  Like a diffuse field at two ears ~20 cm apart, the channels share
    most of their reverberation in the bass, so the steady pipes it colours do
    not pull the organ to one side.  Each partition fades in and out over 32
    samples and its top bins taper off, so dropping the bins above fmax keeps
    the convolution time-invariant (no block-rate artefacts)."""

    def __init__(self, rng, seconds=9.0, fmax=3300.0, per_block=12):
        self.K = int(seconds * SR / BLOCK)
        self.nb = int(fmax * NFFT / SR) + 1
        self.H = np.zeros((2, self.K, self.nb), np.complex64)
        self.fdl = np.zeros((self.K, self.nb), np.complex64)
        self.p = 0
        self.prev = np.zeros(BLOCK)
        self.order = np.arange(self.K)
        self.built = 0
        self.per_block = per_block
        self.noise = SpectralNoise(rng)
        self.common = SpectralNoise(rng, channels=1)
        kd = TAU * F_SAFE * 0.2 / 343.0
        c = np.clip(np.sin(kd) / kd, 0.0, 1.0)                  # interaural coherence of a diffuse field
        self.coh, self.sep = np.sqrt(c), np.sqrt(1.0 - c)
        self.edge = np.ones(BLOCK)
        self.edge[:32] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(32) + 0.5) / 32)
        self.edge[-32:] = self.edge[31::-1]
        self.top = np.ones(self.nb)
        self.top[-32:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(1, 33) / 33)
        f = F_SAFE
        t60 = 8.5 / (1.0 + (f / 600.0) ** 0.9)                 # ~8 s at 100 Hz, 3.3 s at 1 kHz
        self.decay = 10.0 ** (-3.0 * DT / t60)                  # amplitude per partition
        self.colour = lowpass(1200, 4) * highpass(45, 2)
        tk = (np.arange(self.K) + 0.5) * DT
        x = np.clip((tk - 0.07) / 0.35, 0.0, 1.0)               # 70 ms pre-delay, slow bloom
        self.onset = x * x * (3 - 2 * x)
        # A block of SpectralNoise with magnitude m has E|FFT|^2 ~ m^2 per bin, so a steady tone at f
        # gains sum_k m_k(f)^2 in power.  Normalise that to 1 at 200 Hz (longer T60 below: more bass).
        i = int(200.0 * NFFT / SR)
        g2 = float(np.sum((self.colour[i] * self.decay[i] ** np.arange(self.K) * self.onset) ** 2))
        self.scale = 1.0 / math.sqrt(max(g2 * float(np.mean(self.edge ** 2)), 1e-12))

    def _build(self, n):
        z = np.zeros((2, BLOCK))
        for _ in range(n):
            k = self.built
            if k >= self.K:
                return
            mag = self.colour * (self.decay ** k) * (self.onset[k] * self.scale)
            blk = (self.common.render(mag * self.coh) + self.noise.render(mag * self.sep)) * self.edge
            self.H[:, k] = np.fft.rfft(np.concatenate([blk, z], axis=1), axis=1)[:, :self.nb] * self.top
            self.built += 1

    def process(self, x):
        if self.built < self.K:
            self._build(self.per_block)
        buf = np.concatenate([self.prev, x])
        self.prev = x
        self.p = (self.p + 1) % self.K
        self.fdl[self.p] = np.fft.rfft(buf)[:self.nb]
        Y = np.einsum("kf,ckf->cf", self.fdl[(self.p - self.order) % self.K], self.H)
        full = np.zeros((2, NFFT // 2 + 1), np.complex64)
        full[:, :self.nb] = Y
        return np.fft.irfft(full, n=NFFT, axis=1)[:, BLOCK:]


class Organ:
    """A quiet pipe organ improvising modally during night adoration, with Gregorian chant on a flute."""

    GAIN = 0.160          # output level: the generator's RMS matched to the stasis and halo drones
    TAIL = 0.42           # dark cathedral tail level
    WET = 0.2             # the mixer's hall send (other drones 0.5): its own tail is its cathedral, and
                          # more hall would pull the held chords to one side (its two channels differ)
    LEVEL = 0.106         # slow leveller: above this short-term RMS (about -19.5 dBFS) it eases off 3:1
    KNEE, CEIL = 0.32, 0.39   # transparent limiter for the rare coincident peaks

    def __init__(self, rng):
        self.rng = r = np.random.default_rng(int(rng.integers(1 << 62)))
        self.t = 0.0                      # start of the block being rendered (seconds)
        self.notes = []                   # sounding / scheduled pipes
        self.q = []                       # timed actions (heap)
        self._seq = itertools.count()
        self.vo = [None] * 4              # Swell voices (MIDI or None)
        self.keys = [dict() for _ in range(4)]     # voice -> {rank: note}
        self.ped, self.ped_note = None, None
        self.reg = {GEDACKT, SALICIONAL, CELESTE}
        self.swell = Param(0.3, 6.0)      # swell box opening 0 (closed) .. 1
        self.lev_ms, self.lev_g = 0.0, 1.0   # leveller: short-term mean square, current gain
        self.sw_prev = np.zeros((2, BLOCK))
        self.sw_tail = np.zeros((2, BLOCK))
        self.bus_sw = GrainBus(2.0)       # chiffs of enclosed pipes
        self.bus_dir = GrainBus(2.0)      # chiffs of the pedal and the solo flute
        self.br_sw = SpectralNoise(r)     # wind noise of the pipes
        self.br_dir = SpectralNoise(r)
        self._bcache = {}
        self.tail = _DarkTail(r)
        self.mode = _wchoice(r, _CENTRES)
        self.state = "improv"
        self.next_chant = r.uniform(100.0, 160.0)
        self.chant = None
        self.hint, self.hint_t = None, -1e9
        self.played = {}
        self.log = None                   # set to a list to record what it plays (tests / curiosity)
        self._at(0.0, self._open)

    # -- scene hints ----------------------------------------------------------
    def event(self, name, theme="", **kw):
        """("hymn", theme="marian" | "eucharistic" | "mercy") from the cathedral view: the next chants
        come from that theme until another hint (an explicit theme=None clears it; hints fade after
        an hour).  Anything else is ignored."""
        if not isinstance(name, str) or name != "hymn":
            return
        if theme is None:
            self.hint = None
        elif isinstance(theme, str) and theme.strip().lower() in THEMES:
            self.hint, self.hint_t = theme.strip().lower(), self.t

    # -- scheduling -----------------------------------------------------------
    def _at(self, t, fn, *args):
        heapq.heappush(self.q, (t, next(self._seq), fn, args))

    def _note(self, rank, midi, t, scale=1.0, chiff=1.0):
        """Schedule a pipe speaking at time t."""
        rk = RANKS[rank]
        r = self.rng
        n = _Note()
        n.rank, n.midi = rank, midi
        f0 = _hz(midi - 12 if rank == PEDALR else midi)             # the pedal table starts at 16' pitch
        n.f = f0 * 2.0 ** ((rk.detune + r.normal(0.0, 0.5 if rank != CELESTE else 0.8)) / 1200.0)
        n.ph = r.random()
        n.lvl = rk.level * scale
        if rank == PEDALR and midi < 40:
            n.lvl *= 10.0 ** (-0.5 * (40 - midi) / 20.0)        # regulate the deepest pipes a little
        n.ton = max(t, self.t)
        n.toff = math.inf
        n.ta = rk.attack * (1.0 + 0.8 * max(0.0, (60 - midi) / 24.0)) * r.uniform(0.9, 1.15)
        n.tr = rk.release
        if rk.div == SOLO:
            pan = 0.08
        else:
            side = 1.0 if midi % 2 == 0 else -1.0                   # C and C# chests
            wide = 0.2 if rk.div == PEDAL else 0.12 + 0.2 * min(max((72 - midi) / 36.0, 0.0), 1.0)
            pan = side * wide * rk.pan_k
        n.gl, n.gr = pan_gains(pan)
        n.chiff = chiff if rk.chiff else 0.0
        self.notes.append(n)
        return n

    def _off(self, n, t):
        if n is not None:
            n.toff = max(t, self.t, n.ton)

    def _key(self, v, midi, t, reg=None):
        """Swell voice v plays `midi` (None = silent) from time t, legato."""
        reg = self.reg if reg is None else reg
        held = self.keys[v]
        if midi == self.vo[v]:
            if midi is not None:                     # same key: follow a registration change
                for rk in list(held):
                    if rk not in reg:
                        self._off(held.pop(rk), t)
                for rk in reg:
                    if rk not in held:
                        held[rk] = self._note(rk, midi, t + self.rng.uniform(0.0, 0.3), chiff=0.5)
            return
        for rk, n in held.items():
            self._off(n, t + 0.03)
        held.clear()
        if midi is not None:
            for rk in reg:
                held[rk] = self._note(rk, midi, t)
        self.vo[v] = midi

    def _pedal(self, midi, t):
        if midi == self.ped:
            return
        self._off(self.ped_note, t + 0.05)
        self.ped_note = self._note(PEDALR, midi, t) if midi is not None else None
        self.ped = midi

    def _log(self, t, what):
        if self.log is not None:
            self.log.append((round(t, 2), what))

    # -- harmony --------------------------------------------------------------
    def _pick(self, mode, degs, kinds, bass, avoid=None, no_lt=False):
        """A random valid chord; no_lt rejects a leading tone (a semitone below the final), so the
        approach to a final is plagal or modal, never a dominant-tonic cadence."""
        r = self.rng
        lt = (MODES[mode][0] - 1) % 12
        for _ in range(40):
            deg = _wchoice(r, {d: w for d, w in degs.items() if d != avoid} or degs)
            kind = _wchoice(r, kinds)
            b = MODES[mode][0] if bass == "final" else None
            ch = _Chord(mode, deg, kind, 0)
            if b is None:
                b = ch.root
                if kind in ("triad", "7", "add9") and r.random() < 0.18:      # first inversion
                    steps = MODES[mode][1]
                    b = (MODES[mode][0] + steps[(deg + 2) % 7]) % 12
            ch.bass = b
            if ch.ok(mode) and not (no_lt and lt in ch.pcs):
                if kind in _RESOLVE and r.random() < 0.6:
                    res = _Chord(mode, deg, _RESOLVE[kind], b)
                    if res.ok(mode):
                        ch.resolve = res
                return ch
        return _Chord(mode, 0, "5", MODES[mode][0])

    def _phrase(self, mode, n, start_tonic=True):
        """A phrase of n chords: tonic, colours (often over a pedal on the final), approach, tonic."""
        r = self.rng
        roots, approach = MODES[mode][2], MODES[mode][3]
        chords = [self._pick(mode, {0: 1.0}, _TONIC, "final")] if start_tonic else []
        pedal = int(r.choice([0, 1, 2, 2, 3]))
        prev = 0
        for i in range(1, n - 1):
            last = i == n - 2
            ch = self._pick(mode, approach if last else roots, _COLOUR,
                            "final" if (i <= pedal and not last) else "root", avoid=prev, no_lt=last)
            chords.append(ch)
            prev = ch.deg
        chords.append(self._pick(mode, {0: 1.0}, _TONIC, "final"))
        return chords

    def _voicing(self, chord, nv):
        """Voice-lead the Swell voices to `chord` with nv voices sounding: every combination of
        chord tones near each voice is scored at once (motion, spacing, rubs, doublings).  No voice
        sits under the pedal's 8' pipe or a step above it, and low seconds and minor ninths cost."""
        need = set(chord.pcs)
        if chord.root == chord.bass and len(need) > 2:
            need.discard(chord.root)                 # the pedal already sounds the root
        if len(need) > nv and chord.fifth in need and chord.fifth != chord.root:
            need.discard(chord.fifth)
        pcs = set(chord.pcs)
        bass = self._bass_midi(chord.bass)
        for reach in (7, 24):                        # chord tones near each voice; leap if none of those will do
            cands = []
            for v in range(nv):
                lo, hi = _RANGES[v]
                p = self.vo[v]
                c = [m for m in range(lo, hi + 1) if m % 12 in pcs and (p is None or abs(m - p) <= reach)]
                cands.append(c or [m for m in range(lo, hi + 1) if m % 12 in pcs])
            G = np.stack(np.meshgrid(*cands, indexing="ij"), -1).reshape(-1, nv)
            D = np.diff(G, axis=1)
            P = G % 12
            miss = np.zeros(len(G), int)
            for pc in need:
                miss += ~(P == pc).any(axis=1)
            ok = (D > 0).all(axis=1)
            if ok.any():
                ok &= miss == miss[ok].min()             # every needed tone, or as many as the voices hold
            clear = ok & ((G == bass) | (G - bass >= 3)).all(axis=1)   # nothing under the pedal's 8' or a step above
            if clear.any():
                ok = clear
                break
        if not ok.any():
            return [min(cands[v], key=lambda m: abs(m - _IDEAL[v])) for v in range(nv)] + [None] * (4 - nv)
        G, D, P = G[ok], D[ok], P[ok]
        cost = 0.05 * np.abs(G[:, -1] - _IDEAL[nv - 1])            # keep the top voice in its home
        for v in range(nv):
            p = self.vo[v]
            if p is None:
                cost += 0.12 * np.abs(G[:, v] - _IDEAL[v])
            else:                                                   # held 0, step 0.8, leaps dearer
                d = np.abs(G[:, v] - p)
                cost += np.where(d == 0, 0.0, np.where(d <= 2, 0.8, 2.0 + 1.1 * (d - 2)))
        if nv > 1:
            cost += np.where(D == 1, np.where(G[:, :-1] < 60, 5.0, 2.5), 0.0).sum(axis=1)   # semitone rubs
            lim = np.array([12] + [9] * (nv - 2))
            cost += 0.7 * np.maximum(D - lim, 0).sum(axis=1)                               # wide gaps
            cost += np.where((G[:, 0] < 52) & (D[:, 0] < 5), 1.5, 0.0)                     # mud down low
        cost += np.where(G[:, 0] - bass < 7, 2.0, 0.0)                                     # crowding the pedal
        S = np.column_stack([G, np.full(len(G), bass - 12), np.full(len(G), bass)])       # + the pedal's 16' and 8'
        a, b = np.triu_indices(S.shape[1], 1)
        d = np.abs(S[:, a] - S[:, b])
        cost += 3.5 * (((d == 1) | (d == 13)) & (np.minimum(S[:, a], S[:, b]) < 60)).sum(axis=1)  # low m2 / m9
        for pc in pcs:
            if pc not in (chord.root, chord.fifth):
                cost += np.maximum((P == pc).sum(axis=1) - 1, 0)                           # doubled 3rds etc.
        return [int(m) for m in G[int(np.argmin(cost))]] + [None] * (4 - nv)

    def _bass_midi(self, pc):
        """Pedal key for a pitch class: C2..D3, the octave nearest the last pedal note."""
        prev = self.ped if self.ped is not None else 41
        opts = [m for m in range(36, 51) if m % 12 == pc]
        return min(opts, key=lambda m: abs(m - prev))

    def _play_chord(self, t, chord, nv, swell):
        r = self.rng
        new = self._voicing(chord, nv)
        self._pedal(self._bass_midi(chord.bass), t - r.uniform(0.0, 0.25))
        for v in range(4):                 # held keys only follow registration changes
            self._key(v, new[v], t + r.uniform(0.0, 0.12))
        self.swell.set(swell)
        self._log(t, f"{self.mode} {chord.name()} {new}")

    def _resolve(self, t, chord):
        """Suspension resolves: the voice holding the 4th (or 2nd) steps to the third."""
        old = set(self.vo) - {None}
        tgt = set(chord.pcs)
        ped = self.ped if self.ped is not None else 0
        for v, m in enumerate(self.vo):
            if m is not None and m % 12 not in tgt:
                near = [m + d for d in (-2, -1, 1, 2) if (m + d) % 12 in tgt]
                step = min((s for s in near if s == ped or s - ped >= 3), key=lambda x: abs(x - m),
                           default=None)                        # never under the pedal's 8' or a step above it
                if step is not None and step not in old:
                    self._key(v, step, t)
        self._log(t, f"  resolve -> {chord.name()}")

    def _schedule_phrase(self, t, chords, then):
        r = self.rng
        nv = 4 if r.random() < 0.72 else 3
        n = len(chords)
        lo, hi = r.uniform(0.2, 0.28), r.uniform(0.4, 0.55)
        for i, ch in enumerate(chords):
            x = i / max(1, n - 1)
            sw = lo + (hi - lo) * math.sin(math.pi * x) ** 1.2 + r.uniform(-0.05, 0.05)
            if i == n - 1:
                sw = lo
            dur = r.uniform(16, 25) if i == n - 1 else r.uniform(12, 21)
            self._at(t, self._play_chord, ch, nv, sw)
            if ch.resolve is not None:
                self._at(t + dur * r.uniform(0.4, 0.62), self._resolve, ch.resolve)
            t += dur
        self._at(t, then)

    def _open(self, t):
        """Start: the first chord blooms from the pedal up."""
        ch = self._pick(self.mode, {0: 1.0}, _TONIC, "final")
        new = self._voicing(ch, 4)
        self._pedal(self._bass_midi(ch.bass), t)
        for v in range(4):
            self._key(v, new[v], t + 0.35 + 0.6 * v + self.rng.uniform(0.0, 0.2))
        self.swell.set(0.28)
        self._log(t, f"open {self.mode} {ch.name()} {new}")
        self._at(t + self.rng.uniform(14, 20), self._next_phrase)

    def _next_phrase(self, t):
        r = self.rng
        if t >= self.next_chant:
            key = self._pick_chant()
            self.chant = key
            target = CHANTS[key]["mode"]
            if target != self.mode:
                self.mode = target
                chords = self._phrase(target, 3)[1:] if r.random() < 0.5 else self._phrase(target, 4)[1:]
            else:
                chords = [self._pick(target, MODES[target][3], _COLOUR, "root", no_lt=True),
                          self._pick(target, {0: 1.0}, _TONIC, "final")]
            self._log(t, f"approach {target} for {key}")
            self._schedule_phrase(t, chords, self._thin)
            return
        if r.random() < 0.38:
            self.mode = _wchoice(r, {m: w for m, w in _CENTRES.items() if m != self.mode})
            chords = self._phrase(self.mode, int(r.integers(4, 7)))
        else:
            chords = self._phrase(self.mode, int(r.integers(4, 7)))[1:]
        if r.random() < 0.3:                          # the organist changes stops now and then
            self.reg = {GEDACKT, SALICIONAL, CELESTE} if PRINCIPAL in self.reg else \
                {GEDACKT, SALICIONAL, CELESTE, PRINCIPAL}
        self._schedule_phrase(t, chords, self._next_phrase)

    def _drone_notes(self):
        final = MODES[self.mode][0]
        lo = 48 + final                               # final in the octave above C3
        hi = None if self.mode == "phrygian" else lo + 7   # mode III's fifth B grates on its reciting tone C
        return lo, hi, self._bass_midi(final)

    def _thin(self, t):
        """The chords thin out to an open fifth on the final (in Phrygian the final alone); then the chant."""
        r = self.rng
        lo, hi, ped = self._drone_notes()
        reg = {GEDACKT, SALICIONAL, CELESTE}
        self._pedal(ped, t)
        self._key(3, None, t + 0.6)
        self._key(2, None, t + r.uniform(2.5, 3.5))
        self._key(0, lo, t + r.uniform(0.0, 0.3), reg)          # (the Principal is pushed in)
        self._key(1, hi, t + r.uniform(0.8, 1.4), reg)
        self.swell.set(0.16, 7.0)
        self._log(t, f"thin to {_NAMES[lo % 12]}{'' if hi is None else '-' + _NAMES[hi % 12]} drone")
        self._at(t + r.uniform(7.0, 9.0), self._sing)

    def _sing(self, t):
        r = self.rng
        key = self.chant or "salve"
        c = CHANTS[key]
        unit = r.uniform(0.72, 0.86)
        notes, end = perform(parse_gabc(c["gabc"]), unit, r, t, c["repeat"])
        for ton, toff, midi, art in notes:
            n = self._note(FLUTE, midi, ton, 1.0, 1.0 if art else 0.3)
            n.toff = toff
            if not art:
                n.ta *= 0.7
        self.played[key] = t
        self.state = "chant"
        self._log(t, f"chant {key} ({len(notes)} notes, {end - t:.0f} s, unit {unit:.2f} s)")
        self._at(end + r.uniform(2.5, 4.0), self._return)

    def _return(self, t):
        """After the chant the chords come back, voice by voice."""
        r = self.rng
        self.state = "improv"
        self.chant = None
        self.next_chant = t + r.uniform(240.0, 420.0)
        ch = self._pick(self.mode, {0: 1.0}, _TONIC, "final")
        new = self._voicing(ch, 4)
        for i, v in enumerate((2, 3, 0, 1)):
            self._key(v, new[v], t + i * r.uniform(1.6, 2.6))
        self.swell.set(0.3, 8.0)
        self._log(t, f"return {self.mode} {ch.name()} {new}")
        self._at(t + r.uniform(14, 20), self._next_phrase)

    def _pick_chant(self):
        r = self.rng
        theme = self.hint if self.t - self.hint_t < HINT_SECS else None
        pool = [k for k, c in CHANTS.items() if theme is None or c["theme"] == theme]
        pool.sort(key=lambda k: (self.played.get(k, -1e9), r.random()))
        return pool[1] if len(pool) > 2 and r.random() < 0.2 else pool[0]

    # -- synthesis ------------------------------------------------------------
    def _breath_shape(self, rank, midi):
        key = (rank, midi)
        s = self._bcache.get(key)
        if s is None:
            (bumps, broad, rms) = RANKS[rank].breath
            f0 = _hz(midi) * (0.5 if rank == PEDALR else 1.0)
            s = sum(peak(f0 * k, 0.12, w) for k, w in bumps) + broad * bandpass(1800, 7000, 1)
            s = normalize(s, rms)
            if len(self._bcache) > 600:
                self._bcache.clear()
            self._bcache[key] = s
        return s

    def _chiff(self, n, offset):
        """Speech transient of a pipe: a blip of one harmonic plus a puff of band noise."""
        r = self.rng
        k, ta, tt, na, nt = RANKS[n.rank].chiff
        L = 4096
        t = np.arange(L) / SR
        rise = 1.0 - np.exp(-t / 0.0025)
        g = np.zeros(L)
        if ta > 0 and k * n.f < 15000:
            g += ta * np.sin(TAU * k * n.f * t + r.uniform(0, TAU)) * np.exp(-t / tt)
        fc = max(140.0, 3.0 * n.f) if k < 0 else min(k * n.f, 9000.0)
        spec = (r.standard_normal(L // 2 + 1) + 1j * r.standard_normal(L // 2 + 1))
        f = np.fft.rfftfreq(L, 1.0 / SR)
        spec *= np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / fc) / 0.6) ** 2)
        nz = np.fft.irfft(spec, L)
        nz *= 1.0 / (np.abs(nz).max() + 1e-9)
        g += na * nz * np.exp(-t / nt)
        g *= rise * n.lvl * n.chiff * 0.7
        bus = self.bus_sw if RANKS[n.rank].div == SWELL else self.bus_dir
        bus.add(np.stack([g * n.gl, g * n.gr]), offset)

    def _swell_box(self, x, s0, s1):
        """Enclosed division through the swell shutters: STFT shelf low-pass + level."""
        s = 0.5 * (s0 + s1)
        fc = 450.0 * 2.0 ** (3.6 * s)
        rr = (F_SAFE / fc) ** 2
        H = 10.0 ** ((-7.0 + 7.0 * s) / 20.0) * np.sqrt((1.0 + rr / 25.0) / (1.0 + rr))
        frame = np.concatenate([self.sw_prev, x], axis=1) * _WIN
        self.sw_prev = x
        y = np.fft.irfft(np.fft.rfft(frame, axis=1) * H, n=NFFT, axis=1) * _WIN
        out = self.sw_tail + y[:, :BLOCK]
        self.sw_tail = y[:, BLOCK:]
        return out

    def render(self):
        t0 = self.t
        t1 = t0 + DT
        while self.q and self.q[0][0] < t1:
            ta, _, fn, args = heapq.heappop(self.q)
            fn(max(ta, t0), *args)
        if not self.q:                                 # never fall silent, whatever happened
            self._at(t1 + 1.0, self._next_phrase)
        live = []
        for n in self.notes:
            if n.toff < t0 and t0 - n.toff > 12.0 * n.tr:
                continue                               # finished (below -100 dB)
            if n.toff <= n.ton:
                continue                               # released before it ever spoke
            live.append(n)
        self.notes = live
        act = [n for n in live if n.ton < t1]
        sw = np.zeros((2, BLOCK))
        dr = np.zeros((2, BLOCK))
        solo = np.zeros((2, BLOCK))
        mag_sw = np.zeros(NFFT // 2 + 1)
        mag_dr = np.zeros(NFFT // 2 + 1)
        if act:
            f = np.array([n.f for n in act])
            ph = np.array([n.ph for n in act])
            base = np.array([n.rank * (TN + 1) for n in act])
            inc = f / SR                                # phase in cycles per sample
            P = ph[:, None] + inc[:, None] * _IDX[None, :]
            for n, p in zip(act, (ph + inc * BLOCK) % 1.0):
                n.ph = p
            P -= np.floor(P)                            # -> table position, linear interpolation
            P *= TN
            i = P.astype(np.int32)
            fr = P - i
            i += base[:, None].astype(np.int32)
            a = _TABLES[i]
            sig = a + (_TABLES[i + 1] - a) * fr
            g = np.array([n.lvl for n in act])
            tt = t0 + _TT
            dyn = [k for k, n in enumerate(act) if n.ton + n.ta > t0 or n.toff < t1]
            if dyn:
                ton = np.array([act[k].ton for k in dyn])[:, None]
                tat = np.array([act[k].ta for k in dyn])[:, None]
                toff = np.array([act[k].toff for k in dyn])[:, None]
                trl = np.array([act[k].tr for k in dyn])[:, None]
                x = np.clip((tt[None, :] - ton) / tat, 0.0, 1.0)
                env = x * x * (3.0 - 2.0 * x)
                rel = tt[None, :] - toff
                env *= np.exp(-np.maximum(rel, 0.0) / trl)
                sig[dyn] *= env
            gl = np.array([n.gl for n in act]) * g
            gr = np.array([n.gr for n in act]) * g
            divs = np.array([RANKS[n.rank].div for n in act])
            for d, dst in ((SWELL, sw), (PEDAL, dr), (SOLO, solo)):
                m = divs == d
                if m.any():
                    dst[0] += gl[m] @ sig[m]
                    dst[1] += gr[m] @ sig[m]
            # wind noise of the pipes, following each pipe's envelope at the end of the block
            te = t1
            for n in act:
                xa = min(max((te - n.ton) / n.ta, 0.0), 1.0)
                e = n.lvl * xa * xa * (3 - 2 * xa) * (math.exp(-(te - n.toff) / n.tr) if te > n.toff else 1.0)
                if e > 1e-4:
                    s = self._breath_shape(n.rank, n.midi)
                    if RANKS[n.rank].div == SWELL:
                        mag_sw += e * s
                    else:
                        mag_dr += e * s
                if n.chiff and n.ton >= t0:
                    self._chiff(n, int((n.ton - t0) * SR))
                    n.chiff = 0.0
        sw += self.br_sw.render(mag_sw) + self.bus_sw.pull()
        s0 = self.swell.v
        self.swell.ramp()
        sw = self._swell_box(sw, s0, self.swell.v)
        out = sw + dr + solo + self.br_dir.render(mag_dr) + self.bus_dir.pull()
        out += self.tail.process(out[0] + out[1]) * self.TAIL
        self.t = t1
        out *= self.GAIN
        # leveller: like an organist easing the swell when the chords pile up (1.5 s detector, ~1 s gain)
        self.lev_ms += (float(np.mean(out * out)) - self.lev_ms) * (1.0 - math.exp(-DT / 1.5))
        over = 10.0 * math.log10(max(self.lev_ms, 1e-12) / self.LEVEL ** 2)
        g = 10.0 ** (-max(over, 0.0) * (2.0 / 3.0) / 20.0)
        g = self.lev_g + (g - self.lev_g) * (1.0 - math.exp(-DT / 1.0))
        out *= self.lev_g + (g - self.lev_g) * (_TT / DT)
        self.lev_g = g
        a = np.abs(out)
        over = a > self.KNEE
        if over.any():                     # soft knee: only samples beyond ~3.4x the RMS are touched
            k, c = self.KNEE, self.CEIL
            out[over] = np.sign(out[over]) * (k + (c - k) * np.tanh((a[over] - k) / (c - k)))
        return out


# =============================================================================
# THE CATHEDRAL AT NIGHT (scene ambience "cathedral")
# =============================================================================
# rosary sounds: resonant modes (Hz, decay s, amplitude), a soft body thump, chance of a
# second click as the beads settle, peak level, attack time
_BEADS = {
    "small": dict(modes=((2350, .0060, 1.0), (3900, .0042, .65), (5650, .0028, .40)), body=(430, .005, .22),
                  double=.35, gain=.26, attack=.00025, length=.07),
    "large": dict(modes=((1800, .0082, 1.0), (3050, .0058, .60), (4500, .0036, .32)), body=(330, .007, .35),
                  double=.30, gain=.32, attack=.0003, length=.08),
    "cross": dict(modes=((3150, .048, 1.0), (5240, .034, .70), (7630, .022, .45), (9870, .015, .30)),
                  body=(520, .004, .15), double=.12, gain=.16, attack=.0004, length=.30),
    "medal": dict(modes=((1680, .11, 1.0), (2790, .085, .62), (4350, .05, .38), (6150, .03, .22)),
                  body=(0, 0, 0), double=0.0, gain=.12, attack=.0015, length=.42),
}


class CathedralAir:
    """The hush of a vast stone church at night, with rosary beads from the scene.

    Same interface as sources.Source (sources.py registers a Source subclass of it
    as the "cathedral" ambience; this module cannot import sources.py)."""

    key = "cathedral"
    label = "CATHEDRAL"
    presets = None
    wet = 0.3
    gain = 1.0
    VERB = 2.4            # reverberant / direct energy of a bead click

    def __init__(self, rng):
        self.rng = r = np.random.default_rng(int(rng.integers(1 << 62)))
        self.preset = None
        self.t = 0.0
        self.air = SpectralNoise(r)
        self.air_mag = normalize(lowpass(240, 2) * highpass(35, 2) * tilt(-0.8), 0.033)   # less inaudible rumble
        self.hush = SpectralNoise(r)
        self.hush_mag = normalize(bandpass(200, 1400, 2) * tilt(-0.6), 0.013)
        self.breathe = RandomWalks(2, r, rate=0.025, smooth=9.0)
        self.wind = SpectralNoise(r)
        self.gust = RandomWalks(3, r, rate=0.05, smooth=4.5)
        self.wind_base = lowpass(420, 2) * highpass(40, 2) * tilt(-0.5)
        self.flame = SpectralNoise(r, channels=1)
        self.flame_mag = normalize(bandpass(90, 900, 1) * tilt(-0.4), 0.015)
        self.flick_f = r.uniform(6.0, 14.0, 3)
        self.flick_ph = r.uniform(0, TAU, 3)
        self.flick_w = RandomWalks(3, r, rate=0.6, smooth=0.4)
        self.draft = Param(0.04, 1.2)
        self.draft_end = 0.0
        self.draft_next = r.uniform(12.0, 35.0)
        self.sputter_next = r.uniform(60.0, 150.0)
        self.bus = GrainBus(1.5)
        self.verb = SpectralNoise(r)
        t60 = 7.5 / (1.0 + (F_SAFE / 2000.0) ** 1.1)          # 7 s low, 5.3 s at 1 kHz, 1.3 s at 8 kHz
        self.dE = np.exp(-13.8155 * DT / t60)                  # energy decay per block
        self.E = np.zeros(NFFT // 2 + 1)        # reverberant energy spectrum
        self.E_pend = np.zeros(NFFT // 2 + 1)   # injected last block (one block of pre-delay)
        self.E_next = np.zeros(NFFT // 2 + 1)   # injected by events since the last block
        self.n_ev = 0

    def set_preset(self, name, instant=False):
        pass

    # -- rosary ---------------------------------------------------------------
    def event(self, name, kind="small", pan=0.0, vel=0.7, **kw):
        """("bead", kind="small"|"large"|"cross"|"medal", pan=-1..1, vel=0..1); anything else is ignored."""
        if not isinstance(name, str) or name != "bead" or self.n_ev >= 3:     # a burst: keep it bounded
            return
        kind = kind if isinstance(kind, str) and kind in _BEADS else "small"
        pan = _num(pan, -1.0, 1.0, 0.0)
        vel = _num(vel, 0.0, 1.0, 0.7)
        if vel <= 0.0:
            return
        self.n_ev += 1
        r = self.rng
        g = self._click(kind, vel)
        spec = _BEADS[kind]
        if r.random() < spec["double"]:                      # the beads settle: a softer second touch
            off = int(r.uniform(0.018, 0.045) * SR)
            g2 = self._click(kind, vel * r.uniform(0.3, 0.6))
            g = np.concatenate([g, np.zeros(max(0, off + len(g2) - len(g)))])
            g[off:off + len(g2)] += g2
        o = int(r.uniform(0, 96))
        self.bus.add(g, o, 1.0, pan * 0.8, itd=0.0005 * abs(pan))
        for d, a in ((r.uniform(0.011, 0.02), 0.2), (r.uniform(0.024, 0.04), 0.14),
                     (r.uniform(0.045, 0.065), 0.1), (r.uniform(0.07, 0.095), 0.07)):
            self.bus.add(g, o + int(d * SR), a * r.uniform(0.7, 1.2), r.uniform(-0.8, 0.8))
        X = np.fft.rfft(g[:NFFT], n=NFFT)
        # statistical tail: inject the click's energy spectrum, scaled so the whole tail carries VERB x its energy
        # (each channel resynthesises E, so each gets half)
        self.E_next += (X.real ** 2 + X.imag ** 2) * (self.VERB * 0.5 * (1.0 - self.dE))

    def _click(self, kind, vel):
        r = self.rng
        spec = _BEADS[kind]
        n = int(spec["length"] * SR)
        t = np.arange(n) / SR
        g = np.zeros(n)
        bright = 0.5 + 0.5 * vel                             # a gentle touch is darker
        for j, (f, tau, a) in enumerate(spec["modes"]):
            g += (a * bright ** j * np.sin(TAU * f * r.uniform(0.92, 1.08) * t + r.uniform(0, TAU))
                  * np.exp(-t / (tau * r.uniform(0.85, 1.15))))
        bf, bt, ba = spec["body"]
        if ba:
            g += ba * np.sin(TAU * bf * r.uniform(0.9, 1.1) * t) * np.exp(-t / bt)
        k = min(n, 96)
        g[:k] += 0.2 * np.diff(r.standard_normal(k + 1)) * np.exp(-np.arange(k) / SR / 0.0004)
        a = max(2, int(spec["attack"] * SR))
        g[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)     # soft onset: no digital click
        g[-64:] *= np.linspace(1.0, 0.0, 64)
        return g * (spec["gain"] * vel ** 0.9 / (np.abs(g).max() + 1e-9))

    # -- render ---------------------------------------------------------------
    def render(self):
        r = self.rng
        b = self.breathe.step()
        out = self.air.render(self.air_mag * (0.72 + 0.45 * b[0]))
        out += self.hush.render(self.hush_mag * (0.55 + 0.8 * b[1]))
        # wind outside, heard through stone and glass: a muffled, slow sigh
        g = self.gust.step()
        mags = []
        for c in range(2):
            gc = 0.7 * g[0] + 0.3 * g[1 + c]
            m = self.wind_base * (0.3 + gc) + peak(210 + 280 * gc, 0.5, 1.2 * gc ** 2)
            mags.append(normalize(m, 0.002 + 0.021 * gc ** 2.5))
        out += self.wind.render(np.stack(mags))
        # candle flames flutter when a draft passes
        if self.t >= self.draft_next:
            self.draft_end = self.t + r.uniform(2.5, 7.0)
            self.draft_next = self.draft_end + r.uniform(20.0, 70.0)
            self.draft.set(r.uniform(0.5, 1.0))
        elif self.t >= self.draft_end:
            self.draft.set(0.04)
        w = self.flick_w.step()
        self.flick_f = np.clip(self.flick_f + (w - 0.5) * 0.8, 5.0, 16.0)
        ph = self.flick_ph[:, None] + TAU * self.flick_f[:, None] * _TT[None, :]
        self.flick_ph = (self.flick_ph + TAU * self.flick_f * DT) % TAU
        flick = (0.5 + 0.5 * np.tanh(1.5 * np.sin(ph).sum(axis=0))) ** 2
        fl = self.flame.render(self.flame_mag)[0] * flick * self.draft.ramp()
        out += np.stack([fl * 0.8, fl * 0.6])
        # once in a while a wick sputters, very softly
        if self.t >= self.sputter_next:
            self.sputter_next = self.t + r.uniform(90.0, 240.0)
            pan = r.uniform(-0.5, 0.5)
            o = 0
            for _ in range(int(r.integers(2, 6))):
                o += int(r.exponential(0.05) * SR)
                k = int(r.uniform(0.002, 0.005) * SR)
                c = np.convolve(r.standard_normal(k), np.ones(6) / 6, "same") * np.hanning(k)
                self.bus.add(c / (np.abs(c).max() + 1e-9), o, r.uniform(0.006, 0.014), pan)
        # rosary beads: the dry clicks and their early reflections, then the long tail
        out += self.bus.pull()
        self.E *= self.dE
        self.E += self.E_pend
        self.E_pend, self.E_next = self.E_next, self.E_pend
        self.E_next[:] = 0.0
        if self.E.max() > 1e-13:
            out += self.verb.render(np.sqrt(self.E))
        elif self.E.any():
            self.E[:] = 0.0                    # died away: stop before it decays into denormals
        self.n_ev = 0
        self.t += DT
        return out
