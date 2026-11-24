"""Audio engine: a mixer running in its own process, streaming raw PCM to a
system player (paplay / pw-cat / pacat / aplay / sox / ffplay).

The UI process talks to it through a multiprocessing pipe with small tuples,
so heavy numpy synthesis never stalls the animation (no shared GIL).
"""

import math
import multiprocessing as mp
import os
import select
import shutil
import signal
import subprocess
import sys
import time
import wave

import numpy as np

from .dsp import BLOCK, SR, Param, Reverb
from .sources import LAYER_KEYS, LAYER_LABELS, LAYER_PRESETS, LAYERS

DEFAULT_LEVELS = {
    "drone": 0.6, "noise": 0.3, "ambience": 0.5, "rain": 0.0, "ocean": 0.0, "wind": 0.0,
    "stream": 0.0, "crickets": 0.0, "fire": 0.0, "deepsea": 0.0, "shiphum": 0.0, "taps": 0.8,
}
DEFAULT_PRESETS = {k: (p[0] if p else None) for k, p in LAYER_PRESETS.items()}

BACKENDS = [
    ("paplay", ["paplay", "--raw", "--format=s16le", f"--rate={SR}", "--channels=2",
                "--latency-msec=120", "--client-name=StasisKit", "--stream-name=StasisKit"]),
    ("pw-cat", ["pw-cat", "--playback", "--raw", "--format", "s16", "--rate", str(SR), "--channels", "2",
                "--latency", "120ms", "-"]),
    ("pacat", ["pacat", "--playback", "--raw", "--format=s16le", f"--rate={SR}", "--channels=2",
               "--latency-msec=120", "--client-name=StasisKit"]),
    ("aplay", ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "2", "--buffer-time=200000"]),
    ("sox", ["play", "-q", "-t", "raw", "-r", str(SR), "-e", "signed", "-b", "16", "-c", "2", "-"]),
    ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-f", "s16le", "-ar", str(SR),
                "-ch_layout", "stereo", "-i", "-"]),
]

STALL_SECS = 5.0       # a player that accepts nothing for this long is restarted
HEALTHY_SECS = 60.0    # this much good playback forgives earlier player crashes
# RIFF sizes are 32-bit: whole 16-bit stereo blocks that fit (~372 min)
WAV_MAX_BLOCKS = (0xFFFFFFFF - 36) // (BLOCK * 4)
WAV_MAX_SECONDS = WAV_MAX_BLOCKS * BLOCK / SR


def vol_curve(v):
    """Slider position 0..1 -> linear gain (roughly perceptual)."""
    v = max(0.0, min(1.0, v))
    return v * v


class Mixer:
    def __init__(self, seed=None, levels=None, presets=None, master=0.8):
        self.rng = np.random.default_rng(seed)
        self.sources = {cls.key: cls(self.rng) for cls in LAYERS}
        self.vol = {k: Param(0.0, 0.35) for k in LAYER_KEYS}
        self.master = Param(vol_curve(master), 0.3)
        self.fade = Param(1.0, 2.0)
        self.muted = False
        self.reverb = Reverb(self.rng, 4.8)
        levels = dict(DEFAULT_LEVELS, **(levels or {}))
        for k, v in levels.items():
            if k in self.vol:
                self.vol[k].jump(vol_curve(v))
        for k, p in (presets or {}).items():
            if k in self.sources and p:
                self.sources[k].set_preset(p, instant=True)   # saved state: no crossfade from defaults

    def apply(self, msg):
        op = msg[0]
        if op == "vol" and msg[1] in self.vol:
            self.vol[msg[1]].set(vol_curve(msg[2]))
        elif op == "preset" and msg[1] in self.sources:
            self.sources[msg[1]].set_preset(msg[2])
        elif op == "master":
            self.master.set(vol_curve(msg[1]))
        elif op == "mute":
            self.muted = bool(msg[1])
        elif op == "fade":
            self.fade.set(msg[1], max(0.05, msg[2] / 4.0))
        elif op == "ambience":
            self.sources["ambience"].set_scene(msg[1])
        elif op == "event":
            layer = msg[2].get("layer", "taps")
            src = self.sources.get(layer)
            if src is not None and layer in self.vol and self.vol[layer].v > 1e-4:
                try:
                    src.event(msg[1], **{k: v for k, v in msg[2].items() if k != "layer"})
                except (TypeError, ValueError):
                    pass    # a malformed event must never take the audio process down

    def render(self):
        dry = np.zeros((2, BLOCK))
        send = np.zeros(BLOCK)
        for key, src in self.sources.items():
            vp = self.vol[key]
            if vp.v < 1e-5 and vp.target < 1e-5:
                if key == "taps":
                    src.render()   # drain the grain bus
                continue
            g = vp.ramp()
            sig = src.render() * g
            dry += sig * (1.0 - 0.4 * src.wet)
            if src.wet > 0:
                send += (sig[0] + sig[1]) * (0.5 * src.wet)
        out = dry + self.reverb.process(send) * 0.6
        m = self.master.ramp() * self.fade.ramp()
        if self.muted:
            m = 0.0
        out = np.tanh(out * m * 2.6)
        return out


def _close_player(proc, grace=0.3):
    """Stop and reap a player. It is signalled before stdin is closed, so the
    close can never block on a full pipe (and never leaves an orphan behind)."""
    if proc is None:
        return
    for stop in (proc.terminate, proc.kill):
        try:
            stop()
            proc.wait(timeout=grace)
            break
        except Exception:
            pass
    try:
        proc.stdin.close()
    except Exception:
        pass


def _open_backend(preferred=None):
    order = BACKENDS
    if preferred and preferred != "auto":
        order = [b for b in BACKENDS if b[0] == preferred] + [b for b in BACKENDS if b[0] != preferred]
    for name, cmd in order:
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            continue
        fd = proc.stdin.fileno()
        try:
            import fcntl
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, BLOCK * 4 * 2)
        except (OSError, AttributeError, ImportError):
            pass
        # all writes go straight to the fd, non-blocking, so a stuck player can never hang us
        os.set_blocking(fd, False)
        try:
            os.write(fd, bytes(BLOCK * 4 * 2))
        except BlockingIOError:
            pass
        except OSError:
            _close_player(proc)
            continue
        time.sleep(0.25)
        if proc.poll() is None:
            return proc, name
        _close_player(proc)
    return None, None


def _log_path():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "stasiskit")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "audio.log")


def _tell(conn, msg):
    """Status message to the UI; dropped rather than blocking if the UI is not reading."""
    try:
        if select.select([], [conn], [], 0)[1]:
            conn.send(msg)
    except (OSError, ValueError):
        pass


def audio_main(conn, init, backend="auto", ui_conn=None):
    """Entry point of the audio process."""
    if ui_conn is not None:
        ui_conn.close()   # drop the inherited UI end, so recv() sees EOF if the UI dies
    ppid0 = init.get("ppid") or os.getppid()
    try:
        fd = os.open(_log_path(), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    try:
        mixer = Mixer(init.get("seed"), init.get("levels"), init.get("presets"), init.get("master", 0.8))
        mixer.muted = bool(init.get("muted"))
        if init.get("ambience"):
            mixer.sources["ambience"].set_scene(init["ambience"])
    except Exception as e:  # pragma: no cover - logged for the user
        _tell(conn, ("error", f"synth init failed: {e}"))
        raise
    proc, name = _open_backend(backend)
    if proc is None:
        _tell(conn, ("error", "no audio player found (install pulseaudio-utils/pipewire, alsa-utils or sox)"))
        return
    _tell(conn, ("ready", name))
    restarts = 0
    started = last_io = time.monotonic()
    pending = b""          # rendered PCM the player has not taken yet
    announce = False       # tell the UI once a restarted player takes data
    try:
        while True:
            stuck = pending and time.monotonic() - last_io > 0.5
            while conn.poll():
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    return
                if msg[0] == "quit":
                    return
                if not (stuck and msg[0] == "event"):   # don't pile up taps while the player is stuck
                    mixer.apply(msg)
            if os.getppid() != ppid0:   # orphaned (reparented to init or a subreaper)
                return
            if not pending:
                block = mixer.render()
                pending = memoryview((np.clip(block.T, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            try:
                n = os.write(proc.stdin.fileno(), pending)
            except BlockingIOError:
                n = 0
            except OSError:          # player exited
                n = -1
            now = time.monotonic()
            if n > 0:
                pending = pending[n:]
                last_io = now
                if announce:
                    announce = False
                    _tell(conn, ("ready", name))
                continue
            if n == 0 and now - last_io < STALL_SECS:
                select.select([conn], [proc.stdin], [], 0.1)   # room in the pipe, or a message
                continue
            # the player died or stopped taking data: replace it
            _close_player(proc)
            if n == 0:
                _tell(conn, ("error", f"{name} stopped taking audio - restarting it"))
            else:
                if last_io - started >= HEALTHY_SECS:
                    restarts = 0
                restarts += 1
                if restarts > 5:
                    _tell(conn, ("error", f"{name} keeps exiting"))
                    return
            proc, name = _open_backend(backend)
            if proc is None:
                _tell(conn, ("error", "audio player died"))
                return
            started = last_io = time.monotonic()
            announce = True
    finally:
        _close_player(proc)


class AudioClient:
    """UI-side handle. Keeps a mirror of mixer state for drawing the mixer."""

    def __init__(self, enabled=True, levels=None, presets=None, master=0.8, muted=False, backend="auto"):
        self.enabled = enabled
        self.levels = dict(DEFAULT_LEVELS, **(levels or {}))
        self.presets = dict(DEFAULT_PRESETS, **{k: v for k, v in (presets or {}).items() if k in DEFAULT_PRESETS})
        for k, v in list(self.presets.items()):
            ps = LAYER_PRESETS.get(k)
            if ps and v not in ps:
                self.presets[k] = ps[0]
        self.master = master
        self.muted = muted
        self.backend = backend
        self.status = "off" if not enabled else "starting"
        self.backend_name = None
        self.conn = None
        self.proc = None
        self.unsent = {}          # settings waiting for room in the pipe, latest per control
        self.fade_level = 1.0
        self.ambience_key = None

    # -- process control --------------------------------------------------
    def start(self, ambience=None):
        if not self.enabled:
            return
        self.ambience_key = ambience
        ctx = mp.get_context("fork")
        parent, child = ctx.Pipe()
        init = dict(levels=self.levels, presets=self.presets, master=self.master, muted=self.muted,
                    ambience=ambience, seed=int(time.time()) & 0xFFFFFF, ppid=os.getpid())
        # the child also gets our end, only to close its inherited copy of it
        self.proc = ctx.Process(target=audio_main, args=(child, init, self.backend, parent), daemon=True,
                                name="stasiskit-audio")
        self.proc.start()
        child.close()
        self.conn = parent

    def poll(self):
        if not self.conn:
            return
        try:
            while self.conn.poll():
                msg = self.conn.recv()
                if msg[0] == "ready":
                    self.status = "on"
                    self.backend_name = msg[1]
                elif msg[0] == "error":
                    self.status = "error: " + msg[1]
        except (EOFError, OSError):
            self._lost()
        if self.unsent:
            self._flush()

    def _lost(self):
        if not self.status.startswith("error"):   # keep the reason the audio process gave
            self.status = "stopped"
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = None
        self.unsent.clear()

    def stop(self):
        self._send("quit")    # best effort, never blocks
        if self.proc is not None:
            self.proc.join(timeout=1.5)
            for kill in (self.proc.terminate, self.proc.kill):
                if not self.proc.is_alive():
                    break
                kill()
                self.proc.join(timeout=1.0)
        if self.conn:
            self.conn.close()
        self.conn = None

    def _send(self, *msg):
        """Never blocks the UI. If the audio process is not draining its pipe (e.g. stuck
        on a player), taps are dropped and settings wait in `unsent` until there is room."""
        if not self.conn:
            return
        if msg[0] == "event":
            self._flush(msg)
        else:
            self.unsent[msg[:2] if msg[0] in ("vol", "preset") else msg[:1]] = msg
            self._flush()

    def _flush(self, event=None):
        if not self.conn:
            return

        def room():
            return select.select([], [self.conn], [], 0)[1]
        try:
            while self.unsent and room():
                self.conn.send(self.unsent.pop(next(iter(self.unsent))))
            if event and not self.unsent and room():
                self.conn.send(event)
        except (OSError, ValueError):
            self._lost()

    # -- controls -----------------------------------------------------------
    @property
    def active(self):
        return self.status == "on"

    def set_level(self, layer, v):
        v = round(max(0.0, min(1.0, v)), 3)
        self.levels[layer] = v
        self._send("vol", layer, v)

    def set_preset(self, layer, name):
        self.presets[layer] = name
        self._send("preset", layer, name)

    def cycle_preset(self, layer, step=1):
        ps = LAYER_PRESETS.get(layer)
        if not ps:
            return None
        cur = self.presets.get(layer) or ps[0]
        i = (ps.index(cur) + step) % len(ps) if cur in ps else 0
        self.set_preset(layer, ps[i])
        return ps[i]

    def set_master(self, v):
        self.master = round(max(0.0, min(1.0, v)), 3)
        self._send("master", self.master)

    def toggle_mute(self):
        self.muted = not self.muted
        self._send("mute", self.muted)
        return self.muted

    def fade_to(self, level, seconds):
        self.fade_level = level
        self._send("fade", float(level), float(seconds))

    def ambience(self, key):
        if key != self.ambience_key:
            self.ambience_key = key
            self._send("ambience", key)

    def event(self, name, **kw):
        """Scene-triggered sound (nail taps, breaking waves...).  `layer` picks the
        mixer layer that plays it (default "taps"); muted layers drop it here."""
        if self.levels.get(kw.get("layer", "taps"), 0) > 0:
            self._send("event", name, kw)

    def summary(self):
        parts = []
        for k in LAYER_KEYS:
            v = self.levels.get(k, 0)
            if v > 0.01 and k != "taps":
                p = self.presets.get(k)
                lbl = LAYER_LABELS[k].lower()
                parts.append(f"{lbl}{':' + p if p and k in ('drone', 'noise') else ''}")
        return " · ".join(parts) if parts else "silence"


def render_wav(path, seconds, levels=None, presets=None, master=0.8, ambience=None, seed=None,
               progress=None):
    """Offline render of the current mix to a 16-bit stereo WAV file."""
    nblocks = seconds * SR / BLOCK
    if not nblocks <= WAV_MAX_BLOCKS + 1e-6:   # checked up front, not after hours of rendering (also NaN/inf)
        m = int(WAV_MAX_SECONDS // 60)
        raise ValueError(f"WAV files are limited to {m // 60} h {m % 60} min of 48 kHz stereo audio "
                         f"({seconds / 60:g} min requested); export {m} min or less")
    nblocks = min(int(math.ceil(nblocks)), WAV_MAX_BLOCKS)
    mixer = Mixer(seed, levels, presets, master)
    if ambience:
        mixer.sources["ambience"].set_scene(ambience)
    fade_blocks = int(min(10.0, seconds * 0.1) * SR / BLOCK)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        for i in range(nblocks):
            blk = mixer.render()
            if i < fade_blocks:
                blk *= np.linspace(i, i + 1, BLOCK) / fade_blocks
            elif i >= nblocks - fade_blocks:
                j = nblocks - i
                blk *= np.linspace(j, j - 1, BLOCK) / fade_blocks
            wf.writeframes((np.clip(blk.T, -1, 1) * 32767).astype("<i2").tobytes())
            if progress and i % 50 == 0:
                progress(i / nblocks)
    return path
