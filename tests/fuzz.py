"""StasisKit fuzz/soak harness (fake clock, no audio, no TTY).

  python3 -I tests/fuzz.py soak     # every scene x mode x size, long simulated runs, render after every update
  python3 -I tests/fuzz.py app      # whole App loop: random keys + resizes + timers, hours of simulated time
"""
import os
import random
import sys
import time as _real_time
import traceback
import types
from multiprocessing import get_context

import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TMP = tempfile.mkdtemp(prefix="stasiskit-fuzz-")
os.environ["XDG_CONFIG_HOME"] = _TMP        # never touch the real config/state
os.environ["XDG_CACHE_HOME"] = _TMP
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from stasiskit.canvas import AnsiEncoder, Canvas  # noqa: E402
from stasiskit.scenes import SCENES  # noqa: E402

SIZES = [(40, 12), (41, 13), (63, 17), (120, 36), (300, 90), (400, 12), (40, 100)]


def _fail(tag, exc):
    return f"FAIL {tag}\n" + "".join(traceback.format_exception(exc))


# ----------------------------------------------------------------------------- scene soak
def soak_one(job):
    cls_i, mode, (w, h), sim_minutes, dt, churn = job
    cls = SCENES[cls_i]
    rng = random.Random(hash((cls_i, mode, w, h)) & 0xFFFF)
    tag = f"{cls.key} mode={mode} size={w}x{h}"
    try:
        sc = cls(None, seed=rng.randrange(1 << 30))
        sc.ensure_size(w, h)
        cv = Canvas(w, h)
        enc = AnsiEncoder()
        steps = int(sim_minutes * 60 / dt)
        for i in range(steps):
            sc.update(dt)
            if churn and rng.random() < churn:
                r = rng.random()
                if r < 0.4:
                    sc.next_variant(rng.choice((1, -1)))
                elif r < 0.6:
                    sc.set_variant(rng.randrange(len(sc.variants)))
                elif r < 0.8:
                    nw, nh = rng.choice(SIZES)
                    sc.ensure_size(nw, nh)
                    cv.resize(nw, nh)
                    w, h = nw, nh
                else:
                    sc.on_enter()
            sc.ensure_size(w, h)
            sc.render(cv, mode)
            if i % 7 == 0:
                enc.encode(cv)
            if not np.isfinite(cv.fg).all() or not np.isfinite(cv.bg).all():
                return f"FAIL {tag}: non-finite colours at sim t={sc.t:.1f}s"
        return f"ok   {tag} ({steps} steps, sim {sim_minutes} min)"
    except Exception as e:
        return _fail(f"{tag} at sim t={getattr(sc, 't', -1):.1f}s", e)


def run_soak():
    jobs = []
    for ci, cls in enumerate(SCENES):
        long_min = 40 if cls.key in ("garden", "cards") else 15
        for mode in ("ascii", "blocks"):
            jobs.append((ci, mode, (60, 20), long_min, 0.1, 0.0))            # long lifecycle soak
            jobs.append((ci, mode, (60, 20), 10, 0.1, 0.02))                 # with variant/size churn
            for size in SIZES:
                jobs.append((ci, mode, size, 1.5, 0.1, 0.01))                # size sweep
    print(f"{len(jobs)} soak jobs")
    with get_context("fork").Pool(20) as pool:
        out = list(pool.imap_unordered(soak_one, jobs))
    fails = [o for o in out if o.startswith("FAIL")]
    for o in fails:
        print(o)
    print(f"\nsoak: {len(out) - len(fails)} ok, {len(fails)} failed")
    return len(fails)


# ----------------------------------------------------------------------------- app fuzz
class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.base = 1_800_000_000.0

    def time(self):
        return self.base + self.t

    def perf_counter(self):
        return self.t

    def strftime(self, fmt, *a):
        return _real_time.strftime(fmt, _real_time.localtime(self.base + self.t))

    def localtime(self, *a):
        return _real_time.localtime(self.base + self.t)

    def sleep(self, s):
        self.t += s


NORMAL_KEYS = (["right", "left", "l", "h", "n", "p", "pgup", "pgdn", "v", "V", "m", "t", "?", "f1", "b", "c", "C",
                "s", "r", "[", "]", "-", "+", "=", "_", "M", "a", "i", "tab", "ctrl-l", "space", "enter", "x", "z",
                "alt-x", "unknown", "home", "end", "insert", "delete", "backspace", "shift-tab", "0", "€"]
               + [str(d) for d in range(1, 10)])
PANEL_KEYS = ["up", "down", "left", "right", "k", "j", "h", "l", "enter", "space", "backspace", "0", "M", "e",
              "pgup", "pgdn", "+", "-", "=", "tab", "esc", "m", "t", "x"]


class FuzzTerm:
    def __init__(self, clock, rng, app_ref, sim_seconds, key_rate):
        self.clock, self.rng, self.app_ref = clock, rng, app_ref
        self.w, self.h = 120, 36
        self.end = sim_seconds
        self.key_rate = key_rate
        self.next_resize = rng.uniform(5, 60)
        self.frames = 0
        self.bytes = 0
        self.keylog = []

    def size(self):
        return self.w, self.h

    def read_keys(self, timeout):
        self.clock.t += max(timeout, 0.0)
        rng = self.rng
        t = self.clock.t
        if t > self.end:
            return ["q"] if self.app_ref[0].panel is None else ["esc", "esc", "q"]
        if t > self.next_resize:
            self.next_resize = t + rng.uniform(5, 90)
            self.w, self.h = rng.choice(SIZES + [(39, 11), (30, 8), (200, 60), (80, 24)])
        keys = []
        if rng.random() < self.key_rate:
            app = self.app_ref[0]
            if app.panel in ("mixer", "timer"):
                keys = [rng.choice(PANEL_KEYS)]
            else:
                keys = [rng.choice(NORMAL_KEYS)]
        self.keylog.extend((round(t, 2), k) for k in keys)
        del self.keylog[:-40]
        return keys

    def write(self, s):
        self.frames += 1
        self.bytes += len(s)


def app_one(job):
    seed, sim_minutes, key_rate, end_action, companion, mode = job
    import stasiskit.app as appmod
    import stasiskit.ui as uimod
    clock = FakeClock()
    appmod.time = clock
    uimod.time = clock
    from stasiskit.app import App
    from stasiskit.config import load_config
    rng = random.Random(seed)
    random.seed(seed)
    cfg = load_config()
    cfg["audio"] = False
    cfg["on_timer_end"] = end_action
    cfg["scene_cycle_minutes"] = rng.choice([0.5, 2.0, 7.0])
    cfg["chattiness"] = rng.choice([0, 1, 2, 3])
    cfg["boot_sequence"] = rng.random() < 0.5
    opts = types.SimpleNamespace(no_boot=False, screensaver=False, scene=rng.choice([None, "garden", "cards", "3", "abyss"]),
                                 companion=companion, mode=mode, fps=8, no_audio=True, mute=True,
                                 timer=rng.choice([None, 0.2, 1.0, 6.0]))
    ref = [None]
    term = None
    try:
        app = App(cfg, {}, opts)
        ref[0] = app
        term = FuzzTerm(clock, rng, ref, sim_minutes * 60, key_rate)
        app.run(term=term)
        return (f"ok   seed={seed} end={end_action} {companion} sim={clock.t / 60:.1f}min frames={term.frames} "
                f"avgKB={term.bytes / max(1, term.frames) / 1024:.1f} final scene={app.scene.key} phase={app.timer_phase}")
    except Exception as e:
        ctx = f"seed={seed} end={end_action} sim t={clock.t:.1f}s size={term.w}x{term.h}" if term else f"seed={seed}"
        if ref[0] is not None:
            a = ref[0]
            ctx += f" scene={a.scene_classes[a.scene_idx].key} panel={a.panel} phase={a.timer_phase} mode={a.mode}"
        if term:
            ctx += f"\n  last keys: {term.keylog[-15:]}"
        return _fail(ctx, e)


def run_app():
    jobs = []
    for seed in range(16):
        jobs.append((seed, 90, 0.08, ["dark", "dim", "exit"][seed % 3], ["gemma", "neon", "claude"][seed % 3],
                     ["ascii", "blocks"][(seed // 2) % 2]))
    for seed in range(100, 104):   # key storms
        jobs.append((seed, 20, 0.9, "dark", "gemma", "blocks"))
    print(f"{len(jobs)} app fuzz jobs")
    with get_context("fork").Pool(20) as pool:
        out = list(pool.imap_unordered(app_one, jobs))
    fails = [o for o in out if o.startswith("FAIL")]
    for o in sorted(out):
        print(o if o.startswith("FAIL") else o[:160])
    print(f"\napp fuzz: {len(out) - len(fails)} ok, {len(fails)} failed")
    return len(fails)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "soak"
    t0 = _real_time.time()
    n = run_soak() if which == "soak" else run_app()
    print(f"({_real_time.time() - t0:.0f}s wall)")
    sys.exit(1 if n else 0)
