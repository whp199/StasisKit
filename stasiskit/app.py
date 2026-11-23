"""The StasisKit application: main loop, input, overlays, sleep timer."""

import math
import random
import signal
import sys
import time

from . import ui
from .audio.engine import AudioClient
from .audio.sources import LAYER_KEYS, LAYER_PRESETS
from .canvas import AnsiEncoder, Canvas
from .companion import Companion, load_all
from .config import clean_config, clean_state, config_snapshot, resolve_state, save_state
from .scenes import SCENES
from .term import Terminal
from .util import fmt_duration

IDLE_GAP = {1: (600, 1000), 2: (260, 520), 3: (120, 240)}
SCENE_LINE_P = {0: 0.0, 1: 0.3, 2: 0.65, 3: 0.95}


def _units(n, unit):
    return f"{n} {unit}" + ("" if n == 1 else "s")


def human_duration(minutes):
    secs = max(1, round(minutes * 60))
    if secs < 60:
        return _units(secs, "second")
    h, m = divmod(int(minutes + 0.5), 60)
    if not h:
        return _units(m, "minute")
    return _units(h, "hour") + (f" {_units(m, 'minute')}" if m else "")


class App:
    def __init__(self, cfg, state, opts):
        self.cfg = cfg = clean_config(cfg)          # in place: callers may keep a reference
        self.state = state = resolve_state(cfg, clean_state(state))
        self.opts = opts
        self.screensaver = bool(getattr(opts, "screensaver", False))
        self.companions = load_all()
        cid = getattr(opts, "companion", None) or state.get("companion") or cfg["companion"]
        if cid not in self.companions:
            cid = "gemma" if "gemma" in self.companions else next(iter(self.companions))
        self.companion = Companion(self.companions[cid], cfg.get("human_name") or None)

        self.scene_classes = SCENES
        self.scenes = {}
        self.variants = dict(state.get("variants", {}))
        self.scene_idx = self._resolve_scene(getattr(opts, "scene", None) or cfg.get("start_scene")
                                             or state.get("scene") or SCENES[0].key)
        self.mode = getattr(opts, "mode", None) or state.get("mode") or cfg["render_mode"]
        if self.mode not in ("ascii", "blocks"):
            self.mode = "ascii"
        self.brightness = float(state.get("brightness", cfg["brightness"]))
        self.autocycle = True if self.screensaver else bool(state.get("autocycle", cfg["autocycle"]))
        self.fps = max(4, min(60, int(getattr(opts, "fps", None) or cfg["fps"])))

        audio_on = bool(cfg["audio"]) and not getattr(opts, "no_audio", False)
        self.audio = AudioClient(
            enabled=audio_on, levels=state.get("levels"), presets=state.get("presets"),
            master=float(state.get("master", cfg["master_volume"])),
            muted=bool(getattr(opts, "mute", False) or state.get("muted", False)),
            backend=cfg["audio_backend"])

        self.cv = Canvas(80, 24)
        self.enc = AnsiEncoder()
        self.term = None
        self.running = True
        self.panel = None
        self.mixer_sel = 1
        self._mixer_mem = {}
        self.timer_sel = int(state.get("timer_minutes", cfg["timer_minutes"]))
        self.speech = None
        self.hud_until = 0.0
        self.hud_pin = None          # None auto, True always, False never
        self.timer_end = None
        self.timer_phase = None
        self.timer_fade = 0.0
        self.timer_warned = False
        self.sleep_gain = 1.0
        self.sleep_target = 1.0
        self.asleep_t0 = 0.0
        self.trans = None
        self.trans_gain = 1.0
        self.boot_t0 = 0.0
        self.boot_greeted = False
        self.breathe_t0 = 0.0
        self.started = time.time()
        self.next_idle = self.started + self._idle_gap()
        self.cycle_at = self.started + self._cycle_secs()
        self.pending_timer = getattr(opts, "timer", None)
        self._cli_mute = bool(getattr(opts, "mute", False))
        self._mute_toggled = False

    # ------------------------------------------------------------------ scenes
    def _resolve_scene(self, key):
        if key is None:
            return 0
        s = str(key).strip().lower()
        if s.isdigit():
            return (int(s) - 1) % len(SCENES)
        for i, cls in enumerate(SCENES):
            if cls.key == s or cls.title.lower() == s:
                return i
        for i, cls in enumerate(SCENES):
            if s in cls.key or s in cls.title.lower():
                return i
        return 0

    @property
    def scene(self):
        sc = self.scenes.get(self.scene_idx)
        if sc is None:
            cls = SCENES[self.scene_idx]
            sc = cls(self, seed=random.randrange(1 << 30))
            v = self.variants.get(cls.key)
            if v is not None:
                sc.set_variant(int(v))
            self.scenes[self.scene_idx] = sc
        return sc

    def _cycle_secs(self):
        m = self.cfg["scene_cycle_minutes"]
        return m * 60 if m > 0 else math.inf       # 0 = never switch by itself

    def _target(self):
        """The view being shown, or the one a running fade-out is heading for."""
        tr = self.trans
        return tr["to"] if tr and tr["phase"] == "out" else self.scene_idx

    def goto(self, idx, by_user=True):
        idx %= len(SCENES)
        tr, now = self.trans, time.time()
        if tr and tr["phase"] == "out":
            if idx == self.scene_idx:              # changed our mind: fade back in from here
                self.trans = {"phase": "in", "t0": now - self.trans_gain * 0.7, "to": idx, "user": by_user}
            else:                                  # retarget, keep the fade going
                tr["to"], tr["user"] = idx, tr["user"] or by_user
        elif idx != self.scene_idx:                # (fading in: fade out from the current gain)
            self.trans = {"phase": "out", "t0": now - (1.0 - self.trans_gain) * 0.45, "to": idx, "user": by_user}

    def _switch_now(self, idx, by_user):
        self.scene.on_exit()
        self.scene_idx = idx
        sc = self.scene
        sc.on_enter()
        self.audio.ambience(sc.ambience)
        self.cycle_at = time.time() + self._cycle_secs()
        p = SCENE_LINE_P.get(self.cfg["chattiness"], 0.6) * (1.0 if by_user else 0.5)
        if not self.screensaver and random.random() < p:
            self.say(f"scene.{sc.key}")

    def audio_event(self, name, **kw):
        self.audio.event(name, **kw)

    # ------------------------------------------------------------------ speech
    def say(self, category, happy=None, **kw):
        if self.screensaver:
            return None
        text = self.companion.line(category, **kw)
        if text:
            if happy is None:
                happy = category in ("greeting", "wake", "breathe_end", "timer_set", "quit")
            self.speech = ui.Speech(text, time.time(), self.companion.cps, happy=happy)
        return text

    def _idle_gap(self):
        gap = IDLE_GAP.get(self.cfg["chattiness"])
        return random.uniform(*gap) if gap else 1e12

    # ------------------------------------------------------------------ timer
    def fade_seconds(self, dur):
        return max(10.0, min(self.cfg["timer_fade_minutes"] * 60.0, dur / 3.0))

    def start_timer(self, minutes):
        had = self.timer_end is not None           # wake() clears it
        self.wake(silent=True)
        if minutes <= 0:
            if had:
                self.say("timer_off")
            self.timer_end = None
            self.timer_phase = None
            return
        now = time.time()
        dur = minutes * 60.0
        self.timer_end = now + dur
        self.timer_phase = "run"
        self.timer_fade = self.fade_seconds(dur)
        self.timer_warned = dur <= 6 * 60
        self.say("timer_set", duration=human_duration(minutes))

    def wake(self, silent=False):
        was = self.timer_phase in ("fade", "asleep")
        self.timer_end = None
        self.timer_phase = None
        self.sleep_target = 1.0
        if was:
            self.audio.fade_to(1.0, 4.0)
            self.enc.invalidate()
            if not silent:
                self.say("wake")

    def _update_timer(self, now, dt):
        ph = self.timer_phase
        if ph is None:
            self.sleep_target = 1.0
        elif ph in ("run", "fade"):
            rem = self.timer_end - now
            if ph == "run":
                if not self.timer_warned and rem <= 300:
                    self.timer_warned = True
                    self.say("timer_warn")
                if rem <= self.timer_fade:
                    self.timer_phase = "fade"
                    self.audio.fade_to(0.0, max(1.0, rem))
                    self.say("goodnight", happy=True)
                    if self.speech:
                        self.speech.hold = min(self.speech.hold + 6, rem * 0.6)
            else:
                k = max(0.0, min(1.0, rem / max(1.0, self.timer_fade)))
                self.sleep_target = 0.06 + 0.94 * k ** 1.3
                if rem <= 0:
                    self.timer_phase = "asleep"
                    self.asleep_t0 = now
                    self.speech = None
                    self.panel = None
        if self.timer_phase == "asleep":
            end = self.cfg["on_timer_end"]
            self.sleep_target = 0.08 if end == "dim" else 0.0
            if end == "exit" and now - self.asleep_t0 > 4.0:
                self.running = False
        rate = 0.5 if self.sleep_target < self.sleep_gain else 1.2
        self.sleep_gain += (self.sleep_target - self.sleep_gain) * min(1.0, dt * rate)
        if abs(self.sleep_gain - self.sleep_target) < 0.002:
            self.sleep_gain = self.sleep_target

    # ------------------------------------------------------------------ input
    def on_key(self, k, now):
        if self.timer_phase == "asleep":
            self.wake()
            return
        if self.screensaver:
            self.running = False
            return
        if self.panel == "boot":
            self.panel = None
            if not self.boot_greeted:
                self.boot_greeted = True
                self.say("greeting")
            return
        if self.panel in ("help", "breathe"):
            self.panel = None
            return
        if self.panel == "mixer":
            return self._key_mixer(k)
        if self.panel == "timer":
            return self._key_timer(k)
        self.hud_until = now + self.cfg["hud_seconds"]
        n = len(SCENES)
        if k in ("q", "esc", "ctrl-c", "ctrl-d"):
            self.running = False
        elif k in ("right", "l", "n", "pgdn"):
            self.goto(self._target() + 1)
        elif k in ("left", "h", "p", "pgup"):
            self.goto(self._target() - 1)
        elif len(k) == 1 and k.isdigit() and k != "0":
            if int(k) <= n:
                self.goto(int(k) - 1)
        elif k in ("v", "V"):
            sc = self.scene
            sc.next_variant(1 if k == "v" else -1)
            self.variants[sc.key] = sc.variant
        elif k == "m":
            self.panel = "mixer"
            if random.random() < 0.35:
                self.say("mixer")
        elif k == "t":
            self.panel = "timer"
            if self.timer_end:
                self.timer_sel = max(1, int(round((self.timer_end - now) / 60)))
        elif k in ("?", "f1"):
            self.panel = "help"
        elif k == "b":
            self.panel = "breathe"
            self.breathe_t0 = now
            self.say("breathe_start")
            if self.speech:
                self.speech.hold = 2.0
        elif k == "c":
            self.say("talk")
        elif k == "C":
            ids = sorted(self.companions)
            i = (ids.index(self.companion.id) + 1) % len(ids) if self.companion.id in ids else 0
            self.companion = Companion(self.companions[ids[i]], self.cfg.get("human_name") or None)
            self.say("greeting")
        elif k == "r":
            self.mode = "blocks" if self.mode == "ascii" else "ascii"
        elif k == "[":
            self.brightness = max(0.1, round(self.brightness - 0.1, 2))
        elif k == "]":
            self.brightness = min(1.0, round(self.brightness + 0.1, 2))
        elif k in ("-", "_"):
            self.audio.set_master(self.audio.master - 0.05)
        elif k in ("+", "="):
            self.audio.set_master(self.audio.master + 0.05)
        elif k == "M":
            self.audio.toggle_mute()
            self._mute_toggled = True
        elif k == "a":
            self.autocycle = not self.autocycle
            self.cycle_at = now + self._cycle_secs()
        elif k in ("i", "tab"):
            self.hud_pin = {None: True, True: False, False: None}[self.hud_pin]
        elif k == "ctrl-l":
            self.enc.invalidate()

    def _key_mixer(self, k):
        rows = ["master"] + LAYER_KEYS
        sel = self.mixer_sel
        key = rows[sel]
        au = self.audio
        level = au.master if key == "master" else au.levels.get(key, 0.0)

        def setv(v):
            if key == "master":
                au.set_master(v)
            else:
                au.set_level(key, v)

        if k in ("up", "k"):
            self.mixer_sel = (sel - 1) % len(rows)
        elif k in ("down", "j", "tab"):
            self.mixer_sel = (sel + 1) % len(rows)
        elif k in ("left", "h", "-"):
            setv(level - 0.05)
        elif k in ("right", "l", "+", "="):
            setv(level + 0.05)
        elif k == "pgdn":
            setv(level - 0.2)
        elif k == "pgup":
            setv(level + 0.2)
        elif k in ("enter", "space"):
            if key != "master" and LAYER_PRESETS.get(key):
                au.cycle_preset(key)
        elif k == "backspace":
            if key != "master" and LAYER_PRESETS.get(key):
                au.cycle_preset(key, -1)
        elif k == "0":
            if level > 0:
                self._mixer_mem[key] = level
                setv(0.0)
            else:
                setv(self._mixer_mem.get(key, 0.5))
        elif k == "M":
            au.toggle_mute()
            self._mute_toggled = True
        elif k in ("esc", "m", "q", "ctrl-c"):
            self.panel = None
            self._save()

    def _key_timer(self, k):
        ch = ui.TIMER_CHOICES
        m = self.timer_sel
        if k in ("left", "h"):
            lower = [c for c in ch if c < m]
            self.timer_sel = lower[-1] if lower else 0
        elif k in ("right", "l"):
            higher = [c for c in ch if c > m]
            self.timer_sel = higher[0] if higher else m
        elif k in ("up", "k", "+", "="):
            self.timer_sel = min(600, m + 1)
        elif k in ("down", "j", "-"):
            self.timer_sel = max(0, m - 1)
        elif k == "e":
            acts = list(ui.END_ACTIONS)
            cur = self.cfg["on_timer_end"]
            self.cfg["on_timer_end"] = acts[(acts.index(cur) + 1) % len(acts)] if cur in acts else acts[0]
        elif k in ("enter", "space"):
            self.panel = None
            self.start_timer(self.timer_sel)
            self._save()
        elif k in ("esc", "t", "q", "ctrl-c"):
            self.panel = None

    # ------------------------------------------------------------------ frame
    def step(self, dt, now):
        self.audio.poll()
        if self.term is not None:
            w, h = self.term.size()
            if (w, h) != (self.cv.w, self.cv.h):
                self.cv.resize(w, h)
                self.enc.invalidate()
        # scene transition
        if self.trans:
            tr = self.trans
            el = now - tr["t0"]
            if tr["phase"] == "out":
                self.trans_gain = max(0.0, 1.0 - el / 0.45)
                if el >= 0.45:
                    self._switch_now(tr["to"], tr["user"])
                    tr["phase"] = "in"
                    tr["t0"] = now
            else:
                self.trans_gain = min(1.0, el / 0.7)
                if el >= 0.7:
                    self.trans = None
                    self.trans_gain = 1.0
        self._update_timer(now, dt)
        if self.pending_timer:
            self.start_timer(self.pending_timer)
            self.pending_timer = None
        # boot
        if self.panel == "boot":
            done_t = 0.5 + len(ui.boot_lines(self)) * 0.42
            el = now - self.boot_t0
            if el > done_t and not self.boot_greeted:
                self.boot_greeted = True
                self.say("greeting")
            if self.boot_greeted and self.speech and now > self.speech.t0 + self.speech.type_time + 3.0:
                self.panel = None
        # breathing
        if self.panel == "breathe":
            n, *_ = ui.breath_state(now - self.breathe_t0)
            if n >= ui.BREATH_CYCLES:
                self.panel = None
                self.say("breathe_end")
        # auto-cycle
        if (self.autocycle and self.panel is None and self.trans is None and self.timer_phase != "asleep"
                and now >= self.cycle_at):
            self.goto(self.scene_idx + 1, by_user=False)
            self.cycle_at = now + self._cycle_secs()
        # idle chatter
        if now >= self.next_idle:
            self.next_idle = now + self._idle_gap()
            if self.panel is None and self.speech is None and self.timer_phase in (None, "run"):
                self.say("idle")
        if self.speech and self.speech.done(now):
            self.speech = None
        if self.timer_phase == "asleep" and self.sleep_gain < 0.005:
            return
        sc = self.scene
        sc.ensure_size(self.cv.w, self.cv.h)
        sc.update(dt)

    def hud_alpha(self, now):
        if self.screensaver or self.hud_pin is False:
            return 0.0
        if self.hud_pin or self.panel in ("mixer", "timer", "help"):
            return 1.0
        return max(0.0, min(1.0, (self.hud_until - now) / 0.8))

    def draw(self, now):
        cv = self.cv
        if cv.w < 40 or cv.h < 12:
            cv.clear()
            msg = "StasisKit needs at least 40x12"
            cv.text(max(0, (cv.w - len(msg)) // 2), cv.h // 2, msg[:cv.w], ui.DIM)
            return 1.0
        if self.timer_phase == "asleep" and self.sleep_gain < 0.005:
            cv.clear()
            return 0.0
        sc = self.scene
        sc.render(cv, self.mode)
        if self.panel == "boot":
            cv.scale(0.45)
            ui.draw_boot(cv, self, now)
        elif self.panel == "breathe":
            ui.draw_breathe(cv, self, now)
        a = self.hud_alpha(now)
        if self.panel != "boot":
            if a > 0.02:
                ui.draw_hud(cv, self, a)
            elif not self.screensaver:
                ui.draw_mini_timer(cv, self)
        ui.draw_speech(cv, self, now)
        if self.panel == "mixer":
            ui.draw_mixer(cv, self)
        elif self.panel == "timer":
            ui.draw_timer(cv, self)
        elif self.panel == "help":
            ui.draw_help(cv, self)
        return self.brightness * self.trans_gain * self.sleep_gain

    # ------------------------------------------------------------------ run
    def _save(self):
        au = self.audio
        save_state(dict(
            levels=au.levels, presets=au.presets, master=au.master,
            muted=au.muted if (self._mute_toggled or not self._cli_mute) else bool(self.state.get("muted", False)),
            companion=self.companion.id, scene=SCENES[self.scene_idx].key, mode=self.mode,
            brightness=self.brightness, autocycle=self.autocycle if not self.screensaver else
            self.state.get("autocycle", self.cfg["autocycle"]), variants=self.variants,
            timer_minutes=self.timer_sel, _config_seen=config_snapshot(self.cfg)))

    def run(self, term=None):
        self.audio.start(ambience=self.scene.ambience)
        own_term = term is None
        term = term or Terminal()
        try:
            if own_term:
                term.start()
            self.term = term
            now = time.time()
            if self.cfg["boot_sequence"] and not self.screensaver and not getattr(self.opts, "no_boot", False):
                self.panel = "boot"
                self.boot_t0 = now
            elif not self.screensaver:
                self.say("greeting")
                self.hud_until = now + self.cfg["hud_seconds"]
            self.scene.on_enter()
            self._loop()
        finally:
            self._shutdown(term if own_term else None)
        return self

    def _shutdown(self, term):
        """Save, restore the tty, stop audio: each step runs even if another one fails (a hung-up
        tty), and a second SIGTERM/SIGHUP (systemd sends both) cannot cut the cleanup short."""
        old = {}
        if term is not None:
            for sig in (signal.SIGTERM, signal.SIGHUP):
                try:
                    old[sig] = signal.signal(sig, signal.SIG_IGN)
                except (ValueError, OSError):
                    pass
        err = None
        for fn in (self._save, term and term.stop, self.audio.stop):
            try:
                if fn:
                    fn()
            except Exception as e:
                err = err or e
        for sig, h in old.items():
            signal.signal(sig, h if h is not None else signal.SIG_DFL)
        if err is not None and sys.exception() is None:   # don't mask why the loop ended
            raise err

    def _loop(self):
        term = self.term
        frame_dt = 1.0 / self.fps
        next_frame = time.perf_counter()
        last = time.time()
        while self.running:
            fdt = frame_dt if not (self.timer_phase == "asleep" and self.sleep_gain < 0.01) else 0.25
            keys = term.read_keys(max(0.0, next_frame - time.perf_counter()))
            now = time.time()
            for k in keys:
                self.on_key(k, now)
                if not self.running:
                    break
            if not self.running:
                break
            pc = time.perf_counter()
            if pc < next_frame:
                continue
            next_frame = next_frame + fdt if pc - next_frame < fdt else pc + fdt
            dt = max(0.0, min(0.1, now - last))
            last = now
            self.step(dt, now)
            gain = self.draw(now)
            out = self.enc.encode(self.cv, gain)
            if out:
                term.write(out)

    def farewell(self):
        line = self.companion.line("quit")
        r, g, b = (int(v * 255) for v in self.companion.accent)
        mins = (time.time() - self.started) / 60
        s = f"\x1b[38;2;{r};{g};{b}m{self.companion.name}\x1b[0m: {line}" if line else ""
        s += f"\n\x1b[2m  session {human_duration(mins) if mins >= 1 else 'under a minute'}\x1b[0m"
        return s
