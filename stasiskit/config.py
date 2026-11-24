"""User configuration (~/.config/stasiskit/config.toml) and saved state (state.json).

Both files are hand-editable, so every value is checked on load: a wrong type
falls back to the default, a number out of range is clamped."""

import json
import math
import os
import tomllib

DEFAULTS = {
    "companion": "gemma",        # companion id (file name in companions/)
    "always_show_companion": False,  # keep the companion portrait visible even when quiet
    "human_name": "",            # what your companion calls you ("" = companion's default)
    "fps": 24,
    "render_mode": "ascii",      # "ascii" or "blocks" (hi-res half-block pixels)
    "start_scene": "",           # scene key; "" = resume the last one
    "scene_cycle_minutes": 7.0,  # auto-cycle interval when auto-cycle is on (0 = never)
    "autocycle": True,
    "chattiness": 2,             # 0 silent .. 3 chatty
    "boot_sequence": True,
    "timer_minutes": 45,         # default selection in the timer panel
    "timer_fade_minutes": 3.0,   # length of the fade at the end of a sleep timer
    "on_timer_end": "dark",      # "dark", "dim" or "exit"
    "audio": True,
    "audio_backend": "auto",     # auto, paplay, pw-cat, pacat, aplay, sox, ffplay
    "hud_seconds": 7.0,          # HUD auto-hides after this many idle seconds
    "brightness": 1.0,
    "master_volume": 0.8,
}

RENDER_MODES = ("ascii", "blocks")
END_ACTIONS = ("dark", "dim", "exit")
AUDIO_BACKENDS = ("auto", "paplay", "pw-cat", "pacat", "aplay", "sox", "ffplay")   # engine.BACKENDS
CHOICES = {"render_mode": RENDER_MODES, "on_timer_end": END_ACTIONS, "audio_backend": AUDIO_BACKENDS}
RANGES = {"fps": (4, 60), "chattiness": (0, 3), "timer_minutes": (0, 600),
          "brightness": (0.1, 1.0), "master_volume": (0.0, 1.0)}   # other numbers: >= 0

# config keys that saved state overrides (the in-app choices resume), and their state.json names
STATE_KEYS = {"companion": "companion", "always_show_companion": "always_show_companion",
              "render_mode": "mode", "autocycle": "autocycle",
              "timer_minutes": "timer_minutes", "master_volume": "master", "brightness": "brightness"}


def config_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "stasiskit")


def config_path():
    return os.path.join(config_dir(), "config.toml")


def state_path():
    return os.path.join(config_dir(), "state.json")


# ---------------------------------------------------------------------- checking values
# Each checker returns a valid value or raises TypeError / ValueError / AttributeError.
def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    s = v.strip().lower()
    if s in ("true", "yes", "on", "1"):
        return True
    if s in ("false", "no", "off", "0"):
        return False
    raise ValueError(v)


def _num(v, lo=0.0, hi=math.inf):
    """A finite number (or numeric string) clamped to lo..hi."""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise TypeError(v)
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(v)
    return min(hi, max(lo, x))


def _int(v, lo=0, hi=math.inf):
    x = _num(v, lo, hi)
    if not x.is_integer():
        raise ValueError(v)
    return int(x)


def _text(v):
    if not isinstance(v, str) or not v:
        raise ValueError(v)
    return v


def _choice(v, choices):
    if v not in choices:
        raise ValueError(v)
    return v


def _scalar(v):
    if not isinstance(v, (str, int, float, bool)):
        raise TypeError(v)
    return v


def _each(d, fn, keys=None):
    """dict -> dict of the entries `fn` accepts (only `keys`, if given)."""
    out = {}
    for k, v in d.items():
        if keys is None or k in keys:
            try:
                out[k] = fn(v)
            except (TypeError, ValueError, AttributeError):
                pass
    return out


def clean_value(key, v):
    """`v` as a valid value for config key `key`, or the default."""
    d = DEFAULTS[key]
    try:
        if isinstance(d, bool):
            return _bool(v)
        if key in CHOICES:
            return _choice(v.strip().lower(), CHOICES[key])
        if isinstance(d, str):
            if key == "start_scene" and type(v) is int:
                v = str(v)                       # start_scene = 3 means view 3
            return _text(v.strip())
        if key == "master_volume" and _num(v) > 2:
            v = _num(v) / 100                    # master_volume = 80 means 80 %; 1..2 clamps to full
        x = _num(v, *RANGES.get(key, (0.0, math.inf)))
        return int(round(x)) if isinstance(d, int) else x
    except (TypeError, ValueError, AttributeError):
        return d


def clean_config(cfg):
    """Check every known key of `cfg` in place (missing keys get defaults); return it."""
    cfg.update({k: clean_value(k, cfg.get(k, d)) for k, d in DEFAULTS.items()})
    return cfg


def clean_state(st):
    """Drop malformed fields of a loaded state.json, so the config value is used instead."""
    if not isinstance(st, dict):
        return {}
    from .audio.sources import LAYER_KEYS

    def unit(v):
        return _num(v, 0.0, 1.0)

    fields = {
        "levels": lambda v: _each(v, unit, LAYER_KEYS),
        "presets": lambda v: _each(v, _text),
        "variants": lambda v: _each(v, _int),
        "_config_seen": lambda v: _each(v, _scalar, STATE_KEYS),
        "master": unit,
        "brightness": lambda v: _num(v, 0.1, 1.0),
        "timer_minutes": lambda v: int(round(_num(v, 0, 600))),
        "muted": _bool,
        "autocycle": _bool,
        "always_show_companion": _bool,
        "companion": _text,
        "scene": _text,
        "mode": lambda v: _choice(v, RENDER_MODES),
    }
    out = dict(st)                               # unknown keys pass through untouched
    for k, fn in fields.items():
        if k in out:
            try:
                out[k] = fn(out[k])
            except (TypeError, ValueError, AttributeError):
                del out[k]
    return out


def config_snapshot(cfg):
    """The config values saved state can override, stored with the state (see resolve_state)."""
    return {k: cfg[k] for k in STATE_KEYS}


def resolve_state(cfg, state):
    """Saved state normally wins over config.toml, so in-app choices resume.  But a key whose
    config.toml value changed since the state was saved was edited by the user: config wins."""
    seen = state.get("_config_seen")
    seen = seen if isinstance(seen, dict) else {}    # older state.json: compare with the defaults
    st = dict(state)
    for ck, sk in STATE_KEYS.items():
        if cfg[ck] != seen.get(ck, DEFAULTS[ck]):
            st.pop(sk, None)
    return st


# ---------------------------------------------------------------------- files
def load_config():
    cfg = dict(DEFAULTS)
    p = config_path()
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                user = tomllib.load(f)
            for k, v in user.items():
                if k in DEFAULTS:
                    cfg[k] = v
        except (OSError, ValueError, RecursionError) as e:   # ValueError: bad TOML or not UTF-8
            cfg = dict(DEFAULTS)
            cfg["_config_error"] = str(e)
    return clean_config(cfg)


def load_state():
    try:
        with open(state_path(), encoding="utf-8") as f:
            return clean_state(json.load(f))
    except (OSError, ValueError, RecursionError):
        return {}


def save_state(state):
    data = json.dumps(state, indent=1, sort_keys=True)   # before touching the file
    try:
        os.makedirs(config_dir(), exist_ok=True)
        tmp = state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, state_path())
    except OSError:
        pass


EXAMPLE_CONFIG = """\
# StasisKit configuration - every key is optional.
# companion, always_show_companion, render_mode, autocycle and timer_minutes are
# also remembered from the last session (C, s, r, a and the timer panel); an edit
# here takes over from that.
companion = "gemma"          # "gemma", "neon", "claude", or the id of your own companion file
always_show_companion = false # keep the companion portrait on screen when quiet (toggle with s)
human_name = ""              # what your companion calls you
fps = 24                     # 4 .. 60
render_mode = "ascii"        # "ascii" or "blocks"
autocycle = true
scene_cycle_minutes = 7      # 0 = never switch views by itself
chattiness = 2               # 0 silent, 1 quiet, 2 normal, 3 chatty
boot_sequence = true
timer_minutes = 45
timer_fade_minutes = 3
on_timer_end = "dark"        # "dark", "dim" or "exit"
audio = true
audio_backend = "auto"       # auto, paplay, pw-cat, pacat, aplay, sox, ffplay
hud_seconds = 7
"""
