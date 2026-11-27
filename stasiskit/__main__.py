"""Command line entry point:  python3 -m stasiskit  (or the ./stasiskit launcher)."""

import argparse
import math
import os
import re
import sys

from . import __version__


_NUM = r"((?>\d+(?:\.\d*)?|\.\d+))\s*"        # atomic: '1030s' is not 103 min + 0 s
_DURATION = re.compile(rf"(?:{_NUM}h(?:ours?|rs?)?)?\s*(?:{_NUM}(?:m(?:in(?:ute)?s?)?)?)?\s*"
                       rf"(?:{_NUM}s(?:ec(?:ond)?s?)?)?")


def parse_minutes(s):
    """'45' -> 45, '45m' -> 45, '1h30m' / '1h30' -> 90, '1.5h' -> 90, '90s' -> 1.5"""
    m = _DURATION.fullmatch(str(s).strip().lower().replace(",", "."))
    if not m or not any(m.groups()):
        raise argparse.ArgumentTypeError(f"can't read duration {s!r} (try 45, 45m, 1h30 or 90s)")
    h, mins, secs = (float(g or 0) for g in m.groups())
    return h * 60 + mins + secs / 60


def parse_size(s):
    """'160x48' -> (160, 48)"""
    m = re.fullmatch(r"\s*(\d+)\s*[x×*]\s*(\d+)\s*", str(s).lower())
    w, h = (int(m[1]), int(m[2])) if m else (0, 0)
    if not (20 <= w <= 1000 and 10 <= h <= 500):
        raise argparse.ArgumentTypeError(f"bad size {s!r}: use WIDTHxHEIGHT in cells, 20x10 up to 1000x500")
    return w, h


def build_parser():
    p = argparse.ArgumentParser(
        prog="stasiskit",
        description="StasisKit - an ASCII hypersleep companion: animated views, drones, noise and nature sounds.")
    p.add_argument("-c", "--companion", help="companion id (gemma, neon, claude, or your own)")
    p.add_argument("--show-companion", action="store_true", help="always show the companion portrait")
    p.add_argument("-s", "--scene", help="start on this view (name or number)")
    p.add_argument("-t", "--timer", type=parse_minutes, help="start a sleep timer, e.g. 45, 45m, 1h30m")
    p.add_argument("--mode", choices=("ascii", "blocks"), help="render style")
    p.add_argument("--fps", type=int, help="frames per second (default 24)")
    p.add_argument("--no-audio", action="store_true", help="disable sound entirely")
    p.add_argument("--mute", action="store_true", help="start muted")
    p.add_argument("--no-boot", action="store_true", help="skip the boot sequence")
    p.add_argument("--screensaver", action="store_true",
                   help="no HUD or chatter, auto-cycle views, any key exits")
    p.add_argument("--list", action="store_true", help="list views, companions and sound layers")
    p.add_argument("--init-config", action="store_true", help="write an example ~/.config/stasiskit/config.toml")
    p.add_argument("--snapshot", metavar="VIEW", help="render a view to a PNG instead of running")
    p.add_argument("--out", default="stasiskit.png", help="output file for --snapshot")
    p.add_argument("--size", type=parse_size, default="160x48", help="snapshot size in cells, e.g. 160x48")
    p.add_argument("--at", type=float, default=12.0, help="snapshot time in seconds")
    p.add_argument("--variant", help="snapshot variant (name or number, see --list)")
    p.add_argument("--export-audio", metavar="FILE.wav", help="render the saved mix to a WAV file")
    p.add_argument("--minutes", type=parse_minutes, default=30, help="length for --export-audio")
    p.add_argument("--version", action="version", version=f"StasisKit {__version__}")
    return p


def cmd_list():
    from .audio.sources import LAYERS
    from .companion import load_all
    from .scenes import SCENES
    print("VIEWS")
    for i, s in enumerate(SCENES, 1):
        print(f"  {i:>2}. {s.key:<10} {s.title:<22} {s.subtitle}")
        if len(s.variants) > 1:
            print(f"      {'':<10} variants: {', '.join(s.variants)}")
    print("\nCOMPANIONS")
    for cid, d in sorted(load_all().items()):
        print(f"  {cid:<10} {d.get('name', cid)} - {d.get('title', '')}  ({d['_path']})")
    print("\nSOUND LAYERS")
    for c in LAYERS:
        ps = ", ".join(c.presets) if c.presets else "follows the current view"
        print(f"  {c.key:<10} {c.label:<15} {ps}")


def pick_variant(cls, name):
    """Variant index for a name (or unique part of one) or a 1-based number; exit if unknown."""
    names = [v.lower() for v in cls.variants]
    q = name.strip().lower()
    if q.isdigit() and 1 <= int(q) <= len(names):
        return int(q) - 1
    hits = [names.index(q)] if q in names else [i for i, v in enumerate(names) if q and q in v]
    if len(hits) == 1:
        return hits[0]
    known = ", ".join(f"{i}. {v}" for i, v in enumerate(cls.variants, 1))
    sys.exit(f"unknown variant {name!r} for {cls.key}; choose one of: {known}")


def cmd_snapshot(args):
    import random
    from .canvas import Canvas, save_png
    from .scenes import SCENES
    w, h = args.size
    key = args.snapshot.lower()
    cls = next((c for i, c in enumerate(SCENES, 1)
                if c.key == key or str(i) == key or (key == "0" and i == 10)), None)   # 0 = view 10
    if cls is None:
        sys.exit(f"unknown view {args.snapshot!r}; try --list")
    if not math.isfinite(args.at):
        sys.exit("--at needs a number of seconds")
    sc = cls(None, seed=random.randrange(1 << 30))
    if args.variant is not None:
        sc.set_variant(pick_variant(cls, args.variant))
    sc.ensure_size(w, h)
    cv = Canvas(w, h)
    dt = 1 / 24
    for _ in range(max(1, int(max(0.0, args.at) / dt))):   # scenes expect an update before a render
        sc.update(dt)
    sc.render(cv, args.mode or "ascii")
    save_png(cv, args.out)
    print(f"wrote {args.out}")


def cmd_export(args):
    from .audio.dsp import BLOCK, SR
    from .audio.engine import render_wav
    from .config import load_config, load_state, resolve_state
    secs = args.minutes * 60
    # 16-bit stereo WAV: the RIFF sizes are 32-bit, so a file holds at most ~6 h 12 min
    max_blocks = (0xFFFFFFFF - 36) // 4 // BLOCK
    if secs <= 0:
        sys.exit("--minutes must be more than 0")
    if math.ceil(secs * SR / BLOCK) > max_blocks:
        lim = int(max_blocks * BLOCK / SR / 60)
        sys.exit(f"--minutes {args.minutes:g} is too long: a WAV file holds at most {lim} minutes "
                 f"({lim // 60} h {lim % 60} min) - export the night in parts")
    cfg = load_config()
    st = resolve_state(cfg, load_state())

    def prog(f):
        print(f"\r  rendering… {f * 100:5.1f}%", end="", file=sys.stderr, flush=True)

    render_wav(args.export_audio, secs, st.get("levels"), st.get("presets"), st.get("master", cfg["master_volume"]),
               ambience=None, progress=prog)
    print(f"\r  wrote {args.export_audio} ({args.minutes:g} min)          ", file=sys.stderr)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list:
        return cmd_list()
    if args.init_config:
        from .config import EXAMPLE_CONFIG, config_path
        p = config_path()
        if os.path.exists(p):
            sys.exit(f"{p} already exists - not overwriting")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(EXAMPLE_CONFIG)
        print(f"wrote {p}")
        return
    if args.snapshot:
        return cmd_snapshot(args)
    if args.export_audio:
        return cmd_export(args)

    from .app import App
    from .config import load_config, load_state
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("StasisKit needs an interactive terminal.")
    cfg = load_config()
    app = App(cfg, load_state(), args)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    if not args.screensaver:
        try:
            print(app.farewell())
        except OSError:                  # the terminal went away (hang-up)
            pass


if __name__ == "__main__":
    main()
