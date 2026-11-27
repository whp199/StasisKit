"""Development previews for the CATHEDRAL view (no audio, never touches your config).

  python3 -I tests/cathedral_preview.py subject mary [--heights 24,48,96,160] [--out DIR] [--seed N]
  python3 -I tests/cathedral_preview.py subject mercy
  python3 -I tests/cathedral_preview.py subject monstrance
  python3 -I tests/cathedral_preview.py scene [--variants all|name,name] [--size 160x48] [--at 20]
                                              [--modes ascii,blocks] [--out DIR]
                                              [--rosary SECONDS] [--weekday 0-6] [--sheet]

`subject` renders the sprite at each height with the standard candle preview rig
(ASCII | half-blocks | lit pixels, one PNG per height) and prints the render time.
`scene` renders the whole view, one PNG per variant and mode, and prints the
time of the first frame (G-buffer build) and of later frames (update + render,
measured over the last simulated second).  --rosary prays that many seconds of
the rosary first (to see the captions inside a decade: ~260 is the first decade),
--weekday fixes the mysteries (0 = Monday), and --sheet also writes one contact
sheet of every variant and mode.
"""
import argparse
import importlib
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("XDG_CONFIG_HOME", tempfile.mkdtemp(prefix="stasiskit-preview-"))


def cmd_subject(a):
    from stasiskit.scenes.cathedral import kit
    mod = importlib.import_module(f"stasiskit.scenes.cathedral.{a.name}")
    os.makedirs(a.out, exist_ok=True)
    for h in [int(v) for v in a.heights.split(",")]:
        t0 = time.perf_counter()
        spr = mod.render(h, seed=a.seed)
        dt = time.perf_counter() - t0
        spr.check()
        p = os.path.join(a.out, f"{a.name}_h{h}.png")
        kit.preview(spr, p, glow=a.glow)
        print(f"{a.name} h={h:4d} -> {spr.w}x{spr.h} px in {dt * 1000:7.1f} ms   {p}")
        print(f"    anchors: {', '.join(sorted(spr.anchors))}")


def cmd_scene(a):
    import numpy as np
    from stasiskit.canvas import Canvas, save_png
    from stasiskit.scenes.cathedral import Cathedral, rosary, subjects
    os.makedirs(a.out, exist_ok=True)
    w, h = (int(v) for v in a.size.lower().split("x"))
    names = list(Cathedral.variants) if a.variants == "all" else a.variants.split(",")
    written = []
    for name in names:
        sc = Cathedral(None, seed=a.seed)
        if a.weekday is not None:
            sc.rosary = rosary.Rosary(weekday=a.weekday)
        if a.rosary:
            sc.rosary.update(a.rosary)
        sc.set_variant(name)
        t0 = time.perf_counter()
        sc.ensure_size(w, h)
        sc.update(1 / 24)
        cv = Canvas(w, h)
        sc.render(cv, "ascii")
        first = time.perf_counter() - t0
        steps = max(1, int(a.at * 24))
        n, spent = 0, 0.0
        for i in range(steps):
            t1 = time.perf_counter()
            sc.update(1 / 24)
            if i >= steps - 24:
                sc.render(cv, "blocks" if i % 2 else "ascii")
                spent += time.perf_counter() - t1
                n += 1
        per = spent / max(1, n)
        v = sc.view
        cards = ", ".join(f"{k} {'real' if c.real else 'STAND-IN'} {c.spr.w}x{c.spr.h}px" for k, c in v.cards.items())
        print(f"{name:>14} subjects: {cards or 'none on screen'}")
        for k, c in v.cards.items():
            if not c.real:                       # why the subject's own module was not used
                err = subjects.ERRORS.get((k, int(round(c.h_px))), "(no error recorded)")
                print(f"{'':>14} {k} stand-in because:\n" + "\n".join("    " + ln for ln in err.rstrip().splitlines()))
        for mode in a.modes.split(","):
            sc.render(cv, mode)
            assert np.isfinite(cv.fg).all() and np.isfinite(cv.bg).all(), "non-finite colours"
            p = os.path.join(a.out, f"cathedral_{name.replace(' ', '-')}_{mode}_{w}x{h}.png")
            save_png(cv, p)
            written.append((name, mode, p))
            print(f"{name:>14} {mode:6s} {w}x{h}: first frame {first * 1000:6.0f} ms, "
                  f"then ~{per * 1000:5.1f} ms/frame (update+render)   {p}")
    if a.sheet and written:
        from PIL import Image
        rows = {}
        for name, mode, p in written:
            rows.setdefault(name, []).append(Image.open(p).convert("RGB"))
        tw = max(sum(i.width for i in r) for r in rows.values())
        th = sum(max(i.height for i in r) for r in rows.values())
        sheet = Image.new("RGB", (tw + 8, th + 8 * len(rows)), (60, 60, 70))
        y = 0
        for r in rows.values():
            x = 0
            for im in r:
                sheet.paste(im, (x, y))
                x += im.width + 8
            y += max(i.height for i in r) + 8
        k = min(1.0, 2000 / sheet.width)
        sheet = sheet.resize((int(sheet.width * k), int(sheet.height * k)), Image.LANCZOS)
        p = os.path.join(a.out, f"cathedral_sheet_{w}x{h}.png")
        sheet.save(p)
        print(f"sheet: {p}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subject")
    s.add_argument("name", choices=("mary", "mercy", "monstrance"))
    s.add_argument("--heights", default="24,48,96,160")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--glow", type=float, default=0.6)
    s.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "cathedral-preview"))
    c = sub.add_parser("scene")
    c.add_argument("--variants", default="all")
    c.add_argument("--size", default="160x48")
    c.add_argument("--at", type=float, default=20.0)
    c.add_argument("--modes", default="ascii,blocks")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--rosary", type=float, default=0.0, help="seconds of the rosary to pray first")
    c.add_argument("--weekday", type=int, default=None, help="0 = Monday ... 6 = Sunday")
    c.add_argument("--sheet", action="store_true", help="also write a contact sheet")
    c.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "cathedral-preview"))
    a = p.parse_args()
    (cmd_subject if a.cmd == "subject" else cmd_scene)(a)


if __name__ == "__main__":
    main()
