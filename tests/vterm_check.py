"""Replays the ANSI encoder's diffed output through a minimal virtual terminal and checks that
every cell matches the canvas, for every view in both render modes.  python3 -I tests/vterm_check.py"""
import sys, re, unicodedata, types, time, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="stasiskit-vterm-")
import numpy as np
from stasiskit.canvas import Canvas, AnsiEncoder
from stasiskit.scenes import SCENES
from stasiskit import ui

class VT:
    def __init__(s, w, h):
        s.w, s.h = w, h; s.ch = [[" "]*w for _ in range(h)]; s.fg = [[0]*w for _ in range(h)]; s.bg = [[0]*w for _ in range(h)]
        s.x = s.y = 0; s.f = s.b = 0
    def feed(s, data):
        i = 0; n = len(data)
        while i < n:
            c = data[i]
            if c == "\x1b":
                if data[i+1] == "[":
                    m = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])").match(data, i)
                    p, f = m.group(1), m.group(2); i = m.end()
                    if f == "H":
                        a = [int(v) for v in p.split(";")] if p else [1, 1]
                        s.y, s.x = a[0]-1, a[1]-1
                    elif f == "m":
                        a = [int(v) for v in p.split(";")] if p else [0]
                        if a[0] == 38: s.f = (a[2]<<16)|(a[3]<<8)|a[4]
                        elif a[0] == 48: s.b = (a[2]<<16)|(a[3]<<8)|a[4]
                    elif f == "J":
                        for yy in range(s.h):
                            for xx in range(s.w): s.ch[yy][xx] = " "; s.bg[yy][xx] = s.b
                    continue
                raise ValueError(repr(data[i:i+10]))
            else:
                if 0 <= s.y < s.h and 0 <= s.x < s.w:
                    s.ch[s.y][s.x] = c; s.fg[s.y][s.x] = s.f; s.bg[s.y][s.x] = s.b
                s.x += 1; i += 1

def pack(a):
    a8 = (np.clip(a*255+0.5,0,255).astype(np.uint32)) & 0xFE
    return (a8[...,0]<<16)|(a8[...,1]<<8)|a8[...,2]

w, h = 120, 30
cv = Canvas(w, h); enc = AnsiEncoder(); vt = VT(w, h)
bad = 0; frames = 0
glyphs = set()
for cls in SCENES:
    sc = cls(None, seed=3); sc.ensure_size(w, h)
    for mode in ("ascii", "blocks"):
        for k in range(40):
            sc.update(1/24); sc.render(cv, mode)
            if k % 13 == 5: cv.scale(0.45)      # simulate overlays toggling
            vt.feed(enc.encode(cv)); frames += 1
            glyphs |= set(cv.ch.ravel().tolist())
            fgp, bgp = pack(cv.fg), pack(cv.bg)
            for y in range(h):
                for x in range(w):
                    c = cv.ch[y, x]
                    if vt.ch[y][x] != c or vt.bg[y][x] != bgp[y, x] or (c != " " and vt.fg[y][x] != fgp[y, x]):
                        bad += 1
print(f"frames {frames}  mismatched cells {bad}")
if bad:
    sys.exit(1)
amb = sorted(g for g in glyphs if unicodedata.east_asian_width(g) in ("A", "W", "F"))
print("ambiguous/wide glyphs used by scenes:", " ".join(amb))
ui_glyphs = set("◆│⏾⟳♪✗◀▌▸•·─╭╮╰╯▏▎▍▋▊▉█←→↑↓●◂▸⏎♥°◤◥◣◢▀▄▐✦")
print("ambiguous/wide glyphs in UI:", " ".join(sorted(g for g in ui_glyphs if unicodedata.east_asian_width(g) in ("A","W","F"))))
