"""The rosary, prayed in real time on Our Lady's beads.

`Rosary` walks the prayers bead by bead, in the order the USCCB gives: the Sign
of the Cross and the Apostles' Creed on the crucifix, an Our Father, three Hail
Marys and a Glory Be on the pendant, then five decades (the mystery is
announced, an Our Father on the large bead, ten Hail Marys, the Glory Be and the
Fatima prayer), then on the centre medal the Hail, Holy Queen, its versicle
("Pray for us, O holy Mother of God...") and the concluding prayer ("Let us
pray: O God, whose Only Begotten Son..."), and the Sign of the Cross again on
the crucifix; after a short silence it begins again, about twenty minutes in
all.  The mysteries are those of the day (Monday and Saturday Joyful, Tuesday
and Friday Sorrowful, Wednesday and Sunday Glorious, Thursday Luminous; the
Sundays of Advent take the Joyful and the Sundays of Lent the Sorrowful
Mysteries), chosen when each rosary begins.

The bead order comes from the statue sprite: `anchors["bead_kind"]` (0 crucifix,
1 large, 2 small, 3 medal) in prayer order.  The scene asks `glint()` where the
fingers are (a bead index, or a point on the chain between two beads), and
`update()` returns the index of each new bead they reach (the scene sounds a
bead click for it).
"""

import datetime
import re
import time
from dataclasses import dataclass

MYSTERIES = {
    "Joyful": ("The Annunciation", "The Visitation", "The Nativity", "The Presentation in the Temple",
               "The Finding in the Temple"),
    "Sorrowful": ("The Agony in the Garden", "The Scourging at the Pillar", "The Crowning with Thorns",
                  "The Carrying of the Cross", "The Crucifixion"),
    "Glorious": ("The Resurrection", "The Ascension", "The Descent of the Holy Spirit", "The Assumption",
                 "The Coronation of Mary"),
    "Luminous": ("The Baptism of the Lord", "The Wedding at Cana", "The Proclamation of the Kingdom",
                 "The Transfiguration", "The Institution of the Eucharist"),
}
# Monday = 0 ... Sunday = 6
BY_WEEKDAY = ("Joyful", "Sorrowful", "Glorious", "Luminous", "Sorrowful", "Joyful", "Glorious")
ROMAN = ("I", "II", "III", "IV", "V")
ORDINAL = ("First", "Second", "Third", "Fourth", "Fifth")

# seconds for each prayer, most at the unhurried pace of the Hail Mary (about 2.8 words a second);
# the Hail, Holy Queen and the closing prayer a little slower
DUR = dict(sign=6.0, creed=40.0, our_father=20.0, hail_mary=15.0, glory=11.0, announce=6.0, glory_fatima=21.0,
           queen=35.0, versicle=7.0, closing=25.0, silence=12.0)

KIND_NAMES = {0: "cross", 1: "large", 2: "small", 3: "medal"}

# shorter forms of the longer prayer captions, for a narrow screen (see short_forms)
_SHORT = {
    "The Sign of the Cross": ("Sign of the Cross",),
    "The Apostles' Creed": ("The Creed",),
    "Glory Be · O my Jesus": ("Glory Be",),
    "Pray for us, O holy Mother of God": ("Pray for us",),
}


def short_forms(prayer):
    """Shorter forms of a prayer caption for a narrow screen, longest first (maybe none):
    "Hail Mary 4 of 10" -> "Hail Mary 4/10", "4/10"; "The Second Joyful Mystery" ->
    "Second Mystery"; "Pray for us, O holy Mother of God" -> "Pray for us"..."""
    m = re.fullmatch(r"Hail Mary (\d+) of (\d+)", prayer)
    if m:
        return (f"Hail Mary {m[1]}/{m[2]}", f"{m[1]}/{m[2]}")
    m = re.fullmatch(r"The (\w+) \w+ Mystery", prayer)
    if m:
        return (f"{m[1]} Mystery",)
    return _SHORT.get(prayer, ())


@dataclass
class Step:
    prayer: str            # what is being prayed ("Hail Mary 4 of 10", "Our Father", ...)
    dur: float
    bead: int = -1         # bead index being held, or -1 between beads
    gap: tuple = None      # (bead a, bead b): fingers on the chain between them
    decade: int = 0        # 1..5 inside a decade, 0 before, 6 after
    kind: str = ""         # "announce", "silence", ...


def mysteries_for(weekday):
    return BY_WEEKDAY[int(weekday) % 7]


def easter(year):
    """Gregorian Easter Sunday of a year (the Meeus/Jones/Butcher computus)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return datetime.date(year, month, day + 1)


def mysteries_on(date):
    """The mysteries for a calendar date: those of its weekday, except that the Sundays of
    Advent (from the Sunday between 27 November and 3 December to Christmas Eve) take the
    Joyful and the Sundays of Lent (the First Sunday of Lent to Palm Sunday) the Sorrowful."""
    wd = date.weekday()
    if wd == 6:
        if (date.month, date.day) >= (11, 27) and (date.month, date.day) <= (12, 24):
            return "Joyful"
        e = easter(date.year)
        if e - datetime.timedelta(days=42) <= date <= e - datetime.timedelta(days=7):
            return "Sorrowful"
    return mysteries_for(wd)


def build_steps(kinds):
    """Prayer steps for a rosary whose beads (in prayer order) have these kinds.
    Tolerates odd bead counts: missing beads fall back to the nearest valid one."""
    kinds = [int(k) for k in kinds]
    n = len(kinds)
    cross = next((i for i, k in enumerate(kinds) if k == 0), 0)
    large = [i for i, k in enumerate(kinds) if k == 1] or [min(1, n - 1)]
    small = [i for i, k in enumerate(kinds) if k == 2] or [min(2, n - 1)]
    medal = next((i for i, k in enumerate(kinds) if k == 3), large[min(1, len(large) - 1)])

    def L(i):
        return large[min(i, len(large) - 1)]

    def S(i):
        return small[min(i, len(small) - 1)]

    st = [Step("The Sign of the Cross", DUR["sign"], bead=cross),
          Step("The Apostles' Creed", DUR["creed"], bead=cross),
          Step("Our Father", DUR["our_father"], bead=L(0))]
    for j in range(3):
        st.append(Step(f"Hail Mary {j + 1} of 3", DUR["hail_mary"], bead=S(j)))
    st.append(Step("Glory Be", DUR["glory"], gap=(S(2), L(1))))
    for d in range(5):
        lb = L(d + 1)
        st.append(Step("", DUR["announce"], bead=lb, decade=d + 1, kind="announce"))
        st.append(Step("Our Father", DUR["our_father"], bead=lb, decade=d + 1))
        for j in range(10):
            st.append(Step(f"Hail Mary {j + 1} of 10", DUR["hail_mary"], bead=S(3 + 10 * d + j), decade=d + 1))
        nxt = L(d + 2) if d < 4 else medal
        st.append(Step("Glory Be · O my Jesus", DUR["glory_fatima"], gap=(S(3 + 10 * d + 9), nxt), decade=d + 1))
    st.append(Step("Hail, Holy Queen", DUR["queen"], bead=medal, decade=6))
    st.append(Step("Pray for us, O holy Mother of God", DUR["versicle"], bead=medal, decade=6))
    st.append(Step("Let us pray", DUR["closing"], bead=medal, decade=6))
    # back on the crucifix, where the next rosary begins (one click, not two)
    st.append(Step("The Sign of the Cross", DUR["sign"], bead=cross, decade=6))
    st.append(Step("", DUR["silence"], bead=cross, decade=6, kind="silence"))
    return st


class Rosary:
    """Real-time rosary.  update(dt) returns the bead index reached this step (or None)."""

    def __init__(self, kinds=None, weekday=None, clock=time.localtime):
        self.kinds = list(kinds) if kinds is not None else default_kinds()
        self.steps = build_steps(self.kinds)
        self.total = sum(s.dur for s in self.steps)
        self.weekday = weekday
        self.clock = clock
        self.t = 0.0
        self.i = 0
        self.set_name = self._today()

    def _today(self):
        if self.weekday is not None:                 # a fixed weekday (tests, previews): no seasons
            return mysteries_for(self.weekday)
        tm = self.clock()
        try:
            return mysteries_on(datetime.date(tm.tm_year, tm.tm_mon, tm.tm_mday))
        except (AttributeError, TypeError, ValueError):
            return mysteries_for(tm.tm_wday)

    def set_kinds(self, kinds):
        """Re-map onto another statue's beads, keeping our place in the prayers."""
        kinds = list(kinds)
        if kinds == self.kinds:
            return
        frac = self.t
        self.kinds = kinds
        self.steps = build_steps(kinds)
        self.total = sum(s.dur for s in self.steps)
        self.t, self.i = 0.0, 0
        self.update(frac % self.total)

    @property
    def step(self):
        return self.steps[self.i]

    def elapsed_in_step(self):
        return self.t - sum(s.dur for s in self.steps[:self.i])

    def update(self, dt):
        """Advance; returns the index of a bead the fingers just arrived at, else None."""
        before = self.held()
        self.t += max(0.0, dt)
        start = sum(s.dur for s in self.steps[:self.i])
        while self.t >= start + self.steps[self.i].dur:
            start += self.steps[self.i].dur
            self.i += 1
            if self.i >= len(self.steps):            # the rosary begins again, with today's mysteries
                self.i = 0
                self.t -= self.total
                start = 0.0
                self.set_name = self._today()
        now = self.held()
        return now if (now is not None and now != before) else None

    def held(self):
        s = self.steps[self.i]
        return s.bead if s.bead >= 0 else None

    def glint(self):
        """Where the fingers are: (bead a, bead b, fraction) - a == b on a bead."""
        s = self.steps[self.i]
        if s.gap is not None:
            return s.gap[0], s.gap[1], 0.5
        return s.bead, s.bead, 0.0

    def kind_of(self, bead):
        return KIND_NAMES.get(self.kinds[bead] if 0 <= bead < len(self.kinds) else 2, "small")

    def caption_lines(self, set_name=None):
        """Every text each caption line can show in this rosary, with today's mysteries or
        those of `set_name`: (titles, mysteries, prayers), as caption() words them."""
        name = set_name or self.set_name
        titles = [f"The {name} Mysteries"]
        mysteries = ["The opening prayers", "In conclusion"]
        mysteries += [f"{ROMAN[i]} · {m}" for i, m in enumerate(MYSTERIES[name])]
        prayers = [f"The {o} {name} Mystery" for o in ORDINAL] + ["· · ·"]
        prayers += [s.prayer for s in self.steps if s.prayer]
        return titles, mysteries, prayers

    def caption_width(self):
        """The longest caption line this rosary shows (with today's mysteries)."""
        return max(len(t) for group in self.caption_lines() for t in group)

    def caption(self):
        """Three short lines: the set of mysteries, the mystery (or part), the prayer."""
        s = self.steps[self.i]
        title = f"The {self.set_name} Mysteries"
        if 1 <= s.decade <= 5:
            mystery = f"{ROMAN[s.decade - 1]} · {MYSTERIES[self.set_name][s.decade - 1]}"
        elif s.decade == 0:
            mystery = "The opening prayers"
        else:
            mystery = "In conclusion"
        if s.kind == "announce":
            prayer = f"The {ORDINAL[s.decade - 1]} {self.set_name} Mystery"
        elif s.kind == "silence":
            prayer = "· · ·"
        else:
            prayer = s.prayer
        return title, mystery, prayer


def default_kinds():
    """Bead kinds of a standard five-decade rosary in prayer order, as Our Lady's
    statue lists them (61 anchors): crucifix, large, 3 small, large, then five
    decades of 10 small beads with a large bead between each, and the medal."""
    k = [0, 1, 2, 2, 2, 1]
    for d in range(5):
        k += [2] * 10
        if d < 4:
            k.append(1)
    return k + [3]
