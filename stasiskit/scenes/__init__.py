"""Scene registry - the order here is the order of the channels."""

from .viewport import Viewport

SCENES = [Viewport]


def _register(modname, clsname):
    import importlib
    try:
        mod = importlib.import_module(f"{__name__}.{modname}")
        SCENES.append(getattr(mod, clsname))
    except ImportError:
        pass


for _m, _c in [("abyss", "Abyss"), ("reef", "Reef"), ("tideline", "Tideline"), ("poolrooms", "PoolRooms"),
               ("garden", "Garden"), ("cards", "Cards"), ("nails", "Nails"), ("julia", "Julia")]:
    _register(_m, _c)
