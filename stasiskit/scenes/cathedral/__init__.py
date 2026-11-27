"""CATHEDRAL - a candlelit night vigil: Our Lady praying the rosary, the Divine
Mercy image, and the Blessed Sacrament exposed in a monstrance.

kit.py      shared sprite / lighting toolkit (the subjects and the scene use it)
nave.py     the church: SDF geometry, furnishings, cameras, G-buffer
light.py    candle groups, soft shadows, haze in-scatter, moonlight through glass
subjects.py the subjects as camera-facing cards (with stand-ins as fallback)
fx.py       flames, flicker, incense, dust, glints
rosary.py   the rosary prayed in real time
scene.py    the Cathedral view
mary.py, mercy.py, monstrance.py   the subjects' sprites
"""

from .scene import Cathedral

__all__ = ["Cathedral"]
