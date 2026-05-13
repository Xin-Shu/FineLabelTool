from PyQt5.QtGui import QColor

_PALETTE = [
    (255,  59,  48),   # 0  red
    ( 52, 199,  89),   # 1  green
    (  0, 122, 255),   # 2  blue
    (255, 149,   0),   # 3  orange
    (175,  82, 222),   # 4  purple
    ( 90, 200, 250),   # 5  sky
    (255, 204,   0),   # 6  yellow
    ( 88,  86, 214),   # 7  indigo
    (  0, 199, 190),   # 8  teal
    (255,  45,  85),   # 9  pink
    (162, 132,  94),   # 10 brown
    ( 48, 209,  88),   # 11 mint
    (100, 210, 255),   # 12 light-blue
    (255, 159,  10),   # 13 amber
    (255,  55,  95),   # 14 crimson
    (142, 142, 147),   # 15 gray
    (215, 100,  50),   # 16 burnt-orange
    ( 50, 173, 230),   # 17 process-blue
    (220,  20,  60),   # 18 crimson-red
    ( 30, 144, 255),   # 19 dodger-blue
]


def get_color(identity_id: int) -> QColor:
    if identity_id < 0:
        return QColor(160, 160, 160)
    r, g, b = _PALETTE[identity_id % len(_PALETTE)]
    return QColor(r, g, b)
