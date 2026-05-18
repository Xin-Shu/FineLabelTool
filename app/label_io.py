from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import List


@dataclass
class Box:
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 0      # detection class (from det label first column)
    identity: int = -1     # user-assigned tracking ID (-1 = unassigned)


def _round_pixel(value: float) -> int:
    return int(floor(value + 0.5))


def snap_box_to_pixel_grid(box: Box, img_w: int, img_h: int) -> None:
    """Mutate a normalized box so its pixel x1/y1/x2/y2 are integers."""
    if img_w <= 0 or img_h <= 0:
        return

    left = _round_pixel((box.x_center - box.width / 2) * img_w)
    top = _round_pixel((box.y_center - box.height / 2) * img_h)
    right = _round_pixel((box.x_center + box.width / 2) * img_w)
    bottom = _round_pixel((box.y_center + box.height / 2) * img_h)

    left = max(0, min(img_w - 1, left))
    top = max(0, min(img_h - 1, top))
    right = max(left + 1, min(img_w, right))
    bottom = max(top + 1, min(img_h, bottom))

    box.x_center = ((left + right) / 2) / img_w
    box.y_center = ((top + bottom) / 2) / img_h
    box.width = (right - left) / img_w
    box.height = (bottom - top) / img_h


def snap_boxes_to_pixel_grid(boxes: List[Box], img_w: int, img_h: int) -> None:
    for box in boxes:
        snap_box_to_pixel_grid(box, img_w, img_h)


def read_det_labels(path: Path) -> List[Box]:
    """Read detection labels: class_id x y w h [conf]"""
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    vals = [float(p) for p in parts]
                    boxes.append(Box(
                        class_id=int(vals[0]),
                        x_center=vals[1],
                        y_center=vals[2],
                        width=vals[3],
                        height=vals[4],
                        confidence=vals[5] if len(vals) > 5 else 1.0,
                    ))
                except (ValueError, IndexError):
                    continue
    except OSError:
        pass
    return boxes


def read_gt_labels(path: Path) -> List[Box]:
    """Read ground-truth labels: identity x y w h"""
    boxes: List[Box] = []
    if not path.exists():
        return boxes
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    vals = [float(p) for p in parts]
                    boxes.append(Box(
                        identity=int(vals[0]),
                        x_center=vals[1],
                        y_center=vals[2],
                        width=vals[3],
                        height=vals[4],
                    ))
                except (ValueError, IndexError):
                    continue
    except OSError:
        pass
    return boxes


def write_gt_labels(path: Path, boxes: List[Box]) -> bool:
    """Write GT labels: identity x y w h (only boxes with assigned identity).
    Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for b in boxes:
                if b.identity >= 0:
                    f.write(
                        f"{b.identity} {b.x_center:.6f} {b.y_center:.6f} "
                        f"{b.width:.6f} {b.height:.6f}\n"
                    )
        return True
    except OSError:
        return False
