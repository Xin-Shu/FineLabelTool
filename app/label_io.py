from dataclasses import dataclass
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
