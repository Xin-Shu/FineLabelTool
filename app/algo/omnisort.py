from __future__ import annotations

from typing import List, Sequence, Tuple


def _xyxy(box) -> Tuple[float, float, float, float]:
    return (
        box.x_center - box.width / 2,
        box.y_center - box.height / 2,
        box.x_center + box.width / 2,
        box.y_center + box.height / 2,
    )


def _giou_distance(a, b) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(a)
    bx1, by1, bx2, by2 = _xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0

    cx1 = min(ax1, bx1)
    cy1 = min(ay1, by1)
    cx2 = max(ax2, bx2)
    cy2 = max(ay2, by2)
    area_enclose = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    giou = iou
    if area_enclose > 0:
        giou -= (area_enclose - union) / area_enclose
    return 1.0 - ((giou + 1.0) / 2.0)


def _omni_center_distance(a, b) -> float:
    dx = abs(a.x_center - b.x_center)
    dy = abs(a.y_center - b.y_center)
    dx = min(dx, 1.0 - dx)
    dy = min(dy, 1.0 - dy)
    return (dx * dx + dy * dy) ** 0.5 / (2.0 ** 0.5)


def suggest_ids_from_previous(
    previous_boxes: Sequence,
    current_boxes: Sequence,
    threshold: float = 0.30,
) -> List[Tuple[int, int]]:
    """Return ``(current_index, previous_index)`` matches using OmniSORT-style costs.

    The app uses this for one-step ID suggestions, not full sequence tracking. It
    combines GIoU distance with omnidirectional center distance, then applies a
    greedy one-to-one assignment. Inputs are normalized boxes in app format.
    """
    candidates = []
    for curr_idx, curr in enumerate(current_boxes):
        for prev_idx, prev in enumerate(previous_boxes):
            cost = 0.5 * _giou_distance(curr, prev) + 0.5 * _omni_center_distance(curr, prev)
            if cost < threshold:
                candidates.append((cost, curr_idx, prev_idx))

    candidates.sort()
    used_current = set()
    used_previous = set()
    matches = []
    for _, curr_idx, prev_idx in candidates:
        if curr_idx in used_current or prev_idx in used_previous:
            continue
        matches.append((curr_idx, prev_idx))
        used_current.add(curr_idx)
        used_previous.add(prev_idx)
    return matches
