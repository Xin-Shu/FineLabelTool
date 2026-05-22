from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from label_io import Box, read_gt_labels


class DetectorError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    boxes: List[Box]
    source: str
    device: str


@dataclass
class TrainingResult:
    model_path: Path
    frame_count: int
    box_count: int
    device: str


def _load_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise DetectorError(
            "Ultralytics YOLO is not available or one of its dependencies is missing. "
            f"Import error: {exc}. Install it in this environment with "
            "`pip install ultralytics`, then restart the app."
        ) from exc
    return YOLO


def _select_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def detector_root(clip_path: Path) -> Path:
    return clip_path / ".detector"


def trained_model_path(clip_path: Path) -> Path:
    return detector_root(clip_path) / "latest.pt"


def _active_weights(clip_path: Path, base_model: str) -> str:
    trained = trained_model_path(clip_path)
    return str(trained) if trained.exists() else base_model


def suggest_detections(
    image_path: Path,
    clip_path: Path,
    *,
    base_model: str,
    confidence: float,
    image_size: int,
    device: str = "auto",
) -> DetectionResult:
    YOLO = _load_yolo()
    selected_device = _select_device(device)
    weights = _active_weights(clip_path, base_model)
    try:
        model = YOLO(weights)
        results = model.predict(
            source=str(image_path),
            conf=float(confidence),
            imgsz=int(image_size),
            device=selected_device,
            verbose=False,
        )
    except Exception as exc:
        raise DetectorError(f"Detector inference failed: {exc}") from exc

    boxes: List[Box] = []
    for result in results:
        if result.boxes is None:
            continue
        img_h, img_w = result.orig_shape[:2]
        xyxy = result.boxes.xyxy.detach().cpu().tolist()
        confs = result.boxes.conf.detach().cpu().tolist()
        classes = result.boxes.cls.detach().cpu().tolist()
        for coords, conf, cls in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = coords
            x1 = max(0.0, min(float(img_w - 1), float(x1)))
            y1 = max(0.0, min(float(img_h - 1), float(y1)))
            x2 = max(x1 + 1.0, min(float(img_w), float(x2)))
            y2 = max(y1 + 1.0, min(float(img_h), float(y2)))
            boxes.append(
                Box(
                    x_center=((x1 + x2) / 2.0) / img_w,
                    y_center=((y1 + y2) / 2.0) / img_h,
                    width=(x2 - x1) / img_w,
                    height=(y2 - y1) / img_h,
                    confidence=float(conf),
                    class_id=int(cls),
                    identity=-1,
                )
            )

    source = "trained detector" if Path(weights).exists() else base_model
    return DetectionResult(boxes=boxes, source=source, device=selected_device)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_yolo_dataset(
    clip_path: Path,
    frame_paths: Sequence[Path],
    completed_indices: Sequence[int],
) -> tuple[Path, int, int]:
    root = detector_root(clip_path) / "train_data"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    train_images: List[str] = []
    frame_count = 0
    box_count = 0
    for idx in sorted(set(completed_indices)):
        if idx < 0 or idx >= len(frame_paths):
            continue
        frame_path = Path(frame_paths[idx])
        gt_path = clip_path / "label_gt" / f"{frame_path.stem}.txt"
        boxes = [box for box in read_gt_labels(gt_path) if box.identity >= 0]
        if not boxes:
            continue

        safe_stem = f"{idx:06d}_{frame_path.stem}"
        image_dst = image_dir / f"{safe_stem}{frame_path.suffix.lower()}"
        label_dst = label_dir / f"{safe_stem}.txt"
        _link_or_copy(frame_path, image_dst)
        _write_text(
            label_dst,
            "".join(
                f"0 {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n"
                for box in boxes
            ),
        )
        train_images.append(image_dst.resolve().as_posix())
        frame_count += 1
        box_count += len(boxes)

    if not train_images:
        raise DetectorError("No completed frames with assigned labels are available for detector training.")

    train_txt = root / "train.txt"
    _write_text(train_txt, "\n".join(train_images) + "\n")
    data_yaml = root / "data.yaml"
    _write_text(
        data_yaml,
        "\n".join(
            [
                f"path: {root.resolve().as_posix()}",
                f"train: {train_txt.resolve().as_posix()}",
                f"val: {train_txt.resolve().as_posix()}",
                "nc: 1",
                "names:",
                "  0: object",
                "",
            ]
        ),
    )
    return data_yaml, frame_count, box_count


def train_detector(
    clip_path: Path,
    frame_paths: Sequence[Path],
    completed_indices: Sequence[int],
    *,
    base_model: str,
    epochs: int,
    image_size: int,
    device: str = "auto",
) -> TrainingResult:
    YOLO = _load_yolo()
    selected_device = _select_device(device)
    root = detector_root(clip_path)
    root.mkdir(parents=True, exist_ok=True)
    data_yaml, frame_count, box_count = _prepare_yolo_dataset(clip_path, frame_paths, completed_indices)
    weights = _active_weights(clip_path, base_model)

    try:
        model = YOLO(weights)
        result = model.train(
            data=str(data_yaml),
            epochs=max(1, int(epochs)),
            imgsz=int(image_size),
            batch=-1,
            device=selected_device,
            workers=max(1, min(4, os.cpu_count() or 1)),
            project=str(root / "runs"),
            name="latest",
            exist_ok=True,
            pretrained=True,
            single_cls=True,
            patience=max(2, min(5, int(epochs))),
            verbose=False,
        )
    except Exception as exc:
        raise DetectorError(f"Detector training failed: {exc}") from exc

    save_dir = Path(getattr(result, "save_dir", root / "runs" / "latest"))
    if not save_dir.exists() and getattr(model, "trainer", None) is not None:
        save_dir = Path(getattr(model.trainer, "save_dir", save_dir))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    source = best if best.exists() else last
    if not source.exists():
        raise DetectorError("Detector training completed, but no model weights were produced.")
    latest = trained_model_path(clip_path)
    shutil.copy2(source, latest)
    return TrainingResult(
        model_path=latest,
        frame_count=frame_count,
        box_count=box_count,
        device=selected_device,
    )
