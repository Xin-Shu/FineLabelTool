from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from label_io import Box, read_gt_labels

DEFAULT_TILE_OVERLAP = 0.25
TILE_NMS_IOU = 0.50
MIN_TILE_LABEL_AREA_FRACTION = 0.35
MIN_TILE_LABEL_SIZE_PX = 2.0


class DetectorError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    boxes: List[Box]
    source: str
    device: str
    tile_count: int = 0


@dataclass
class TrainingResult:
    model_path: Path
    frame_count: int
    tile_count: int
    box_count: int
    device: str
    updated: bool = True
    skipped_frame_count: int = 0


def _is_yolox_model(base_model: str) -> bool:
    return str(base_model).startswith("yolox:")


def _ensure_supported_backend(base_model: str) -> None:
    if not _is_yolox_model(base_model):
        return
    variant = str(base_model).split(":", 1)[1].upper()
    raise DetectorError(
        f"YOLOX-{variant} is available in the model menu, but this build still uses the "
        "Ultralytics YOLO backend for tiled inference and fine-tuning. YOLOX requires a "
        "separate backend with official YOLOX checkpoints and its training pipeline. "
        "Choose YOLO11 Nano for now, or install/add the YOLOX backend before selecting this model."
    )


def _load_yolo():
    config_dir = Path(tempfile.gettempdir()) / "FineLabelTool" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
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


def training_manifest_path(clip_path: Path) -> Path:
    return detector_root(clip_path) / "training_manifest.json"


def _read_training_manifest(clip_path: Path) -> dict:
    path = training_manifest_path(clip_path)
    if not path.exists():
        return {"frames": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"frames": {}}
    if not isinstance(data, dict):
        return {"frames": {}}
    frames = data.get("frames")
    if not isinstance(frames, dict):
        data["frames"] = {}
    return data


def _write_training_manifest(clip_path: Path, manifest: dict) -> None:
    path = training_manifest_path(clip_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _label_signature(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        digest.update(path.read_bytes())
    except OSError:
        return ""
    return digest.hexdigest()


def _completed_label_signatures(
    clip_path: Path,
    frame_paths: Sequence[Path],
    completed_indices: Sequence[int],
) -> dict:
    signatures = {}
    for idx in sorted(set(completed_indices)):
        if idx < 0 or idx >= len(frame_paths):
            continue
        frame_path = Path(frame_paths[idx])
        gt_path = clip_path / "label_gt" / f"{frame_path.stem}.txt"
        signature = _label_signature(gt_path)
        if signature:
            try:
                mtime_ns = gt_path.stat().st_mtime_ns
            except OSError:
                mtime_ns = 0
            signatures[str(idx)] = {
                "frame": frame_path.name,
                "label": gt_path.name,
                "sha256": signature,
                "mtime_ns": mtime_ns,
            }
    return signatures


def _pending_training_indices(
    clip_path: Path,
    frame_paths: Sequence[Path],
    completed_indices: Sequence[int],
) -> tuple[List[int], dict]:
    current = _completed_label_signatures(clip_path, frame_paths, completed_indices)
    manifest = _read_training_manifest(clip_path)
    trained_frames = manifest.get("frames", {})
    latest_exists = _usable_weight_file(trained_model_path(clip_path))
    if latest_exists and not trained_frames:
        try:
            model_mtime_ns = trained_model_path(clip_path).stat().st_mtime_ns
        except OSError:
            model_mtime_ns = 0
        pending = [
            int(idx)
            for idx, info in current.items()
            if int(info.get("mtime_ns", 0)) > model_mtime_ns
        ]
        if not pending:
            manifest["frames"] = current
            _write_training_manifest(clip_path, manifest)
        return sorted(pending), manifest
    if not latest_exists:
        return sorted(int(idx) for idx in current), manifest

    pending = [
        int(idx)
        for idx, info in current.items()
        if trained_frames.get(idx, {}).get("sha256") != info.get("sha256")
    ]
    return sorted(pending), manifest


def _usable_weight_file(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 100_000
    except OSError:
        return False


def _base_weights_path(clip_path: Path, base_model: str) -> str:
    base_path = Path(base_model)
    if base_path.is_absolute() or base_path.parent != Path(".") or "://" in base_model:
        return base_model

    weights_dir = detector_root(clip_path) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    cached = weights_dir / base_model
    if _usable_weight_file(cached):
        return str(cached)

    cwd_copy = Path.cwd() / base_model
    if _usable_weight_file(cwd_copy):
        shutil.copy2(cwd_copy, cached)
        return str(cached)

    return base_model


def _active_weights(clip_path: Path, base_model: str) -> str:
    trained = trained_model_path(clip_path)
    return str(trained) if _usable_weight_file(trained) else _base_weights_path(clip_path, base_model)


def _load_model(YOLO, clip_path: Path, base_model: str):
    weights = _active_weights(clip_path, base_model)
    base_path = Path(base_model)
    if weights == base_model and base_path.parent == Path(".") and "://" not in base_model:
        weights_dir = detector_root(clip_path) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(base_model)
        cached = weights_dir / base_model
        cwd_copy = Path.cwd() / base_model
        if not _usable_weight_file(cached) and _usable_weight_file(cwd_copy):
            shutil.copy2(cwd_copy, cached)
        return model, str(cached) if _usable_weight_file(cached) else base_model
    return YOLO(weights), weights


def _load_pil_image(path: Path):
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        return Image.open(path).convert("RGB")
    except Exception as exc:
        raise DetectorError(f"Could not read image for tiled detector workflow: {exc}") from exc


def _tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_windows(width: int, height: int, tile_size: int, overlap: float = DEFAULT_TILE_OVERLAP) -> List[Tuple[int, int, int, int]]:
    tile_size = max(32, int(tile_size))
    overlap = max(0.0, min(0.80, float(overlap)))
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = _tile_starts(width, tile_size, stride)
    ys = _tile_starts(height, tile_size, stride)
    windows = []
    for y in ys:
        for x in xs:
            windows.append((x, y, min(width, x + tile_size), min(height, y + tile_size)))
    return windows


def _box_to_xyxy_pixels(box: Box, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (box.x_center - box.width / 2.0) * img_w
    y1 = (box.y_center - box.height / 2.0) * img_h
    x2 = (box.x_center + box.width / 2.0) * img_w
    y2 = (box.y_center + box.height / 2.0) * img_h
    return x1, y1, x2, y2


def _xyxy_to_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    *,
    confidence: float,
    class_id: int,
) -> Box:
    x1 = max(0.0, min(float(img_w - 1), float(x1)))
    y1 = max(0.0, min(float(img_h - 1), float(y1)))
    x2 = max(x1 + 1.0, min(float(img_w), float(x2)))
    y2 = max(y1 + 1.0, min(float(img_h), float(y2)))
    return Box(
        x_center=((x1 + x2) / 2.0) / img_w,
        y_center=((y1 + y2) / 2.0) / img_h,
        width=(x2 - x1) / img_w,
        height=(y2 - y1) / img_h,
        confidence=float(confidence),
        class_id=int(class_id),
        identity=-1,
    )


def _box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = (
        a.x_center - a.width / 2.0,
        a.y_center - a.height / 2.0,
        a.x_center + a.width / 2.0,
        a.y_center + a.height / 2.0,
    )
    bx1, by1, bx2, by2 = (
        b.x_center - b.width / 2.0,
        b.y_center - b.height / 2.0,
        b.x_center + b.width / 2.0,
        b.y_center + b.height / 2.0,
    )
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_boxes(boxes: List[Box], iou_threshold: float = TILE_NMS_IOU) -> List[Box]:
    kept: List[Box] = []
    for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if any(box.class_id == kept_box.class_id and _box_iou(box, kept_box) >= iou_threshold for kept_box in kept):
            continue
        kept.append(box)
    return kept


def suggest_detections(
    image_path: Path,
    clip_path: Path,
    *,
    base_model: str,
    confidence: float,
    image_size: int,
    device: str = "auto",
) -> DetectionResult:
    _ensure_supported_backend(base_model)
    YOLO = _load_yolo()
    selected_device = _select_device(device)
    image = _load_pil_image(image_path)
    img_w, img_h = image.size
    windows = _tile_windows(img_w, img_h, image_size)
    try:
        model, weights = _load_model(YOLO, clip_path, base_model)
        try:
            import numpy as np
        except Exception as exc:
            raise DetectorError(f"NumPy is required for tiled detector inference: {exc}") from exc

        boxes: List[Box] = []
        for left, top, right, bottom in windows:
            tile = image.crop((left, top, right, bottom))
            results = model.predict(
                source=np.asarray(tile),
                conf=float(confidence),
                imgsz=int(image_size),
                device=selected_device,
                verbose=False,
            )
            for result in results:
                if result.boxes is None:
                    continue
                xyxy = result.boxes.xyxy.detach().cpu().tolist()
                confs = result.boxes.conf.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                for coords, conf, cls in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = coords
                    boxes.append(
                        _xyxy_to_box(
                            left + x1,
                            top + y1,
                            left + x2,
                            top + y2,
                            img_w,
                            img_h,
                            confidence=float(conf),
                            class_id=int(cls),
                        )
                    )
    except Exception as exc:
        raise DetectorError(f"Detector inference failed: {exc}") from exc

    boxes = _nms_boxes(boxes)
    source = "trained detector" if _usable_weight_file(trained_model_path(clip_path)) else Path(weights).name
    return DetectionResult(
        boxes=boxes,
        source=f"{source}, tiled {image_size}px/{int(DEFAULT_TILE_OVERLAP * 100)}%",
        device=selected_device,
        tile_count=len(windows),
    )


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
    tile_size: int = 640,
) -> tuple[Path, int, int, int]:
    root = detector_root(clip_path) / "train_data"
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    train_images: List[str] = []
    frame_count = 0
    box_count = 0
    tile_count = 0
    for idx in sorted(set(completed_indices)):
        if idx < 0 or idx >= len(frame_paths):
            continue
        frame_path = Path(frame_paths[idx])
        gt_path = clip_path / "label_gt" / f"{frame_path.stem}.txt"
        boxes = [box for box in read_gt_labels(gt_path) if box.identity >= 0]
        if not boxes:
            continue
        image = _load_pil_image(frame_path)
        img_w, img_h = image.size

        for tile_idx, (left, top, right, bottom) in enumerate(_tile_windows(img_w, img_h, tile_size=tile_size)):
            tile_w = right - left
            tile_h = bottom - top
            label_lines = []
            for box in boxes:
                x1, y1, x2, y2 = _box_to_xyxy_pixels(box, img_w, img_h)
                clip_x1 = max(x1, left)
                clip_y1 = max(y1, top)
                clip_x2 = min(x2, right)
                clip_y2 = min(y2, bottom)
                clip_w = clip_x2 - clip_x1
                clip_h = clip_y2 - clip_y1
                if clip_w < MIN_TILE_LABEL_SIZE_PX or clip_h < MIN_TILE_LABEL_SIZE_PX:
                    continue
                original_area = max(1.0, (x2 - x1) * (y2 - y1))
                clipped_area = clip_w * clip_h
                center_in_tile = left <= (x1 + x2) / 2.0 <= right and top <= (y1 + y2) / 2.0 <= bottom
                if clipped_area / original_area < MIN_TILE_LABEL_AREA_FRACTION and not center_in_tile:
                    continue
                local_x1 = clip_x1 - left
                local_y1 = clip_y1 - top
                local_x2 = clip_x2 - left
                local_y2 = clip_y2 - top
                xc = ((local_x1 + local_x2) / 2.0) / tile_w
                yc = ((local_y1 + local_y2) / 2.0) / tile_h
                w = (local_x2 - local_x1) / tile_w
                h = (local_y2 - local_y1) / tile_h
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            if not label_lines:
                continue

            safe_stem = f"{idx:06d}_{frame_path.stem}_tile{tile_idx:04d}_{left}_{top}"
            image_dst = image_dir / f"{safe_stem}.png"
            label_dst = label_dir / f"{safe_stem}.txt"
            image.crop((left, top, right, bottom)).save(image_dst)
            _write_text(label_dst, "".join(label_lines))
            train_images.append(image_dst.resolve().as_posix())
            tile_count += 1
            box_count += len(label_lines)
        frame_count += 1

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
    return data_yaml, frame_count, tile_count, box_count


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
    _ensure_supported_backend(base_model)
    YOLO = _load_yolo()
    selected_device = _select_device(device)
    root = detector_root(clip_path)
    root.mkdir(parents=True, exist_ok=True)
    pending_indices, manifest = _pending_training_indices(clip_path, frame_paths, completed_indices)
    current_signatures = _completed_label_signatures(clip_path, frame_paths, completed_indices)
    skipped_count = max(0, len(current_signatures) - len(pending_indices))
    latest = trained_model_path(clip_path)
    if not pending_indices and _usable_weight_file(latest):
        manifest["frames"] = current_signatures
        _write_training_manifest(clip_path, manifest)
        return TrainingResult(
            model_path=latest,
            frame_count=0,
            tile_count=0,
            box_count=0,
            device=selected_device,
            updated=False,
            skipped_frame_count=skipped_count,
        )
    data_yaml, frame_count, tile_count, box_count = _prepare_yolo_dataset(
        clip_path,
        frame_paths,
        pending_indices,
        tile_size=int(image_size),
    )
    batch_size = 8 if selected_device not in ("cpu", "mps") else 4

    try:
        model, _ = _load_model(YOLO, clip_path, base_model)
        result = model.train(
            data=str(data_yaml),
            epochs=max(1, int(epochs)),
            imgsz=int(image_size),
            batch=batch_size,
            device=selected_device,
            workers=0,
            project=str(root / "runs"),
            name="latest",
            exist_ok=True,
            pretrained=True,
            single_cls=True,
            val=False,
            plots=False,
            cache=False,
            amp=False,
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
    manifest["frames"] = current_signatures
    _write_training_manifest(clip_path, manifest)
    return TrainingResult(
        model_path=latest,
        frame_count=frame_count,
        tile_count=tile_count,
        box_count=box_count,
        device=selected_device,
        skipped_frame_count=skipped_count,
    )
