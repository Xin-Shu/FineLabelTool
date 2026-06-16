<div align="center">

# Label & Track

**A desktop tool for precise multi-object tracking annotation.**<br>
*<em>Fully implemented by <a href="https://openai.com/codex/"><img src="docs/codex.png" alt="Codex" height="18" /></a> and <a href="https://www.anthropic.com/claude-code"><img src="docs/claudecode-color.svg" alt="Claude Code" height="18" /></a>*<br>
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-41CD52?style=flat-square)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square)](https://github.com)

</div>

---

## Overview

Label & Track is a PyQt5 annotation app for image-sequence MOT datasets. It focuses on fast box editing, identity consistency, frame-to-frame review, and optional detector/tracker assistance.

<p align="center">
  <img src="docs/screenshots/main_interface.png" width="860" alt="Label & Track main annotation interface"/>
  <br/>
  <em>Main interface with canvas, timeline, and annotation sidebar</em>
</p>

## Main Features

- Draw, move, resize, delete, copy, and paste bounding boxes.
- Assign and clear tracking IDs; new IDs start from `1`.
- Select all boxes, undo edits, hide ID text, and customize hotkeys.
- Mark frames completed only after labels are saved and sanity checks pass.
- Review adjacent frames with ghost overlays: previous (`Q`), next (`W`), detections (`D`).
- Show single-ID or all-ID trajectories, including disappeared tracks.
- Use Direct navigation to jump to unassigned boxes or disappeared-ID locations.
- Keep large sequences responsive with async frame loading, async thumbnails, adaptive frame caching, and an optional OpenGL canvas.

## ID And Detection Assist

- **ID Summary** compares the current frame against the previous frame and highlights added, disappeared, stayed, and unassigned IDs.
- **OmniSORT suggestion** proposes IDs for current-frame boxes from the previous completed frame. This is an interactive adaptation of [Xin-Shu/OmniSORT](https://github.com/Xin-Shu/OmniSORT), not a full-sequence batch tracker.
- **Suggest Detections** runs tiled YOLO inference on the current frame, maps tile detections back to full-frame coordinates, and merges overlap duplicates.
- **Accept New** adds only detector suggestions that do not overlap existing boxes.
- **Update Detector** fine-tunes YOLO on tiled crops from completed saved labels. Incremental updates train only on new or changed completed labels after the first detector model is available.
- GPU acceleration is used when PyTorch detects CUDA or Apple MPS; otherwise the app falls back to CPU.

Detector assist is optional and uses [Ultralytics YOLO](https://docs.ultralytics.com/). Review its AGPL-3.0 terms before distributing detector-enabled builds. YOLOX entries are visible as planned backend options, but the current built-in detector workflow uses YOLO11 Nano.

## Install

```bash
git clone https://github.com/Xin-Shu/FineLabelTool.git
cd FineLabelTool

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -r requirements-detector.txt   # optional detector assist
```

## Run

```bash
bash run.sh      # Linux / macOS
run.bat          # Windows

# or directly
python app/main.py
```

## Dataset Layout

Place datasets under `data/`. The app expects one clip per folder:

```text
data/
└── my_dataset/
    ├── frame/
    │   ├── img0001.png
    │   ├── img0002.png
    │   └── ...
    ├── label_det/          # optional detector labels
    │   ├── img0001.txt
    │   └── ...
    └── label_gt/           # app-written GT labels
        ├── img0001.txt
        └── ...
```

Large datasets are intentionally not versioned. The repo ignores `data/`, model weights, detector caches, and generated MOT exports.

## Label Formats

Detection labels in `label_det/*.txt`:

```text
class_id  x_center  y_center  width  height  [confidence]
```

Ground-truth labels in `label_gt/*.txt`:

```text
identity  x_center  y_center  width  height
```

Both app-native formats use normalized `[0, 1]` center coordinates. Only boxes with assigned identities are saved to `label_gt`.

MOT exports use:

```text
frame,id,x1,y1,w,h,conf,-1,-1,-1
```

MOT frame numbers and IDs are 1-based. MOT boxes use pixel upper-left `x1,y1,width,height`.

## CVIP360 Notes

The local CVIP360 conversion follows the dataset README from Mazzola et al., *A dataset of annotated omnidirectional videos for distancing applications* (Journal of Imaging, 2021). Source rows contain repeated pixel `[x,y,w,h]` boxes and no explicit identities, so converted MOT IDs are assigned by box order within each annotation row.

Prepared local CVIP360 copy:

- `17` clips
- `18,488` decoded `3840 x 2160` PNG frames
- `56,074` MOT rows
- `5` GT overlay preview images under `data/cvip360/gt_preview_samples/`

Some CVIP360 videos decode to 1-2 fewer frames than their annotation rows. The converter caps `gt.txt` to the actual decoded `imgXXXX.png` count so MOT rows do not point to missing frames.

## Shortcuts

| Key | Action |
|---|---|
| `Left` / `Right` | Previous / next frame |
| `Ctrl+G` | Jump to frame |
| `B` | Draw box |
| `Delete` | Delete selected boxes |
| `Escape` | Cancel draw mode / deselect |
| `Ctrl+A` | Select all boxes on current frame |
| `Ctrl+Z` | Undo current-frame edit |
| `Ctrl+C` / `Ctrl+V` | Copy / paste boxes across frames |
| `Ctrl+S` | Save current frame |
| `Ctrl+Enter` | Save and attempt completion |
| `F` | Fit image to view |
| `Ctrl++` / `Ctrl+-` / `Ctrl+0` | UI zoom in / out / reset |
| Hold `Q` / `W` / `D` | Overlay previous / next / detector boxes |
| Middle mouse | Pan while zoomed |
| Scroll wheel | Canvas zoom |

Shortcuts can be changed from the in-app Hotkeys dialog.

## Project Layout

```text
FineLabelTool/
├── app/
│   ├── main.py
│   ├── main_window.py
│   ├── canvas.py
│   ├── timeline.py
│   ├── label_io.py
│   └── algo/
│       ├── detector.py
│       └── omnisort.py
├── data/
├── requirements.txt
├── requirements-detector.txt
├── run.sh
└── run.bat
```

## License

MIT License. See [`LICENSE`](LICENSE) for details.
