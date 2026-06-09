<div align="center">

# Label & Track

**A desktop tool for annotating multi-object tracking datasets — fast, keyboard-driven, and built for precision.**<br>
*<em>Fully implemented by <a href="https://openai.com/codex/"><img src="docs/codex.png" alt="Codex" height="18" /></a> and <a href="https://www.anthropic.com/claude-code"><img src="docs/claudecode-color.svg" alt="Claude Code" height="18" /></a>*<br>
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-41CD52?style=flat-square)](https://pypi.org/project/PyQt5/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square)](https://github.com)

</div>

---

## Why This Exists

Multi-object tracking annotation needs more than drawing boxes. It also needs **consistent identities across long sequences**, quick frame-to-frame comparison, and a workflow that does not slow the annotator down.

**Label & Track** is built for that job. It focuses on fast box editing, identity assignment, frame overlays, and lightweight tracking-based ID suggestions so users can spend time judging labels instead of managing the tool.

---

## Screenshots

<p align="center">
  <img src="docs/screenshots/main_interface.png" width="860" alt="Label & Track — main annotation interface showing image canvas, timeline, and sidebar"/>
  <br/>
  <em>Main interface: canvas with bounding boxes, frame timeline, and annotation sidebar</em>
</p>

<p align="center">
  <img src="docs/screenshots/trajectory_overlay.png" width="860" alt="Trajectory overlay showing the path of a tracked identity across previous frames"/>
  <br/>
  <em>Trajectory overlay — review an object's full history before confirming a tracking ID</em>
</p>

---

## Features

### Annotation
- **Draw, move, resize, copy, and paste boxes** directly on the canvas
- **Assign tracking IDs** with keyboard-first controls; new IDs start from 1
- **Assign the next unused ID** with the `New ID` button
- **Hide ID Numbers** keeps box outlines visible while hiding in-frame identity text
- **Undo recent edits** on each frame with `Ctrl+Z`

### Smart ID Assignment
- **Same-frame conflict detection** shows when an ID is already in use
- **Past trajectory review** helps check whether an ID belongs to the same object
- **OmniSORT-based ID suggestion** propagates IDs from the previous completed frame
- **ID Summary against previous frame** shows added, disappeared, stayed, and unassigned IDs with sanity checks

**About OmniSORT.** Label & Track uses a lightweight, interactive adaptation of OmniSORT for annotation-time ID suggestion. Instead of running a full tracker across the entire sequence, it matches the current frame's unassigned boxes against identities from the previous completed frame and proposes one-to-one assignments to speed up manual labeling. The original OmniSORT tracker is also publicly available at [Xin-Shu/OmniSORT](https://github.com/Xin-Shu/OmniSORT).

### Detection Assist
- **Suggest Detections** runs tiled YOLO inference on the current frame without changing labels
- **Accept New** converts non-overlapping detector suggestions into editable, unassigned boxes
- **Update Detector** fine-tunes the detector from tiled crops generated from completed saved labels
- **Incremental detector updates** train only on new or changed completed labels after the first trained model is available
- **Auto-update after completed save** can update the detector in the background as annotation grows
- **GPU acceleration** is used automatically when PyTorch detects CUDA or Apple MPS; CPU is used otherwise

The detector workflow is tile-first for high-resolution frames. The app splits each frame into overlapping tiles, runs YOLO on the tiles, maps detections back into full-frame coordinates, then merges duplicate boxes from overlap regions. Fine-tuning uses the same tiled crop layout so training and inference see similar object scale.

Detector assist is optional and uses [Ultralytics YOLO](https://docs.ultralytics.com/). For open-source use, review the Ultralytics AGPL-3.0 licensing terms and keep the repository license compatible with the detector dependency.

YOLOX model sizes are shown in the detector selector as planned backend options. They require a separate YOLOX backend and checkpoints before they can run; use the YOLO11 option for the current built-in detector workflow.

### Navigation & Overlays
- **Frame timeline** with async thumbnail loading and completion indicators
- **Jump to any frame** directly by number (Ctrl+G)
- **Direct navigation** jumps to unassigned boxes or the last location of disappeared IDs
- **Hold Q** to ghost the previous frame's boxes over the current view
- **Hold W** to ghost the next frame's boxes over the current view
- **Hold D** to overlay the raw detector output for the current frame
- **Trajectory overlay** shown automatically when an identity is fully tracked up to the current frame
- **Show All Trajectories** overlays every ID history seen up to the current frame, including disappeared IDs
- **Minimap** appears when zoomed in

### Workflow
- **Searchable dataset picker** with screen-aware sizing
- **Async frame loading** keeps the UI responsive on large image sequences
- **Adaptive frame cache** limits memory use for very large PNG datasets
- **OpenGL canvas acceleration** is enabled automatically when Qt can create an OpenGL viewport
- **Mark frame as completed** with `Ctrl+Enter`; labels are saved immediately
- **Sanity-gated completion** auto-marks a saved frame completed only when all checks pass
- **Dataset switching** with unsaved-change protection
- **Auto-skips to first unlabelled frame** when loading a dataset
- **Customizable hotkeys** from the in-app Hotkeys dialog
- **HUD warning badges** on the canvas for all conflict and overlay states
- **Task badge** reports active frame, detector, and training work

---

## Installation

**Requirements:** Python 3.8 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/Xin-Shu/FineLabelTool.git
cd FineLabelTool

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# Optional: enable Detection Assist
pip install -r requirements-detector.txt
```

---

## Running the App

```bash
# Linux / macOS
bash run.sh

# Windows
run.bat

# Or directly
python app/main.py
```

---

## Dataset Structure

Place your datasets inside the `data/` folder. Each dataset is a subfolder with the following layout:

```
data/
└── my_dataset/
    ├── frame/               # Input images — one PNG per frame
    │   ├── img0001.png
    │   ├── img0002.png
    │   └── ...
    ├── label_det/           # (Optional) Detector output — one .txt per frame
    │   ├── img0001.txt
    │   └── ...
    └── label_gt/            # Ground-truth output — created by the app on save
        ├── img0001.txt
        └── ...
```

Large datasets are intentionally kept out of Git. The repository ignores `data/`, model weights, detector caches, and generated MOT exports.

### Label File Formats

**Detection labels** (`label_det/*.txt`) are read-only input from an external detector:
```
class_id  x_center  y_center  width  height  [confidence]
```
Coordinates are normalised to `[0, 1]`. Confidence is optional.

**Ground-truth labels** (`label_gt/*.txt`) are written by the app on save:
```
identity  x_center  y_center  width  height
```
Only boxes with an assigned identity are saved. Unassigned boxes are discarded at save time.

**MOT exports** (`gt.txt`) use the standard 10-column MOTChallenge-style layout:
```
frame,id,x1,y1,w,h,conf,-1,-1,-1
```
Frame numbers and IDs are 1-based. MOT box coordinates are pixel-based upper-left `x1,y1,width,height`, while app-native `label_gt` files stay normalised.

### Prepared Dataset Notes

The local CVIP360 conversion follows the dataset README from Mazzola et al., *A dataset of annotated omnidirectional videos for distancing applications* (Journal of Imaging, 2021). CVIP360 annotations are repeated pixel `[x,y,w,h]` boxes per video frame and do not include explicit identity IDs, so the conversion assigns MOT IDs by box order within each annotation row.

For the prepared local CVIP360 copy:
- `17` clips are organised as `clip/frame/`, `clip/gt.txt`, and `clip/cvip360_meta.json`
- `18,488` decoded `3840 x 2160` PNG frames are available
- `56,074` MOT rows are written across all `gt.txt` files
- `5` full-resolution GT overlay preview PNGs are saved under `data/cvip360/gt_preview_samples/`

Some CVIP360 videos decode to 1-2 fewer frames than their annotation text rows. The converter caps `gt.txt` to the actual decoded `imgXXXX.png` count so MOT rows never point to missing frames.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `←` / `→` | Previous / Next frame |
| `Ctrl+G` | Jump to frame by number |
| `B` | Toggle Draw Box mode |
| `Delete` | Delete selected box |
| `Escape` | Cancel draw mode / deselect box |
| `Ctrl+Z` | Undo last change on current frame |
| `Ctrl+C` | Copy selected box |
| `Ctrl+V` | Paste copied box (works across frames) |
| `Ctrl+S` | Save current frame |
| `Ctrl+Enter` | Toggle frame completion (saves automatically) |
| `F` | Fit image to viewport |
| `Ctrl++` / `Ctrl+-` | Zoom UI in / out |
| `Ctrl+0` | Reset UI zoom |
| `Hold Q` | Overlay previous frame's boxes |
| `Hold W` | Overlay next frame's boxes |
| `Hold D` | Overlay detection boxes |
| `Middle mouse` | Pan the canvas |
| `Scroll wheel` | Zoom canvas in / out |

---

## Annotation Workflow

A typical session looks like this:

1. **Open a dataset** — click *Dataset…* or *Open Dataset…* in the sidebar. The app opens at the first unlabelled frame.
2. **Review detections** — hold `D` to see the detector's raw proposals as a ghost overlay.
3. **Assign IDs** — click a box, enter the identity in the sidebar, and press `Enter` or click *Assign ID*.
   - If the same ID already exists in the frame, the conflict is highlighted.
   - If the ID was previously used by a different object, the past trajectory is shown before confirmation.
4. **Find unfinished work** — use *Direct* beside *Unassigned* to jump to unlabelled boxes, or beside *Disappeared* to inspect where a missing ID was last seen in the previous frame.
5. **Add missing boxes** — press `B` (or click *Draw Box*), drag a rectangle on the canvas, then assign an ID.
6. **Use ID suggestions** — click *Suggest IDs* in the *ID Suggestions* panel to auto-propagate IDs from the previous completed frame using the app's interactive OmniSORT-based matcher.
7. **Use detector assist** — click *Suggest Detections* to preview tiled detector boxes mapped back onto the full frame. Click *Accept New* to add only suggestions that do not overlap existing boxes.
8. **Improve the detector** — leave *Auto-update after completed save* enabled or click *Update Detector* to fine-tune YOLO on tiled crops from new or changed completed labels.
9. **Compare with adjacent frames** — hold `Q` or `W` to ghost the previous or next frame's boxes over the current view. This helps verify spatial consistency without navigating away.
10. **Save or complete** — press `Ctrl+S` to save. If the frame passes sanity checks, it is automatically marked completed. Pressing `Ctrl+Enter` also saves first, then completes only when sanity checks pass.

---

## Project Structure

```
FineLabelTool/
├── app/
│   ├── main.py            # Entry point
│   ├── main_window.py     # Application logic and UI orchestration
│   ├── canvas.py          # Interactive image canvas (QGraphicsView)
│   ├── timeline.py        # Thumbnail timeline with async loading
│   ├── label_io.py        # Label file reading and writing
│   ├── colors.py          # Per-identity colour palette
│   └── algo/
│       ├── detector.py    # Optional YOLO detector suggestion and fine-tuning
│       └── omnisort.py    # GIoU + omni-distance ID suggestion
├── data/                  # Datasets go here
├── requirements.txt
├── requirements-detector.txt
├── run.sh                 # Linux / macOS launcher
└── run.bat                # Windows launcher
```

---

## Contributing

Issues and pull requests are welcome. Please keep the keyboard-first workflow in mind when adding features.

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
