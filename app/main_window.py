from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QApplication, QComboBox, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QShortcut, QSplitter, QStatusBar,
    QVBoxLayout, QWidget,
)

from canvas import ImageCanvas
from algo.omnisort import suggest_ids_from_previous
from label_io import (
    Box,
    read_det_labels,
    read_gt_labels,
    write_gt_labels,
)
from timeline import TimelineWidget


class DatasetDialog(QDialog):
    def __init__(self, datasets: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset")
        self.selected: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose a dataset to label:"))

        self._list = QListWidget()
        self._list.addItems(datasets)
        self._list.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self._list)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self._fit_to_screen()

    def _fit_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if not screen:
            self.resize(420, 320)
            return
        available = screen.availableGeometry()
        w = min(max(420, int(available.width() * 0.32)), int(available.width() * 0.80))
        h = min(max(320, int(available.height() * 0.45)), int(available.height() * 0.80))
        self.setMinimumSize(min(380, w), min(280, h))
        self.resize(w, h)
        self.move(
            available.x() + (available.width() - w) // 2,
            available.y() + (available.height() - h) // 2,
        )

    def _accept(self):
        items = self._list.selectedItems()
        if items:
            self.selected = items[0].text()
            self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Label & Track")

        self._clip_path: Optional[Path] = None
        self._frame_paths: List[Path] = []
        self._current_index: int = 0
        self._boxes_per_frame: Dict[int, List[Box]] = {}
        self._completed: Dict[int, bool] = {}
        self._dirty_frames: set[int] = set()
        self._undo_stack: Dict[int, List[List[Box]]] = {}
        self._copied_box: Optional[Box] = None
        self._selected_box: Optional[Box] = None
        self._active_overlay_key: Optional[int] = None
        self._pending_id_confirm: Optional[int] = None
        self._current_boxes_hidden_for_overlay = False
        self._first_load = True
        self._shortcuts: List[QShortcut] = []
        font = QApplication.font()
        self._base_font_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
        self._ui_zoom = 1.0

        self._init_ui()
        self._setup_shortcuts()
        self._update_save_state()
        self._update_id_summary()
        self._apply_ui_zoom()
        self._fit_main_window_to_screen()
        QApplication.instance().installEventFilter(self)
        self._select_dataset()

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #eef1f4;
                color: #161a1d;
            }
            QGraphicsView {
                border: 1px solid #15181b;
            }
            QGroupBox {
                background: #f8fafc;
                border: 1px solid #c8d0d9;
                border-radius: 6px;
                font-weight: 600;
                margin-top: 10px;
                padding: 8px 8px 8px 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #b8c2cc;
                border-radius: 4px;
                min-height: 26px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background: #edf5ff;
                border-color: #7aa7d9;
            }
            QPushButton:checked {
                background: #dff3e6;
                border-color: #5fab72;
            }
            QLineEdit {
                background: #ffffff;
                border: 1px solid #aab5c0;
                border-radius: 4px;
                min-height: 24px;
                padding: 2px 5px;
            }
            QLineEdit:focus {
                border-color: #3178c6;
            }
            QStatusBar {
                background: #f8fafc;
                border-top: 1px solid #c8d0d9;
            }
        """)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(6)

        # Top bar
        top = QHBoxLayout()
        self._lbl_dataset = QLabel("No dataset loaded")
        dataset_font = QFont(QApplication.font())
        dataset_font.setBold(True)
        dataset_font.setPointSizeF(max(dataset_font.pointSizeF(), self._base_font_size + 1.0))
        self._lbl_dataset.setFont(dataset_font)
        self._lbl_frame = QLabel("Frame: —")
        self._lbl_save_state = QLabel("No edits")
        self._lbl_save_state.setAlignment(Qt.AlignCenter)
        self._lbl_save_state.setMinimumWidth(140)
        btn_dataset = QPushButton("Dataset...")
        btn_dataset.clicked.connect(self._select_dataset)
        top.addWidget(self._lbl_dataset)
        top.addWidget(btn_dataset)
        top.addStretch()
        top.addWidget(self._lbl_save_state)
        top.addWidget(self._lbl_frame)
        layout.addLayout(top)

        # Splitter: canvas | sidebar
        splitter = QSplitter(Qt.Horizontal)
        self._canvas = ImageCanvas()
        self._canvas.box_selected.connect(self._on_box_selected)
        self._canvas.box_deselected.connect(self._on_box_deselected)
        self._canvas.box_change_started.connect(self._on_box_change_started)
        self._canvas.box_changed.connect(self._on_box_changed)
        self._canvas.box_drawn.connect(self._on_box_drawn)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._make_sidebar())
        splitter.setSizes([1100, 260])
        layout.addWidget(splitter, 1)

        # Timeline
        self._timeline = TimelineWidget()
        self._timeline.frame_clicked.connect(self._goto_frame)
        layout.addWidget(self._timeline)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a dataset to begin.")

    def _make_sidebar(self) -> QWidget:
        content = QWidget()
        content.setMinimumWidth(240)
        content.setMaximumWidth(330)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 0, 4, 0)
        layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(360)
        scroll.setWidget(content)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        w = content

        # --- box group ---
        box_grp = QGroupBox("Selected Box")
        bg = QVBoxLayout(box_grp)

        self._lbl_box_info = QLabel("No box selected")
        self._lbl_box_info.setWordWrap(True)
        bg.addWidget(self._lbl_box_info)

        row = QHBoxLayout()
        row.addWidget(QLabel("Identity:"))
        self._id_input = QLineEdit()
        self._id_input.setPlaceholderText("e.g. 1")
        self._id_input.setEnabled(False)
        self._id_input.returnPressed.connect(self._assign_identity)
        self._id_input.textChanged.connect(self._on_id_input_changed)
        row.addWidget(self._id_input)
        bg.addLayout(row)

        self._btn_assign = QPushButton("Assign ID")
        self._btn_assign.setEnabled(False)
        self._btn_assign.setToolTip("Assign the typed ID to the selected box (Enter)")
        self._btn_assign.clicked.connect(self._assign_identity)
        bg.addWidget(self._btn_assign)

        self._btn_remove_id = QPushButton("Clear Identity")
        self._btn_remove_id.setEnabled(False)
        self._btn_remove_id.setToolTip("Remove the identity from the selected box")
        self._btn_remove_id.clicked.connect(self._remove_identity)
        bg.addWidget(self._btn_remove_id)

        self._btn_delete_box = QPushButton("Delete Box")
        self._btn_delete_box.setEnabled(False)
        self._btn_delete_box.setToolTip("Delete the selected box (Delete key)")
        self._btn_delete_box.clicked.connect(self._delete_selected_box)
        bg.addWidget(self._btn_delete_box)

        self._chk_lock_position = QCheckBox("Lock box position")
        self._chk_lock_position.stateChanged.connect(self._toggle_box_geometry_locks)
        bg.addWidget(self._chk_lock_position)

        self._chk_lock_size = QCheckBox("Lock box size")
        self._chk_lock_size.stateChanged.connect(self._toggle_box_geometry_locks)
        bg.addWidget(self._chk_lock_size)

        layout.addWidget(box_grp)

        # --- frame group ---
        frm_grp = QGroupBox("Frame")
        fg = QVBoxLayout(frm_grp)

        self._btn_complete = QPushButton("Mark Completed")
        self._btn_complete.setCheckable(True)
        self._btn_complete.setToolTip("Mark this frame as fully annotated and save (Ctrl+Enter / Cmd+Enter)")
        self._btn_complete.clicked.connect(self._toggle_completed)
        fg.addWidget(self._btn_complete)

        self._btn_save = QPushButton("Save  (Ctrl+S)")
        self._btn_save.setToolTip("Save annotations for the current frame (Ctrl+S)")
        self._btn_save.clicked.connect(self._save_current)
        fg.addWidget(self._btn_save)

        self._btn_draw_box = QPushButton("Draw Box  (B)")
        self._btn_draw_box.setCheckable(True)
        self._btn_draw_box.setToolTip("Draw a new bounding box on the image (B key, Escape to cancel)")
        self._btn_draw_box.toggled.connect(self._toggle_draw_box)
        fg.addWidget(self._btn_draw_box)

        layout.addWidget(frm_grp)

        # --- navigation group ---
        nav_grp = QGroupBox("Navigation")
        ng = QVBoxLayout(nav_grp)

        row2 = QHBoxLayout()
        btn_prev = QPushButton("Prev")
        btn_prev.setToolTip("Go to previous frame (← arrow key)")
        btn_prev.clicked.connect(self._prev_frame)
        btn_next = QPushButton("Next")
        btn_next.setToolTip("Go to next frame (→ arrow key)")
        btn_next.clicked.connect(self._next_frame)
        row2.addWidget(btn_prev)
        row2.addWidget(btn_next)
        ng.addLayout(row2)

        btn_fit = QPushButton("Fit View  (F)")
        btn_fit.setToolTip("Fit the image to the viewport (F key)")
        btn_fit.clicked.connect(self._canvas.fit_view)
        ng.addWidget(btn_fit)

        lbl_overlay = QLabel(
            "Hold Q: prev boxes  |  Hold W: next\n"
            "Hold D: detections\n"
            "B: draw box  |  Del: delete box\n"
            "Ctrl+G: go to frame"
        )
        lbl_overlay.setWordWrap(True)
        ng.addWidget(lbl_overlay)

        btn_open = QPushButton("Open Dataset…")
        btn_open.setToolTip("Switch to a different dataset")
        btn_open.clicked.connect(self._select_dataset)
        ng.addWidget(btn_open)

        layout.addWidget(nav_grp)

        # --- tracker group ---
        tracker_grp = QGroupBox("ID Suggestions")
        tg = QVBoxLayout(tracker_grp)
        self._tracker_algo = QComboBox()
        self._tracker_algo.addItems(["OmniSORT"])
        tg.addWidget(self._tracker_algo)
        self._btn_suggest_ids = QPushButton("Suggest IDs")
        self._btn_suggest_ids.clicked.connect(self._suggest_ids_from_previous)
        tg.addWidget(self._btn_suggest_ids)
        layout.addWidget(tracker_grp)

        summary_grp = QGroupBox("ID Summary (against prev)")
        sg = QVBoxLayout(summary_grp)
        self._lbl_id_summary = QLabel("No dataset loaded.")
        self._lbl_id_summary.setTextFormat(Qt.RichText)
        self._lbl_id_summary.setWordWrap(True)
        self._lbl_id_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._lbl_id_summary.setStyleSheet(
            "QLabel {"
            "background: #f1f5f9; border: 1px solid #d8e0ea; border-radius: 5px; "
            "padding: 7px; color: #111827;"
            "}"
        )
        sg.addWidget(self._lbl_id_summary)
        layout.addWidget(summary_grp)

        layout.addStretch()
        return scroll

    def _setup_shortcuts(self):
        self._shortcuts = []
        for sequence, callback in (
            (QKeySequence(Qt.Key_Right), self._next_frame),
            (QKeySequence(Qt.Key_Left), self._prev_frame),
            (QKeySequence(Qt.Key_F), self._fit_view_requested),
            (QKeySequence(Qt.CTRL + Qt.Key_Return), self._toggle_completed),
            (QKeySequence(Qt.META + Qt.Key_Return), self._toggle_completed),
        ):
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        for standard_key, callback in (
            (QKeySequence.Save, self._save_current),
            (QKeySequence.Copy, self._copy_selected_box),
            (QKeySequence.Paste, self._paste_copied_box),
            (QKeySequence.Undo, self._undo_current_frame),
        ):
            shortcut = QShortcut(standard_key, self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        for sequence in ("Ctrl++", "Ctrl+="):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self._zoom_ui_in)
            self._shortcuts.append(shortcut)
        for sequence in ("Meta++", "Meta+=", "Ctrl+Shift+=", "Meta+Shift+="):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self._zoom_ui_in)
            self._shortcuts.append(shortcut)
        shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self._zoom_ui_out)
        self._shortcuts.append(shortcut)
        shortcut = QShortcut(QKeySequence("Meta+-"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self._zoom_ui_out)
        self._shortcuts.append(shortcut)
        shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self._reset_ui_zoom)
        self._shortcuts.append(shortcut)
        shortcut = QShortcut(QKeySequence("Meta+0"), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self._reset_ui_zoom)
        self._shortcuts.append(shortcut)
        for sequence, callback in (
            ("Ctrl+G", self._goto_frame_dialog),
            ("Meta+G", self._goto_frame_dialog),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        for key in (Qt.Key_Delete, Qt.Key_Backspace):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self._delete_selected_box)
            self._shortcuts.append(shortcut)

    def _apply_ui_zoom(self):
        font = QApplication.font()
        font.setPointSizeF(max(8.0, min(18.0, self._base_font_size * self._ui_zoom)))
        QApplication.setFont(font)
        self.updateGeometry()

    def _zoom_ui_in(self):
        self._ui_zoom = min(1.6, self._ui_zoom * 1.1)
        self._apply_ui_zoom()
        self._fit_main_window_to_screen()

    def _zoom_ui_out(self):
        self._ui_zoom = max(0.75, self._ui_zoom / 1.1)
        self._apply_ui_zoom()
        self._fit_main_window_to_screen()

    def _reset_ui_zoom(self):
        self._ui_zoom = 1.0
        self._apply_ui_zoom()
        self._fit_main_window_to_screen()

    def _fit_view_requested(self):
        if self._active_overlay_key is not None:
            self._status.showMessage("Release overlay key before changing zoom.")
            return
        self._canvas.fit_view()

    # ------------------------------------------------------------------ dataset

    def _select_dataset(self):
        if not self._handle_dirty_before_context_change("loading another dataset"):
            return
        data_dir = Path("data")
        if not data_dir.exists():
            self._status.showMessage("'data/' folder not found next to app/.")
            return
        datasets = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
        if not datasets:
            self._status.showMessage("No subfolders found in 'data/'.")
            return
        dlg = DatasetDialog(datasets, self)
        if dlg.exec_() == QDialog.Accepted and dlg.selected:
            self._load_dataset(data_dir / dlg.selected)
        else:
            self._fit_main_window_to_screen()

    def _load_dataset(self, clip_path: Path):
        self._clip_path = clip_path
        self._lbl_dataset.setText(f"Dataset: {clip_path.name}")

        frame_dir = clip_path / "frame"
        self._frame_paths = sorted(frame_dir.glob("*.png"))
        if not self._frame_paths:
            self._status.showMessage(f"No .png frames found in {frame_dir}")
            return

        self._boxes_per_frame.clear()
        self._completed.clear()
        self._dirty_frames.clear()
        self._undo_stack.clear()
        self._copied_box = None
        self._first_load = True

        # Detect previously completed frames
        gt_dir = clip_path / "label_gt"
        first_unlabelled = 0
        found_unlabelled = False
        for i, fp in enumerate(self._frame_paths):
            gt_path = gt_dir / (fp.stem + ".txt")
            done = gt_path.exists() and gt_path.stat().st_size > 0
            self._completed[i] = done
            if not done and not found_unlabelled:
                first_unlabelled = i
                found_unlabelled = True

        self._timeline.load_frames(self._frame_paths, eager_index=first_unlabelled)
        for i, done in self._completed.items():
            if done:
                self._timeline.set_completed(i, True)

        self._goto_frame(first_unlabelled)
        self._update_save_state()
        self._fit_main_window_to_screen()
        self._status.showMessage(
            f"Loaded {len(self._frame_paths)} frames from '{clip_path.name}'. "
            f"Starting at frame {first_unlabelled + 1}."
        )

    def _fit_main_window_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if not screen:
            self.resize(1280, 720)
            return
        available = screen.availableGeometry()
        max_w = int(available.width() * 0.92)
        max_h = int(available.height() * 0.88)
        w = max_w
        h = int(w * 9 / 16)
        if h > max_h:
            h = max_h
            w = int(h * 16 / 9)
        min_w = min(960, max_w)
        min_h = int(min_w * 9 / 16)
        if w < min_w and min_h <= max_h:
            w, h = min_w, min_h
        self.setMinimumSize(min(720, max_w), min(405, max_h))
        self.setGeometry(
            available.x() + (available.width() - w) // 2,
            available.y() + (available.height() - h) // 2,
            w,
            h,
        )

    def _format_frame_ranges(self, indices: List[int]) -> str:
        if not indices:
            return "None"
        frames = sorted({idx + 1 for idx in indices})
        ranges = []
        start = prev = frames[0]
        for frame in frames[1:]:
            if frame == prev + 1:
                prev = frame
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = frame
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ", ".join(ranges)

    def _format_id_list(self, ids: List[int]) -> str:
        return ", ".join(str(identity) for identity in ids) if ids else "None"

    def _id_summary_html(self, rows: List[tuple]) -> str:
        rendered_rows = []
        for label, value, color in rows:
            rendered_rows.append(
                "<tr>"
                f"<td style='padding:2px 8px 2px 0; color:#334155; white-space:nowrap;'><b>{label}</b></td>"
                f"<td style='padding:2px 0; color:{color};'>{value}</td>"
                "</tr>"
            )
        return (
            "<div style='line-height:1.35;'>"
            "<table cellspacing='0' cellpadding='0' width='100%'>"
            + "".join(rendered_rows)
            + "</table>"
            "</div>"
        )

    def _set_save_state_label(self, text: str, *, background: str, border: str, foreground: str):
        self._lbl_save_state.setText(text)
        self._lbl_save_state.setStyleSheet(
            "QLabel {"
            f"background: {background}; color: {foreground}; border: 1px solid {border}; "
            "border-radius: 10px; padding: 4px 10px; font-weight: 700;"
            "}"
        )

    def _update_save_state(self):
        if not self._frame_paths:
            self._set_save_state_label("No edits", background="#e5e7eb", border="#cbd5e1", foreground="#334155")
            return
        if self._dirty_frames:
            current_dirty = self._current_index in self._dirty_frames
            frame_text = f"frame {self._current_index + 1}" if current_dirty else f"{len(self._dirty_frames)} frames"
            self._set_save_state_label(
                f"Unsaved: {frame_text}",
                background="#fff7ed",
                border="#f59e0b",
                foreground="#9a3412",
            )
            return
        self._set_save_state_label(
            f"Saved {datetime.now().strftime('%H:%M:%S')}",
            background="#dcfce7",
            border="#16a34a",
            foreground="#166534",
        )

    def _update_id_summary(self):
        if not self._frame_paths:
            self._lbl_id_summary.setText(
                self._id_summary_html([("Status", "No dataset loaded.", "#64748b")])
            )
            return
        curr_boxes = self._get_boxes(self._current_index)
        curr_ids = {box.identity for box in curr_boxes if box.identity >= 0}
        unassigned = sum(1 for box in curr_boxes if box.identity < 0)
        total = len(curr_boxes)
        if self._current_index <= 0:
            self._lbl_id_summary.setText(
                self._id_summary_html([
                    ("Against", "No previous frame", "#64748b"),
                    ("Stayed", "None", "#64748b"),
                    ("Added", self._format_id_list(sorted(curr_ids)), "#166534"),
                    ("Disappeared", "None", "#64748b"),
                    ("Unassigned boxes", f"{unassigned} / {total}", "#9a3412" if unassigned else "#166534"),
                ])
            )
            return
        prev_ids = {box.identity for box in self._get_boxes(self._current_index - 1) if box.identity >= 0}
        stayed = sorted(prev_ids & curr_ids)
        added = sorted(curr_ids - prev_ids)
        disappeared = sorted(prev_ids - curr_ids)
        self._lbl_id_summary.setText(
            self._id_summary_html([
                ("Against", f"Frame {self._current_index}", "#475569"),
                ("Stayed", f"{len(stayed)} - {self._format_id_list(stayed)}", "#1d4ed8"),
                ("Added", f"{len(added)} - {self._format_id_list(added)}", "#166534"),
                ("Disappeared", f"{len(disappeared)} - {self._format_id_list(disappeared)}", "#b91c1c"),
                ("Unassigned boxes", f"{unassigned} / {total}", "#9a3412" if unassigned else "#166534"),
            ])
        )

    def _handle_dirty_before_context_change(self, action: str) -> bool:
        if not self._dirty_frames:
            return True
        frame_list = sorted(self._dirty_frames)
        summary = self._format_frame_ranges(frame_list)
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved Changes")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"Save edits to {len(frame_list)} frame(s) before {action}?")
        box.setInformativeText(f"Unsaved frames: {summary}")
        box.setDetailedText("\n".join(f"Frame {idx + 1}: {self._frame_paths[idx].name}" for idx in frame_list))
        box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Save)
        answer = box.exec_()
        if answer == QMessageBox.Save:
            for idx in sorted(self._dirty_frames):
                if not self._save_frame(idx):
                    return False
            return True
        if answer == QMessageBox.Discard:
            self._dirty_frames.clear()
            return True
        return False

    # ------------------------------------------------------------------ boxes

    def _get_boxes(self, index: int) -> List[Box]:
        if index not in self._boxes_per_frame:
            stem = self._frame_paths[index].stem
            gt_path  = self._clip_path / "label_gt"  / f"{stem}.txt"
            det_path = self._clip_path / "label_det" / f"{stem}.txt"
            if gt_path.exists():
                boxes = read_gt_labels(gt_path)
            else:
                boxes = read_det_labels(det_path)
            self._boxes_per_frame[index] = boxes
        return self._boxes_per_frame[index]

    def _get_detection_boxes(self, index: int) -> List[Box]:
        stem = self._frame_paths[index].stem
        det_path = self._clip_path / "label_det" / f"{stem}.txt"
        return read_det_labels(det_path)

    def _box_xyxy(self, box: Box) -> List[float]:
        return [
            box.x_center - box.width / 2,
            box.y_center - box.height / 2,
            box.x_center + box.width / 2,
            box.y_center + box.height / 2,
        ]

    def _copy_box(self, box: Box) -> Box:
        return Box(
            x_center=box.x_center,
            y_center=box.y_center,
            width=box.width,
            height=box.height,
            confidence=box.confidence,
            class_id=box.class_id,
            identity=box.identity,
        )

    def _copy_boxes(self, boxes: List[Box]) -> List[Box]:
        return [self._copy_box(box) for box in boxes]

    def _push_undo(self, index: Optional[int] = None):
        if not self._frame_paths:
            return
        idx = self._current_index if index is None else index
        stack = self._undo_stack.setdefault(idx, [])
        stack.append(self._copy_boxes(self._get_boxes(idx)))
        if len(stack) > 50:
            stack.pop(0)

    def _undo_current_frame(self):
        if not self._frame_paths:
            return
        if self._id_input.hasFocus():
            self._id_input.undo()
            return
        stack = self._undo_stack.get(self._current_index, [])
        if not stack:
            self._status.showMessage("Nothing to undo.", 3000)
            return
        self._boxes_per_frame[self._current_index] = stack.pop()
        self._selected_box = None
        self._mark_dirty(self._current_index)
        self._goto_frame(self._current_index)
        self._status.showMessage("Undid last change.", 3000)

    def _reload_boxes_for_frame(self, index: int):
        self._boxes_per_frame.pop(index, None)
        if index == self._current_index:
            self._goto_frame(index)

    # ------------------------------------------------------------------ navigation

    def _goto_frame(self, index: int):
        if not self._frame_paths:
            return
        if hasattr(self, "_btn_draw_box") and self._btn_draw_box.isChecked():
            self._btn_draw_box.setChecked(False)
        index = max(0, min(index, len(self._frame_paths) - 1))
        self._current_index = index

        pix = QPixmap(str(self._frame_paths[index]))
        if pix.isNull():
            self._status.showMessage(f"Cannot read frame {self._frame_paths[index]}")
            return

        boxes = self._get_boxes(index)
        keep_zoom = not self._first_load
        self._canvas.load_frame(pix, boxes, keep_zoom=keep_zoom)
        self._canvas.clear_reference_boxes()
        self._canvas.clear_warning_notices()
        self._canvas.set_current_boxes_visible(True)
        self._active_overlay_key = None
        self._current_boxes_hidden_for_overlay = False
        self._first_load = False

        self._timeline.set_current(index)
        n = len(self._frame_paths)
        dirty = " *" if index in self._dirty_frames else ""
        self._lbl_frame.setText(f"Frame: {index + 1} / {n}{dirty}")
        self._update_save_state()
        self._update_id_summary()

        done = self._completed.get(index, False)
        self._btn_complete.setChecked(done)
        self._btn_complete.setText("Completed" if done else "Mark Completed")

        self._on_box_deselected()

    def _prev_frame(self):
        self._goto_frame(self._current_index - 1)

    def _next_frame(self):
        self._goto_frame(self._current_index + 1)

    def _show_adjacent_overlay(self, key: int):
        if not self._frame_paths:
            return
        if self._canvas.is_interacting():
            self._status.showMessage("Release the mouse before showing frame overlays.")
            return
        if key == Qt.Key_Q:
            target = self._current_index - 1
            label = "prev"
            notice = "Overlay: previous frame"
        elif key == Qt.Key_W:
            target = self._current_index + 1
            label = "next"
            notice = "Overlay: next frame"
        elif key == Qt.Key_D:
            target = self._current_index
            label = "det"
            notice = "Overlay: detections"
        else:
            return

        if not 0 <= target < len(self._frame_paths):
            self._canvas.clear_reference_boxes()
            self._status.showMessage("No adjacent frame available for overlay.")
            return

        self._active_overlay_key = key
        self._pending_id_confirm = None
        self._canvas.clear_highlight()
        boxes = self._get_detection_boxes(target) if key == Qt.Key_D else self._get_boxes(target)
        if key in (Qt.Key_Q, Qt.Key_W):
            self._canvas.set_current_boxes_visible(False)
            self._current_boxes_hidden_for_overlay = True
        self._canvas.show_reference_boxes(boxes, label)
        self._canvas.set_overlay_notice(notice, notice_id="adjacent")
        self._status.showMessage(f"Showing {label} boxes while key is held.")

    def _clear_adjacent_overlay(self, key: int):
        if self._active_overlay_key == key:
            self._active_overlay_key = None
            try:
                if self._current_boxes_hidden_for_overlay:
                    self._canvas.set_current_boxes_visible(True)
                    self._current_boxes_hidden_for_overlay = False
                self._canvas.clear_reference_boxes()
            except RuntimeError:
                self._current_boxes_hidden_for_overlay = False
            self._status.showMessage("Reference overlay hidden.", 2000)

    # ------------------------------------------------------------------ box interaction

    def _on_box_selected(self, box: Box):
        self._clear_pending_id_state()
        self._selected_box = box
        self._canvas.clear_warning_notices()
        self._id_input.setEnabled(True)
        self._btn_assign.setEnabled(True)
        self._btn_remove_id.setEnabled(True)
        self._btn_delete_box.setEnabled(True)
        if box.identity >= 0:
            self._id_input.setText(str(box.identity))
            info = f"Identity: {box.identity}"
        else:
            self._id_input.clear()
            info = "Unassigned box"
        if box.confidence < 1.0:
            info += f"\nConf: {box.confidence:.2f}"
        self._lbl_box_info.setText(info)
        self._id_input.setFocus()
        self._id_input.selectAll()

    def _on_box_deselected(self):
        self._clear_pending_id_state()
        self._selected_box = None
        self._canvas.clear_warning_notices()
        self._id_input.setEnabled(False)
        self._btn_assign.setEnabled(False)
        self._btn_remove_id.setEnabled(False)
        self._btn_delete_box.setEnabled(False)
        self._id_input.clear()
        self._lbl_box_info.setText("No box selected")

    def _assign_identity(self):
        if self._selected_box is None:
            return
        try:
            identity = int(self._id_input.text())
            if identity < 0:
                raise ValueError
        except ValueError:
            self._clear_pending_id_state()
            self._show_unavailable_warning("ID must be a non-negative integer.")
            return

        # Case 1: same-frame duplicate — highlight the conflicting box
        conflict_box = next(
            (b for b in self._get_boxes(self._current_index)
             if b is not self._selected_box and b.identity == identity),
            None,
        )
        if conflict_box is not None:
            self._clear_pending_id_state()
            self._canvas.highlight_box(conflict_box)
            self._canvas.set_warning_notice(f"⚠ ID {identity} in use — see highlighted box")
            self._status.showMessage(
                f"ID {identity} is already assigned in this frame. Conflicting box is highlighted.", 5000
            )
            return

        # Case 2: ID belongs to a different object in previous frame (low IoU)
        if self._identity_taken_by_previous_object(identity, self._selected_box):
            if self._pending_id_confirm == identity:
                # Second click: confirmed — fall through to assign
                self._pending_id_confirm = None
                self._canvas.clear_reference_boxes()
                self._canvas.clear_highlight()
            else:
                # First click: show trajectory and ask for confirmation
                self._pending_id_confirm = identity
                self._canvas.clear_highlight()
                self._show_identity_trajectory_partial(identity)
                self._canvas.set_warning_notice(
                    f"⚠ ID {identity}: trajectory shown — Assign again to confirm"
                )
                self._status.showMessage(
                    f"ID {identity} belongs to another track. "
                    "Inspect the trajectory overlay, then click Assign ID again to confirm.",
                    7000,
                )
                return

        self._push_undo()
        self._selected_box.identity = identity
        self._canvas.refresh_boxes()
        self._canvas.clear_warning_notices()
        self._canvas.clear_highlight()
        self._lbl_box_info.setText(f"Identity: {identity}")
        self._mark_dirty(self._current_index)
        self._pending_id_confirm = None
        self._show_identity_trajectory_if_complete(identity)
        self._update_id_summary()
        self._status.showMessage(f"Assigned identity {identity}.", 3000)

    def _on_id_input_changed(self, text: str):
        if self._pending_id_confirm is None:
            return
        try:
            new_id = int(text)
            if new_id == self._pending_id_confirm:
                return
        except ValueError:
            pass
        self._clear_pending_id_state()

    def _clear_pending_id_state(self):
        self._pending_id_confirm = None
        self._canvas.clear_highlight()
        if self._active_overlay_key is None:
            self._canvas.clear_reference_boxes()

    def _show_identity_trajectory_partial(self, identity: int):
        trajectory = []
        for idx in range(self._current_index):
            match = next(
                (b for b in self._get_boxes(idx) if b.identity == identity),
                None,
            )
            if match is not None:
                trajectory.append(match)
        if trajectory:
            self._active_overlay_key = None
            self._canvas.show_trajectory(trajectory, identity)

    def _remove_identity(self):
        if self._selected_box is None:
            return
        self._push_undo()
        self._selected_box.identity = -1
        self._id_input.clear()
        self._clear_pending_id_state()
        self._canvas.refresh_boxes()
        self._canvas.clear_warning_notices()
        self._lbl_box_info.setText("Unassigned box")
        self._mark_dirty(self._current_index)
        self._update_id_summary()
        self._status.showMessage("Identity cleared.", 3000)

    def _delete_selected_box(self):
        if self._selected_box is None:
            return
        boxes = self._get_boxes(self._current_index)
        if self._selected_box in boxes:
            self._push_undo()
            boxes.remove(self._selected_box)
            self._selected_box = None
            self._mark_dirty(self._current_index)
            self._canvas.clear_warning_notices()
            self._goto_frame(self._current_index)
            self._status.showMessage("Box deleted.", 3000)

    def _copy_selected_box(self):
        if self._selected_box is None:
            self._status.showMessage("No selected box to copy.", 3000)
            return
        self._copied_box = self._copy_box(self._selected_box)
        self._canvas.clear_warning_notices()
        self._status.showMessage("Box copied.", 3000)

    def _paste_copied_box(self):
        if self._copied_box is None:
            self._status.showMessage("No copied box to paste.", 3000)
            return
        box = self._copy_box(self._copied_box)
        if box.identity >= 0 and self._identity_used_in_current_frame(box.identity, box):
            box.identity = -1
            self._status.showMessage("Box pasted without ID — that ID is already used in this frame.", 4000)
        else:
            self._status.showMessage("Box pasted.", 3000)
        self._push_undo()
        self._get_boxes(self._current_index).append(box)
        self._mark_dirty(self._current_index)
        self._canvas.clear_warning_notices()
        self._goto_frame(self._current_index)

    def _toggle_draw_box(self, enabled: bool):
        self._canvas.set_draw_mode(enabled)
        self._status.showMessage("Draw box mode enabled." if enabled else "Draw box mode disabled.", 3000)

    def _on_box_drawn(self, box: Box):
        boxes = self._get_boxes(self._current_index)
        self._push_undo()
        boxes.append(box)
        self._mark_dirty(self._current_index)
        self._btn_draw_box.setChecked(False)
        self._canvas.clear_warning_notices()
        self._goto_frame(self._current_index)
        self._status.showMessage("New box drawn.", 3000)

    def _on_box_changed(self, box: Box):
        self._mark_dirty(self._current_index)
        self._update_id_summary()
        self._status.showMessage("Box geometry changed.", 2000)

    def _on_box_change_started(self, box: Box):
        self._push_undo()

    def _identity_used_in_current_frame(self, identity: int, selected: Box) -> bool:
        return any(
            box is not selected and box.identity == identity
            for box in self._get_boxes(self._current_index)
        )

    def _identity_taken_by_previous_object(self, identity: int, selected: Box) -> bool:
        if self._current_index <= 0:
            return False
        prev_box = next(
            (box for box in self._get_boxes(self._current_index - 1) if box.identity == identity),
            None,
        )
        if prev_box is None:
            return False
        return self._box_iou(prev_box, selected) < 0.05

    def _box_iou(self, a: Box, b: Box) -> float:
        ax1, ay1, ax2, ay2 = self._box_xyxy(a)
        bx1, by1, bx2, by2 = self._box_xyxy(b)
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _show_unavailable_warning(self, message: str):
        self._canvas.set_warning_notice(message)
        self._status.showMessage(message)

    def _show_identity_trajectory_if_complete(self, identity: int):
        if self._current_index <= 0:
            return
        trajectory = []
        for idx in range(self._current_index):
            match = next(
                (box for box in self._get_boxes(idx) if box.identity == identity),
                None,
            )
            if match is None:
                return
            trajectory.append(match)
        self._active_overlay_key = None
        self._canvas.show_trajectory(trajectory, identity)

    def _mark_dirty(self, index: int):
        self._dirty_frames.add(index)
        if self._completed.get(index, False):
            self._completed[index] = False
            self._timeline.set_completed(index, False)
            if index == self._current_index:
                self._btn_complete.setChecked(False)
                self._btn_complete.setText("Mark Completed")
        self._lbl_frame.setText(
            f"Frame: {index + 1} / {len(self._frame_paths)} *"
        )
        self._update_save_state()

    def _toggle_box_geometry_locks(self, state: int):
        position_locked = self._chk_lock_position.isChecked()
        size_locked = self._chk_lock_size.isChecked()
        self._canvas.set_geometry_locks(position_locked, size_locked)
        if position_locked and size_locked:
            msg = "Box position and size locked."
        elif position_locked:
            msg = "Box position locked."
        elif size_locked:
            msg = "Box size locked."
        else:
            msg = "Box position and size unlocked."
        self._status.showMessage(msg)

    def _suggest_ids_from_previous(self):
        idx = self._current_index
        prev_idx = idx - 1
        if prev_idx < 0:
            self._status.showMessage("No previous frame available for ID suggestions.")
            return
        if not self._completed.get(prev_idx, False):
            self._status.showMessage("Previous frame must be completed before suggesting IDs.")
            return

        prev_boxes = [b for b in self._get_boxes(prev_idx) if b.identity >= 0]
        curr_boxes = self._get_boxes(idx)
        targets = [i for i, b in enumerate(curr_boxes) if b.identity < 0]
        if not prev_boxes or not targets:
            self._status.showMessage("No eligible previous IDs or current unassigned boxes.")
            return

        matches = suggest_ids_from_previous(prev_boxes, [curr_boxes[i] for i in targets])
        applied = 0
        if matches:
            self._push_undo(idx)
        for curr_local_idx, prev_idx_match in matches:
            curr_box = curr_boxes[targets[curr_local_idx]]
            prev_box = prev_boxes[prev_idx_match]
            if curr_box.identity < 0:
                curr_box.identity = prev_box.identity
                applied += 1

        if applied:
            self._mark_dirty(idx)
            self._canvas.refresh_boxes()
            self._canvas.clear_warning_notices()
            self._update_id_summary()
            self._status.showMessage(f"Suggested {applied} ID(s) with {self._tracker_algo.currentText()}.", 4000)
        else:
            self._status.showMessage("No confident ID suggestions found.", 4000)

    # ------------------------------------------------------------------ save / complete

    def _toggle_completed(self):
        idx = self._current_index
        done = self._btn_complete.isChecked()
        self._completed[idx] = done
        self._timeline.set_completed(idx, done)
        self._btn_complete.setText("Completed" if done else "Mark Completed")
        if done:
            if not self._save_current():
                self._completed[idx] = False
                self._timeline.set_completed(idx, False)
                self._btn_complete.setChecked(False)
                self._btn_complete.setText("Mark Completed")
                return
        self._reload_boxes_for_frame(idx)

    def _save_current(self):
        if not self._clip_path or not self._frame_paths:
            return False
        return self._save_frame(self._current_index)

    def _save_frame(self, idx: int) -> bool:
        if not self._clip_path or not self._frame_paths:
            return False
        boxes = self._get_boxes(idx)
        gt_path = self._gt_path_for_index(idx)
        if not write_gt_labels(gt_path, boxes):
            QMessageBox.critical(
                self, "Save Error",
                f"Could not write labels to:\n{gt_path}\n\nCheck disk space and permissions.",
            )
            return False
        self._dirty_frames.discard(idx)
        if idx == self._current_index:
            self._canvas.clear_reference_boxes()
            self._lbl_frame.setText(f"Frame: {idx + 1} / {len(self._frame_paths)}")
            self._update_id_summary()
        self._update_save_state()
        self._status.showMessage(f"Saved frame {idx + 1}: {gt_path.name}", 3000)
        return True

    def _gt_path_for_index(self, idx: int) -> Path:
        stem = self._frame_paths[idx].stem
        return self._clip_path / "label_gt" / f"{stem}.txt"

    def _goto_frame_dialog(self):
        if not self._frame_paths:
            return
        n = len(self._frame_paths)
        frame_num, ok = QInputDialog.getInt(
            self,
            "Go to Frame",
            f"Enter frame number (1 – {n}):",
            self._current_index + 1,
            1, n,
        )
        if ok:
            self._goto_frame(frame_num - 1)

    def closeEvent(self, event):
        if self._handle_dirty_before_context_change("closing"):
            event.accept()
        else:
            event.ignore()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            if self._handle_global_shortcut(event):
                return True
        return super().eventFilter(obj, event)

    def _handle_global_shortcut(self, event) -> bool:
        if QApplication.activeModalWidget() is not None:
            return False
        focus = QApplication.focusWidget()
        if focus is not None and focus is not self and not self.isAncestorOf(focus):
            return False
        mods = event.modifiers()
        key = event.key()
        primary_only = bool(mods & (Qt.ControlModifier | Qt.MetaModifier)) and not (mods & Qt.AltModifier)
        if primary_only and key == Qt.Key_S:
            self._save_current()
            return True
        if primary_only and key == Qt.Key_C:
            self._copy_selected_box()
            return True
        if primary_only and key == Qt.Key_V:
            self._paste_copied_box()
            return True
        if primary_only and key == Qt.Key_Z:
            self._undo_current_frame()
            return True
        if primary_only and key == Qt.Key_G:
            self._goto_frame_dialog()
            return True
        if primary_only and key == Qt.Key_Return:
            self._toggle_completed()
            return True
        if mods == Qt.NoModifier and key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected_box()
            return True
        return False

    # ------------------------------------------------------------------ key events

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        key = event.key()
        id_focused = hasattr(self, "_id_input") and self._id_input.hasFocus()
        if key in (Qt.Key_Q, Qt.Key_W, Qt.Key_D) and not id_focused:
            self._show_adjacent_overlay(key)
        elif key == Qt.Key_Right and not id_focused:
            self._next_frame()
        elif key == Qt.Key_Left and not id_focused:
            self._prev_frame()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace) and not id_focused:
            self._delete_selected_box()
        elif key == Qt.Key_B and not id_focused:
            self._btn_draw_box.setChecked(not self._btn_draw_box.isChecked())
        elif key == Qt.Key_Escape:
            if self._btn_draw_box.isChecked():
                self._btn_draw_box.setChecked(False)
            elif not id_focused:
                self._canvas.clear_selection()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
        if event.key() in (Qt.Key_Q, Qt.Key_W, Qt.Key_D):
            self._clear_adjacent_overlay(event.key())
        else:
            super().keyReleaseEvent(event)
