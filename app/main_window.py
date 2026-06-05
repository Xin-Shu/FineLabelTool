from __future__ import annotations
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QEvent, QObject, QRunnable, QSize, QSettings, Qt, QThread, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QImageReader, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QApplication, QComboBox, QHeaderView, QKeySequenceEdit, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QScrollArea, QShortcut, QSpinBox, QSplitter, QStatusBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from canvas import ImageCanvas
from algo.detector import DetectorError, suggest_detections, train_detector, trained_model_path
from algo.omnisort import suggest_ids_from_previous
from label_io import (
    Box,
    read_det_labels,
    read_gt_labels,
    snap_box_to_pixel_grid,
    snap_boxes_to_pixel_grid,
    write_gt_labels,
)
from timeline import TimelineWidget


HOTKEY_DEFS = [
    ("prev_frame", "Previous frame", "Left"),
    ("next_frame", "Next frame", "Right"),
    ("save", "Save current frame", "Ctrl+S"),
    ("complete", "Toggle frame completion", "Ctrl+Return"),
    ("fit_view", "Fit image to view", "F"),
    ("draw_box", "Draw box mode", "B"),
    ("delete_box", "Delete selected box", "Del"),
    ("select_all", "Select all boxes", "Ctrl+A"),
    ("copy", "Copy selected boxes", "Ctrl+C"),
    ("paste", "Paste copied box", "Ctrl+V"),
    ("undo", "Undo current-frame edit", "Ctrl+Z"),
    ("goto_frame", "Go to frame", "Ctrl+G"),
    ("ui_zoom_in", "Zoom UI in", "Ctrl+="),
    ("ui_zoom_out", "Zoom UI out", "Ctrl+-"),
    ("ui_zoom_reset", "Reset UI zoom", "Ctrl+0"),
    ("overlay_prev", "Hold previous-frame overlay", "Q"),
    ("overlay_next", "Hold next-frame overlay", "W"),
    ("overlay_det", "Hold detection overlay", "D"),
]


HOTKEY_DEFAULTS = {key: default for key, _, default in HOTKEY_DEFS}


class DatasetDialog(QDialog):
    def __init__(self, datasets: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset")
        self.selected: Optional[str] = None
        self._datasets = list(datasets)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Choose a dataset")
        title_font = QFont(QApplication.font())
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 3)
        title.setFont(title_font)
        layout.addWidget(title)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter datasets...")
        self._filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self._filter)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setUniformItemSizes(True)
        for name in datasets:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(0, 36))
            self._list.addItem(item)
        self._list.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self._list)

        self._count_label = QLabel()
        layout.addWidget(self._count_label)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.setStyleSheet("""
            QDialog { background: #eef1f4; color: #111827; }
            QLineEdit {
                background: #ffffff; border: 1px solid #b8c2cc;
                border-radius: 6px; padding: 7px 9px;
            }
            QListWidget {
                background: #ffffff; border: 1px solid #c8d0d9;
                border-radius: 8px; padding: 5px; outline: 0;
            }
            QListWidget::item {
                border-radius: 6px; padding: 6px 9px;
            }
            QListWidget::item:selected {
                background: #dbeafe; color: #0f172a;
            }
            QListWidget::item:hover {
                background: #edf5ff;
            }
        """)
        self._fit_to_screen()
        self._apply_filter("")

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
        font = QFont(QApplication.font())
        base = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
        font.setPointSizeF(min(18.0, max(base + 1.0, h / 38.0)))
        self.setFont(font)
        self.move(
            available.x() + (available.width() - w) // 2,
            available.y() + (available.height() - h) // 2,
        )

    def _apply_filter(self, text: str):
        needle = text.strip().lower()
        visible = 0
        first_visible = None
        for row in range(self._list.count()):
            item = self._list.item(row)
            match = needle in item.text().lower()
            item.setHidden(not match)
            if match:
                visible += 1
                if first_visible is None:
                    first_visible = item
        if first_visible is not None and not self._list.selectedItems():
            self._list.setCurrentItem(first_visible)
        self._count_label.setText(f"{visible} dataset(s)")

    def _accept(self):
        items = [item for item in self._list.selectedItems() if not item.isHidden()]
        item = items[0] if items else self._list.currentItem()
        if item is not None and not item.isHidden():
            self.selected = item.text()
            self.accept()


class HotkeyDialog(QDialog):
    def __init__(self, hotkeys: Dict[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        self._edits: Dict[str, QKeySequenceEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)

        title = QLabel("Keyboard Shortcuts")
        title_font = QFont(QApplication.font())
        title_font.setBold(True)
        title_font.setPointSizeF(title_font.pointSizeF() + 2)
        title.setFont(title_font)
        layout.addWidget(title)

        self._table = QTableWidget(len(HOTKEY_DEFS), 3)
        self._table.setHorizontalHeaderLabels(["Action", "Shortcut", "Default"])
        self._table.verticalHeader().hide()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

        for row, (key, label, default) in enumerate(HOTKEY_DEFS):
            action_item = QTableWidgetItem(label)
            default_item = QTableWidgetItem(default)
            action_item.setFlags(action_item.flags() & ~Qt.ItemIsEditable)
            default_item.setFlags(default_item.flags() & ~Qt.ItemIsEditable)
            edit = QKeySequenceEdit(QKeySequence(hotkeys.get(key, default)))
            if hasattr(edit, "setMaximumSequenceLength"):
                edit.setMaximumSequenceLength(1)
            self._edits[key] = edit
            self._table.setItem(row, 0, action_item)
            self._table.setCellWidget(row, 1, edit)
            self._table.setItem(row, 2, default_item)

        self._table.resizeRowsToContents()
        layout.addWidget(self._table, 1)

        self._hint = QLabel("Changes are saved for this user and applied immediately after OK.")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        restore = btns.button(QDialogButtonBox.RestoreDefaults)
        if restore is not None:
            restore.clicked.connect(self._restore_defaults)
        layout.addWidget(btns)

        self._fit_to_screen()

    def _fit_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if not screen:
            self.resize(680, 520)
            return
        available = screen.availableGeometry()
        w = min(max(640, int(available.width() * 0.38)), int(available.width() * 0.85))
        h = min(max(500, int(available.height() * 0.68)), int(available.height() * 0.88))
        self.resize(w, h)
        self.move(
            available.x() + (available.width() - w) // 2,
            available.y() + (available.height() - h) // 2,
        )

    def _restore_defaults(self):
        for key, _, default in HOTKEY_DEFS:
            self._edits[key].setKeySequence(QKeySequence(default))

    def hotkeys(self) -> Dict[str, str]:
        values = {}
        for key, edit in self._edits.items():
            values[key] = edit.keySequence().toString(QKeySequence.PortableText)
        return values


class _DetectorPredictWorker(QObject):
    finished = pyqtSignal(int, object, str, str, int)
    failed = pyqtSignal(int, str)

    def __init__(self, index: int, image_path: Path, clip_path: Path, base_model: str, confidence: float, image_size: int):
        super().__init__()
        self.index = index
        self.image_path = image_path
        self.clip_path = clip_path
        self.base_model = base_model
        self.confidence = confidence
        self.image_size = image_size

    def run(self):
        try:
            result = suggest_detections(
                self.image_path,
                self.clip_path,
                base_model=self.base_model,
                confidence=self.confidence,
                image_size=self.image_size,
            )
            self.finished.emit(self.index, result.boxes, result.source, result.device, result.tile_count)
        except DetectorError as exc:
            self.failed.emit(self.index, str(exc))
        except Exception as exc:
            self.failed.emit(self.index, f"Unexpected detector error: {exc}")


class _DetectorTrainWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        clip_path: Path,
        frame_paths: List[Path],
        completed_indices: List[int],
        base_model: str,
        epochs: int,
        image_size: int,
    ):
        super().__init__()
        self.clip_path = clip_path
        self.frame_paths = list(frame_paths)
        self.completed_indices = list(completed_indices)
        self.base_model = base_model
        self.epochs = epochs
        self.image_size = image_size

    def run(self):
        try:
            result = train_detector(
                self.clip_path,
                self.frame_paths,
                self.completed_indices,
                base_model=self.base_model,
                epochs=self.epochs,
                image_size=self.image_size,
            )
            self.finished.emit(result)
        except DetectorError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Unexpected detector training error: {exc}")


class _FrameLoadSignals(QObject):
    loaded = pyqtSignal(int, int, QImage)
    failed = pyqtSignal(int, int, str)


class _FrameLoadTask(QRunnable):
    def __init__(self, generation: int, index: int, path: Path, signals: _FrameLoadSignals):
        super().__init__()
        self.generation = generation
        self.index = index
        self.path = path
        self.signals = signals

    def run(self):
        reader = QImageReader(str(self.path))
        reader.setAutoTransform(True)
        image = reader.read()
        try:
            if image.isNull():
                self.signals.failed.emit(self.generation, self.index, reader.errorString())
            else:
                self.signals.loaded.emit(self.generation, self.index, image)
        except RuntimeError:
            pass


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
        self._copied_boxes: List[Box] = []
        self._selected_box: Optional[Box] = None
        self._active_overlay_key: Optional[int] = None
        self._pending_id_confirm: Optional[int] = None
        self._current_boxes_hidden_for_overlay = False
        self._first_load = True
        self._direct_unassigned_cursor = 0
        self._direct_disappeared_cursor = 0
        self._suggested_detection_boxes: List[Box] = []
        self._detector_predict_thread: Optional[QThread] = None
        self._detector_train_thread: Optional[QThread] = None
        self._detector_predict_worker: Optional[QObject] = None
        self._detector_train_worker: Optional[QObject] = None
        self._detector_training = False
        self._detector_prediction = False
        self._detector_train_queued = False
        self._frame_image_cache: OrderedDict[int, QImage] = OrderedDict()
        self._frame_loading_indexes: set[int] = set()
        self._frame_load_generation = 0
        self._frame_cache_max = 3
        self._frame_prefetch_radius = 1
        self._pending_frame_index: Optional[int] = None
        self._displayed_frame_index: Optional[int] = None
        self._frame_load_signals = _FrameLoadSignals()
        self._frame_load_signals.loaded.connect(self._on_frame_image_loaded)
        self._frame_load_signals.failed.connect(self._on_frame_image_failed)
        self._frame_pool = QThreadPool(self)
        self._frame_pool.setMaxThreadCount(1)
        self._shortcuts: List[QShortcut] = []
        self._settings = QSettings("Xin-Shu", "FineLabelTool")
        self._hotkeys = self._load_hotkeys()
        font = QApplication.font()
        self._base_font_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
        self._ui_zoom = 1.0
        self._detector_auto_timer = QTimer(self)
        self._detector_auto_timer.setSingleShot(True)
        self._detector_auto_timer.timeout.connect(self._start_detector_training)

        self._init_ui()
        self._setup_shortcuts()
        self._update_save_state()
        self._update_id_summary()
        self._refresh_detector_status()
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
                margin-top: 8px;
                padding: 6px 7px 6px 7px;
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
                min-height: 22px;
                padding: 2px 7px;
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
                min-height: 22px;
                padding: 2px 5px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 22px;
                padding: 1px 5px;
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
        self._lbl_task = QLabel("Task: Ready")
        self._lbl_task.setStyleSheet(
            "QLabel { background: #e5e7eb; color: #334155; border: 1px solid #cbd5e1; "
            "border-radius: 8px; padding: 3px 8px; font-weight: 600; }"
        )
        self._task_progress = QProgressBar()
        self._task_progress.setTextVisible(False)
        self._task_progress.setFixedSize(90, 8)
        self._task_progress.hide()
        btn_dataset = QPushButton("Dataset...")
        btn_dataset.clicked.connect(self._select_dataset)
        top.addWidget(self._lbl_dataset)
        top.addWidget(btn_dataset)
        top.addStretch()
        top.addWidget(self._lbl_task)
        top.addWidget(self._task_progress)
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
        layout.setContentsMargins(7, 0, 4, 0)
        layout.setSpacing(4)

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
        bg.setContentsMargins(7, 9, 7, 7)
        bg.setSpacing(4)

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
        assign_row = QHBoxLayout()
        assign_row.addWidget(self._btn_assign, 1)
        self._btn_assign_next_id = QPushButton("New ID: -")
        self._btn_assign_next_id.setEnabled(False)
        self._btn_assign_next_id.setToolTip("Assign the next unused tracking ID")
        self._btn_assign_next_id.clicked.connect(self._assign_next_identity)
        assign_row.addWidget(self._btn_assign_next_id)
        bg.addLayout(assign_row)

        self._btn_remove_id = QPushButton("Clear Identity")
        self._btn_remove_id.setEnabled(False)
        self._btn_remove_id.setToolTip("Remove identities from the selected box or boxes")
        self._btn_remove_id.clicked.connect(self._remove_identity)
        bg.addWidget(self._btn_remove_id)

        self._btn_delete_box = QPushButton("Delete Box")
        self._btn_delete_box.setEnabled(False)
        self._btn_delete_box.setToolTip("Delete the selected box (Delete key)")
        self._btn_delete_box.clicked.connect(self._delete_selected_box)
        bg.addWidget(self._btn_delete_box)

        layout.addWidget(box_grp)

        # --- frame group ---
        frm_grp = QGroupBox("Frame")
        fg = QVBoxLayout(frm_grp)
        fg.setContentsMargins(7, 9, 7, 7)
        fg.setSpacing(4)

        self._btn_complete = QPushButton("Mark Completed")
        self._btn_complete.setCheckable(True)
        self._btn_complete.setToolTip("Mark this frame as fully annotated and save")
        self._btn_complete.clicked.connect(self._toggle_completed)
        fg.addWidget(self._btn_complete)

        self._btn_save = QPushButton("Save")
        self._btn_save.setToolTip("Save annotations for the current frame")
        self._btn_save.clicked.connect(self._save_current)
        fg.addWidget(self._btn_save)

        self._btn_draw_box = QPushButton("Draw Box")
        self._btn_draw_box.setCheckable(True)
        self._btn_draw_box.setToolTip("Draw a new bounding box on the image")
        self._btn_draw_box.toggled.connect(self._toggle_draw_box)
        fg.addWidget(self._btn_draw_box)

        self._btn_hide_ids = QPushButton()
        self._btn_hide_ids.setCheckable(True)
        self._btn_hide_ids.setToolTip("Hide or show ID numbers drawn inside bounding boxes")
        hide_ids_value = self._settings.value("view/hide_ids", False)
        hide_ids = str(hide_ids_value).lower() in ("true", "1", "yes")
        self._btn_hide_ids.setChecked(hide_ids)
        self._set_id_numbers_hidden(hide_ids, show_status=False)
        self._btn_hide_ids.toggled.connect(self._set_id_numbers_hidden)
        fg.addWidget(self._btn_hide_ids)

        layout.addWidget(frm_grp)

        # --- detector assist group ---
        detector_grp = QGroupBox("Detection Assist")
        dg = QVBoxLayout(detector_grp)
        dg.setContentsMargins(7, 9, 7, 7)
        dg.setSpacing(4)

        detector_model_row = QHBoxLayout()
        detector_model_row.addWidget(QLabel("Model:"))
        self._detector_model = QComboBox()
        self._detector_model.addItem("YOLO11 Nano", "yolo11n.pt")
        self._detector_model.addItem("YOLOX Nano", "yolox:nano")
        self._detector_model.addItem("YOLOX Tiny", "yolox:tiny")
        self._detector_model.addItem("YOLOX Small", "yolox:s")
        self._detector_model.addItem("YOLOX Medium", "yolox:m")
        self._detector_model.addItem("YOLOX Large", "yolox:l")
        self._detector_model.addItem("YOLOX X-Large", "yolox:x")
        saved_model = self._settings.value("detector/base_model", "yolo11n.pt")
        for row_idx in range(self._detector_model.count()):
            if self._detector_model.itemData(row_idx) == saved_model:
                self._detector_model.setCurrentIndex(row_idx)
                break
        self._detector_model.currentIndexChanged.connect(self._on_detector_model_changed)
        detector_model_row.addWidget(self._detector_model, 1)
        dg.addLayout(detector_model_row)

        detector_opts_row = QHBoxLayout()
        detector_opts_row.addWidget(QLabel("Conf:"))
        self._detector_conf = QDoubleSpinBox()
        self._detector_conf.setRange(0.01, 0.95)
        self._detector_conf.setSingleStep(0.05)
        self._detector_conf.setDecimals(2)
        self._detector_conf.setValue(float(self._settings.value("detector/confidence", 0.25)))
        self._detector_conf.valueChanged.connect(self._save_detector_settings)
        detector_opts_row.addWidget(self._detector_conf)
        detector_opts_row.addWidget(QLabel("Epochs:"))
        self._detector_epochs = QSpinBox()
        self._detector_epochs.setRange(1, 50)
        self._detector_epochs.setValue(int(self._settings.value("detector/epochs", 6)))
        self._detector_epochs.valueChanged.connect(self._save_detector_settings)
        detector_opts_row.addWidget(self._detector_epochs)
        dg.addLayout(detector_opts_row)

        self._btn_suggest_detections = QPushButton("Suggest Detections")
        self._btn_suggest_detections.setToolTip("Run tiled detector inference on the current frame without changing labels")
        self._btn_suggest_detections.clicked.connect(self._suggest_detections_current_frame)
        dg.addWidget(self._btn_suggest_detections)

        detector_accept_row = QHBoxLayout()
        self._btn_accept_detections = QPushButton("Accept New")
        self._btn_accept_detections.setEnabled(False)
        self._btn_accept_detections.setToolTip("Add suggested boxes that do not overlap existing boxes")
        self._btn_accept_detections.clicked.connect(self._accept_suggested_detections)
        self._btn_clear_detections = QPushButton("Clear")
        self._btn_clear_detections.setEnabled(False)
        self._btn_clear_detections.setToolTip("Hide current detector suggestions")
        self._btn_clear_detections.clicked.connect(self._clear_suggested_detections)
        detector_accept_row.addWidget(self._btn_accept_detections)
        detector_accept_row.addWidget(self._btn_clear_detections)
        dg.addLayout(detector_accept_row)

        self._chk_detector_auto = QCheckBox("Auto-update after completed save")
        auto_value = self._settings.value("detector/auto_update", True)
        self._chk_detector_auto.setChecked(str(auto_value).lower() not in ("false", "0", "no"))
        self._chk_detector_auto.stateChanged.connect(self._save_detector_settings)
        dg.addWidget(self._chk_detector_auto)

        self._btn_update_detector = QPushButton("Update Detector")
        self._btn_update_detector.setToolTip("Fine-tune the tiled detector on all completed saved labels")
        self._btn_update_detector.clicked.connect(self._update_detector_now)
        dg.addWidget(self._btn_update_detector)

        self._lbl_detector_status = QLabel("Detector: tiled pretrained model until updated")
        self._lbl_detector_status.setWordWrap(True)
        self._lbl_detector_status.setStyleSheet(
            "QLabel { background: #f1f5f9; border: 1px solid #d8e0ea; "
            "border-radius: 5px; padding: 6px; color: #475569; }"
        )
        dg.addWidget(self._lbl_detector_status)
        layout.addWidget(detector_grp)

        # --- navigation group ---
        nav_grp = QGroupBox("Navigation")
        ng = QVBoxLayout(nav_grp)
        ng.setContentsMargins(7, 9, 7, 7)
        ng.setSpacing(4)

        btn_fit = QPushButton("Fit View  (F)")
        btn_fit.setToolTip("Fit the image to the viewport")
        btn_fit.clicked.connect(self._canvas.fit_view)
        ng.addWidget(btn_fit)

        btn_open = QPushButton("Open Dataset…")
        btn_open.setToolTip("Switch to a different dataset")
        btn_open.clicked.connect(self._select_dataset)
        ng.addWidget(btn_open)

        btn_hotkeys = QPushButton("Hotkeys...")
        btn_hotkeys.setToolTip("View and customize keyboard shortcuts")
        btn_hotkeys.clicked.connect(self._open_hotkey_dialog)
        ng.addWidget(btn_hotkeys)

        layout.addWidget(nav_grp)

        # --- tracker group ---
        tracker_grp = QGroupBox("ID Suggestions")
        tg = QVBoxLayout(tracker_grp)
        tg.setContentsMargins(7, 9, 7, 7)
        tg.setSpacing(4)
        self._tracker_algo = QComboBox()
        self._tracker_algo.addItems(["OmniSORT"])
        tg.addWidget(self._tracker_algo)
        self._btn_suggest_ids = QPushButton("Suggest IDs")
        self._btn_suggest_ids.clicked.connect(self._suggest_ids_from_previous)
        tg.addWidget(self._btn_suggest_ids)
        layout.addWidget(tracker_grp)

        summary_grp = QGroupBox("ID Summary (against prev)")
        sg = QVBoxLayout(summary_grp)
        sg.setContentsMargins(7, 9, 7, 7)
        sg.setSpacing(4)
        unassigned_row = QHBoxLayout()
        self._lbl_unassigned_title = QLabel("<b>Unassigned</b>")
        self._lbl_unassigned_title.setTextFormat(Qt.RichText)
        self._lbl_unassigned_value = QLabel("- / -")
        self._lbl_unassigned_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._btn_direct_unassigned = QPushButton("Direct")
        self._btn_direct_unassigned.setEnabled(False)
        self._btn_direct_unassigned.setToolTip("Jump to the next unassigned box")
        self._btn_direct_unassigned.clicked.connect(self._direct_unassigned_box)
        unassigned_row.addWidget(self._lbl_unassigned_title)
        unassigned_row.addStretch()
        unassigned_row.addWidget(self._lbl_unassigned_value)
        unassigned_row.addWidget(self._btn_direct_unassigned)
        sg.addLayout(unassigned_row)
        disappeared_row = QHBoxLayout()
        self._lbl_disappeared_title = QLabel("<b>Disappeared</b>")
        self._lbl_disappeared_title.setTextFormat(Qt.RichText)
        self._lbl_disappeared_value = QLabel("0")
        self._lbl_disappeared_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._btn_direct_disappeared = QPushButton("Direct")
        self._btn_direct_disappeared.setEnabled(False)
        self._btn_direct_disappeared.setToolTip("Jump to where the next disappeared ID was last seen")
        self._btn_direct_disappeared.clicked.connect(self._direct_disappeared_box)
        disappeared_row.addWidget(self._lbl_disappeared_title)
        disappeared_row.addStretch()
        disappeared_row.addWidget(self._lbl_disappeared_value)
        disappeared_row.addWidget(self._btn_direct_disappeared)
        sg.addLayout(disappeared_row)
        self._btn_show_trajectories = QPushButton("Show All Trajectories")
        self._btn_show_trajectories.setCheckable(True)
        self._btn_show_trajectories.setEnabled(False)
        self._btn_show_trajectories.setToolTip("Overlay trajectories for every ID seen up to this frame")
        self._btn_show_trajectories.clicked.connect(self._toggle_all_trajectories)
        sg.addWidget(self._btn_show_trajectories)
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
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts = []
        # Key execution is centralized in eventFilter so focused child widgets
        # cannot swallow app-level annotation shortcuts.

    def _load_hotkeys(self) -> Dict[str, str]:
        hotkeys = dict(HOTKEY_DEFAULTS)
        self._settings.beginGroup("hotkeys")
        try:
            for key in hotkeys:
                value = self._settings.value(key, hotkeys[key])
                if value:
                    hotkeys[key] = str(value)
        finally:
            self._settings.endGroup()
        return hotkeys

    def _save_hotkeys(self):
        self._settings.beginGroup("hotkeys")
        try:
            for key, value in self._hotkeys.items():
                self._settings.setValue(key, value)
        finally:
            self._settings.endGroup()
            self._settings.sync()

    def _set_id_numbers_hidden(self, hidden: bool, show_status: bool = True):
        if not hasattr(self, "_btn_hide_ids"):
            return
        hidden = bool(hidden)
        self._canvas.set_identity_labels_visible(not hidden)
        self._btn_hide_ids.setText("Show ID Numbers" if hidden else "Hide ID Numbers")
        self._settings.setValue("view/hide_ids", hidden)
        self._settings.sync()
        if show_status and hasattr(self, "_status"):
            self._status.showMessage("ID numbers hidden." if hidden else "ID numbers shown.", 3000)

    def _save_detector_settings(self, *args):
        if not hasattr(self, "_detector_model"):
            return
        self._settings.setValue("detector/base_model", self._detector_base_model())
        self._settings.setValue("detector/confidence", self._detector_conf.value())
        self._settings.setValue("detector/epochs", self._detector_epochs.value())
        self._settings.setValue("detector/auto_update", self._chk_detector_auto.isChecked())
        self._settings.sync()

    def _on_detector_model_changed(self, *args):
        self._save_detector_settings()
        self._suggested_detection_boxes = []
        self._canvas.clear_reference_boxes()
        self._refresh_detector_status()

    def _detector_base_model(self) -> str:
        if not hasattr(self, "_detector_model"):
            return "yolo11n.pt"
        return str(self._detector_model.currentData() or self._detector_model.currentText())

    def _detector_needs_yolox_backend(self) -> bool:
        return self._detector_base_model().startswith("yolox:")

    def _detector_image_size(self) -> int:
        return 640

    def _set_detector_status(self, text: str, color: str = "#475569"):
        if not hasattr(self, "_lbl_detector_status"):
            return
        self._lbl_detector_status.setText(text)
        self._lbl_detector_status.setStyleSheet(
            "QLabel { background: #f1f5f9; border: 1px solid #d8e0ea; "
            f"border-radius: 5px; padding: 6px; color: {color}; }}"
        )

    def _set_detector_controls_enabled(self):
        if not hasattr(self, "_btn_suggest_detections"):
            return
        has_dataset = bool(self._clip_path and self._frame_paths)
        busy = self._detector_prediction or self._detector_training
        backend_ready = not self._detector_needs_yolox_backend()
        self._btn_suggest_detections.setEnabled(has_dataset and not busy and backend_ready)
        self._btn_update_detector.setEnabled(has_dataset and not self._detector_training and backend_ready)
        self._btn_accept_detections.setEnabled(bool(self._suggested_detection_boxes) and not busy)
        self._btn_clear_detections.setEnabled(bool(self._suggested_detection_boxes))

    def _refresh_detector_status(self):
        if not hasattr(self, "_lbl_detector_status"):
            return
        if not self._clip_path:
            self._set_detector_status("Detector: open a dataset to begin")
            self._set_detector_controls_enabled()
            return
        if self._detector_needs_yolox_backend():
            self._set_detector_status("Detector: YOLOX backend not installed yet", "#9a3412")
            self._set_detector_controls_enabled()
            return
        model_path = trained_model_path(self._clip_path)
        if model_path.exists():
            self._set_detector_status(f"Detector: trained tiled model ready ({model_path.name})", "#166534")
        else:
            self._set_detector_status("Detector: tiled pretrained model until updated")
        self._set_detector_controls_enabled()

    def _shortcut_callbacks(self):
        return {
            "prev_frame": self._prev_frame,
            "next_frame": self._next_frame,
            "save": self._save_current,
            "complete": self._toggle_completed,
            "fit_view": self._fit_view_requested,
            "draw_box": self._toggle_draw_box_shortcut,
            "delete_box": self._delete_selected_box,
            "select_all": self._select_all_boxes,
            "copy": self._copy_selected_box,
            "paste": self._paste_copied_box,
            "undo": self._undo_current_frame,
            "goto_frame": self._goto_frame_dialog,
            "ui_zoom_in": self._zoom_ui_in,
            "ui_zoom_out": self._zoom_ui_out,
            "ui_zoom_reset": self._reset_ui_zoom,
        }

    def _open_hotkey_dialog(self):
        dialog = HotkeyDialog(self._hotkeys, self)
        if dialog.exec_() == QDialog.Accepted:
            self._hotkeys = dialog.hotkeys()
            self._save_hotkeys()
            self._setup_shortcuts()
            self._status.showMessage("Hotkeys updated.", 3000)

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

    def _fit_current_frame_after_layout(self):
        if not self._frame_paths:
            return
        self._canvas.fit_view()
        if hasattr(self._timeline, "center_current"):
            self._timeline.center_current()

    def _set_task_message(self, text: str, *, active: bool = False, color: str = "#334155"):
        if not hasattr(self, "_lbl_task"):
            return
        self._lbl_task.setText(f"Task: {text}")
        self._lbl_task.setStyleSheet(
            "QLabel { background: #e5e7eb; border: 1px solid #cbd5e1; "
            f"color: {color}; border-radius: 8px; padding: 3px 8px; font-weight: 600; }}"
        )
        if active:
            self._task_progress.setRange(0, 0)
            self._task_progress.show()
        else:
            self._task_progress.hide()

    def _configure_frame_loading_policy(self, focus_index: int):
        sample_indices = sorted({
            0,
            max(0, min(focus_index, len(self._frame_paths) - 1)),
            max(0, len(self._frame_paths) // 2),
            max(0, len(self._frame_paths) - 1),
        })
        sizes = []
        for idx in sample_indices:
            try:
                sizes.append(self._frame_paths[idx].stat().st_size)
            except OSError:
                pass
        max_size = max(sizes) if sizes else 0
        mib = max_size / (1024 * 1024)
        if mib >= 70:
            cache_max, radius, threads, thumb_threads, thumb_prefetch = 1, 0, 1, 1, 1
        elif mib >= 35:
            cache_max, radius, threads, thumb_threads, thumb_prefetch = 3, 1, 1, 1, 2
        elif mib >= 12:
            cache_max, radius, threads, thumb_threads, thumb_prefetch = 5, 2, 2, 1, 4
        else:
            cache_max, radius, threads, thumb_threads, thumb_prefetch = 12, 6, 2, 2, 8
        self._frame_cache_max = cache_max
        self._frame_prefetch_radius = radius
        self._frame_pool.setMaxThreadCount(threads)
        if hasattr(self._timeline, "set_loading_policy"):
            self._timeline.set_loading_policy(max_threads=thumb_threads, prefetch=thumb_prefetch)

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
        self._set_task_message("Scanning dataset...", active=True, color="#1d4ed8")
        QApplication.processEvents()

        frame_dir = clip_path / "frame"
        self._frame_paths = sorted(frame_dir.glob("*.png"))
        if not self._frame_paths:
            self._status.showMessage(f"No .png frames found in {frame_dir}")
            self._set_task_message("No frames", color="#b91c1c")
            return

        self._frame_load_generation += 1
        self._frame_image_cache.clear()
        self._frame_loading_indexes.clear()
        self._pending_frame_index = None
        self._displayed_frame_index = None
        self._boxes_per_frame.clear()
        self._completed.clear()
        self._dirty_frames.clear()
        self._undo_stack.clear()
        self._copied_boxes = []
        self._suggested_detection_boxes = []
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

        self._configure_frame_loading_policy(first_unlabelled)
        self._timeline.load_frames(self._frame_paths, eager_index=first_unlabelled)
        for i, done in self._completed.items():
            if done:
                self._timeline.set_completed(i, True)

        self._fit_main_window_to_screen()
        self._goto_frame(first_unlabelled)
        self._update_save_state()
        self._refresh_detector_status()
        QTimer.singleShot(0, self._fit_current_frame_after_layout)
        QTimer.singleShot(120, self._fit_current_frame_after_layout)
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
        values = sorted(set(ids))
        if not values:
            return "None"
        return ", ".join(str(identity) for identity in values)

    def _trajectory_ids_through_current(self) -> List[int]:
        if not self._frame_paths:
            return []
        identities = set()
        for idx in range(self._current_index + 1):
            identities.update(self._ids_in_frame(self._get_boxes(idx)))
        return sorted(identities)

    def _frame_sanity(self, index: int) -> Dict[str, object]:
        curr_boxes = self._get_boxes(index)
        curr_id_list = self._ids_in_frame(curr_boxes)
        curr_ids = set(curr_id_list)
        unassigned = sum(1 for box in curr_boxes if box.identity < 0)
        total = len(curr_boxes)
        sanity = []
        if unassigned:
            sanity.append("label all boxes")
        curr_duplicates = self._duplicate_ids(curr_id_list)
        if curr_duplicates:
            sanity.append(f"resolve duplicate current IDs: {self._format_id_list(curr_duplicates)}")

        prev_ids = set()
        prev_duplicates = []
        stayed = []
        added = sorted(curr_ids)
        disappeared = []
        if index > 0:
            prev_boxes = self._get_boxes(index - 1)
            prev_id_list = self._ids_in_frame(prev_boxes)
            prev_ids = set(prev_id_list)
            prev_duplicates = self._duplicate_ids(prev_id_list)
            if prev_duplicates:
                sanity.append(f"previous frame has duplicate IDs: {self._format_id_list(prev_duplicates)}")
            stayed = sorted(prev_ids & curr_ids)
            added = sorted(curr_ids - prev_ids)
            disappeared = sorted(prev_ids - curr_ids)
            if len(prev_ids) != len(stayed) + len(disappeared):
                sanity.append("previous IDs must equal stayed + disappeared")
            if len(curr_ids) != len(stayed) + len(added):
                sanity.append("current IDs must equal stayed + added")

        if len(curr_ids) != total - unassigned:
            sanity.append("current boxes must equal assigned IDs + unassigned")

        return {
            "passed": not sanity,
            "messages": sanity,
            "total": total,
            "unassigned": unassigned,
            "curr_ids": curr_ids,
            "prev_ids": prev_ids,
            "stayed": stayed,
            "added": added,
            "disappeared": disappeared,
        }

    def _ids_in_frame(self, boxes: List[Box]) -> List[int]:
        return [box.identity for box in boxes if box.identity >= 0]

    def _duplicate_ids(self, ids: List[int]) -> List[int]:
        seen = set()
        duplicates = set()
        for identity in ids:
            if identity in seen:
                duplicates.add(identity)
            seen.add(identity)
        return sorted(duplicates)

    def _next_unused_identity(self) -> int:
        used = set()
        for boxes in self._boxes_per_frame.values():
            used.update(self._ids_in_frame(boxes))
        if self._frame_paths:
            for idx in (self._current_index, self._current_index - 1, self._current_index + 1):
                if 0 <= idx < len(self._frame_paths):
                    used.update(self._ids_in_frame(self._get_boxes(idx)))
        return max(max(used) + 1, 1) if used else 1

    def _update_next_id_button(self):
        if not hasattr(self, "_btn_assign_next_id"):
            return
        next_id = self._next_unused_identity() if self._frame_paths else 1
        self._btn_assign_next_id.setText(f"New ID: {next_id}")
        enabled = self._selected_box is not None
        self._btn_assign_next_id.setEnabled(enabled)

    def _update_unassigned_direct(self, unassigned: int, total: int):
        if not hasattr(self, "_btn_direct_unassigned"):
            return
        if not self._frame_paths:
            self._lbl_unassigned_value.setText("- / -")
            self._btn_direct_unassigned.setEnabled(False)
            return
        self._lbl_unassigned_value.setText(f"{unassigned} / {total}")
        self._lbl_unassigned_value.setStyleSheet(
            f"color: {'#9a3412' if unassigned else '#166534'}; font-weight: 700;"
        )
        self._btn_direct_unassigned.setEnabled(unassigned > 0)

    def _update_disappeared_direct(self, disappeared_ids: List[int]):
        if not hasattr(self, "_btn_direct_disappeared"):
            return
        count = len(disappeared_ids)
        self._lbl_disappeared_value.setText(str(count))
        self._lbl_disappeared_value.setStyleSheet(
            f"color: {'#b91c1c' if count else '#64748b'}; font-weight: 700;"
        )
        self._btn_direct_disappeared.setEnabled(count > 0)

    def _unassigned_boxes_current(self) -> List[Box]:
        if not self._frame_paths:
            return []
        return [box for box in self._get_boxes(self._current_index) if box.identity < 0]

    def _disappeared_boxes_from_previous(self) -> List[Box]:
        if not self._frame_paths or self._current_index <= 0:
            return []
        curr_ids = {box.identity for box in self._get_boxes(self._current_index) if box.identity >= 0}
        boxes = [
            box for box in self._get_boxes(self._current_index - 1)
            if box.identity >= 0 and box.identity not in curr_ids
        ]
        return sorted(boxes, key=lambda box: box.identity)

    def _direct_unassigned_box(self):
        boxes = self._unassigned_boxes_current()
        if not boxes:
            self._status.showMessage("No unassigned boxes in this frame.", 3000)
            return
        if self._current_boxes_hidden_for_overlay:
            self._canvas.set_current_boxes_visible(True)
            self._current_boxes_hidden_for_overlay = False
        self._active_overlay_key = None
        self._canvas.clear_reference_boxes()
        self._set_trajectory_button_checked(False)
        self._canvas.clear_warning_notices()
        target = boxes[self._direct_unassigned_cursor % len(boxes)]
        self._direct_unassigned_cursor = (self._direct_unassigned_cursor + 1) % len(boxes)
        if self._canvas.focus_box(target, flashes=3):
            self._status.showMessage("Focused next unassigned box.", 3000)

    def _direct_disappeared_box(self):
        boxes = self._disappeared_boxes_from_previous()
        if not boxes:
            self._status.showMessage("No disappeared IDs from the previous frame.", 3000)
            return
        if self._current_boxes_hidden_for_overlay:
            self._canvas.set_current_boxes_visible(True)
            self._current_boxes_hidden_for_overlay = False
        self._active_overlay_key = None
        self._set_trajectory_button_checked(False)
        self._canvas.clear_warning_notices()
        target = boxes[self._direct_disappeared_cursor % len(boxes)]
        self._direct_disappeared_cursor = (self._direct_disappeared_cursor + 1) % len(boxes)
        if self._canvas.focus_reference_box(target, label="prev", flashes=3):
            self._status.showMessage(f"Focused last location of disappeared ID {target.identity}.", 3000)

    def _set_trajectory_button_checked(self, checked: bool):
        if not hasattr(self, "_btn_show_trajectories"):
            return
        self._btn_show_trajectories.blockSignals(True)
        self._btn_show_trajectories.setChecked(checked)
        self._btn_show_trajectories.blockSignals(False)

    def _toggle_all_trajectories(self, checked: bool):
        if not self._frame_paths:
            self._set_trajectory_button_checked(False)
            return
        if not checked:
            self._canvas.clear_reference_boxes()
            self._status.showMessage("Trajectory overlay hidden.", 3000)
            return
        if self._current_boxes_hidden_for_overlay:
            self._canvas.set_current_boxes_visible(True)
            self._current_boxes_hidden_for_overlay = False
        self._active_overlay_key = None
        self._suggested_detection_boxes = []
        self._set_detector_controls_enabled()
        trajectory_ids = self._trajectory_ids_through_current()
        if not trajectory_ids:
            self._set_trajectory_button_checked(False)
            self._status.showMessage("No assigned IDs up to this frame.", 3000)
            return
        trajectories = {identity: [] for identity in trajectory_ids}
        wanted = set(trajectory_ids)
        for idx in range(self._current_index + 1):
            seen_this_frame = set()
            for box in self._get_boxes(idx):
                if box.identity in wanted and box.identity not in seen_this_frame:
                    trajectories[box.identity].append(self._copy_box(box))
                    seen_this_frame.add(box.identity)
        self._canvas.clear_warning_notices()
        self._canvas.show_trajectories(trajectories)
        self._status.showMessage(f"Showing trajectories for {len(trajectory_ids)} ID(s) seen up to this frame.", 4000)

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
            self._update_unassigned_direct(0, 0)
            self._update_disappeared_direct([])
            self._update_next_id_button()
            if hasattr(self, "_btn_show_trajectories"):
                self._btn_show_trajectories.setEnabled(False)
            return
        result = self._frame_sanity(self._current_index)
        curr_ids = result["curr_ids"]
        unassigned = result["unassigned"]
        total = result["total"]
        self._update_unassigned_direct(unassigned, total)
        if hasattr(self, "_btn_show_trajectories"):
            self._btn_show_trajectories.setEnabled(bool(self._trajectory_ids_through_current()))
        sanity = result["messages"]
        if self._current_index <= 0:
            sanity_text = (
                "<b style='color:#166534;'>All good!</b>"
                if result["passed"] else
                f"<i style='color:#b91c1c;'>{'; '.join(sanity)}</i>"
            )
            self._lbl_id_summary.setText(
                self._id_summary_html([
                    ("Against", "No previous frame", "#64748b"),
                    ("Sanity", sanity_text, "#111827"),
                    (f"Added ({len(curr_ids)})", self._format_id_list(sorted(curr_ids)), "#166534"),
                    ("Stayed (0)", "None", "#64748b"),
                ])
            )
            self._update_disappeared_direct([])
            self._update_next_id_button()
            return
        stayed = result["stayed"]
        added = result["added"]
        disappeared = result["disappeared"]
        self._update_disappeared_direct(disappeared)
        sanity_text = (
            "<b style='color:#166534;'>All good!</b>"
            if result["passed"] else
            f"<i style='color:#b91c1c;'>{'; '.join(sanity)}</i>"
        )
        self._lbl_id_summary.setText(
            self._id_summary_html([
                ("Against", f"Frame {self._current_index}", "#475569"),
                ("Sanity", sanity_text, "#111827"),
                (f"Added ({len(added)})", self._format_id_list(added), "#166534"),
                (f"Stayed ({len(stayed)})", self._format_id_list(stayed), "#1d4ed8"),
            ])
        )
        self._update_next_id_button()

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
            self._snap_boxes_for_index(index, boxes)
            self._boxes_per_frame[index] = boxes
        return self._boxes_per_frame[index]

    def _get_detection_boxes(self, index: int) -> List[Box]:
        stem = self._frame_paths[index].stem
        det_path = self._clip_path / "label_det" / f"{stem}.txt"
        boxes = read_det_labels(det_path)
        self._snap_boxes_for_index(index, boxes)
        return boxes

    def _frame_image_size(self, index: int) -> Optional[QSize]:
        if not 0 <= index < len(self._frame_paths):
            return None
        cached = self._frame_image_cache.get(index)
        if cached is not None and not cached.isNull():
            return QSize(cached.width(), cached.height())
        size = QImageReader(str(self._frame_paths[index])).size()
        return size if size.isValid() else None

    def _snap_boxes_for_index(self, index: int, boxes: List[Box]) -> None:
        size = self._frame_image_size(index)
        if size is not None:
            snap_boxes_to_pixel_grid(boxes, size.width(), size.height())

    def _snap_box_for_index(self, index: int, box: Box) -> None:
        size = self._frame_image_size(index)
        if size is not None:
            snap_box_to_pixel_grid(box, size.width(), size.height())

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

    # ------------------------------------------------------------------ detector assist

    def _suggest_detections_current_frame(self):
        if not self._clip_path or not self._frame_paths or self._detector_prediction:
            return
        if self._pending_frame_index is not None:
            self._status.showMessage("Wait for the current frame to finish loading before running detector suggestions.", 4000)
            return
        self._clear_suggested_detections(show_status=False)
        self._detector_prediction = True
        self._set_detector_controls_enabled()
        self._set_detector_status("Detector: running on current frame...", "#1d4ed8")
        self._set_task_message("Running detector...", active=True, color="#1d4ed8")
        self._status.showMessage("Running detector suggestions for this frame.", 3000)

        thread = QThread(self)
        worker = _DetectorPredictWorker(
            self._current_index,
            self._frame_paths[self._current_index],
            self._clip_path,
            self._detector_base_model(),
            self._detector_conf.value(),
            self._detector_image_size(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detector_suggestions_ready)
        worker.failed.connect(self._on_detector_suggestions_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detector_prediction_finished)
        self._detector_predict_thread = thread
        self._detector_predict_worker = worker
        thread.start()

    def _on_detector_suggestions_ready(self, index: int, boxes: List[Box], source: str, device: str, tile_count: int):
        if index != self._current_index:
            self._set_task_message("Ready")
            self._status.showMessage(
                f"Detector suggestions for frame {index + 1} were discarded after frame change.",
                4000,
            )
            return
        self._snap_boxes_for_index(index, boxes)
        self._suggested_detection_boxes = self._copy_boxes(boxes)
        self._set_detector_controls_enabled()
        if not boxes:
            self._canvas.clear_reference_boxes()
            self._set_trajectory_button_checked(False)
            self._set_detector_status(f"Detector: no boxes at conf {self._detector_conf.value():.2f}", "#9a3412")
            self._set_task_message("Ready")
            self._status.showMessage("Detector found no boxes on this frame.", 4000)
            return
        self._set_trajectory_button_checked(False)
        self._canvas.show_reference_boxes(self._suggested_detection_boxes, "suggested")
        self._canvas.set_overlay_notice(f"Detector suggestions: {len(boxes)}", notice_id="detector")
        self._set_detector_status(
            f"Detector: {len(boxes)} suggestion(s) from {source} on {device}, {tile_count} tile(s)",
            "#166534",
        )
        self._status.showMessage(
            f"Detector suggested {len(boxes)} box(es). Review, then Accept New or Clear.",
            5000,
        )
        self._set_task_message("Ready")

    def _on_detector_suggestions_failed(self, index: int, message: str):
        self._suggested_detection_boxes = []
        self._canvas.clear_reference_boxes()
        self._set_trajectory_button_checked(False)
        self._set_detector_status("Detector: suggestion failed", "#b91c1c")
        self._set_task_message("Detector failed", color="#b91c1c")
        QMessageBox.warning(self, "Detector Suggestion Failed", message)

    def _on_detector_prediction_finished(self):
        self._detector_prediction = False
        self._detector_predict_thread = None
        self._detector_predict_worker = None
        self._set_detector_controls_enabled()

    def _clear_suggested_detections(self, show_status: bool = True):
        self._suggested_detection_boxes = []
        self._canvas.clear_reference_boxes()
        self._set_trajectory_button_checked(False)
        self._set_detector_controls_enabled()
        self._refresh_detector_status()
        if show_status:
            self._status.showMessage("Detector suggestions cleared.", 3000)

    def _accept_suggested_detections(self):
        if not self._suggested_detection_boxes:
            self._status.showMessage("No detector suggestions to accept.", 3000)
            return
        current_boxes = self._get_boxes(self._current_index)
        accepted: List[Box] = []
        skipped = 0
        for candidate in self._copy_boxes(self._suggested_detection_boxes):
            candidate.identity = -1
            self._snap_box_for_index(self._current_index, candidate)
            if any(self._box_iou(candidate, existing) >= 0.55 for existing in current_boxes + accepted):
                skipped += 1
                continue
            accepted.append(candidate)

        if not accepted:
            self._status.showMessage("All detector suggestions overlap existing boxes.", 4000)
            return

        self._push_undo()
        current_boxes.extend(accepted)
        self._mark_dirty(self._current_index)
        self._suggested_detection_boxes = []
        self._canvas.clear_reference_boxes()
        self._goto_frame(self._current_index)
        if skipped:
            self._status.showMessage(
                f"Accepted {len(accepted)} new detector box(es); skipped {skipped} overlapping suggestion(s).",
                5000,
            )
        else:
            self._status.showMessage(f"Accepted {len(accepted)} detector box(es).", 4000)

    def _completed_indices_for_detector(self) -> List[int]:
        return sorted(idx for idx, done in self._completed.items() if done)

    def _queue_detector_training(self):
        if not hasattr(self, "_chk_detector_auto") or not self._chk_detector_auto.isChecked():
            return
        if self._detector_training:
            self._detector_train_queued = True
            self._set_detector_status("Detector: update queued after current training", "#1d4ed8")
            return
        self._detector_train_queued = True
        self._set_detector_status("Detector: update queued from completed labels", "#1d4ed8")
        self._detector_auto_timer.start(1800)

    def _update_detector_now(self):
        self._detector_train_queued = True
        self._start_detector_training()

    def _start_detector_training(self):
        if not self._clip_path or not self._frame_paths:
            return
        if self._detector_training:
            self._detector_train_queued = True
            return
        completed_indices = self._completed_indices_for_detector()
        if not completed_indices:
            self._detector_train_queued = False
            self._set_detector_status("Detector: no completed labels to train from", "#9a3412")
            self._set_task_message("Ready")
            self._status.showMessage("Complete and save at least one labelled frame before updating the detector.", 5000)
            return

        self._detector_train_queued = False
        self._detector_training = True
        self._set_detector_controls_enabled()
        self._set_detector_status(f"Detector: training on {len(completed_indices)} completed frame(s)...", "#1d4ed8")
        self._set_task_message("Training detector...", active=True, color="#1d4ed8")
        self._status.showMessage("Detector training started in the background.", 4000)

        thread = QThread(self)
        worker = _DetectorTrainWorker(
            self._clip_path,
            self._frame_paths,
            completed_indices,
            self._detector_base_model(),
            self._detector_epochs.value(),
            self._detector_image_size(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_detector_training_finished)
        worker.failed.connect(self._on_detector_training_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detector_training_thread_finished)
        self._detector_train_thread = thread
        self._detector_train_worker = worker
        thread.start()

    def _on_detector_training_finished(self, result):
        if not result.updated:
            self._set_detector_status(
                f"Detector: already current; {result.skipped_frame_count} completed frame(s) reused",
                "#166534",
            )
            self._set_task_message("Ready")
            self._status.showMessage("Detector already includes the completed labels. No training needed.", 5000)
            return
        self._set_detector_status(
            f"Detector: updated on {result.frame_count} new/changed frame(s), {result.tile_count} tile(s), {result.box_count} label(s) using {result.device}",
            "#166534",
        )
        self._set_task_message("Ready")
        self._status.showMessage("Detector update finished. New suggestions will use the trained model.", 5000)

    def _on_detector_training_failed(self, message: str):
        self._set_detector_status("Detector: update failed", "#b91c1c")
        self._set_task_message("Detector failed", color="#b91c1c")
        QMessageBox.warning(self, "Detector Update Failed", message)

    def _on_detector_training_thread_finished(self):
        self._detector_training = False
        self._detector_train_thread = None
        self._detector_train_worker = None
        self._set_detector_controls_enabled()
        if self._detector_train_queued:
            self._detector_auto_timer.start(1200)

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

    def _cache_frame_image(self, index: int, image: QImage):
        self._frame_image_cache[index] = image
        self._frame_image_cache.move_to_end(index)
        while len(self._frame_image_cache) > self._frame_cache_max:
            evicted = False
            for key in list(self._frame_image_cache.keys()):
                if key not in (self._current_index, self._pending_frame_index):
                    self._frame_image_cache.pop(key, None)
                    evicted = True
                    break
            if not evicted:
                break

    def _request_frame_image(self, index: int, *, priority: bool = False):
        if not 0 <= index < len(self._frame_paths):
            return
        if index in self._frame_image_cache or index in self._frame_loading_indexes:
            return
        self._frame_loading_indexes.add(index)
        task = _FrameLoadTask(
            self._frame_load_generation,
            index,
            self._frame_paths[index],
            self._frame_load_signals,
        )
        self._frame_pool.start(task, 1 if priority else 0)

    def _prefetch_frames_around(self, index: int):
        if self._frame_prefetch_radius <= 0:
            return
        for offset in range(1, self._frame_prefetch_radius + 1):
            self._request_frame_image(index - offset)
            self._request_frame_image(index + offset)

    def _on_frame_image_loaded(self, generation: int, index: int, image: QImage):
        self._frame_loading_indexes.discard(index)
        if generation != self._frame_load_generation:
            return
        self._cache_frame_image(index, image)
        if index == self._pending_frame_index:
            self._display_frame_image(index, image)

    def _on_frame_image_failed(self, generation: int, index: int, message: str):
        self._frame_loading_indexes.discard(index)
        if generation != self._frame_load_generation:
            return
        if index == self._pending_frame_index:
            self._pending_frame_index = None
            self._canvas.setEnabled(True)
            self._set_task_message("Frame load failed", color="#b91c1c")
            self._status.showMessage(f"Cannot read frame {index + 1}: {message}", 6000)

    def _display_frame_image(self, index: int, image: QImage):
        self._pending_frame_index = None
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            self._canvas.setEnabled(True)
            self._set_task_message("Frame load failed", color="#b91c1c")
            self._status.showMessage(f"Cannot display frame {self._frame_paths[index]}", 6000)
            return

        boxes = self._get_boxes(index)
        keep_zoom = not self._first_load and self._displayed_frame_index is not None
        self._canvas.load_frame(pix, boxes, keep_zoom=keep_zoom)
        self._canvas.setEnabled(True)
        self._displayed_frame_index = index
        self._suggested_detection_boxes = []
        self._canvas.clear_reference_boxes()
        self._set_trajectory_button_checked(False)
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
        self._set_detector_controls_enabled()

        done = self._completed.get(index, False)
        self._btn_complete.setChecked(done)
        self._btn_complete.setText("Completed" if done else "Mark Completed")

        self._on_box_deselected()
        self._set_task_message("Ready")
        self._prefetch_frames_around(index)

    def _goto_frame(self, index: int):
        if not self._frame_paths:
            return
        if hasattr(self, "_btn_draw_box") and self._btn_draw_box.isChecked():
            self._btn_draw_box.setChecked(False)
        index = max(0, min(index, len(self._frame_paths) - 1))
        if index != self._current_index:
            self._direct_unassigned_cursor = 0
            self._direct_disappeared_cursor = 0
        self._current_index = index

        self._pending_frame_index = index
        self._canvas.setEnabled(False)
        self._suggested_detection_boxes = []
        self._canvas.clear_reference_boxes()
        self._set_trajectory_button_checked(False)
        self._canvas.clear_warning_notices()
        self._active_overlay_key = None
        self._current_boxes_hidden_for_overlay = False

        self._timeline.set_current(index)
        n = len(self._frame_paths)
        dirty = " *" if index in self._dirty_frames else ""
        self._lbl_frame.setText(f"Frame: {index + 1} / {n}{dirty}")
        self._update_save_state()
        self._set_detector_controls_enabled()

        done = self._completed.get(index, False)
        self._btn_complete.setChecked(done)
        self._btn_complete.setText("Completed" if done else "Mark Completed")

        self._on_box_deselected()
        cached = self._frame_image_cache.get(index)
        if cached is not None:
            self._frame_image_cache.move_to_end(index)
            self._display_frame_image(index, cached)
            return

        self._set_task_message(f"Loading frame {index + 1}/{n}", active=True, color="#1d4ed8")
        self._status.showMessage(f"Loading frame {index + 1}...", 3000)
        self._request_frame_image(index, priority=True)

    def _prev_frame(self):
        self._goto_frame(self._current_index - 1)

    def _next_frame(self):
        self._goto_frame(self._current_index + 1)

    def _show_adjacent_overlay(self, key: int):
        if not self._frame_paths:
            return
        if self._pending_frame_index is not None:
            self._status.showMessage("Wait for the current frame to finish loading before showing overlays.")
            return
        if self._canvas.is_interacting():
            self._status.showMessage("Release the mouse before showing frame overlays.")
            return
        if self._suggested_detection_boxes:
            self._clear_suggested_detections(show_status=False)
        self._set_trajectory_button_checked(False)
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
            self._set_trajectory_button_checked(False)
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
        self._update_next_id_button()
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
        self._update_next_id_button()
        self._btn_remove_id.setEnabled(False)
        self._btn_delete_box.setEnabled(False)
        self._id_input.clear()
        self._lbl_box_info.setText("No box selected")

    def _on_multiple_boxes_selected(self, count: int):
        self._clear_pending_id_state()
        self._selected_box = None
        self._canvas.clear_warning_notices()
        self._id_input.setEnabled(False)
        self._btn_assign.setEnabled(False)
        self._update_next_id_button()
        self._btn_remove_id.setEnabled(count > 0)
        self._btn_delete_box.setEnabled(count > 0)
        self._id_input.clear()
        self._lbl_box_info.setText(f"{count} boxes selected")

    def _assign_identity(self):
        if self._selected_box is None:
            return
        try:
            identity = int(self._id_input.text())
            if identity < 1:
                raise ValueError
        except ValueError:
            self._clear_pending_id_state()
            self._show_unavailable_warning("ID must be a positive integer starting from 1.")
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

    def _assign_next_identity(self):
        if self._selected_box is None:
            return
        identity = self._next_unused_identity()
        self._id_input.setText(str(identity))
        self._assign_identity()

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
            if self._suggested_detection_boxes:
                self._suggested_detection_boxes = []
                self._set_detector_controls_enabled()
            self._canvas.clear_reference_boxes()
            self._set_trajectory_button_checked(False)

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
            self._set_trajectory_button_checked(False)
            self._canvas.show_trajectory(trajectory, identity)

    def _remove_identity(self):
        selected_boxes = self._canvas.get_selected_boxes()
        if not selected_boxes and self._selected_box is not None:
            selected_boxes = [self._selected_box]
        if not selected_boxes:
            return
        boxes_to_clear = [box for box in selected_boxes if box.identity >= 0]
        if not boxes_to_clear:
            self._status.showMessage("Selected boxes already have no identity.", 3000)
            return
        self._push_undo()
        for box in boxes_to_clear:
            box.identity = -1
        self._id_input.clear()
        self._clear_pending_id_state()
        self._canvas.refresh_boxes()
        self._canvas.clear_warning_notices()
        if len(selected_boxes) == 1:
            self._lbl_box_info.setText("Unassigned box")
        else:
            self._selected_box = None
            self._lbl_box_info.setText(f"{len(selected_boxes)} boxes selected")
        self._mark_dirty(self._current_index)
        self._update_id_summary()
        cleared_count = len(boxes_to_clear)
        self._status.showMessage(
            "Identity cleared." if cleared_count == 1 else f"Cleared identities from {cleared_count} boxes.",
            3000,
        )

    def _delete_selected_box(self):
        selected_boxes = self._canvas.get_selected_boxes()
        if not selected_boxes and self._selected_box is not None:
            selected_boxes = [self._selected_box]
        if not selected_boxes:
            return
        boxes = self._get_boxes(self._current_index)
        selected_ids = {id(box) for box in selected_boxes}
        kept_boxes = [box for box in boxes if id(box) not in selected_ids]
        deleted_count = len(boxes) - len(kept_boxes)
        if deleted_count:
            self._push_undo()
            boxes[:] = kept_boxes
            self._selected_box = None
            self._mark_dirty(self._current_index)
            self._canvas.clear_warning_notices()
            self._goto_frame(self._current_index)
            self._status.showMessage(
                "Box deleted." if deleted_count == 1 else f"{deleted_count} boxes deleted.",
                3000,
            )

    def _select_all_boxes(self):
        if not self._frame_paths:
            return
        selected_boxes = self._canvas.select_all_boxes()
        count = len(selected_boxes)
        if count == 0:
            self._on_box_deselected()
            self._status.showMessage("No boxes in this frame.", 3000)
            return
        self._on_multiple_boxes_selected(count)
        self._status.showMessage(f"Selected {count} boxes.", 3000)

    def _copy_selected_box(self):
        selected_boxes = self._canvas.get_selected_boxes()
        if not selected_boxes and self._selected_box is not None:
            selected_boxes = [self._selected_box]
        if not selected_boxes:
            self._status.showMessage("No selected boxes to copy.", 3000)
            return
        self._copied_boxes = self._copy_boxes(selected_boxes)
        self._canvas.clear_warning_notices()
        count = len(self._copied_boxes)
        self._status.showMessage(
            "Box copied." if count == 1 else f"{count} boxes copied.",
            3000,
        )

    def _paste_copied_box(self):
        if not self._copied_boxes:
            self._status.showMessage("No copied boxes to paste.", 3000)
            return
        existing_boxes = self._get_boxes(self._current_index)
        used_ids = {box.identity for box in existing_boxes if box.identity >= 0}
        pasted_boxes = self._copy_boxes(self._copied_boxes)
        cleared_ids = 0
        for box in pasted_boxes:
            self._snap_box_for_index(self._current_index, box)
            if box.identity >= 0 and box.identity in used_ids:
                box.identity = -1
                cleared_ids += 1
            elif box.identity >= 0:
                used_ids.add(box.identity)
        self._push_undo()
        existing_boxes.extend(pasted_boxes)
        self._mark_dirty(self._current_index)
        self._canvas.clear_warning_notices()
        self._goto_frame(self._current_index)
        count = len(pasted_boxes)
        if cleared_ids:
            self._status.showMessage(
                f"Pasted {count} boxes; {cleared_ids} IDs cleared because they were already used here.",
                5000,
            )
        else:
            self._status.showMessage("Box pasted." if count == 1 else f"{count} boxes pasted.", 3000)

    def _toggle_draw_box(self, enabled: bool):
        self._canvas.set_draw_mode(enabled)
        self._status.showMessage("Draw box mode enabled." if enabled else "Draw box mode disabled.", 3000)

    def _toggle_draw_box_shortcut(self):
        self._btn_draw_box.setChecked(not self._btn_draw_box.isChecked())

    def _on_box_drawn(self, box: Box):
        boxes = self._get_boxes(self._current_index)
        self._snap_box_for_index(self._current_index, box)
        self._push_undo()
        boxes.append(box)
        self._mark_dirty(self._current_index)
        self._btn_draw_box.setChecked(False)
        self._canvas.clear_warning_notices()
        self._goto_frame(self._current_index)
        self._status.showMessage("New box drawn.", 3000)

    def _on_box_changed(self, box: Box):
        self._snap_box_for_index(self._current_index, box)
        self._canvas.refresh_boxes()
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
        self._set_trajectory_button_checked(False)
        self._canvas.show_trajectory(trajectory, identity)

    def _mark_dirty(self, index: int):
        if index == self._current_index and self._suggested_detection_boxes:
            self._suggested_detection_boxes = []
            self._canvas.clear_reference_boxes()
            self._set_trajectory_button_checked(False)
            self._set_detector_controls_enabled()
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

    def _set_completed_state(self, idx: int, completed: bool):
        self._completed[idx] = completed
        self._timeline.set_completed(idx, completed)
        if idx == self._current_index:
            self._btn_complete.setChecked(completed)
            self._btn_complete.setText("Completed" if completed else "Mark Completed")

    def _toggle_completed(self):
        idx = self._current_index
        done = self._btn_complete.isChecked()
        if done:
            if not self._save_current():
                self._set_completed_state(idx, False)
                return
            if not self._completed.get(idx, False):
                self._set_completed_state(idx, False)
                self._status.showMessage("Frame saved, but sanity check failed. Not marked completed.", 5000)
                return
        else:
            self._set_completed_state(idx, False)
        self._reload_boxes_for_frame(idx)

    def _save_current(self):
        if not self._clip_path or not self._frame_paths:
            return False
        if self._pending_frame_index is not None:
            self._status.showMessage("Wait for the current frame to finish loading before saving.", 3000)
            return False
        return self._save_frame(self._current_index)

    def _save_frame(self, idx: int) -> bool:
        if not self._clip_path or not self._frame_paths:
            return False
        boxes = self._get_boxes(idx)
        self._snap_boxes_for_index(idx, boxes)
        gt_path = self._gt_path_for_index(idx)
        if not write_gt_labels(gt_path, boxes):
            QMessageBox.critical(
                self, "Save Error",
                f"Could not write labels to:\n{gt_path}\n\nCheck disk space and permissions.",
            )
            return False
        sanity = self._frame_sanity(idx)
        self._set_completed_state(idx, bool(sanity["passed"]))
        self._dirty_frames.discard(idx)
        if idx == self._current_index:
            self._canvas.clear_reference_boxes()
            self._lbl_frame.setText(f"Frame: {idx + 1} / {len(self._frame_paths)}")
            self._update_id_summary()
        self._update_save_state()
        if sanity["passed"]:
            self._status.showMessage(f"Saved and completed frame {idx + 1}: {gt_path.name}", 3000)
            self._queue_detector_training()
        else:
            self._status.showMessage(
                f"Saved frame {idx + 1}, but not completed: {'; '.join(sanity['messages'])}",
                6000,
            )
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

    def _wait_for_thread_pool(self, pool: QThreadPool, wait_ms: int) -> bool:
        try:
            return bool(pool.waitForDone(max(0, int(wait_ms))))
        except TypeError:
            pool.waitForDone()
            return True

    def _shutdown_background_loaders(self):
        self._set_task_message("Closing loaders...", active=True, color="#1d4ed8")
        QApplication.processEvents()

        self._frame_load_generation += 1
        self._pending_frame_index = None
        self._frame_loading_indexes.clear()
        self._frame_pool.clear()
        self._frame_image_cache.clear()

        if hasattr(self, "_timeline"):
            self._timeline.shutdown_loading(wait_ms=250)
        self._wait_for_thread_pool(self._frame_pool, 250)

        if hasattr(self, "_canvas"):
            self._canvas.release_resources()
        self._set_task_message("Closing")
        QApplication.processEvents()

    def closeEvent(self, event):
        if self._detector_training or self._detector_prediction:
            QMessageBox.information(
                self,
                "Detector Busy",
                "A detector job is still running. Please wait for it to finish before closing the app.",
            )
            event.ignore()
            return
        if self._handle_dirty_before_context_change("closing"):
            self._shutdown_background_loaders()
            event.accept()
        else:
            event.ignore()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat():
            if self._handle_global_shortcut(event):
                return True
        if event.type() == QEvent.KeyRelease and not event.isAutoRepeat():
            if self._event_matches_hotkey(event, "overlay_prev"):
                self._clear_adjacent_overlay(Qt.Key_Q)
                return True
            if self._event_matches_hotkey(event, "overlay_next"):
                self._clear_adjacent_overlay(Qt.Key_W)
                return True
            if self._event_matches_hotkey(event, "overlay_det"):
                self._clear_adjacent_overlay(Qt.Key_D)
                return True
        return super().eventFilter(obj, event)

    def _handle_global_shortcut(self, event) -> bool:
        if QApplication.activeModalWidget() is not None:
            return False
        focus = QApplication.focusWidget()
        if focus is not None and focus is not self and not self.isAncestorOf(focus):
            return False
        id_focused = isinstance(focus, QLineEdit)
        for action, key_value in (
            ("overlay_prev", Qt.Key_Q),
            ("overlay_next", Qt.Key_W),
            ("overlay_det", Qt.Key_D),
        ):
            if not id_focused and self._event_matches_hotkey(event, action):
                self._canvas.stop_flash()
                self._show_adjacent_overlay(key_value)
                return True

        callbacks = self._shortcut_callbacks()
        for action, callback in callbacks.items():
            if self._event_matches_hotkey(event, action):
                if self._pending_frame_index is not None and action not in (
                    "prev_frame", "next_frame", "goto_frame",
                    "ui_zoom_in", "ui_zoom_out", "ui_zoom_reset",
                ):
                    self._status.showMessage("Wait for the current frame to finish loading.", 3000)
                    return True
                if id_focused and not self._event_has_primary_modifier(event) and action not in ("delete_box",):
                    return False
                self._canvas.stop_flash()
                callback()
                return True
        return False

    def _event_has_primary_modifier(self, event) -> bool:
        return bool(event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier))

    def _event_matches_hotkey(self, event, action: str) -> bool:
        configured = self._hotkeys.get(action, "")
        if not configured:
            return False
        sequence = QKeySequence(configured)
        if sequence.isEmpty():
            return False
        modifiers = int(event.modifiers() & (
            Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier
        ))
        event_sequence = QKeySequence(modifiers | int(event.key()))
        if event_sequence.matches(sequence) == QKeySequence.ExactMatch:
            return True
        if configured.startswith("Ctrl+"):
            meta_sequence = QKeySequence("Meta+" + configured[5:])
            if event_sequence.matches(meta_sequence) == QKeySequence.ExactMatch:
                return True
        return False

    # ------------------------------------------------------------------ key events

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        self._canvas.stop_flash()
        id_focused = hasattr(self, "_id_input") and self._id_input.hasFocus()
        if self._event_matches_hotkey(event, "overlay_prev") and not id_focused:
            self._show_adjacent_overlay(Qt.Key_Q)
        elif self._event_matches_hotkey(event, "overlay_next") and not id_focused:
            self._show_adjacent_overlay(Qt.Key_W)
        elif self._event_matches_hotkey(event, "overlay_det") and not id_focused:
            self._show_adjacent_overlay(Qt.Key_D)
        elif self._event_matches_hotkey(event, "next_frame") and not id_focused:
            self._next_frame()
        elif self._event_matches_hotkey(event, "prev_frame") and not id_focused:
            self._prev_frame()
        elif self._event_matches_hotkey(event, "delete_box") and not id_focused:
            self._delete_selected_box()
        elif self._event_matches_hotkey(event, "draw_box") and not id_focused:
            self._toggle_draw_box_shortcut()
        elif event.key() == Qt.Key_Escape:
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
        if self._event_matches_hotkey(event, "overlay_prev"):
            self._clear_adjacent_overlay(Qt.Key_Q)
        elif self._event_matches_hotkey(event, "overlay_next"):
            self._clear_adjacent_overlay(Qt.Key_W)
        elif self._event_matches_hotkey(event, "overlay_det"):
            self._clear_adjacent_overlay(Qt.Key_D)
        else:
            super().keyReleaseEvent(event)
