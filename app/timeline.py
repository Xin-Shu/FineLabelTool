from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import QWidget, QScrollArea, QHBoxLayout, QLabel

THUMB_W = 80
THUMB_H = 55
MAX_THUMB_THREADS = 2
THUMB_STRIDE = THUMB_W + 10
THUMB_PREFETCH = 8


class _ThumbSignals(QObject):
    loaded = pyqtSignal(int, int, QImage)


class _ThumbLoadTask(QRunnable):
    def __init__(self, generation: int, index: int, path: Path, signals: _ThumbSignals):
        super().__init__()
        self.generation = generation
        self.index = index
        self.path = path
        self.signals = signals

    def run(self):
        image = QImage(str(self.path))
        if not image.isNull():
            image = image.scaled(
                THUMB_W,
                THUMB_H,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        try:
            self.signals.loaded.emit(self.generation, self.index, image)
        except RuntimeError:
            pass


class FrameThumb(QLabel):
    clicked = pyqtSignal(int)

    def __init__(
        self,
        index: int,
        path: Path,
        placeholder: QPixmap,
        parent=None,
    ):
        super().__init__(parent)
        self.index = index
        self._path = path
        self._src = placeholder
        self._loaded = False
        self._is_current = False
        self._is_completed = False
        self.setFixedSize(THUMB_W + 4, THUMB_H + 18)
        self.setCursor(Qt.PointingHandCursor)
        self._redraw()

    def set_image(self, image: QImage):
        if self._loaded:
            return
        if not image.isNull():
            self._src = QPixmap.fromImage(image)
        self._loaded = True
        self._redraw()

    def set_current(self, v: bool):
        if self._is_current != v:
            self._is_current = v
            self._redraw()

    def set_completed(self, v: bool):
        if self._is_completed != v:
            self._is_completed = v
            self._redraw()

    def _redraw(self):
        W, H = THUMB_W + 4, THUMB_H + 18
        canvas = QPixmap(W, H)
        canvas.fill(QColor(30, 30, 30))
        p = QPainter(canvas)

        border_color = QColor(0, 120, 255) if self._is_current else QColor(70, 70, 70)
        bw = 3 if self._is_current else 1
        p.setPen(QPen(border_color, bw))
        p.drawRect(bw // 2, bw // 2, W - bw, THUMB_H + 2)

        ox = (THUMB_W - self._src.width()) // 2 + 2
        oy = (THUMB_H - self._src.height()) // 2 + 1
        p.drawPixmap(ox, oy, self._src)

        if self._is_completed:
            p.fillRect(2, 1, THUMB_W, THUMB_H, QColor(52, 199, 89, 95))
            p.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(QColor(14, 122, 44), 3)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawLine(W - 22, 15, W - 18, 20)
            p.drawLine(W - 18, 20, W - 10, 10)

        p.setPen(QPen(QColor(180, 180, 180)))
        font = QFont(self.font())
        font.setPointSize(max(7, font.pointSize()))
        p.setFont(font)
        p.drawText(2, THUMB_H + 14, str(self.index + 1))
        p.end()
        self.setPixmap(canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)


class TimelineWidget(QWidget):
    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(THUMB_H + 44)

        self._scroll = QScrollArea(self)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidgetResizable(False)
        self._scroll.setFixedHeight(THUMB_H + 40)
        self._scroll.setStyleSheet(
            "QScrollArea { border: none; background: #1f2328; }"
            "QScrollBar:horizontal { height: 10px; background: #2b3036; }"
            "QScrollBar::handle:horizontal { background: #5d6875; min-width: 24px; }"
        )

        self._container = QWidget()
        self._container.setStyleSheet("background: #1f2328;")
        self._layout = QHBoxLayout(self._container)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(8, 6, 8, 4)
        self._scroll.setWidget(self._container)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self._thumbs: List[FrameThumb] = []
        self._paths: List[Path] = []
        self._current = 0
        self._generation = 0
        self._loading_indexes: set[int] = set()
        self._signals = _ThumbSignals()
        self._signals.loaded.connect(self._on_thumb_loaded)
        self._pool = QThreadPool.globalInstance()
        self._pool.setMaxThreadCount(max(1, min(MAX_THUMB_THREADS, self._pool.maxThreadCount())))
        self._scroll.horizontalScrollBar().valueChanged.connect(
            lambda _: self._request_visible_thumbs()
        )

    def load_frames(self, paths: List[Path], eager_index: Optional[int] = None):
        self._generation += 1
        generation = self._generation
        self._paths = list(paths)
        self._loading_indexes.clear()
        for t in self._thumbs:
            t.deleteLater()
        self._thumbs.clear()
        while self._layout.count():
            self._layout.takeAt(0)

        placeholder = QPixmap(THUMB_W, THUMB_H)
        placeholder.fill(QColor(55, 55, 55))

        for i, path in enumerate(paths):
            t = FrameThumb(i, path, placeholder)
            t.clicked.connect(self.frame_clicked)
            self._layout.addWidget(t)
            self._thumbs.append(t)

        self._container.setFixedSize(
            len(paths) * THUMB_STRIDE + 16,
            THUMB_H + 28,
        )
        if eager_index is not None and 0 <= eager_index < len(self._thumbs):
            self._request_thumb(eager_index, generation)
        QTimer.singleShot(0, self._request_visible_thumbs)

    def _on_thumb_loaded(self, generation: int, index: int, image: QImage):
        self._loading_indexes.discard(index)
        if generation != self._generation:
            return
        if 0 <= index < len(self._thumbs):
            self._thumbs[index].set_image(image)

    def _request_thumb(self, index: int, generation: Optional[int] = None):
        if not 0 <= index < len(self._thumbs):
            return
        if self._thumbs[index]._loaded or index in self._loading_indexes:
            return
        generation = self._generation if generation is None else generation
        self._loading_indexes.add(index)
        self._pool.start(_ThumbLoadTask(generation, index, self._paths[index], self._signals))

    def _request_visible_thumbs(self):
        if not self._thumbs:
            return
        bar = self._scroll.horizontalScrollBar()
        first = max(0, bar.value() // THUMB_STRIDE - THUMB_PREFETCH)
        visible_count = max(1, self._scroll.viewport().width() // THUMB_STRIDE + 1)
        last = min(len(self._thumbs) - 1, first + visible_count + THUMB_PREFETCH * 2)
        for idx in range(first, last + 1):
            self._request_thumb(idx)

    def set_current(self, index: int):
        if 0 <= self._current < len(self._thumbs):
            self._thumbs[self._current].set_current(False)
        self._current = index
        if 0 <= index < len(self._thumbs):
            self._thumbs[index].set_current(True)
            self._center_on_index(index)
            QTimer.singleShot(0, lambda i=index: self._center_on_index(i))
            QTimer.singleShot(120, lambda i=index: self._center_on_index(i))
            for idx in range(max(0, index - THUMB_PREFETCH), min(len(self._thumbs), index + THUMB_PREFETCH + 1)):
                self._request_thumb(idx)
            QTimer.singleShot(0, self._request_visible_thumbs)

    def center_current(self):
        self._center_on_index(self._current)
        QTimer.singleShot(0, lambda: self._center_on_index(self._current))

    def _center_on_index(self, index: int):
        bar = self._scroll.horizontalScrollBar()
        target = index * THUMB_STRIDE + THUMB_STRIDE // 2 - self._scroll.viewport().width() // 2
        target = max(bar.minimum(), min(bar.maximum(), target))
        bar.setValue(target)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.center_current)

    def set_completed(self, index: int, completed: bool):
        if 0 <= index < len(self._thumbs):
            self._thumbs[index].set_completed(completed)
