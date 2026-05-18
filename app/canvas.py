from __future__ import annotations
import os
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QPixmap,
                          QFont, QCursor, QPainterPath, QFontDatabase)
from PyQt5.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsItem,
                              QGraphicsPathItem, QGraphicsPixmapItem,
                              QGraphicsRectItem, QLabel)

from colors import get_color
from label_io import Box, snap_box_to_pixel_grid, snap_boxes_to_pixel_grid

HANDLE_SIZE = 9   # visible pixels
BOX_STROKE_PX = 1.45
SELECTED_BOX_STROKE_PX = 2.0
REFERENCE_BOX_STROKE_PX = 1.25
HIGHLIGHT_STROKE_PX = 2.6
HANDLE_STROKE_PX = 0.8
DRAW_BOX_STROKE_PX = 1.45
TRAJECTORY_STROKE_PX = 1.25
_H_TL, _H_TM, _H_TR = 0, 1, 2
_H_ML, _H_MR        = 3, 4
_H_BL, _H_BM, _H_BR = 5, 6, 7

_CURSOR_MAP = {
    _H_TL: Qt.SizeFDiagCursor, _H_BR: Qt.SizeFDiagCursor,
    _H_TR: Qt.SizeBDiagCursor, _H_BL: Qt.SizeBDiagCursor,
    _H_TM: Qt.SizeVerCursor,   _H_BM: Qt.SizeVerCursor,
    _H_ML: Qt.SizeHorCursor,   _H_MR: Qt.SizeHorCursor,
}


class BoxItem(QGraphicsItem):
    def __init__(
        self,
        box: Box,
        img_w: int,
        img_h: int,
        *,
        reference_label: Optional[str] = None,
        on_change_started=None,
        on_changed=None,
    ):
        super().__init__()
        self.box = box
        self.img_w = img_w
        self.img_h = img_h
        self.reference_label = reference_label
        self._on_change_started = on_change_started
        self._on_changed = on_changed
        self.position_locked = False
        self.size_locked = False
        self._active_handle: Optional[int] = None
        self._drag_start: Optional[QPointF] = None
        self._orig: Optional[tuple] = None   # (xc, yc, w, h) at drag start
        self.highlighted: bool = False

        if self.is_reference:
            self.setAcceptedMouseButtons(Qt.NoButton)
            self.setZValue(2)
        else:
            self.setFlags(QGraphicsItem.ItemIsSelectable)
            self.setAcceptHoverEvents(True)
            self.setZValue(self._current_box_z())

    @property
    def is_reference(self) -> bool:
        return self.reference_label is not None

    def _current_box_z(self) -> int:
        return 4 if self.box.identity >= 0 else 3

    def _view_scale(self) -> float:
        if self.scene() and self.scene().views():
            transform = self.scene().views()[0].transform()
            return max(abs(transform.m11()), abs(transform.m22()), 1e-6)
        return 1.0

    def _handle_size_scene(self) -> float:
        return HANDLE_SIZE / self._view_scale()

    def _scene_units_for_view_pixels(self, pixels: float) -> float:
        return max(pixels / self._view_scale(), 0.05)

    def _adaptive_pen(self, color, pixels: float, style=Qt.SolidLine) -> QPen:
        pen = QPen(color)
        pen.setWidthF(self._scene_units_for_view_pixels(pixels))
        pen.setStyle(style)
        pen.setCosmetic(False)
        return pen

    def set_geometry_locks(self, position_locked: bool, size_locked: bool):
        self.position_locked = position_locked
        self.size_locked = size_locked
        self.setCursor(QCursor(Qt.ArrowCursor))
        self.update()

    # ------------------------------------------------------------------ geom

    def _pixel_rect(self) -> QRectF:
        b = self.box
        x = (b.x_center - b.width  / 2) * self.img_w
        y = (b.y_center - b.height / 2) * self.img_h
        return self._snap_rect_to_pixel_grid(QRectF(x, y, b.width * self.img_w, b.height * self.img_h))

    def _snap_rect_to_pixel_grid(self, rect: QRectF) -> QRectF:
        rect = rect.normalized()
        left = max(0, min(self.img_w - 1, int(rect.left() + 0.5)))
        top = max(0, min(self.img_h - 1, int(rect.top() + 0.5)))
        right = max(left + 1, min(self.img_w, int(rect.right() + 0.5)))
        bottom = max(top + 1, min(self.img_h, int(rect.bottom() + 0.5)))
        return QRectF(left, top, right - left, bottom - top)

    def _handle_rects(self) -> dict:
        r = self._pixel_rect()
        handle_size = self._handle_size_scene()
        s = handle_size / 2
        cx, cy = r.center().x(), r.center().y()
        return {
            _H_TL: QRectF(r.left()-s,  r.top()-s,    handle_size, handle_size),
            _H_TM: QRectF(cx-s,        r.top()-s,    handle_size, handle_size),
            _H_TR: QRectF(r.right()-s, r.top()-s,    handle_size, handle_size),
            _H_ML: QRectF(r.left()-s,  cy-s,         handle_size, handle_size),
            _H_MR: QRectF(r.right()-s, cy-s,         handle_size, handle_size),
            _H_BL: QRectF(r.left()-s,  r.bottom()-s, handle_size, handle_size),
            _H_BM: QRectF(cx-s,        r.bottom()-s, handle_size, handle_size),
            _H_BR: QRectF(r.right()-s, r.bottom()-s, handle_size, handle_size),
        }

    def _handle_at(self, pos: QPointF) -> Optional[int]:
        for h, rect in self._handle_rects().items():
            if rect.contains(pos):
                return h
        return None

    def boundingRect(self) -> QRectF:
        pad = self._handle_size_scene() + 2 / self._view_scale()
        return self._pixel_rect().adjusted(-pad, -pad, pad, pad)

    def shape(self):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    # ------------------------------------------------------------------ paint

    def paint(self, painter: QPainter, option, widget=None):
        r = self._pixel_rect()
        color = get_color(self.box.identity)
        selected = self.isSelected()
        if not self.is_reference:
            self.setZValue(self._current_box_z())

        # Box outline. Width is kept in view pixels so it does not grow with zoom.
        stroke_px = SELECTED_BOX_STROKE_PX if selected else BOX_STROKE_PX
        style = Qt.SolidLine
        if self.is_reference:
            stroke_px = REFERENCE_BOX_STROKE_PX
            style = Qt.DashLine
        pen = self._adaptive_pen(color, stroke_px, style)
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.NoBrush))
        painter.drawRect(r)

        # Identity label, drawn inside the top-right corner without a filled badge.
        if self.box.identity >= 0:
            label = str(self.box.identity)
            font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            font.setPointSize(11)
            font.setBold(True)
            if self.is_reference:
                font.setItalic(True)
            painter.setFont(font)
            fm = painter.fontMetrics()
            margin = 4
            tw = fm.boundingRect(label).width()
            th = fm.height()
            label_rect = QRectF(
                r.right() - tw - margin,
                r.top() + margin,
                tw,
                th,
            )
            painter.setPen(QPen(color, 1))
            painter.drawText(label_rect, Qt.AlignRight | Qt.AlignTop, label)

        # Conflict-highlight ring (drawn outside the box rect)
        if self.highlighted and not self.is_reference:
            pad = self._scene_units_for_view_pixels(3)
            h_pen = self._adaptive_pen(QColor(255, 100, 0), HIGHLIGHT_STROKE_PX, Qt.DashLine)
            painter.setPen(h_pen)
            painter.setBrush(QBrush(Qt.NoBrush))
            painter.drawRect(r.adjusted(-pad, -pad, pad, pad))

        # Resize handles (only when selected)
        if selected and not self.size_locked:
            painter.setPen(self._adaptive_pen(Qt.white, HANDLE_STROKE_PX))
            painter.setBrush(QBrush(color))
            for hr in self._handle_rects().values():
                painter.drawRect(hr)

    # ------------------------------------------------------------------ hover

    def hoverMoveEvent(self, event):
        if self.position_locked and self.size_locked:
            self.setCursor(QCursor(Qt.ArrowCursor))
            super().hoverMoveEvent(event)
            return
        h = self._handle_at(event.pos())
        if h is not None and not self.size_locked:
            self.setCursor(QCursor(_CURSOR_MAP.get(h, Qt.ArrowCursor)))
        elif self._pixel_rect().contains(event.pos()) and not self.position_locked:
            self.setCursor(QCursor(Qt.SizeAllCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(QCursor(Qt.ArrowCursor))
        super().hoverLeaveEvent(event)

    # ------------------------------------------------------------------ drag

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            handle = self._handle_at(event.pos())
            wants_resize = handle is not None
            wants_move = handle is None and self._pixel_rect().contains(event.pos())
            if (wants_resize and self.size_locked) or (wants_move and self.position_locked):
                super().mousePressEvent(event)
                return
            self._active_handle = handle
            self._drag_start = event.scenePos()
            b = self.box
            self._orig = (b.x_center, b.y_center, b.width, b.height)
            if self._on_change_started:
                self._on_change_started(self.box)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return

        delta = event.scenePos() - self._drag_start
        xc0, yc0, w0, h0 = self._orig

        # Convert orig box to pixel rect
        x0 = (xc0 - w0 / 2) * self.img_w
        y0 = (yc0 - h0 / 2) * self.img_h
        w0p = w0 * self.img_w
        h0p = h0 * self.img_h
        r = QRectF(x0, y0, w0p, h0p)

        dx, dy = delta.x(), delta.y()
        h = self._active_handle

        if h is None:
            if self.position_locked:
                super().mouseMoveEvent(event)
                return
            r.translate(dx, dy)
        else:
            if self.size_locked:
                super().mouseMoveEvent(event)
                return
            if h in (_H_TL, _H_TM, _H_TR):
                r.setTop(r.top() + dy)
            if h in (_H_BL, _H_BM, _H_BR):
                r.setBottom(r.bottom() + dy)
            if h in (_H_TL, _H_ML, _H_BL):
                r.setLeft(r.left() + dx)
            if h in (_H_TR, _H_MR, _H_BR):
                r.setRight(r.right() + dx)
            r = r.normalized()

        # Clamp to image
        img_rect = QRectF(0, 0, self.img_w, self.img_h)
        r = self._snap_rect_to_pixel_grid(r.intersected(img_rect))
        if r.width() < 2 or r.height() < 2:
            return

        self.prepareGeometryChange()
        b = self.box
        b.x_center = (r.left() + r.width() / 2) / self.img_w
        b.y_center = (r.top() + r.height() / 2) / self.img_h
        b.width = r.width() / self.img_w
        b.height = r.height() / self.img_h
        self.update()

    def mouseReleaseEvent(self, event):
        changed = False
        if self._orig is not None:
            b = self.box
            changed = (b.x_center, b.y_center, b.width, b.height) != self._orig
        self._drag_start = None
        self._active_handle = None
        self._orig = None
        if changed and self._on_changed:
            self._on_changed(self.box)
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------

class ImageCanvas(QGraphicsView):
    box_selected   = pyqtSignal(object)   # emits Box
    box_deselected = pyqtSignal()
    box_change_started = pyqtSignal(object)
    box_changed    = pyqtSignal(object)
    box_drawn      = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(35, 35, 35)))
        self._try_enable_opengl_viewport()
        self._overlay_notices: Dict[str, QLabel] = {}
        self._warning_notices: Dict[str, QLabel] = {}
        self._minimap = QLabel(self)
        self._minimap.setStyleSheet(
            "background: rgba(17, 24, 39, 190); border: 1px solid #f8fafc;"
        )
        self._minimap.setFixedSize(190, 110)
        self._minimap.hide()

        self._box_items: List[BoxItem] = []
        self._reference_items: List[BoxItem] = []
        self._trajectory_items: List[QGraphicsItem] = []
        self._frame_pixmap: Optional[QPixmap] = None
        self._minimap_base: Optional[QPixmap] = None
        self._minimap_base_offset = QPointF(0, 0)
        self._img_w = 1
        self._img_h = 1
        self._boxes_locked = False
        self._position_locked = False
        self._size_locked = False
        self._panning = False
        self._pan_start = None
        self._pan_h0 = 0
        self._pan_v0 = 0
        self._overlay_active = False
        self._draw_mode = False
        self._draw_start: Optional[QPointF] = None
        self._draw_item: Optional[QGraphicsRectItem] = None
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(140)
        self._flash_timer.timeout.connect(self._advance_flash)
        self._flash_item: Optional[BoxItem] = None
        self._flash_remaining = 0
        self._flash_restore_highlight = False

        self.horizontalScrollBar().valueChanged.connect(self._update_minimap)
        self.verticalScrollBar().valueChanged.connect(self._update_minimap)

    def _try_enable_opengl_viewport(self):
        if os.environ.get("APP_LABEL_USE_OPENGL") != "1":
            return
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        try:
            from PyQt5.QtWidgets import QOpenGLWidget
            self.setViewport(QOpenGLWidget())
        except Exception:
            pass

    # ------------------------------------------------------------------ public

    def clear_selection(self):
        self.stop_flash()
        self._scene.clearSelection()
        self.box_deselected.emit()

    def highlight_box(self, box: Box):
        self.stop_flash()
        self.clear_highlight()
        for item in self._box_items:
            if item.box is box:
                item.highlighted = True
                item.update()
                break

    def clear_highlight(self):
        for item in self._box_items:
            if item.highlighted:
                item.highlighted = False
                item.update()

    def _item_for_box(self, box: Box) -> Optional[BoxItem]:
        for item in self._box_items:
            if item.box is box:
                return item
        return None

    def _snap_rect_to_pixel_grid(self, rect: QRectF) -> QRectF:
        rect = rect.normalized()
        left = max(0, min(self._img_w - 1, int(rect.left() + 0.5)))
        top = max(0, min(self._img_h - 1, int(rect.top() + 0.5)))
        right = max(left + 1, min(self._img_w, int(rect.right() + 0.5)))
        bottom = max(top + 1, min(self._img_h, int(rect.bottom() + 0.5)))
        return QRectF(left, top, right - left, bottom - top)

    def load_frame(self, pixmap: QPixmap, boxes: List[Box], keep_zoom: bool = False):
        old_transform = self.transform() if keep_zoom else None
        old_center = self.mapToScene(self.viewport().rect().center()) if keep_zoom else None

        self.stop_flash()
        self._scene.clear()
        self._box_items.clear()
        self._reference_items.clear()
        self._trajectory_items.clear()
        self._draw_item = None
        self._draw_start = None
        self._panning = False
        self._pan_start = None
        self.setCursor(QCursor(Qt.ArrowCursor))

        self._img_w = pixmap.width()
        self._img_h = pixmap.height()
        self._frame_pixmap = pixmap
        self._minimap_base = None
        snap_boxes_to_pixel_grid(boxes, self._img_w, self._img_h)

        self._scene.addItem(QGraphicsPixmapItem(pixmap))
        for box in boxes:
            item = BoxItem(
                box,
                self._img_w,
                self._img_h,
                on_change_started=self.box_change_started.emit,
                on_changed=self.box_changed.emit,
            )
            item.set_geometry_locks(self._position_locked, self._size_locked)
            self._scene.addItem(item)
            self._box_items.append(item)

        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)

        if keep_zoom and old_transform is not None:
            self.setTransform(old_transform)
            self.centerOn(old_center)
        else:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._update_minimap()

    def fit_view(self):
        if self._overlay_active:
            return
        scene_rect = self._scene.sceneRect()
        if scene_rect.isEmpty() or self._frame_pixmap is None:
            return
        self.fitInView(scene_rect, Qt.KeepAspectRatio)
        self._update_minimap()

    def refresh_boxes(self):
        for item in self._box_items:
            item.prepareGeometryChange()
            item.update()

    def focus_box(self, box: Box, flashes: int = 3):
        item = self._item_for_box(box)
        if item is None:
            return False
        return self._focus_item(item, box, flashes=flashes, select=True)

    def focus_reference_box(self, box: Box, label: str = "prev", flashes: int = 3):
        self.clear_reference_boxes()
        self._overlay_active = True
        self._panning = False
        item = BoxItem(
            box,
            self._img_w,
            self._img_h,
            reference_label=label,
        )
        item.highlighted = True
        self._scene.addItem(item)
        self._reference_items.append(item)
        self.set_overlay_notice(f"Last seen: ID {box.identity}", notice_id="direct")
        return self._focus_item(item, box, flashes=flashes, select=False)

    def _focus_item(self, item: BoxItem, box: Box, flashes: int = 3, select: bool = True):
        self.stop_flash()
        if select:
            self._scene.clearSelection()
            item.setSelected(True)
            self.box_selected.emit(box)

        rect = item._pixel_rect()
        target = self._focus_rect(rect)
        if not target.isEmpty():
            self.fitInView(target, Qt.KeepAspectRatio)
            self.centerOn(rect.center())
            self._update_minimap()

        self._flash_item = item
        self._flash_restore_highlight = item.highlighted
        self._flash_remaining = max(1, flashes) * 2
        item.highlighted = False
        item.update()
        self._advance_flash()
        self._flash_timer.start()
        return True

    def _focus_rect(self, rect: QRectF) -> QRectF:
        scene_rect = self._scene.sceneRect()
        if scene_rect.isEmpty():
            return rect
        margin_x = max(rect.width() * 2.0, scene_rect.width() * 0.04, 20.0)
        margin_y = max(rect.height() * 2.0, scene_rect.height() * 0.04, 20.0)
        target = rect.adjusted(-margin_x, -margin_y, margin_x, margin_y)
        min_w = scene_rect.width() * 0.08
        min_h = scene_rect.height() * 0.08
        if target.width() < min_w:
            extra = (min_w - target.width()) / 2
            target.adjust(-extra, 0, extra, 0)
        if target.height() < min_h:
            extra = (min_h - target.height()) / 2
            target.adjust(0, -extra, 0, extra)
        return target.intersected(scene_rect)

    def stop_flash(self):
        if self._flash_timer.isActive():
            self._flash_timer.stop()
        if self._flash_item is not None:
            self._flash_item.highlighted = self._flash_restore_highlight
            self._flash_item.update()
        self._flash_item = None
        self._flash_remaining = 0
        self._flash_restore_highlight = False

    def _advance_flash(self):
        if self._flash_item is None:
            self._flash_timer.stop()
            return
        if self._flash_remaining <= 0:
            self.stop_flash()
            return
        self._flash_item.highlighted = not self._flash_item.highlighted
        self._flash_item.update()
        self._flash_remaining -= 1

    def set_boxes_locked(self, locked: bool):
        self.set_geometry_locks(locked, locked)

    def set_geometry_locks(self, position_locked: bool, size_locked: bool):
        self._position_locked = position_locked
        self._size_locked = size_locked
        self._boxes_locked = position_locked and size_locked
        for item in self._box_items:
            item.set_geometry_locks(position_locked, size_locked)

    def set_current_boxes_visible(self, visible: bool):
        for item in self._box_items:
            item.setOpacity(1.0 if visible else 0.0)
            item.setEnabled(visible)

    def is_interacting(self) -> bool:
        if self._panning or self._draw_mode or self._draw_item is not None:
            return True
        return any(item._drag_start is not None for item in self._box_items)

    def show_reference_boxes(self, boxes: List[Box], label: str):
        self.clear_reference_boxes()
        self._overlay_active = True
        self._panning = False
        snap_boxes_to_pixel_grid(boxes, self._img_w, self._img_h)
        for box in boxes:
            item = BoxItem(
                box,
                self._img_w,
                self._img_h,
                reference_label=label,
            )
            self._scene.addItem(item)
            self._reference_items.append(item)

    def show_trajectory(self, boxes: List[Box], identity: int):
        self.clear_reference_boxes()
        if not boxes:
            return
        self._overlay_active = True
        self._panning = False
        snap_boxes_to_pixel_grid(boxes, self._img_w, self._img_h)
        color = get_color(identity)
        centers = []
        for box in boxes:
            item = BoxItem(
                box,
                self._img_w,
                self._img_h,
                reference_label="track",
            )
            self._scene.addItem(item)
            self._reference_items.append(item)
            centers.append(QPointF(box.x_center * self._img_w, box.y_center * self._img_h))

        if len(centers) >= 2:
            path = QPainterPath(centers[0])
            for point in centers[1:]:
                path.lineTo(point)
            line = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setWidthF(TRAJECTORY_STROKE_PX)
            pen.setStyle(Qt.DotLine)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setZValue(2.5)
            self._scene.addItem(line)
            self._trajectory_items.append(line)
        self.set_overlay_notice(f"Trajectory: ID {identity}", notice_id="trajectory")

    def _notice_style(self, kind: str) -> str:
        if kind == "warning":
            return (
                "background: #dc2626; color: #ffffff; border: 1px solid #7f1d1d; "
                "border-radius: 6px; padding: 8px 12px; font-weight: 800; "
                "font-size: 15px;"
            )
        return (
            "background: #ffd166; color: #111827; border: 1px solid #9a6700; "
            "border-radius: 6px; padding: 8px 12px; font-weight: 800; "
            "font-size: 15px;"
        )

    def set_overlay_notice(self, text: str, notice_id: str = "overlay"):
        self._set_notice(text, "overlay", notice_id)

    def set_warning_notice(self, text: str, notice_id: str = "warning"):
        self._set_notice(text, "warning", notice_id)

    def _notice_map(self, kind: str) -> Dict[str, QLabel]:
        return self._warning_notices if kind == "warning" else self._overlay_notices

    def _set_notice(self, text: str, kind: str, notice_id: str):
        notices = self._notice_map(kind)
        label = notices.get(notice_id)
        if label is None:
            label = QLabel(self.viewport())
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            label.hide()
            notices[notice_id] = label
        if text:
            label.setStyleSheet(self._notice_style(kind))
            label.setText(text)
            label.adjustSize()
            label.show()
            label.raise_()
        else:
            label.hide()
        self._position_notice_stack(kind)

    def clear_overlay_notices(self):
        self._clear_notice_stack("overlay")

    def clear_warning_notices(self):
        self._clear_notice_stack("warning")

    def _clear_notice_stack(self, kind: str):
        for label in self._notice_map(kind).values():
            label.hide()
        self._position_notice_stack(kind)

    def clear_reference_boxes(self):
        for item in self._reference_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._reference_items.clear()
        for item in self._trajectory_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._trajectory_items.clear()
        self._overlay_active = False
        self.clear_overlay_notices()

    def set_draw_mode(self, enabled: bool):
        self._draw_mode = enabled
        self.setCursor(QCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor))
        if not enabled:
            self._draw_start = None
            if self._draw_item is not None:
                self._scene.removeItem(self._draw_item)
                self._draw_item = None

    def get_selected_box(self) -> Optional[Box]:
        for item in self._box_items:
            if item.isSelected():
                return item.box
        return None

    # ------------------------------------------------------------------ events

    def wheelEvent(self, event):
        self.stop_flash()
        if self._overlay_active:
            event.ignore()
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._update_minimap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_notice_stack("overlay")
        self._position_notice_stack("warning")
        self._position_minimap()
        self._update_minimap()

    def _position_notice_stack(self, kind: str):
        visible = [label for label in self._notice_map(kind).values() if label.isVisible()]
        if not visible:
            return
        margin = 12
        gap = 8
        y = margin
        for label in visible:
            if kind == "warning":
                x = self.viewport().width() - label.width() - margin
                label.move(max(margin, x), y)
            else:
                label.move(margin, y)
            y += label.height() + gap

    def _position_minimap(self):
        margin = 12
        self._minimap.move(
            self.viewport().width() - self._minimap.width() - margin,
            self.viewport().height() - self._minimap.height() - margin,
        )

    def _update_minimap(self):
        if self._frame_pixmap is None or self._scene.sceneRect().isEmpty():
            self._minimap.hide()
            return

        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        scene_rect = self._scene.sceneRect()
        if visible.width() >= scene_rect.width() * 0.98 and visible.height() >= scene_rect.height() * 0.98:
            self._minimap.hide()
            return

        if self._minimap_base is None:
            base = QPixmap(self._minimap.size())
            base.fill(QColor(17, 24, 39, 210))
            scaled = self._frame_pixmap.scaled(
                self._minimap.width() - 12,
                self._minimap.height() - 12,
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
            if scaled.isNull():
                self._minimap.hide()
                return
            ox = (self._minimap.width() - scaled.width()) // 2
            oy = (self._minimap.height() - scaled.height()) // 2
            painter = QPainter(base)
            try:
                painter.drawPixmap(ox, oy, scaled)
            finally:
                painter.end()
            self._minimap_base = base
            self._minimap_base_offset = QPointF(ox, oy)

        canvas = QPixmap(self._minimap_base)
        scaled_w = self._minimap_base.width() - 12
        scaled_h = self._minimap_base.height() - 12
        if scaled_w <= 0 or scaled_h <= 0:
            self._minimap.hide()
            return
        ox = self._minimap_base_offset.x()
        oy = self._minimap_base_offset.y()
        draw_w = self._minimap_base.width() - 2 * ox
        draw_h = self._minimap_base.height() - 2 * oy

        sx = draw_w / scene_rect.width()
        sy = draw_h / scene_rect.height()
        vr = QRectF(
            ox + visible.left() * sx,
            oy + visible.top() * sy,
            visible.width() * sx,
            visible.height() * sy,
        ).intersected(QRectF(ox, oy, draw_w, draw_h))
        painter = QPainter(canvas)
        try:
            painter.setPen(QPen(QColor(255, 209, 102), 2))
            painter.setBrush(QBrush(Qt.NoBrush))
            painter.drawRect(vr)
        finally:
            painter.end()

        self._minimap.setPixmap(canvas)
        self._position_minimap()
        self._minimap.show()
        self._minimap.raise_()

    def mousePressEvent(self, event):
        self.stop_flash()
        if self._draw_mode and event.button() == Qt.LeftButton and self._frame_pixmap is not None:
            self._draw_start = self.mapToScene(event.pos())
            self._draw_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
            pen = QPen(QColor(255, 209, 102))
            pen.setWidthF(DRAW_BOX_STROKE_PX)
            pen.setStyle(Qt.DashLine)
            pen.setCosmetic(True)
            self._draw_item.setPen(pen)
            self._draw_item.setBrush(QBrush(Qt.NoBrush))
            self._draw_item.setZValue(6)
            self._scene.addItem(self._draw_item)
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self._pan_h0 = self.horizontalScrollBar().value()
            self._pan_v0 = self.verticalScrollBar().value()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        item = self.itemAt(event.pos())
        if isinstance(item, BoxItem) and not item.is_reference:
            selected = [box_item for box_item in self._box_items if box_item.isSelected()]
            if selected and item not in selected:
                event.accept()
                return
            self.box_selected.emit(item.box)
        else:
            self._scene.clearSelection()
            self.box_deselected.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_mode and self._draw_item is not None and self._draw_start is not None:
            current = self.mapToScene(event.pos())
            img_rect = QRectF(0, 0, self._img_w, self._img_h)
            rect = self._snap_rect_to_pixel_grid(
                QRectF(self._draw_start, current).normalized().intersected(img_rect)
            )
            self._draw_item.setRect(rect)
            event.accept()
            return
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self._pan_h0 - delta.x())
            self.verticalScrollBar().setValue(self._pan_v0 - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton and self._draw_item is not None:
            rect = self._snap_rect_to_pixel_grid(self._draw_item.rect().normalized())
            self._scene.removeItem(self._draw_item)
            self._draw_item = None
            self._draw_start = None
            if rect.width() >= 2 and rect.height() >= 2:
                box = Box(
                    x_center=(rect.left() + rect.width() / 2) / self._img_w,
                    y_center=(rect.top() + rect.height() / 2) / self._img_h,
                    width=rect.width() / self._img_w,
                    height=rect.height() / self._img_h,
                    confidence=1.0,
                    class_id=0,
                    identity=-1,
                )
                snap_box_to_pixel_grid(box, self._img_w, self._img_h)
                self.box_drawn.emit(box)
            event.accept()
            return
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(QCursor(Qt.ArrowCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)
