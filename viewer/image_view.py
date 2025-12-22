"""QGraphicsView-based image viewer with overlay and window/level support."""
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPathItem
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPen, QColor
from PyQt5.QtCore import Qt
import numpy as np
from skimage import measure


class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pix_item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.current_image = None  # numpy 2D
        self.window = None
        self.level = None
        self.overlays = []
        self.slice_index = 0
        self.mouse_slice_callback = None
        self.contour_items = []
        self.setRenderHints(self.renderHints() | Qt.SmoothTransformation)
        
        # Window/Level adjustment
        self.wl_adjust_mode = False
        self.last_mouse_pos = None
        self.wl_sensitivity = 2.0
        self.cached_rgba = None
        self.first_image_loaded = False
        self.wl_adjust_mode = False
        self.last_mouse_pos = None
        self.wl_sensitivity = 2.0

    def display_image(self, img2d, window=None, level=None):
        self.current_image = img2d.astype(float)
        
        if self.window is None or self.level is None:
            if window is None or level is None:
                amin, amax = np.min(self.current_image), np.max(self.current_image)
                window = amax - amin if amax != amin else 1.0
                level = (amax + amin) / 2.0
            self.window = float(window)
            self.level = float(level)
        
        rgba = self._compose_rgba()
        self.cached_rgba = rgba.copy()
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.setSceneRect(0, 0, rgba.shape[1], rgba.shape[0])
        self._draw_contours()
        
        if not self.first_image_loaded:
            self.fitInView(self.pix_item, Qt.KeepAspectRatio)
            self.first_image_loaded = True
    
    def set_cached_rgba(self, rgba):
        self.cached_rgba = rgba
    
    def get_current_rgba(self):
        return self.cached_rgba.copy() if self.cached_rgba is not None else None
    
    def display_cached_image(self):
        if self.cached_rgba is None:
            return
        rgba = self.cached_rgba
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.setSceneRect(0, 0, rgba.shape[1], rgba.shape[0])
        self._draw_contours()

    def _compose_rgba(self):
        img = self.current_image
        lo = self.level - self.window / 2.0
        hi = self.level + self.window / 2.0
        mapped = (img - lo) / (hi - lo)
        mapped = np.clip(mapped, 0.0, 1.0)
        gray = (mapped * 255).astype('uint8')
        h, w = gray.shape
        rgba = np.zeros((h, w, 4), dtype='uint8')
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        rgba[..., 3] = 255
        for overlay_data in self.overlays:
            mask, color, alpha = overlay_data[0], overlay_data[1], overlay_data[2]
            display_mode = overlay_data[3] if len(overlay_data) > 3 else 'fill'
            
            if mask is None:
                continue
            if display_mode == 'contour':
                continue
                
            a = np.clip(alpha, 0.0, 1.0)
            color_arr = np.zeros((h, w, 3), dtype='uint8')
            color_arr[..., 0] = color[0]
            color_arr[..., 1] = color[1]
            color_arr[..., 2] = color[2]
            mask_bool = mask.astype(bool)
            inv_a = 1.0 - a
            for c in range(3):
                rgba[..., c][mask_bool] = (a * color_arr[..., c][mask_bool] + inv_a * rgba[..., c][mask_bool]).astype('uint8')
        return rgba

    def _draw_contours(self):
        for item in self.contour_items:
            self.scene().removeItem(item)
        self.contour_items = []
        
        if self.current_image is None:
            return
            
        h, w = self.current_image.shape
        
        for overlay_data in self.overlays:
            mask, color, alpha = overlay_data[0], overlay_data[1], overlay_data[2]
            display_mode = overlay_data[3] if len(overlay_data) > 3 else 'fill'
            contour_width = overlay_data[4] if len(overlay_data) > 4 else 2.0
            
            if mask is None:
                continue
            
            if display_mode not in ('contour', 'both'):
                continue
                
            try:
                contours = measure.find_contours(mask.astype(np.uint8), 0.5)
                
                for contour in contours:
                    if len(contour) < 2:
                        continue
                    
                    if len(contour) > 100:
                        step = max(1, len(contour) // 100)
                        contour = contour[::step]
                        
                    path = QPainterPath()
                    first_point = True
                    for point in contour:
                        y, x = point
                        if first_point:
                            path.moveTo(x, y)
                            first_point = False
                        else:
                            path.lineTo(x, y)
                    if len(contour) > 0:
                        y, x = contour[0]
                        path.lineTo(x, y)
                    
                    path_item = QGraphicsPathItem(path)
                    pen = QPen(QColor(color[0], color[1], color[2]))
                    pen.setWidthF(contour_width)
                    pen.setCosmetic(False)
                    pen.setCapStyle(Qt.RoundCap)
                    pen.setJoinStyle(Qt.RoundJoin)
                    path_item.setPen(pen)
                    
                    self.scene().addItem(path_item)
                    self.contour_items.append(path_item)
                    
            except Exception as e:
                print(f"Error drawing contours: {e}")

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            delta = event.angleDelta().y()
            if delta > 0:
                step = 1
            else:
                step = -1
            if self.mouse_slice_callback:
                self.mouse_slice_callback(step)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.wl_adjust_mode = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.CrossCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.wl_adjust_mode and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            window_delta = delta.x() * self.wl_sensitivity
            level_delta = -delta.y() * self.wl_sensitivity
            
            self.window = max(1.0, self.window + window_delta)
            self.level = self.level + level_delta
            
            if self.current_image is not None:
                self.cached_rgba = None
                rgba = self._compose_rgba()
                self.cached_rgba = rgba.copy()
                qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
                self.pix_item.setPixmap(QPixmap.fromImage(qimg))
            
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.wl_adjust_mode = False
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def add_overlay(self, mask, color=(255, 0, 0), alpha=0.4, display_mode='fill', contour_width=2.0):
        self.overlays.append((mask, color, alpha, display_mode, contour_width))

    def clear_overlays(self):
        self.overlays = []
        for item in self.contour_items:
            self.scene().removeItem(item)
        self.contour_items = []

    def set_overlays(self, overlays):
        self.overlays = overlays
