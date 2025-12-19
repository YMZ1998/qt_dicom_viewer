"""
QGraphicsView-based image viewer with overlay blending.

Supports: display 2D numpy image (grayscale), window/level mapping, simple zoom/pan,
mouse wheel emits slice change (caller handles bounds), overlay boolean masks with color/alpha.
Supports contour display mode with adjustable thickness.
"""
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
        self.overlays = []  # list of tuples (mask(y,x), (r,g,b), alpha, display_mode, contour_width)
        self.slice_index = 0
        self.mouse_slice_callback = None
        self.contour_items = []  # list of QGraphicsPathItem for contours
        # enable high quality transforms
        self.setRenderHints(self.renderHints() | Qt.SmoothTransformation)

    def display_image(self, img2d, window=None, level=None):
        """
        img2d: 2D numpy array (y,x)
        window/level optional; if None, auto compute
        """
        self.current_image = img2d.astype(float)
        if window is None or level is None:
            amin, amax = np.min(self.current_image), np.max(self.current_image)
            window = amax - amin if amax != amin else 1.0
            level = (amax + amin) / 2.0
        self.window = float(window)
        self.level = float(level)
        rgba = self._compose_rgba()
        qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.setSceneRect(0, 0, rgba.shape[1], rgba.shape[0])
        self._draw_contours()
        self.fitInView(self.pix_item, Qt.KeepAspectRatio)

    def _compose_rgba(self):
        """
        Map current_image to uint8 grayscale and blend overlays into RGBA uint8 array.
        Only blends fill mode overlays.
        """
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
        # blend overlays (simple alpha compositing) - only fill mode
        for overlay_data in self.overlays:
            mask, color, alpha = overlay_data[0], overlay_data[1], overlay_data[2]
            display_mode = overlay_data[3] if len(overlay_data) > 3 else 'fill'
            
            if mask is None:
                continue
            # Skip contour-only mode here (will be drawn as vectors)
            if display_mode == 'contour':
                continue
                
            # color: (r,g,b) 0-255, alpha: 0.0-1.0
            a = np.clip(alpha, 0.0, 1.0)
            # create color layers
            color_arr = np.zeros((h, w, 3), dtype='uint8')
            color_arr[..., 0] = color[0]
            color_arr[..., 1] = color[1]
            color_arr[..., 2] = color[2]
            mask_bool = mask.astype(bool)
            # perform composite for masked pixels
            inv_a = 1.0 - a
            for c in range(3):
                rgba[..., c][mask_bool] = (a * color_arr[..., c][mask_bool] + inv_a * rgba[..., c][mask_bool]).astype('uint8')
            # alpha channel remains 255 (opaque)
        return rgba

    def _draw_contours(self):
        """
        Draw contours for overlays with contour or both display mode.
        """
        # Clear existing contour items
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
            
            # Only draw contours for 'contour' or 'both' mode
            if display_mode not in ('contour', 'both'):
                continue
                
            # Find contours using skimage
            try:
                contours = measure.find_contours(mask.astype(np.uint8), 0.5)
                
                for contour in contours:
                    if len(contour) < 2:
                        continue
                        
                    # Create QPainterPath
                    path = QPainterPath()
                    # contour is (N, 2) array with (row, col) coordinates
                    # Qt uses (x, y) so we need to swap: col=x, row=y
                    first_point = True
                    for point in contour:
                        y, x = point
                        if first_point:
                            path.moveTo(x, y)
                            first_point = False
                        else:
                            path.lineTo(x, y)
                    # Close the path
                    if len(contour) > 0:
                        y, x = contour[0]
                        path.lineTo(x, y)
                    
                    # Create path item
                    path_item = QGraphicsPathItem(path)
                    pen = QPen(QColor(color[0], color[1], color[2]))
                    pen.setWidthF(contour_width)
                    pen.setCosmetic(False)  # Width in scene coordinates
                    path_item.setPen(pen)
                    
                    # Add to scene
                    self.scene().addItem(path_item)
                    self.contour_items.append(path_item)
                    
            except Exception as e:
                print(f"Error drawing contours: {e}")

    def wheelEvent(self, event):
        # treat wheel as slice navigation by default
        delta = event.angleDelta().y()
        if delta > 0:
            step = 1
        else:
            step = -1
        if self.mouse_slice_callback:
            self.mouse_slice_callback(step)

    def add_overlay(self, mask, color=(255, 0, 0), alpha=0.4, display_mode='fill', contour_width=2.0):
        """
        Add overlay mask (2D boolean array) with color tuple (r,g,b), alpha, display mode and contour width.
        display_mode: 'fill', 'contour', or 'both'
        contour_width: width of contour line in pixels
        """
        self.overlays.append((mask, color, alpha, display_mode, contour_width))

    def clear_overlays(self):
        self.overlays = []
        # Clear contour items
        for item in self.contour_items:
            self.scene().removeItem(item)
        self.contour_items = []

    def set_overlays(self, overlays):
        """
        Set overlays list to overlays: list of (mask, color, alpha, display_mode, contour_width)
        """
        self.overlays = overlays
