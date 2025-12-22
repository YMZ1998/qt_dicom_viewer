"""DICOM Viewer main window with series/ROI management and RTSTRUCT support."""
import os
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QPushButton, QLabel, QSlider, QListWidgetItem, QAction, QMenuBar,
    QComboBox, QDoubleSpinBox, QGroupBox, QCheckBox, QProgressDialog, QStatusBar,
    QToolBar, QMessageBox, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QKeySequence
from .dicom_io import group_dicom_series, read_series_as_volume, get_rescale_from_file, apply_rescale
from .image_view import ImageView
from .rtstruct import parse_rtstruct, contours_to_slice_masks


class SeriesLoaderThread(QThread):
    finished = pyqtSignal(object, object, object, object, float, float)
    error = pyqtSignal(str)
    
    def __init__(self, files):
        super().__init__()
        self.files = files
    
    def run(self):
        try:
            vol, origin, spacing, direction = read_series_as_volume(self.files)
            try:
                slope, intercept = get_rescale_from_file(self.files[0])
            except Exception:
                slope, intercept = 1.0, 0.0
            vol = apply_rescale(vol, slope=slope, intercept=intercept)
            self.finished.emit(vol, origin, spacing, direction, slope, intercept)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DICOM Viewer')
        self.resize(1400, 800)
        self.current_theme = 'dark'
        self._apply_stylesheet()
        self._build_ui()
        self.volume = None  # shape (z,y,x)
        self.origin = None
        self.spacing = None
        self.direction = None
        self.slice_index = 0
        self.roi_masks = {}  # roi_name -> {k: mask}
        self.current_series_files = []
        self.current_folder = None  # store current opened folder
        self.rtstruct_files = []  # list of available RTSTRUCT files
        self.current_rtstruct_index = 0  # current RTSTRUCT index
        self.display_mode = 'both'  # 'fill', 'contour', or 'both'
        self.contour_width = 1.0
        self.alpha_value = 0.0
        self.image_cache = {}
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update)
        self.pending_update = False
        self.loader_thread = None
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_PageUp), self, lambda: self.on_wheel_slice(5))
        QShortcut(QKeySequence(Qt.Key_PageDown), self, lambda: self.on_wheel_slice(-5))
        QShortcut(QKeySequence(Qt.Key_Up), self, lambda: self.on_wheel_slice(1))
        QShortcut(QKeySequence(Qt.Key_Down), self, lambda: self.on_wheel_slice(-1))
        QShortcut(QKeySequence(Qt.Key_Home), self, self.goto_first_slice)
        QShortcut(QKeySequence(Qt.Key_End), self, self.goto_last_slice)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_all_rois)
    
    def goto_first_slice(self):
        if self.volume is not None:
            self.slice_index = 0
            self.slider.setValue(self.slice_index)
    
    def goto_last_slice(self):
        if self.volume is not None:
            self.slice_index = self.volume.shape[0] - 1
            self.slider.setValue(self.slice_index)
    
    def toggle_all_rois(self):
        if self.roi_list.count() == 0:
            return
        any_checked = any(
            self.roi_list.item(i).checkState() == Qt.Checked 
            for i in range(self.roi_list.count())
        )
        new_state = Qt.Unchecked if any_checked else Qt.Checked
        for i in range(self.roi_list.count()):
            self.roi_list.item(i).setCheckState(new_state)
        self.update_viewer()
    
    def _apply_stylesheet(self):
        self.setStyleSheet("""
        QMainWindow { background-color: #2b2b2b; }
        QWidget { background-color: #2b2b2b; color: #e0e0e0; }
        QPushButton { background-color: #3c3c3c; border: 1px solid #555; border-radius: 4px; padding: 6px 12px; }
        QPushButton:hover { background-color: #4a4a4a; }
        QListWidget { background-color: #353535; border: 1px solid #555; }
        QListWidget::item:selected { background-color: #0d47a1; }
        QSlider::groove:horizontal { border: 1px solid #555; height: 8px; background: #3c3c3c; }
        QSlider::handle:horizontal { background: #0d47a1; width: 16px; margin: -4px 0; border-radius: 8px; }
        QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 8px; padding-top: 12px; }
        QToolBar { background-color: #353535; border: none; }
        QStatusBar { background-color: #353535; border-top: 1px solid #555; }
        """)
    
    def _build_ui(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')
        self._build_toolbar()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        left_w = QWidget()
        left = QVBoxLayout(left_w)
        self.open_btn = QPushButton('Open DICOM Folder')
        self.open_btn.clicked.connect(self.open_folder)
        left.addWidget(self.open_btn)
        left.addWidget(QLabel('Series:'))
        self.series_list = QListWidget()
        self.series_list.itemClicked.connect(self.on_series_selected)
        left.addWidget(self.series_list)
        left.addWidget(QLabel('ROIs:'))
        self.roi_list = QListWidget()
        self.roi_list.setSelectionMode(QListWidget.MultiSelection)
        self.roi_list.itemChanged.connect(self.on_roi_toggled)
        left.addWidget(self.roi_list)
        main_layout.addWidget(left_w, 1)
        center_w = QWidget()
        center = QVBoxLayout(center_w)
        self.viewer = ImageView()
        self.viewer.mouse_slice_callback = self.on_wheel_slice
        center.addWidget(self.viewer, 10)
        controls = QHBoxLayout()
        self.slice_label = QLabel('Slice: 0 / 0')
        controls.addWidget(self.slice_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        controls.addWidget(self.slider, 1)
        center.addLayout(controls)
        main_layout.addWidget(center_w, 6)
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        self.info_label = QLabel('No series loaded.')
        right.addWidget(self.info_label)
        rt_group = QGroupBox('RTStruct')
        rt_layout = QVBoxLayout()
        self.load_rt_btn = QPushButton('Load RTSTRUCT')
        self.load_rt_btn.clicked.connect(self.load_rt)
        rt_layout.addWidget(self.load_rt_btn)
        rt_selector_layout = QHBoxLayout()
        self.rt_prev_btn = QPushButton('<')
        self.rt_prev_btn.setMaximumWidth(40)
        self.rt_prev_btn.clicked.connect(self.load_previous_rtstruct)
        self.rt_prev_btn.setEnabled(False)
        rt_selector_layout.addWidget(self.rt_prev_btn)
        
        self.rt_label = QLabel('No RTStruct')
        self.rt_label.setAlignment(Qt.AlignCenter)
        rt_selector_layout.addWidget(self.rt_label, 1)
        
        self.rt_next_btn = QPushButton('>')
        self.rt_next_btn.setMaximumWidth(40)
        self.rt_next_btn.clicked.connect(self.load_next_rtstruct)
        self.rt_next_btn.setEnabled(False)
        rt_selector_layout.addWidget(self.rt_next_btn)
        
        rt_layout.addLayout(rt_selector_layout)
        rt_group.setLayout(rt_layout)
        right.addWidget(rt_group)
        display_group = QGroupBox('Display Settings')
        display_layout = QVBoxLayout()
        display_layout.addWidget(QLabel('Display Mode:'))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Fill', 'Contour', 'Both'])
        self.mode_combo.setCurrentText('Both')
        self.mode_combo.currentTextChanged.connect(self.on_display_mode_changed)
        display_layout.addWidget(self.mode_combo)
        display_layout.addWidget(QLabel('Contour Width:'))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setMinimum(0.5)
        self.width_spin.setMaximum(10.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(1.0)
        self.width_spin.valueChanged.connect(self.on_contour_width_changed)
        display_layout.addWidget(self.width_spin)
        display_layout.addWidget(QLabel('Fill Opacity:'))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setMinimum(0.0)
        self.alpha_spin.setMaximum(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(0.0)
        self.alpha_spin.valueChanged.connect(self.on_alpha_changed)
        display_layout.addWidget(self.alpha_spin)
        
        display_group.setLayout(display_layout)
        right.addWidget(display_group)
        right.addStretch()
        main_layout.addWidget(right_w, 1)
        menu = QMenuBar(self)
        file_menu = menu.addMenu('File')
        open_action = QAction('Open DICOM Folder', self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)
        load_rt_action = QAction('Load RTSTRUCT', self)
        load_rt_action.triggered.connect(self.load_rt)
        file_menu.addAction(load_rt_action)
        self.setMenuBar(menu)
    
    def _build_toolbar(self):
        toolbar = QToolBar('Main Toolbar')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        open_action = QAction('Open', self)
        open_action.triggered.connect(self.open_folder)
        toolbar.addAction(open_action)
        
        load_rt_action = QAction('Load RT', self)
        load_rt_action.triggered.connect(self.load_rt)
        toolbar.addAction(load_rt_action)
        
        toolbar.addSeparator()
        window_label = QLabel(' Window: ')
        toolbar.addWidget(window_label)
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setMinimum(1.0)
        self.window_spin.setMaximum(10000.0)
        self.window_spin.setValue(400.0)
        self.window_spin.setMaximumWidth(100)
        self.window_spin.valueChanged.connect(self.on_window_changed)
        toolbar.addWidget(self.window_spin)
        
        level_label = QLabel(' Level: ')
        toolbar.addWidget(level_label)
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setMinimum(-10000.0)
        self.level_spin.setMaximum(10000.0)
        self.level_spin.setValue(40.0)
        self.level_spin.setMaximumWidth(100)
        self.level_spin.valueChanged.connect(self.on_level_changed)
        toolbar.addWidget(self.level_spin)
        
        toolbar.addSeparator()
        self.theme_btn = QPushButton('☀ Light')
        self.theme_btn.clicked.connect(self.toggle_theme)
        toolbar.addWidget(self.theme_btn)
    
    def on_window_changed(self, value):
        if self.volume is not None and self.viewer.current_image is not None:
            self.viewer.window = value
            self.image_cache.clear()
            self.update_viewer()
    
    def on_level_changed(self, value):
        if self.volume is not None and self.viewer.current_image is not None:
            self.viewer.level = value
            self.image_cache.clear()
            self.update_viewer()
    
    def toggle_theme(self):
        if self.current_theme == 'dark':
            self.current_theme = 'light'
            self._apply_light_theme()
            self.theme_btn.setText('🌙 Dark')
        else:
            self.current_theme = 'dark'
            self._apply_stylesheet()
            self.theme_btn.setText('☀ Light')
    
    def _apply_light_theme(self):
        self.setStyleSheet("""
        QMainWindow { background-color: #f5f5f5; }
        QWidget { background-color: #f5f5f5; color: #212121; }
        QPushButton { background-color: #fff; border: 1px solid #bdbdbd; border-radius: 4px; padding: 6px 12px; }
        QPushButton:hover { background-color: #e3f2fd; }
        QListWidget { background-color: #fff; border: 1px solid #bdbdbd; }
        QListWidget::item:selected { background-color: #2196f3; color: #fff; }
        QSlider::groove:horizontal { border: 1px solid #bdbdbd; height: 8px; background: #e0e0e0; }
        QSlider::handle:horizontal { background: #2196f3; width: 16px; margin: -4px 0; border-radius: 8px; }
        QGroupBox { border: 1px solid #bdbdbd; border-radius: 4px; margin-top: 8px; padding-top: 12px; }
        QToolBar { background-color: #fafafa; border: none; }
        QStatusBar { background-color: #fafafa; border-top: 1px solid #bdbdbd; }
        """)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open DICOM Folder', os.getcwd())
        if not folder:
            return
        self.current_folder = folder
        groups = group_dicom_series(folder)
        self.series_list.clear()
        for sid, files in groups.items():
            if len(files) > 1:
                item = QListWidgetItem(f'{sid} ({len(files)} files)')
                item.setData(Qt.UserRole, files)
                self.series_list.addItem(item)

    def on_series_selected(self, item):
        files = item.data(Qt.UserRole)
        if not files:
            return
        try:
            vol, origin, spacing, direction = read_series_as_volume(files)
            try:
                slope, intercept = get_rescale_from_file(files[0])
            except Exception:
                slope, intercept = 1.0, 0.0
            vol = apply_rescale(vol, slope=slope, intercept=intercept)
            self.volume = vol
            self.origin = origin
            self.spacing = spacing
            self.direction = direction
            self.current_series_files = files
            z, y, x = vol.shape
            self.slice_index = max(0, z // 2)
            self.slider.setMaximum(z - 1)
            self.slider.setValue(self.slice_index)
            self.update_viewer()
            self.info_label.setText(f'Series loaded: {len(files)} slices\nSpacing: {spacing}\nOrigin: {origin}')
            self._auto_load_rtstruct()
        except Exception as e:
            self.info_label.setText(f'Failed to load series: {e}')

    def update_viewer(self):
        if not self.pending_update:
            self.pending_update = True
            self.update_timer.start(50)
    
    def _delayed_update(self):
        self.pending_update = False
        if self.volume is None:
            return
        z, y, x = self.volume.shape
        if self.slice_index < 0: self.slice_index = 0
        if self.slice_index >= z: self.slice_index = z - 1
        
        img2d = self.volume[self.slice_index]
        self.viewer.clear_overlays()
        overlays = []
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            if item.checkState() == Qt.Checked:
                roi_name = item.text()
                per_slice = self.roi_masks.get(roi_name, {})
                mask = per_slice.get(self.slice_index, None)
                if mask is not None:
                    color = self._color_for_name(roi_name)
                    overlays.append((mask, color, self.alpha_value, self.display_mode, self.contour_width))
        self.viewer.set_overlays(overlays)
        self.viewer.display_image(img2d)
        self.slice_label.setText(f'Slice: {self.slice_index+1} / {z}')
        if self.viewer.window is not None:
            self.window_spin.blockSignals(True)
            self.window_spin.setValue(self.viewer.window)
            self.window_spin.blockSignals(False)
        if self.viewer.level is not None:
            self.level_spin.blockSignals(True)
            self.level_spin.setValue(self.viewer.level)
            self.level_spin.blockSignals(False)

    def on_slider_changed(self, val):
        self.slice_index = int(val)
        self.update_viewer()

    def on_wheel_slice(self, step):
        if self.volume is None:
            return
        self.slice_index = max(0, min(self.volume.shape[0] - 1, self.slice_index + step))
        self.slider.setValue(self.slice_index)

    def load_rt(self):
        if self.volume is None:
            self.info_label.setText('Load a series first before loading RTSTRUCT.')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'Open RTSTRUCT', os.getcwd(), 'DICOM Files (*.dcm *.dicom);;All Files (*)')
        if not path:
            return
        try:
            rois = parse_rtstruct(path)
            masks = contours_to_slice_masks(rois, self.origin, self.spacing, self.direction, self.volume.shape)
            self.roi_masks = masks
            self.roi_list.clear()
            for rname in masks.keys():
                item = QListWidgetItem(rname)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.roi_list.addItem(item)
            self.update_viewer()
            self.info_label.setText(f'Loaded RTSTRUCT: {len(masks)} ROIs')
        except Exception as e:
            self.info_label.setText(f'Failed to load RTSTRUCT: {e}')

    def on_roi_toggled(self, item):
        self.update_viewer()

    def on_display_mode_changed(self, text):
        self.display_mode = text.lower()
        self.update_viewer()
    
    def on_contour_width_changed(self, value):
        self.contour_width = value
        self.update_viewer()
    
    def on_alpha_changed(self, value):
        self.alpha_value = value
        self.update_viewer()

    def _auto_load_rtstruct(self):
        if not self.current_folder:
            return
        self.rtstruct_files = self._find_all_rtstruct_files(self.current_folder)
        if self.rtstruct_files:
            self.current_rtstruct_index = 0
            self._load_rtstruct_by_index(0)
            self._update_rtstruct_controls()
        else:
            self.rt_label.setText('No RTStruct')
            self.rt_prev_btn.setEnabled(False)
            self.rt_next_btn.setEnabled(False)
    
    def _find_all_rtstruct_files(self, folder):
        import pydicom
        rtstruct_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.dcm', '.dicom')):
                    path = os.path.join(root, f)
                    try:
                        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                        modality = getattr(ds, 'Modality', None)
                        if modality == 'RTSTRUCT':
                            rtstruct_files.append(path)
                    except Exception:
                        continue
        return rtstruct_files
    
    def _load_rtstruct_by_index(self, index):
        if not self.rtstruct_files or index < 0 or index >= len(self.rtstruct_files):
            return
        
        rt_file = self.rtstruct_files[index]
        try:
            rois = parse_rtstruct(rt_file)
            masks = contours_to_slice_masks(rois, self.origin, self.spacing, self.direction, self.volume.shape)
            self.roi_masks = masks
            self.roi_list.clear()
            for rname in masks.keys():
                item = QListWidgetItem(rname)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.roi_list.addItem(item)
            self.update_viewer()
            current_text = self.info_label.text()
            rt_filename = os.path.basename(rt_file)
            self.info_label.setText(f'{current_text}\nRTStruct: {rt_filename} ({len(masks)} ROIs)')
        except Exception as e:
            self.info_label.setText(f'{self.info_label.text()}\nFailed to load RTStruct: {e}')
    
    def _update_rtstruct_controls(self):
        if not self.rtstruct_files:
            self.rt_label.setText('No RTStruct')
            self.rt_prev_btn.setEnabled(False)
            self.rt_next_btn.setEnabled(False)
        else:
            total = len(self.rtstruct_files)
            self.rt_label.setText(f'{self.current_rtstruct_index + 1} / {total}')
            self.rt_prev_btn.setEnabled(self.current_rtstruct_index > 0)
            self.rt_next_btn.setEnabled(self.current_rtstruct_index < total - 1)
    
    def load_previous_rtstruct(self):
        if self.current_rtstruct_index > 0:
            self.current_rtstruct_index -= 1
            self._load_rtstruct_by_index(self.current_rtstruct_index)
            self._update_rtstruct_controls()
    
    def load_next_rtstruct(self):
        if self.current_rtstruct_index < len(self.rtstruct_files) - 1:
            self.current_rtstruct_index += 1
            self._load_rtstruct_by_index(self.current_rtstruct_index)
            self._update_rtstruct_controls()

    def _color_for_name(self, name):
        h = sum(ord(c) for c in name) % 360
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h/360.0, 0.8, 0.9)
        return (int(r*255), int(g*255), int(b*255))
