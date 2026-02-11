import os
import sys
import threading

from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QListWidget, QListWidgetItem
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你已有的核心算法模块
from rtss_split_non_overlap import load_rois_from_rtss, build_conflict_graph, dsatur_coloring, split_into_sets


class ROIApp(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI 非重叠分组工具")
        self.resize(700, 600)

        # 初始化配置存储
        self.settings = QSettings("ROIApp", "RTSSSplitter")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # RTSTRUCT 文件选择
        self.rtss_layout = QHBoxLayout()
        self.rtss_label = QLabel("RTSTRUCT 文件:")
        self.rtss_path = QLineEdit()
        self.rtss_btn = QPushButton("选择文件")
        self.rtss_btn.clicked.connect(self.select_rtss_file)
        self.rtss_layout.addWidget(self.rtss_label)
        self.rtss_layout.addWidget(self.rtss_path)
        self.rtss_layout.addWidget(self.rtss_btn)
        self.layout.addLayout(self.rtss_layout)

        # 解析状态标签
        self.parse_status = QLabel("状态: 未选择文件")
        self.layout.addWidget(self.parse_status)

        # ROI 选择列表
        self.roi_list_label = QLabel("请选择要处理的 ROI (多选):")
        self.layout.addWidget(self.roi_list_label)

        self.roi_list_widget = QListWidget()
        self.roi_list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.layout.addWidget(self.roi_list_widget)

        # 运行按钮
        self.run_btn = QPushButton("开始处理")
        self.run_btn.clicked.connect(self.run_process)
        self.layout.addWidget(self.run_btn)

        # 日志显示
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.layout.addWidget(self.log)

        # 绑定信号，实现线程安全日志
        self.log_signal.connect(self.log_message)

        # 存储解析的ROI数据
        self.rois_data = None

        # 当文件路径改变时自动解析 ROI
        self.rtss_path.textChanged.connect(self.auto_parse_rois)

    # -----------------------------
    # 日志函数
    # -----------------------------
    def log_message(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    # -----------------------------
    # 选择 RTSTRUCT 文件
    # -----------------------------
    def select_rtss_file(self):
        last_dir = self.settings.value("last_directory", "")
        path, _ = QFileDialog.getOpenFileName(self, "选择 RTSTRUCT 文件", last_dir, "DICOM Files (*.dcm)")
        if path:
            self.rtss_path.setText(path)
            # 保存选择的目录
            self.settings.setValue("last_directory", os.path.dirname(path))

    # -----------------------------
    # 自动解析 ROI
    # -----------------------------
    def auto_parse_rois(self):
        rtss_file = self.rtss_path.text().strip()
        if not rtss_file:
            self.parse_status.setText("状态: 未选择文件")
            self.roi_list_widget.clear()
            self.rois_data = None
            return

        try:
            self.log_signal.emit(f"[INFO] 开始解析 RTSTRUCT 文件: {rtss_file}")
            # 解析所有ROI，不进行过滤
            rois = load_rois_from_rtss(rtss_file, z_tol=1e-3, roi_names_filter=None)

            if len(rois) == 0:
                self.log_signal.emit("[ERROR] 没有找到任何有效的 ROI")
                self.parse_status.setText("状态: 未找到有效ROI")
                self.roi_list_widget.clear()
                self.rois_data = None
                QMessageBox.warning(self, "解析失败", "没有找到任何有效的 ROI")
                return

            self.log_signal.emit(f"[INFO] 成功加载 {len(rois)} 个 ROI")

            # 清空现有的列表项
            self.roi_list_widget.clear()

            # 添加ROI到列表中
            for roi in rois:
                item = QListWidgetItem(roi.name)
                item.setCheckState(Qt.CheckState.Checked)  # 默认选中所有ROI
                self.roi_list_widget.addItem(item)

            self.rois_data = rois
            self.parse_status.setText(f"状态: 成功加载 {len(rois)} 个ROI")
            self.log_signal.emit("[INFO] ROI 解析完成，可以在列表中选择要处理的 ROI")

        except Exception as e:
            self.log_signal.emit(f"[ERROR] 解析 ROI 过程中发生错误: {e}")
            self.parse_status.setText("状态: 解析失败")
            self.roi_list_widget.clear()
            self.rois_data = None
            QMessageBox.critical(self, "解析错误", f"解析RTSTRUCT文件时发生错误:\n{str(e)}")

    # -----------------------------
    # 运行处理
    # -----------------------------
    def run_process(self):
        rtss_file = self.rtss_path.text().strip()
        if not rtss_file:
            QMessageBox.warning(self, "提示", "请先选择 RTSTRUCT 文件")
            return

        if self.rois_data is None:
            QMessageBox.warning(self, "提示", "请先解析 ROI")
            return

        # 获取用户选中的ROI名称
        selected_rois = []
        for i in range(self.roi_list_widget.count()):
            item = self.roi_list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_rois.append(item.text())

        if len(selected_rois) == 0:
            QMessageBox.warning(self, "提示", "请至少选择一个 ROI 进行处理")
            return

        # 使用线程运行，避免阻塞 GUI
        thread = threading.Thread(target=self.run_task_thread,
                                  args=(rtss_file, selected_rois))
        thread.start()

    # -----------------------------
    # 线程执行任务
    # -----------------------------
    def run_task_thread(self, rtss_path, selected_rois):
        try:
            self.log_signal.emit(f"[INFO] 开始处理 RTSTRUCT 文件: {rtss_path}")

            # 从已解析的 ROI 数据中筛选用户选择的 ROI
            filtered_rois = [roi for roi in self.rois_data if roi.name in selected_rois]

            if len(filtered_rois) == 0:
                self.log_signal.emit("[ERROR] 没有找到任何有效的 ROI")
                return

            self.log_signal.emit(f"[INFO] 正在处理 {len(filtered_rois)} 个选中的 ROI")
            self.log_signal.emit("[INFO] 构建冲突图...")

            graph = build_conflict_graph(filtered_rois)

            self.log_signal.emit("[INFO] 应用 DSATUR 算法进行图着色...")
            colors = dsatur_coloring(graph)

            roi_sets = split_into_sets(colors)

            self.log_signal.emit(f"[RESULT] 分割为 {len(roi_sets)} 个非重叠集合:")
            for idx in sorted(roi_sets.keys()):
                self.log_signal.emit(f"第 {idx} 组: {roi_sets[idx]}")

            self.log_signal.emit("[INFO] 处理完成!")

        except Exception as e:
            self.log_signal.emit(f"[ERROR] 处理过程中发生错误: {e}")

    def closeEvent(self, event):
        current_path = self.rtss_path.text().strip()
        if current_path:
            self.settings.setValue("last_rtss_path", current_path)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ROIApp()
    window.show()
    sys.exit(app.exec())
