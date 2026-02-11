"""RTSS非重叠分割对话框"""
import os
import sys
import threading

from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QListWidget, QListWidgetItem,
    QProgressDialog
)

# 导入核心算法模块
from .rtss_split_non_overlap import load_rois_from_rtss, build_conflict_graph, dsatur_coloring, split_into_sets


class RTSSSplitDialog(QDialog):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, parent=None, rtss_data=None):
        super().__init__(parent)
        self.setWindowTitle("ROI 非重叠分组工具")
        self.resize(700, 600)
        self.setModal(False)  # 非模态对话框
        
        # 接收RTSS数据
        self.rtss_data = rtss_data
        
        # 初始化配置存储
        self.settings = QSettings("ROIApp", "RTSSSplitter")
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 显示当前RTSS信息
        self.rtss_info_label = QLabel("当前RTSS文件:")
        self.layout.addWidget(self.rtss_info_label)
        
        if self.rtss_data:
            rtss_name = self.rtss_data.get("filename", "Unknown")
            self.rtss_info_display = QLabel(f"  {rtss_name}")
            self.layout.addWidget(self.rtss_info_display)
            roi_count = len(self.rtss_data.get("rois", []))
            self.parse_status = QLabel(f"状态: 已加载 {roi_count} 个ROI")
        else:
            self.rtss_info_display = QLabel("  未加载RTSS文件")
            self.layout.addWidget(self.rtss_info_display)
            self.parse_status = QLabel("状态: 请先在主窗口加载RTSTRUCT文件")
        self.layout.addWidget(self.parse_status)
        
        # ROI 选择列表
        self.roi_list_label = QLabel("请选择要处理的 ROI (多选):")
        self.layout.addWidget(self.roi_list_label)
        
        self.roi_list_widget = QListWidget()
        self.roi_list_widget.setSelectionMode(QListWidget.MultiSelection)
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
        self.progress_signal.connect(self.update_progress)
        self.finished_signal.connect(self.on_processing_finished)
        
        # 存储解析的ROI数据
        self.rois_data = None
        self.processing_result = None
        
        # 进度对话框
        self.progress_dialog = None
        
        # 如果有RTSS数据，直接解析
        if self.rtss_data:
            self.auto_parse_rois_from_data()
        else:
            self.roi_list_widget.clear()
            self.rois_data = None
    
    # -----------------------------
    # 日志函数
    # -----------------------------
    def log_message(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
    
    def update_progress(self, value):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
    
    def on_processing_finished(self, result):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        self.processing_result = result
        self.log_signal.emit("[INFO] 处理完成!")
        
        # 显示结果摘要
        msg = f"分割完成！共分为 {len(result)} 个非重叠集合:\n\n"
        for idx in sorted(result.keys()):
            msg += f"第 {idx} 组 ({len(result[idx])} 个ROI):\n"
            for name in result[idx]:
                msg += f"  - {name}\n"
            msg += "\n"
        
        QMessageBox.information(self, "处理完成", msg)
    
    # -----------------------------
    # 从主窗口数据解析 ROI
    # -----------------------------
    def auto_parse_rois_from_data(self):
        if not self.rtss_data:
            self.parse_status.setText("状态: 未加载RTSS数据")
            self.roi_list_widget.clear()
            self.rois_data = None
            return
        
        try:
            rois = self.rtss_data.get("rois", [])
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
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)  # 默认选中所有ROI
                self.roi_list_widget.addItem(item)
            
            self.rois_data = rois
            self.parse_status.setText(f"状态: 成功加载 {len(rois)} 个ROI")
            self.log_signal.emit("[INFO] ROI 解析完成，可以在列表中选择要处理的 ROI")
            
        except Exception as e:
            self.log_signal.emit(f"[ERROR] 解析 ROI 过程中发生错误: {e}")
            self.parse_status.setText("状态: 解析失败")
            self.roi_list_widget.clear()
            self.rois_data = None
            QMessageBox.critical(self, "解析错误", f"解析RTSTRUCT数据时发生错误:\n{str(e)}")
    

    
    # -----------------------------
    # 运行处理
    # -----------------------------
    def run_process(self):
        if not self.rtss_data:
            QMessageBox.warning(self, "提示", "请先在主窗口加载RTSTRUCT文件")
            return
        
        if self.rois_data is None:
            QMessageBox.warning(self, "提示", "请先解析 ROI")
            return
        
        # 获取用户选中的ROI名称
        selected_rois = []
        for i in range(self.roi_list_widget.count()):
            item = self.roi_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_rois.append(item.text())
        
        if len(selected_rois) == 0:
            QMessageBox.warning(self, "提示", "请至少选择一个 ROI 进行处理")
            return
        
        # 显示进度对话框
        self.progress_dialog = QProgressDialog("正在处理RTSS...", "取消", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setWindowTitle("处理中")
        self.progress_dialog.show()
        
        # 使用线程运行，避免阻塞 GUI
        thread = threading.Thread(target=self.run_task_thread,
                                  args=(self.rtss_data, selected_rois))
        thread.start()
    
    # -----------------------------
    # 线程执行任务
    # -----------------------------
    def run_task_thread(self, rtss_data, selected_rois):
        try:
            self.log_signal.emit(f"[INFO] 开始处理RTSS数据")
            self.progress_signal.emit(10)
            
            # 从已解析的 ROI 数据中筛选用户选择的 ROI
            filtered_rois = [roi for roi in self.rois_data if roi.name in selected_rois]
            
            if len(filtered_rois) == 0:
                self.log_signal.emit("[ERROR] 没有找到任何有效的 ROI")
                return
            
            self.log_signal.emit(f"[INFO] 正在处理 {len(filtered_rois)} 个选中的 ROI")
            self.progress_signal.emit(30)
            
            self.log_signal.emit("[INFO] 构建冲突图...")
            graph = build_conflict_graph(filtered_rois)
            self.progress_signal.emit(60)
            
            self.log_signal.emit("[INFO] 应用 DSATUR 算法进行图着色...")
            colors = dsatur_coloring(graph)
            self.progress_signal.emit(80)
            
            roi_sets = split_into_sets(colors)
            self.progress_signal.emit(90)
            
            self.log_signal.emit(f"[RESULT] 分割为 {len(roi_sets)} 个非重叠集合:")
            for idx in sorted(roi_sets.keys()):
                self.log_signal.emit(f"第 {idx} 组: {roi_sets[idx]}")
            
            self.progress_signal.emit(100)
            self.finished_signal.emit(roi_sets)
            
        except Exception as e:
            self.log_signal.emit(f"[ERROR] 处理过程中发生错误: {e}")
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
    
    def get_result(self):
        """获取处理结果"""
        return self.processing_result
    
    def closeEvent(self, event):
        # 不需要保存路径，因为我们使用主窗口的数据
        event.accept()