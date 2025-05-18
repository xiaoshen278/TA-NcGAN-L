import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QMessageBox, QProgressBar,
                             QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette
import cv2
import numpy as np

import args_fusion
import test
import notebook.services.shutdown

class VideoFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("红外与可见光视频融合系统")
        self.setGeometry(100, 100, 800, 600)
        #加载模型
        self.model = test.load_model(args_fusion.args.model_default, deepsupervision = False)
        # 控制标志
        self.enable_interpolation = True  # 默认启用插帧
        self.enable_fps_limit = False  # 默认不启用帧率限制
        self.target_fps_limit = 6  # 目标限制帧率
        self.output_fps = None  # 实际输出帧率
        self.last_fused_frame = None  # 用于存储上一帧用于插值

        # 视频文件路径
        self.ir_video_path = None
        self.visible_video_path = None
        self.output_path = None

        # 视频捕获对象
        self.ir_cap = None
        self.visible_cap = None
        self.original_fps = None  # 原始帧率
        self.frame_size = None
        self.original_size = None  # 保存原始尺寸
        self.frame_interval = 1  # 帧间隔计数器
        self.frame_counter = 0  # 帧计数器

        # 初始化UI
        self.init_ui()

    def set_background_color(self):
        """设置窗口背景色"""
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 240, 245))  # 浅灰色背景
        palette.setColor(QPalette.WindowText, QColor(50, 50, 50))  # 深灰色文字
        self.setPalette(palette)

    def init_ui(self):
        # 主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        main_layout.setContentsMargins(15, 15, 15, 15)  # 设置边距
        main_layout.setSpacing(15)  # 设置间距

        # 文件选择区域
        file_layout = QVBoxLayout()
        file_layout.setSpacing(10)

        # 红外视频选择
        ir_layout = QVBoxLayout()
        self.ir_label = QLabel("红外视频: 未选择")
        self.ir_label.setWordWrap(True)
        ir_btn = QPushButton("选择红外视频")
        ir_btn.setObjectName("irButton")  # 为按钮设置对象名用于样式表
        ir_btn.clicked.connect(lambda: self.select_video("ir"))
        ir_layout.addWidget(self.ir_label)
        ir_layout.addWidget(ir_btn)

        # 可见光视频选择
        visible_layout = QVBoxLayout()
        self.visible_label = QLabel("可见光视频: 未选择")
        self.visible_label.setWordWrap(True)
        visible_btn = QPushButton("选择可见光视频")
        visible_btn.setObjectName("visibleButton")
        visible_btn.clicked.connect(lambda: self.select_video("visible"))
        visible_layout.addWidget(self.visible_label)
        visible_layout.addWidget(visible_btn)

        # 将两个选择布局放入一个水平布局中
        h_layout = QHBoxLayout()
        h_layout.addLayout(ir_layout)
        h_layout.addLayout(visible_layout)
        h_layout.setSpacing(20)

        file_layout.addLayout(h_layout)
        main_layout.addLayout(file_layout)

        # 视频显示区域
        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 360)
        self.video_display.setStyleSheet("""
            QLabel {
                background-color: #333333;
                border: 2px solid #555555;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.video_display)

        # 控制区域
        control_group = QGroupBox("融合控制")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)

        # 第一行控制按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        # 融合按钮
        self.fusion_btn = QPushButton("开始融合")
        self.fusion_btn.setObjectName("fusionButton")
        self.fusion_btn.clicked.connect(self.start_fusion)
        self.fusion_btn.setEnabled(False)

        # 输出路径选择
        output_btn = QPushButton("选择输出路径")
        output_btn.setObjectName("outputButton")
        output_btn.clicked.connect(self.select_output_path)

        btn_layout.addWidget(self.fusion_btn)
        btn_layout.addWidget(output_btn)
        control_layout.addLayout(btn_layout)

        # 第二行选项
        option_layout = QHBoxLayout()
        option_layout.setSpacing(20)

        # 插帧控制复选框
        self.interpolation_checkbox = QCheckBox("启用插帧算法")
        self.interpolation_checkbox.setChecked(self.enable_interpolation)
        self.interpolation_checkbox.stateChanged.connect(self.toggle_interpolation)
        option_layout.addWidget(self.interpolation_checkbox)

        # 帧率限制复选框
        self.fps_limit_checkbox = QCheckBox("限制最大帧率")
        self.fps_limit_checkbox.setChecked(self.enable_fps_limit)
        self.fps_limit_checkbox.stateChanged.connect(self.toggle_fps_limit)
        option_layout.addWidget(self.fps_limit_checkbox)

        option_layout.addStretch()
        control_layout.addLayout(option_layout)
        main_layout.addWidget(control_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #aaaaaa;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("准备就绪")

        # 应用样式表
        self.setStyleSheet("""
            QPushButton {
                background-color: #5D9B9B;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }

            QPushButton:hover {
                background-color: #4A7B7B;
            }

            QPushButton:pressed {
                background-color: #3A6B6B;
            }

            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }

            #fusionButton {
                background-color: #4CAF50;
            }

            #fusionButton:hover {
                background-color: #3E8E41;
            }

            #fusionButton:pressed {
                background-color: #2E7D32;
            }

            QLabel {
                color: #333333;
                font-size: 14px;
            }

            QGroupBox {
                border: 1px solid #dddddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }

            QCheckBox {
                color: #333333;
                font-weight: bold;
            }

            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }

            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
            }

            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 2px solid #cccccc;
            }
        """)

    def toggle_interpolation(self, state):
        """切换插帧算法状态"""
        self.enable_interpolation = (state == Qt.Checked)
        self.animate_button(self.interpolation_checkbox)
        self.update_status_message()

    def toggle_fps_limit(self, state):
        """切换帧率限制状态"""
        self.enable_fps_limit = (state == Qt.Checked)
        self.animate_button(self.fps_limit_checkbox)
        self.update_status_message()

    def update_status_message(self):
        """更新状态栏消息"""
        if hasattr(self, 'original_fps') and self.original_fps is not None:
            if self.original_fps >= self.target_fps_limit:
                if self.enable_fps_limit:
                    self.status_bar.showMessage(
                        f"原始帧率{self.original_fps:.1f}fps > 6fps，已限制为6fps并插帧到30fps")
                else:
                    if self.enable_interpolation:
                        self.status_bar.showMessage(
                            f"原始帧率{self.original_fps:.1f}fps > 6fps，未限制帧率，插帧到{self.original_fps * 5:.1f}fps")
                    else:
                        self.status_bar.showMessage(
                            f"原始帧率{self.original_fps:.1f}fps > 6fps，未限制帧率，保持原始帧率")
            else:
                if self.enable_interpolation:
                    self.status_bar.showMessage(
                        f"原始帧率{self.original_fps:.1f}fps < 6fps，已插帧到{self.original_fps * 5:.1f}fps")
                else:
                    self.status_bar.showMessage(
                        f"原始帧率{self.original_fps:.1f}fps < 6fps，保持原始帧率")
        else:
            self.status_bar.showMessage("准备就绪")

    def animate_button(self, widget):
        """按钮动画效果"""
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(100)
        animation.setEasingCurve(QEasingCurve.OutQuad)

        original_geometry = widget.geometry()
        animation.setStartValue(original_geometry)
        animation.setEndValue(original_geometry.adjusted(-2, -2, 2, 2))
        animation.start()

    def select_video(self, video_type):
        """选择视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"选择{video_type}视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )

        if file_path:
            if video_type == "ir":
                self.ir_video_path = file_path
                self.ir_label.setText(f"红外视频: {file_path}")
            else:
                self.visible_video_path = file_path
                self.visible_label.setText(f"可见光视频: {file_path}")

            # 检查是否两个视频都已选择
            if self.ir_video_path and self.visible_video_path:
                self.fusion_btn.setEnabled(True)
                self.animate_button(self.fusion_btn)

    def select_output_path(self):
        """选择输出路径"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件", "",
            "MP4视频 (*.mp4);;AVI视频 (*.avi);;所有文件 (*.*)"
        )

        if file_path:
            self.output_path = file_path
            self.status_bar.showMessage(f"输出路径: {file_path}")


    def fusion_algorithm(self, ir_frame, visible_frame):
        try:
            fusion = test.process_(visible_frame, ir_frame, self.model)
            return fusion

        except Exception as e:
            print(f"融合算法错误: {str(e)}")
            return None

    def display_frame(self, frame):
        """在QLabel中显示帧"""
        if frame is None:
            return

        # 将OpenCV图像转换为Qt图像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 缩放以适应显示区域
        scaled_pixmap = pixmap.scaled(
            self.video_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_display.setPixmap(scaled_pixmap)

    def start_fusion(self):
        if not self.output_path:
            QMessageBox.warning(self, "警告", "请先选择输出路径！")
            return

        try:
            # 打开视频文件
            self.ir_cap = cv2.VideoCapture(self.ir_video_path)
            self.visible_cap = cv2.VideoCapture(self.visible_video_path)

            # 检查视频是否成功打开
            if not self.ir_cap.isOpened() or not self.visible_cap.isOpened():
                raise ValueError("无法打开视频文件")

            # 检查视频属性
            ir_frames = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            visible_frames = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            min_frames = min(ir_frames, visible_frames)

            if min_frames == 0:
                raise ValueError("视频文件无效或无法读取")

            # 获取视频参数并检查一致性
            self.original_fps = self.ir_cap.get(cv2.CAP_PROP_FPS)
            ir_width = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ir_height = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            visible_width = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            visible_height = int(self.visible_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 确定最终输出尺寸（取两个视频的最小尺寸）
            output_width = min(ir_width, visible_width)
            output_height = min(ir_height, visible_height)
            self.original_size = (output_width, output_height)  # 保存原始尺寸
            self.frame_size = self.original_size  # 处理时使用原始尺寸

            # 检查视频尺寸是否一致
            if (ir_width, ir_height) != (visible_width, visible_height):
                reply = QMessageBox.question(self, '提示',
                                             '两个视频的尺寸不一致，将自动调整为较小的尺寸继续处理。是否继续？',
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    self.reset_video_processing()
                    return

            # 计算输出帧率
            if self.original_fps >= self.target_fps_limit:
                # 原始帧率较高的情况
                if self.enable_fps_limit:
                    # 限制帧率到10fps，并插帧到30fps
                    self.output_fps = 30
                    self.frame_interval = int(round(self.original_fps / self.target_fps_limit))
                else:
                    # 不限制帧率，根据插帧选项决定
                    if self.enable_interpolation:
                        self.output_fps = self.original_fps * 3
                    else:
                        self.output_fps = self.original_fps
                    self.frame_interval = 1
            else:
                # 原始帧率较低的情况
                if self.enable_interpolation:
                    self.output_fps = self.original_fps * 3
                else:
                    self.output_fps = self.original_fps
                self.frame_interval = 1

            # 更新状态信息
            self.update_status_message()

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.output_fps, self.original_size,isColor=False)

            if not self.video_writer.isOpened():
                raise ValueError("无法创建输出视频文件")

            # 禁用按钮
            self.fusion_btn.setEnabled(False)
            self.animate_button(self.fusion_btn)

            # 设置进度条
            self.progress_bar.setMaximum(min_frames)

            # 启动定时器处理视频帧
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_frame)

            # 计算定时器间隔(毫秒)
            # 这里使用固定间隔，不根据帧率变化，因为我们只显示处理后的帧
            self.timer.start(50)  # 20fps的显示速率

        except Exception as e:
            QMessageBox.critical(self, "错误", f"视频处理出错: {str(e)}")
            self.reset_video_processing()

    def process_frame(self):
        try:
            # 读取当前帧
            ir_ret, ir_frame = self.ir_cap.read()
            visible_ret, visible_frame = self.visible_cap.read()

            if not ir_ret or not visible_ret:
                self.finish_processing()
                return

            # 更新帧计数器
            self.frame_counter += 1

            # 检查是否需要跳过帧（帧率限制）
            if self.frame_counter % self.frame_interval != 0:
                return

            # 调整到原始尺寸（如果需要）
            if (ir_frame.shape[1], ir_frame.shape[0]) != self.original_size:
                ir_frame = cv2.resize(ir_frame, self.original_size)
            if (visible_frame.shape[1], visible_frame.shape[0]) != self.original_size:
                visible_frame = cv2.resize(visible_frame, self.original_size)

            # 使用融合算法接口
            fused_frame = self.fusion_algorithm(ir_frame, visible_frame)

            if fused_frame is None:
                raise ValueError("图像融合失败")

            # 写入输出视频（根据插帧状态决定是否插帧）
            self.write_with_interpolation(fused_frame)

            # 更新显示（只显示实际处理的帧）
            self.display_frame(fused_frame)

            # 更新进度条
            current_frame = int(self.ir_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.progress_bar.setValue(current_frame)

        except Exception as e:
            self.timer.stop()
            QMessageBox.critical(self, "错误", f"处理帧时出错: {str(e)}")
            self.reset_video_processing()

    def write_with_interpolation(self, current_frame):
        """带插帧的视频写入方法"""
        if current_frame is None:
            return

        if self.last_fused_frame is None:
            # 第一帧，直接写入
            self.video_writer.write(current_frame)
            self.last_fused_frame = current_frame
            return

        # 只有在启用插帧时才进行插帧
        if self.enable_interpolation:
            # 写入插值帧（当前帧和前帧之间插入两帧）
            alpha1 = 0.20  # 第一帧插值权重
            alpha2 = 0.40  # 第二帧插值权重
            alpha3 = 0.60
            alpha4 = 0.80

            interpolated_frame1 = cv2.addWeighted(
                self.last_fused_frame, 1 - alpha1,
                current_frame, alpha1,
                0)
            interpolated_frame2 = cv2.addWeighted(
                self.last_fused_frame, 1 - alpha2,
                current_frame, alpha2,
                0)
            interpolated_frame3 = cv2.addWeighted(
                self.last_fused_frame, 1 - alpha3,
                current_frame, alpha3,
                0)
            interpolated_frame4 = cv2.addWeighted(
                self.last_fused_frame, 1 - alpha4,
                current_frame, alpha4,
                0)

            self.video_writer.write(interpolated_frame1)
            self.video_writer.write(interpolated_frame2)
            self.video_writer.write(interpolated_frame3)
            self.video_writer.write(interpolated_frame4)

            # 写入当前帧
        self.video_writer.write(current_frame)
        self.last_fused_frame = current_frame
    def finish_processing(self):
        """完成视频处理"""
        self.timer.stop()
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()

        self.reset_video_processing()
        QMessageBox.information(self, "完成", "视频融合处理完成！")
        self.status_bar.showMessage("处理完成")

    def reset_video_processing(self):
        """重置视频处理状态"""
        if hasattr(self, 'timer') and self.timer:
            self.timer.stop()

        if hasattr(self, 'ir_cap') and self.ir_cap:
            self.ir_cap.release()

        if hasattr(self, 'visible_cap') and self.visible_cap:
            self.visible_cap.release()

        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()

        self.ir_cap = None
        self.visible_cap = None
        self.video_writer = None
        self.last_fused_frame = None
        self.original_fps = None
        self.output_fps = None
        self.original_size = None
        self.frame_counter = 0
        self.frame_interval = 1

        # 重置按钮状态
        self.fusion_btn.setEnabled(self.ir_video_path is not None and self.visible_video_path is not None)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("准备就绪")

    def closeEvent(self, event):
        self.reset_video_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoFusionApp()
    window.show()
    sys.exit(app.exec_())