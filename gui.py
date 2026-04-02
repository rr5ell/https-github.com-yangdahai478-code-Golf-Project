"""
高尔夫推杆轨迹模拟器 - GUI界面
使用PyQt6实现交互式界面
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QLabel, QSlider,
                              QDoubleSpinBox, QFileDialog, QGroupBox, QCheckBox,
                              QMessageBox, QTabWidget, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QDoubleValidator
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from physics_engine import TerrainModel, PhysicsEngine, create_demo_point_cloud


class Terrain3DViewer(gl.GLViewWidget):
    """3D地形可视化组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=20, elevation=30, azimuth=45)
        self.setBackgroundColor('w')

        # 点云显示
        self.scatter = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]),
                                            color=(0.3, 0.7, 0.3, 0.6),
                                            size=2)
        self.addItem(self.scatter)

        # 轨迹线
        self.trajectory_line = gl.GLLinePlotItem(pos=np.array([[0, 0, 0]]),
                                                 color=(1, 0, 0, 1),
                                                 width=3)
        self.addItem(self.trajectory_line)

        # 球体
        self.ball_mesh = None

        # 添加坐标轴
        self.add_axes()

    def add_axes(self):
        """添加坐标轴"""
        length = 10
        # X轴 - 红色
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [length, 0, 0]]),
                                   color=(1, 0, 0, 1), width=2)
        # Y轴 - 绿色
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, length, 0]]),
                                   color=(0, 1, 0, 1), width=2)
        # Z轴 - 蓝色
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, length/2]]),
                                   color=(0, 0, 1, 1), width=2)

        self.addItem(x_axis)
        self.addItem(y_axis)
        self.addItem(z_axis)

    def update_terrain(self, points):
        """更新地形点云"""
        # 标准化颜色
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            normalized_z = (z - z_min) / (z_max - z_min)
        else:
            normalized_z = np.zeros_like(z)

        # 绿色到黄色的渐变
        colors = np.column_stack([
            normalized_z,           # R
            0.7 - 0.3 * normalized_z,  # G
            0.1,                   # B
            np.ones_like(z) * 0.6  # Alpha
        ])

        self.scatter.setData(pos=points, color=colors, size=3)

        # 调整相机
        center = points.mean(axis=0)
        max_range = (points.max(axis=0) - points.min(axis=0)).max()
        self.setCameraPosition(distance=max_range * 2, elevation=45, azimuth=45)

    def update_trajectory(self, trajectory):
        """更新轨迹显示"""
        if len(trajectory) > 0:
            self.trajectory_line.setData(pos=trajectory)
            self.trajectory_line.setVisible(True)
        else:
            self.trajectory_line.setVisible(False)

    def clear_trajectory(self):
        """清除轨迹"""
        self.trajectory_line.setVisible(False)


class TopDownViewer(pg.PlotWidget):
    """俯视图可视化组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True)
        self.setLabel('bottom', 'X (m)')
        self.setLabel('left', 'Y (m)')

        # 点云散点图
        self.scatter = pg.ScatterPlotItem()
        self.addItem(self.scatter)

        # 轨迹线
        self.trajectory_line = pg.PlotCurveItem(pen=pg.mkPen('r', width=2))
        self.addItem(self.trajectory_line)

        # 力方向箭头
        self.force_arrow = pg.ArrowItem(angle=0, tipAngle=30, baseAngle=20,
                                         headLen=15, tailLen=10, tailWidth=3,
                                         pen=pg.mkPen('b', width=2))
        self.force_arrow.setVisible(False)
        self.addItem(self.force_arrow)

        # 起点标记
        self.start_marker = pg.ScatterPlotItem()
        self.addItem(self.start_marker)

    def update_terrain(self, points):
        """更新地形显示（带颜色映射）"""
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()

        if z_max > z_min:
        # 颜色映射：绿色到黄色
            normalized_z = (z - z_min) / (z_max - z_min)
            brushes = [pg.mkBrush(int(255 * nz), int(255 * (0.7 - 0.3 * nz)), int(25), 180)
                      for nz in normalized_z]
        else:
            brushes = [pg.mkBrush(0, 180, 25, 180)] * len(points)

        spots = [{'pos': [x, y], 'brush': brush, 'size': 5}
                 for (x, y, _), brush in zip(points, brushes)]

        self.scatter.setData(spots)

        # 设置视图范围
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        self.setXRange(points[:, 0].min() - x_range * 0.1,
                       points[:, 0].max() + x_range * 0.1)
        self.setYRange(points[:, 1].min() - y_range * 0.1,
                       points[:, 1].max() + y_range * 0.1)

    def update_trajectory(self, trajectory):
        """更新轨迹显示"""
        if len(trajectory) > 0:
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            self.trajectory_line.setData(x, y)
            self.trajectory_line.setVisible(True)
        else:
            self.trajectory_line.setVisible(False)

    def update_force_direction(self, start_pos, force_vector):
        """更新力方向显示"""
        if np.linalg.norm(force_vector) > 0:
            angle = np.degrees(np.arctan2(force_vector[1], force_vector[0]))
            self.force_arrow.setPos(start_pos[0], start_pos[1])
            self.force_arrow.setStyle(angle=angle)
            self.force_arrow.setVisible(True)
        else:
            self.force_arrow.setVisible(False)

    def update_start_position(self, start_pos):
        """更新起始位置标记"""
        self.start_marker.setData(x=[start_pos[0]], y=[start_pos[1]],
                                 symbol='o', size=15,
                                 brush=pg.mkBrush('b'), pen=pg.mkPen('w', width=2))

    def clear_trajectory(self):
        """清除轨迹"""
        self.trajectory_line.setVisible(False)
        self.force_arrow.setVisible(False)


class GolfPuttingSimulator(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('高尔夫推杆轨迹模拟器')
        self.resize(1400, 900)

        # 数据
        self.points = None
        self.terrain_model = None
        self.trajectory = None
        self.start_position = np.array([0.0, 0.0, 0.0])

        self.init_ui()
        self.load_demo_terrain()

    def init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # 右侧可视化面板（标签页）
        viz_panel = QTabWidget()
        self.terrain_3d = Terrain3DViewer()
        self.top_down = TopDownViewer()

        viz_panel.addTab(self.top_down, '俯视图')
        viz_panel.addTab(self.terrain_3d, '3D视图')

        main_layout.addWidget(viz_panel, 3)

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. 地形加载
        terrain_group = QGroupBox('地形设置')
        terrain_layout = QVBoxLayout()

        load_btn = QPushButton('加载点云文件')
        load_btn.clicked.connect(self.load_point_cloud)
        terrain_layout.addWidget(load_btn)

        demo_btn = QPushButton('使用演示地形')
        demo_btn.clicked.connect(self.load_demo_terrain)
        terrain_layout.addWidget(demo_btn)

        terrain_group.setLayout(terrain_layout)
        layout.addWidget(terrain_group)

        # 2. 起始位置
        start_group = QGroupBox('起始位置 (m)')
        start_layout = QVBoxLayout()

        self.start_x = self.create_spinbox(-100, 100, 0.0)
        self.start_y = self.create_spinbox(-100, 100, 0.0)
        self.start_z = self.create_spinbox(-10, 10, 0.0)

        self.start_x.valueChanged.connect(self.update_start_position_ui)
        self.start_y.valueChanged.connect(self.update_start_position_ui)
        self.start_z.valueChanged.connect(self.update_start_position_ui)

        start_layout.addWidget(QLabel('X:'))
        start_layout.addWidget(self.start_x)
        start_layout.addWidget(QLabel('Y:'))
        start_layout.addWidget(self.start_y)
        start_layout.addWidget(QLabel('Z:'))
        start_layout.addWidget(self.start_z)

        start_group.setLayout(start_layout)
        layout.addWidget(start_group)

        # 3. 推杆力度（大小和方向）
        force_group = QGroupBox('推杆设置')
        force_layout = QVBoxLayout()

        # 力大小
        force_layout.addWidget(QLabel('推杆力度:'))
        self.force_magnitude = self.create_spinbox(0.01, 10.0, 1.0)
        self.force_magnitude.setSingleStep(0.1)
        force_layout.addWidget(self.force_magnitude)

        # 力方向 - 滑块
        force_layout.addWidget(QLabel('方向角度 (度):'))
        self.force_angle = QSlider(Qt.Orientation.Horizontal)
        self.force_angle.setRange(0, 360)
        self.force_angle.setValue(0)
        self.force_angle.valueChanged.connect(self.update_force_display)
        force_layout.addWidget(self.force_angle)

        self.angle_label = QLabel('0°')
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        force_layout.addWidget(self.angle_label)

        force_group.setLayout(force_layout)
        layout.addWidget(force_group)

        # 4. 物理参数
        physics_group = QGroupBox('物理参数')
        physics_layout = QVBoxLayout()

        physics_layout.addWidget(QLabel('滚动摩擦系数 (0.05-0.20):'))
        self.friction_coeff = self.create_spinbox(0.05, 0.20, 0.10)
        self.friction_coeff.setSingleStep(0.01)
        physics_layout.addWidget(self.friction_coeff)

        self.air_resistance = QCheckBox('启用空气阻力')
        self.air_resistance.setChecked(True)
        physics_layout.addWidget(self.air_resistance)

        self.adaptive_timestep = QCheckBox('自适应时间步长')
        self.adaptive_timestep.setChecked(True)
        physics_layout.addWidget(self.adaptive_timestep)

        physics_group.setLayout(physics_layout)
        layout.addWidget(physics_group)

        # 5. 模拟控制
        sim_group = QGroupBox('模拟控制')
        sim_layout = QVBoxLayout()

        simulate_btn = QPushButton('开始模拟')
        simulate_btn.setStyleSheet('''
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        ''')
        simulate_btn.clicked.connect(self.run_simulation)
        sim_layout.addWidget(simulate_btn)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # 6. 结果显示
        result_group = QGroupBox('结果')
        result_layout = QVBoxLayout()

        self.result_label = QLabel('等待模拟...')
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)

        sim_group.setLayout(sim_layout)
        layout.addWidget(result_group)

        layout.addStretch()

        return panel

    def create_spinbox(self, min_val, max_val, default_val):
        """创建双精度旋转框"""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default_val)
        spinbox.setSingleStep(0.01)
        spinbox.setDecimals(3)
        return spinbox

    def load_point_cloud(self):
        """加载点云文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择点云文件',
            '',
            '文本文件 (*.txt *.csv);;所有文件 (*.*)'
        )

        if file_path:
            try:
                # 尝试读取文件
                if file_path.endswith('.csv'):
                    data = np.loadtxt(file_path, delimiter=',')
                else:
                    data = np.loadtxt(file_path)

                # 检查数据格式
                if data.shape[1] < 3:
                    raise ValueError('点云文件至少需要3列 (X, Y, Z)')

                self.points = data[:, :3]
                self.init_terrain()

            except Exception as e:
                QMessageBox.critical(self, '错误', f'加载点云失败: {str(e)}')

    def load_demo_terrain(self):
        """加载演示地形"""
        self.points = create_demo_point_cloud()
        self.init_terrain()

    def init_terrain(self):
        """初始化地形模型"""
        if self.points is None:
            return

        try:
            # 创建地形模型
            self.terrain_model = TerrainModel(self.points, smoothing_factor=0.1)

            # 更新可视化
            self.terrain_3d.update_terrain(self.points)
            self.top_down.update_terrain(self.points)

            # 更新起始位置到地形中心
            center = self.points.mean(axis=0)
            self.start_position = center.copy()
            self.start_x.setValue(center[0])
            self.start_y.setValue(center[1])
            self.start_z.setValue(center[2])

            # 清除轨迹
            self.clear_trajectory()

            print('地形加载成功')
            print(f'点云数量: {len(self.points)}')
            print(f'范围: X[{self.points[:, 0].min():.2f}, {self.points[:, 0].max():.2f}], '
                  f'Y[{self.points[:, 1].min():.2f}, {self.points[:, 1].max():.2f}], '
                  f'Z[{self.points[:, 2].min():.3f}, {self.points[:, 2].max():.3f}]')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'初始化地形失败: {str(e)}')

    def update_start_position_ui(self):
        """更新起始位置"""
        self.start_position[0] = self.start_x.value()
        self.start_position[1] = self.start_y.value()
        self.start_position[2] = self.start_z.value()

        if self.terrain_model:
            # 更新z到地形高度
            self.start_position[2] = self.terrain_model.get_elevation(
                self.start_position[0], self.start_position[1]
            )
            self.start_z.setValue(self.start_position[2])

        self.top_down.update_start_position(self.start_position)

    def update_force_display(self):
        """更新力方向显示"""
        angle = self.force_angle.value()
        self.angle_label.setText(f'{angle}°')

        # 更新俯视图中的箭头
        if self.terrain_model:
            magnitude = self.force_magnitude.value()
            angle_rad = np.radians(angle)
            force_vector = np.array([
                magnitude * np.cos(angle_rad),
                magnitude * np.sin(angle_rad),
                0.0
            ])
            self.top_down.update_force_direction(self.start_position, force_vector)

    def run_simulation(self):
        """运行模拟"""
        if self.terrain_model is None:
            QMessageBox.warning(self, '警告', '请先加载地形')
            return

        try:
            # 获取参数
            start_pos = self.start_position.copy()
            force_magnitude = self.force_magnitude.value()
            angle_deg = self.force_angle.value()
            friction_coeff = self.friction_coeff.value()

            # 计算初始速度（推杆力度转换为速度）
            angle_rad = np.radians(angle_deg)
            velocity = np.array([
                force_magnitude * np.cos(angle_rad),
                force_magnitude * np.sin(angle_rad),
                0.0
            ])

            # 创建物理引擎
            physics = PhysicsEngine(
                self.terrain_model,
                rolling_friction_coeff=friction_coeff,
                air_resistance=self.air_resistance.isChecked(),
                adaptive_timestep=self.adaptive_timestep.isChecked()
            )

            # 运行模拟
            self.trajectory, times = physics.simulate(start_pos, velocity)

            # 更新可视化
            self.terrain_3d.update_trajectory(self.trajectory)
            self.top_down.update_trajectory(self.trajectory)

            # 显示结果
            if len(self.trajectory) > 0:
                total_distance = np.sum(np.linalg.norm(
                    np.diff(self.trajectory[:, :2], axis=0), axis=1
                ))
                final_pos = self.trajectory[-1]
                start_pos_2d = self.trajectory[0, :2]
                displacement = np.linalg.norm(final_pos[:2] - start_pos_2d)

                result_text = f'''
模拟完成!

轨迹点数: {len(self.trajectory)}
总行驶距离: {total_distance:.3f} m
净位移: {displacement:.3f} m
最终位置: X={final_pos[0]:.2f} m, Y={final_pos[1]:.2f} m
最终高度: {final_pos[2]:.3f} m
推杆力度: {force_magnitude:.2f} m/s
方向: {angle_deg}°
摩擦系数: {friction_coeff:.3f}
                '''
                self.result_label.setText(result_text)

        except Exception as e:
            QMessageBox.critical(self, '错误', f'模拟失败: {str(e)}')
            import traceback
            traceback.print_exc()

    def clear_trajectory(self):
        """清除轨迹"""
        self.trajectory = None
        self.terrain_3d.clear_trajectory()
        self.top_down.clear_trajectory()
        self.result_label.setText('等待模拟...')


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = GolfPuttingSimulator()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
