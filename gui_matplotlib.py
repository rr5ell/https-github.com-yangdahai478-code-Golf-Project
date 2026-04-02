"""
高尔夫推杆轨迹模拟器 - Matplotlib 版 GUI
不依赖 pyqtgraph 和 PyQt6，使用 matplotlib 和 tkinter
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, FancyArrowPatch
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

from physics_engine import TerrainModel, PhysicsEngine, create_demo_point_cloud


class GolfPuttingSimulatorMatplotlib:
    """基于 Matplotlib 的推杆模拟器"""

    def __init__(self):
        super().__init__()

        # 数据
        self.points = None
        self.display_points = None
        self.terrain_model = None
        self.trajectory = None
        self.times = None
        self.start_position = np.array([0.0, 0.0, 0.0])

        # 模拟参数
        self.force_magnitude = 2.0  # m/s
        self.direction_angle = 0.0   # degrees
        self.friction_coeff = 0.10

        # 物理引擎
        self.physics = None

        # 初始化 GUI
        self.setup_gui()

    def setup_gui(self):
        """设置 GUI"""
        plt.style.use('default')

        # 创建图形窗口
        self.fig = plt.figure(figsize=(16, 10.5))
        self.fig.canvas.manager.set_window_title('高尔夫推杆轨迹模拟器')

        # 俯视图
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.set_title('俯视图 (点击设置起点)', fontsize=14, fontweight='bold')
        self.ax_2d.set_xlabel('X (m)')
        self.ax_2d.set_ylabel('Y (m)')
        self.ax_2d.grid(True, alpha=0.3)

        # 初始空图
        self.terrain_scatter = None
        self.start_marker = Circle((0, 0), 0.1, color='red', fill=True,
                                  alpha=0.8, label='起点', zorder=10)
        self.ax_2d.add_patch(self.start_marker)
        self.start_marker.set_visible(False)

        self.force_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->',
                                           mutation_scale=20, linewidth=3,
                                           color='red', zorder=9)
        self.ax_2d.add_patch(self.force_arrow)
        self.force_arrow.set_visible(False)

        self.trajectory_line, = self.ax_2d.plot([], [], 'r-', linewidth=3,
                                                 alpha=0.7, zorder=8, label='轨迹')

        # 3D 视图
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.ax_3d.set_title('3D 视图', fontsize=14, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        self.trajectory_line_3d, = self.ax_3d.plot([], [], [], 'r-', linewidth=3, label='轨迹')
        self.start_marker_3d, = self.ax_3d.plot([], [], [], 'ro', markersize=10, label='起点')

        # 设置控件区域
        self.setup_controls()

        # 绑定事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 信息显示
        self.info_text = self.fig.text(0.02, 0.95, '请先加载点云文件',
                                       fontsize=12, fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

    def setup_controls(self):
        """设置控制面板"""
        plt.subplots_adjust(bottom=0.25, left=0.05, right=0.95, top=0.90)

        # 加载文件按钮
        ax_load = plt.axes([0.05, 0.18, 0.12, 0.05])
        self.btn_load = Button(ax_load, '加载点云', hovercolor='lightblue')
        self.btn_load.on_clicked(self.load_point_cloud)

        # 演示地形按钮
        ax_demo = plt.axes([0.18, 0.18, 0.12, 0.05])
        self.btn_demo = Button(ax_demo, '演示地形', hovercolor='lightgreen')
        self.btn_demo.on_clicked(self.load_demo_terrain)

        # 力度滑块
        ax_force = plt.axes([0.35, 0.18, 0.25, 0.03])
        self.slider_force = Slider(ax_force, '力度 (m/s)', 0.1, 5.0,
                                   valinit=self.force_magnitude)
        self.slider_force.on_changed(self.on_force_change)

        # 方向滑块
        ax_angle = plt.axes([0.35, 0.13, 0.25, 0.03])
        self.slider_angle = Slider(ax_angle, '方向 (度)', 0, 360,
                                   valinit=self.direction_angle)
        self.slider_angle.on_changed(self.on_angle_change)

        # 摩擦系数滑块
        ax_friction = plt.axes([0.35, 0.08, 0.25, 0.03])
        self.slider_friction = Slider(ax_friction, '摩擦系数', 0.05, 0.20,
                                      valinit=self.friction_coeff)
        self.slider_friction.on_changed(self.on_friction_change)

        # 模拟按钮
        ax_sim = plt.axes([0.65, 0.12, 0.1, 0.06])
        self.btn_sim = Button(ax_sim, '开始模拟', hovercolor='lightgreen')
        self.btn_sim.on_clicked(self.run_simulation)

        # 清除按钮
        ax_clear = plt.axes([0.76, 0.12, 0.1, 0.06])
        self.btn_clear = Button(ax_clear, '清除轨迹', hovercolor='lightcoral')
        self.btn_clear.on_clicked(self.clear_trajectory)

        # 模式选择
        ax_mode = plt.axes([0.65, 0.03, 0.12, 0.12])
        self.radio_mode = RadioButtons(ax_mode, ('点击起点', '拖动方向'))
        self.radio_mode.on_clicked(self.on_mode_change)

        # 结果显示
        self.result_text = self.fig.text(0.78, 0.20, '',
                                         fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    def load_point_cloud(self, event=None):
        """加载点云文件"""
        root = tk.Tk()
        root.withdraw()

        filepath = filedialog.askopenfilename(
            title='选择点云文件',
            filetypes=[
                ('XYZ Files', '*.xyz'),
                ('Text Files', '*.txt'),
                ('CSV Files', '*.csv'),
                ('All Files', '*.*')
            ],
            initialdir=r'C:\Users\HS\Desktop\1117'
        )
        root.destroy()

        if not filepath:
            return

        try:
            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().split()

            if len(first_line) == 4:
                # 带索引格式
                data = np.loadtxt(filepath, encoding='utf-8')
                if data.shape[1] >= 4:
                    data = data[:, 1:4]
            else:
                # 纯坐标格式
                data = np.loadtxt(filepath, encoding='utf-8')

            self.points = data
            self.init_terrain()

        except Exception as e:
            self.show_error(f'加载点云失败: {str(e)}')

    def load_demo_terrain(self, event=None):
        """加载演示地形"""
        try:
            self.points = create_demo_point_cloud()
            self.init_terrain()
        except Exception as e:
            self.show_error(f'加载演示地形失败: {str(e)}')

    def init_terrain(self):
        """初始化地形模型"""
        if self.points is None:
            return

        try:
            # 降采样
            if len(self.points) > 5000:
                step = len(self.points) // 5000
                self.display_points = self.points[::step]
            else:
                self.display_points = self.points

            # 创建地形模型
            self.terrain_model = TerrainModel(self.points, smoothing_factor=0.1)

            # 创建物理引擎
            self.physics = PhysicsEngine(
                self.terrain_model,
                rolling_friction_coeff=self.friction_coeff,
                air_resistance=True,
                adaptive_timestep=True
            )

            # 更新可视化
            self.update_terrain_display()

            # 设置起始位置
            center = self.points.mean(axis=0)
            self.start_position = center.copy()
            self.start_position[2] = self.terrain_model.get_elevation(center[0], center[1])

            # 更新起点标记
            self.update_start_marker()

            # 清除轨迹
            self.clear_trajectory()

            self.update_info()
            print(f'地形加载成功！原始点数: {len(self.points)}, 显示点数: {len(self.display_points)}')

        except Exception as e:
            self.show_error(f'初始化地形失败: {str(e)}')

    def update_terrain_display(self):
        """更新地形显示"""
        if self.display_points is None:
            return

        # 俯视图
        if self.terrain_scatter:
            self.terrain_scatter.remove()

        z = self.display_points[:, 2]
        self.terrain_scatter = self.ax_2d.scatter(
            self.display_points[:, 0],
            self.display_points[:, 1],
            c=z, cmap='GnBu', s=10, alpha=0.6
        )

        # 3D 视图（大幅降采样）
        display_3d = self.display_points[::5] if len(self.display_points) > 1000 else self.display_points
        z = display_3d[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            normalized_z = (z - z_min) / (z_max - z_min)
        else:
            normalized_z = np.zeros_like(z)
        colors = plt.cm.GnBu(0.3 + 0.7 * normalized_z)

        self.ax_3d.clear()
        self.ax_3d.scatter(display_3d[:, 0], display_3d[:, 1], display_3d[:, 2],
                           c=colors, s=5, alpha=0.6)
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        # 重绘轨迹线
        self.trajectory_line_3d, = self.ax_3d.plot([], [], [], 'r-', linewidth=3)
        self.start_marker_3d, = self.ax_3d.plot([], [], [], 'ro', markersize=10)

        # 设置视图范围
        x_range = self.points[:, 0].max() - self.points[:, 0].min()
        y_range = self.points[:, 1].max() - self.points[:, 1].min()
        z_range = self.points[:, 2].max() - self.points[:, 2].min()

        self.ax_2d.set_xlim(self.points[:, 0].min() - x_range * 0.1,
                           self.points[:, 0].max() + x_range * 0.1)
        self.ax_2d.set_ylim(self.points[:, 1].min() - y_range * 0.1,
                           self.points[:, 1].max() + y_range * 0.1)

        self.ax_3d.set_xlim(self.points[:, 0].min(), self.points[:, 0].max())
        self.ax_3d.set_ylim(self.points[:, 1].min(), self.points[:, 1].max())
        self.ax_3d.set_zlim(self.points[:, 2].min(), self.points[:, 2].max())

        self.fig.canvas.draw()

    def update_start_marker(self):
        """更新起点标记"""
        self.start_marker.center = (self.start_position[0], self.start_position[1])
        self.start_marker.set_visible(True)
        self.start_marker_3d.set_data([self.start_position[0]], [self.start_position[1]])
        self.start_marker_3d.set_3d_properties([self.start_position[2]])
        self.update_force_arrow()

    def on_click(self, event):
        """鼠标点击事件"""
        if self.terrain_model is None or event.inaxes != self.ax_2d:
            return

        if self.radio_mode.value_selected == '点击起点':
            self.start_position[0] = event.xdata
            self.start_position[1] = event.ydata
            self.start_position[2] = self.terrain_model.get_elevation(event.xdata, event.ydata)

            self.update_start_marker()
            self.update_info()
            print(f'起点设置: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]')

    def on_mouse_move(self, event):
        """鼠标移动事件"""
        if self.terrain_model is None:
            return

        if self.radio_mode.value_selected != '拖动方向':
            return

        if event.inaxes != self.ax_2d or not event.button:
            return

        dx = event.xdata - self.start_position[0]
        dy = event.ydata - self.start_position[1]

        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360

        self.direction_angle = angle
        self.slider_angle.set_val(angle)
        self.update_force_arrow()
        self.update_info()

    def on_force_change(self, val):
        """力度滑块变化"""
        self.force_magnitude = val
        self.update_force_arrow()
        self.update_info()

    def on_angle_change(self, val):
        """角度滑块变化"""
        self.direction_angle = val
        self.update_force_arrow()
        self.update_info()

    def on_friction_change(self, val):
        """摩擦系数滑块变化"""
        self.friction_coeff = val
        if self.physics:
            self.physics.mu = val
        self.update_info()

    def on_mode_change(self, label):
        """模式选择变化"""
        if label == '点击起点':
            self.ax_2d.set_title('俯视图 (点击设置起点)', fontsize=14, fontweight='bold')
        else:
            self.ax_2d.set_title('俯视图 (拖动设置方向)', fontsize=14, fontweight='bold')
        self.fig.canvas.draw()

    def update_force_arrow(self):
        """更新力方向箭头"""
        angle_rad = np.radians(self.direction_angle)
        arrow_length = self.force_magnitude * 0.3

        dx = arrow_length * np.cos(angle_rad)
        dy = arrow_length * np.sin(angle_rad)

        self.force_arrow.set_positions(
            (self.start_position[0], self.start_position[1]),
            (self.start_position[0] + dx, self.start_position[1] + dy)
        )
        self.force_arrow.set_visible(True)
        self.fig.canvas.draw_idle()

    def update_info(self):
        """更新信息显示"""
        if self.terrain_model is None:
            info = '请先加载点云文件'
        else:
            info = f'起点: [{self.start_position[0]:.2f}, {self.start_position[1]:.2f}, {self.start_position[2]:.3f}]\n'
            info += f'力度: {self.force_magnitude:.2f} m/s\n'
            info += f'方向: {self.direction_angle:.1f}°\n'
            info += f'摩擦系数: {self.friction_coeff:.3f}'
        self.info_text.set_text(info)

    def run_simulation(self, event=None):
        """运行模拟"""
        if self.terrain_model is None:
            self.show_error('请先加载地形')
            return

        print(f'\n开始模拟...')
        print(f'起点: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]')
        print(f'力度: {self.force_magnitude:.2f} m/s, 方向: {self.direction_angle:.1f}°')

        # 计算初始速度
        angle_rad = np.radians(self.direction_angle)
        initial_velocity = np.array([
            self.force_magnitude * np.cos(angle_rad),
            self.force_magnitude * np.sin(angle_rad),
            0.0
        ])

        # 运行模拟
        try:
            self.trajectory, self.times = self.physics.simulate(self.start_position, initial_velocity)
            print(f'模拟完成！轨迹点数: {len(self.trajectory)}, 时间: {self.times[-1]:.3f}s')
        except Exception as e:
            self.show_error(f'模拟失败: {str(e)}')
            return

        # 更新轨迹显示
        if len(self.trajectory) > 0:
            self.trajectory_line.set_data(self.trajectory[:, 0], self.trajectory[:, 1])
            self.trajectory_line_3d.set_data(self.trajectory[:, 0], self.trajectory[:, 1])
            self.trajectory_line_3d.set_3d_properties(self.trajectory[:, 2])

            # 计算统计
            total_distance = np.sum(np.linalg.norm(
                np.diff(self.trajectory[:, :2], axis=0), axis=1
            ))
            final_pos = self.trajectory[-1]
            displacement = np.linalg.norm(final_pos[:2] - self.start_position[:2])

            result = f'结果:\n'
            result += f'点数: {len(self.trajectory)}\n'
            result += f'时间: {self.times[-1]:.3f}s\n'
            result += f'距离: {total_distance:.3f}m\n'
            result += f'位移: {displacement:.3f}m\n'
            result += f'终点: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.3f}]'
            self.result_text.set_text(result)

        self.fig.canvas.draw()

    def clear_trajectory(self, event=None):
        """清除轨迹"""
        self.trajectory = None
        self.times = None
        self.trajectory_line.set_data([], [])
        self.trajectory_line_3d.set_data([], [])
        self.trajectory_line_3d.set_3d_properties([])
        self.result_text.set_text('')
        self.fig.canvas.draw()

    def show_error(self, message):
        """显示错误信息"""
        print(f'错误: {message}')
        # 使用 tkinter 显示错误对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('错误', message)
        root.destroy()

    def show(self):
        """显示界面"""
        print('\n' + '=' * 60)
        print('高尔夫推杆轨迹模拟器')
        print('=' * 60)
        print('操作说明:')
        print('  - 点击"加载点云"或"演示地形"加载地形')
        print('  - 选择"点击起点"模式，在俯视图上点击设置位置')
        print('  - 选择"拖动方向"模式，拖动鼠标设置方向')
        print('  - 使用滑块调整力度、方向和摩擦系数')
        print('  - 点击"开始模拟"运行模拟')
        print('=' * 60)
        plt.show()


def main():
    """主函数"""
    app = GolfPuttingSimulatorMatplotlib()
    app.show()


if __name__ == '__main__':
    main()
