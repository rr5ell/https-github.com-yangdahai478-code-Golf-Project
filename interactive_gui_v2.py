"""
高尔夫推杆轨迹模拟器 - 优化版交互式界面
- 性能优化：降采样点云显示
- 修复乱码：使用英文界面
- 文件选择：支持手动选择点云文件
"""

import sys
import numpy as np
import matplotlib
matplotlib.pyplot.switch_backend('TkAgg')  # 使用 TkAgg 后端
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, FancyArrowPatch
import tkinter as tk
from tkinter import filedialog

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

from physics_engine import TerrainModel, PhysicsEngine


class InteractivePuttingSimulator:
    """交互式推杆模拟器"""

    def __init__(self, points, smoothing_factor=0.1):
        """
        初始化模拟器

        Args:
            points: 点云数据 (N x 3)
            smoothing_factor: 地形平滑因子
        """
        self.original_points = np.asarray(points, dtype=np.float64)

        # 降采样优化性能
        if len(self.original_points) > 5000:
            self.display_points = self.original_points[::len(self.original_points)//5000]
        else:
            self.display_points = self.original_points

        print(f"Loaded points: {len(self.original_points)} total, {len(self.display_points)} displayed")

        # 初始化地形（使用全部点）
        print("\nInitializing terrain model...")
        self.terrain = TerrainModel(self.original_points, smoothing_factor=smoothing_factor)
        print("Terrain model initialized!")

        # 初始化物理引擎
        print("Initializing physics engine...")
        self.physics = PhysicsEngine(
            self.terrain,
            rolling_friction_coeff=0.10,
            air_resistance=True,
            adaptive_timestep=True
        )
        print("Physics engine initialized!")

        # 模拟参数
        self.start_position = self.original_points.mean(axis=0)
        self.start_position[2] = self.terrain.get_elevation(
            self.start_position[0], self.start_position[1]
        )

        self.force_magnitude = 2.0  # m/s
        self.direction_angle = 0.0   # degrees

        self.trajectory = None
        self.times = None

        # 创建图形
        self.setup_gui()

    def setup_gui(self):
        """设置 GUI"""
        plt.style.use('default')

        # 创建图形
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Golf Putting Trajectory Simulator')

        # 俯视图
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.set_title('Top View (Click to set start position)', fontsize=14, fontweight='bold')
        self.ax_2d.set_xlabel('X (m)')
        self.ax_2d.set_ylabel('Y (m)')
        self.ax_2d.grid(True, alpha=0.3)

        # 绘制地形（使用降采样后的点）
        z = self.display_points[:, 2]
        scatter = self.ax_2d.scatter(self.display_points[:, 0], self.display_points[:, 1],
                                     c=z, cmap='GnBu', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=self.ax_2d, label='Elevation Z (m)')

        # 起点标记
        self.start_marker = Circle((self.start_position[0], self.start_position[1]),
                                   0.1, color='red', fill='r', alpha=0.8,
                                   label='Start', zorder=10)
        self.ax_2d.add_patch(self.start_marker)

        # 力方向箭头
        self.force_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->',
                                            mutation_scale=20, linewidth=3,
                                            color='red', zorder=9)
        self.ax_2d.add_patch(self.force_arrow)

        # 轨迹线
        self.trajectory_line, = self.ax_2d.plot([], [], 'r-', linewidth=3,
                                                 alpha=0.7, zorder=8)

        # 3D 视图
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.ax_3d.set_title('3D View', fontsize=14, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        # 绘制 3D 地形（大幅降采样）
        display_3d_points = self.display_points[::5] if len(self.display_points) > 1000 else self.display_points
        z = display_3d_points[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            normalized_z = (z - z_min) / (z_max - z_min)
        else:
            normalized_z = np.zeros_like(z)
        colors = plt.cm.GnBu(0.3 + 0.7 * normalized_z)
        self.ax_3d.scatter(display_3d_points[:, 0], display_3d_points[:, 1], display_3d_points[:, 2],
                           c=colors, s=5, alpha=0.6)

        # 3D 轨迹线
        self.trajectory_line_3d, = self.ax_3d.plot([], [], [], 'r-', linewidth=3)

        # 3D 起点标记
        self.start_marker_3d, = self.ax_3d.plot([self.start_position[0]],
                                                 [self.start_position[1]],
                                                 [self.start_position[2]],
                                                 'ro', markersize=10, label='Start')

        # 调整视图范围
        x_range = self.original_points[:, 0].max() - self.original_points[:, 0].min()
        y_range = self.original_points[:, 1].max() - self.original_points[:, 1].min()
        z_range = self.original_points[:, 2].max() - self.original_points[:, 2].min()

        self.ax_2d.set_xlim(self.original_points[:, 0].min() - x_range * 0.1,
                           self.original_points[:, 0].max() + x_range * 0.1)
        self.ax_2d.set_ylim(self.original_points[:, 1].min() - y_range * 0.1,
                           self.original_points[:, 1].max() + y_range * 0.1)

        self.ax_3d.set_xlim(self.original_points[:, 0].min() - x_range * 0.1,
                           self.original_points[:, 0].max() + x_range * 0.1)
        self.ax_3d.set_ylim(self.original_points[:, 1].min() - y_range * 0.1,
                           self.original_points[:, 1].max() + y_range * 0.1)
        self.ax_3d.set_zlim(self.original_points[:, 2].min(), self.original_points[:, 2].max())

        # 添加控件区域
        self.setup_controls()

        # 绑定鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        # 初始显示
        self.update_force_arrow()
        plt.tight_layout()

        # 信息文本
        self.info_text = self.fig.text(0.02, 0.95, '', fontsize=12,
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.update_info()

    def setup_controls(self):
        """设置控制面板"""
        # 为控件留出空间
        plt.subplots_adjust(bottom=0.25, left=0.05, right=0.95, top=0.90)

        # 力度滑块
        ax_force = plt.axes([0.15, 0.15, 0.25, 0.03])
        self.slider_force = Slider(ax_force, 'Force (m/s)', 0.1, 5.0,
                                    valinit=self.force_magnitude)
        self.slider_force.on_changed(self.on_force_change)

        # 方向滑块
        ax_angle = plt.axes([0.15, 0.10, 0.25, 0.03])
        self.slider_angle = Slider(ax_angle, 'Direction (deg)', 0, 360,
                                    valinit=self.direction_angle)
        self.slider_angle.on_changed(self.on_angle_change)

        # 摩擦系数滑块
        ax_friction = plt.axes([0.15, 0.05, 0.25, 0.03])
        self.slider_friction = Slider(ax_friction, 'Friction', 0.05, 0.20,
                                      valinit=0.10)
        self.slider_friction.on_changed(self.on_friction_change)

        # 模拟按钮
        ax_sim = plt.axes([0.50, 0.12, 0.1, 0.06])
        self.btn_sim = Button(ax_sim, 'Simulate', hovercolor='lightgreen')
        self.btn_sim.on_clicked(self.run_simulation)

        # 清除按钮
        ax_clear = plt.axes([0.50, 0.04, 0.1, 0.06])
        self.btn_clear = Button(ax_clear, 'Clear', hovercolor='lightcoral')
        self.btn_clear.on_clicked(self.clear_trajectory)

        # 模式选择
        ax_mode = plt.axes([0.70, 0.05, 0.15, 0.12])
        self.radio_mode = RadioButtons(ax_mode, ('Click Start', 'Drag Direction'))
        self.radio_mode.on_clicked(self.on_mode_change)

        # 结果显示
        self.result_text = self.fig.text(0.70, 0.20, '', fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    def on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes != self.ax_2d:
            return

        if self.radio_mode.value_selected == 'Click Start':
            # 设置起点
            self.start_position[0] = event.xdata
            self.start_position[1] = event.ydata
            self.start_position[2] = self.terrain.get_elevation(event.xdata, event.ydata)

            # 更新起点标记
            self.start_marker.center = (event.xdata, event.ydata)
            self.start_marker_3d.set_data([event.xdata], [event.ydata])
            self.start_marker_3d.set_3d_properties([self.start_position[2]])

            self.update_force_arrow()
            self.update_info()
            print(f"Start position: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]")

    def on_mouse_move(self, event):
        """鼠标移动事件 - 用于拖动设置方向"""
        if self.radio_mode.value_selected != 'Drag Direction':
            return

        if event.inaxes != self.ax_2d or not event.button:
            return

        # 计算方向角度
        dx = event.xdata - self.start_position[0]
        dy = event.ydata - self.start_position[1]

        angle = np.degrees(np.arctan2(dy, dx))
        if angle < 0:
            angle += 360

        self.direction_angle = angle
        self.slider_angle.set_val(angle)
        self.update_force_arrow()

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
        self.physics.mu = val
        self.update_info()

    def on_mode_change(self, label):
        """模式选择变化"""
        if label == 'Click Start':
            self.ax_2d.set_title('Top View (Click to set start position)', fontsize=14, fontweight='bold')
        else:
            self.ax_2d.set_title('Top View (Hold and drag to set direction)', fontsize=14, fontweight='bold')
        self.fig.canvas.draw()

    def update_force_arrow(self):
        """更新力方向箭头"""
        angle_rad = np.radians(self.direction_angle)
        arrow_length = self.force_magnitude * 0.3  # 缩放箭头长度

        dx = arrow_length * np.cos(angle_rad)
        dy = arrow_length * np.sin(angle_rad)

        self.force_arrow.set_positions(
            (self.start_position[0], self.start_position[1]),
            (self.start_position[0] + dx, self.start_position[1] + dy)
        )
        self.fig.canvas.draw_idle()

    def update_info(self):
        """更新信息显示"""
        info = f'Start: [{self.start_position[0]:.2f}, {self.start_position[1]:.2f}, {self.start_position[2]:.3f}]\n'
        info += f'Force: {self.force_magnitude:.2f} m/s\n'
        info += f'Direction: {self.direction_angle:.1f}°\n'
        info += f'Friction: {self.physics.mu:.3f}'
        self.info_text.set_text(info)

    def run_simulation(self, event=None):
        """运行模拟"""
        print("\nStarting simulation...")
        print(f"Start: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]")
        print(f"Force: {self.force_magnitude:.2f} m/s, Direction: {self.direction_angle:.1f}°")

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
            print(f"Simulation complete! Points: {len(self.trajectory)}, Time: {self.times[-1]:.3f}s")
        except Exception as e:
            print(f"Simulation failed: {e}")
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

            result = f'Results:\n'
            result += f'Points: {len(self.trajectory)}\n'
            result += f'Time: {self.times[-1]:.3f}s\n'
            result += f'Distance: {total_distance:.3f}m\n'
            result += f'Displacement: {displacement:.3f}m\n'
            result += f'End: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.3f}]'
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

    def show(self):
        """显示界面"""
        plt.show()


def load_xyz_file(filepath):
    """加载 XYZ 格式的点云文件"""
    print(f"Loading: {filepath}")

    try:
        # 先读取一行检测格式
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip().split()

        if len(first_line) == 4:
            # 带索引的格式：index X Y Z
            data = np.loadtxt(filepath, encoding='utf-8')
            points = data[:, 1:4]  # 取 X Y Z
        else:
            # 纯坐标格式：X Y Z
            points = np.loadtxt(filepath, encoding='utf-8')

        print(f"Loaded {len(points)} points")
        print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

        return points
    except Exception as e:
        print(f"Failed to load: {e}")
        return None


def select_file_dialog():
    """打开文件选择对话框"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    filepath = filedialog.askopenfilename(
        title='Select Point Cloud File',
        filetypes=[
            ('XYZ Files', '*.xyz'),
            ('Text Files', '*.txt'),
            ('CSV Files', '*.csv'),
            ('All Files', '*.*')
        ],
        initialdir=r'C:\Users\HS\Desktop\1117'
    )

    root.destroy()
    return filepath


def main():
    """主函数"""
    print("=" * 60)
    print("Golf Putting Trajectory Simulator - Interactive GUI")
    print("=" * 60)

    # 选择点云文件
    print("\nPlease select a point cloud file...")
    filepath = select_file_dialog()

    if not filepath:
        print("No file selected. Using demo data...")
        from physics_engine import create_demo_point_cloud
        points = create_demo_point_cloud()
    else:
        points = load_xyz_file(filepath)

        if points is None:
            print("Failed to load file. Using demo data...")
            from physics_engine import create_demo_point_cloud
            points = create_demo_point_cloud()

    # Simulator
    simulator = InteractivePuttingSimulator(points)
    print("\n" + "=" * 60)
    print("Interface launched!")
    print("Instructions:")
    print("  - Select 'Click Start' mode, click on top view to set position")
    print("  - Select 'Drag Direction' mode, hold and drag to set direction")
    print("  - Use sliders to adjust force, direction, and friction")
    print("  - Click 'Simulate' button to run simulation")
    print("=" * 60)

    simulator.show()


if __name__ == '__main__':
    main()
