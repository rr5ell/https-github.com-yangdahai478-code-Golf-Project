"""
高尔夫推杆轨迹模拟器 - 交互式界面（基于 matplotlib）
支持鼠标点击选择位置、拖动调整力度和方向
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Circle, FancyArrowPatch

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
        self.points = np.asarray(points, dtype=np.float64)

        print(f"加载点云: {len(self.points)} 个点")
        print(f"X范围: [{self.points[:, 0].min():.3f}, {self.points[:, 0].max():.3f}]")
        print(f"Y范围: [{self.points[:, 1].min():.3f}, {self.points[:, 1].max():.3f}]")
        print(f"Z范围: [{self.points[:, 2].min():.3f}, {self.points[:, 2].max():.3f}]")

        # 初始化地形
        print("\n初始化地形模型...")
        self.terrain = TerrainModel(self.points, smoothing_factor=smoothing_factor)
        print("地形模型初始化成功!")

        # 初始化物理引擎
        print("初始化物理引擎...")
        self.physics = PhysicsEngine(
            self.terrain,
            rolling_friction_coeff=0.10,
            air_resistance=True,
            adaptive_timestep=True
        )
        print("物理引擎初始化成功!")

        # 模拟参数
        self.start_position = self.points.mean(axis=0)
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
        self.fig.canvas.manager.set_window_title('高尔夫推杆轨迹模拟器')

        # 俯视图
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_2d.set_aspect('equal')
        self.ax_2d.set_title('俯视图 (点击设置起点)', fontsize=14, fontweight='bold')
        self.ax_2d.set_xlabel('X (m)')
        self.ax_2d.set_ylabel('Y (m)')
        self.ax_2d.grid(True, alpha=0.3)

        # 绘制地形
        z = self.points[:, 2]
        scatter = self.ax_2d.scatter(self.points[:, 0], self.points[:, 1],
                                     c=z, cmap='GnBu', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=self.ax_2d, label='高度 Z (m)')

        # 起点标记
        self.start_marker = Circle((self.start_position[0], self.start_position[1]),
                                   0.1, color='red', fill='r', alpha=0.8,
                                   label='起点', zorder=10)
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
        self.ax_3d.set_title('3D 视图', fontsize=14, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        # 绘制 3D 地形
        z = self.points[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            normalized_z = (z - z_min) / (z_max - z_min)
        else:
            normalized_z = np.zeros_like(z)
        colors = plt.cm.GnBu(0.3 + 0.7 * normalized_z)
        self.ax_3d.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                           c=colors, s=5, alpha=0.6)

        # 3D 轨迹线
        self.trajectory_line_3d, = self.ax_3d.plot([], [], [], 'r-', linewidth=3)

        # 3D 起点标记
        self.start_marker_3d, = self.ax_3d.plot([self.start_position[0]],
                                                 [self.start_position[1]],
                                                 [self.start_position[2]],
                                                 'ro', markersize=10, label='起点')

        # 调整视图范围
        x_range = self.points[:, 0].max() - self.points[:, 0].min()
        y_range = self.points[:, 1].max() - self.points[:, 1].min()
        z_range = self.points[:, 2].max() - self.points[:, 2].min()

        self.ax_2d.set_xlim(self.points[:, 0].min() - x_range * 0.1,
                           self.points[:, 0].max() + x_range * 0.1)
        self.ax_2d.set_ylim(self.points[:, 1].min() - y_range * 0.1,
                           self.points[:, 1].max() + y_range * 0.1)

        self.ax_3d.set_xlim(self.points[:, 0].min() - x_range * 0.1,
                           self.points[:, 0].max() + x_range * 0.1)
        self.ax_3d.set_ylim(self.points[:, 1].min() - y_range * 0.1,
                           self.points[:, 1].max() + y_range * 0.1)
        self.ax_3d.set_zlim(self.points[:, 2].min(), self.points[:, 2].max())

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
        self.slider_force = Slider(ax_force, '力度 (m/s)', 0.1, 5.0,
                                    valinit=self.force_magnitude)
        self.slider_force.on_changed(self.on_force_change)

        # 方向滑块
        ax_angle = plt.axes([0.15, 0.10, 0.25, 0.03])
        self.slider_angle = Slider(ax_angle, '方向 (°)', 0, 360,
                                    valinit=self.direction_angle)
        self.slider_angle.on_changed(self.on_angle_change)

        # 摩擦系数滑块
        ax_friction = plt.axes([0.15, 0.05, 0.25, 0.03])
        self.slider_friction = Slider(ax_friction, '摩擦系数', 0.05, 0.20,
                                      valinit=0.10)
        self.slider_friction.on_changed(self.on_friction_change)

        # 模拟按钮
        ax_sim = plt.axes([0.50, 0.12, 0.1, 0.06])
        self.btn_sim = Button(ax_sim, '开始模拟', hovercolor='lightgreen')
        self.btn_sim.on_clicked(self.run_simulation)

        # 清除按钮
        ax_clear = plt.axes([0.50, 0.04, 0.1, 0.06])
        self.btn_clear = Button(ax_clear, '清除轨迹', hovercolor='lightcoral')
        self.btn_clear.on_clicked(self.clear_trajectory)

        # 模式选择
        ax_mode = plt.axes([0.70, 0.05, 0.15, 0.12])
        self.radio_mode = RadioButtons(ax_mode, ('点击设置起点', '拖动设置方向'))
        self.radio_mode.on_clicked(self.on_mode_change)

        # 结果显示
        self.result_text = self.fig.text(0.70, 0.20, '', fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    def on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes != self.ax_2d:
            return

        if self.radio_mode.value_selected == '点击设置起点':
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
            print(f"起点设置为: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]")

    def on_mouse_move(self, event):
        """鼠标移动事件 - 用于拖动设置方向"""
        if self.radio_mode.value_selected != '拖动设置方向':
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
        if label == '点击设置起点':
            self.ax_2d.set_title('俯视图 (点击设置起点)', fontsize=14, fontweight='bold')
        else:
            self.ax_2d.set_title('俯视图 (按住拖动设置方向)', fontsize=14, fontweight='bold')
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
        info = f'起点: [{self.start_position[0]:.2f}, {self.start_position[1]:.2f}, {self.start_position[2]:.3f}]\n'
        info += f'力度: {self.force_magnitude:.2f} m/s\n'
        info += f'方向: {self.direction_angle:.1f}°\n'
        info += f'摩擦系数: {self.physics.mu:.3f}'
        self.info_text.set_text(info)

    def run_simulation(self, event=None):
        """运行模拟"""
        print("\n开始模拟...")
        print(f"起点: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]")
        print(f"力度: {self.force_magnitude:.2f} m/s, 方向: {self.direction_angle:.1f}°")

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
            print(f"模拟完成! 轨迹点数: {len(self.trajectory)}, 消耗时间: {self.times[-1]:.3f}s")
        except Exception as e:
            print(f"模拟失败: {e}")
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

            result = f'模拟结果:\n'
            result += f'轨迹点数: {len(self.trajectory)}\n'
            result += f'模拟时间: {self.times[-1]:.3f}s\n'
            result += f'总距离: {total_distance:.3f}m\n'
            result += f'净位移: {displacement:.3f}m\n'
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

    def show(self):
        """显示界面"""
        plt.show()


def load_xyz_file(filepath):
    """加载 XYZ 格式的点云文件"""
    # 先读取一行检测格式
    with open(filepath, 'r') as f:
        first_line = f.readline().strip().split()

    if len(first_line) == 4:
        # 带索引的格式：index X Y Z
        data = np.loadtxt(filepath)
        points = data[:, 1:4]  # 取 X Y Z
    else:
        # 纯坐标格式：X Y Z
        points = np.loadtxt(filepath)

    return points


def main():
    """主函数"""
    print("=" * 60)
    print("高尔夫推杆轨迹模拟器 - 交互式界面")
    print("=" * 60)

    # 加载点云文件
    xyz_file = r"C:\Users\HS\Desktop\1117\FirstFrame_20251117_134654.xyz"
    print(f"\n正在加载点云文件: {xyz_file}")

    try:
        points = load_xyz_file(xyz_file)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # Simulator
    simulator = InteractivePuttingSimulator(points)
    print("\n" + "=" * 60)
    print("界面已启动!")
    print("操作说明:")
    print("  - 选择'点击设置起点'模式，在俯视图点击设置起点")
    print("  - 选择'拖动设置方向'模式，按住鼠标拖动设置方向")
    print("  - 使用滑块调整力度、方向和摩擦系数")
    print("  - 点击'开始模拟'按钮运行模拟")
    print("=" * 60)

    simulator.show()


if __name__ == '__main__':
    main()
