"""
加载 XYZ 点云文件并运行模拟
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from physics_engine import TerrainModel, PhysicsEngine


def load_xyz_file(filepath):
    """
    加载 XYZ 格式的点云文件

    格式：每行可能是 4 列（索引 + X Y Z）或 3 列（X Y Z）
    """
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

    print(f"加载点云: {len(points)} 个点")
    print(f"X范围: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"Y范围: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    return points


def visualize_terrain_3d(points, trajectory=None):
    """3D 可视化地形和轨迹"""
    fig = plt.figure(figsize=(14, 8))

    # 3D 视图
    ax3d = fig.add_subplot(121, projection='3d')

    # 绘制点云
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        normalized_z = (z - z_min) / (z_max - z_min)
    else:
        normalized_z = np.zeros_like(z)

    # 颜色渐变
    colors = plt.cm.GnBu(0.3 + 0.7 * normalized_z)
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                 c=colors, s=2, alpha=0.6, label='地形')

    # 绘制轨迹
    if trajectory is not None and len(trajectory) > 0:
        ax3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                  'r-', linewidth=3, label='轨迹')
        ax3d.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                    c='b', s=100, marker='o', label='起点')
        ax3d.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                    c='g', s=100, marker='*', label='终点')

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D 视图')
    ax3d.legend()

    # 俯视图
    ax2d = fig.add_subplot(122)
    ax2d.set_aspect('equal')

    # 绘制点云（按高度着色）
    scatter = ax2d.scatter(points[:, 0], points[:, 1], c=z,
                          cmap='GnBu', s=5, alpha=0.6)
    plt.colorbar(scatter, ax=ax2d, label='高度 Z (m)')

    # 绘制轨迹
    if trajectory is not None and len(trajectory) > 0:
        ax2d.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=3)
        ax2d.scatter(trajectory[0, 0], trajectory[0, 1], c='b', s=100, marker='o', label='起点')
        ax2d.scatter(trajectory[-1, 0], trajectory[-1, 1], c='g', s=100, marker='*', label='终点')

        # 绘制方向箭头
        if len(trajectory) > 1:
            start = trajectory[0]
            next_point = trajectory[1]
            direction = next_point - start
            direction = direction / (np.linalg.norm(direction) + 1e-10) * 0.5
            ax2d.arrow(start[0], start[1], direction[0], direction[1],
                      head_width=0.1, head_length=0.1, fc='r', ec='r')

    ax2d.set_xlabel('X (m)')
    ax2d.set_ylabel('Y (m)')
    ax2d.set_title('俯视图')
    ax2d.grid(True, alpha=0.3)
    ax2d.legend()

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("高尔夫推杆轨迹模拟器 - XYZ 点云加载")
    print("=" * 60)

    # 加载点云文件
    xyz_file = r"C:\Users\HS\Desktop\1117\FirstFrame_20251117_134654.xyz"
    print(f"\n正在加载点云文件: {xyz_file}")

    try:
        points = load_xyz_file(xyz_file)
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 初始化地形模型
    print("\n初始化地形模型...")
    try:
        terrain = TerrainModel(points, smoothing_factor=0.1)
        print("地形模型初始化成功!")
    except Exception as e:
        print(f"地形模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建物理引擎
    print("\n初始化物理引擎...")
    physics = PhysicsEngine(
        terrain,
        rolling_friction_coeff=0.10,
        air_resistance=True,
        adaptive_timestep=True
    )
    print("物理引擎初始化成功!")

    # 设置起始位置（使用点云中心）
    center = points.mean(axis=0)
    start_pos = center.copy()
    start_pos[2] = terrain.get_elevation(center[0], center[1])

    print(f"\n起始位置: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")

    # 设置推杆参数
    force_magnitude = 2.0  # m/s
    direction_angle = 30.0  # 度
    angle_rad = np.radians(direction_angle)

    initial_velocity = np.array([
        force_magnitude * np.cos(angle_rad),
        force_magnitude * np.sin(angle_rad),
        0.0
    ])

    print(f"推杆力度: {force_magnitude:.2f} m/s")
    print(f"方向角度: {direction_angle:.1f}°")
    print(f"初始速度: [{initial_velocity[0]:.3f}, {initial_velocity[1]:.3f}, {initial_velocity[2]:.3f}] m/s")

    # 运行模拟
    print("\n开始模拟...")
    try:
        trajectory, times = physics.simulate(start_pos, initial_velocity)
        print(f"模拟完成! 轨迹点数: {len(trajectory)}, 消耗时间: {times[-1]:.3f}s")
    except Exception as e:
        print(f"模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 显示结果
    print("\n" + "=" * 60)
    print("模拟结果")
    print("=" * 60)

    if len(trajectory) > 0:
        total_distance = np.sum(np.linalg.norm(
            np.diff(trajectory[:, :2], axis=0), axis=1
        ))
        print(f"总行驶距离: {total_distance:.3f} 米")

        final_pos = trajectory[-1]
        displacement = np.linalg.norm(final_pos[:2] - start_pos[:2])
        print(f"净位移: {displacement:.3f} 米")
        print(f"最终位置: X={final_pos[0]:.3f}, Y={final_pos[1]:.3f}, Z={final_pos[2]:.3f}")

        # 显示关键点
        print("\n轨迹关键点 (每隔10%):")
        step = max(1, len(trajectory) // 10)
        for i in range(0, len(trajectory), step):
            pos = trajectory[i]
            t = times[i]
            print(f"  t={t:.2f}s: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # 可视化
    print("\n启动可视化窗口...")
    visualize_terrain_3d(points, trajectory)
    print("\n完成!")


if __name__ == '__main__':
    main()
