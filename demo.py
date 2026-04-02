"""
演示脚本 - 无GUI运行模拟
"""

import numpy as np
from physics_engine import create_demo_point_cloud, TerrainModel, PhysicsEngine


def main():
    print("=" * 50)
    print("高尔夫推杆轨迹模拟器 - 演示")
    print("=" * 50)

    # 1. 创建演示地形
    print("\n1. 创建地形...")
    points = create_demo_point_cloud()
    print(f"   点云数量: {len(points)}")
    print(f"   X范围: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"   Y范围: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"   Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # 2. 初始化地形模型
    print("\n2. 初始化地形模型...")
    terrain = TerrainModel(points, smoothing_factor=0.1)
    print("   地形模型初始化成功")

    # 3. 创建物理引擎
    print("\n3. 初始化物理引擎...")
    physics = PhysicsEngine(
        terrain,
        rolling_friction_coeff=0.10,
        air_resistance=True,
        adaptive_timestep=True
    )
    print("   物理引擎初始化成功")

    # 4. 设置模拟参数
    print("\n4. 设置模拟参数...")
    start_pos = np.array([0.0, 0.0, terrain.get_elevation(0.0, 0.0)])
    print(f"   起始位置: [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.3f}]")

    force_magnitude = 2.0  # m/s
    direction_angle = 45.0  # 度
    angle_rad = np.radians(direction_angle)
    initial_velocity = np.array([
        force_magnitude * np.cos(angle_rad),
        force_magnitude * np.sin(angle_rad),
        0.0
    ])
    print(f"   初始速度: [{initial_velocity[0]:.3f}, {initial_velocity[1]:.3f}, {initial_velocity[2]:.3f}] m/s")
    print(f"   推杆力度: {force_magnitude:.2f} m/s")
    print(f"   方向角度: {direction_angle:.1f}°")

    # 5. 运行模拟
    print("\n5. 运行模拟...")
    trajectory, times = physics.simulate(start_pos, initial_velocity)

    # 6. 显示结果
    print("\n" + "=" * 50)
    print("模拟结果")
    print("=" * 50)
    print(f"轨迹点数: {len(trajectory)}")
    print(f"模拟时间: {times[-1]:.3f} 秒")

    total_distance = np.sum(np.linalg.norm(
        np.diff(trajectory[:, :2], axis=0), axis=1
    ))
    print(f"总行驶距离: {total_distance:.3f} 米")

    final_pos = trajectory[-1]
    start_pos_2d = trajectory[0, :2]
    displacement = np.linalg.norm(final_pos[:2] - start_pos_2d)
    print(f"净位移: {displacement:.3f} 米")
    print(f"最终位置: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}, Z={final_pos[2]:.3f}")

    # 显示每米距离
    print("\n轨迹关键点:")
    for i, (pos, t) in enumerate(zip(trajectory[::len(trajectory)//10], times[::len(times)//10])):
        print(f"  t={t:.2f}s: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.3f}]")

    print("\n模拟完成!")


if __name__ == '__main__':
    main()
