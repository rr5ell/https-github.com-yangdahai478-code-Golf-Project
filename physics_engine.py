"""
高尔夫推杆物理模拟引擎
基于经典力学和地形分析
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree, cKDTree
import warnings

warnings.filterwarnings('ignore')


class TerrainModel:
    """
    地形模型
    从点云数据构建可查询的高程场

    主要问题及解决方案：
    1. 点云噪点 -> 使用KDTree和局部加权平均
    2. 密度不均 -> 自适应邻域半径
    3. 边界问题 -> 边界检测和限制
    """

    def __init__(self, points, smoothing_factor=0.1):
        """
        初始化地形模型

        Args:
            points: Nx3 array, [x, y, z] 坐标
            smoothing_factor: 平滑因子，0-1之间
        """
        self.points = np.asarray(points, dtype=np.float64)

        if len(self.points.shape) != 2 or self.points.shape[1] != 3:
            raise ValueError("点云必须是Nx3数组")

        # 标准化坐标，避免数值问题
        self.min_coords = self.points.min(axis=0)
        self.max_coords = self.points.max(axis=0)
        self.normalized_points = self._normalize_points(self.points)

        # 构建KD树用于快速查询
        self.kdtree = cKDTree(self.normalized_points[:, :2])

        # 自适应计算邻域半径
        distances, _ = self.kdtree.query(self.normalized_points[:, :2], k=6)
        self.neighbor_radius = np.percentile(distances[:, -1], 50) * 2
        self.smoothing_factor = smoothing_factor

        # 计算边界
        self._compute_boundary()

    def _normalize_points(self, points):
        """归一化坐标到[0,1]范围"""
        normalized = points.copy()
        normalized[:, 0] = (points[:, 0] - self.min_coords[0]) / (self.max_coords[0] - self.min_coords[0])
        normalized[:, 1] = (points[:, 1] - self.min_coords[1]) / (self.max_coords[1] - self.min_coords[1])
        normalized[:, 2] = (points[:, 2] - self.min_coords[2]) / (self.max_coords[2] - self.min_coords[2])
        return normalized

    def _denormalize_point(self, point):
        """反归一化坐标"""
        x = point[0] * (self.max_coords[0] - self.min_coords[0]) + self.min_coords[0]
        y = point[1] * (self.max_coords[1] - self.min_coords[1]) + self.min_coords[1]
        z = point[2] * (self.max_coords[2] - self.min_coords[2]) + self.min_coords[2]
        return np.array([x, y, z])

    def _compute_boundary(self):
        """计算凸包边界用于碰撞检测"""
        # 使用角度排序法获取边界点
        centroid = self.normalized_points[:, :2].mean(axis=0)
        angles = np.arctan2(self.normalized_points[:, 1] - centroid[1],
                            self.normalized_points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        self.boundary_points = self.normalized_points[sorted_indices][:, :2]

    def is_inside_boundary(self, x, y):
        """检查点是否在地形边界内"""
        # 归一化输入坐标
        if self.max_coords[0] > self.min_coords[0]:
            norm_x = (x - self.min_coords[0]) / (self.max_coords[0] - self.min_coords[0])
        else:
            norm_x = 0.0

        if self.max_coords[1] > self.min_coords[1]:
            norm_y = (y - self.min_coords[1]) / (self.max_coords[1] - self.min_coords[1])
        else:
            norm_y = 0.0

        # 使用射线法
        point = np.array([norm_x, norm_y])
        inside = False
        n = len(self.boundary_points)

        for i in range(n):
            j = (i + 1) % n
            if ((self.boundary_points[i, 1] > point[1]) != (self.boundary_points[j, 1] > point[1])):
                intersect = (self.boundary_points[j, 0] - self.boundary_points[i, 0]) * \
                           (point[1] - self.boundary_points[i, 1]) / \
                       (self.boundary_points[j, 1] - self.boundary_points[i, 1]) + \
                           self.boundary_points[i, 0]
                if point[0] < intersect:
                    inside = not inside

        return inside

    def get_elevation(self, x, y):
        """
        获取指定位置的高程

        使用局部加权平均平滑噪点
        """
        query_point = np.array([x, y])
        distances, indices = self.kdtree.query(query_point, k=min(10, len(self.points)))

        if isinstance(indices, np.ndarray):
            indices = indices[indices >= 0]
            distances = distances[:len(indices)]
        else:
            indices = np.array([indices])
            distances = np.array([distances])

        if len(indices) == 0:
            return 0.0

        # 距离加权，使用高斯核
        weights = np.exp(-distances / (self.neighbor_radius * self.smoothing_factor + 1e-10))
        weights = weights / (weights.sum() + 1e-10)

        elevations = self.normalized_points[indices, 2]
        z_normalized = np.sum(weights * elevations)

        # 反归一化
        z = z_normalized * (self.max_coords[2] - self.min_coords[2]) + self.min_coords[2]

        return z

    def get_gradient(self, x, y, delta=0.001):
        """
        计算指定位置的梯度（坡度方向和大小）

        Returns:
            grad_x: x方向梯度
            grad_y: y方向梯度
            slope_magnitude: 坨度大小
        """
        z = self.get_elevation(x, y)
        z_dx = self.get_elevation(x + delta, y)
        z_dy = self.get_elevation(x, y + delta)

        grad_x = (z_dx - z) / delta
        grad_y = (z_dy - z) / delta
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return grad_x, grad_y, slope_magnitude

    def get_surface_normal(self, x, y, delta=0.001):
        """
        计算表面法向量
        用于计算受力方向
        """
        grad_x, grad_y, _ = self.get_gradient(x, y, delta)

        # 法向量：[-dz/dx, -dz/dy, 1] 归一化
        normal = np.array([-grad_x, -grad_y, 1.0])
        normal = normal / (np.linalg.norm(normal) + 1e-10)

        return normal


class PhysicsEngine:
    """
    物理引擎
    模拟高尔夫球在地形上的运动

    物理模型：
    1. 重力分量：mg * sin(slope)
    2. 滚动摩擦：f * N = f * mg * cos(slope)
    3. 空气阻力：0.5 * rho * Cd * A * v^2
    4. 自适应时间步长确保数值稳定性
    """

    # 物理常量
    GRAVITY = 9.81              # m/s^2
    BALL_RADIUS = 0.02135       # m (标准高尔夫球直径 42.67mm)
    BALL_MASS = 0.04593         # kg (标准高尔夫球质量)
    AIR_DENSITY = 1.225         # kg/m^3
    DRAG_COEFFICIENT = 0.47     # 球体阻力系数
    CROSS_SECTIONAL_AREA = np.pi * BALL_RADIUS**2

    def __init__(self, terrain_model,
                 rolling_friction_coeff=0.1,  # 草坪滚动摩擦系数
                 air_resistance=True,
                 adaptive_timestep=True):
        """
        初始化物理引擎

        Args:
            terrain_model: TerrainModel 实例
            rolling_friction_coeff: 滚动摩擦系数，草坪通常0.05-0.15
            air_resistance: 是否考虑空气阻力
            adaptive_timestep: 是否使用自适应时间步长
        """
        self.terrain = terrain_model
        self.mu = rolling_friction_coeff
        self.air_resistance = air_resistance
        self.adaptive_timestep = adaptive_timestep

        # 自适应时间步长参数
        self.min_timestep = 0.0001  # 最小时间步长
        self.max_timestep = 0.01     # 最大时间步长
        self.velocity_threshold = 0.0005  # 停止速度阈值 (m/s)
        self.max_iterations = 10000  # 最大迭代次数

    def compute_forces(self, position, velocity):
        """
        计算所有作用在球上的力

        Args:
            position: [x, y, z] 位置
            velocity: [vx, vy, vz] 速度

        Returns:
            total_force: 总力向量
        """
        x, y, z = position
        speed = np.linalg.norm(velocity[:2])  # 只考虑水平速度

        if speed < 1e-10:
            return np.array([0.0, 0.0, 0.0])

        # 1. 获取坡度信息
        grad_x, grad_y, slope_magnitude = self.terrain.get_gradient(x, y)
        surface_normal = self.terrain.get_surface_normal(x, y)

        # 2. 计算重力沿坡度方向的分量
        # 坡度方向（重力滑下坡的方向）
        slope_direction = np.array([grad_x, grad_y, 0.0])
        slope_direction = slope_direction / (np.linalg.norm(slope_direction) + 1e-10)

        # 重力沿坡度方向的力
        gravity_force = self.BALL_MASS * self.GRAVITY * slope_magnitude * slope_direction

        # 3. 计算滚动摩擦力（与运动方向相反）
        if speed > 0:
            velocity_direction = velocity[:2] / speed
            normal_force = self.BALL_MASS * self.GRAVITY * np.cos(slope_magnitude)
            friction_magnitude = self.mu * normal_force

            # 摩擦力与运动方向相反
            friction_force = -friction_magnitude * np.array([velocity_direction[0], velocity_direction[1], 0.0])
        else:
            friction_force = np.array([0.0, 0.0, 0.0])

        # 4. 计算空气阻力
        if self.air_resistance:
            drag_magnitude = 0.5 * self.AIR_DENSITY * self.DRAG_COEFFICIENT * \
                           self.CROSS_SECTIONAL_AREA * speed**2

            if speed > 0:
                drag_force = -drag_magnitude * np.array([velocity[0]/speed, velocity[1]/speed, 0.0])
            else:
                drag_force = np.array([0.0, 0.0, 0.0])
        else:
            drag_force = np.array([0.0, 0.0, 0.0])

        # 总力
        total_force = gravity_force + friction_force + drag_force

        return total_force

    def rk4_step(self, position, velocity, dt):
        """
        四阶龙格-库塔积分
        提高数值精度
        """
        def derivatives(pos, vel):
            force = self.compute_forces(pos, vel)
            acceleration = force / self.BALL_MASS
            # 只更新水平速度，垂直由地形决定
            return acceleration

        # k1
        a1 = derivatives(position, velocity)
        v1 = velocity

        # k2
        pos2 = position + v1 * 0.5 * dt
        vel2 = velocity + a1 * 0.5 * dt
        a2 = derivatives(pos2, vel2)
        v2 = vel2

        # k3
        pos3 = position + v2 * 0.5 * dt
        vel3 = velocity + a2 * 0.5 * dt
        a3 = derivatives(pos3, vel3)
        v3 = vel3

        # k4
        pos4 = position + v3 * dt
        vel4 = velocity + a3 * dt
        a4 = derivatives(pos4, vel4)
        v4 = vel4

        # 组合
        new_velocity = velocity + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
        new_position = position + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)

        # 更新z坐标为地形高度
        new_position[2] = self.terrain.get_elevation(new_position[0], new_position[1])

        return new_position, new_velocity

    def adaptive_step(self, position, velocity):
        """
        自适应时间步长
        根据速度调整步长以保持稳定性
        """
        speed = np.linalg.norm(velocity)

        if speed > 1.0:
            # 高速时使用小步长
            dt = self.min_timestep
        elif speed > 0.1:
            dt = 0.001
        elif speed > 0.01:
            dt = 0.005
        else:
            dt = self.max_timestep

        return dt

    def simulate(self, start_position, initial_velocity):
        """
        完整的推杆模拟

        Args:
            start_position: [x, y, z] 起始位置
            initial_velocity: [vx, vy, vz] 初始速度向量

        Returns:
            trajectory: 轨迹点列表 [[x, y, z], ...]
            times: 对应时间点列表
        """
        trajectory = []
        times = []
        position = np.array(start_position, dtype=np.float64)
        velocity = np.array(initial_velocity, dtype=np.float64)
        t = 0.0

        # 确保起始z在地形上
        position[2] = self.terrain.get_elevation(position[0], position[1])

        trajectory.append(position.copy())
        times.append(t)

        for iteration in range(self.max_iterations):
            speed = np.linalg.norm(velocity[:2])

            # 检查停止条件
            if speed < self.velocity_threshold:
                print(f"球已停止。迭代次数: {iteration}, 最终距离: {t:.2f}s")
                break

            # 检查边界
            if not self.terrain.is_inside_boundary(position[0], position[1]):
                print(f"球离开边界。迭代次数: {iteration}")
                break

            # 选择时间步长
            if self.adaptive_timestep:
                dt = self.adaptive_step(position, velocity)
            else:
                dt = self.max_timestep

            # 数值积分
            position, velocity = self.rk4_step(position, velocity, dt)
            t += dt

            trajectory.append(position.copy())
            times.append(t)

        return np.array(trajectory), np.array(times)


def create_demo_point_cloud():
    """
    创建演示用的点云数据
    生成一个带有坡度的高尔夫果岭地形
    """
    # 创建网格点
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # 生成高度：带有轻微坡度和一些"草纹"效果
    Z = 0.1 * (X + 0.3 * Y)  # 基本坡度
    Z += 0.02 * np.sin(X * 3) * np.cos(Y * 2)  # 草纹效果
    Z += 0.01 * np.random.randn(*X.shape)  # 少量噪点

    # 展平
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    return points
