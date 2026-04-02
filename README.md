# 高尔夫推杆轨迹模拟器

基于点云数据和物理模拟的高尔夫球推杆轨迹预测系统。

## 功能特性

- **地形建模**: 从点云文件构建 3D 地形模型
- **物理模拟**: 基于经典力学的高尔夫球运动模拟
- **可视化界面**: 实时 3D 和俯视图显示
- **交互控制**: 推杆力度、方向和物理参数调整

## 技术方案分析

### 已采用的核心方案

1. **地形表示**: 使用 KDTree + 局部加权平均
   - 避免了 RBF 插值在稀疏数据时的过拟合问题
   - 处理点云密度不均匀和噪点

2. **数值积分**: 四阶龙格-库塔方法 (RK4)
   - 比欧拉法精度高一个量级
   - 避免速度较快时的数值发散

3. **自适应时间步长**
   - 根据速度动态调整步长
   - 避免高速时的数值不稳定

### 方案比较

| 方案 | 优点 | 缺点 | 本项目选择 |
|------|------|------|-----------|
| RK4 | 高精度，稳定 | 计算量较大 | ✅ 采用 |
| Verlet | 能量守恒好 | 低速时精度差 | ❌ 未采用 |
| 欧拉法 | 简单快速 | 精度低，不稳定 | ❌ 避免 |
| RBF 插值 | 光滑 | 对噪点敏感，计算慢 | ❌ 避免 |
| KDTree + 加权 | 快速，抗噪点 | 边界处理复杂 | ✅ 采用 |

### 避免的问题

1. **数值不稳定**: 使用 RK4 + 自适应步长，避免高速球飞出
2. **点云噪点**: 局部加权平均而非直接插值
3. **边界检测**: 射线法检测判断是否离开果岭
4. **速度归零**: 避免球在零速度附近无限震荡

## 安装依赖

```bash
cd golf_trajectory_simulator
pip install -r requirements.txt
```

## 使用方法

### 启动程序

有两种界面可供选择：

#### 1. Matplotlib 版本（推荐，无需额外依赖）

```bash
python gui_matplotlib.py
```

这个版本不依赖 pyqtgraph 和 PyQt6，仅使用 matplotlib 和 tkinter，系统自带。

#### 2. PyQt6 版本（需要安装 pyqtgraph）

```bash
python gui.py
```

需要先安装依赖：
```bash
pip install PyQt6 pyqtgraph
```

### 点云文件格式

支持 TXT 或 CSV 格式，每行包含 X, Y, Z 坐标（单位：米）：

```
0.0 0.0 0.0
0.1 0.0 0.001
...
```

或 CSV:

```
x,y,z
0.0,0.0,0.0
0.1,0.0,0.001
...
```

### 操作步骤

1. **加载地形**: 点击"加载点云文件"或使用"使用演示地形"
2. **设置起始位置**: 在"起始位置"面板调整 X, Y 坐标
3. **设置推杆**:
   - 调整"推杆力度"（单位：m/s）
   - 拖动"方向角度"滑块调整方向
4. **调整参数**: 根据需要修改滚动摩擦系数等
5. **运行模拟**: 点击"开始模拟"查看轨迹

## 物理模型

### 受力分析

```
总力 = 重力分量 + 滚动摩擦力 + 空气阻力
```

1. **重力沿坡度分量**
   ```
   F_gravity = m * g * sin(slope) * slope_direction
   ```

2. **滚动摩擦力**
   ```
   F_friction = μ * N * (-velocity_direction)
   N = m * g * cos(slope)
   ```

3. **空气阻力**
   ```
   F_drag = 0.5 * ρ * Cd * A * v^2 * (-velocity_direction)
   ```

### 物理常数

- 常数高尔夫球直径: 42.67 mm
- 常数高尔夫球质量: 45.93 g
- 草坪滚动摩擦系数: 0.05-0.15 (可调)
- 空气阻力系数: 0.47

## 项目结构

```
golf_trajectory_simulator/
├── physics_engine.py    # 物理引擎核心
├── gui.py              # Qt 图形界面
├── requirements.txt     # Python 依赖
└── README.md           # 项目说明
```

## 扩展建议

- 支持更多点云格式 (PLY, LAS)
- 添加目标洞口和进洞检测
- 导出轨迹数据为标准格式
- 添加风速影响
- 支持多球同时模拟

## 参考资料

- [Scipy KDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html)
- [RK4 积分方法](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
- [PyQt6 文档](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
