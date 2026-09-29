# pick_and_place_task

ROS 2 软件包，实现基于腕部视觉的 UR 机械臂两阶段颜色方块抓放任务：

- **Task 1** — 静态场景：HSV 颜色检测 + 三维点云反投影，按绿/红/黄分色堆叠 9 个方块。
- **Task 2** — 动态场景：最小二乘圆轨迹拟合 + 线性相位预测，抓取旋转转盘上的红色方块并连续堆叠至 10 层。

课程「交叉项目训练-机器人智能操作」（2026 春）组队项目，成员：[苟左](https://github.com/tactino)、[张家杰](https://github.com/z007-jj)；指导教师：李翔。
演示视频见 [`demo/`](demo/)，完整实验报告见 [`report/report.pdf`](report/report.pdf)。[English README](README.md)

---

## 硬件要求

| 设备 | 型号/接口 |
|------|-----------|
| 机械臂 | Universal Robots UR 系列，RTDE 接口（默认 IP `192.168.56.3`） |
| 相机 | Intel RealSense D 系列，腕部安装（eye-in-hand），已完成手眼标定 |
| 吸盘 | 串口 Modbus，连接至 `/dev/ttyUSB0` |

---

## 软件依赖

**ROS 2 包（`package.xml` 中已声明）：**

```
rclpy  sensor_msgs  cv_bridge  message_filters  tf2_ros
```

**Python 库：**

```bash
pip install opencv-python numpy scipy pyserial
pip install pyrealsense2          # RealSense SDK Python 绑定（可选，相机由 ROS 驱动）
pip install ur-rtde               # UR RTDE 接口
```

---

## 安装

```bash
# 将本包放入 ROS 2 工作空间 src 目录
cd ~/ros2_ws/src
# （已克隆或复制至此处）

# 编译
cd ~/ros2_ws
colcon build --packages-select pick_and_place_task
source install/setup.bash
```

---

## 运行

### 完整两阶段任务（推荐）

```bash
ros2 launch pick_and_place_task pick_and_place.launch.py
```

Task 1 节点启动后立即执行；Task 1 正常退出后等待 5 秒，Task 2 自动启动。

### 仅运行 Task 1

```bash
ros2 run pick_and_place_task pick_and_place.py
```

### 仅运行 Task 2

```bash
python3 install/lib/pick_and_place_task/pick_and_place_moving.py \
    --initial-stack-level 3 \
    --ros-args -p base_frame:=base
```

`--initial-stack-level` 指定初始堆叠层数（Task 1 完成后应传入实际已放置层数）。

---

## Launch 参数

所有参数均有默认值，按需覆盖：

```bash
ros2 launch pick_and_place_task pick_and_place.launch.py \
    task1_approach_offset:=0.13 \
    task1_color_adaptive_sv_relax:=30
```

### Task 1 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task1_approach_offset` | `0.13` | 接近位到目标的偏置（m） |
| `task1_grasp_offset` | `0.045` | 抓取位到目标的偏置（m） |
| `task1_grasp_descent_offset` | `0.10` | 从接近位额外下降量（m） |
| `task1_approach_to_grasp_end_speed_scale` | `0.40` | 末段速度比例（相对标称） |
| `task1_approach_to_grasp_slowdown_stages` | `4` | 渐速下降阶段数 |
| `task1_color_adaptive_sv_relax` | `25` | HSV S/V 通道自适应松弛量 |
| `task1_pick_camera_z_compensation_m` | `0.0` | 相机 z 轴系统误差补偿（m） |
| `task1_initial_green_count` | `0` | 绿色已放置初始数量 |
| `task1_initial_red_count` | `0` | 红色已放置初始数量 |
| `task1_initial_yellow_count` | `0` | 黄色已放置初始数量 |

### Task 2 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task2_approach_offset` | `0.10` | 接近位偏置（m） |
| `task2_grasp_offset` | `0.05` | 抓取位偏置（m） |
| `task2_pick_camera_z_compensation_m` | `0.01` | 相机 z 轴补偿（m） |

---

## 调试工具

| 脚本 | 用途 |
|------|------|
| `scripts/color_threshold_tuner.py` | 实时调整 HSV 颜色阈值，确认分割效果 |
| `scripts/pick_pose_debug.py` | 可视化接近/抓取位规划结果 |
| `scripts/pose_camera_monitor.py` | 监控相机位姿与 TF 变换 |

所有主节点均支持 `show_debug_window:=true` 参数，启用后弹出 OpenCV 调试窗口。

---

## 项目结构

```
pick_and_place_task/
├── scripts/
│   ├── pick_and_place.py          # Task 1：静态抓放控制节点
│   ├── pick_and_place_moving.py   # Task 2：动态目标抓取节点
│   ├── color_threshold_tuner.py   # HSV 颜色阈值调试工具
│   ├── pick_pose_debug.py         # 抓取位姿可视化调试工具
│   └── pose_camera_monitor.py     # 相机位姿监控工具
├── launch/
│   └── pick_and_place.launch.py   # 两阶段任务启动文件
├── demo/
│   ├── task1_demo.mp4             # Task 1 演示视频
│   └── task2_demo.mp4             # Task 2 演示视频（堆叠至 10 层）
├── report/
│   ├── report.tex                 # 实验报告 LaTeX 源文件
│   ├── report.pdf                 # 编译生成的实验报告
│   └── img/                       # 报告插图
├── CMakeLists.txt
└── package.xml
```

---

## 关键算法

**Task 1 — 静态抓放**

1. 高斯模糊 → HSV 转换 → CLAHE 亮度均衡 → 多阈值分割（支持 S/V 自适应松弛）
2. 连通域分析 + 矩形度/凸四边形过滤 → 目标中心提取
3. 针孔模型反投影 + SVD 法向量估算 → 3D 位姿
4. 多帧稳定确认（位置 ≤ 8 mm，法向量夹角 ≤ 12°，连续 3 帧）
5. 三段式路径（接近 → 渐速下降 → 提升）+ 偏航补偿 + 放置后视觉核验

**Task 2 — 动态抓取**

1. 观测位采样（默认 7~8 s，间隔 50 ms），记录目标 xy 坐标与偏航角
2. 最小二乘圆轨迹拟合 + `numpy.unwrap` 线性相位回归 → 圆心、半径、角速度 ω
3. 提前量 Δt = 移动时间 + 下降时间 + 余量 → 预测抓取时刻位置
4. 到达接近位后实时修正：中心位置 + 边缘偏航（消除 π 歧义）
5. 分阶段旋转下降（默认 3 段） → 激活吸盘 → 放置并返回观测位

---

## 位姿格式说明

所有预设位姿（`home_pose`、`place_pose_*` 等）均采用 `[x, y, z, qx, qy, qz, qw]` 四元数格式，
在 launch 文件或节点参数中配置。节点初始化时自动转换为 RTDE `moveL` 所需的旋转向量（rotvec）格式。

当 `apply_pose_frame_transform: true`（默认）时，所有位姿会自动绕 z 轴旋转 180° 以对齐手眼标定坐标系约定。
