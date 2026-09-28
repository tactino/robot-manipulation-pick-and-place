"""
pick_and_place.launch.py

两阶段抓放任务的启动文件：
  Task 1 — pick_and_place_node (ROS2 Node)
      使用相机颜色检测对静态方块执行抓取和放置。
  Task 2 — pick_and_place_moving_cmd (ExecuteProcess)
      对运动目标执行抓取和放置，初始堆叠层数为 3。
      在 Task 1 正常退出后延迟 5 s 自动启动。

所有可调参数均通过 launch argument 暴露，并附有合理默认值。
"""

import os
import pathlib

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 在函数内导入，避免在未安装 ROS2 的环境中 import 本文件时报错
    from ament_index_python.packages import get_package_share_directory

    pkg_share_dir = get_package_share_directory('pick_and_place_task')
    # 将 share 目录映射到安装脚本目录：<install>/lib/pick_and_place_task/
    pkg_lib_dir = str(
        pathlib.Path(pkg_share_dir).parent.parent / 'lib' / 'pick_and_place_task'
    )

    # ── Task 1：静态抓放 Node ─────────────────────────────────────────────────
    pick_and_place_node = Node(
        package='pick_and_place_task',
        executable='pick_and_place.py',
        output='screen',
        parameters=[{
            # 机器人与传感器基础配置
            'robot_ip':           '192.168.56.3',
            'base_frame':         LaunchConfiguration('task1_base_frame'),
            'camera_frame':       'camera_color_optical_frame',
            'suction_device':     '/dev/ttyUSB0',
            'rgb_topic':          '/camera/camera/color/image_raw',
            'depth_topic':        '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic':  '/camera/camera/color/camera_info',
            'speed':              1.0,
            'acceleration':       1.0,

            # 接近 / 抓取几何参数
            'approach_offset':                         LaunchConfiguration('task1_approach_offset'),
            'grasp_offset':                            LaunchConfiguration('task1_grasp_offset'),
            'grasp_descent_offset':                    LaunchConfiguration('task1_grasp_descent_offset'),
            'approach_to_grasp_progressive_slowdown':  True,
            'approach_to_grasp_slowdown_stages':       LaunchConfiguration('task1_approach_to_grasp_slowdown_stages'),
            'approach_to_grasp_end_speed_scale':       LaunchConfiguration('task1_approach_to_grasp_end_speed_scale'),
            'approach_to_grasp_last_stage_ratio':      LaunchConfiguration('task1_approach_to_grasp_last_stage_ratio'),

            # 吸盘控制
            'suction_on_delay_sec': LaunchConfiguration('task1_suction_on_delay_sec'),

            # 方块 / TF 拾取配置
            'use_tf_cube_center_for_pick':      LaunchConfiguration('task1_use_tf_cube_center_for_pick'),
            'pick_cube_frame':                  LaunchConfiguration('task1_pick_cube_frame'),
            'pick_camera_z_compensation_m':     LaunchConfiguration('task1_pick_camera_z_compensation_m'),
            'pick_camera_z_compensation_frame': LaunchConfiguration('task1_pick_camera_z_compensation_frame'),
            'pick_camera_axis_in_tool':         LaunchConfiguration('task1_pick_camera_axis_in_tool'),
            'pick_pose_z_half_turn_correction': False,

            # 帧同步参数
            'frame_sync_queue_size':             LaunchConfiguration('task1_frame_sync_queue_size'),
            'frame_sync_slop_sec':               LaunchConfiguration('task1_frame_sync_slop_sec'),
            'max_accepted_frame_age_sec':        LaunchConfiguration('task1_max_accepted_frame_age_sec'),
            'drop_out_of_order_frames':          LaunchConfiguration('task1_drop_out_of_order_frames'),
            'clear_camera_cache_before_observe': LaunchConfiguration('task1_clear_camera_cache_before_observe'),
            'home_observe_required_new_frames':  LaunchConfiguration('task1_home_observe_required_new_frames'),
            'home_observe_wait_timeout_sec':     LaunchConfiguration('task1_home_observe_wait_timeout_sec'),

            # 运动几何固定值
            'post_pick_short_lift_offset': 0.08,
            'lift_offset':                 0.06,
            'fixed_lift_z':                0.35,
            'settle_sec':                  0.20,
            'enable_motion':               True,

            # 颜色检测参数
            'color_min_area':                    300.0,
            'color_adaptive_sv_relax':           LaunchConfiguration('task1_color_adaptive_sv_relax'),
            'color_value_equalize_clip_limit':   LaunchConfiguration('task1_color_value_equalize_clip_limit'),
            'color_mask_morph_kernel':           LaunchConfiguration('task1_color_mask_morph_kernel'),

            # 放置几何固定值
            'first_place_drop_height': 0.10,
            'place_clearance':         0.012,
            'stack_height_step':       0.03,
            'place_hover_offset':      0.32,
            'place_hover_observe_sec': 1.0,

            # 各颜色方块初始数量
            'initial_green_count':  LaunchConfiguration('task1_initial_green_count'),
            'initial_red_count':    LaunchConfiguration('task1_initial_red_count'),
            'initial_yellow_count': LaunchConfiguration('task1_initial_yellow_count'),
            'apply_pose_frame_transform': True,

            # 标定位姿，格式：[x, y, z, qx, qy, qz, qw]
            'home_pose':              [-0.163687,  0.207756, 0.497297,  0.364787,  0.924170,  0.100803, -0.051757],
            'place_pose_green':       [ 0.310575,  0.234747, 0.090584,  0.832468,  0.553170, -0.029218, -0.012106],
            'place_pose_red':         [ 0.076850,  0.335232, 0.091866,  0.312712,  0.949606,  0.005125, -0.020816],
            'place_pose_yellow':      [ 0.204962,  0.359774, 0.092745,  0.697613,  0.716347,  0.012685, -0.004696],
            'place_hover_pose_red':   [-0.002263,  0.257214, 0.353725,  0.790075,  0.612940,  0.004711, -0.007983],
            'place_hover_pose_yellow':[ 0.118099,  0.319657, 0.355338,  0.857946,  0.513306,  0.020869, -0.003208],
            'place_hover_pose_green': [ 0.205854,  0.230666, 0.372731,  0.942018,  0.334388,  0.002118,  0.027958],
            'place_mid_pose_green':   [ 0.310575,  0.234747, 0.372731,  0.832468,  0.553170, -0.029218, -0.012106],
            'place_mid_pose_red':     [ 0.076850,  0.335232, 0.353725,  0.312712,  0.949606,  0.005125, -0.020816],
            'place_mid_pose_yellow':  [ 0.204962,  0.359774, 0.355338,  0.697613,  0.716347,  0.012685, -0.004696],
        }],
    )

    # ── Task 2：运动目标抓放进程 ───────────────────────────────────────────────
    # 以 ExecuteProcess 而非 Node 启动，以便将 --initial-stack-level 作为
    # 位置参数传入，同时通过 --ros-args 传递 ROS2 参数。
    pick_and_place_moving_cmd = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(pkg_lib_dir, 'pick_and_place_moving.py'),
            '--initial-stack-level', '3',
            '--ros-args',
            '-p', ['base_frame:=',                       LaunchConfiguration('task2_base_frame')],
            '-p', ['approach_offset:=',                  LaunchConfiguration('task2_approach_offset')],
            '-p', ['grasp_offset:=',                     LaunchConfiguration('task2_grasp_offset')],
            '-p', ['pick_camera_z_compensation_m:=',     LaunchConfiguration('task2_pick_camera_z_compensation_m')],
            '-p', ['pick_camera_z_compensation_frame:=', LaunchConfiguration('task2_pick_camera_z_compensation_frame')],
        ],
        output='screen',
    )

    # Task 1 退出后延迟 5 s 自动启动 Task 2
    on_first_exit = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=pick_and_place_node,
            on_exit=[TimerAction(period=5.0, actions=[pick_and_place_moving_cmd])],
        ),
    )

    # ── 启动参数声明 ──────────────────────────────────────────────────────────
    return LaunchDescription([
        # Task 1 参数
        DeclareLaunchArgument('task1_base_frame',                         default_value='base'),
        DeclareLaunchArgument('task1_pick_camera_axis_in_tool',           default_value='camera_z'),
        DeclareLaunchArgument('task1_approach_offset',                    default_value='0.13'),
        DeclareLaunchArgument('task1_grasp_offset',                       default_value='0.045'),
        DeclareLaunchArgument('task1_grasp_descent_offset',               default_value='0.10'),
        DeclareLaunchArgument('task1_approach_to_grasp_slowdown_stages',  default_value='4'),
        DeclareLaunchArgument('task1_approach_to_grasp_end_speed_scale',  default_value='0.40'),
        DeclareLaunchArgument('task1_approach_to_grasp_last_stage_ratio', default_value='0.12'),
        DeclareLaunchArgument('task1_suction_on_delay_sec',               default_value='0.10'),
        DeclareLaunchArgument('task1_use_tf_cube_center_for_pick',        default_value='false'),
        DeclareLaunchArgument('task1_pick_cube_frame',                    default_value='cube_frame'),
        DeclareLaunchArgument('task1_pick_camera_z_compensation_m',       default_value='0.0'),
        DeclareLaunchArgument('task1_pick_camera_z_compensation_frame',   default_value='camera_link'),
        DeclareLaunchArgument('task1_frame_sync_queue_size',              default_value='3'),
        DeclareLaunchArgument('task1_frame_sync_slop_sec',                default_value='0.04'),
        DeclareLaunchArgument('task1_max_accepted_frame_age_sec',         default_value='0.45'),
        DeclareLaunchArgument('task1_drop_out_of_order_frames',           default_value='true'),
        DeclareLaunchArgument('task1_clear_camera_cache_before_observe',  default_value='true'),
        DeclareLaunchArgument('task1_home_observe_required_new_frames',   default_value='2'),
        DeclareLaunchArgument('task1_home_observe_wait_timeout_sec',      default_value='3.0'),
        DeclareLaunchArgument('task1_color_adaptive_sv_relax',            default_value='25'),
        DeclareLaunchArgument('task1_color_value_equalize_clip_limit',    default_value='2.0'),
        DeclareLaunchArgument('task1_color_mask_morph_kernel',            default_value='5'),
        DeclareLaunchArgument('task1_initial_green_count',                default_value='0'),
        DeclareLaunchArgument('task1_initial_red_count',                  default_value='0'),
        DeclareLaunchArgument('task1_initial_yellow_count',               default_value='0'),
        # Task 2 参数
        DeclareLaunchArgument('task2_base_frame',                         default_value='base'),
        DeclareLaunchArgument('task2_approach_offset',                    default_value='0.10'),
        DeclareLaunchArgument('task2_grasp_offset',                       default_value='0.05'),
        DeclareLaunchArgument('task2_pick_camera_z_compensation_m',       default_value='0.01'),
        DeclareLaunchArgument('task2_pick_camera_z_compensation_frame',   default_value='camera_link'),
        # 动作
        pick_and_place_node,
        on_first_exit,
    ])
