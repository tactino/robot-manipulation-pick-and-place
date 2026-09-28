#!/usr/bin/env python3
"""
pick_and_place_moving.py — Phase 2: 运动目标红色方块抓取任务

节点 Phase2MovingRedPick 在观测位采集一段时间的目标轨迹样本，
拟合匀速圆周运动参数，预测方块抓取时刻位置，执行接近-抓取-放置闭环。
放置时根据相机深度判断当前堆叠层数，自动选择对应的放置位姿。
任务循环直到堆叠层数达到上限（10 层）后退出。

用法（通过 launch 文件传参）::
    python3 pick_and_place_moving.py --initial-stack-level 3 \\
        --ros-args -p base_frame:=base ...
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

import serial

rtde_control = None
rtde_receive = None


@dataclass
class RedTarget:
    """单帧中检测到的红色目标，包含像素与相机坐标系下的几何信息。"""

    area: float
    contour: np.ndarray
    center_px: tuple[int, int]
    center_cam: np.ndarray


class Phase2MovingRedPick(Node):
    """Phase 2 运动目标拾取节点：观测 → 拟合圆轨迹 → 预测抓取点 → 抓取 → 放置堆叠。"""

    def __init__(self):
        super().__init__('phase2_moving_red_pick')

        # 机器人与传感器基础配置
        self.declare_parameter('robot_ip', '192.168.56.3')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('suction_device', '/dev/ttyUSB0')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('speed', 0.5)
        self.declare_parameter('acceleration', 0.5)
        # 轨迹预测参数
        self.declare_parameter('travel_speed_for_prediction', 0.20)
        self.declare_parameter('prediction_extra_sec', 0.15)
        self.declare_parameter('pick_prediction_horizon_sec', 5.0)
        self.declare_parameter('pick_timing_advance_sec', 0.4)
        # 接近-抓取运动参数
        self.declare_parameter('approach_to_grasp_speed', 0.14)
        self.declare_parameter('approach_to_grasp_acceleration', 0.35)
        self.declare_parameter('approach_to_grasp_extra_sec', 0.5)
        self.declare_parameter('descent_rotation_stages', 3)
        self.declare_parameter('descent_vertical_only', True)
        self.declare_parameter('near_grasp_rotate_speed', 0.35)
        self.declare_parameter('near_grasp_rotate_acceleration', 0.80)
        # final_parallel_align 已由 near-grasp + staged refine 替代，保留参数但默认禁用（max_iters=0）
        self.declare_parameter('final_parallel_align_max_iters', 0)
        self.declare_parameter('final_parallel_align_tol_deg', 2.0)
        # 几何偏移量
        self.declare_parameter('approach_offset', 0.10)
        self.declare_parameter('grasp_offset', 0.05)
        self.declare_parameter('post_pick_lift_offset', 0.10)
        # 采样与观测参数
        self.declare_parameter('observe_settle_sec', 0.50)
        self.declare_parameter('first_sample_duration_sec', 8.0)
        self.declare_parameter('sample_duration_sec', 7.0)
        self.declare_parameter('sample_interval_sec', 0.05)
        self.declare_parameter('min_samples', 20)
        # TF 与运动开关
        self.declare_parameter('tf_lookup_timeout_sec', 0.10)
        self.declare_parameter('enable_motion', True)
        # 颜色检测与调试
        self.declare_parameter('color_min_area', 300.0)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('apply_pose_frame_transform', True)
        # 相机 z 轴补偿
        self.declare_parameter('pick_camera_z_compensation_m', 0.015)
        self.declare_parameter('pick_camera_z_compensation_frame', 'camera_link')
        self.declare_parameter(
            'observe_pose',
            [-0.203774, 0.252514, 0.465453, 0.399701, 0.914671, 0.045801, -0.038977],
        )
        self.declare_parameter(
            'home_pose',
            [-0.161277, 0.180922, 0.481411, 0.391481, 0.917038, 0.057670, -0.049580],
        )
        # 堆叠层数管理
        self.declare_parameter('initial_stack_level', 3)
        self.declare_parameter('stack_min_level', 4)
        self.declare_parameter('stack_max_level', 10)
        self.declare_parameter('place_check_settle_sec', 0.20)
        self.declare_parameter(
            'place_observe_pose',
            [-0.006123, 0.155230, 0.508925, 0.839010, 0.537765, 0.081651, -0.014281],
        )
        self.declare_parameter(
            'stack_pose_4',
            [0.073069, 0.331429, 0.338227, 0.626369, 0.779349, -0.014945, 0.007325],
        )
        self.declare_parameter(
            'stack_pose_5',
            [0.059257, 0.334054, 0.419633, 0.610151, 0.792217, -0.008006, 0.006572],
        )
        # 放置后短提升参数
        self.declare_parameter('post_place_short_lift_offset', 0.012)
        self.declare_parameter('post_place_short_lift_speed_scale', 0.35)
        self.declare_parameter('post_place_short_lift_acc_scale', 0.35)
        self.declare_parameter(
            'stack_pose_6',
            [-0.032231, 0.240574, 0.378881, 0.482881, 0.497557, 0.720535, -0.009604],
        )
        self.declare_parameter(
            'stack_pose_6_pre',
            [-0.118163, 0.257136, 0.379495, 0.123326, 0.681091, 0.614068, -0.379244],
        )
        self.declare_parameter(
            'stack_pose_7',
            [-0.030755, 0.250147, 0.463470, 0.485131, 0.511809, 0.709013, -0.001038],
        )
        self.declare_parameter(
            'stack_pose_7_pre',
            [-0.082057, 0.268016, 0.463573, 0.278065, 0.632016, 0.682200, -0.240497],
        )
        self.declare_parameter(
            'stack_pose_8',
            [-0.030813, 0.253748, 0.545515, 0.512448, 0.490112, 0.704037, 0.038973],
        )
        self.declare_parameter(
            'stack_pose_8_pre',
            [-0.088713, 0.289781, 0.540681, 0.216779, 0.661537, 0.652313, -0.299772],
        )
        self.declare_parameter(
            'stack_pose_9',
            [-0.016908, 0.245958, 0.620554, 0.453642, 0.535685, 0.710903, -0.043223],
        )
        self.declare_parameter(
            'stack_pose_9_pre',
            [-0.069472, 0.267504, 0.620920, 0.223254, 0.663110, 0.651129, -0.294064],
        )
        self.declare_parameter(
            'stack_pose_10',
            [0.011767, 0.216268, 0.689697, 0.349220, 0.567592, 0.728519, -0.158573],
        )
        self.declare_parameter(
            'stack_pose_10_pre',
            [-0.046682, 0.206781, 0.685269, 0.074879, 0.666391, 0.659007, -0.340626],
        )

        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.suction_device = self.get_parameter('suction_device').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.speed = float(self.get_parameter('speed').get_parameter_value().double_value)
        self.acceleration = float(self.get_parameter('acceleration').get_parameter_value().double_value)
        self.travel_speed_for_prediction = float(
            self.get_parameter('travel_speed_for_prediction').get_parameter_value().double_value
        )
        self.prediction_extra_sec = float(self.get_parameter('prediction_extra_sec').get_parameter_value().double_value)
        self.pick_prediction_horizon_sec = float(
            self.get_parameter('pick_prediction_horizon_sec').get_parameter_value().double_value
        )
        if self.pick_prediction_horizon_sec <= 0.0:
            self.pick_prediction_horizon_sec = 3.0
        self.pick_timing_advance_sec = float(
            self.get_parameter('pick_timing_advance_sec').get_parameter_value().double_value
        )
        self.approach_to_grasp_speed = float(
            self.get_parameter('approach_to_grasp_speed').get_parameter_value().double_value
        )
        self.approach_to_grasp_acceleration = float(
            self.get_parameter('approach_to_grasp_acceleration').get_parameter_value().double_value
        )
        self.approach_to_grasp_extra_sec = float(
            self.get_parameter('approach_to_grasp_extra_sec').get_parameter_value().double_value
        )
        self.descent_rotation_stages = int(self.get_parameter('descent_rotation_stages').get_parameter_value().integer_value)
        self.descent_vertical_only = bool(self.get_parameter('descent_vertical_only').get_parameter_value().bool_value)
        self.near_grasp_rotate_speed = float(
            self.get_parameter('near_grasp_rotate_speed').get_parameter_value().double_value
        )
        self.near_grasp_rotate_acceleration = float(
            self.get_parameter('near_grasp_rotate_acceleration').get_parameter_value().double_value
        )
        self.final_parallel_align_max_iters = int(
            self.get_parameter('final_parallel_align_max_iters').get_parameter_value().integer_value
        )
        if self.final_parallel_align_max_iters < 0:
            self.final_parallel_align_max_iters = 0
        self.final_parallel_align_tol_deg = float(
            self.get_parameter('final_parallel_align_tol_deg').get_parameter_value().double_value
        )
        if self.final_parallel_align_tol_deg < 0.0:
            self.final_parallel_align_tol_deg = 0.0
        self.approach_offset = float(self.get_parameter('approach_offset').get_parameter_value().double_value)
        self.grasp_offset = float(self.get_parameter('grasp_offset').get_parameter_value().double_value)
        self.post_pick_lift_offset = float(self.get_parameter('post_pick_lift_offset').get_parameter_value().double_value)
        self.observe_settle_sec = float(self.get_parameter('observe_settle_sec').get_parameter_value().double_value)
        self.first_sample_duration_sec = float(
            self.get_parameter('first_sample_duration_sec').get_parameter_value().double_value
        )
        if self.first_sample_duration_sec <= 0.0:
            self.first_sample_duration_sec = 8.0
        self.sample_duration_sec = float(self.get_parameter('sample_duration_sec').get_parameter_value().double_value)
        self.sample_interval_sec = float(self.get_parameter('sample_interval_sec').get_parameter_value().double_value)
        self.min_samples = int(self.get_parameter('min_samples').get_parameter_value().integer_value)
        self.tf_lookup_timeout_sec = float(self.get_parameter('tf_lookup_timeout_sec').get_parameter_value().double_value)
        self.enable_motion = bool(self.get_parameter('enable_motion').get_parameter_value().bool_value)
        self.color_min_area = float(self.get_parameter('color_min_area').get_parameter_value().double_value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').get_parameter_value().bool_value)
        self.apply_pose_frame_transform = bool(
            self.get_parameter('apply_pose_frame_transform').get_parameter_value().bool_value
        )
        self.pick_camera_z_compensation_m = float(
            self.get_parameter('pick_camera_z_compensation_m').get_parameter_value().double_value
        )
        self.pick_camera_z_compensation_frame = (
            self.get_parameter('pick_camera_z_compensation_frame').get_parameter_value().string_value.strip()
        )
        if not self.pick_camera_z_compensation_frame:
            self.pick_camera_z_compensation_frame = 'camera_link'
        if abs(self.pick_camera_z_compensation_m) > 1e-9:
            self.get_logger().info(
                'Pick camera z-axis compensation enabled: %.4f m in frame %s'
                % (self.pick_camera_z_compensation_m, self.pick_camera_z_compensation_frame)
            )
        self.initial_stack_level = int(self.get_parameter('initial_stack_level').get_parameter_value().integer_value)
        self.stack_min_level = int(self.get_parameter('stack_min_level').get_parameter_value().integer_value)
        self.stack_max_level = int(self.get_parameter('stack_max_level').get_parameter_value().integer_value)
        self.place_check_settle_sec = float(self.get_parameter('place_check_settle_sec').get_parameter_value().double_value)
        self.post_place_short_lift_offset = float(
            self.get_parameter('post_place_short_lift_offset').get_parameter_value().double_value
        )
        if self.post_place_short_lift_offset < 0.0:
            self.post_place_short_lift_offset = 0.0
        self.post_place_short_lift_speed_scale = float(
            self.get_parameter('post_place_short_lift_speed_scale').get_parameter_value().double_value
        )
        if self.post_place_short_lift_speed_scale <= 0.0:
            self.post_place_short_lift_speed_scale = 0.35
        self.post_place_short_lift_acc_scale = float(
            self.get_parameter('post_place_short_lift_acc_scale').get_parameter_value().double_value
        )
        if self.post_place_short_lift_acc_scale <= 0.0:
            self.post_place_short_lift_acc_scale = 0.35
        self.camera_z_axis_in_tool = np.array([0.721943, 0.679312, 0.131655], dtype=np.float64)

        self.observe_pose = self._read_pose_parameter(
            'observe_pose',
            [-0.270359, 0.219932, 0.452308, 0.991532, -0.109283, -0.030250, -0.063294],
        )
        self.home_pose = self._read_pose_parameter(
            'home_pose',
            [-0.161277, 0.180922, 0.481411, 0.391481, 0.917038, 0.057670, -0.049580],
        )
        self.place_observe_pose = self._read_pose_parameter(
            'place_observe_pose',
            [-0.011198, 0.163262, 0.528755, 0.666224, 0.723629, 0.180258, -0.003720],
        )
        self.stack_place_poses = {
            4: self._read_pose_parameter(
                'stack_pose_4',
                [0.073069, 0.331429, 0.338227, 0.626369, 0.779349, -0.014945, 0.007325],
            ),
            5: self._read_pose_parameter(
                'stack_pose_5',
                [0.059257, 0.334054, 0.419633, 0.610151, 0.792217, -0.008006, 0.006572],
            ),
            6: self._read_pose_parameter(
                'stack_pose_6',
                [-0.032231, 0.240574, 0.370881, 0.482881, 0.497557, 0.720535, -0.009604],
            ),
            7: self._read_pose_parameter(
                'stack_pose_7',
                [-0.030755, 0.250147, 0.456470, 0.485131, 0.511809, 0.709013, -0.001038],
            ),
            8: self._read_pose_parameter(
                'stack_pose_8',
                [-0.030813, 0.253748, 0.541515, 0.512448, 0.490112, 0.704037, 0.038973],
            ),
            9: self._read_pose_parameter(
                'stack_pose_9',
                [-0.016908, 0.245958, 0.617554, 0.453642, 0.535685, 0.710903, -0.043223],
            ),
            10: self._read_pose_parameter(
                'stack_pose_10',
                [0.011767, 0.216268, 0.689697, 0.349220, 0.567592, 0.728519, -0.158573],
            ),
        }
        self.stack_place_pre_poses = {
            6: self._read_pose_parameter(
                'stack_pose_6_pre',
                [-0.118163, 0.257136, 0.371495, 0.123326, 0.681091, 0.614068, -0.379244],
            ),
            7: self._read_pose_parameter(
                'stack_pose_7_pre',
                [-0.082057, 0.268016, 0.454573, 0.278065, 0.632016, 0.682200, -0.240497],
            ),
            8: self._read_pose_parameter(
                'stack_pose_8_pre',
                [-0.088713, 0.289781, 0.537681, 0.216779, 0.661537, 0.652313, -0.299772],
            ),
            9: self._read_pose_parameter(
                'stack_pose_9_pre',
                [-0.069472, 0.267504, 0.619920, 0.223254, 0.663110, 0.651129, -0.294064],
            ),
            10: self._read_pose_parameter(
                'stack_pose_10_pre',
                [-0.046682, 0.206781, 0.685269, 0.074879, 0.666391, 0.659007, -0.340626],
            ),
        }
        self.current_stack_level = int(self.initial_stack_level)

        self.color_ranges_red = [
            (np.array([0, 100, 80], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
            (np.array([160, 100, 80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
        ]

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_stamp = None
        self._debug_window_failed = False
        self._viz_lock = threading.Lock()
        self._fitted_center_base = None
        self._fitted_radius = None
        self._fitted_z_ref = None
        self._predicted_pick_base = None
        self._predicted_contour_base = None
        self._predicted_edge_dir_base = None
        self._orbit_center_xy = None
        self._locked_orbit_center_xy = None
        self._orbit_radius = None
        self._orbit_omega = None
        self._edge_yaw_rate = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rtde_c = None
        self.rtde_r = None
        self.serial_suction = None
        self.task_started = False
        self._first_cycle_sampling_done = False
        self._recorded_z_rotation_rad = 0.0
        self._observe_depart_pose = None

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self._init_robot()
        self._init_suction()

        self.timer = self.create_timer(0.3, self._trigger_once)

    def _trigger_once(self):
        """定时器回调，仅在第一次触发时在后台线程中启动任务，之后自动停止调用。"""
        if self.task_started:
            return
        self.task_started = True
        threading.Thread(target=self._run_task, daemon=True).start()

    def _read_pose_parameter(self, name: str, default_pose: list[float]) -> list[float]:
        """读取位姿参数并转换为 rotvec 格式；支持 6 元素 rotvec 或 7 元素四元数输入，失败时退回默认值。"""
        value = [float(v) for v in self.get_parameter(name).get_parameter_value().double_array_value]
        if len(value) == 6:
            if self.apply_pose_frame_transform:
                pos = np.asarray(value[:3], dtype=np.float64)
                rotvec = np.asarray(value[3:6], dtype=np.float64)
                t_pos, t_quat = self._transform_pose_frame(pos, Rotation.from_rotvec(rotvec).as_quat())
                t_rotvec = Rotation.from_quat(t_quat).as_rotvec()
                return [float(t_pos[0]), float(t_pos[1]), float(t_pos[2]), float(t_rotvec[0]), float(t_rotvec[1]), float(t_rotvec[2])]
            return value

        if len(value) == 7:
            pos = np.asarray(value[:3], dtype=np.float64)
            quat = np.asarray(value[3:7], dtype=np.float64)
            if np.linalg.norm(quat) > 0.0:
                if self.apply_pose_frame_transform:
                    pos, quat = self._transform_pose_frame(pos, quat)
                rotvec = Rotation.from_quat(quat).as_rotvec()
                return [float(pos[0]), float(pos[1]), float(pos[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

        if len(default_pose) == 7:
            pos = np.asarray(default_pose[:3], dtype=np.float64)
            quat = np.asarray(default_pose[3:7], dtype=np.float64)
            if np.linalg.norm(quat) > 0.0:
                if self.apply_pose_frame_transform:
                    pos, quat = self._transform_pose_frame(pos, quat)
                rotvec = Rotation.from_quat(quat).as_rotvec()
                return [float(pos[0]), float(pos[1]), float(pos[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

        self.get_logger().warning('Parameter %s is invalid, fallback to default.' % name)
        return [float(v) for v in default_pose[:6]]

    def _transform_pose_frame(self, position: np.ndarray, quaternion_xyzw: np.ndarray):
        """对位置和四元数施加绕 z 轴 180° 旋转，用于在不同坐标系约定之间转换。"""
        frame_rotation = Rotation.from_euler('z', np.pi)
        t_pos = frame_rotation.apply(np.asarray(position, dtype=np.float64))
        t_quat = (frame_rotation * Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64))).as_quat()
        return t_pos, t_quat

    def _init_robot(self):
        """延迟导入并连接 RTDE 控制/接收接口；连接失败时禁用运动模式。"""
        if not self.enable_motion:
            self.get_logger().warning('enable_motion is false, robot motion disabled.')
            return

        global rtde_control, rtde_receive
        if rtde_control is None:
            try:
                import importlib

                rtde_control = importlib.import_module('rtde_control')
            except Exception:
                self.get_logger().error('Cannot import rtde_control, disabling robot motion.')
                self.enable_motion = False
                return

        if rtde_receive is None:
            try:
                import importlib

                rtde_receive = importlib.import_module('rtde_receive')
            except Exception:
                self.get_logger().warning('Cannot import rtde_receive, actual TCP pose verification will fallback.')

        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.get_logger().info('Connected RTDE control to %s' % self.robot_ip)
        except Exception as exc:
            self.get_logger().error('Failed to connect RTDE control: %s' % exc)
            self.enable_motion = False

        if rtde_receive is not None:
            try:
                self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
                self.get_logger().info('Connected RTDE receive to %s' % self.robot_ip)
            except Exception as exc:
                self.rtde_r = None
                self.get_logger().warning('Failed to connect RTDE receive: %s' % exc)

    def _init_suction(self):
        """打开串口连接吸盘控制器；失败时记录警告并以无吸盘模式继续。"""
        try:
            self.serial_suction = serial.Serial(self.suction_device, 115200, timeout=1, bytesize=8)
            self.get_logger().info('Opened suction serial %s' % self.suction_device)
        except Exception as exc:
            self.get_logger().warning('Cannot open suction device %s: %s' % (self.suction_device, exc))

    def camera_info_callback(self, msg: CameraInfo):
        """接收相机内参消息，提取 3×3 内参矩阵。"""
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def image_callback(self, rgb_msg: Image, depth_msg: Image):
        """时间同步 RGB+深度帧回调：转换为 numpy 数组并更新调试窗口。"""
        try:
            self.latest_rgb = np.asarray(self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8'))
            self.latest_depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'))
            self.latest_stamp = rgb_msg.header.stamp
        except CvBridgeError as exc:
            self.get_logger().error('CvBridge conversion failed: %s' % exc)
            return

        if self.show_debug_window and not self._debug_window_failed:
            self._update_debug_window()

    def _update_debug_window(self):
        """在调试窗口上叠加当前检测目标轮廓与轨迹预测可视化。"""
        if self.latest_rgb is None:
            return
        vis = self.latest_rgb.copy()

        self._draw_motion_prediction_overlay(vis)

        target = self._detect_largest_red_target(self.latest_rgb, self.latest_depth)
        if target is not None:
            cv2.drawContours(vis, [target.contour], -1, (0, 0, 255), 2)
            cv2.circle(vis, target.center_px, 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                'red area: %.0f' % target.area,
                (target.center_px[0] + 8, target.center_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        try:
            cv2.imshow('Phase2 Moving Red Monitor', vis)
            cv2.waitKey(1)
        except Exception as exc:
            self._debug_window_failed = True
            self.get_logger().warning('Debug window disabled: %s' % exc)

    def _draw_motion_prediction_overlay(self, vis_image: np.ndarray):
        """在图像上绘制拟合圆轨迹、预测抓取中心、预测轮廓和边缘方向箭头。"""
        with self._viz_lock:
            center = None if self._fitted_center_base is None else np.asarray(self._fitted_center_base, dtype=np.float64)
            radius = self._fitted_radius
            z_ref = self._fitted_z_ref
            predicted_pick = None if self._predicted_pick_base is None else np.asarray(self._predicted_pick_base, dtype=np.float64)
            predicted_contour = None if self._predicted_contour_base is None else np.asarray(self._predicted_contour_base, dtype=np.float64)
            predicted_edge = None if self._predicted_edge_dir_base is None else np.asarray(self._predicted_edge_dir_base, dtype=np.float64)

        if center is not None and radius is not None and z_ref is not None and radius > 1e-5:
            traj_points_px = []
            for theta in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
                p_base = np.array(
                    [
                        center[0] + radius * np.cos(theta),
                        center[1] + radius * np.sin(theta),
                        z_ref,
                    ],
                    dtype=np.float64,
                )
                uv = self._project_base_point_to_pixel(p_base)
                if uv is not None:
                    traj_points_px.append(uv)

            if len(traj_points_px) >= 3:
                pts = np.asarray(traj_points_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_image, [pts], True, (255, 255, 0), 2)
                cv2.putText(
                    vis_image,
                    'fitted trajectory',
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        if predicted_pick is not None:
            uv = self._project_base_point_to_pixel(predicted_pick)
            if uv is not None:
                u, v = uv
                cv2.drawMarker(
                    vis_image,
                    (u, v),
                    (255, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=18,
                    thickness=2,
                )
                cv2.putText(
                    vis_image,
                    'predicted pick center',
                    (u + 8, max(20, v - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if predicted_contour is not None and len(predicted_contour) >= 3:
            contour_px = []
            for p in predicted_contour:
                uv = self._project_base_point_to_pixel(p)
                if uv is not None:
                    contour_px.append(uv)
            if len(contour_px) >= 3:
                pts = np.asarray(contour_px, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis_image, [pts], True, (0, 255, 255), 2)
                cv2.putText(
                    vis_image,
                    'predicted contour',
                    (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if predicted_pick is not None and predicted_edge is not None:
            edge_end = predicted_pick + predicted_edge * 0.05
            p0 = self._project_base_point_to_pixel(predicted_pick)
            p1 = self._project_base_point_to_pixel(edge_end)
            if p0 is not None and p1 is not None:
                cv2.arrowedLine(vis_image, p0, p1, (255, 0, 0), 2, tipLength=0.2)
                cv2.putText(
                    vis_image,
                    'edge // camera_link z',
                    (max(10, p0[0] + 8), max(20, p0[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

    def _extract_valid_contours(self, mask: np.ndarray):
        """从二值掩膜中提取面积达标且形状为四边形的有效轮廓列表。"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.color_min_area:
                clean_mask[labels == label] = 255

        contours_result = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_result) == 2:
            contours, _ = contours_result
        else:
            _, contours, _ = contours_result

        valid = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0.0:
                continue
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            valid.append((contour, float(area)))
        return valid

    def _detect_largest_red_target(self, rgb_image: np.ndarray, depth_image: np.ndarray):
        """在 RGB+深度帧中检测面积最大的红色四边形目标，返回带 3D 中心坐标的 RedTarget。"""
        if rgb_image is None or depth_image is None:
            return None

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges_red:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        valid = self._extract_valid_contours(mask)
        if not valid:
            return None

        contour, area = max(valid, key=lambda item: item[1])
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

        points_3d = self._project_contour_to_3d(contour, rgb_image, depth_image)
        if points_3d is None or len(points_3d) == 0:
            center_cam = self._pixel_to_3d(cx, cy, depth_image)
            if center_cam is None:
                return None
        else:
            valid_mask = np.isfinite(points_3d).all(axis=1)
            valid_mask = np.logical_and(valid_mask, points_3d[:, 2] > 0.05)
            valid_mask = np.logical_and(valid_mask, points_3d[:, 2] < 2.0)
            points = points_3d[valid_mask]
            if len(points) == 0:
                center_cam = self._pixel_to_3d(cx, cy, depth_image)
                if center_cam is None:
                    return None
            else:
                center_cam = np.median(points, axis=0)

        return RedTarget(
            area=area,
            contour=contour,
            center_px=(cx, cy),
            center_cam=np.asarray(center_cam, dtype=np.float64),
        )

    def _pixel_to_3d(self, u: int, v: int, depth_image: np.ndarray):
        """将像素坐标反投影为相机坐标系下的 3D 点；深度无效时返回 None。"""
        if self.camera_matrix is None:
            return None
        if v < 0 or v >= depth_image.shape[0] or u < 0 or u >= depth_image.shape[1]:
            return None

        depth_value = depth_image[v, u]
        if depth_value == 0:
            return None

        if np.issubdtype(depth_image.dtype, np.integer):
            z = float(depth_value) / 1000.0
        else:
            z = float(depth_value)
        if z <= 0.0:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    def _project_contour_to_3d(self, contour: np.ndarray, rgb_image: np.ndarray, depth_image: np.ndarray):
        """将轮廓内所有像素反投影为 3D 点云，用于获取目标区域的深度分布。"""
        region_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
        pixel_indices = np.column_stack(np.where(region_mask > 0))

        points_3d = []
        for v, u in pixel_indices:
            p = self._pixel_to_3d(int(u), int(v), depth_image)
            if p is None:
                continue
            points_3d.append(p)

        if not points_3d:
            return None
        return np.asarray(points_3d, dtype=np.float64)

    def _lookup_transform(self, target_frame: str = None, source_frame: str = None):
        """查询 TF 变换（默认 camera_frame→base_frame），超时返回 None。"""
        tgt = self.base_frame if target_frame is None else target_frame
        src = self.camera_frame if source_frame is None else source_frame
        try:
            return self.tf_buffer.lookup_transform(
                tgt,
                src,
                rclpy.time.Time(),
                timeout=Duration(seconds=max(0.0, self.tf_lookup_timeout_sec)),
            )
        except Exception:
            return None

    def _project_base_point_to_pixel(self, point_base: np.ndarray):
        """将 base 坐标系下的 3D 点投影到图像像素坐标，用于调试可视化。"""
        if self.camera_matrix is None:
            return None

        tf_base_to_cam = self._lookup_transform(self.camera_frame, self.base_frame)
        if tf_base_to_cam is None:
            return None

        point_cam = self._transform_point(np.asarray(point_base, dtype=np.float64), tf_base_to_cam)
        z = float(point_cam[2])
        if z <= 1e-6:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        u = int((point_cam[0] * fx / z) + cx)
        v = int((point_cam[1] * fy / z) + cy)
        return (u, v)

    def _transform_point(self, point: np.ndarray, transform_msg):
        """将 3D 点通过 TF 消息变换到目标坐标系。"""
        translation = np.array(
            [
                transform_msg.transform.translation.x,
                transform_msg.transform.translation.y,
                transform_msg.transform.translation.z,
            ],
            dtype=np.float64,
        )
        rotation = Rotation.from_quat(
            [
                transform_msg.transform.rotation.x,
                transform_msg.transform.rotation.y,
                transform_msg.transform.rotation.z,
                transform_msg.transform.rotation.w,
            ]
        )
        return rotation.apply(point) + translation

    def _target_to_base(self, target: RedTarget):
        """将 RedTarget 的相机坐标系中心点变换到 base 坐标系。"""
        tf_msg = self._lookup_transform()
        if tf_msg is None:
            return None
        return self._transform_point(target.center_cam, tf_msg)

    def _transform_vector(self, vector: np.ndarray, transform_msg):
        """Transform a vector (direction) using only rotation part of transform."""
        rotation = Rotation.from_quat(
            [
                transform_msg.transform.rotation.x,
                transform_msg.transform.rotation.y,
                transform_msg.transform.rotation.z,
                transform_msg.transform.rotation.w,
            ]
        )
        return rotation.apply(vector)

    def _camera_z_axis_in_base(self, source_frame: str):
        """Get camera link z-axis direction expressed in base frame."""
        transform = self._lookup_transform(self.base_frame, source_frame)
        if transform is None:
            return None

        axis_base = self._transform_vector(np.array([0.0, 0.0, 1.0], dtype=np.float64), transform)
        axis_norm = float(np.linalg.norm(axis_base))
        if axis_norm <= 1e-9:
            return None
        return axis_base / axis_norm

    def _pixel_to_3d_with_search(self, u: int, v: int, depth_image: np.ndarray, max_radius: int = 4):
        """反投影像素到 3D；中心深度无效时在周围方形邻域内逐步扩大搜索。"""
        p = self._pixel_to_3d(u, v, depth_image)
        if p is not None:
            return p
        for r in range(1, max_radius + 1):
            for dv in range(-r, r + 1):
                for du in range(-r, r + 1):
                    if abs(du) != r and abs(dv) != r:
                        continue
                    p2 = self._pixel_to_3d(u + du, v + dv, depth_image)
                    if p2 is not None:
                        return p2
        return None

    def _extract_contour_vertices_px(self, contour: np.ndarray):
        """用最小外接矩形拟合轮廓，返回四个角点的像素坐标。"""
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.asarray(box, dtype=np.float64)

    def _contour_polygon_base(self, contour: np.ndarray, depth_image: np.ndarray, tf_msg):
        """将轮廓四角点反投影并变换到 base 坐标系，返回 3D 多边形点数组。"""
        vertices_px = self._extract_contour_vertices_px(contour)
        points_base = []
        for p in vertices_px:
            u = int(round(float(p[0])))
            v = int(round(float(p[1])))
            p_cam = self._pixel_to_3d_with_search(u, v, depth_image)
            if p_cam is None:
                continue
            points_base.append(self._transform_point(p_cam, tf_msg))
        if len(points_base) < 3:
            return None
        return np.asarray(points_base, dtype=np.float64)

    def _longest_edge_direction_xy(self, polygon_base: np.ndarray):
        """计算 3D 多边形在 XY 平面内最长边的单位方向向量。"""
        if polygon_base is None or len(polygon_base) < 2:
            return None
        n = len(polygon_base)
        best_vec = None
        best_len = 0.0
        for i in range(n):
            j = (i + 1) % n
            vec = polygon_base[j] - polygon_base[i]
            vec_xy = np.array([vec[0], vec[1], 0.0], dtype=np.float64)
            l = np.linalg.norm(vec_xy[:2])
            if l > best_len:
                best_len = l
                best_vec = vec_xy
        if best_vec is None or best_len < 1e-6:
            return None
        best_vec = best_vec / np.linalg.norm(best_vec[:2])
        return best_vec

    def _box_dimensions_from_polygon(self, polygon_base: np.ndarray):
        """从 3D 多边形各边长中提取长轴和短轴尺寸（米）。"""
        if polygon_base is None or len(polygon_base) < 3:
            return None
        n = len(polygon_base)
        lengths = []
        for i in range(n):
            j = (i + 1) % n
            vec = polygon_base[j] - polygon_base[i]
            lengths.append(float(np.linalg.norm(vec[:2])))
        lengths = sorted([l for l in lengths if l > 1e-6])
        if len(lengths) < 2:
            return None
        return float(lengths[-1]), float(lengths[0])

    def _predict_rotated_polygon(self, polygon_last: np.ndarray, center_last: np.ndarray, center_pred: np.ndarray, delta_yaw: float):
        """将当前多边形平移到预测中心并绕 z 轴旋转 delta_yaw，预测未来姿态轮廓。"""
        rot = Rotation.from_euler('z', delta_yaw).as_matrix()
        predicted = []
        for p in polygon_last:
            rel = p - center_last
            rel_rot = rot @ rel
            predicted.append(center_pred + rel_rot)
        return np.asarray(predicted, dtype=np.float64)

    def _build_rectangle_polygon(self, center_base: np.ndarray, yaw: float, length_long: float, length_short: float):
        """根据中心、偏航角和长宽构造 base 坐标系下的矩形四顶点多边形。"""
        cx, cy, cz = float(center_base[0]), float(center_base[1]), float(center_base[2])
        hl = 0.5 * float(length_long)
        hs = 0.5 * float(length_short)
        x_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
        y_dir = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)

        corners = [
            np.array([cx, cy, cz], dtype=np.float64) + hl * x_dir + hs * y_dir,
            np.array([cx, cy, cz], dtype=np.float64) - hl * x_dir + hs * y_dir,
            np.array([cx, cy, cz], dtype=np.float64) - hl * x_dir - hs * y_dir,
            np.array([cx, cy, cz], dtype=np.float64) + hl * x_dir - hs * y_dir,
        ]
        return np.asarray(corners, dtype=np.float64), x_dir

    def _move_to_pose(self, pose: list[float], speed: float = None, acceleration: float = None):
        """向机器人发送 moveL 指令，连接断开时自动重试；非运动模式下仅打印目标位姿。"""
        use_speed = self.speed if speed is None else float(speed)
        use_acc = self.acceleration if acceleration is None else float(acceleration)

        if self.enable_motion and self.rtde_c is not None:
            while rclpy.ok():
                try:
                    ok = self.rtde_c.moveL(pose, use_speed, use_acc)
                    if ok:
                        return True
                except Exception:
                    pass

                if hasattr(self.rtde_c, 'reuploadScript'):
                    try:
                        self.rtde_c.reuploadScript()
                    except Exception:
                        pass
                time.sleep(0.3)
            return False

        self.get_logger().info('Motion disabled, skip moveL: %s' % np.array2string(np.asarray(pose), precision=5))
        return True

    def _read_actual_tcp_pose(self):
        """读取机器人当前 TCP 位姿（rotvec 格式）；非运动模式或读取失败时返回 None。"""
        if self.enable_motion and self.rtde_r is not None and hasattr(self.rtde_r, 'getActualTCPPose'):
            try:
                pose = self.rtde_r.getActualTCPPose()
                if pose is not None and len(pose) >= 6:
                    return [float(v) for v in pose[:6]]
            except Exception:
                pass

        if self.enable_motion and self.rtde_c is not None and hasattr(self.rtde_c, 'getActualTCPPose'):
            try:
                pose = self.rtde_c.getActualTCPPose()
                if pose is not None and len(pose) >= 6:
                    return [float(v) for v in pose[:6]]
            except Exception:
                pass

        return None

    def suction_suck(self):
        """向串口发送吸气 Modbus 指令，激活吸盘吸取物块。"""
        if self.serial_suction is None:
            self.get_logger().warning('Suction serial not ready, skip suction command.')
            return False
        try:
            self.serial_suction.write(bytes.fromhex('01 06 00 02 00 01 E9 CA'))
            return True
        except Exception as exc:
            self.get_logger().error('Suction command failed: %s' % exc)
            return False

    def suction_release(self):
        """向串口发送放气 Modbus 指令，释放吸盘。"""
        if self.serial_suction is None:
            self.get_logger().warning('Suction serial not ready, skip release command.')
            return False
        try:
            self.serial_suction.write(bytes.fromhex('01 06 00 02 00 02 A9 CB'))
            return True
        except Exception as exc:
            self.get_logger().error('Release command failed: %s' % exc)
            return False

    def _has_red_target_in_current_view(self):
        """检查当前帧中是否存在有效红色目标，用于确认抓取后物块已离开视野。"""
        rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
        depth = self.latest_depth
        if rgb is None or depth is None:
            return False
        target = self._detect_largest_red_target(rgb, depth)
        return target is not None

    def _move_to_stack_pose_and_confirm(self):
        """移动到放置堆叠位姿并释放吸盘；视野中无目标时逐层降低后重试。"""
        next_level = int(self.current_stack_level) + 1
        next_level = max(int(self.stack_min_level), min(int(self.stack_max_level), next_level))
        self.current_stack_level = next_level

        while True:
            place_pose = self.stack_place_poses.get(int(self.current_stack_level))
            if place_pose is None:
                self.get_logger().error('No place pose configured for stack level %d.' % self.current_stack_level)
                return False

            self.get_logger().info('Move to stack level %d pose for place check.' % self.current_stack_level)
            if not self._move_to_pose(place_pose):
                return False

            if self.place_check_settle_sec > 0.0:
                time.sleep(self.place_check_settle_sec)

            red_found = self._has_red_target_in_current_view()
            self.get_logger().info(
                'Stack level %d check: red_detected=%s' % (self.current_stack_level, str(red_found))
            )

            if red_found:
                break

            if self.current_stack_level <= int(self.stack_min_level):
                self.get_logger().warning(
                    'Reached minimum stack level %d without red detection, use this level for release.'
                    % self.current_stack_level
                )
                break

            self.current_stack_level -= 1

        released = self.suction_release()
        if not released:
            self.get_logger().warning('Release command failed at stack level %d.' % self.current_stack_level)
        return True

    def _estimate_stack_top_z_cam(self):
        """从当前深度帧估计放置堆叠顶面的相机 z 坐标（米），用于推算当前层高。"""
        rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
        depth = self.latest_depth
        if rgb is None or depth is None:
            return None

        z_cam_min = 0.23
        z_cam_max = 0.37

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges_red:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        valid = self._extract_valid_contours(mask)
        if not valid:
            return None

        best_z_cam = None
        for contour, _ in valid:
            points_3d = self._project_contour_to_3d(contour, rgb, depth)
            contour_z_cam = None

            if points_3d is not None and len(points_3d) > 0:
                valid_mask = np.isfinite(points_3d).all(axis=1)
                valid_mask = np.logical_and(valid_mask, points_3d[:, 2] > 0.05)
                valid_mask = np.logical_and(valid_mask, points_3d[:, 2] < 2.0)
                valid_points = points_3d[valid_mask]
                if len(valid_points) > 0:
                    z_candidates = valid_points[:, 2]
                    z_candidates = z_candidates[np.isfinite(z_candidates)]
                    z_candidates = z_candidates[np.logical_and(z_candidates >= z_cam_min, z_candidates < z_cam_max)]
                    if len(z_candidates) > 0:
                        contour_z_cam = float(np.percentile(z_candidates, 10.0))

            if contour_z_cam is None:
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx = x + w // 2
                    cy = y + h // 2
                center_cam = self._pixel_to_3d(cx, cy, depth)
                if center_cam is not None:
                    center_z_cam = float(center_cam[2])
                    if np.isfinite(center_z_cam) and z_cam_min <= center_z_cam < z_cam_max:
                        contour_z_cam = center_z_cam

            # Ignore this contour when z_cam is unavailable (n/a).
            if contour_z_cam is None:
                continue
            if best_z_cam is None or contour_z_cam < best_z_cam:
                best_z_cam = contour_z_cam

        if best_z_cam is None:
            self.get_logger().warning(
                'No valid red target in z_cam range [%.1f,%.1f) for layer check (n/a or out-of-range).'
                % (z_cam_min, z_cam_max)
            )
            return None

        return float(best_z_cam)

    def _place_to_next_layer_and_return_observe(self):
        """推算目标放置层数，移动放置并返回观测位姿，完成一次完整的拾取-放置循环。"""
        target_level = None
        top_z_cam = None
        if int(self.current_stack_level) >= 5:
            target_level = min(10, int(self.current_stack_level) + 1)
            self.get_logger().info(
                'Layer check: recorded n=%d -> sequential place to level %d (k* -> k).'
                % (int(self.current_stack_level), int(target_level))
            )
        else:
            if not self._move_to_pose(self.place_observe_pose):
                self.get_logger().error('Failed to move to place_observe_pose for layer check.')
                return False

            # Wait for pose settling
            if self.place_check_settle_sec > 0.0:
                time.sleep(self.place_check_settle_sec)

            # Wait for robot to stabilize by receiving at least one new camera frame after arrival
            stabilize_timeout_sec = 3.0
            old_stamp = self.latest_stamp
            start_stabilize = time.monotonic()
            while time.monotonic() - start_stabilize < stabilize_timeout_sec:
                if self.latest_stamp is not None and self.latest_stamp != old_stamp:
                    # New frame received after arrival, robot has stabilized
                    break
                time.sleep(0.05)

            if time.monotonic() - start_stabilize >= stabilize_timeout_sec:
                self.get_logger().warning('Robot stabilization timeout at place_observe_pose, continuing anyway...')

            # Now wait for red target to appear if not found (wait up to 10s)
            wait_timeout_sec = 10.0
            wait_start_time = time.monotonic()
            while True:
                top_z_cam = self._estimate_stack_top_z_cam()
                if top_z_cam is not None:
                    break
                
                elapsed = time.monotonic() - wait_start_time
                if elapsed >= wait_timeout_sec:
                    self.get_logger().warning(
                        'Place-layer check timeout: no valid red targets found after %.1f seconds at place_observe_pose.'
                        % wait_timeout_sec
                    )
                    return False
                
                self.get_logger().info(
                    'Waiting for red target at place_observe_pose (elapsed=%.1fs)...'
                    % elapsed
                )
                time.sleep(0.5)

            if 0.3 <= top_z_cam < 0.4:
                target_level = 4
            elif 0.2 <= top_z_cam < 0.3:
                target_level = 5

            if target_level is None:
                self.get_logger().warning(
                    'Stack min z_cam=%.4f outside supported ranges [0.3,0.4)->L4 and [0.2,0.3)->L5.'
                    % top_z_cam
                )
                return False

        if top_z_cam is not None:
            self.get_logger().info(
                'Layer check: min_z_cam=%.4f (lowest valid face), n=%d -> place to level %d.'
                % (top_z_cam, int(self.current_stack_level), int(target_level))
            )

        place_pose = self.stack_place_poses.get(int(target_level))
        if place_pose is None:
            self.get_logger().error('No place pose configured for target level %d.' % target_level)
            return False

        # Reset per-cycle recorded z-rotation; it will be measured from actual poses at release.
        self._recorded_z_rotation_rad = 0.0

        if int(target_level) >= 6:
            place_pre_pose = self.stack_place_pre_poses.get(int(target_level))
            if place_pre_pose is None:
                self.get_logger().error('No pre-place pose configured for target level %d.' % target_level)
                return False
            if not self._move_to_pose(place_pre_pose):
                return False

        if not self._move_to_pose(place_pose):
            return False

        # Measure actual TCP self-z rotation from observe departure to actual release pose.
        release_pose_actual = self._read_actual_tcp_pose()
        pose_from = self._observe_depart_pose if self._observe_depart_pose is not None else self.observe_pose
        pose_to = release_pose_actual if release_pose_actual is not None else place_pose
        self._recorded_z_rotation_rad = self._compute_z_axis_rotation_from_poses(pose_from, pose_to)
        if abs(self._recorded_z_rotation_rad) > 1e-6:
            self.get_logger().info(
                'Recorded z-axis rotation from observe to release: %.4f rad (%.2f deg), source=%s'
                % (
                    self._recorded_z_rotation_rad,
                    np.degrees(self._recorded_z_rotation_rad),
                    'actual' if (self._observe_depart_pose is not None and release_pose_actual is not None) else 'fallback',
                )
            )
        else:
            self.get_logger().info('Recorded z-axis rotation from observe to release is near zero.')

        released = self.suction_release()
        if not released:
            self.get_logger().warning('Release command failed at stack level %d.' % int(target_level))

        if int(target_level) >= 6:
            short_lift_pose = self._offset_pose_along_tcp_local_z(
                place_pose,
                float(self.post_place_short_lift_offset),
                reverse=True,
            )
        else:
            short_lift_pose = list(place_pose)
            short_lift_pose[2] = float(place_pose[2]) + float(self.post_place_short_lift_offset)
        short_lift_speed = max(0.01, float(self.speed) * float(self.post_place_short_lift_speed_scale))
        short_lift_acc = max(0.01, float(self.acceleration) * float(self.post_place_short_lift_acc_scale))
        if not self._move_to_pose(short_lift_pose, speed=short_lift_speed, acceleration=short_lift_acc):
            return False

        recorded_level = int(self.current_stack_level)
        inferred_level = int(target_level)
        if inferred_level > recorded_level:
            self.current_stack_level = inferred_level
            self.get_logger().info(
                'Stack level record updated: n=%d -> %d.'
                % (recorded_level, int(self.current_stack_level))
            )
        elif inferred_level < recorded_level:
            self.get_logger().warning(
                'Stack level mismatch: recorded=%d, inferred=%d. Keep recorded level.'
                % (recorded_level, inferred_level)
            )
        else:
            self.get_logger().info('Stack level record unchanged: n=%d.' % recorded_level)

        # Apply reverse z-axis compensation on stack->home return motion.
        home_pose_to_use = list(self.home_pose)
        if abs(self._recorded_z_rotation_rad) > 1e-6:
            home_pose_to_use = self._apply_z_rotation_compensation_to_pose(
                home_pose_to_use,
                -self._recorded_z_rotation_rad,
            )
            self.get_logger().info(
                'Applying reverse z compensation on stack->home: %.4f rad (%.2f deg)'
                % (-self._recorded_z_rotation_rad, np.degrees(-self._recorded_z_rotation_rad))
            )

        if not self._move_to_pose(home_pose_to_use):
            self.get_logger().error('Placed successfully but failed to return to home_pose with z compensation.')
            return False

        # Verify whether reverse compensation really takes effect at home.
        home_pose_actual = self._read_actual_tcp_pose()
        if home_pose_actual is not None:
            residual_home = self._compute_z_axis_rotation_from_poses(home_pose_to_use, home_pose_actual)
            nominal_home_offset = self._compute_z_axis_rotation_from_poses(self.home_pose, home_pose_to_use)
            self.get_logger().info(
                'Post-compensation home yaw residual: %.4f rad (%.2f deg), nominal offset from home: %.4f rad (%.2f deg)'
                % (
                    residual_home,
                    np.degrees(residual_home),
                    nominal_home_offset,
                    np.degrees(nominal_home_offset),
                )
            )
        else:
            self.get_logger().warning('Could not read actual home TCP pose; skip residual verification log.')

        if not self._move_to_pose(self.observe_pose):
            self.get_logger().error('Placed successfully but failed to return to observe_pose.')
            return False

        return True

    def _build_tool_pose(
        self,
        point_base: np.ndarray,
        suction_dir_base: np.ndarray,
        camera_z_hint_base: np.ndarray = None,
    ):
        """根据目标点和吸盘方向构造 TCP 位姿（rotvec），可选利用相机 z 轴提示对齐工具偏航。"""
        z_axis = np.asarray(suction_dir_base, dtype=np.float64)
        n = np.linalg.norm(z_axis)
        if n == 0.0:
            return None
        z_axis = z_axis / n

        x_axis = None
        y_axis = None
        if camera_z_hint_base is not None:
            hint = np.asarray(camera_z_hint_base, dtype=np.float64)
            hint_proj = hint - np.dot(hint, z_axis) * z_axis
            hint_norm = np.linalg.norm(hint_proj)
            if hint_norm > 1e-6:
                look_dir = hint_proj / hint_norm
                camera_z_tool = np.asarray(self.camera_z_axis_in_tool, dtype=np.float64)
                camera_xy_norm = float(np.linalg.norm(camera_z_tool[:2]))
                if camera_xy_norm > 1e-6:
                    theta = float(np.arctan2(-camera_z_tool[1], camera_z_tool[0]))
                    basis_x = look_dir
                    basis_y = np.cross(z_axis, basis_x)
                    basis_y_norm = np.linalg.norm(basis_y)
                    if basis_y_norm > 1e-6:
                        basis_y = basis_y / basis_y_norm
                        x_axis = np.cos(theta) * basis_x + np.sin(theta) * basis_y
                        y_axis = np.cross(z_axis, x_axis)
                        x_norm = np.linalg.norm(x_axis)
                        y_norm = np.linalg.norm(y_axis)
                        if x_norm > 1e-6 and y_norm > 1e-6:
                            x_axis = x_axis / x_norm
                            y_axis = y_axis / y_norm
                        else:
                            x_axis = None
                            y_axis = None
                else:
                    x_axis = look_dir
                    y_axis = np.cross(z_axis, x_axis)

        if x_axis is None:
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(seed, z_axis)) > 0.95:
                seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            x_axis = np.cross(seed, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm == 0.0:
                return None
            x_axis = x_axis / x_norm

        if y_axis is None:
            y_axis = np.cross(z_axis, x_axis)
            y_norm = np.linalg.norm(y_axis)
            if y_norm == 0.0:
                return None
            y_axis = y_axis / y_norm

        rot = np.column_stack((x_axis, y_axis, z_axis))
        rotvec = Rotation.from_matrix(rot).as_rotvec()
        return [
            float(point_base[0]),
            float(point_base[1]),
            float(point_base[2]),
            float(rotvec[0]),
            float(rotvec[1]),
            float(rotvec[2]),
        ]

    def _rotate_pose_about_local_z(self, pose: list[float], delta_yaw: float):
        """在工具本地 z 轴绕附加偏航角，返回旋转后的新位姿。"""
        rot_curr = Rotation.from_rotvec(np.asarray(pose[3:6], dtype=np.float64))
        rot_new = rot_curr * Rotation.from_euler('z', float(delta_yaw))
        rotvec_new = rot_new.as_rotvec()
        return [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
            float(rotvec_new[0]),
            float(rotvec_new[1]),
            float(rotvec_new[2]),
        ]

    def _offset_pose_along_tcp_local_z(self, pose: list[float], distance: float, reverse: bool = False):
        """沿工具本地 z 轴（或反向）偏移 distance 米，方向由工具姿态决定。"""
        rot = Rotation.from_rotvec(np.asarray(pose[3:6], dtype=np.float64)).as_matrix()
        z_axis = np.asarray(rot[:, 2], dtype=np.float64)
        direction = -z_axis if reverse else z_axis
        pos = np.asarray(pose[:3], dtype=np.float64) + direction * float(distance)
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(pose[3]),
            float(pose[4]),
            float(pose[5]),
        ]

    def _compute_z_axis_rotation_from_poses(self, pose_from: list[float], pose_to: list[float]) -> float:
        """Compute relative yaw (z-axis) rotation from pose_from to pose_to."""
        if pose_from is None or pose_to is None or len(pose_from) < 6 or len(pose_to) < 6:
            return 0.0

        rotvec_from = np.asarray(pose_from[3:6], dtype=np.float64)
        rotvec_to = np.asarray(pose_to[3:6], dtype=np.float64)

        rot_from = Rotation.from_rotvec(rotvec_from)
        rot_to = Rotation.from_rotvec(rotvec_to)
        rot_relative = rot_from.inv() * rot_to

        euler_angles = rot_relative.as_euler('zyx', degrees=False)
        return float(euler_angles[0])

    def _apply_z_rotation_compensation_to_pose(self, pose: list[float], rotation_compensation_rad: float) -> list[float]:
        """Apply local TCP z-axis rotation compensation to a pose."""
        if pose is None or len(pose) < 6 or abs(rotation_compensation_rad) < 1e-6:
            return pose

        pos = np.asarray(pose[:3], dtype=np.float64)
        rotvec = np.asarray(pose[3:6], dtype=np.float64)

        rot_current = Rotation.from_rotvec(rotvec)
        rot_z_compensation = Rotation.from_euler('z', rotation_compensation_rad, degrees=False)
        rot_compensated = rot_current * rot_z_compensation
        rotvec_compensated = rot_compensated.as_rotvec()

        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(rotvec_compensated[0]),
            float(rotvec_compensated[1]),
            float(rotvec_compensated[2]),
        ]

    def _pose_axes(self, pose: list[float]):
        """从 rotvec 位姿提取 x/y/z 三个工具轴方向向量（base 坐标系）。"""
        rot = Rotation.from_rotvec(np.asarray(pose[3:6], dtype=np.float64)).as_matrix()
        x_axis = rot[:, 0]
        y_axis = rot[:, 1]
        z_axis = rot[:, 2]
        return x_axis, y_axis, z_axis

    def _signed_angle_around_axis(self, v1: np.ndarray, v2: np.ndarray, axis: np.ndarray):
        """计算 v1 → v2 绕 axis 的有符号旋转角（弧度）。"""
        a1 = np.asarray(v1, dtype=np.float64)
        a2 = np.asarray(v2, dtype=np.float64)
        ax = np.asarray(axis, dtype=np.float64)
        n1 = np.linalg.norm(a1)
        n2 = np.linalg.norm(a2)
        na = np.linalg.norm(ax)
        if n1 < 1e-8 or n2 < 1e-8 or na < 1e-8:
            return 0.0
        a1 = a1 / n1
        a2 = a2 / n2
        ax = ax / na
        sin_val = np.dot(ax, np.cross(a1, a2))
        cos_val = np.clip(np.dot(a1, a2), -1.0, 1.0)
        return float(np.arctan2(sin_val, cos_val))

    def _realtime_edge_direction_base(self):
        """从当前帧实时检测红色目标并返回其在 base 坐标系下的最长边方向。"""
        rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
        depth = self.latest_depth
        if rgb is None or depth is None:
            return None

        target = self._detect_largest_red_target(rgb, depth)
        if target is None:
            return None

        tf_msg = self._lookup_transform()
        if tf_msg is None:
            return None

        polygon_base = self._contour_polygon_base(target.contour, depth, tf_msg)
        return self._longest_edge_direction_xy(polygon_base)

    def _realtime_target_center_base(self):
        """从当前帧实时检测红色目标并返回其在 base 坐标系下的中心点。"""
        rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
        depth = self.latest_depth
        if rgb is None or depth is None:
            return None

        target = self._detect_largest_red_target(rgb, depth)
        if target is None:
            return None

        tf_msg = self._lookup_transform()
        if tf_msg is None:
            return None
        return self._transform_point(target.center_cam, tf_msg)

    def _compute_camera_z_parallel_delta_yaw(self, pose: list[float], edge_dir: np.ndarray):
        """计算使相机 z 轴在 XY 平面内平行于目标边缘方向所需的偏航调整量（弧度）。"""
        if edge_dir is None:
            return None

        x_curr, y_curr, z_curr = self._pose_axes(pose)
        camera_z_curr = (
            x_curr * float(self.camera_z_axis_in_tool[0])
            + y_curr * float(self.camera_z_axis_in_tool[1])
            + z_curr * float(self.camera_z_axis_in_tool[2])
        )
        camera_z_curr_proj = camera_z_curr - np.dot(camera_z_curr, z_curr) * z_curr
        camera_z_curr_proj_norm = np.linalg.norm(camera_z_curr_proj)
        if camera_z_curr_proj_norm < 1e-8:
            return None
        camera_z_curr_proj = camera_z_curr_proj / camera_z_curr_proj_norm

        edge_proj = np.asarray(edge_dir, dtype=np.float64) - np.dot(edge_dir, z_curr) * z_curr
        edge_proj_norm = np.linalg.norm(edge_proj)
        if edge_proj_norm < 1e-8:
            return None
        edge_proj = edge_proj / edge_proj_norm

        # Parallel condition allows opposite direction; choose the closer one for minimal yaw correction.
        if np.dot(camera_z_curr_proj, edge_proj) < 0.0:
            edge_proj = -edge_proj

        return self._signed_angle_around_axis(camera_z_curr_proj, edge_proj, z_curr)

    def _final_align_camera_z_parallel_at_grasp(self, grasp_pose: list[float], lift_pose: list[float]):
        """在抓取点迭代微调工具偏航，使相机 z 轴对齐目标边缘方向直到误差低于阈值。"""
        max_iters = int(self.final_parallel_align_max_iters)
        tol_rad = float(np.deg2rad(self.final_parallel_align_tol_deg))
        if max_iters <= 0:
            return grasp_pose, lift_pose, False, 0.0

        pose_grasp = list(grasp_pose)
        pose_lift = list(lift_pose)
        applied = False
        last_delta = 0.0

        for _ in range(max_iters):
            edge_dir = self._realtime_edge_direction_base()
            if edge_dir is None and self._predicted_edge_dir_base is not None:
                edge_dir = np.asarray(self._predicted_edge_dir_base, dtype=np.float64)
            delta_yaw = self._compute_camera_z_parallel_delta_yaw(pose_grasp, edge_dir)
            if delta_yaw is None:
                break

            last_delta = float(delta_yaw)
            if abs(delta_yaw) <= tol_rad:
                break

            pose_grasp = self._rotate_pose_about_local_z(pose_grasp, delta_yaw)
            pose_lift = self._rotate_pose_about_local_z(pose_lift, delta_yaw)
            if not self._move_to_pose(
                pose_grasp,
                speed=self.near_grasp_rotate_speed,
                acceleration=self.near_grasp_rotate_acceleration,
            ):
                return grasp_pose, lift_pose, False, last_delta
            applied = True

        return pose_grasp, pose_lift, applied, last_delta

    def _estimate_segment_travel_time(self, distance: float, speed: float, acceleration: float):
        """根据梯形速度曲线估算给定距离的移动时间（秒）。"""
        d = max(0.0, float(distance))
        v = max(1e-3, float(speed))
        a = max(1e-3, float(acceleration))
        t_ramp = v / a
        d_ramp = 0.5 * a * t_ramp * t_ramp
        if d <= 2.0 * d_ramp:
            # Triangular profile.
            return 2.0 * np.sqrt(d / a)
        d_flat = d - 2.0 * d_ramp
        return 2.0 * t_ramp + d_flat / v

    def _refine_grasp_center_near_grasp(
        self,
        approach_pose: list[float],
        grasp_pose: list[float],
        lift_pose: list[float],
    ):
        """在接近抓取点时用实时目标中心（或轨道预测）修正 grasp/lift 位姿的 XY 坐标。"""
        center_now = self._realtime_target_center_base()
        if center_now is None:
            self.get_logger().warning('Near-grasp center refine skipped: no realtime target center.')
            return grasp_pose, lift_pose, False

        center_pick = np.asarray(center_now, dtype=np.float64)
        source = 'realtime_now'

        if self._orbit_center_xy is not None and self._orbit_radius is not None and self._orbit_omega is not None:
            orbit_center = np.asarray(self._orbit_center_xy, dtype=np.float64)
            rel = center_pick[:2] - orbit_center
            if np.linalg.norm(rel) > 1e-6:
                descent_dist = np.linalg.norm(
                    np.asarray(grasp_pose[:3], dtype=np.float64) - np.asarray(approach_pose[:3], dtype=np.float64)
                )
                lead_dt = self._estimate_segment_travel_time(
                    descent_dist,
                    self.approach_to_grasp_speed,
                    self.approach_to_grasp_acceleration,
                )
                theta_now = float(np.arctan2(rel[1], rel[0]))
                theta_pick = theta_now + float(self._orbit_omega) * lead_dt
                center_pick[0] = orbit_center[0] + float(self._orbit_radius) * np.cos(theta_pick)
                center_pick[1] = orbit_center[1] + float(self._orbit_radius) * np.sin(theta_pick)
                source = 'orbit_lead'

        grasp_pose_new = grasp_pose.copy()
        lift_pose_new = lift_pose.copy()
        old_xy = np.asarray(grasp_pose[:2], dtype=np.float64)
        new_xy = np.asarray(center_pick[:2], dtype=np.float64)

        grasp_pose_new[0] = float(new_xy[0])
        grasp_pose_new[1] = float(new_xy[1])
        lift_pose_new[0] = float(new_xy[0])
        lift_pose_new[1] = float(new_xy[1])

        delta_mm = 1000.0 * float(np.linalg.norm(new_xy - old_xy))
        self.get_logger().info(
            'Near-grasp center refine: source=%s, xy_shift=%.1f mm -> (%.4f, %.4f)'
            % (source, delta_mm, new_xy[0], new_xy[1])
        )
        return grasp_pose_new, lift_pose_new, True

    def _refine_grasp_orientation_near_grasp(self, grasp_pose: list[float], lift_pose: list[float]):
        """在接近抓取点时用实时边缘方向（含角速度前瞻）修正 grasp/lift 位姿的偏航角。"""
        edge_dir = self._realtime_edge_direction_base()
        source = 'realtime'
        edge_lead_deg = 0.0
        if edge_dir is not None and self._edge_yaw_rate is not None:
            lead_time = self._estimate_approach_to_grasp_direction_lead_time()
            lead_yaw = float(self._edge_yaw_rate) * lead_time
            edge_dir = Rotation.from_euler('z', lead_yaw).apply(np.asarray(edge_dir, dtype=np.float64))
            edge_lead_deg = float(np.degrees(lead_yaw))
            source = 'realtime+lead'
        if edge_dir is None and self._predicted_edge_dir_base is not None:
            edge_dir = np.asarray(self._predicted_edge_dir_base, dtype=np.float64)
            source = 'predicted_fallback'
        if edge_dir is None:
            self.get_logger().warning('Near-grasp orientation refine skipped: no realtime/predicted edge direction.')
            return grasp_pose, lift_pose, 0.0, False

        delta_yaw = self._compute_camera_z_parallel_delta_yaw(grasp_pose, edge_dir)
        if delta_yaw is None:
            self.get_logger().warning('Near-grasp orientation refine skipped: invalid camera-z/edge geometry.')
            return grasp_pose, lift_pose, 0.0, False
        if abs(delta_yaw) < 1e-4:
            self.get_logger().info(
                'Near-grasp orientation refine not needed: delta_yaw=%.3f deg (%s, edge_lead=%.3f deg).'
                % (np.degrees(delta_yaw), source, edge_lead_deg)
            )
            return grasp_pose, lift_pose, delta_yaw, False

        grasp_pose_new = self._rotate_pose_about_local_z(grasp_pose, delta_yaw)
        lift_pose_new = self._rotate_pose_about_local_z(lift_pose, delta_yaw)
        self.get_logger().info(
            'Near-grasp orientation refine: delta_yaw=%.3f deg (%s, edge_lead=%.3f deg, camera_link z // contour edge)'
            % (np.degrees(delta_yaw), source, edge_lead_deg)
        )
        return grasp_pose_new, lift_pose_new, delta_yaw, True

    def _build_descend_stage_pose(self, approach_pose: list[float], grasp_pose: list[float], delta_yaw: float, ratio: float):
        """按 ratio 在接近→抓取路径上插值并叠加比例偏航旋转，构造下降阶段中间位姿。"""
        r = float(np.clip(ratio, 0.0, 1.0))
        if self.descent_vertical_only:
            z_val = float(approach_pose[2]) + r * (float(grasp_pose[2]) - float(approach_pose[2]))
            pos = np.array([float(approach_pose[0]), float(approach_pose[1]), z_val], dtype=np.float64)
        else:
            pos = np.asarray(approach_pose[:3], dtype=np.float64) + r * (
                np.asarray(grasp_pose[:3], dtype=np.float64) - np.asarray(approach_pose[:3], dtype=np.float64)
            )
        stage_pose = [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(approach_pose[3]),
            float(approach_pose[4]),
            float(approach_pose[5]),
        ]
        return self._rotate_pose_about_local_z(stage_pose, float(delta_yaw) * r)

    def _move_approach_to_grasp_with_staged_rotation(
        self,
        approach_pose: list[float],
        grasp_pose: list[float],
        delta_yaw_refine: float,
    ):
        """分多阶段从接近位移动到抓取位，同步完成偏航旋转修正以减少末端轨迹偏差。"""
        stages = max(1, int(self.descent_rotation_stages))
        if abs(delta_yaw_refine) < 1e-4 or stages == 1:
            return self._move_to_pose(
                grasp_pose,
                speed=self.approach_to_grasp_speed,
                acceleration=self.approach_to_grasp_acceleration,
            )

        self.get_logger().info(
            'Approach->grasp staged descent: stages=%d, total_z_rot=%.3f deg'
            % (stages, np.degrees(delta_yaw_refine))
        )
        for i in range(1, stages + 1):
            ratio = float(i) / float(stages)
            stage_pose = self._build_descend_stage_pose(approach_pose, grasp_pose, delta_yaw_refine, ratio)
            if not self._move_to_pose(
                stage_pose,
                speed=self.approach_to_grasp_speed,
                acceleration=self.approach_to_grasp_acceleration,
            ):
                return False
        return True

    def _wait_for_initial_data(self, timeout_sec: float = 5.0):
        """等待 RGB、深度帧和相机内参均就绪，超时返回 False。"""
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            if self.latest_rgb is not None and self.latest_depth is not None and self.camera_matrix is not None:
                return True
            time.sleep(0.05)
        return False

    def _collect_motion_samples(self, sample_duration_sec: float = None):
        """在观测期内按固定间隔采样目标中心、偏航和尺寸，返回时间序列字典。"""
        duration_sec = float(self.sample_duration_sec if sample_duration_sec is None else sample_duration_sec)
        if duration_sec <= 0.0:
            duration_sec = float(self.sample_duration_sec)

        samples_t = []
        samples_xy = []
        samples_z = []
        samples_yaw = []
        samples_long = []
        samples_short = []

        start = time.monotonic()
        while time.monotonic() - start < duration_sec:
            rgb = None if self.latest_rgb is None else self.latest_rgb.copy()
            depth = self.latest_depth
            if rgb is None or depth is None:
                time.sleep(self.sample_interval_sec)
                continue

            target = self._detect_largest_red_target(rgb, depth)
            if target is None:
                time.sleep(self.sample_interval_sec)
                continue

            tf_msg = self._lookup_transform()
            if tf_msg is None:
                time.sleep(self.sample_interval_sec)
                continue

            center_base = self._transform_point(target.center_cam, tf_msg)
            polygon_base = self._contour_polygon_base(target.contour, depth, tf_msg)
            edge_dir = self._longest_edge_direction_xy(polygon_base)
            dims = self._box_dimensions_from_polygon(polygon_base)
            if edge_dir is None:
                time.sleep(self.sample_interval_sec)
                continue
            if dims is None:
                time.sleep(self.sample_interval_sec)
                continue
            yaw = float(np.arctan2(edge_dir[1], edge_dir[0]))
            length_long, length_short = dims

            t_now = time.monotonic()
            samples_t.append(t_now)
            samples_xy.append([float(center_base[0]), float(center_base[1])])
            samples_z.append(float(center_base[2]))
            samples_yaw.append(yaw)
            samples_long.append(length_long)
            samples_short.append(length_short)
            time.sleep(self.sample_interval_sec)

        if len(samples_t) < max(3, self.min_samples):
            return None

        return {
            't': np.asarray(samples_t, dtype=np.float64),
            'xy': np.asarray(samples_xy, dtype=np.float64),
            'z': np.asarray(samples_z, dtype=np.float64),
            'yaw': np.asarray(samples_yaw, dtype=np.float64),
            'long': np.asarray(samples_long, dtype=np.float64),
            'short': np.asarray(samples_short, dtype=np.float64),
        }

    def _fit_circle(self, xy: np.ndarray):
        """用最小二乘法拟合 XY 采样点的圆轨迹，返回圆心坐标和半径。"""
        x = xy[:, 0]
        y = xy[:, 1]
        a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        b = x * x + y * y
        sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy, c = sol
        r = np.sqrt(max(1e-9, c + cx * cx + cy * cy))
        return np.array([cx, cy], dtype=np.float64), float(r)

    def _estimate_angular_velocity(self, t: np.ndarray, xy: np.ndarray, center: np.ndarray):
        """对目标相对圆心的相位序列做线性拟合，估算轨道角速度（rad/s）。"""
        rel = xy - center[None, :]
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        angles_unwrapped = np.unwrap(angles)
        t0 = t[0]
        coeff = np.polyfit(t - t0, angles_unwrapped, 1)
        omega = float(coeff[0])
        return omega, float(angles_unwrapped[-1]), float(t[-1])

    def _estimate_edge_yaw_rate(self, t: np.ndarray, yaw_samples: np.ndarray):
        """用双角展开线性拟合估算目标边缘偏航角速度（rad/s），消除平行边 π 歧义。"""
        # Use doubled-angle unwrap so parallel edges (theta and theta+pi) are treated equivalent.
        yaw2 = 2.0 * yaw_samples
        yaw2_unwrapped = np.unwrap(yaw2)
        coeff = np.polyfit(t - t[0], yaw2_unwrapped, 1)
        omega_yaw = float(coeff[0] / 2.0)
        yaw_last = float(yaw2_unwrapped[-1] / 2.0)
        return omega_yaw, yaw_last

    def _fit_yaw_from_orbit_phase(self, theta_samples: np.ndarray, yaw_samples: np.ndarray):
        """将双倍偏航对轨道相位做线性拟合，返回斜率与截距，用于预测任意相位时的边缘朝向。"""
        # Fit doubled yaw against orbit phase to reduce pi-flip ambiguity.
        theta_u = np.unwrap(theta_samples)
        yaw2_u = np.unwrap(2.0 * yaw_samples)
        coef = np.polyfit(theta_u, yaw2_u, 1)
        k, b = float(coef[0]), float(coef[1])
        return k, b

    def _predict_target_position(self, center: np.ndarray, radius: float, angle_last: float, omega: float, dt: float, z_val: float):
        """给定圆轨迹参数和提前量 dt，预测目标在 dt 秒后的 3D 位置。"""
        angle_pred = angle_last + omega * dt
        x = center[0] + radius * np.cos(angle_pred)
        y = center[1] + radius * np.sin(angle_pred)
        return np.array([x, y, z_val], dtype=np.float64)

    def _estimate_travel_time(self, target_base: np.ndarray):
        """估算从观测位到抓取点全程所需时间，包含移动、下降和额外余量。"""
        # Prediction starts after sampling at observe pose, not at home pose.
        observe_pos = np.asarray(self.observe_pose[:3], dtype=np.float64)
        dist = np.linalg.norm(np.asarray(target_base[:3], dtype=np.float64) - observe_pos)
        v = max(1e-3, float(self.travel_speed_for_prediction))
        t_to_approach = float(dist / v + self.prediction_extra_sec)

        # Account for approach->grasp descent time so XY prediction aligns to suction moment.
        descent_dist = abs(float(self.approach_offset) - float(self.grasp_offset))
        t_descent = self._estimate_segment_travel_time(
            descent_dist,
            self.approach_to_grasp_speed,
            self.approach_to_grasp_acceleration,
        )
        t_descent_extra = max(0.0, float(self.approach_to_grasp_extra_sec))
        t_motion = float(t_to_approach + t_descent + t_descent_extra)
        return t_motion, float(t_to_approach), float(t_descent), float(t_descent_extra)

    def _estimate_approach_to_grasp_direction_lead_time(self):
        """估算从接近位到抓取位的下降时间，用于边缘方向前瞻补偿。"""
        descent_dist = abs(float(self.approach_offset) - float(self.grasp_offset))
        t_descent = self._estimate_segment_travel_time(
            descent_dist,
            self.approach_to_grasp_speed,
            self.approach_to_grasp_acceleration,
        )
        t_descent_extra = max(0.0, float(self.approach_to_grasp_extra_sec))
        return float(t_descent + t_descent_extra)

    def _run_cycle_once(self):
        """执行单次完整循环：观测→采样→拟合圆→预测→接近→抓取→放置，返回是否成功。"""
        if not self._move_to_pose(self.observe_pose):
            self.get_logger().error('Failed to move to observe_pose.')
            return False
        self.get_logger().info('Arrived at observe_pose.')

        if self.observe_settle_sec > 0.0:
            time.sleep(self.observe_settle_sec)

        self._observe_depart_pose = self._read_actual_tcp_pose()
        if self._observe_depart_pose is not None:
            self.get_logger().info('Captured actual observe departure pose for z-compensation reference.')
        else:
            self.get_logger().warning(
                'Could not read actual observe pose; z-compensation will fallback to configured poses.'
            )

        sample_duration_this_cycle = float(self.sample_duration_sec)
        if not self._first_cycle_sampling_done:
            sample_duration_this_cycle = float(self.first_sample_duration_sec)
            self.get_logger().info(
                'First cycle observation enabled: sampling for %.2fs (subsequent cycles: %.2fs).'
                % (sample_duration_this_cycle, float(self.sample_duration_sec))
            )

        samples = self._collect_motion_samples(sample_duration_this_cycle)
        if samples is None:
            self.get_logger().error('Not enough red-target samples for trajectory fitting.')
            return False

        self._first_cycle_sampling_done = True

        t = samples['t']
        xy = samples['xy']
        z = samples['z']
        yaw_samples = samples['yaw']
        samples_long = samples['long']
        samples_short = samples['short']

        fitted_center, radius = self._fit_circle(xy)
        if self._locked_orbit_center_xy is None:
            self._locked_orbit_center_xy = fitted_center.copy()
            self.get_logger().info(
                'Orbit center locked from first cycle: (%.4f, %.4f)'
                % (self._locked_orbit_center_xy[0], self._locked_orbit_center_xy[1])
            )
        center = self._locked_orbit_center_xy.copy()

        if np.linalg.norm(fitted_center - center) > 1e-4:
            self.get_logger().info(
                'Use locked orbit center: fitted=(%.4f, %.4f), locked=(%.4f, %.4f)'
                % (fitted_center[0], fitted_center[1], center[0], center[1])
            )

        omega, angle_last, t_last = self._estimate_angular_velocity(t, xy, center)
        omega_yaw, _ = self._estimate_edge_yaw_rate(t, yaw_samples)
        z_ref = float(np.median(z))
        long_ref = float(np.median(samples_long))
        short_ref = float(np.median(samples_short))

        theta_samples = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
        theta_samples_u = np.unwrap(theta_samples)
        yaw_k, yaw_b = self._fit_yaw_from_orbit_phase(theta_samples_u, yaw_samples)

        with self._viz_lock:
            self._fitted_center_base = center.copy()
            self._fitted_radius = float(radius)
            self._fitted_z_ref = float(z_ref)
        self._orbit_center_xy = center.copy()
        self._orbit_radius = float(radius)
        self._orbit_omega = float(omega)
        self._edge_yaw_rate = float(omega_yaw)

        self.get_logger().info(
            'Fitted trajectory: center=(%.4f, %.4f), radius=%.4f, omega=%.4f rad/s'
            % (center[0], center[1], radius, omega)
        )

        dt = float(self.pick_prediction_horizon_sec)
        predicted = self._predict_target_position(center, radius, angle_last, omega, dt, z_ref)
        theta_pred = angle_last + omega * dt
        yaw_pred = 0.5 * (yaw_k * theta_pred + yaw_b)
        predicted_polygon_base, predicted_edge_dir_base = self._build_rectangle_polygon(
            predicted,
            yaw_pred,
            long_ref,
            short_ref,
        )

        with self._viz_lock:
            self._predicted_pick_base = predicted.copy()
            self._predicted_contour_base = None if predicted_polygon_base is None else predicted_polygon_base.copy()
            self._predicted_edge_dir_base = None if predicted_edge_dir_base is None else predicted_edge_dir_base.copy()
        self.get_logger().info(
            'Predicted target at fixed horizon dt=%.3fs: [%.4f, %.4f, %.4f], yaw_pred=%.4f, box=(%.4f, %.4f), contour_yaw_rate=%.4f rad/s'
            % (
                dt,
                predicted[0],
                predicted[1],
                predicted[2],
                yaw_pred,
                long_ref,
                short_ref,
                omega_yaw,
            )
        )

        suction_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        
        # Optionally apply camera z-axis compensation to predicted point
        # to account for systematic center bias in detection
        camera_z_comp = float(self.pick_camera_z_compensation_m)
        if abs(camera_z_comp) > 1e-9:
            comp_frame = self.pick_camera_z_compensation_frame
            cam_z_axis_base = self._camera_z_axis_in_base(comp_frame)
            if cam_z_axis_base is not None:
                comp_delta = cam_z_axis_base * camera_z_comp
                predicted = np.asarray(predicted, dtype=np.float64) + comp_delta
                self.get_logger().debug(
                    'Applied camera z-axis compensation: %.4f m, delta=[%.4f, %.4f, %.4f]'
                    % (camera_z_comp, comp_delta[0], comp_delta[1], comp_delta[2])
                )
            else:
                self.get_logger().warning(
                    'pick_camera_z_compensation_m=%.4f requested but TF %s -> %s unavailable, skip compensation.'
                    % (camera_z_comp, comp_frame, self.base_frame)
                )
        
        approach_point = predicted - suction_dir * self.approach_offset
        grasp_point = predicted - suction_dir * self.grasp_offset
        lift_point = grasp_point - suction_dir * self.post_pick_lift_offset

        approach_pose = self._build_tool_pose(approach_point, suction_dir, camera_z_hint_base=predicted_edge_dir_base)
        grasp_pose = self._build_tool_pose(grasp_point, suction_dir, camera_z_hint_base=predicted_edge_dir_base)
        lift_pose = self._build_tool_pose(lift_point, suction_dir, camera_z_hint_base=predicted_edge_dir_base)
        if approach_pose is None or grasp_pose is None or lift_pose is None:
            self.get_logger().error('Failed to build predicted pick poses.')
            return False

        approach_start_time = time.monotonic()
        if not self._move_to_pose(approach_pose):
            return False
        arrive_time = time.monotonic()
        t_travel = max(0.0, float(arrive_time - approach_start_time))
        t_elapsed_since_last_sample = max(0.0, float(arrive_time - t_last))
        effective_horizon = float(self.pick_prediction_horizon_sec) - float(self.pick_timing_advance_sec)
        wait_before_descent = max(0.0, effective_horizon - t_elapsed_since_last_sample)
        self.get_logger().info(
            'Approach timing: travel=%.3fs, elapsed_since_last_sample=%.3fs, wait=%.3fs (horizon=%.3fs, advance=%.3fs).'
            % (t_travel, t_elapsed_since_last_sample, wait_before_descent, float(self.pick_prediction_horizon_sec), float(self.pick_timing_advance_sec))
        )
        if wait_before_descent > 0.0:
            time.sleep(wait_before_descent)

        grasp_pose, lift_pose, delta_yaw_refine, refine_applied = self._refine_grasp_orientation_near_grasp(
            grasp_pose,
            lift_pose,
        )

        if self.descent_vertical_only:
            grasp_pose[0] = float(approach_pose[0])
            grasp_pose[1] = float(approach_pose[1])
            lift_pose[0] = float(approach_pose[0])
            lift_pose[1] = float(approach_pose[1])
            center_refined = False
            self.get_logger().info('Near-grasp center refine skipped: descent_vertical_only=true (XY locked to approach).')
        else:
            grasp_pose, lift_pose, center_refined = self._refine_grasp_center_near_grasp(
                approach_pose,
                grasp_pose,
                lift_pose,
            )

        self.get_logger().info(
            'Approach->grasp move: speed=%.3f, acc=%.3f, z-rot-correction=%s (delta=%.3f deg), center-refine=%s'
            % (
                self.approach_to_grasp_speed,
                self.approach_to_grasp_acceleration,
                str(refine_applied),
                np.degrees(delta_yaw_refine),
                str(center_refined),
            )
        )

        if not self._move_approach_to_grasp_with_staged_rotation(
            approach_pose,
            grasp_pose,
            delta_yaw_refine,
        ):
            return False

        grasp_pose, lift_pose, final_align_applied, final_align_delta = self._final_align_camera_z_parallel_at_grasp(
            grasp_pose,
            lift_pose,
        )
        self.get_logger().info(
            'Final grasp parallel align: applied=%s, residual_delta=%.3f deg, tol=%.3f deg, max_iters=%d'
            % (
                str(final_align_applied),
                np.degrees(final_align_delta),
                self.final_parallel_align_tol_deg,
                self.final_parallel_align_max_iters,
            )
        )

        if not self.suction_suck():
            return False

        time.sleep(0.10)
        if not self._place_to_next_layer_and_return_observe():
            self.get_logger().error('Suction succeeded but place-to-next-layer flow failed.')
            return False

        self.get_logger().info('Suction succeeded, placed to next layer, and returned to observe_pose.')
        return True

    def _run_task(self):
        """任务主循环：等待数据就绪后持续执行拾放循环，直至完成 10 次放置或节点关闭。"""
        self.get_logger().info('Phase2 task started.')

        if not self._wait_for_initial_data(timeout_sec=8.0):
            self.get_logger().error('No synchronized RGB/depth/camera info. Abort.')
            return

        self.get_logger().info(
            'Stack logic enabled: initial=%d, min=%d, max=%d'
            % (self.current_stack_level, self.stack_min_level, self.stack_max_level)
        )

        while rclpy.ok():
            ok = self._run_cycle_once()
            if not ok:
                self.get_logger().warning('Cycle aborted due to failure, retry after short pause.')
                time.sleep(0.5)
                continue
            self.get_logger().info('Successful cycle completed, continue to next cycle.')
            
            # Check if we have reached 10 placed objects
            if int(self.current_stack_level) >= 10:
                self.get_logger().info(
                    'Task completed: reached target stack level %d (10 objects placed).'
                    % int(self.current_stack_level)
                )
                break
            
            time.sleep(0.2)

    def destroy_node(self):
        """关闭调试窗口、串口和 RTDE 连接后销毁节点。"""
        if self.show_debug_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if self.serial_suction is not None:
            try:
                self.serial_suction.close()
            except Exception:
                pass

        if self.rtde_c is not None:
            try:
                self.rtde_c.stopScript()
            except Exception:
                pass

        super().destroy_node()


def main(args=None):
    # --initial-stack-level 是自定义位置参数，必须在 --ros-args 之前解析，
    # 不能直接交给 rclpy.init()，因此手动分割两段 argv。
    parser = argparse.ArgumentParser(description='Phase2 Moving Red Pick Task')
    parser.add_argument(
        '--initial-stack-level',
        type=int,
        default=3,
        help='Initial stack level (default: 3, minimum: 3)',
    )

    if args is None:
        args = sys.argv[1:]
    else:
        args = list(args)

    ros_args_idx = args.index('--ros-args') if '--ros-args' in args else len(args)
    custom_args = args[:ros_args_idx]
    ros_args = args[ros_args_idx:] if ros_args_idx < len(args) else []

    parsed = parser.parse_args(custom_args)
    initial_level = int(parsed.initial_stack_level)

    if initial_level < 3:
        print(f"Error: --initial-stack-level must be >= 3, got {initial_level}", file=sys.stderr)
        sys.exit(1)

    ros2_argv = ['program_name']
    if ros_args:
        ros2_argv.extend(ros_args)
    ros2_argv.extend(['--ros-args', '-p', f'initial_stack_level:={initial_level}'])

    os.environ.setdefault('RCUTILS_CONSOLE_OUTPUT_FORMAT', '{message}')
    rclpy.init(args=ros2_argv[1:])
    node = Phase2MovingRedPick()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
