#!/usr/bin/env python3
"""
pick_and_place.py — Task 1: 静态彩色方块抓放控制器

节点 CubePickPlaceController 在 home_pose 观察各色方块，按优先级选取最高的
绿/红/黄方块，规划接近-抓取-提升路径，执行吸盘抓取，再移动至对应颜色的放置
列堆叠。每色放置 3 个（共 9 次）后进行视觉验证，确认完成后退出。

主要流程::
    home_pose -> 检测目标 -> 接近 -> 抓取 -> 提升 ->
    place_hover_pose -> place_mid_pose -> drop_pose -> release ->
    hover_pose -> home_pose（循环）
"""

import os
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

try:
    import open3d as o3d
except ImportError:
    o3d = None

rtde_control = None


@dataclass
class ColorTarget:
    """单帧中检测到的颜色方块，包含像素、相机坐标系几何信息及颜色标签。"""
    color: str
    area: float
    contour: np.ndarray
    center_px: tuple[int, int]
    center_cam: np.ndarray
    top_cam: np.ndarray
    normal_cam: np.ndarray


class CubePickPlaceController(Node):
    """Task 1 静态颜色方块拾取-放置控制器：检测方块颜色、规划路径、执行抓放并核验堆叠数量。"""
    def __init__(self):
        super().__init__('detect_cube_and_suck')

        # 机器人与传感器基础配置
        self.declare_parameter('robot_ip', '192.168.56.3')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('suction_device', '/dev/ttyUSB0')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        # 帧同步参数
        self.declare_parameter('frame_sync_queue_size', 3)
        self.declare_parameter('frame_sync_slop_sec', 0.04)
        self.declare_parameter('max_accepted_frame_age_sec', 0.25)
        self.declare_parameter('drop_out_of_order_frames', True)
        self.declare_parameter('clear_camera_cache_before_observe', True)
        # 运动速度与几何偏移
        self.declare_parameter('speed', 0.15)
        self.declare_parameter('acceleration', 0.30)
        self.declare_parameter('approach_offset', 0.10)
        self.declare_parameter('grasp_offset', 0.055)
        self.declare_parameter('grasp_descent_offset', 0.02)
        self.declare_parameter('post_pick_short_lift_offset', 0.08)
        self.declare_parameter('lift_offset', 0.20)
        self.declare_parameter('post_pick_center_hover_z', 0.35)
        self.declare_parameter('fixed_lift_z', 0.40)
        self.declare_parameter(
            'pick_transition_pose',
            [-0.194905, 0.259999, 0.341792, -0.437524, 0.896344, 0.069700, 0.016794],
        )
        self.declare_parameter('settle_sec', 0.20)
        # 接近-抓取渐进减速参数
        self.declare_parameter('approach_to_grasp_progressive_slowdown', True)
        self.declare_parameter('approach_to_grasp_slowdown_stages', 4)
        self.declare_parameter('approach_to_grasp_end_speed_scale', 0.40)
        self.declare_parameter('approach_to_grasp_last_stage_ratio', 0.12)
        self.declare_parameter('approach_to_grasp_blend_radius', 0.004)
        # 吸盘控制
        self.declare_parameter('suction_on_delay_sec', 0.10)
        # 观测与完成验证参数
        self.declare_parameter('home_observe_settle_sec', 0.60)
        self.declare_parameter('home_observe_required_new_frames', 3)
        self.declare_parameter('home_observe_wait_timeout_sec', 1.5)
        self.declare_parameter('completion_observe_settle_sec', 0.80)
        self.declare_parameter('completion_observe_required_new_frames', 2)
        self.declare_parameter('completion_observe_wait_timeout_sec', 2.0)
        # TF 与运动开关
        self.declare_parameter('tf_lookup_timeout_sec', 0.03)
        self.declare_parameter('tf_startup_grace_sec', 0.80)
        self.declare_parameter('enable_motion', True)
        # 颜色检测参数
        self.declare_parameter('color_min_area', 300.0)
        self.declare_parameter('color_adaptive_sv_relax', 25)
        self.declare_parameter('color_value_equalize_clip_limit', 2.0)
        self.declare_parameter('color_blur_kernel', 5)
        self.declare_parameter('color_mask_morph_kernel', 5)
        self.declare_parameter('color_mask_open_iters', 1)
        self.declare_parameter('color_mask_close_iters', 1)
        self.declare_parameter('color_min_rectangularity', 0.55)
        # 放置几何参数
        self.declare_parameter('place_drop_heights', [0.095, 0.175, 0.265])
        self.declare_parameter('first_place_drop_height', 0.10)
        self.declare_parameter('place_clearance', 0.012)
        self.declare_parameter('stack_height_step', 0.030)
        self.declare_parameter('place_layer_height_tolerance', 0.08)
        self.declare_parameter('place_hover_offset', 0.35)
        self.declare_parameter('place_hover_observe_sec', 1.0)
        # 调试与方向控制
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('enable_pick_diagnostics', True)
        self.declare_parameter('apply_pose_frame_transform', True)
        self.declare_parameter('pick_camera_face_workspace_center', True)
        self.declare_parameter('use_pick_transition_pose', True)
        self.declare_parameter('pick_camera_axis_in_tool', 'camera_z')
        self.declare_parameter('pick_pose_z_half_turn_correction', False)
        self.declare_parameter('place_pose_z_half_turn_correction', False)
        self.declare_parameter('pick_rotation_speed_scale', 2.0)
        self.declare_parameter('alternate_tcp_rotation_during_pick', True)
        self.declare_parameter('tcp_rotation_step_deg', 25.0)
        # 目标稳定性判断参数
        self.declare_parameter('pick_stability_required_frames', 3)
        self.declare_parameter('pick_stability_timeout_sec', 1.5)
        self.declare_parameter('pick_stability_position_tol_m', 0.008)
        self.declare_parameter('pick_stability_normal_tol_deg', 12.0)
        self.declare_parameter('pick_stability_require_live_tf', False)
        # 吸盘重试与状态查询
        self.declare_parameter('suction_retry_feed_step_m', 0.005)
        self.declare_parameter('suction_retry_max_attempts', 2)
        self.declare_parameter('suction_status_query_cmd_hex', '')
        self.declare_parameter('suction_status_response_len', 7)
        self.declare_parameter('suction_status_success_byte_index', 3)
        self.declare_parameter('suction_status_success_mask', 1)
        self.declare_parameter('suction_status_success_value', 1)
        self.declare_parameter('suction_status_read_timeout_sec', 0.08)
        self.declare_parameter('suction_retry_when_status_unknown', True)
        self.declare_parameter('release_settle_sec', 0.5)
        # TF 缓存与方块 TF 覆盖
        self.declare_parameter('tf_cache_max_age_sec', 0.6)
        self.declare_parameter('use_tf_cube_center_for_pick', False)
        self.declare_parameter('pick_cube_frame', 'cube_frame')
        # 相机 z 轴补偿（消除检测中心偏差）
        self.declare_parameter('pick_camera_z_compensation_m', 0.05)
        self.declare_parameter('pick_camera_z_compensation_frame', 'camera_link')
        # 工作空间定义（base 坐标系）
        self.declare_parameter('workspace_center_x', -0.22)
        self.declare_parameter('workspace_center_y', 0.29)
        self.declare_parameter('workspace_center_z', 0.10)
        self.declare_parameter('pick_workspace_x_min', -0.42)
        self.declare_parameter('pick_workspace_x_max', -0.02)
        self.declare_parameter('pick_workspace_y_min', 0.16)
        self.declare_parameter('pick_workspace_y_max', 0.42)
        self.declare_parameter('pick_workspace_z_min', -0.02)
        self.declare_parameter('pick_workspace_z_max', 0.4)
        self.declare_parameter('detected_center_z_offset', 0.07)
        self.declare_parameter('auto_correct_detection_xy_flip', False)
        # 各颜色方块初始已放置数量
        self.declare_parameter('initial_green_count', 0)
        self.declare_parameter('initial_red_count', 0)
        self.declare_parameter('initial_yellow_count', 0)
        self.declare_parameter(
            'home_pose',
            [-0.163687, 0.207756, 0.497297, 0.364787, 0.924170, 0.100803, -0.051757],
        )
        self.declare_parameter(
            'completion_observe_pose',
            [0.021420, 0.165615, 0.477136, 0.812487, 0.541495, 0.210298, 0.049223],
        )
        self.declare_parameter(
            'place_pose_green',
            [0.310575, 0.234747, 0.094584, 0.832468, 0.553170, -0.029218, -0.012106],
        )
        self.declare_parameter(
            'place_pose_red',
            [0.076850, 0.335232, 0.095866, 0.312712, 0.949606, 0.005125, -0.020816],
        )
        self.declare_parameter(
            'place_pose_yellow',
            [0.204962, 0.359774, 0.096745, 0.697613, 0.716347, 0.012685, -0.004696],
        )
        self.declare_parameter(
            'place_hover_pose_green',
            [0.205854, 0.230666, 0.372731, 0.942018, 0.334388, 0.002118, 0.027958],
        )
        self.declare_parameter(
            'place_hover_pose_red',
            [-0.002263, 0.257214, 0.353725, 0.790075, 0.612940, 0.004711, -0.007983],
        )
        self.declare_parameter(
            'place_hover_pose_yellow',
            [0.118099, 0.319657, 0.355338, 0.857946, 0.513306, 0.020869, -0.003208],
        )
        self.declare_parameter(
            'place_mid_pose_green',
            [0.310575, 0.234747, 0.372731, 0.832468, 0.553170, -0.029218, -0.012106],
        )
        self.declare_parameter(
            'place_mid_pose_red',
            [0.076850, 0.335232, 0.353725, 0.312712, 0.949606, 0.005125, -0.020816],
        )
        self.declare_parameter(
            'place_mid_pose_yellow',
            [0.204962, 0.359774, 0.355338, 0.697613, 0.716347, 0.012685, -0.004696],
        )

        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.suction_device = self.get_parameter('suction_device').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.frame_sync_queue_size = int(
            self.get_parameter('frame_sync_queue_size').get_parameter_value().integer_value
        )
        if self.frame_sync_queue_size < 1:
            self.frame_sync_queue_size = 1
        self.frame_sync_slop_sec = float(
            self.get_parameter('frame_sync_slop_sec').get_parameter_value().double_value
        )
        if self.frame_sync_slop_sec < 0.0:
            self.frame_sync_slop_sec = 0.0
        self.max_accepted_frame_age_sec = float(
            self.get_parameter('max_accepted_frame_age_sec').get_parameter_value().double_value
        )
        if self.max_accepted_frame_age_sec < 0.0:
            self.max_accepted_frame_age_sec = 0.0
        self.drop_out_of_order_frames = bool(
            self.get_parameter('drop_out_of_order_frames').get_parameter_value().bool_value
        )
        self.clear_camera_cache_before_observe = bool(
            self.get_parameter('clear_camera_cache_before_observe').get_parameter_value().bool_value
        )
        self.speed = float(self.get_parameter('speed').get_parameter_value().double_value)
        self.acceleration = float(self.get_parameter('acceleration').get_parameter_value().double_value)
        self.approach_offset = float(self.get_parameter('approach_offset').get_parameter_value().double_value)
        self.grasp_offset = float(self.get_parameter('grasp_offset').get_parameter_value().double_value)
        self.grasp_descent_offset = float(
            self.get_parameter('grasp_descent_offset').get_parameter_value().double_value
        )
        if self.grasp_descent_offset < 0.0:
            self.grasp_descent_offset = 0.0
        self.post_pick_short_lift_offset = float(
            self.get_parameter('post_pick_short_lift_offset').get_parameter_value().double_value
        )
        self.lift_offset = float(self.get_parameter('lift_offset').get_parameter_value().double_value)
        self.post_pick_center_hover_z = float(
            self.get_parameter('post_pick_center_hover_z').get_parameter_value().double_value
        )
        self.fixed_lift_z = float(self.get_parameter('fixed_lift_z').get_parameter_value().double_value)
        self.settle_sec = float(self.get_parameter('settle_sec').get_parameter_value().double_value)
        self.approach_to_grasp_progressive_slowdown = bool(
            self.get_parameter('approach_to_grasp_progressive_slowdown').get_parameter_value().bool_value
        )
        self.approach_to_grasp_slowdown_stages = int(
            self.get_parameter('approach_to_grasp_slowdown_stages').get_parameter_value().integer_value
        )
        if self.approach_to_grasp_slowdown_stages < 1:
            self.approach_to_grasp_slowdown_stages = 1
        self.approach_to_grasp_end_speed_scale = float(
            self.get_parameter('approach_to_grasp_end_speed_scale').get_parameter_value().double_value
        )
        if self.approach_to_grasp_end_speed_scale <= 0.0:
            self.approach_to_grasp_end_speed_scale = 0.40
        if self.approach_to_grasp_end_speed_scale > 1.0:
            self.approach_to_grasp_end_speed_scale = 1.0
        self.approach_to_grasp_last_stage_ratio = float(
            self.get_parameter('approach_to_grasp_last_stage_ratio').get_parameter_value().double_value
        )
        if self.approach_to_grasp_last_stage_ratio <= 0.0:
            self.approach_to_grasp_last_stage_ratio = 0.05
        if self.approach_to_grasp_last_stage_ratio >= 0.9:
            self.approach_to_grasp_last_stage_ratio = 0.9
        self.approach_to_grasp_blend_radius = float(
            self.get_parameter('approach_to_grasp_blend_radius').get_parameter_value().double_value
        )
        if self.approach_to_grasp_blend_radius < 0.0:
            self.approach_to_grasp_blend_radius = 0.0
        self.suction_on_delay_sec = float(
            self.get_parameter('suction_on_delay_sec').get_parameter_value().double_value
        )
        if self.suction_on_delay_sec < 0.0:
            self.suction_on_delay_sec = 0.0
        self.home_observe_settle_sec = float(
            self.get_parameter('home_observe_settle_sec').get_parameter_value().double_value
        )
        if self.home_observe_settle_sec < 0.0:
            self.home_observe_settle_sec = 0.0
        self.home_observe_required_new_frames = int(
            self.get_parameter('home_observe_required_new_frames').get_parameter_value().integer_value
        )
        if self.home_observe_required_new_frames < 1:
            self.home_observe_required_new_frames = 1
        self.home_observe_wait_timeout_sec = float(
            self.get_parameter('home_observe_wait_timeout_sec').get_parameter_value().double_value
        )
        if self.home_observe_wait_timeout_sec < 0.3:
            self.home_observe_wait_timeout_sec = 0.3
        self.completion_observe_settle_sec = float(
            self.get_parameter('completion_observe_settle_sec').get_parameter_value().double_value
        )
        self.completion_observe_required_new_frames = int(
            self.get_parameter('completion_observe_required_new_frames').get_parameter_value().integer_value
        )
        if self.completion_observe_required_new_frames < 1:
            self.completion_observe_required_new_frames = 1
        self.completion_observe_wait_timeout_sec = float(
            self.get_parameter('completion_observe_wait_timeout_sec').get_parameter_value().double_value
        )
        if self.completion_observe_wait_timeout_sec < 0.3:
            self.completion_observe_wait_timeout_sec = 0.3
        self.tf_lookup_timeout_sec = float(
            self.get_parameter('tf_lookup_timeout_sec').get_parameter_value().double_value
        )
        self.tf_startup_grace_sec = float(
            self.get_parameter('tf_startup_grace_sec').get_parameter_value().double_value
        )
        self.enable_motion = bool(self.get_parameter('enable_motion').get_parameter_value().bool_value)
        self.color_min_area = float(self.get_parameter('color_min_area').get_parameter_value().double_value)
        self.color_adaptive_sv_relax = int(
            self.get_parameter('color_adaptive_sv_relax').get_parameter_value().integer_value
        )
        if self.color_adaptive_sv_relax < 0:
            self.color_adaptive_sv_relax = 0
        self.color_value_equalize_clip_limit = float(
            self.get_parameter('color_value_equalize_clip_limit').get_parameter_value().double_value
        )
        if self.color_value_equalize_clip_limit < 0.0:
            self.color_value_equalize_clip_limit = 0.0
        self.color_blur_kernel = int(self.get_parameter('color_blur_kernel').get_parameter_value().integer_value)
        if self.color_blur_kernel < 1:
            self.color_blur_kernel = 1
        if self.color_blur_kernel % 2 == 0:
            self.color_blur_kernel += 1
        self.color_mask_morph_kernel = int(
            self.get_parameter('color_mask_morph_kernel').get_parameter_value().integer_value
        )
        if self.color_mask_morph_kernel < 1:
            self.color_mask_morph_kernel = 1
        if self.color_mask_morph_kernel % 2 == 0:
            self.color_mask_morph_kernel += 1
        self.color_mask_open_iters = int(
            self.get_parameter('color_mask_open_iters').get_parameter_value().integer_value
        )
        self.color_mask_close_iters = int(
            self.get_parameter('color_mask_close_iters').get_parameter_value().integer_value
        )
        self.color_mask_open_iters = max(0, self.color_mask_open_iters)
        self.color_mask_close_iters = max(0, self.color_mask_close_iters)
        self.color_min_rectangularity = float(
            self.get_parameter('color_min_rectangularity').get_parameter_value().double_value
        )
        if self.color_min_rectangularity < 0.0:
            self.color_min_rectangularity = 0.0
        if self.color_min_rectangularity > 1.0:
            self.color_min_rectangularity = 1.0
        self.place_drop_heights = [
            float(v) for v in self.get_parameter('place_drop_heights').get_parameter_value().double_array_value
        ]
        if not self.place_drop_heights:
            self.place_drop_heights = [0.09, 0.18, 0.27]
        self.first_place_drop_height = float(
            self.get_parameter('first_place_drop_height').get_parameter_value().double_value
        )
        self.place_clearance = float(self.get_parameter('place_clearance').get_parameter_value().double_value)
        self.stack_height_step = float(self.get_parameter('stack_height_step').get_parameter_value().double_value)
        self.place_layer_height_tolerance = float(
            self.get_parameter('place_layer_height_tolerance').get_parameter_value().double_value
        )
        self.place_hover_offset = float(self.get_parameter('place_hover_offset').get_parameter_value().double_value)
        self.place_hover_observe_sec = float(self.get_parameter('place_hover_observe_sec').get_parameter_value().double_value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').get_parameter_value().bool_value)
        self.enable_pick_diagnostics = bool(
            self.get_parameter('enable_pick_diagnostics').get_parameter_value().bool_value
        )
        self.o3d_available = o3d is not None
        self.icp_threshold = 0.03
        self.use_base_icp_suction_logic = False
        self.apply_pose_frame_transform = bool(
            self.get_parameter('apply_pose_frame_transform').get_parameter_value().bool_value
        )
        self.flip_pose_parameters_for_base = False
        base_frame_tag = self.base_frame.strip().lower()
        if base_frame_tag == 'base':
            self.use_base_icp_suction_logic = True
        if base_frame_tag == 'base' and self.apply_pose_frame_transform:
            self.get_logger().warning(
                'base_frame=base detected, force disable apply_pose_frame_transform to avoid double frame conversion.'
            )
            self.apply_pose_frame_transform = False
            self.flip_pose_parameters_for_base = True
            self.get_logger().info(
                'base_frame=base: enabling parameter-only pose frame flip while keeping runtime transform disabled.'
            )
        elif base_frame_tag == 'base_link':
            pass
        else:
            self.get_logger().info(
                'base_frame=%s, apply_pose_frame_transform=%s'
                % (self.base_frame, str(self.apply_pose_frame_transform))
            )
        if self.use_base_icp_suction_logic and not self.o3d_available:
            self.get_logger().warning(
                'base_frame=base enabled but open3d is unavailable, fallback to plane-fit normal for suction.'
            )
        self.pick_transition_pose = self._read_pose_parameter(
            'pick_transition_pose',
            [-0.194905, 0.259999, 0.341792, -0.437524, 0.896344, 0.069700, 0.016794],
        )
        self.pick_camera_face_workspace_center = bool(
            self.get_parameter('pick_camera_face_workspace_center').get_parameter_value().bool_value
        )
        self.use_pick_transition_pose = bool(
            self.get_parameter('use_pick_transition_pose').get_parameter_value().bool_value
        )
        self.pick_pose_z_half_turn_correction = bool(
            self.get_parameter('pick_pose_z_half_turn_correction').get_parameter_value().bool_value
        )
        self.place_pose_z_half_turn_correction = bool(
            self.get_parameter('place_pose_z_half_turn_correction').get_parameter_value().bool_value
        )
        self.pick_camera_axis_in_tool = (
            self.get_parameter('pick_camera_axis_in_tool').get_parameter_value().string_value.strip().lower()
        )
        if self.pick_camera_axis_in_tool not in ('x', '-x', 'y', '-y', 'camera_z'):
            self.get_logger().warning(
                'Invalid pick_camera_axis_in_tool=%s, fallback to x.' % self.pick_camera_axis_in_tool
            )
            self.pick_camera_axis_in_tool = 'x'
        self.pick_rotation_speed_scale = float(
            self.get_parameter('pick_rotation_speed_scale').get_parameter_value().double_value
        )
        if self.pick_rotation_speed_scale <= 0.0:
            self.get_logger().warning('Invalid pick_rotation_speed_scale=%.3f, fallback to 1.0.' % self.pick_rotation_speed_scale)
            self.pick_rotation_speed_scale = 1.0
        self.alternate_tcp_rotation_during_pick = bool(
            self.get_parameter('alternate_tcp_rotation_during_pick').get_parameter_value().bool_value
        )
        self.tcp_rotation_step_deg = float(
            self.get_parameter('tcp_rotation_step_deg').get_parameter_value().double_value
        )
        if self.tcp_rotation_step_deg < 0.0:
            self.tcp_rotation_step_deg = 0.0
        if self.tcp_rotation_step_deg > 180.0:
            self.tcp_rotation_step_deg = 180.0
        self.pick_stability_required_frames = int(
            self.get_parameter('pick_stability_required_frames').get_parameter_value().integer_value
        )
        if self.pick_stability_required_frames < 1:
            self.pick_stability_required_frames = 1
        self.pick_stability_timeout_sec = float(
            self.get_parameter('pick_stability_timeout_sec').get_parameter_value().double_value
        )
        if self.pick_stability_timeout_sec < 0.2:
            self.pick_stability_timeout_sec = 0.2
        self.pick_stability_position_tol_m = float(
            self.get_parameter('pick_stability_position_tol_m').get_parameter_value().double_value
        )
        if self.pick_stability_position_tol_m < 0.0005:
            self.pick_stability_position_tol_m = 0.0005
        self.pick_stability_normal_tol_deg = float(
            self.get_parameter('pick_stability_normal_tol_deg').get_parameter_value().double_value
        )
        if self.pick_stability_normal_tol_deg < 0.5:
            self.pick_stability_normal_tol_deg = 0.5
        if self.pick_stability_normal_tol_deg > 90.0:
            self.pick_stability_normal_tol_deg = 90.0
        self.pick_stability_require_live_tf = bool(
            self.get_parameter('pick_stability_require_live_tf').get_parameter_value().bool_value
        )
        self.suction_retry_feed_step_m = float(
            self.get_parameter('suction_retry_feed_step_m').get_parameter_value().double_value
        )
        if self.suction_retry_feed_step_m < 0.0:
            self.suction_retry_feed_step_m = 0.0
        self.suction_retry_max_attempts = int(
            self.get_parameter('suction_retry_max_attempts').get_parameter_value().integer_value
        )
        if self.suction_retry_max_attempts < 0:
            self.suction_retry_max_attempts = 0
        self.suction_status_query_cmd_hex = (
            self.get_parameter('suction_status_query_cmd_hex').get_parameter_value().string_value.strip()
        )
        self.suction_status_response_len = int(
            self.get_parameter('suction_status_response_len').get_parameter_value().integer_value
        )
        if self.suction_status_response_len < 1:
            self.suction_status_response_len = 1
        self.suction_status_success_byte_index = int(
            self.get_parameter('suction_status_success_byte_index').get_parameter_value().integer_value
        )
        self.suction_status_success_mask = int(
            self.get_parameter('suction_status_success_mask').get_parameter_value().integer_value
        )
        if self.suction_status_success_mask < 0:
            self.suction_status_success_mask = 0
        self.suction_status_success_value = int(
            self.get_parameter('suction_status_success_value').get_parameter_value().integer_value
        )
        if self.suction_status_success_value < 0:
            self.suction_status_success_value = 0
        self.suction_status_read_timeout_sec = float(
            self.get_parameter('suction_status_read_timeout_sec').get_parameter_value().double_value
        )
        if self.suction_status_read_timeout_sec < 0.01:
            self.suction_status_read_timeout_sec = 0.01
        self.suction_retry_when_status_unknown = bool(
            self.get_parameter('suction_retry_when_status_unknown').get_parameter_value().bool_value
        )
        self.release_settle_sec = float(
            self.get_parameter('release_settle_sec').get_parameter_value().double_value
        )
        if self.release_settle_sec < 0.0:
            self.release_settle_sec = 0.0
        self.tf_cache_max_age_sec = float(
            self.get_parameter('tf_cache_max_age_sec').get_parameter_value().double_value
        )
        if self.tf_cache_max_age_sec < 0.05:
            self.tf_cache_max_age_sec = 0.05
        self.use_tf_cube_center_for_pick = bool(
            self.get_parameter('use_tf_cube_center_for_pick').get_parameter_value().bool_value
        )
        self.pick_cube_frame = self.get_parameter('pick_cube_frame').get_parameter_value().string_value.strip()
        if not self.pick_cube_frame:
            self.pick_cube_frame = 'cube_frame'
        self.pick_camera_z_compensation_m = float(
            self.get_parameter('pick_camera_z_compensation_m').get_parameter_value().double_value
        )
        self.pick_camera_z_compensation_frame = (
            self.get_parameter('pick_camera_z_compensation_frame').get_parameter_value().string_value.strip()
        )
        if not self.pick_camera_z_compensation_frame:
            self.pick_camera_z_compensation_frame = 'camera_link'
        self.get_logger().info(
            'Pick orientation config: camera_axis=%s, z_half_turn_correction=%s'
            % (self.pick_camera_axis_in_tool, str(self.pick_pose_z_half_turn_correction))
        )
        if self.use_tf_cube_center_for_pick:
            self.get_logger().info(
                'Pick center source: TF override enabled, frame=%s.' % self.pick_cube_frame
            )
        if abs(self.pick_camera_z_compensation_m) > 1e-9:
            self.get_logger().info(
                'Pick center camera-z compensation enabled: %.4fm along %s +z.'
                % (self.pick_camera_z_compensation_m, self.pick_camera_z_compensation_frame)
            )
        if self.place_pose_z_half_turn_correction:
            self.get_logger().info('Place pose z-half-turn correction: enabled')
        # Camera-link z axis expressed in the wrist_3_link / tool frame from the saved eye-in-hand calibration.
        self.camera_z_axis_in_tool = np.array([0.721943, 0.679312, 0.131655], dtype=np.float64)
        self.workspace_center = np.array(
            [
                float(self.get_parameter('workspace_center_x').get_parameter_value().double_value),
                float(self.get_parameter('workspace_center_y').get_parameter_value().double_value),
                float(self.get_parameter('workspace_center_z').get_parameter_value().double_value),
            ],
            dtype=np.float64,
        )
        self.pick_workspace = {
            'x_min': float(self.get_parameter('pick_workspace_x_min').get_parameter_value().double_value),
            'x_max': float(self.get_parameter('pick_workspace_x_max').get_parameter_value().double_value),
            'y_min': float(self.get_parameter('pick_workspace_y_min').get_parameter_value().double_value),
            'y_max': float(self.get_parameter('pick_workspace_y_max').get_parameter_value().double_value),
            'z_min': float(self.get_parameter('pick_workspace_z_min').get_parameter_value().double_value),
            'z_max': float(self.get_parameter('pick_workspace_z_max').get_parameter_value().double_value),
        }
        if self.flip_pose_parameters_for_base:
            self.workspace_center[0] = -float(self.workspace_center[0])
            self.workspace_center[1] = -float(self.workspace_center[1])

            raw_x_min = float(self.pick_workspace['x_min'])
            raw_x_max = float(self.pick_workspace['x_max'])
            raw_y_min = float(self.pick_workspace['y_min'])
            raw_y_max = float(self.pick_workspace['y_max'])

            self.pick_workspace['x_min'] = min(-raw_x_max, -raw_x_min)
            self.pick_workspace['x_max'] = max(-raw_x_max, -raw_x_min)
            self.pick_workspace['y_min'] = min(-raw_y_max, -raw_y_min)
            self.pick_workspace['y_max'] = max(-raw_y_max, -raw_y_min)

            self.get_logger().info(
                'Applied parameter-only frame flip to workspace_center/pick_workspace for base_frame=base.'
            )
        self.detected_center_z_offset = float(
            self.get_parameter('detected_center_z_offset').get_parameter_value().double_value
        )
        self.auto_correct_detection_xy_flip = bool(
            self.get_parameter('auto_correct_detection_xy_flip').get_parameter_value().bool_value
        )

        self.home_pose = self._read_pose_parameter(
            'home_pose',
            [-0.163687, 0.207756, 0.497297, 0.364787, 0.924170, 0.100803, -0.051757],
        )
        self.completion_observe_pose = self._read_pose_parameter(
            'completion_observe_pose',
            [0.116134, 0.309904, 0.475088, 0.874830, -0.464678, -0.000302, -0.136915],
        )
        self.place_poses = {
            'green': self._read_pose_parameter(
                'place_pose_green',
                [0.310575, 0.234747, 0.094584, 0.832468, 0.553170, -0.029218, -0.012106],
            ),
            'red': self._read_pose_parameter(
                'place_pose_red',
                [0.076850, 0.335232, 0.095866, 0.312712, 0.949606, 0.005125, -0.020816],
            ),
            'yellow': self._read_pose_parameter(
                'place_pose_yellow',
                [0.204962, 0.359774, 0.096745, 0.697613, 0.716347, 0.012685, -0.004696],
            ),
        }
        self.place_hover_poses = {
            'green': self._read_pose_parameter(
                'place_hover_pose_green',
                [0.205854, 0.230666, 0.372731, 0.942018, 0.334388, 0.002118, 0.027958],
            ),
            'red': self._read_pose_parameter(
                'place_hover_pose_red',
                [-0.002263, 0.257214, 0.353725, 0.790075, 0.612940, 0.004711, -0.007983],
            ),
            'yellow': self._read_pose_parameter(
                'place_hover_pose_yellow',
                [0.118099, 0.319657, 0.355338, 0.857946, 0.513306, 0.020869, -0.003208],
            ),
        }
        self.place_mid_poses = {
            'green': self._read_pose_parameter(
                'place_mid_pose_green',
                [0.310575, 0.234747, 0.372731, 0.832468, 0.553170, -0.029218, -0.012106],
            ),
            'red': self._read_pose_parameter(
                'place_mid_pose_red',
                [0.076850, 0.335232, 0.353725, 0.312712, 0.949606, 0.005125, -0.020816],
            ),
            'yellow': self._read_pose_parameter(
                'place_mid_pose_yellow',
                [0.204962, 0.359774, 0.355338, 0.697613, 0.716347, 0.012685, -0.004696],
            ),
        }

        # If configured, apply a 180-degree rotation about local Z to all
        # configured place poses and place-hover poses so the runtime does
        # not need to special-case them on each placement.
        if self.place_pose_z_half_turn_correction:
            for k, p in self.place_poses.items():
                try:
                    rotvec = np.asarray(p[3:6], dtype=np.float64)
                    quat = Rotation.from_rotvec(rotvec).as_quat()
                    quat = (Rotation.from_euler('z', np.pi) * Rotation.from_quat(quat)).as_quat()
                    new_rotvec = Rotation.from_quat(quat).as_rotvec()
                    self.place_poses[k][3] = float(new_rotvec[0])
                    self.place_poses[k][4] = float(new_rotvec[1])
                    self.place_poses[k][5] = float(new_rotvec[2])
                except Exception:
                    pass

            for k, p in self.place_hover_poses.items():
                try:
                    rotvec = np.asarray(p[3:6], dtype=np.float64)
                    quat = Rotation.from_rotvec(rotvec).as_quat()
                    quat = (Rotation.from_euler('z', np.pi) * Rotation.from_quat(quat)).as_quat()
                    new_rotvec = Rotation.from_quat(quat).as_rotvec()
                    self.place_hover_poses[k][3] = float(new_rotvec[0])
                    self.place_hover_poses[k][4] = float(new_rotvec[1])
                    self.place_hover_poses[k][5] = float(new_rotvec[2])
                except Exception:
                    pass

            for k, p in self.place_mid_poses.items():
                try:
                    rotvec = np.asarray(p[3:6], dtype=np.float64)
                    quat = Rotation.from_rotvec(rotvec).as_quat()
                    quat = (Rotation.from_euler('z', np.pi) * Rotation.from_quat(quat)).as_quat()
                    new_rotvec = Rotation.from_quat(quat).as_rotvec()
                    self.place_mid_poses[k][3] = float(new_rotvec[0])
                    self.place_mid_poses[k][4] = float(new_rotvec[1])
                    self.place_mid_poses[k][5] = float(new_rotvec[2])
                except Exception:
                    pass

            self.get_logger().info('Applied 180deg Z rotation to configured place/place-hover poses.')

        self.color_ranges = {
            'green': [
                (np.array([50, 94, 90], dtype=np.uint8), np.array([80, 150, 190], dtype=np.uint8)),
            ],
            'yellow': [
                (np.array([20, 110, 100], dtype=np.uint8), np.array([35, 255, 230], dtype=np.uint8)),
            ],
            'red': [
                (np.array([0, 100, 80], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
                (np.array([160, 100, 80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
            ],
        }
        self.color_order = ('green', 'red', 'yellow')
        self.target_goal_count = 3
        self.initial_counts = {
            'green': int(self.get_parameter('initial_green_count').get_parameter_value().integer_value),
            'red': int(self.get_parameter('initial_red_count').get_parameter_value().integer_value),
            'yellow': int(self.get_parameter('initial_yellow_count').get_parameter_value().integer_value),
        }
        for color in self.color_order:
            raw_val = self.initial_counts[color]
            clamped_val = max(0, min(self.target_goal_count, raw_val))
            if raw_val != clamped_val:
                self.get_logger().warning(
                    'initial_%s_count=%d out of range, clamped to %d.' % (color, raw_val, clamped_val)
                )
            self.initial_counts[color] = clamped_val

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_stamp = None
        self._last_image_stamp_sec = None
        self._last_frame_drop_warn_time = 0.0
        self._camera_frame_capture_enabled = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rtde_c = None
        self.serial_suction = None
        self.busy = False
        self.finished = False
        self.motion_paused = False
        self.color_counts = dict(self.initial_counts)
        self._debug_window_failed = False
        self._awaiting_home_refresh = False
        self._home_stamp_before_refresh = None
        self._busy_lock = threading.Lock()
        self._last_tf_missing_warn_time = 0.0
        self._last_tf_cache_warn_time = 0.0
        self._last_cube_tf_warn_time = 0.0
        self._last_camera_z_comp_warn_time = 0.0
        self._startup_time = time.monotonic()
        self._cached_cam_to_base_tf = None
        self._cached_cam_to_base_tf_time = 0.0
        self._last_cam_to_base_tf_source = 'unknown'
        self._base_frame_fallbacks = ['base', 'base_link']
        self._next_tcp_rotation_clockwise = True
        self._warned_suction_status_query_missing = False
        self._recorded_z_rotation_rad = 0.0  # Record TCP z-axis rotation from home to approach
        self._color_mask_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.color_mask_morph_kernel, self.color_mask_morph_kernel),
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=int(self.frame_sync_queue_size),
            slop=float(self.frame_sync_slop_sec),
        )
        self.ts.registerCallback(self.image_callback)

        self._init_robot()
        self._init_suction()

        self.get_logger().info('Moving to the initial observation pose before starting the cycle.')
        if self._move_to_pose(self.home_pose):
            self.get_logger().info('Arrived at home_pose: %s' % self._format_pose(self.home_pose))
            self._awaiting_home_refresh = True
            self._home_stamp_before_refresh = None
            self._startup_time = time.monotonic()
            self.latest_rgb = None
            self.latest_depth = None
            self.latest_stamp = None

        self.get_logger().info(
            'Ready. The node will pick the largest green/red/yellow block, place it by color, and stop after 3 per color.'
        )
        self.get_logger().info(
            'Initial placed counts: green=%d, red=%d, yellow=%d'
            % (self.color_counts['green'], self.color_counts['red'], self.color_counts['yellow'])
        )

        self.timer = self.create_timer(0.2, self.control_loop)

    def _read_pose_parameter(self, name: str, default_pose: list[float]) -> list[float]:
        """读取位姿参数并转换为 rotvec 格式；支持 6 元素 rotvec 或 7 元素四元数输入，失败时退回默认值。"""
        should_transform_pose = self.apply_pose_frame_transform or self.flip_pose_parameters_for_base
        value = [float(v) for v in self.get_parameter(name).get_parameter_value().double_array_value]
        if len(value) == 6:
            if should_transform_pose:
                position = np.asarray(value[:3], dtype=np.float64)
                rotvec = np.asarray(value[3:6], dtype=np.float64)
                transformed_position, transformed_quat = self._transform_pose_frame(position, Rotation.from_rotvec(rotvec).as_quat())
                transformed_rotvec = Rotation.from_quat(transformed_quat).as_rotvec()
                return [
                    float(transformed_position[0]),
                    float(transformed_position[1]),
                    float(transformed_position[2]),
                    float(transformed_rotvec[0]),
                    float(transformed_rotvec[1]),
                    float(transformed_rotvec[2]),
                ]
            return value

        if len(value) == 7:
            position = np.asarray(value[:3], dtype=np.float64)
            quat = np.asarray(value[3:7], dtype=np.float64)
            if np.linalg.norm(quat) > 0.0:
                if should_transform_pose:
                    position, quat = self._transform_pose_frame(position, quat)
                rotvec = Rotation.from_quat(quat).as_rotvec()
                return [float(position[0]), float(position[1]), float(position[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

        if len(default_pose) == 7:
            position = np.asarray(default_pose[:3], dtype=np.float64)
            quat = np.asarray(default_pose[3:7], dtype=np.float64)
            if np.linalg.norm(quat) > 0.0:
                if should_transform_pose:
                    position, quat = self._transform_pose_frame(position, quat)
                rotvec = Rotation.from_quat(quat).as_rotvec()
                return [float(position[0]), float(position[1]), float(position[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

        self.get_logger().warning('Parameter %s is invalid, using default pose.' % name)
        return [float(v) for v in default_pose[:6]]

    def _transform_pose_frame(self, position: np.ndarray, quaternion_xyzw: np.ndarray):
        """对位置和四元数施加绕 z 轴 180° 旋转，用于在不同坐标系约定之间转换。"""
        frame_rotation = Rotation.from_euler('z', np.pi)
        transformed_position = frame_rotation.apply(np.asarray(position, dtype=np.float64))
        transformed_quaternion = (frame_rotation * Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64))).as_quat()
        return transformed_position, transformed_quaternion

    def _transform_rotvec_pose_frame(self, pose: list[float]) -> list[float]:
        """将 rotvec 格式位姿经 _transform_pose_frame 转换后返回新的 rotvec 位姿。"""
        pos = np.asarray(pose[:3], dtype=np.float64)
        rotvec = np.asarray(pose[3:6], dtype=np.float64)
        quat = Rotation.from_rotvec(rotvec).as_quat()
        t_pos, t_quat = self._transform_pose_frame(pos, quat)
        t_rotvec = Rotation.from_quat(t_quat).as_rotvec()
        return [
            float(t_pos[0]),
            float(t_pos[1]),
            float(t_pos[2]),
            float(t_rotvec[0]),
            float(t_rotvec[1]),
            float(t_rotvec[2]),
        ]

    def _pose_position(self, pose: list[float]) -> np.ndarray:
        """提取 rotvec 位姿前三元素作为 3D 位置向量。"""
        return np.asarray(pose[:3], dtype=np.float64)

    def _build_pose_with_rotvec(self, position: np.ndarray, rotvec: np.ndarray):
        """将位置和 rotvec 拼合为 6 元素位姿列表。"""
        return [float(position[0]), float(position[1]), float(position[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]

    def _format_pose(self, pose):
        """将位姿格式化为 5 位精度的字符串，用于日志输出。"""
        return np.array2string(np.asarray(pose, dtype=np.float64), precision=5, separator=', ')

    def _init_robot(self):
        """延迟导入并连接 RTDE 控制接口；连接失败时禁用运动模式。"""
        if not self.enable_motion:
            self.get_logger().warning('enable_motion is false, robot motion is disabled.')
            return

        global rtde_control
        if rtde_control is None:
            try:
                import importlib

                rtde_control = importlib.import_module('rtde_control')
            except Exception:
                self.get_logger().error('Cannot import rtde_control, disabling robot motion.')
                self.enable_motion = False
                return

        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.get_logger().info('Connected RTDE control to %s' % self.robot_ip)
        except Exception as exc:
            self.get_logger().error('Failed to connect RTDE control: %s' % exc)
            self.enable_motion = False

    def _init_suction(self):
        """打开串口连接吸盘控制器；失败时记录警告并以无吸盘模式继续。"""
        try:
            self.serial_suction = serial.Serial(self.suction_device, 115200, timeout=1, bytesize=8)
            self.get_logger().info('Opened suction serial %s' % self.suction_device)
        except Exception as exc:
            self.get_logger().warning(
                'Cannot open %s (%s). If needed run: sudo chmod 777 /dev/ttyUSB0' % (self.suction_device, exc)
            )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """接收相机内参消息，提取 3×3 内参矩阵和畸变系数。"""
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        """时间同步 RGB+深度帧回调：过滤过时和乱序帧，转换为 numpy 数组并触发调试窗口更新。"""
        if not self._camera_frame_capture_enabled:
            return

        stamp_sec = float(rgb_msg.header.stamp.sec) + float(rgb_msg.header.stamp.nanosec) * 1e-9
        if self.drop_out_of_order_frames and self._last_image_stamp_sec is not None:
            if stamp_sec <= float(self._last_image_stamp_sec):
                return

        if self.max_accepted_frame_age_sec > 0.0:
            now_sec = float(self.get_clock().now().nanoseconds) * 1e-9
            frame_age = now_sec - stamp_sec
            if frame_age > float(self.max_accepted_frame_age_sec):
                now_mono = time.monotonic()
                if now_mono - self._last_frame_drop_warn_time > 2.0:
                    self._last_frame_drop_warn_time = now_mono
                    self.get_logger().warning(
                        'Drop stale camera frame: age=%.3fs > %.3fs'
                        % (frame_age, self.max_accepted_frame_age_sec)
                    )
                return

        try:
            self.latest_rgb = np.asarray(self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8'))
            self.latest_depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'))
            self.latest_stamp = rgb_msg.header.stamp
            self._last_image_stamp_sec = stamp_sec
        except CvBridgeError as exc:
            self.get_logger().error('CvBridge conversion failed: %s' % exc)
            return

        if self.show_debug_window and not self._debug_window_failed:
            self._update_debug_window(self.latest_rgb, self.latest_depth)

    def _find_largest_color_blob_2d(self, color: str, rgb_image: np.ndarray, preprocessed_hsv: np.ndarray = None):
        """在 RGB 图像中找到指定颜色的最大色块，返回包含轮廓和中心像素的字典。"""
        combined_mask = self._build_color_mask(color, rgb_image, preprocessed_hsv=preprocessed_hsv)

        _, best_contour, best_approx, best_area = self._extract_largest_contour(combined_mask)
        if best_contour is None or best_approx is None or best_area <= 0.0:
            return None

        moments = cv2.moments(best_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            x, y, w, h = cv2.boundingRect(best_contour)
            cx = x + w // 2
            cy = y + h // 2

        return {
            'color': color,
            'area': float(best_area),
            'contour': best_contour,
            'approx': best_approx,
            'center': (cx, cy),
        }

    def _preprocess_rgb_for_color_mask(self, rgb_image: np.ndarray):
        """对 RGB 图像做高斯模糊和 CLAHE 均衡化并转换为 HSV，用于颜色掩膜生成。"""
        blur_ksize = int(self.color_blur_kernel)
        processed = rgb_image
        if blur_ksize > 1:
            processed = cv2.GaussianBlur(processed, (blur_ksize, blur_ksize), 0)

        hsv_image = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        if self.color_value_equalize_clip_limit > 0.0:
            hsv_image = hsv_image.copy()
            v_channel = hsv_image[:, :, 2]
            clahe = cv2.createCLAHE(clipLimit=self.color_value_equalize_clip_limit, tileGridSize=(8, 8))
            hsv_image[:, :, 2] = clahe.apply(v_channel)
        return hsv_image

    def _build_color_mask_from_hsv(self, color: str, hsv_image: np.ndarray):
        """从 HSV 图像生成指定颜色的二值掩膜，支持自适应饱和度/明度松弛和形态学后处理。"""
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        relax = int(self.color_adaptive_sv_relax)
        for lower, upper in self.color_ranges[color]:
            lower_bound = lower.copy()
            upper_bound = upper.copy()
            if relax > 0:
                lower_bound[1] = np.uint8(max(0, int(lower_bound[1]) - relax))
                lower_bound[2] = np.uint8(max(0, int(lower_bound[2]) - relax))
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_image, lower_bound, upper_bound))

        if self.color_mask_open_iters > 0:
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_OPEN,
                self._color_mask_kernel,
                iterations=self.color_mask_open_iters,
            )
        if self.color_mask_close_iters > 0:
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_CLOSE,
                self._color_mask_kernel,
                iterations=self.color_mask_close_iters,
            )

        return combined_mask

    def _build_color_mask(self, color: str, rgb_image: np.ndarray, preprocessed_hsv: np.ndarray = None):
        """从 RGB 图像生成指定颜色的二值掩膜；可传入预处理好的 HSV 以避免重复计算。"""
        hsv_image = preprocessed_hsv
        if hsv_image is None:
            hsv_image = self._preprocess_rgb_for_color_mask(rgb_image)

        return self._build_color_mask_from_hsv(color, hsv_image)

    def _project_base_point_to_pixel(self, point_base: np.ndarray):
        """将 base 坐标系下的 3D 点投影到图像像素坐标，用于调试可视化。"""
        if self.camera_matrix is None:
            return None

        tf_base_to_cam = self._lookup_transform(self.camera_frame, self.base_frame, warn=False)
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

    def _draw_workspace_overlay(self, vis_image: np.ndarray):
        """在调试图像上叠加拾取工作空间的三维线框投影。"""
        x_min = float(self.pick_workspace['x_min'])
        x_max = float(self.pick_workspace['x_max'])
        y_min = float(self.pick_workspace['y_min'])
        y_max = float(self.pick_workspace['y_max'])
        z_min = float(self.pick_workspace['z_min'])
        z_max = float(self.pick_workspace['z_max'])

        bottom_corners = [
            np.array([x_min, y_min, z_min], dtype=np.float64),
            np.array([x_max, y_min, z_min], dtype=np.float64),
            np.array([x_max, y_max, z_min], dtype=np.float64),
            np.array([x_min, y_max, z_min], dtype=np.float64),
        ]
        top_corners = [
            np.array([x_min, y_min, z_max], dtype=np.float64),
            np.array([x_max, y_min, z_max], dtype=np.float64),
            np.array([x_max, y_max, z_max], dtype=np.float64),
            np.array([x_min, y_max, z_max], dtype=np.float64),
        ]

        bottom_uv = [self._project_base_point_to_pixel(p) for p in bottom_corners]
        top_uv = [self._project_base_point_to_pixel(p) for p in top_corners]

        if all(p is not None for p in bottom_uv):
            pts = np.asarray(bottom_uv, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis_image, [pts], True, (255, 200, 0), 2)
        if all(p is not None for p in top_uv):
            pts = np.asarray(top_uv, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis_image, [pts], True, (255, 200, 0), 2)
        if all(p is not None for p in bottom_uv) and all(p is not None for p in top_uv):
            for i in range(4):
                cv2.line(vis_image, bottom_uv[i], top_uv[i], (255, 200, 0), 2)

        center_uv = self._project_base_point_to_pixel(self.workspace_center)
        if center_uv is not None:
            cv2.drawMarker(
                vis_image,
                center_uv,
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=16,
                thickness=2,
            )
            cv2.putText(
                vis_image,
                'workspace center',
                (center_uv[0] + 8, max(20, center_uv[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_planned_camera_direction_overlay(self, vis_image: np.ndarray, target: ColorTarget):
        """在调试图像上绘制规划好的相机朝向箭头，可视化工具轴对齐效果。"""
        center_base, normal_base, _ = self._target_to_base(target)
        suction_dir = -np.asarray(normal_base, dtype=np.float64)
        suction_norm = np.linalg.norm(suction_dir)
        if suction_norm < 1e-8:
            return
        suction_dir = suction_dir / suction_norm

        approach_point = np.asarray(center_base, dtype=np.float64) - suction_dir * float(self.approach_offset)
        look_vec = np.asarray(self.workspace_center, dtype=np.float64) - approach_point
        look_vec_proj = look_vec - np.dot(look_vec, suction_dir) * suction_dir
        look_norm = np.linalg.norm(look_vec_proj)
        if look_norm < 1e-8:
            return
        look_dir = look_vec_proj / look_norm

        start_uv = self._project_base_point_to_pixel(approach_point)
        end_uv = self._project_base_point_to_pixel(approach_point + look_dir * 0.06)
        if start_uv is None or end_uv is None:
            return

        cv2.arrowedLine(vis_image, start_uv, end_uv, (255, 0, 255), 2, tipLength=0.25)
        cv2.putText(
            vis_image,
            'planned camera facing dir',
            (start_uv[0] + 8, max(20, start_uv[1] + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_pick_target_pose_overlay(self, vis_image: np.ndarray, target: ColorTarget):
        """在调试图像上绘制规划位姿的接近/抓取点投影，辅助调试拾取路径。"""
        center_base, normal_base, _, approach_pose, grasp_pose, _ = self._plan_pick_poses(target)
        if approach_pose is None or grasp_pose is None:
            return

        center_uv = self._project_base_point_to_pixel(np.asarray(center_base, dtype=np.float64))
        approach_uv = self._project_base_point_to_pixel(np.asarray(approach_pose[:3], dtype=np.float64))
        grasp_uv = self._project_base_point_to_pixel(np.asarray(grasp_pose[:3], dtype=np.float64))
        if approach_uv is None or grasp_uv is None:
            return

        cv2.circle(vis_image, grasp_uv, 5, (0, 255, 255), -1)
        if center_uv is not None:
            cv2.circle(vis_image, center_uv, 4, (255, 255, 255), -1)
            cv2.line(vis_image, grasp_uv, center_uv, (255, 255, 255), 1)

        cv2.arrowedLine(vis_image, approach_uv, grasp_uv, (255, 255, 0), 2, tipLength=0.18)

        rot = Rotation.from_rotvec(np.asarray(grasp_pose[3:6], dtype=np.float64)).as_matrix()
        axis_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]
        axis_length = 0.04
        grasp_point = np.asarray(grasp_pose[:3], dtype=np.float64)
        for axis_idx, axis_color in enumerate(axis_colors):
            axis_end = grasp_point + rot[:, axis_idx] * axis_length
            axis_end_uv = self._project_base_point_to_pixel(axis_end)
            if axis_end_uv is not None:
                cv2.line(vis_image, grasp_uv, axis_end_uv, axis_color, 2)

        suction_dir = -np.asarray(normal_base, dtype=np.float64)
        suction_norm = np.linalg.norm(suction_dir)
        if suction_norm > 1e-8:
            suction_dir = suction_dir / suction_norm
            suction_end_uv = self._project_base_point_to_pixel(grasp_point + suction_dir * 0.05)
            if suction_end_uv is not None:
                cv2.arrowedLine(vis_image, grasp_uv, suction_end_uv, (0, 255, 255), 2, tipLength=0.2)

        cv2.putText(
            vis_image,
            'grasp TCP pose',
            (grasp_uv[0] + 8, max(20, grasp_uv[1] + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _update_debug_window(self, rgb_image: np.ndarray, depth_image: np.ndarray = None) -> None:
        """在调试窗口上叠加颜色掩膜、工作空间框线和拾取目标位姿可视化。"""
        vis_image = rgb_image.copy()
        self._draw_workspace_overlay(vis_image)
        preprocessed_hsv = self._preprocess_rgb_for_color_mask(rgb_image)

        bgr_map = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
        }

        color_blobs = []
        for color in self.color_order:
            blob = self._find_largest_color_blob_2d(color, rgb_image, preprocessed_hsv=preprocessed_hsv)
            if blob is None:
                continue
            color_blobs.append(blob)

            bgr = bgr_map[color]
            z_text = 'z: n/a'
            if depth_image is not None:
                target = self._build_color_target_from_contour(
                    color,
                    blob['contour'],
                    blob['area'],
                    rgb_image,
                    depth_image,
                )
                if target is not None:
                    center_base, _, _ = self._target_to_base(target)
                    z_text = 'z: %.4f' % float(center_base[2])

            cv2.drawContours(vis_image, [blob['approx']], -1, bgr, 2)
            cv2.circle(vis_image, blob['center'], 5, bgr, -1)
            cv2.putText(
                vis_image,
                '%s area: %.0f %s' % (color, blob['area'], z_text),
                (blob['center'][0] + 8, blob['center'][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )

        if color_blobs and depth_image is not None:
            planned_target, _ = self._select_pick_target_from_frame(
                rgb_image,
                depth_image,
                log_reject=False,
                preprocessed_hsv=preprocessed_hsv,
            )
            if planned_target is not None:
                planned_blob = next((b for b in color_blobs if b['color'] == planned_target.color), None)
                if planned_blob is not None:
                    x, y, w, h = cv2.boundingRect(planned_blob['contour'])
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(
                        vis_image,
                        'planned pick: %s' % planned_target.color,
                        (max(0, x), max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                self._draw_planned_camera_direction_overlay(vis_image, planned_target)
                self._draw_pick_target_pose_overlay(vis_image, planned_target)

        try:
            cv2.imshow('PickAndPlace RGB', vis_image)
            cv2.waitKey(1)
        except Exception as exc:
            self._debug_window_failed = True
            self.get_logger().warning('Debug window disabled: %s' % exc)

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

    def _parse_hex_command(self, hex_text: str):
        """将带空格的十六进制字符串解析为 bytes，格式无效时返回 None。"""
        if hex_text is None:
            return None
        stripped = ''.join(hex_text.split())
        if len(stripped) == 0 or len(stripped) % 2 != 0:
            return None
        try:
            return bytes.fromhex(stripped)
        except Exception:
            return None

    def _read_suction_attached_state(self):
        """向吸盘发送状态查询指令并解析响应字节，返回是否已吸附或 None（无法判断）。"""
        if self.serial_suction is None:
            return None

        query_cmd = self._parse_hex_command(self.suction_status_query_cmd_hex)
        if query_cmd is None:
            if not self._warned_suction_status_query_missing:
                self._warned_suction_status_query_missing = True
                self.get_logger().warning(
                    'suction_status_query_cmd_hex is empty/invalid, cannot verify suction state from hardware.'
                )
            return None

        try:
            old_timeout = self.serial_suction.timeout
            self.serial_suction.timeout = self.suction_status_read_timeout_sec
            self.serial_suction.reset_input_buffer()
            self.serial_suction.write(query_cmd)
            response = self.serial_suction.read(self.suction_status_response_len)
            self.serial_suction.timeout = old_timeout
        except Exception as exc:
            self.get_logger().warning('Suction status read failed: %s' % exc)
            return None

        if response is None or len(response) <= self.suction_status_success_byte_index:
            return None

        data_byte = int(response[self.suction_status_success_byte_index])
        active_value = data_byte & int(self.suction_status_success_mask)
        return active_value == int(self.suction_status_success_value)

    def _compute_z_axis_rotation_from_poses(self, pose_from: list[float], pose_to: list[float]) -> float:
        """计算从 pose_from 到 pose_to 的 z 轴（偏航）旋转量（弧度）。"""
        if pose_from is None or pose_to is None or len(pose_from) < 6 or len(pose_to) < 6:
            return 0.0

        rot_from = Rotation.from_rotvec(np.asarray(pose_from[3:6], dtype=np.float64))
        rot_to = Rotation.from_rotvec(np.asarray(pose_to[3:6], dtype=np.float64))
        rot_relative = rot_from.inv() * rot_to
        return float(rot_relative.as_euler('zyx', degrees=False)[0])

    def _apply_z_rotation_compensation_to_pose(self, pose: list[float], rotation_compensation_rad: float) -> list[float]:
        """在工具 z 轴方向对位姿叠加补偿旋转（弧度），用于抵消积累的偏航偏差。"""
        if pose is None or len(pose) < 6 or abs(rotation_compensation_rad) < 1e-6:
            return pose

        pos = np.asarray(pose[:3], dtype=np.float64)
        rot_compensated = (
            Rotation.from_rotvec(np.asarray(pose[3:6], dtype=np.float64))
            * Rotation.from_euler('z', rotation_compensation_rad, degrees=False)
        )
        rotvec_compensated = rot_compensated.as_rotvec()
        return [
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(rotvec_compensated[0]),
            float(rotvec_compensated[1]),
            float(rotvec_compensated[2]),
        ]

    def _build_tool_pose(
        self,
        point_base: np.ndarray,
        suction_dir_base: np.ndarray,
        look_at_point_base: np.ndarray = None,
        camera_axis_in_tool: str = 'x',
    ):
        """根据目标点和吸盘方向构造 TCP 位姿（rotvec），可选用 look_at 约束对齐相机轴。"""
        z_axis = np.asarray(suction_dir_base, dtype=np.float64)
        norm = np.linalg.norm(z_axis)
        if norm == 0.0:
            return None
        z_axis = z_axis / norm

        x_axis = None
        y_axis = None

        if look_at_point_base is not None:
            look_vec = np.asarray(look_at_point_base, dtype=np.float64) - np.asarray(point_base, dtype=np.float64)
            look_vec_proj = look_vec - np.dot(look_vec, z_axis) * z_axis
            look_norm = np.linalg.norm(look_vec_proj)
            if look_norm > 1e-6:
                look_dir = look_vec_proj / look_norm
                axis_tag = camera_axis_in_tool.strip().lower()
                if axis_tag == 'camera_z':
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
                            y_norm = np.linalg.norm(y_axis)
                            x_norm = np.linalg.norm(x_axis)
                            if x_norm > 1e-6 and y_norm > 1e-6:
                                x_axis = x_axis / x_norm
                                y_axis = y_axis / y_norm
                            else:
                                x_axis = None
                                y_axis = None
                    else:
                        x_axis = look_dir
                        y_axis = np.cross(z_axis, x_axis)
                if axis_tag in ('x', '-x'):
                    x_axis = look_dir if axis_tag == 'x' else -look_dir
                    y_axis = np.cross(z_axis, x_axis)
                elif axis_tag in ('y', '-y'):
                    y_axis = look_dir if axis_tag == 'y' else -look_dir
                    x_axis = np.cross(y_axis, z_axis)

                if x_axis is not None and y_axis is not None:
                    x_norm = np.linalg.norm(x_axis)
                    y_norm = np.linalg.norm(y_axis)
                    if x_norm > 1e-6 and y_norm > 1e-6:
                        x_axis = x_axis / x_norm
                        y_axis = y_axis / y_norm
                    else:
                        x_axis = None
                        y_axis = None

        if x_axis is None or y_axis is None:
            seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(seed, z_axis)) > 0.95:
                seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)

            x_axis = np.cross(seed, z_axis)
            x_norm = np.linalg.norm(x_axis)
            if x_norm == 0.0:
                return None
            x_axis = x_axis / x_norm

            y_axis = np.cross(z_axis, x_axis)
            y_norm = np.linalg.norm(y_axis)
            if y_norm == 0.0:
                return None
            y_axis = y_axis / y_norm

        # Correct a fixed 180-degree yaw mismatch observed on pick approach/grasp poses
        # by rotating the tool frame around its local z axis.
        if self.pick_pose_z_half_turn_correction:
            x_axis = -x_axis
            y_axis = -y_axis

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

    def _move_to_pose(self, pose: list[float], speed: float = None, acceleration: float = None) -> bool:
        """向机器人发送 moveL 指令，连接中断时自动等待恢复；非运动模式仅打印。"""
        use_speed = self.speed if speed is None else float(speed)
        use_acceleration = self.acceleration if acceleration is None else float(acceleration)
        if self.enable_motion and self.rtde_c is not None:
            while rclpy.ok():
                try:
                    move_ok = self.rtde_c.moveL(pose, use_speed, use_acceleration)
                    if move_ok:
                        if self.motion_paused:
                            self.motion_paused = False
                            self.get_logger().info('RTDE control recovered, resuming motion sequence.')
                        self._cached_cam_to_base_tf = None
                        self._cached_cam_to_base_tf_time = 0.0
                        return True

                    if not self.motion_paused:
                        self.motion_paused = True
                        self.get_logger().warning('RTDE motion paused: moveL returned False. Waiting for recovery...')

                    if hasattr(self.rtde_c, 'reuploadScript'):
                        try:
                            self.rtde_c.reuploadScript()
                        except Exception:
                            pass

                except Exception as exc:
                    if not self.motion_paused:
                        self.motion_paused = True
                        self.get_logger().warning('RTDE motion paused due to moveL exception: %s' % exc)

                    if hasattr(self.rtde_c, 'reuploadScript'):
                        try:
                            self.rtde_c.reuploadScript()
                        except Exception:
                            pass

                time.sleep(0.5)

            return False

        self.get_logger().info('Motion disabled, skipping moveL: %s' % np.array2string(np.asarray(pose), precision=5))
        return True

    def _move_through_poses_with_blend(
        self,
        poses: list[list[float]],
        speeds: list[float],
        accelerations: list[float],
        blend_radius: float,
    ) -> bool:
        """以路径 moveL 执行多段带混合半径的连续运动；控制器不支持时退回到逐段 moveL。"""
        if len(poses) == 0:
            return True

        if not self.enable_motion or self.rtde_c is None:
            for idx, pose in enumerate(poses):
                self.get_logger().info(
                    'Motion disabled, skipping blended stage %d/%d: %s'
                    % (idx + 1, len(poses), np.array2string(np.asarray(pose), precision=5))
                )
            return True

        if len(poses) != len(speeds) or len(poses) != len(accelerations):
            return False

        path = []
        final_idx = len(poses) - 1
        for idx, pose in enumerate(poses):
            blend = float(blend_radius) if idx < final_idx else 0.0
            path.append(
                [
                    float(pose[0]),
                    float(pose[1]),
                    float(pose[2]),
                    float(pose[3]),
                    float(pose[4]),
                    float(pose[5]),
                    float(speeds[idx]),
                    float(accelerations[idx]),
                    float(blend),
                ]
            )

        try:
            move_ok = self.rtde_c.moveL(path)
            if move_ok:
                if self.motion_paused:
                    self.motion_paused = False
                    self.get_logger().info('RTDE control recovered, resuming motion sequence.')
                self._cached_cam_to_base_tf = None
                self._cached_cam_to_base_tf_time = 0.0
                return True
        except Exception:
            pass

        # Fallback for controllers that do not support path moveL API.
        self.get_logger().warning('Blended path moveL unavailable, fallback to staged moveL calls.')
        for idx, pose in enumerate(poses):
            if not self._move_to_pose(pose, speed=speeds[idx], acceleration=accelerations[idx]):
                return False
        return True

    def _read_actual_tcp_pose(self):
        """读取机器人当前 TCP 位姿（rotvec 格式）；未连接或读取失败时返回 None。"""
        if not self.enable_motion or self.rtde_c is None:
            return None

        getter = getattr(self.rtde_c, 'getActualTCPPose', None)
        if getter is None:
            return None

        try:
            pose = getter()
        except Exception:
            return None

        if pose is None:
            return None

        return [float(v) for v in pose]

    def _build_tcp_spin_pose(self, current_pose: list[float], clockwise: bool):
        """在当前位姿基础上绕工具 z 轴旋转一个步进角，用于微调工具偏航。"""
        if current_pose is None or len(current_pose) < 6:
            return None
        if self.tcp_rotation_step_deg <= 0.0:
            return None

        spin_rad = np.deg2rad(self.tcp_rotation_step_deg)
        if clockwise:
            spin_rad = -abs(spin_rad)
        else:
            spin_rad = abs(spin_rad)

        current_rot = Rotation.from_rotvec(np.asarray(current_pose[3:6], dtype=np.float64))
        spun_rot = current_rot * Rotation.from_euler('z', spin_rad)
        spun_rotvec = spun_rot.as_rotvec()
        return [
            float(current_pose[0]),
            float(current_pose[1]),
            float(current_pose[2]),
            float(spun_rotvec[0]),
            float(spun_rotvec[1]),
            float(spun_rotvec[2]),
        ]

    def _move_approach_to_grasp_with_progressive_slowdown(self, approach_pose: list[float], grasp_pose: list[float]):
        """分多阶段从接近位移动到抓取位，速度逐步降低以提高末端定位精度。"""
        if (not self.approach_to_grasp_progressive_slowdown) or self.approach_to_grasp_slowdown_stages <= 1:
            return self._move_to_pose(grasp_pose)

        stages = int(self.approach_to_grasp_slowdown_stages)
        start_speed = float(self.speed)
        end_speed = max(0.01, float(self.speed) * float(self.approach_to_grasp_end_speed_scale))
        start_acc = float(self.acceleration)
        end_acc = max(0.01, float(self.acceleration) * float(self.approach_to_grasp_end_speed_scale))

        self.get_logger().info(
            'Approach->grasp progressive slowdown: stages=%d, speed %.3f->%.3f, acc %.3f->%.3f, blend=%.4f, last_stage_ratio=%.3f'
            % (
                stages,
                start_speed,
                end_speed,
                start_acc,
                end_acc,
                self.approach_to_grasp_blend_radius,
                self.approach_to_grasp_last_stage_ratio,
            )
        )

        approach_pos = np.asarray(approach_pose[:3], dtype=np.float64)
        grasp_pos = np.asarray(grasp_pose[:3], dtype=np.float64)
        stage_poses = []
        stage_speeds = []
        stage_accs = []
        final_ratio = float(self.approach_to_grasp_last_stage_ratio)
        if stages >= 2:
            ratio_list = []
            for stage_idx in range(1, stages):
                ratio_list.append((1.0 - final_ratio) * float(stage_idx) / float(stages - 1))
            ratio_list.append(1.0)
        else:
            ratio_list = [1.0]

        for ratio in ratio_list:
            stage_pos = approach_pos + ratio * (grasp_pos - approach_pos)
            stage_pose = [
                float(stage_pos[0]),
                float(stage_pos[1]),
                float(stage_pos[2]),
                float(grasp_pose[3]),
                float(grasp_pose[4]),
                float(grasp_pose[5]),
            ]
            stage_speed = start_speed + ratio * (end_speed - start_speed)
            stage_acc = start_acc + ratio * (end_acc - start_acc)
            stage_poses.append(stage_pose)
            stage_speeds.append(float(stage_speed))
            stage_accs.append(float(stage_acc))

        return self._move_through_poses_with_blend(
            stage_poses,
            stage_speeds,
            stage_accs,
            float(self.approach_to_grasp_blend_radius),
        )

    def _wait_for_frames(self, old_stamp, timeout_sec: float = 2.0) -> bool:
        """等待至少一帧新的 RGB+深度帧到达，超时后返回当前帧是否有效。"""
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_sec:
            if self.latest_rgb is not None and self.latest_depth is not None and self.latest_stamp != old_stamp:
                return True
            time.sleep(0.05)
        return self.latest_rgb is not None and self.latest_depth is not None

    def _clear_latest_camera_frames(self, reason: str = ''):
        """清除当前缓存的 RGB/深度帧和时间戳，强制下次检测使用新帧。"""
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_stamp = None
        if reason:
            self.get_logger().info('Cleared latest camera frame cache: %s' % reason)

    def _set_camera_frame_capture_enabled(self, enabled: bool, clear_cache: bool = False, reason: str = ''):
        """启用或禁用相机帧捕获，可选同时清除帧缓存。"""
        enabled = bool(enabled)
        if self._camera_frame_capture_enabled == enabled and not clear_cache:
            return

        self._camera_frame_capture_enabled = enabled
        if clear_cache:
            self._clear_latest_camera_frames(reason)
        elif not enabled:
            self._clear_latest_camera_frames(reason)

        state_text = 'enabled' if enabled else 'disabled'
        if reason:
            self.get_logger().info('Camera frame capture %s: %s' % (state_text, reason))
        else:
            self.get_logger().info('Camera frame capture %s.' % state_text)

    def _wait_for_multiple_fresh_frames(self, old_stamp, required_frames: int, timeout_sec: float) -> bool:
        """等待指定数量的新鲜帧到达，超时或任务结束时返回 False。"""
        required = max(1, int(required_frames))
        timeout = max(0.1, float(timeout_sec))
        fresh_count = 0
        last_stamp = old_stamp
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            # Check if task is finished to allow early exit
            if self.finished:
                return False
                
            if self.latest_rgb is None or self.latest_depth is None or self.latest_stamp is None:
                time.sleep(0.03)
                continue

            if self.latest_stamp != last_stamp:
                fresh_count += 1
                last_stamp = self.latest_stamp
                if fresh_count >= required:
                    return True
            time.sleep(0.03)

        return False

    def _wait_for_strict_new_frame(self, old_stamp, timeout_sec: float = 1.0) -> bool:
        """等待一帧严格意义上的新帧（时间戳变化），_wait_for_multiple_fresh_frames 的简便包装。"""
        return self._wait_for_multiple_fresh_frames(old_stamp, required_frames=1, timeout_sec=timeout_sec)

    def _lookup_transform(self, target_frame: str, source_frame: str, warn: bool = True):
        """查询 TF 变换；超时时使用缓存（camera→base），缓存过期则返回 None。"""
        timeout_sec = max(0.0, float(self.tf_lookup_timeout_sec))
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_sec),
            )

            # Cache the frequently used camera->base transform for short TF dropouts.
            if target_frame == self.base_frame and source_frame == self.camera_frame:
                self._cached_cam_to_base_tf = transform
                self._cached_cam_to_base_tf_time = time.monotonic()
                self._last_cam_to_base_tf_source = 'live'
            return transform
        except Exception as exc:
            if (
                target_frame == self.base_frame
                and source_frame == self.camera_frame
                and self._cached_cam_to_base_tf is not None
            ):
                cache_age = time.monotonic() - float(self._cached_cam_to_base_tf_time)
                if cache_age <= float(self.tf_cache_max_age_sec):
                    now = time.monotonic()
                    if now - self._last_tf_cache_warn_time > 2.0:
                        self._last_tf_cache_warn_time = now
                        self.get_logger().warning(
                            'TF lookup timeout, using cached transform (%s -> %s), age=%.3fs.'
                            % (source_frame, target_frame, cache_age)
                        )
                    self._last_cam_to_base_tf_source = 'cached'
                    return self._cached_cam_to_base_tf

                self._cached_cam_to_base_tf = None
                self._cached_cam_to_base_tf_time = 0.0

            if target_frame == self.base_frame and source_frame == self.camera_frame:
                self._last_cam_to_base_tf_source = 'missing'

            if warn:
                self.get_logger().warning('TF not ready (%s -> %s): %s' % (source_frame, target_frame, exc))
            return None

    def _try_resolve_base_frame(self):
        """尝试常见 base frame 别名，自动切换到 TF 树中实际存在的那个。"""
        current = self.base_frame
        candidates = [current] + [name for name in self._base_frame_fallbacks if name != current]
        for candidate in candidates:
            transform = self._lookup_transform(candidate, self.camera_frame, warn=False)
            if transform is not None:
                if candidate != self.base_frame:
                    self.get_logger().warning(
                        'Auto-resolved base_frame from %s to %s for TF lookup.' % (self.base_frame, candidate)
                    )
                    self.base_frame = candidate
                return transform
        return None

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

    def _transform_vector(self, vector: np.ndarray, transform_msg):
        """仅对向量施加旋转变换（忽略平移），用于法向量等方向量的坐标系转换。"""
        rotation = Rotation.from_quat(
            [
                transform_msg.transform.rotation.x,
                transform_msg.transform.rotation.y,
                transform_msg.transform.rotation.z,
                transform_msg.transform.rotation.w,
            ]
        )
        return rotation.apply(vector)

    def pixel_to_3d(self, u: int, v: int, depth_image: np.ndarray):
        """将像素坐标反投影为相机坐标系下的 3D 点；深度为零或无效时返回 None。"""
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

    def project_contour_region_to_3d(self, contour: np.ndarray, rgb_image: np.ndarray, depth_image: np.ndarray):
        """将轮廓填充区域内所有像素反投影为带颜色的 3D 点云。"""
        region_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)

        pixel_indices = np.column_stack(np.where(region_mask > 0))
        points_3d = []
        colors_3d = []

        for v, u in pixel_indices:
            point_3d = self.pixel_to_3d(int(u), int(v), depth_image)
            if point_3d is None:
                continue

            points_3d.append(point_3d)
            b, g, r = rgb_image[v, u]
            colors_3d.append([r / 255.0, g / 255.0, b / 255.0])

        if not points_3d:
            return None, None

        return np.asarray(points_3d, dtype=np.float64), np.asarray(colors_3d, dtype=np.float64)

    def compute_plane_normal(self, points_3d: np.ndarray):
        """用 SVD 对点云拟合平面并返回朝向相机的法向量。"""
        if points_3d is None or len(points_3d) < 3:
            return None

        centered = points_3d - np.mean(points_3d, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        if vh.shape[0] < 3:
            return None

        normal = vh[-1]
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            return None
        normal = normal / norm

        if normal[2] > 0:
            normal = -normal

        return normal

    def _build_virtual_point_cloud(self, real_points: np.ndarray):
        """构建与真实点云对齐的虚拟矩形网格点云，用于 ICP 初始化。"""
        if not self.o3d_available or real_points is None or len(real_points) < 10:
            return None, None

        centroid = np.mean(real_points, axis=0)
        coord_x = real_points[:, 0] - centroid[0]
        coord_y = real_points[:, 1] - centroid[1]
        width = float(np.max(coord_x) - np.min(coord_x))
        height = float(np.max(coord_y) - np.min(coord_y))
        if width <= 0.0 or height <= 0.0:
            return None, None

        xs = np.linspace(-width / 2.0, width / 2.0, 30)
        ys = np.linspace(-height / 2.0, height / 2.0, 30)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_z = np.zeros_like(grid_x)
        virtual_local = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)))

        virtual_pcd = o3d.geometry.PointCloud()
        virtual_pcd.points = o3d.utility.Vector3dVector(virtual_local)

        initial_transform = np.eye(4, dtype=np.float64)
        initial_transform[:3, 3] = centroid
        return virtual_pcd, initial_transform

    def _estimate_icp_transform(self, virtual_pcd, real_points: np.ndarray, initial_transform: np.ndarray):
        """用 ICP 算法将虚拟点云配准到真实点云，返回配准变换结果。"""
        if not self.o3d_available or virtual_pcd is None or real_points is None or initial_transform is None:
            return None

        real_pcd = o3d.geometry.PointCloud()
        real_pcd.points = o3d.utility.Vector3dVector(np.asarray(real_points, dtype=np.float64))
        try:
            return o3d.pipelines.registration.registration_icp(
                virtual_pcd,
                real_pcd,
                float(self.icp_threshold),
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
        except Exception:
            return None

    def _extract_valid_contours(self, mask: np.ndarray):
        """从二值掩膜中提取面积达标且为凸四边形的有效轮廓列表。"""
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

        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0.0:
                continue

            rect = cv2.minAreaRect(contour)
            rect_w = float(rect[1][0])
            rect_h = float(rect[1][1])
            rect_area = rect_w * rect_h
            if rect_area <= 1e-6:
                continue
            rectangularity = float(area) / rect_area
            if rectangularity < self.color_min_rectangularity:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            valid_contours.append((contour, approx, float(area)))

        return clean_mask, valid_contours

    def _extract_largest_contour(self, mask: np.ndarray):
        """从掩膜中提取面积最大的有效轮廓，返回清洁掩膜、轮廓、近似多边形和面积。"""
        clean_mask, valid_contours = self._extract_valid_contours(mask)
        if len(valid_contours) == 0:
            return clean_mask, None, None, 0.0

        best_contour, best_approx, best_area = max(valid_contours, key=lambda item: item[2])
        return clean_mask, best_contour, best_approx, best_area

    def _build_color_target_from_contour(
        self,
        color: str,
        contour: np.ndarray,
        contour_area: float,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ):
        """从像素轮廓构造 ColorTarget：反投影中心、顶面点和接触法向量到相机坐标系。"""
        # Prefer the geometric center of the quadrilateral so suction planning
        # is centered on the block footprint instead of contour mass center.
        center_px_float = None
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad_pts = approx.reshape(-1, 2).astype(np.float64)
            center_px_float = np.mean(quad_pts, axis=0)
        else:
            rect = cv2.minAreaRect(contour)
            center_px_float = np.asarray(rect[0], dtype=np.float64)

        cx = int(round(float(center_px_float[0])))
        cy = int(round(float(center_px_float[1])))

        if cx < 0:
            cx = 0
        if cy < 0:
            cy = 0
        if cx >= depth_image.shape[1]:
            cx = depth_image.shape[1] - 1
        if cy >= depth_image.shape[0]:
            cy = depth_image.shape[0] - 1

        points_3d, _ = self.project_contour_region_to_3d(contour, rgb_image, depth_image)
        center_cam_px = self.pixel_to_3d(cx, cy, depth_image)

        center_depth_samples = []
        patch_radius = 2
        for dv in range(-patch_radius, patch_radius + 1):
            for du in range(-patch_radius, patch_radius + 1):
                px = cx + du
                py = cy + dv
                if py < 0 or py >= depth_image.shape[0] or px < 0 or px >= depth_image.shape[1]:
                    continue
                depth_value = depth_image[py, px]
                if np.issubdtype(depth_image.dtype, np.integer):
                    depth_m = float(depth_value) / 1000.0
                else:
                    depth_m = float(depth_value)
                if not np.isfinite(depth_m) or depth_m <= 0.05 or depth_m >= 2.0:
                    continue
                center_depth_samples.append(depth_m)

        if points_3d is None or len(points_3d) == 0:
            if center_cam_px is None:
                return None
            points_3d = np.asarray([center_cam_px], dtype=np.float64)

        valid_mask = np.isfinite(points_3d).all(axis=1)
        valid_mask = np.logical_and(valid_mask, points_3d[:, 2] > 0.05)
        valid_mask = np.logical_and(valid_mask, points_3d[:, 2] < 2.0)
        valid_points = points_3d[valid_mask]
        if len(valid_points) == 0:
            if center_cam_px is None:
                return None
            valid_points = np.asarray([center_cam_px], dtype=np.float64)

        if center_depth_samples:
            center_depth = float(np.median(np.asarray(center_depth_samples, dtype=np.float64)))
            fx = float(self.camera_matrix[0, 0])
            fy = float(self.camera_matrix[1, 1])
            cx0 = float(self.camera_matrix[0, 2])
            cy0 = float(self.camera_matrix[1, 2])
            center_cam = np.array(
                [
                    (float(cx) - cx0) * center_depth / fx,
                    (float(cy) - cy0) * center_depth / fy,
                    center_depth,
                ],
                dtype=np.float64,
            )
        else:
            center_cam = np.median(valid_points, axis=0)

        z_top = float(np.percentile(valid_points[:, 2], 10.0))
        # Keep grasp XY near the contour center to avoid corner-biased suction.
        # Use top-surface z for vertical contact depth.
        top_cam = np.array([float(center_cam[0]), float(center_cam[1]), z_top], dtype=np.float64)

        normal_cam = None
        if self.use_base_icp_suction_logic and self.o3d_available:
            virtual_pcd, initial_transform = self._build_virtual_point_cloud(valid_points)
            if virtual_pcd is not None and initial_transform is not None:
                icp_result = self._estimate_icp_transform(virtual_pcd, valid_points, initial_transform)
                if icp_result is not None:
                    rotation = np.asarray(icp_result.transformation[:3, :3], dtype=np.float64)
                    normal_cam = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    normal_norm = float(np.linalg.norm(normal_cam))
                    if normal_norm > 0.0:
                        normal_cam = normal_cam / normal_norm
                    if normal_cam is not None and normal_cam[2] > 0.0:
                        normal_cam = -normal_cam

        if normal_cam is None:
            normal_cam = self.compute_plane_normal(valid_points)
        if normal_cam is None:
            normal_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        return ColorTarget(
            color=color,
            area=float(contour_area),
            contour=contour,
            center_px=(cx, cy),
            center_cam=np.asarray(center_cam, dtype=np.float64),
            top_cam=top_cam,
            normal_cam=np.asarray(normal_cam, dtype=np.float64),
        )

    def _detect_color_target(
        self,
        color: str,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        preprocessed_hsv: np.ndarray = None,
    ):
        """检测图像中指定颜色的最大目标方块，返回 ColorTarget 和掩膜。"""
        combined_mask = self._build_color_mask(color, rgb_image, preprocessed_hsv=preprocessed_hsv)

        clean_mask, best_contour, best_approx, best_area = self._extract_largest_contour(combined_mask)
        if best_contour is None or best_approx is None or best_area <= 0.0:
            return None, clean_mask

        target = self._build_color_target_from_contour(
            color,
            best_contour,
            float(best_area),
            rgb_image,
            depth_image,
        )
        return target, clean_mask

    def _detect_highest_color_target(
        self,
        color: str,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        max_center_z_for_count: float = None,
        use_compensated_z: bool = False,
        preprocessed_hsv: np.ndarray = None,
    ):
        """检测图像中指定颜色 base z 最高的目标方块，可选过滤超过高度阈值的候选。"""
        combined_mask = self._build_color_mask(color, rgb_image, preprocessed_hsv=preprocessed_hsv)

        clean_mask, valid_contours = self._extract_valid_contours(combined_mask)
        if len(valid_contours) == 0:
            return None, clean_mask

        highest_target = None
        highest_count_z = -np.inf
        for contour, _, area in valid_contours:
            target = self._build_color_target_from_contour(color, contour, area, rgb_image, depth_image)
            if target is None:
                continue

            center_base, _, top_base = self._target_to_base(target)
            raw_count_z = float(center_base[2])
            if top_base is not None and len(top_base) >= 3:
                raw_count_z = max(raw_count_z, float(top_base[2]))

            judge_count_z = raw_count_z
            if use_compensated_z:
                judge_count_z += float(self.detected_center_z_offset)

            # z-threshold filtering is intentionally applied to judge_count_z
            # (compensated when requested), while raw_count_z itself remains unconstrained.
            if max_center_z_for_count is not None and judge_count_z > float(max_center_z_for_count):
                continue
            if judge_count_z > highest_count_z:
                highest_count_z = judge_count_z
                highest_target = target

        return highest_target, clean_mask

    def _compute_count_height_base_z(self, target: ColorTarget):
        """Return base-frame count height using the higher of center-z and top-z."""
        center_base, _, top_base = self._target_to_base(target)
        center_z = float(center_base[2])
        top_z = center_z
        if top_base is not None and len(top_base) >= 3:
            top_z = float(top_base[2])

        raw_count_z = max(center_z, top_z)
        compensated_count_z = raw_count_z + float(self.detected_center_z_offset)
        return center_base, top_base, raw_count_z, compensated_count_z

    def _select_pick_target_from_frame(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        log_reject: bool = False,
        preprocessed_hsv: np.ndarray = None,
    ):
        """在单帧中跨颜色遍历，选出 base z 最高且在工作空间内的方块目标。"""
        best_target = None
        best_mask = None
        best_center_z = -np.inf
        hsv_image = preprocessed_hsv
        if hsv_image is None:
            hsv_image = self._preprocess_rgb_for_color_mask(rgb_image)
        for color in self.color_order:
            if self.color_counts[color] >= self.target_goal_count:
                continue

            target, mask = self._detect_color_target(
                color,
                rgb_image,
                depth_image,
                preprocessed_hsv=hsv_image,
            )
            if target is None:
                continue

            center_base, _, _ = self._target_to_base(target)
            if not self._is_point_in_pick_workspace(center_base):
                if log_reject:
                    self.get_logger().info(
                        'Reject %s candidate outside pick workspace: center_base=%s'
                        % (color, self._format_pose(center_base))
                    )
                continue

            center_z = float(center_base[2])
            if (
                best_target is None
                or center_z > best_center_z
                or (abs(center_z - best_center_z) <= 1e-6 and target.area > best_target.area)
            ):
                best_target = target
                best_mask = mask
                best_center_z = center_z

        return best_target, best_mask

    def _select_pick_target(self):
        """从当前帧中直接选取拾取目标，日志记录被拒绝的候选。"""
        if self.latest_rgb is None or self.latest_depth is None:
            return None

        return self._select_pick_target_from_frame(self.latest_rgb, self.latest_depth, log_reject=True)

    def _select_stable_pick_target(self):
        """连续多帧稳定确认拾取目标（颜色一致、位置和法向量变化在阈值内），超时返回 None。"""
        required = int(self.pick_stability_required_frames)
        timeout_sec = float(self.pick_stability_timeout_sec)
        pos_tol = float(self.pick_stability_position_tol_m)
        normal_tol_rad = float(np.deg2rad(self.pick_stability_normal_tol_deg))

        stable_count = 0
        ref_color = None
        ref_center = None
        ref_normal = None
        latest_target = None

        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout_sec:
            old_stamp = self.latest_stamp
            self._wait_for_frames(old_stamp, timeout_sec=0.35)
            if self.latest_rgb is None or self.latest_depth is None:
                stable_count = 0
                ref_color = None
                ref_center = None
                ref_normal = None
                continue

            pick_result = self._select_pick_target_from_frame(
                self.latest_rgb,
                self.latest_depth,
                log_reject=False,
            )
            target = pick_result[0] if pick_result is not None else None
            if target is None:
                stable_count = 0
                ref_color = None
                ref_center = None
                ref_normal = None
                continue

            center_base, normal_base, _ = self._target_to_base(target)
            normal_norm = float(np.linalg.norm(normal_base))
            if normal_norm <= 1e-8:
                stable_count = 0
                ref_color = None
                ref_center = None
                ref_normal = None
                continue
            normal_base = np.asarray(normal_base, dtype=np.float64) / normal_norm

            if self.pick_stability_require_live_tf and self._last_cam_to_base_tf_source != 'live':
                stable_count = 0
                ref_color = None
                ref_center = None
                ref_normal = None
                continue

            if ref_color is None:
                ref_color = target.color
                ref_center = np.asarray(center_base, dtype=np.float64)
                ref_normal = np.asarray(normal_base, dtype=np.float64)
                stable_count = 1
                latest_target = target
            else:
                pos_delta = float(np.linalg.norm(np.asarray(center_base, dtype=np.float64) - ref_center))
                cos_sim = float(np.clip(np.dot(normal_base, ref_normal), -1.0, 1.0))
                normal_delta = float(np.arccos(cos_sim))
                same_color = target.color == ref_color

                if same_color and pos_delta <= pos_tol and normal_delta <= normal_tol_rad:
                    stable_count += 1
                    ref_center = 0.5 * ref_center + 0.5 * np.asarray(center_base, dtype=np.float64)
                    ref_normal = 0.5 * ref_normal + 0.5 * np.asarray(normal_base, dtype=np.float64)
                    ref_norm = float(np.linalg.norm(ref_normal))
                    if ref_norm > 1e-8:
                        ref_normal = ref_normal / ref_norm
                    latest_target = target
                else:
                    ref_color = target.color
                    ref_center = np.asarray(center_base, dtype=np.float64)
                    ref_normal = np.asarray(normal_base, dtype=np.float64)
                    stable_count = 1
                    latest_target = target

            if stable_count >= required:
                self.get_logger().info(
                    'Stable pick target confirmed: color=%s, frames=%d, tf_source=%s'
                    % (latest_target.color, stable_count, self._last_cam_to_base_tf_source)
                )
                return latest_target

        self.get_logger().info(
            'Pick target not stable within %.2fs (required_frames=%d).' % (timeout_sec, required)
        )
        return None

    def _is_point_in_pick_workspace(self, point_base: np.ndarray) -> bool:
        """判断 base 坐标系下的 3D 点是否在拾取工作空间的包围盒内。"""
        x, y, z = float(point_base[0]), float(point_base[1]), float(point_base[2])
        return (
            self.pick_workspace['x_min'] <= x <= self.pick_workspace['x_max']
            and self.pick_workspace['y_min'] <= y <= self.pick_workspace['y_max']
            and self.pick_workspace['z_min'] <= z <= self.pick_workspace['z_max']
        )

    def _target_to_base(self, target: ColorTarget):
        """将 ColorTarget 的相机坐标系中心/法向/顶面点变换到 base 坐标系；必要时自动修正 XY 翻转。"""
        transform = self._lookup_transform(self.base_frame, self.camera_frame, warn=False)
        if transform is None:
            return target.center_cam, target.normal_cam, target.top_cam

        center_base = self._transform_point(target.center_cam, transform)
        normal_base = self._transform_vector(target.normal_cam, transform)
        normal_norm = np.linalg.norm(normal_base)
        if normal_norm > 0.0:
            normal_base = normal_base / normal_norm

        top_base = self._transform_point(target.top_cam, transform)

        if self.auto_correct_detection_xy_flip:
            center_mirror = np.array([-center_base[0], -center_base[1], center_base[2]], dtype=np.float64)
            if (not self._is_point_in_pick_workspace(center_base)) and self._is_point_in_pick_workspace(center_mirror):
                top_base = np.array([-top_base[0], -top_base[1], top_base[2]], dtype=np.float64)
                normal_base = np.array([-normal_base[0], -normal_base[1], normal_base[2]], dtype=np.float64)
                normal_norm = np.linalg.norm(normal_base)
                if normal_norm > 0.0:
                    normal_base = normal_base / normal_norm
                center_base = center_mirror
                self.get_logger().warning(
                    'Auto-corrected detection frame by XY flip: center_base=%s' % self._format_pose(center_base)
                )

        tf_center_base = self._lookup_pick_center_from_cube_tf()
        if tf_center_base is not None:
            center_base = np.asarray(tf_center_base, dtype=np.float64)
            if top_base is not None:
                top_base = np.asarray(top_base, dtype=np.float64)
                top_base[0] = float(center_base[0])
                top_base[1] = float(center_base[1])

        return center_base, normal_base, top_base

    def _lookup_pick_center_from_cube_tf(self):
        """从 TF 树查询方块中心坐标以覆盖视觉检测结果；TF 不可用或未启用时返回 None。"""
        if not self.use_tf_cube_center_for_pick:
            return None

        transform = self._lookup_transform(self.base_frame, self.pick_cube_frame, warn=False)
        if transform is None:
            now = time.monotonic()
            if now - self._last_cube_tf_warn_time > 2.0:
                self._last_cube_tf_warn_time = now
                self.get_logger().warning(
                    'Pick center TF override enabled but frame %s is unavailable in %s.'
                    % (self.pick_cube_frame, self.base_frame)
                )
            return None

        return np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )

    def _camera_z_axis_in_base(self, source_frame: str):
        """将相机坐标系 z 轴变换到 base 坐标系，用于工具偏航对齐计算。"""
        transform = self._lookup_transform(self.base_frame, source_frame, warn=False)
        if transform is None:
            return None

        axis_base = self._transform_vector(np.array([0.0, 0.0, 1.0], dtype=np.float64), transform)
        axis_norm = float(np.linalg.norm(axis_base))
        if axis_norm <= 1e-9:
            return None
        return axis_base / axis_norm

    def _execute_pick(self, target: ColorTarget):
        """规划并执行拾取动作：接近→渐速下降到抓取位→激活吸盘→提升→返回观测位。"""
        center_base, normal_base, _, approach_pose, grasp_pose, lift_pose = self._plan_pick_poses(target)

        if approach_pose is None or grasp_pose is None or lift_pose is None:
            self.get_logger().error('Failed to build moveL poses for the pick step.')
            return False

        self.get_logger().info(
            'Picking %s block at area %.1f, center_base=%s'
            % (target.color, target.area, np.array2string(np.asarray(center_base), precision=5))
        )

        self.get_logger().info('Approach pose: %s' % self._format_pose(approach_pose))
        self.get_logger().info('Grasp pose: %s' % self._format_pose(grasp_pose))
        self.get_logger().info('Lift pose along suction-normal direction: %s' % self._format_pose(lift_pose))
        self.get_logger().info('Pick TF source: %s' % self._last_cam_to_base_tf_source)

        # Record z-axis rotation from home_pose to approach_pose for later compensation
        self._recorded_z_rotation_rad = self._compute_z_axis_rotation_from_poses(self.home_pose, approach_pose)
        if abs(self._recorded_z_rotation_rad) > 1e-6:
            self.get_logger().info(
                'Recorded z-axis rotation from home to approach: %.4f rad (%.2f deg)'
                % (self._recorded_z_rotation_rad, np.degrees(self._recorded_z_rotation_rad))
            )

        center_hover_pose = [
            float(self.workspace_center[0]),
            float(self.workspace_center[1]),
            float(self.post_pick_center_hover_z),
            float(lift_pose[3]),
            float(lift_pose[4]),
            float(lift_pose[5]),
        ]

        if self.apply_pose_frame_transform:
            transformed_position, _ = self._transform_pose_frame(
                np.asarray(center_hover_pose[:3], dtype=np.float64),
                Rotation.from_rotvec(np.asarray(center_hover_pose[3:6], dtype=np.float64)).as_quat(),
            )
            center_hover_pose[0] = float(transformed_position[0])
            center_hover_pose[1] = float(transformed_position[1])
            center_hover_pose[2] = float(transformed_position[2])

        # Keep center-hover orientation identical to lift orientation to avoid
        # any TCP self-z rotation during the lift->workspace-center segment.
        self.get_logger().info('Post-pick center-hover keeps lift orientation (no TCP self-z spin).')

        self.get_logger().info('Post-pick workspace-center hover pose: %s' % self._format_pose(center_hover_pose))

        if self.alternate_tcp_rotation_during_pick and self.enable_motion and self.rtde_c is not None:
            clockwise = bool(self._next_tcp_rotation_clockwise)
            current_tcp_pose = self._read_actual_tcp_pose()
            spin_pose = self._build_tcp_spin_pose(current_tcp_pose, clockwise)
            if spin_pose is not None:
                rotate_dir = 'clockwise' if clockwise else 'counterclockwise'
                self.get_logger().info(
                    'Applying pre-approach TCP %s rotation (%.1f deg).'
                    % (rotate_dir, self.tcp_rotation_step_deg)
                )
                if not self._move_to_pose(spin_pose):
                    return False
                self._next_tcp_rotation_clockwise = not clockwise


        # Synchronize TCP reorientation with the home->approach motion only when
        # the camera-axis constraint is not already making the pose solve more sensitive.
        # In camera_z mode, keep approach motion at nominal speed for precision.
        if self.pick_camera_face_workspace_center and self.pick_rotation_speed_scale > 1.0 and self.pick_camera_axis_in_tool != 'camera_z':
            fast_speed = self.speed * self.pick_rotation_speed_scale
            fast_acc = self.acceleration * self.pick_rotation_speed_scale
            self.get_logger().info(
                'Fast synchronized rotate+approach (scale=%.2f): %s'
                % (self.pick_rotation_speed_scale, self._format_pose(approach_pose))
            )
            if not self._move_to_pose(approach_pose, speed=fast_speed, acceleration=fast_acc):
                return False
        else:
            if self.pick_camera_axis_in_tool == 'camera_z' and self.pick_rotation_speed_scale > 1.0:
                self.get_logger().info('camera_z alignment active, using nominal approach speed for precision.')
            if not self._move_to_pose(approach_pose):
                return False

        actual_approach_pose = self._read_actual_tcp_pose()
        if actual_approach_pose is not None:
            self.get_logger().info('Actual arrived approach pose: %s' % self._format_pose(actual_approach_pose))
            approach_delta = np.asarray(actual_approach_pose, dtype=np.float64) - np.asarray(
                approach_pose, dtype=np.float64
            )
            self.get_logger().info('Approach delta (actual - planned): %s' % self._format_pose(approach_delta))

        if not self._move_approach_to_grasp_with_progressive_slowdown(approach_pose, grasp_pose):
            return False

        if self.settle_sec > 0.0:
            time.sleep(self.settle_sec)

        # Execute suction and verify by suction-status feedback.
        sucked = self.suction_suck()
        if not sucked:
            return False

        if self.suction_on_delay_sec > 0.0:
            self.get_logger().info('Suction enabled, waiting %.2fs before lift.' % self.suction_on_delay_sec)
            time.sleep(self.suction_on_delay_sec)

        attached = self._read_suction_attached_state()

        if attached is False:
            self.get_logger().error('Suction not attached, abort this pick.')
            return False

        if attached is None:
            self.get_logger().warning('Suction state unknown, continue with current behavior.')

        # Lift phase: use original lift_pose without rotation compensation (pure vertical lift)
        if not self._move_to_pose(lift_pose):
            return False

        # Apply z-axis rotation compensation when moving from lift pose to center hover pose
        # This cancels out the home->approach rotation recorded earlier
        center_hover_pose_to_use = center_hover_pose
        if abs(self._recorded_z_rotation_rad) > 1e-6:
            center_hover_pose_to_use = self._apply_z_rotation_compensation_to_pose(
                center_hover_pose,
                -self._recorded_z_rotation_rad  # Negative for reverse rotation
            )
            self.get_logger().info(
                'Applying z-axis rotation compensation to center_hover_pose: %.4f rad (%.2f deg)'
                % (-self._recorded_z_rotation_rad, np.degrees(-self._recorded_z_rotation_rad))
            )

        if not self._move_to_pose(center_hover_pose_to_use):
            return False

        return True

    def _plan_pick_poses(self, target: ColorTarget):
        """根据 ColorTarget 和表面法向量规划接近、抓取、提升三个 TCP 位姿。"""
        center_base, normal_base, top_base = self._target_to_base(target)

        # Optionally bias the suction planning point along camera_link z axis
        # (expressed in base frame) to compensate systematic center bias.
        camera_z_comp = float(self.pick_camera_z_compensation_m)
        if abs(camera_z_comp) > 1e-9:
            comp_frame = self.pick_camera_z_compensation_frame
            cam_z_axis_base = self._camera_z_axis_in_base(comp_frame)
            if cam_z_axis_base is not None:
                comp_delta = cam_z_axis_base * camera_z_comp
                center_base = np.asarray(center_base, dtype=np.float64) + comp_delta
                if top_base is not None:
                    top_base = np.asarray(top_base, dtype=np.float64) + comp_delta
            else:
                now = time.monotonic()
                if now - self._last_camera_z_comp_warn_time > 2.0:
                    self._last_camera_z_comp_warn_time = now
                    self.get_logger().warning(
                        'pick_camera_z_compensation_m=%.4f requested but TF %s -> %s unavailable, skip compensation.'
                        % (camera_z_comp, comp_frame, self.base_frame)
                    )

        if self.use_base_icp_suction_logic:
            # Keep suction-direction convention consistent with the original
            # pick pipeline: suction_dir points from tool toward the block.
            suction_dir = -np.asarray(normal_base, dtype=np.float64)
            suction_norm = float(np.linalg.norm(suction_dir))
            if suction_norm > 0.0:
                suction_dir = suction_dir / suction_norm

            pick_base = np.asarray(center_base, dtype=np.float64)
            approach_point = pick_base - suction_dir * float(self.approach_offset)
            grasp_point = pick_base - suction_dir * float(self.grasp_offset)
            lift_point = grasp_point - suction_dir * float(self.lift_offset)

            look_at_point = self.workspace_center if self.pick_camera_face_workspace_center else None

            approach_pose = self._build_tool_pose(
                approach_point,
                suction_dir,
                look_at_point_base=look_at_point,
                camera_axis_in_tool=self.pick_camera_axis_in_tool,
            )
            grasp_pose = self._build_tool_pose(
                grasp_point,
                suction_dir,
                look_at_point_base=look_at_point,
                camera_axis_in_tool=self.pick_camera_axis_in_tool,
            )
            lift_pose = self._build_tool_pose(
                lift_point,
                suction_dir,
                look_at_point_base=look_at_point,
                camera_axis_in_tool=self.pick_camera_axis_in_tool,
            )

            return center_base, normal_base, top_base, approach_pose, grasp_pose, lift_pose

        # In some camera/robot frames the computed surface normal points opposite
        # to the desired suction direction. Use the inverted normal so the TCP
        # points toward the block (suction facing the object).
        suction_dir = -normal_base

        # Use the top-surface center as the pick reference so the final suction
        # point stays closer to the visually observed block center.
        pick_base = np.asarray(top_base if top_base is not None else center_base, dtype=np.float64)
        approach_point = pick_base - suction_dir * self.approach_offset
        # Keep the grasp descent short and independent from approach height.
        # This avoids excessive drop and protective stops when approach_offset is large.
        grasp_descent = min(float(self.grasp_descent_offset), max(0.0, float(self.approach_offset)))
        grasp_point = np.asarray(approach_point, dtype=np.float64) + np.asarray(suction_dir, dtype=np.float64) * grasp_descent
        lift_point = np.asarray(grasp_point, dtype=np.float64) + np.asarray(normal_base, dtype=np.float64) * float(
            self.lift_offset
        )

        look_at_point = self.workspace_center if self.pick_camera_face_workspace_center else None
        approach_pose = self._build_tool_pose(
            approach_point,
            suction_dir,
            look_at_point_base=look_at_point,
            camera_axis_in_tool=self.pick_camera_axis_in_tool,
        )
        grasp_pose = self._build_tool_pose(
            grasp_point,
            suction_dir,
            look_at_point_base=look_at_point,
            camera_axis_in_tool=self.pick_camera_axis_in_tool,
        )
        lift_pose = self._build_tool_pose(
            lift_point,
            suction_dir,
            look_at_point_base=look_at_point,
            camera_axis_in_tool=self.pick_camera_axis_in_tool,
        )

        # Keep dynamic pick poses consistent with the existing global frame
        # transform setting used by configured runtime poses.
        if self.apply_pose_frame_transform:
            approach_pose = self._transform_rotvec_pose_frame(approach_pose)
            grasp_pose = self._transform_rotvec_pose_frame(grasp_pose)
            lift_pose = self._transform_rotvec_pose_frame(lift_pose)

        return center_base, normal_base, top_base, approach_pose, grasp_pose, lift_pose

    def _estimate_place_drop_z(self, color: str, hover_pose: list[float]):
        """根据当前颜色已放置数量估算下一次放置的目标 z 高度（米）。"""
        next_index = self.color_counts[color]
        if next_index < len(self.place_drop_heights):
            return float(self.place_drop_heights[next_index])

        # Safety fallback for counts beyond configured list length.
        last_height = float(self.place_drop_heights[-1])
        extra_levels = next_index - (len(self.place_drop_heights) - 1)
        return float(last_height + extra_levels * self.stack_height_step)

    def _estimate_count_from_place_height(self, center_z: float):
        """根据 base z 高度经验区间估算已放置方块数量（最小和最大猜测值）。"""
        z = float(center_z)
        if 0.01 <= z < 0.11:
            return 1, 1
        if 0.11 <= z < 0.19:
            return 2, 2
        if 0.19 <= z <= 0.28:
            return 3, 3
        return None, None

    def _estimate_count_from_z_cam_hover(self, z_cam: float):
        """根据悬停位深度相机 z 值估算放置堆叠上的方块数量（最小和最大猜测值）。"""
        if z_cam is None:
            return 0, 0

        z = float(z_cam)
        if z > 0.3:
            return 1, 1
        if 0.2 < z < 0.3:
            return 2, 2
        if 0.1 < z < 0.2:
            return 3, 3
        return None, None

    def _read_center_depth_cam_m_from_contour(self, contour: np.ndarray, depth_image: np.ndarray):
        """从深度图读取轮廓中心小邻域内的中位深度（米），用于推算相机 z 坐标。"""
        if depth_image is None or contour is None:
            return None

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

        center_depth_samples = []
        patch_radius = 2
        for dv in range(-patch_radius, patch_radius + 1):
            for du in range(-patch_radius, patch_radius + 1):
                px = cx + du
                py = cy + dv
                if py < 0 or py >= depth_image.shape[0] or px < 0 or px >= depth_image.shape[1]:
                    continue

                depth_value = depth_image[py, px]
                if np.issubdtype(depth_image.dtype, np.integer):
                    depth_m = float(depth_value) / 1000.0
                else:
                    depth_m = float(depth_value)

                if not np.isfinite(depth_m) or depth_m <= 0.05 or depth_m >= 2.0:
                    continue
                center_depth_samples.append(depth_m)

        if not center_depth_samples:
            return None
        return float(np.median(np.asarray(center_depth_samples, dtype=np.float64)))

    def _select_hover_target_with_valid_z_cam(self, color: str, rgb_image: np.ndarray, depth_image: np.ndarray):
        """从悬停视角检测指定颜色方块，选出 base z 最高且深度读数有效的目标。"""
        hsv_image = self._preprocess_rgb_for_color_mask(rgb_image)
        combined_mask = self._build_color_mask(color, rgb_image, preprocessed_hsv=hsv_image)
        clean_mask, valid_contours = self._extract_valid_contours(combined_mask)
        if len(valid_contours) == 0:
            return None, None, 'no_target', 0

        candidates = []
        for contour, _, area in valid_contours:
            target = self._build_color_target_from_contour(color, contour, area, rgb_image, depth_image)
            if target is None:
                continue
            center_base, _, _ = self._target_to_base(target)
            candidates.append((float(center_base[2]), target))

        if len(candidates) == 0:
            return None, None, 'no_valid_z_cam', 0

        # Evaluate from top to bottom; if the top blob has invalid z_cam,
        # fallback to the highest valid blob below it.
        candidates.sort(key=lambda item: item[0], reverse=True)
        skipped_invalid_top = 0
        for _, target in candidates:
            z_cam = self._read_center_depth_cam_m_from_contour(target.contour, depth_image)
            if z_cam is not None:
                return target, z_cam, 'ok', skipped_invalid_top
            skipped_invalid_top += 1

        return None, None, 'no_valid_z_cam', skipped_invalid_top

    def _reconcile_color_count_from_place_view_hover(self, color: str):
        """在放置悬停位观测深度图，根据相机 z 推断堆叠层数并校正计数记录。"""
        old_stamp = self.latest_stamp
        self._wait_for_frames(old_stamp, timeout_sec=1.0)

        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warning('Hover-view reconciliation skipped: no synchronized frames.')
            return

        # Hover-phase layer judgment uses camera-frame depth (z_cam) only.
        target, z_cam, hover_status, skipped_invalid_top = self._select_hover_target_with_valid_z_cam(
            color,
            self.latest_rgb.copy(),
            self.latest_depth,
        )

        if hover_status == 'no_target':
            inferred_count, inferred_layer = 0, 0
            z_cam = None
        elif hover_status == 'no_valid_z_cam':
            self.get_logger().warning(
                'Hover z_cam unavailable for %s: detected blob(s) have invalid center depth patch. '
                'Ignore this color for this reconciliation.'
                % color
            )
            return
        else:
            inferred_count, inferred_layer = self._estimate_count_from_z_cam_hover(z_cam)
            if skipped_invalid_top > 0:
                self.get_logger().info(
                    'Hover z_cam fallback for %s: skipped %d higher blob(s) with invalid z_cam, '
                    'using the highest valid lower blob.'
                    % (color, skipped_invalid_top)
                )

        if inferred_count is None:
            self.get_logger().warning(
                'Hover-view reconciliation skipped for %s: z_cam=%.4f is outside configured ranges '
                '(z_cam>0.3->1, 0.2<z_cam<0.3->2, 0.1<z_cam<0.2->3).'
                % (color, float(z_cam))
            )
            return

        recorded_count = self.color_counts[color]
        if inferred_count != recorded_count:
            if inferred_count > recorded_count:
                self.color_counts[color] = inferred_count
                action_text = 'Updated record.'
            else:
                action_text = 'Keep recorded count.'
            z_cam_text = 'n/a' if z_cam is None else ('%.4f' % float(z_cam))
            self.get_logger().error(
                'Hover z_cam count mismatch for %s: recorded=%d, vision=%d (layer=%d, z_cam=%s). %s'
                % (color, recorded_count, inferred_count, inferred_layer, z_cam_text, action_text)
            )
        else:
            z_cam_text = 'n/a' if z_cam is None else ('%.4f' % float(z_cam))
            self.get_logger().info(
                'Hover z_cam count check passed for %s: recorded=%d, vision=%d (layer=%d, z_cam=%s).'
                % (color, recorded_count, inferred_count, inferred_layer, z_cam_text)
            )

    def _reconcile_color_count_from_place_view(self, color: str):
        """放置完成后在固定俯视位观测，用视觉检测结果核验并修正颜色方块计数。"""
        old_stamp = self.latest_stamp
        if not self._wait_for_strict_new_frame(old_stamp, timeout_sec=1.0):
            self.get_logger().warning(
                'Place-view reconciliation skipped for %s: no strict new frame at completion pose.' % color
            )
            return

        if self.latest_rgb is None or self.latest_depth is None:
            self.get_logger().warning('Place-view reconciliation skipped: no synchronized frames.')
            return

        target, _ = self._detect_highest_color_target(
            color,
            self.latest_rgb.copy(),
            self.latest_depth,
            max_center_z_for_count=0.20,
            use_compensated_z=True,
        )
        if target is None:
            recorded_count = self.color_counts[color]
            if recorded_count != 0:
                self.get_logger().error(
                    'Count mismatch detected for %s: recorded=%d, vision=0 (no block detected). Keep recorded count.'
                    % (color, recorded_count)
                )
            else:
                self.get_logger().info(
                    'Count check passed for %s: recorded=0, vision=0 (no block detected).' % color
                )
            return

        center_base, top_base, raw_count_z, compensated_count_z = self._compute_count_height_base_z(target)
        inferred_count, inferred_layer = self._estimate_count_from_place_height(compensated_count_z)
        if inferred_count is None:
            self.get_logger().warning(
                'Place-view reconciliation skipped: %s highest count z=%.4f (raw=%.4f) is outside configured ranges '
                '[0.01,0.10)->1, [0.10,0.19)->2, [0.19,0.28]->3'
                % (color, compensated_count_z, raw_count_z)
            )
            return

        recorded_count = self.color_counts[color]
        if inferred_count != recorded_count:
            if inferred_count > recorded_count:
                self.color_counts[color] = inferred_count
                action_text = 'Updated record.'
            else:
                action_text = 'Keep recorded count.'
            self.get_logger().error(
                'Count mismatch detected for %s: recorded=%d, vision=%d (layer=%d, '
                'count_z=%.4f, raw_count_z=%.4f, center_z=%.4f, top_z=%.4f). %s'
                % (
                    color,
                    recorded_count,
                    inferred_count,
                    inferred_layer,
                    compensated_count_z,
                    raw_count_z,
                    float(center_base[2]),
                    float(top_base[2]) if top_base is not None and len(top_base) >= 3 else float(center_base[2]),
                    action_text,
                )
            )
        else:
            self.get_logger().info(
                'Count check passed for %s: recorded=%d, vision=%d (layer=%d, '
                'count_z=%.4f, raw_count_z=%.4f, center_z=%.4f, top_z=%.4f).'
                % (
                    color,
                    recorded_count,
                    inferred_count,
                    inferred_layer,
                    compensated_count_z,
                    raw_count_z,
                    float(center_base[2]),
                    float(top_base[2]) if top_base is not None and len(top_base) >= 3 else float(center_base[2]),
                )
            )

    def _execute_place(self, color: str, hover_pose: list[float] = None):
        """执行放置动作：悬停观测→移动→下降到放置高度→释放吸盘→短提升→返回悬停位。"""
        target_pose = list(self.place_poses[color])
        mid_pose = list(self.place_mid_poses[color])
        target_position = self._pose_position(target_pose)
        target_rotvec = np.asarray(target_pose[3:6], dtype=np.float64)

        if hover_pose is None:
            hover_pose = list(self.place_hover_poses[color])

        if not self._move_to_pose(hover_pose):
            return False

        if self.place_hover_observe_sec > 0.0:
            self.get_logger().info(
                'Waiting %.2fs at hover pose for stack-layer observation.' % self.place_hover_observe_sec
            )
            time.sleep(self.place_hover_observe_sec)

        # At place hover, infer current stack layer by z_cam and reconcile count.
        self._set_camera_frame_capture_enabled(True, clear_cache=True, reason='place hover observation')
        self._reconcile_color_count_from_place_view_hover(color)
        self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='leave place hover observation')

        if not self._move_to_pose(mid_pose):
            return False

        drop_z = self._estimate_place_drop_z(color, hover_pose)
        drop_pose = self._build_pose_with_rotvec(
            np.array([target_position[0], target_position[1], drop_z], dtype=np.float64),
            target_rotvec,
        )

        self.get_logger().info(
            'Placing %s block #%d at z=%.4f' % (color, self.color_counts[color] + 1, drop_pose[2])
        )

        if not self._move_to_pose(drop_pose):
            return False

        # Count this placement as soon as the robot reaches drop pose.
        # This keeps recorded counts aligned with the physical placement step.
        self.color_counts[color] += 1
        self.get_logger().info('Recorded %s count updated to %d after reaching drop pose.' % (color, self.color_counts[color]))

        if self.settle_sec > 0.0:
            time.sleep(self.settle_sec)

        released = self.suction_release()
        if not released:
            self.get_logger().warning('Release command failed; continue workflow for count/finish handling.')

        if self.release_settle_sec > 0.0:
            time.sleep(self.release_settle_sec)

        short_lift_z = min(
            float(hover_pose[2]),
            float(drop_pose[2]) + max(float(self.place_clearance), 0.01),
        )
        short_lift_pose = self._build_pose_with_rotvec(
            np.array([target_position[0], target_position[1], short_lift_z], dtype=np.float64),
            target_rotvec,
        )
        short_lift_speed = max(0.01, float(self.speed) * 0.35)
        short_lift_acc = max(0.01, float(self.acceleration) * 0.35)
        self.get_logger().info(
            'Post-place short lift for %s: z %.4f -> %.4f at speed %.3f'
            % (color, float(drop_pose[2]), short_lift_z, short_lift_speed)
        )

        if not self._move_to_pose(short_lift_pose, speed=short_lift_speed, acceleration=short_lift_acc):
            return False

        if not self._move_to_pose(hover_pose):
            return False

        return True

    def _all_colors_finished(self):
        """检查所有颜色的已放置数量是否均已达到目标数量。"""
        return all(self.color_counts[color] >= self.target_goal_count for color in self.color_order)

    def _final_verify_all_colors_with_compensation(self) -> bool:
        """Final completion check at completion_observe_pose using compensated center z."""
        all_ok = True
        for color in self.color_order:
            self._set_camera_frame_capture_enabled(True, clear_cache=False, reason='final completion verification')
            old_stamp = self.latest_stamp
            if not self._wait_for_strict_new_frame(old_stamp, timeout_sec=1.0):
                self.get_logger().warning(
                    'Final verify skipped for %s: no strict new frame at completion pose.' % color
                )
                all_ok = False
                continue

            if self.latest_rgb is None or self.latest_depth is None:
                self.get_logger().warning('Final verify skipped for %s: no synchronized frames.' % color)
                all_ok = False
                continue

            preprocessed_hsv = self._preprocess_rgb_for_color_mask(self.latest_rgb)

            target, _ = self._detect_highest_color_target(
                color,
                self.latest_rgb,
                self.latest_depth,
                max_center_z_for_count=None,
                use_compensated_z=False,
                preprocessed_hsv=preprocessed_hsv,
            )
            if target is None:
                self.color_counts[color] = 0
                self.get_logger().warning('Final verify %s: no block detected, count set to 0.' % color)
                all_ok = False
                continue

            center_base, top_base, raw_count_z, compensated_count_z = self._compute_count_height_base_z(target)
            inferred_count, inferred_layer = self._estimate_count_from_place_height(compensated_count_z)
            if inferred_count is None:
                self.get_logger().warning(
                    'Final verify %s: count_z=%.4f (raw=%.4f) out of count range.'
                    % (color, compensated_count_z, raw_count_z)
                )
                all_ok = False
                continue

            if inferred_count > self.color_counts[color]:
                self.color_counts[color] = inferred_count

            # Vision shows fewer layers than target: actual stack is incomplete.
            # This is the authoritative ground truth — override and fail.
            if inferred_count < self.target_goal_count:
                if self.color_counts[color] >= self.target_goal_count:
                    self.get_logger().error(
                        'Final verify %s: vision shows layer=%d (count=%d) but recorded=%d. '
                        'Stack not complete, marking as failed.'
                        % (color, inferred_layer, inferred_count, self.color_counts[color])
                    )
                    self.color_counts[color] = inferred_count
                all_ok = False
            elif self.color_counts[color] < self.target_goal_count:
                all_ok = False

            self.get_logger().info(
                'Final verify %s: layer=%d, count=%d, count_z=%.4f, raw_count_z=%.4f, center_z=%.4f, top_z=%.4f'
                % (
                    color,
                    inferred_layer,
                    self.color_counts[color],
                    compensated_count_z,
                    raw_count_z,
                    float(center_base[2]),
                    float(top_base[2]) if top_base is not None and len(top_base) >= 3 else float(center_base[2]),
                )
            )

        self._set_camera_frame_capture_enabled(False, clear_cache=False, reason='final completion verification finished')
        return all_ok and self._all_colors_finished()

    def _finish_task(self):
        """移动到完成观测位，进行最终视觉核验；全部颜色通过后标记任务完成。"""
        self.get_logger().info(
            'Recorded counts are full. Moving to completion_observe_pose for visual verification...'
        )
        self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='moving to completion verification pose')
        if not self._move_to_pose(self.completion_observe_pose):
            self.get_logger().warning('Failed to move to completion_observe_pose. Will retry next cycle.')
            self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='completion verification aborted')
            return False

        self.get_logger().info(
            'Arrived at completion_observe_pose: %s' % self._format_pose(self.completion_observe_pose)
        )

        if self.clear_camera_cache_before_observe:
            self._clear_latest_camera_frames('before completion_observe_pose verification')

        completion_stamp_before_wait = self.latest_stamp

        if self.completion_observe_settle_sec > 0.0:
            self.get_logger().info(
                'Waiting %.2fs at completion_observe_pose before height verification...'
                % self.completion_observe_settle_sec
            )
            time.sleep(self.completion_observe_settle_sec)

        self._set_camera_frame_capture_enabled(True, clear_cache=True, reason='completion verification after arrival settle')

        completion_wait_timeout = max(
            float(self.completion_observe_wait_timeout_sec),
            float(self.completion_observe_settle_sec) + 0.8,
        )
        if not self._wait_for_multiple_fresh_frames(
            completion_stamp_before_wait,
            required_frames=self.completion_observe_required_new_frames,
            timeout_sec=completion_wait_timeout,
        ):
            self.get_logger().warning(
                'Completion verification postponed: need %d fresh frames after arriving at completion_observe_pose '
                '(timeout=%.2fs).'
                % (self.completion_observe_required_new_frames, completion_wait_timeout)
            )
            self._set_camera_frame_capture_enabled(False, clear_cache=False, reason='completion verification postponed')
            return False

        # Reconcile each color count by the same height-based logic used previously.
        for color in self.color_order:
            self._reconcile_color_count_from_place_view(color)

        if self._final_verify_all_colors_with_compensation():
            self.finished = True
            self.get_logger().info('Pick and place task 1 completed successfully! ')
            self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='task finished')
            return True

        self.get_logger().warning(
            'Completion verification failed. Updated counts: green=%d, red=%d, yellow=%d. Continue picking.'
            % (self.color_counts['green'], self.color_counts['red'], self.color_counts['yellow'])
        )
        self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='completion verification failed')
        self._awaiting_home_refresh = False
        return False

    def control_loop(self):
        """定时器回调：任务完成时关闭节点，否则在后台线程中执行一次控制循环。"""
        if self.finished:
            # Task completed - shutdown ROS gracefully
            if self.timer is not None:
                self.destroy_timer(self.timer)
                self.timer = None
            # Schedule shutdown on next ROS event loop iteration to avoid context issues
            def shutdown_delayed():
                time.sleep(0.1)
                if rclpy.ok():
                    try:
                        rclpy.shutdown()
                    except Exception as e:
                        self.get_logger().warning('Shutdown exception: %s' % str(e))
            threading.Thread(target=shutdown_delayed, daemon=True).start()
            return

        with self._busy_lock:
            if self.busy:
                return
            self.busy = True

        threading.Thread(target=self._run_cycle_once, daemon=True).start()

    def _run_cycle_once(self):
        """单次控制循环：等待帧→选取最高目标→执行拾取→移动到放置位→执行放置→核验计数。"""
        try:
            if self._all_colors_finished():
                self._finish_task()
                return

            if self.camera_matrix is None:
                self.get_logger().info('Waiting for camera info...')
                return

            if self._camera_frame_capture_enabled and (
                self.latest_rgb is None or self.latest_depth is None
            ):
                self.get_logger().info('Waiting for synchronized RGB/depth and camera info...')
                return

            transform = self._lookup_transform(self.base_frame, self.camera_frame, warn=False)
            if transform is None:
                transform = self._try_resolve_base_frame()
                if transform is not None:
                    self.get_logger().info('TF recovered after base_frame auto-resolution: %s' % self.base_frame)
                    return

                now = time.monotonic()
                if now - self._startup_time < self.tf_startup_grace_sec:
                    return
                if now - self._last_tf_missing_warn_time > 2.0:
                    self._last_tf_missing_warn_time = now
                    self.get_logger().warning(
                        'TF tree disconnected (%s -> %s). Waiting for TF before target selection.'
                        % (self.camera_frame, self.base_frame)
                    )
                return

            if self._awaiting_home_refresh:
                self._set_camera_frame_capture_enabled(True, clear_cache=False, reason='home_pose observation window')
                if self.clear_camera_cache_before_observe and self._home_stamp_before_refresh is None:
                    self._clear_latest_camera_frames('before home_pose observation refresh')
                    self._home_stamp_before_refresh = 'cache_cleared'

                if self.latest_rgb is None or self.latest_depth is None or self.latest_stamp is None:
                    self.get_logger().info('Waiting for first camera frame after home_pose...')
                    return

                if self.home_observe_settle_sec > 0.0:
                    self.get_logger().info(
                        'Waiting %.2fs at home_pose for camera observation stabilization...'
                        % self.home_observe_settle_sec
                    )
                    time.sleep(self.home_observe_settle_sec)

                old_stamp = self.latest_stamp
                required_fresh = int(self.home_observe_required_new_frames)
                wait_timeout = max(
                    float(self.home_observe_wait_timeout_sec),
                    float(self.home_observe_settle_sec) + 0.4,
                )
                got_fresh = self._wait_for_multiple_fresh_frames(
                    old_stamp,
                    required_frames=required_fresh,
                    timeout_sec=wait_timeout,
                )
                if not got_fresh:
                    self.get_logger().warning(
                        'home_pose frame stabilization not ready: required %d new frames within %.2fs, keep waiting.'
                        % (required_fresh, wait_timeout)
                    )
                    return

                self._awaiting_home_refresh = False
                self._home_stamp_before_refresh = None
                self._startup_time = time.monotonic()
                self.get_logger().info('Camera frame refreshed after home_pose, start target selection.')
            else:
                self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='moving to home_pose')
                if not self._move_to_pose(self.home_pose):
                    return
                self.get_logger().info('Arrived at home_pose: %s' % self._format_pose(self.home_pose))
                self._awaiting_home_refresh = True
                self._home_stamp_before_refresh = None
                self._set_camera_frame_capture_enabled(True, clear_cache=True, reason='arrived at home_pose, waiting for fresh frames')
                return

            target = self._select_stable_pick_target()
            if target is None:
                self.get_logger().info('No stable green/red/yellow pick target found in current observation window.')
                return

            self._set_camera_frame_capture_enabled(False, clear_cache=True, reason='stable target selected, entering motion phase')

            center_base, normal_base, top_base, approach_pose, grasp_pose, lift_pose = self._plan_pick_poses(target)
            self.get_logger().info(
                'Detected %s max block: area=%.1f, center_base=%s, top_base=%s, normal_base=%s'
                % (
                    target.color,
                    target.area,
                    self._format_pose(center_base),
                    self._format_pose(top_base),
                    self._format_pose(normal_base),
                )
            )
            self.get_logger().info('Planned approach_pose: %s' % self._format_pose(approach_pose))
            self.get_logger().info('Planned grasp_pose: %s' % self._format_pose(grasp_pose))
            self.get_logger().info('Planned lift_pose: %s' % self._format_pose(lift_pose))

            if not self._execute_pick(target):
                return

            # Predict and log the next place poses (hover and drop) before executing place
            target_pose = list(self.place_poses[target.color])
            target_position = self._pose_position(target_pose)
            target_rotvec = np.asarray(target_pose[3:6], dtype=np.float64)

            # Use the lift_pose z (current gripper height after pick) as hover height
            hover_pose = list(self.place_hover_poses[target.color])
            predicted_drop_z = self._estimate_place_drop_z(target.color, hover_pose)
            drop_pose = self._build_pose_with_rotvec(
                np.array([target_position[0], target_position[1], predicted_drop_z], dtype=np.float64),
                target_rotvec,
            )

            self.get_logger().info('Predicted place hover_pose: %s' % self._format_pose(hover_pose))
            self.get_logger().info('Predicted place drop_pose: %s' % self._format_pose(drop_pose))

            if not self._execute_place(target.color, hover_pose=hover_pose):
                return

            self.get_logger().info(
                'Updated counts: green=%d, red=%d, yellow=%d'
                % (self.color_counts['green'], self.color_counts['red'], self.color_counts['yellow'])
            )

            if self._all_colors_finished():
                self._finish_task()
                return
        finally:
            with self._busy_lock:
                self.busy = False

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
    # 使用紧凑日志格式，避免 ROS2 默认的冗长行首前缀
    os.environ.setdefault('RCUTILS_CONSOLE_OUTPUT_FORMAT', '{message}')
    rclpy.init(args=args)
    node = CubePickPlaceController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()