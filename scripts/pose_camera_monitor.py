#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterUninitializedException
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

rtde_control = None


class PoseCameraMonitor(Node):
    def __init__(self):
        super().__init__('pose_camera_monitor')

        self.declare_parameter('mode', 'monitor')
        self.declare_parameter('robot_ip', '192.168.56.3')
        self.declare_parameter('move_speed', 0.10)
        self.declare_parameter('move_acceleration', 0.20)
        self.declare_parameter('target_pose', Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter('target_pose_runtime', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter('runtime_pose_xy_flip', True)
        self.declare_parameter('exit_after_move', True)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('tool_frame', 'wrist_3_link')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('pose_name', 'home_pose')
        self.declare_parameter('print_period', 1.0)
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'Pose Camera Monitor')
        self.declare_parameter('apply_pose_frame_transform', True)
        self.declare_parameter('color_min_area', 300.0)

        self.mode = self.get_parameter('mode').get_parameter_value().string_value.strip().lower()
        self.robot_ip = self.get_parameter('robot_ip').get_parameter_value().string_value
        self.move_speed = float(self.get_parameter('move_speed').get_parameter_value().double_value)
        self.move_acceleration = float(self.get_parameter('move_acceleration').get_parameter_value().double_value)
        try:
            target_pose_param = self.get_parameter('target_pose')
            if target_pose_param.type_ == Parameter.Type.NOT_SET:
                self.target_pose = []
            else:
                self.target_pose = [float(v) for v in target_pose_param.get_parameter_value().double_array_value]
        except ParameterUninitializedException:
            self.target_pose = []
        self.target_pose_runtime = [
            float(v) for v in self.get_parameter('target_pose_runtime').get_parameter_value().double_array_value
        ]
        self.runtime_pose_xy_flip = bool(
            self.get_parameter('runtime_pose_xy_flip').get_parameter_value().bool_value
        )
        self.exit_after_move = bool(self.get_parameter('exit_after_move').get_parameter_value().bool_value)
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.tool_frame = self.get_parameter('tool_frame').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.pose_name = self.get_parameter('pose_name').get_parameter_value().string_value
        self.print_period = float(self.get_parameter('print_period').get_parameter_value().double_value)
        self.show_window = bool(self.get_parameter('show_window').get_parameter_value().bool_value)
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.apply_pose_frame_transform = bool(
            self.get_parameter('apply_pose_frame_transform').get_parameter_value().bool_value
        )
        self.color_min_area = float(self.get_parameter('color_min_area').get_parameter_value().double_value)

        self.color_ranges = {
            'green': [
                (np.array([35, 70, 60], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
            ],
            'yellow': [
                (np.array([20, 100, 80], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
            ],
            'red': [
                (np.array([0, 100, 80], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
                (np.array([160, 100, 80], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
            ],
        }
        self.color_order = ('green', 'red', 'yellow')

        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_matrix = None
        self.latest_camera_frame = None
        self.latest_depth_frame = None
        self._debug_window_failed = False
        self.rtde_c = None
        self._move_done = False
        self._move_timer = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.mode == 'monitor':
            self.camera_info_sub = self.create_subscription(
                CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
            )
            self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
            self.depth_sub = Subscriber(self, Image, self.depth_topic)
            self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
            self.ts.registerCallback(self.image_callback)
            self.pose_timer = self.create_timer(max(0.1, self.print_period), self.print_pose_block)
            self.get_logger().info(
                'Pose camera monitor started: base_frame=%s, tool_frame=%s, pose_name=%s'
                % (self.base_frame, self.tool_frame, self.pose_name)
            )
        elif self.mode == 'move':
            self.get_logger().info('Pose camera monitor started in move mode.')
            self._move_timer = self.create_timer(0.1, self._run_move_once)
        else:
            self.get_logger().error("Invalid mode=%s, supported: 'monitor' or 'move'." % self.mode)
            raise ValueError('Invalid mode parameter')

    def _init_robot_control(self):
        global rtde_control
        if rtde_control is None:
            try:
                import importlib

                rtde_control = importlib.import_module('rtde_control')
            except Exception as exc:
                self.get_logger().error('Cannot import rtde_control: %s' % exc)
                return False

        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.get_logger().info('Connected RTDE control to %s' % self.robot_ip)
            return True
        except Exception as exc:
            self.get_logger().error('Failed to connect RTDE control: %s' % exc)
            return False

    def _pose_to_movel(self, pose_values, apply_xy_flip: bool):
        if len(pose_values) == 6:
            position = np.asarray(pose_values[:3], dtype=np.float64)
            rotvec = np.asarray(pose_values[3:6], dtype=np.float64)
            quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()

            if apply_xy_flip:
                position, quat_xyzw = self._inverse_transform_pose_frame(position, quat_xyzw)

            quat_norm = np.linalg.norm(quat_xyzw)
            if quat_norm <= 1e-9:
                return None

            quat_xyzw = quat_xyzw / quat_norm
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
            return [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(rotvec[0]),
                float(rotvec[1]),
                float(rotvec[2]),
            ]

        if len(pose_values) == 7:
            position = np.asarray(pose_values[:3], dtype=np.float64)
            quat_xyzw = np.asarray(pose_values[3:7], dtype=np.float64)

            if apply_xy_flip:
                position, quat_xyzw = self._inverse_transform_pose_frame(position, quat_xyzw)

            quat_norm = np.linalg.norm(quat_xyzw)
            if quat_norm <= 1e-9:
                return None

            quat_xyzw = quat_xyzw / quat_norm
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
            return [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(rotvec[0]),
                float(rotvec[1]),
                float(rotvec[2]),
            ]

        return None

    def _runtime_pose_to_movel(self, runtime_pose):
        return self._pose_to_movel(runtime_pose, apply_xy_flip=self.runtime_pose_xy_flip)

    def _target_pose_to_movel(self, target_pose):
        return self._pose_to_movel(target_pose, apply_xy_flip=False)

    def _run_move_once(self):
        if self._move_done:
            return

        self._move_done = True
        if self._move_timer is not None:
            self._move_timer.cancel()

        use_target_pose = len(self.target_pose) > 0
        if use_target_pose:
            if len(self.target_pose) not in (6, 7):
                self.get_logger().error(
                    'target_pose must be 6 values [x,y,z,rx,ry,rz] or 7 values [x,y,z,qx,qy,qz,qw], got %d.'
                    % len(self.target_pose)
                )
                if self.exit_after_move:
                    rclpy.shutdown()
                return
            move_pose = self._target_pose_to_movel(self.target_pose)
        else:
            if len(self.target_pose_runtime) not in (6, 7):
                self.get_logger().error(
                    'target_pose_runtime must be 6 values [x,y,z,rx,ry,rz] or 7 values [x,y,z,qx,qy,qz,qw], got %d.'
                    % len(self.target_pose_runtime)
                )
                if self.exit_after_move:
                    rclpy.shutdown()
                return
            move_pose = self._runtime_pose_to_movel(self.target_pose_runtime)

        if move_pose is None:
            if use_target_pose:
                self.get_logger().error('Failed to convert target_pose to moveL pose.')
            else:
                self.get_logger().error('Failed to convert target_pose_runtime to moveL pose.')
            if self.exit_after_move:
                rclpy.shutdown()
            return

        if use_target_pose:
            self.get_logger().info(
                'Move target(target_pose %dD): %s'
                % (
                    len(self.target_pose),
                    np.array2string(np.asarray(self.target_pose, dtype=np.float64), precision=6, separator=', '),
                )
            )
            self.get_logger().info('target_pose is used without runtime x/y sign flip.')
        else:
            self.get_logger().info(
                'Move target(runtime %dD): %s'
                % (
                    len(self.target_pose_runtime),
                    np.array2string(np.asarray(self.target_pose_runtime, dtype=np.float64), precision=6, separator=', '),
                )
            )
            if self.runtime_pose_xy_flip:
                self.get_logger().info('Applied runtime pose transform: x/y sign flip before moveL conversion.')
        self.get_logger().info(
            'Move target(moveL 6D): %s'
            % np.array2string(np.asarray(move_pose, dtype=np.float64), precision=6, separator=', ')
        )

        if not self._init_robot_control():
            if self.exit_after_move:
                rclpy.shutdown()
            return

        ok = False
        try:
            ok = bool(self.rtde_c.moveL(move_pose, float(self.move_speed), float(self.move_acceleration)))
        except Exception as exc:
            self.get_logger().error('moveL failed: %s' % exc)

        if ok:
            self.get_logger().info('Move completed successfully.')
        else:
            self.get_logger().error('Move failed (moveL returned False).')

        if self.exit_after_move:
            rclpy.shutdown()

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _pixel_to_3d(self, u: int, v: int, depth_image: np.ndarray):
        if self.camera_matrix is None:
            return None
        if v < 0 or v >= depth_image.shape[0] or u < 0 or u >= depth_image.shape[1]:
            return None

        depth_value = depth_image[v, u]
        if np.issubdtype(depth_image.dtype, np.integer):
            z = float(depth_value) / 1000.0
        else:
            z = float(depth_value)

        if not np.isfinite(z) or z <= 0.0:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    def _transform_point(self, point: np.ndarray, transform_msg):
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

    def _read_center_height_base_m(self, blob, rgb_image: np.ndarray, depth_image: np.ndarray):
        if depth_image is None or self.camera_matrix is None:
            return None

        contour = blob.get('contour')
        if contour is None:
            return None

        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

        points_3d, _ = self.project_contour_region_to_3d(contour, rgb_image, depth_image)
        center_cam_px = self._pixel_to_3d(cx, cy, depth_image)

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

        source_frame = self.latest_depth_frame if self.latest_depth_frame else self.camera_frame

        try:
            tf_cam_to_base = self.tf_buffer.lookup_transform(
                self.base_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05),
            )
        except Exception:
            return None

        point_base = self._transform_point(center_cam, tf_cam_to_base)
        return float(point_base[2])

    def _read_center_depth_cam_m(self, blob, depth_image: np.ndarray):
        if depth_image is None:
            return None

        contour = blob.get('contour')
        if contour is None:
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

    def project_contour_region_to_3d(self, contour: np.ndarray, rgb_image: np.ndarray, depth_image: np.ndarray):
        region_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)

        pixel_indices = np.column_stack(np.where(region_mask > 0))
        points_3d = []
        colors_3d = []

        for v, u in pixel_indices:
            point_3d = self._pixel_to_3d(int(u), int(v), depth_image)
            if point_3d is None:
                continue

            points_3d.append(point_3d)
            b, g, r = rgb_image[v, u]
            colors_3d.append([r / 255.0, g / 255.0, b / 255.0])

        if not points_3d:
            return None, None

        return np.asarray(points_3d, dtype=np.float64), np.asarray(colors_3d, dtype=np.float64)

    def image_callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            self.latest_rgb = np.asarray(self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8'))
            self.latest_depth = np.asarray(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'))
            self.latest_camera_frame = rgb_msg.header.frame_id.strip() if rgb_msg.header.frame_id else None
            self.latest_depth_frame = depth_msg.header.frame_id.strip() if depth_msg.header.frame_id else None
        except CvBridgeError as exc:
            self.get_logger().error('CvBridge conversion failed: %s' % exc)
            return

        if self.show_window and not self._debug_window_failed:
            try:
                vis_image = self._build_detection_overlay(self.latest_rgb)
                cv2.imshow(self.window_name, vis_image)
                cv2.waitKey(1)
            except Exception as exc:
                self._debug_window_failed = True
                self.get_logger().warning('Debug window disabled: %s' % exc)

    def _extract_valid_contours(self, mask: np.ndarray):
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

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            valid_contours.append((contour, approx, float(area)))

        return clean_mask, valid_contours

    def _extract_largest_contour(self, mask: np.ndarray):
        clean_mask, valid_contours = self._extract_valid_contours(mask)
        if len(valid_contours) == 0:
            return clean_mask, None, None, 0.0

        best_contour, best_approx, best_area = max(valid_contours, key=lambda item: item[2])
        return clean_mask, best_contour, best_approx, best_area

    def _find_largest_color_blob_2d(self, color: str, rgb_image: np.ndarray):
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for lower, upper in self.color_ranges[color]:
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv_image, lower, upper))

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

    def _build_detection_overlay(self, rgb_image: np.ndarray):
        vis_image = rgb_image.copy()
        bgr_map = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
        }

        color_blobs = []
        img_h, img_w = vis_image.shape[:2]
        for color in self.color_order:
            blob = self._find_largest_color_blob_2d(color, rgb_image)
            if blob is None:
                continue
            color_blobs.append(blob)

            bgr = bgr_map[color]
            center_height_base_m = self._read_center_height_base_m(blob, rgb_image, self.latest_depth)
            center_depth_cam_m = self._read_center_depth_cam_m(blob, self.latest_depth)
            if center_height_base_m is None:
                z_base_text = 'z_base:n/a'
            else:
                z_base_text = 'z_base:%.3fm' % center_height_base_m
            if center_depth_cam_m is None:
                z_cam_text = 'z_cam:n/a'
            else:
                z_cam_text = 'z_cam:%.3fm' % center_depth_cam_m
            cv2.drawContours(vis_image, [blob['approx']], -1, bgr, 2)
            cv2.circle(vis_image, blob['center'], 5, bgr, -1)
            text_x = int(np.clip(blob['center'][0] + 8, 0, max(0, img_w - 180)))
            text_y = int(np.clip(blob['center'][1] - 8, 16, max(16, img_h - 24)))
            cv2.putText(
                vis_image,
                '%s area: %.0f' % (color, blob['area']),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_image,
                '%s %s' % (z_base_text, z_cam_text),
                (text_x, min(img_h - 6, text_y + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )

        if color_blobs:
            global_best = max(color_blobs, key=lambda item: item['area'])
            x, y, w, h = cv2.boundingRect(global_best['contour'])
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            global_height_base_m = self._read_center_height_base_m(global_best, rgb_image, self.latest_depth)
            global_depth_cam_m = self._read_center_depth_cam_m(global_best, self.latest_depth)
            if global_height_base_m is None:
                global_z_base_text = 'z_base:n/a'
            else:
                global_z_base_text = 'z_base:%.3fm' % global_height_base_m
            if global_depth_cam_m is None:
                global_z_cam_text = 'z_cam:n/a'
            else:
                global_z_cam_text = 'z_cam:%.3fm' % global_depth_cam_m
            cv2.putText(
                vis_image,
                'global max: %s %s %s' % (global_best['color'], global_z_base_text, global_z_cam_text),
                (max(0, x), max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis_image

    def _collect_color_blob_z_values(self, rgb_image: np.ndarray):
        z_items = []
        for color in self.color_order:
            blob = self._find_largest_color_blob_2d(color, rgb_image)
            if blob is None:
                continue
            z_items.append(
                {
                    'color': color,
                    'area': float(blob['area']),
                    'z_base': self._read_center_height_base_m(blob, rgb_image, self.latest_depth),
                    'z_cam': self._read_center_depth_cam_m(blob, self.latest_depth),
                }
            )
        return z_items

    def _log_detection_depths(self, rgb_image: np.ndarray, depth_image: np.ndarray):
        if rgb_image is None or depth_image is None:
            self.get_logger().info('Depth monitor: no synchronized RGB/depth frame yet.')
            return

        blobs = []
        for color in self.color_order:
            blob = self._find_largest_color_blob_2d(color, rgb_image)
            if blob is None:
                continue
            blobs.append(blob)

            z_base = self._read_center_height_base_m(blob, rgb_image, depth_image)
            z_cam = self._read_center_depth_cam_m(blob, depth_image)
            z_base_text = 'n/a' if z_base is None else ('%.3fm' % z_base)
            z_cam_text = 'n/a' if z_cam is None else ('%.3fm' % z_cam)
            self.get_logger().info(
                'Depth monitor: %s area=%.0f z_base=%s z_cam=%s'
                % (color, blob['area'], z_base_text, z_cam_text)
            )

        if not blobs:
            self.get_logger().info('Depth monitor: no valid color blob detected.')
            return

        global_best = max(blobs, key=lambda item: item['area'])
        global_z_base = self._read_center_height_base_m(global_best, rgb_image, depth_image)
        global_z_cam = self._read_center_depth_cam_m(global_best, depth_image)
        global_z_base_text = 'n/a' if global_z_base is None else ('%.3fm' % global_z_base)
        global_z_cam_text = 'n/a' if global_z_cam is None else ('%.3fm' % global_z_cam)
        self.get_logger().info(
            'Depth monitor: global max=%s area=%.0f z_base=%s z_cam=%s'
            % (global_best['color'], global_best['area'], global_z_base_text, global_z_cam_text)
        )

    def _inverse_transform_pose_frame(self, position: np.ndarray, quaternion_xyzw: np.ndarray):
        runtime_position = np.asarray(position, dtype=np.float64)
        original_position = runtime_position.copy()
        original_position[0] = -original_position[0]
        original_position[1] = -original_position[1]

        frame_rotation = Rotation.from_euler('z', np.pi)
        original_quaternion = (frame_rotation * Rotation.from_quat(np.asarray(quaternion_xyzw, dtype=np.float64))).as_quat()
        if original_quaternion[3] > 0.0:
            original_quaternion = -original_quaternion
        return original_position, original_quaternion

    def _lookup_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
        except Exception as exc:
            self.get_logger().warning('TF not ready (%s -> %s): %s' % (self.tool_frame, self.base_frame, exc))
            return None

        position = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ],
            dtype=np.float64,
        )
        return position, quaternion

    def _format_parameter_block(self, pose_name: str, values: np.ndarray):
        formatted_values = ', '.join(f'{float(value):.6f}' for value in values)
        return (
            "self.declare_parameter(\n"
            f"    '{pose_name}',\n"
            f"    [{formatted_values}],\n"
            ")"
        )

    def print_pose_block(self):
        result = self._lookup_current_pose()
        if result is None:
            return

        position, quaternion = result
        runtime_pose_values = np.concatenate((position, quaternion), axis=0)
        runtime_block = self._format_parameter_block(self.pose_name + '_runtime', runtime_pose_values)
        self.get_logger().info('\nRuntime pose:\n' + runtime_block)

        if self.mode != 'monitor':
            return
        if self.latest_rgb is None or self.latest_depth is None:
            return

        z_items = self._collect_color_blob_z_values(self.latest_rgb)
        if not z_items:
            self.get_logger().info('z monitor: no valid color blob')
            return

        for item in z_items:
            z_base_text = 'n/a' if item['z_base'] is None else ('%.3fm' % float(item['z_base']))
            z_cam_text = 'n/a' if item['z_cam'] is None else ('%.3fm' % float(item['z_cam']))
            self.get_logger().info(
                'z monitor %s: area=%.0f z_base=%s z_cam=%s'
                % (item['color'], float(item['area']), z_base_text, z_cam_text)
            )

        global_item = max(z_items, key=lambda entry: entry['area'])
        global_z_base_text = 'n/a' if global_item['z_base'] is None else ('%.3fm' % float(global_item['z_base']))
        global_z_cam_text = 'n/a' if global_item['z_cam'] is None else ('%.3fm' % float(global_item['z_cam']))
        self.get_logger().info(
            'z monitor global max: %s area=%.0f z_base=%s z_cam=%s'
            % (
                global_item['color'],
                float(global_item['area']),
                global_z_base_text,
                global_z_cam_text,
            )
        )
        self._log_detection_depths(self.latest_rgb, self.latest_depth)

    def destroy_node(self):
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if self.rtde_c is not None:
            try:
                self.rtde_c.stopScript()
            except Exception:
                pass

        super().destroy_node()


def main(args=None):
    os.environ.setdefault('RCUTILS_CONSOLE_OUTPUT_FORMAT', '{message}')
    rclpy.init(args=args)
    node = PoseCameraMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
