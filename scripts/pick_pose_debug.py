#!/usr/bin/env python3

import time

import rclpy

from pick_and_place import CubePickPlaceController


class PickPoseDebugController(CubePickPlaceController):
    def __init__(self):
        self._last_plan_log_time = 0.0
        self._plan_log_interval_sec = 0.5
        super().__init__()
        self.get_logger().info(
            'Pick pose debug mode active: move to home_pose, plan pick target pose, and render overlay only.'
        )

    def _init_suction(self):
        self.serial_suction = None
        self.get_logger().info('Debug mode: skip suction serial initialization.')

    def _select_pick_target_from_frame(self, rgb_image, depth_image, log_reject=False):
        # Debug mode target selection ignores placed-count gating so the planned
        # pose overlay can always be visualized while tuning camera/pose logic.
        best_in_workspace = None
        best_in_workspace_mask = None
        best_anywhere = None
        best_anywhere_mask = None

        for color in self.color_order:
            target, mask = self._detect_color_target(color, rgb_image.copy(), depth_image)
            if target is None:
                continue

            if best_anywhere is None or target.area > best_anywhere.area:
                best_anywhere = target
                best_anywhere_mask = mask

            center_base, _, _ = self._target_to_base(target)
            if self._is_point_in_pick_workspace(center_base):
                if best_in_workspace is None or target.area > best_in_workspace.area:
                    best_in_workspace = target
                    best_in_workspace_mask = mask

        if best_in_workspace is not None:
            return best_in_workspace, best_in_workspace_mask

        if best_anywhere is not None and log_reject:
            center_base, _, _ = self._target_to_base(best_anywhere)
            self.get_logger().warning(
                'Debug fallback: no target inside pick workspace, use global max %s at center_base=%s'
                % (best_anywhere.color, self._format_pose(center_base))
            )
        return best_anywhere, best_anywhere_mask

    def _run_cycle_once(self):
        try:
            if self.latest_rgb is None or self.latest_depth is None or self.camera_matrix is None:
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
                if self.latest_rgb is None or self.latest_depth is None or self.latest_stamp is None:
                    self.get_logger().info('Waiting for first camera frame after home_pose...')
                    return
                self._awaiting_home_refresh = False
                self._startup_time = time.monotonic()
                self.get_logger().info('Camera frame refreshed after home_pose, start pose planning.')

            pick_result = self._select_pick_target()
            if pick_result is None:
                self.get_logger().info('No valid green/red/yellow cube found in the current view.')
                return

            target, _ = pick_result
            if target is None:
                return

            center_base, normal_base, top_base, approach_pose, grasp_pose, lift_pose = self._plan_pick_poses(target)
            now = time.monotonic()
            if now - self._last_plan_log_time >= self._plan_log_interval_sec:
                self._last_plan_log_time = now
                self.get_logger().info(
                    'Debug target=%s area=%.1f center_base=%s top_base=%s normal_base=%s'
                    % (
                        target.color,
                        target.area,
                        self._format_pose(center_base),
                        self._format_pose(top_base),
                        self._format_pose(normal_base),
                    )
                )
                self.get_logger().info('Debug approach_pose: %s' % self._format_pose(approach_pose))
                self.get_logger().info('Debug grasp_pose: %s' % self._format_pose(grasp_pose))
                self.get_logger().info('Debug lift_pose: %s' % self._format_pose(lift_pose))
        finally:
            with self._busy_lock:
                self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = PickPoseDebugController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_debug_window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
