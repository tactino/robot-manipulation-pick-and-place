#!/usr/bin/env python3

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


class ColorThresholdTuner(Node):
    def __init__(self):
        super().__init__('color_threshold_tuner')

        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('color_min_area', 300.0)

        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.color_min_area = float(self.get_parameter('color_min_area').get_parameter_value().double_value)

        self.bridge = CvBridge()
        self._image_lock = threading.Lock()
        self.latest_rgb = None

        self.color_names = ['green', 'red', 'yellow']
        self.default_ranges = {
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

        self.window_name = 'Color Threshold Tuner'
        self.mask_window_name = 'Color Mask'
        self.raw_window_name = 'Raw RGB'
        self._last_color_id = -1

        self.create_subscription(Image, self.rgb_topic, self.image_callback, 10)
        self._create_ui()
        self.timer = self.create_timer(0.03, self._tick)

        self.get_logger().info('Color threshold tuner started.')
        self.get_logger().info('Press 1/2/3 to switch color, p to print ranges, q or ESC to quit.')

    def _create_ui(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.mask_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.raw_window_name, cv2.WINDOW_NORMAL)

        def _no_op(_):
            return

        # 0: green, 1: red, 2: yellow
        cv2.createTrackbar('COLOR_ID', self.window_name, 0, 2, _no_op)

        # Primary HSV range
        cv2.createTrackbar('H_MIN', self.window_name, 0, 180, _no_op)
        cv2.createTrackbar('H_MAX', self.window_name, 180, 180, _no_op)
        cv2.createTrackbar('S_MIN', self.window_name, 0, 255, _no_op)
        cv2.createTrackbar('S_MAX', self.window_name, 255, 255, _no_op)
        cv2.createTrackbar('V_MIN', self.window_name, 0, 255, _no_op)
        cv2.createTrackbar('V_MAX', self.window_name, 255, 255, _no_op)

        # Secondary HSV range (useful for red wrap-around)
        cv2.createTrackbar('USE_RANGE2', self.window_name, 0, 1, _no_op)
        cv2.createTrackbar('H2_MIN', self.window_name, 0, 180, _no_op)
        cv2.createTrackbar('H2_MAX', self.window_name, 180, 180, _no_op)
        cv2.createTrackbar('S2_MIN', self.window_name, 0, 255, _no_op)
        cv2.createTrackbar('S2_MAX', self.window_name, 255, 255, _no_op)
        cv2.createTrackbar('V2_MIN', self.window_name, 0, 255, _no_op)
        cv2.createTrackbar('V2_MAX', self.window_name, 255, 255, _no_op)

        # Match project preprocessing controls
        cv2.createTrackbar('SV_RELAX', self.window_name, 25, 80, _no_op)
        cv2.createTrackbar('CLAHE_X10', self.window_name, 20, 50, _no_op)
        cv2.createTrackbar('MORPH_KERNEL', self.window_name, 5, 21, _no_op)
        cv2.createTrackbar('OPEN_ITERS', self.window_name, 1, 5, _no_op)
        cv2.createTrackbar('CLOSE_ITERS', self.window_name, 1, 5, _no_op)

        self._load_selected_color_defaults(force=True)

    def image_callback(self, msg: Image):
        try:
            rgb = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
        except CvBridgeError as exc:
            self.get_logger().error('CvBridge conversion failed: %s' % exc)
            return

        with self._image_lock:
            self.latest_rgb = rgb

    def _get_trackbar(self, name: str):
        return cv2.getTrackbarPos(name, self.window_name)

    def _set_trackbar(self, name: str, value: int):
        cv2.setTrackbarPos(name, self.window_name, int(value))

    def _read_thresholds(self):
        h_min = self._get_trackbar('H_MIN')
        h_max = self._get_trackbar('H_MAX')
        s_min = self._get_trackbar('S_MIN')
        s_max = self._get_trackbar('S_MAX')
        v_min = self._get_trackbar('V_MIN')
        v_max = self._get_trackbar('V_MAX')

        h2_min = self._get_trackbar('H2_MIN')
        h2_max = self._get_trackbar('H2_MAX')
        s2_min = self._get_trackbar('S2_MIN')
        s2_max = self._get_trackbar('S2_MAX')
        v2_min = self._get_trackbar('V2_MIN')
        v2_max = self._get_trackbar('V2_MAX')

        use_range2 = self._get_trackbar('USE_RANGE2') > 0
        sv_relax = self._get_trackbar('SV_RELAX')
        clahe_clip = self._get_trackbar('CLAHE_X10') / 10.0
        morph_kernel = self._get_trackbar('MORPH_KERNEL')
        open_iters = self._get_trackbar('OPEN_ITERS')
        close_iters = self._get_trackbar('CLOSE_ITERS')

        if morph_kernel < 1:
            morph_kernel = 1
        if morph_kernel % 2 == 0:
            morph_kernel += 1

        return {
            'range1': (
                np.array([h_min, s_min, v_min], dtype=np.uint8),
                np.array([h_max, s_max, v_max], dtype=np.uint8),
            ),
            'range2': (
                np.array([h2_min, s2_min, v2_min], dtype=np.uint8),
                np.array([h2_max, s2_max, v2_max], dtype=np.uint8),
            ),
            'use_range2': use_range2,
            'sv_relax': int(sv_relax),
            'clahe_clip': float(clahe_clip),
            'morph_kernel': int(morph_kernel),
            'open_iters': int(open_iters),
            'close_iters': int(close_iters),
        }

    def _load_selected_color_defaults(self, force: bool = False):
        color_id = self._get_trackbar('COLOR_ID')
        if not force and color_id == self._last_color_id:
            return

        self._last_color_id = color_id
        color_name = self.color_names[color_id]
        ranges = self.default_ranges[color_name]

        lower1, upper1 = ranges[0]
        self._set_trackbar('H_MIN', int(lower1[0]))
        self._set_trackbar('S_MIN', int(lower1[1]))
        self._set_trackbar('V_MIN', int(lower1[2]))
        self._set_trackbar('H_MAX', int(upper1[0]))
        self._set_trackbar('S_MAX', int(upper1[1]))
        self._set_trackbar('V_MAX', int(upper1[2]))

        if len(ranges) >= 2:
            lower2, upper2 = ranges[1]
            self._set_trackbar('USE_RANGE2', 1)
            self._set_trackbar('H2_MIN', int(lower2[0]))
            self._set_trackbar('S2_MIN', int(lower2[1]))
            self._set_trackbar('V2_MIN', int(lower2[2]))
            self._set_trackbar('H2_MAX', int(upper2[0]))
            self._set_trackbar('S2_MAX', int(upper2[1]))
            self._set_trackbar('V2_MAX', int(upper2[2]))
        else:
            self._set_trackbar('USE_RANGE2', 0)

        self.get_logger().info('Loaded default HSV ranges for %s.' % color_name)

    def _build_color_mask(self, bgr_image: np.ndarray, cfg: dict):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        if cfg['clahe_clip'] > 0.0:
            hsv = hsv.copy()
            clahe = cv2.createCLAHE(clipLimit=cfg['clahe_clip'], tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])

        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        def _relaxed_bounds(lower: np.ndarray, upper: np.ndarray):
            relax = cfg['sv_relax']
            low = lower.copy()
            high = upper.copy()
            low[1] = np.uint8(max(0, int(low[1]) - relax))
            low[2] = np.uint8(max(0, int(low[2]) - relax))
            return low, high

        lower1, upper1 = _relaxed_bounds(cfg['range1'][0], cfg['range1'][1])
        combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lower1, upper1))

        if cfg['use_range2']:
            lower2, upper2 = _relaxed_bounds(cfg['range2'][0], cfg['range2'][1])
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lower2, upper2))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg['morph_kernel'], cfg['morph_kernel']),
        )
        if cfg['open_iters'] > 0:
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=cfg['open_iters'],
            )
        if cfg['close_iters'] > 0:
            combined_mask = cv2.morphologyEx(
                combined_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=cfg['close_iters'],
            )

        return combined_mask

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

    def _find_largest_color_blob_2d(self, color_name: str, rgb_image: np.ndarray, cfg: dict):
        mask = self._build_color_mask(rgb_image, cfg)
        _, best_contour, best_approx, best_area = self._extract_largest_contour(mask)
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
            'color': color_name,
            'area': float(best_area),
            'contour': best_contour,
            'approx': best_approx,
            'center': (cx, cy),
        }

    def _draw_overlay(self, image: np.ndarray, quads: list, largest_blob: dict, color_name: str, cfg: dict):
        vis = image.copy()
        cv2.putText(
            vis,
            'Color: %s | Quads: %d | MinArea: %.0f' % (color_name, len(quads), self.color_min_area),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for contour, approx, area in quads:
            cv2.drawContours(vis, [approx], -1, (0, 255, 0), 2)
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2

            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(
                vis,
                'A=%.0f' % area,
                (cx + 6, max(15, cy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if largest_blob is not None:
            x, y, w, h = cv2.boundingRect(largest_blob['contour'])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(
                vis,
                'largest quad area: %.0f' % largest_blob['area'],
                (max(0, x), max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        low1 = cfg['range1'][0]
        up1 = cfg['range1'][1]
        line1 = 'R1 H[%d,%d] S[%d,%d] V[%d,%d]' % (
            int(low1[0]), int(up1[0]), int(low1[1]), int(up1[1]), int(low1[2]), int(up1[2])
        )
        cv2.putText(vis, line1, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if cfg['use_range2']:
            low2 = cfg['range2'][0]
            up2 = cfg['range2'][1]
            line2 = 'R2 H[%d,%d] S[%d,%d] V[%d,%d]' % (
                int(low2[0]), int(up2[0]), int(low2[1]), int(up2[1]), int(low2[2]), int(up2[2])
            )
            cv2.putText(vis, line2, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return vis

    def _print_current_ranges(self, color_name: str, cfg: dict):
        ranges = [cfg['range1']]
        if cfg['use_range2']:
            ranges.append(cfg['range2'])

        self.get_logger().info('Current color: %s' % color_name)
        for idx, (low, up) in enumerate(ranges, start=1):
            self.get_logger().info(
                'R%d: lower=[%d,%d,%d], upper=[%d,%d,%d]'
                % (idx, int(low[0]), int(low[1]), int(low[2]), int(up[0]), int(up[1]), int(up[2]))
            )

        self.get_logger().info(
            'Python snippet: (np.array([%d, %d, %d], dtype=np.uint8), np.array([%d, %d, %d], dtype=np.uint8))'
            % (
                int(cfg['range1'][0][0]),
                int(cfg['range1'][0][1]),
                int(cfg['range1'][0][2]),
                int(cfg['range1'][1][0]),
                int(cfg['range1'][1][1]),
                int(cfg['range1'][1][2]),
            )
        )

    def _tick(self):
        self._load_selected_color_defaults()

        with self._image_lock:
            if self.latest_rgb is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    rclpy.shutdown()
                return
            rgb = self.latest_rgb.copy()

        color_id = self._get_trackbar('COLOR_ID')
        color_name = self.color_names[color_id]
        cfg = self._read_thresholds()

        mask = self._build_color_mask(rgb, cfg)
        clean_mask, quads = self._extract_valid_contours(mask)
        largest_blob = self._find_largest_color_blob_2d(color_name, rgb, cfg)
        vis = self._draw_overlay(rgb, quads, largest_blob, color_name, cfg)

        cv2.imshow(self.raw_window_name, vis)
        cv2.imshow(self.window_name, vis)
        cv2.imshow(self.mask_window_name, clean_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            self._set_trackbar('COLOR_ID', 0)
            self._load_selected_color_defaults(force=True)
        elif key == ord('2'):
            self._set_trackbar('COLOR_ID', 1)
            self._load_selected_color_defaults(force=True)
        elif key == ord('3'):
            self._set_trackbar('COLOR_ID', 2)
            self._load_selected_color_defaults(force=True)
        elif key == ord('p'):
            self._print_current_ranges(color_name, cfg)
        elif key in (27, ord('q')):
            rclpy.shutdown()

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ColorThresholdTuner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
