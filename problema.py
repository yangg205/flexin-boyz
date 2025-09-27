#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
import math
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse

from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from opposite_detector import SimpleOppositeDetector
from map_navigator import MapNavigator

class RobotState(Enum):
    WAITING_FOR_LINE = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    HANDLING_EVENT = 3
    LEAVING_INTERSECTION = 4
    REACQUIRING_LINE = 5
    DEAD_END = 6
    GOAL_REACHED = 7

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class JetBotController:
    DOMAIN = "https://hackathon2025-dev.fpt.edu.vn"
    ENDPOINT = "/api/maps/get_active_map/"

    def __init__(self, token, map_type):
        rospy.loginfo("Đang khởi tạo JetBot Event-Driven Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        
        self.video_writer = None
        self.initialize_video_writer()

        # Fetch map data
        self.map_data = self.fetch_map(token, map_type)
        rospy.loginfo(f"Map data: {json.dumps(self.map_data, ensure_ascii=False, indent=2)}")

        # Validate map data
        if not isinstance(self.map_data, dict):
            rospy.logerr("Lỗi: Map data không phải là dictionary.")
            raise ValueError("Map data phải là một dictionary.")
        if 'nodes' not in self.map_data or 'edges' not in self.map_data:
            rospy.logerr("Lỗi: Map data thiếu 'nodes' hoặc 'edges'.")
            raise ValueError("Map data thiếu các trường bắt buộc: nodes, edges.")
        
        # Initialize MapNavigator with raw map data
        self.navigator = MapNavigator(self.map_data)
        
        # Validate start_node and end_node
        if self.navigator.start_node is None or self.navigator.end_node is None:
            rospy.logerr(f"Lỗi: start_node ({self.navigator.start_node}) hoặc end_node ({self.navigator.end_node}) không hợp lệ.")
            raise ValueError("start_node hoặc end_node không được khởi tạo đúng trong MapNavigator.")
        
        self.current_node_id = self.navigator.start_node
        self.target_node_id = None
        self.planned_path = None
        self.banned_edges = []
        self.plan_initial_route()

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký vào các topic /scan và /csi_cam_0/image_raw.")
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def create_session_with_retries(self, total_retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
        s = requests.Session()
        try:
            retries = Retry(
                total=total_retries,
                backoff_factor=backoff_factor,
                status_forcelist=status_forcelist,
                allowed_methods=frozenset(["GET", "POST"])
            )
        except TypeError:
            retries = Retry(
                total=total_retries,
                backoff_factor=backoff_factor,
                status_forcelist=status_forcelist,
                method_whitelist=frozenset(["GET", "POST"])
            )
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def fetch_map(self, token: str, map_type: str, timeout=10):
        if not token:
            raise ValueError("Token is required")
        if map_type not in ["map_a", "map_b", "map_z"]:
            raise ValueError("map_type must be one of: map_a, map_b, map_z")
        params = {"token": token, "map_type": map_type}
        url = self.DOMAIN.rstrip("/") + self.ENDPOINT
        session = self.create_session_with_retries()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            rospy.logerr(f"HTTP error khi lấy map: {e} (status {getattr(e.response, 'status_code', None)})")
            try:
                rospy.logerr(f"Response body: {e.response.text}")
            except Exception:
                pass
            raise
        except Exception as e:
            rospy.logerr(f"Lỗi khi kết nối/parse: {e}")
            raise

    def plan_initial_route(self): 
        """Lập kế hoạch đường đi ban đầu từ điểm xuất phát đến đích."""
        rospy.loginfo(f"Đang lập kế hoạch từ node {self.navigator.start_node} đến {self.navigator.end_node}...")
        try:
            self.planned_path = self.navigator.find_path(
                self.navigator.start_node, 
                self.navigator.end_node,
                self.banned_edges
            )
            if self.planned_path and len(self.planned_path) > 1:
                self.target_node_id = self.planned_path[1]
                rospy.loginfo(f"Đã tìm thấy đường đi: {self.planned_path}. Đích đến đầu tiên: {self.target_node_id}")
            else:
                rospy.logerr("Không tìm thấy đường đi hoặc đường đi quá ngắn!")
                self._set_state(RobotState.DEAD_END)
        except Exception as e:
            rospy.logerr(f"Lỗi khi lập kế hoạch đường đi: {e}")
            self._set_state(RobotState.DEAD_END)
            raise

    def initialize_video_writer(self):
        """Khởi tạo đối tượng VideoWriter."""
        try:
            frame_size = (self.WIDTH, self.HEIGHT)
            self.video_writer = cv2.VideoWriter(self.VIDEO_OUTPUT_FILENAME, 
                                                self.VIDEO_FOURCC, 
                                                self.VIDEO_FPS, 
                                                frame_size)
            if self.video_writer.isOpened():
                rospy.loginfo(f"Bắt đầu ghi video vào file '{self.VIDEO_OUTPUT_FILENAME}'")
            else:
                rospy.logerr("Không thể mở file video để ghi.")
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi khi khởi tạo VideoWriter: {e}")
            self.video_writer = None

    def draw_debug_info(self, image):
        """Vẽ các thông tin gỡ lỗi lên một khung hình."""
        if image is None:
            return None
        
        debug_frame = image.copy()
        
        # 1. Vẽ các ROI
        cv2.rectangle(debug_frame, (0, self.ROI_Y), (self.WIDTH-1, self.ROI_Y + self.ROI_H), (0, 255, 0), 1)
        cv2.rectangle(debug_frame, (0, self.LOOKAHEAD_ROI_Y), (self.WIDTH-1, self.LOOKAHEAD_ROI_Y + self.LOOKAHEAD_ROI_H), (0, 255, 255), 1)

        # 2. Vẽ trạng thái hiện tại
        state_text = f"State: {self.current_state.name}"
        cv2.putText(debug_frame, state_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 3. Vẽ line và trọng tâm (nếu robot đang bám line)
        if self.current_state == RobotState.DRIVING_STRAIGHT:
            line_center = self._get_line_center(image, self.ROI_Y, self.ROI_H)
            if line_center is not None:
                cv2.line(debug_frame, (line_center, self.ROI_Y), (line_center, self.ROI_Y + self.ROI_H), (0, 0, 255), 2)

        return debug_frame

    def setup_parameters(self):
        self.WIDTH, self.HEIGHT = 300, 300
        self.BASE_SPEED = 0.16
        self.TURN_SPEED = 0.2
        self.TURN_DURATION_90_DEG = 0.8
        self.ROI_Y = int(self.HEIGHT * 0.85)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.ROI_CENTER_WIDTH_PERCENT = 0.5
        self.LOOKAHEAD_ROI_Y = int(self.HEIGHT * 0.60)
        self.LOOKAHEAD_ROI_H = int(self.HEIGHT * 0.15)
        self.CORRECTION_GAIN = 0.5
        self.SAFE_ZONE_PERCENT = 0.3
        self.LINE_COLOR_LOWER = np.array([0, 0, 0])
        self.LINE_COLOR_UPPER = np.array([180, 255, 75])
        self.INTERSECTION_CLEARANCE_DURATION = 1.5
        self.INTERSECTION_APPROACH_DURATION = 0.5
        self.LINE_REACQUIRE_TIMEOUT = 3.0
        self.SCAN_PIXEL_THRESHOLD = 100
        self.current_state = None
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.MAX_CORRECTION_ADJ = 0.12
        self.VIDEO_OUTPUT_FILENAME = 'jetbot_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
        # Define direction label to enum mapping
        self.LABEL_TO_DIRECTION_ENUM = {
            'N': Direction.NORTH,
            'E': Direction.EAST,
            'S': Direction.SOUTH,
            'W': Direction.WEST
        }

    def initialize_hardware(self):
        try:
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot (động cơ) đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock object. Lỗi: {e}")
            from unittest.mock import Mock
            self.robot = Mock()
    
    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial: rospy.loginfo(f"Chuyển trạng thái: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            if image_msg.encoding.endswith('compressed'):
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding: cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def run(self):
        rospy.loginfo("Bắt đầu vòng lặp. Đợi 3 giây...") 
        time.sleep(3) 
        rospy.loginfo("Hành trình bắt đầu!")
        self.detector.start_scanning()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.current_state == RobotState.WAITING_FOR_LINE:
                rospy.loginfo_throttle(5, "Đang ở trạng thái chờ... Tìm kiếm vạch kẻ đường để bắt đầu.")
                self.robot.stop()
                if self.latest_image is None:
                    rate.sleep()
                    continue
                lookahead_line = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                execution_line = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if lookahead_line is not None and execution_line is not None:
                    rospy.loginfo("Đã tìm thấy vạch kẻ đường! Bắt đầu hành trình.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
            if self.current_state == RobotState.DRIVING_STRAIGHT:
                if self.latest_image is None:
                    rospy.logwarn_throttle(5, "Đang chờ dữ liệu hình ảnh từ topic camera...")
                    self.robot.stop()
                    rate.sleep()
                    continue
                if self.detector.process_detection():
                    rospy.loginfo("SỰ KIỆN (LiDAR): Phát hiện giao lộ. Dừng ngay lập tức.")
                    self.robot.stop()
                    time.sleep(0.5)
                    self.current_node_id = self.target_node_id
                    rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")
                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()
                    continue
                lookahead_line_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                if lookahead_line_center is None:
                    rospy.logwarn("SỰ KIỆN (Dự báo): Vạch kẻ đường biến mất ở phía xa. Chuẩn bị vào giao lộ.")
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue
                execution_line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if execution_line_center is not None:
                    self.correct_course(execution_line_center)
                else:
                    rospy.logwarn("Trạng thái không nhất quán: ROI xa thấy line, ROI gần không thấy. Tạm dừng an toàn.")
                    self.robot.stop()
            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_APPROACH_DURATION:
                    rospy.loginfo("Đã tiến vào trung tâm giao lộ. Dừng lại để xử lý.")
                    self.robot.stop() 
                    time.sleep(0.5)
                    self.current_node_id = self.target_node_id
                    rospy.loginfo(f"==> ĐÃ ĐẾN node {self.current_node_id}.")
                    if self.current_node_id == self.navigator.end_node:
                        rospy.loginfo("ĐÃ ĐẾN ĐÍCH CUỐI CÙNG!")
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()
            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
                if rospy.get_time() - self.state_change_time > self.INTERSECTION_CLEARANCE_DURATION:
                    rospy.loginfo("Đã thoát khỏi khu vực giao lộ. Bắt đầu tìm kiếm line mới.")
                    self._set_state(RobotState.REACQUIRING_LINE)
            elif self.current_state == RobotState.REACQUIRING_LINE:
                self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
                line_center_x = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if line_center_x is not None:
                    rospy.loginfo("Đã tìm thấy line mới! Chuyển sang chế độ bám line.")
                    self._set_state(RobotState.DRIVING_STRAIGHT)
                    continue
                if rospy.get_time() - self.state_change_time > self.LINE_REACQUIRE_TIMEOUT:
                    rospy.logerr("Không thể tìm thấy line mới sau khi rời giao lộ. Dừng lại.")
                    self._set_state(RobotState.DEAD_END)
            elif self.current_state == RobotState.DEAD_END:
                rospy.logwarn("Đã vào ngõ cụt hoặc gặp lỗi không thể phục hồi. Dừng hoạt động.") 
                self.robot.stop() 
                break
            elif self.current_state == RobotState.GOAL_REACHED: 
                rospy.loginfo("ĐÃ HOÀN THÀNH NHIỆM VỤ. Dừng hoạt động.") 
                self.robot.stop()
                break
            self._record_frame()
            rate.sleep()
        self.cleanup()

    def _record_frame(self):
        if self.video_writer is not None and self.latest_image is not None:
            debug_frame = self.draw_debug_info(self.latest_image)
            if debug_frame is not None:
                self.video_writer.write(debug_frame)

    def cleanup(self):
        rospy.loginfo("Dừng robot và giải phóng tài nguyên...") 
        if hasattr(self, 'robot') and self.robot is not None:
            self.robot.stop()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            rospy.loginfo("Đã lưu và đóng file video.")
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.stop_scanning()
        rospy.loginfo("Đã giải phóng tài nguyên. Chương trình kết thúc.")

    def map_absolute_to_relative(self, target_direction_label, current_robot_direction):
        target_dir = self.LABEL_TO_DIRECTION_ENUM.get(target_direction_label)
        if target_dir is None:
            rospy.logerr(f"Hướng không hợp lệ: {target_direction_label}")
            return None
        current_idx = current_robot_direction.value
        target_idx = target_dir.value
        diff = (target_idx - current_idx + 4) % 4 
        if diff == 0:
            return 'straight'
        elif diff == 1:
            return 'right'
        elif diff == 3: 
            return 'left'
        else: 
            return 'turn_around'
        
    def map_relative_to_absolute(self, relative_action, current_robot_direction):
        current_idx = current_robot_direction.value
        if relative_action == 'straight':
            target_idx = current_idx
        elif relative_action == 'right':
            target_idx = (current_idx + 1) % 4
        elif relative_action == 'left':
            target_idx = (current_idx - 1 + 4) % 4
        else:
            return None
        for label, direction in self.LABEL_TO_DIRECTION_ENUM.items():
            if direction.value == target_idx:
                return label
        return None
    
    def _get_line_center(self, image, roi_y, roi_h):
        if image is None:
            return None
        roi = image[roi_y : roi_y + roi_h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(color_mask)
        roi_height, roi_width = focus_mask.shape
        center_width = int(roi_width * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (roi_width - center_width) // 2
        end_x = start_x + center_width
        cv2.rectangle(focus_mask, (start_x, 0), (end_x, roi_height), 255, -1)
        final_mask = cv2.bitwise_and(color_mask, focus_mask)
        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD:
            return None
        M = cv2.moments(c)
        if M["m00"] > 0:
            return int(M["m10"] / M["m00"])
        return None
    
    def correct_course(self, line_center_x):
        error = line_center_x - (self.WIDTH / 2)
        if abs(error) < (self.WIDTH / 2) * self.SAFE_ZONE_PERCENT:
            self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
            return
        adj = (error / (self.WIDTH / 2)) * self.CORRECTION_GAIN
        adj = np.clip(adj, -self.MAX_CORRECTION_ADJ, self.MAX_CORRECTION_ADJ)
        left_motor = self.BASE_SPEED + adj
        right_motor = self.BASE_SPEED - adj
        self.robot.set_motors(left_motor, right_motor)
        
    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý...")
        self.robot.stop() 
        time.sleep(0.5)
        current_direction = self.DIRECTIONS[self.current_direction_index]
        rospy.loginfo("[STEP 3] Lập kế hoạch điều hướng theo bản đồ...")
        final_decision = None
        while True:
            planned_direction_label = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
            if not planned_direction_label:
                rospy.logerr("Lỗi kế hoạch: Không tìm thấy bước tiếp theo.") 
                self._set_state(RobotState.DEAD_END) 
                return
            planned_action = self.map_absolute_to_relative(planned_direction_label, current_direction)
            if planned_action is None:
                rospy.logerr("Không thể xác định hành động từ hướng được đề xuất.")
                self._set_state(RobotState.DEAD_END)
                return
            rospy.loginfo(f"Kế hoạch A* đề xuất: Đi {planned_action} (hướng {planned_direction_label})")
            final_decision = planned_action
            break
        if final_decision == 'straight': 
            rospy.loginfo("[FINAL] Decision: Go STRAIGHT.")
        elif final_decision == 'right': 
            rospy.loginfo("[FINAL] Decision: Turn RIGHT.") 
            self.turn_robot(90, True)
        elif final_decision == 'left': 
            rospy.loginfo("[FINAL] Decision: Turn LEFT.") 
            self.turn_robot(-90, True)
        else:
            rospy.logwarn("[!!!] DEAD END! No valid paths found.") 
            self._set_state(RobotState.DEAD_END)
            return
        next_node_id = self.planned_path[self.planned_path.index(self.current_node_id) + 1]
        self.target_node_id = next_node_id
        rospy.loginfo(f"==> Đang di chuyển đến node tiếp theo: {self.target_node_id}")
        self._set_state(RobotState.LEAVING_INTERSECTION)
    
    def turn_robot(self, degrees, update_main_direction=True):
        duration = abs(degrees) / 90.0 * self.TURN_DURATION_90_DEG
        if degrees > 0: 
            self.robot.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
        elif degrees < 0: 
            self.robot.set_motors(-self.TURN_SPEED, self.TURN_SPEED)
        if degrees != 0: 
            start_time = rospy.get_time()
            while rospy.get_time() - start_time < duration:
                self._record_frame()
                rospy.sleep(1.0 / self.VIDEO_FPS)
        self.robot.stop()
        if update_main_direction and degrees % 90 == 0 and degrees != 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4
            rospy.loginfo(f"==> Hướng đi MỚI: {self.DIRECTIONS[self.current_direction_index].name}")
        time.sleep(0.5)
        self._record_frame()
    
    def _does_path_exist_in_frame(self, image):
        if image is None: 
            return False
        roi = image[self.ROI_Y : self.ROI_Y + self.ROI_H, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        _img, contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APROX_SIMPLE)
        return bool(contours) and cv2.contourArea(max(contours, key=cv2.contourArea)) > self.SCAN_PIXEL_THRESHOLD
    
    def scan_for_available_paths_proactive(self):
        rospy.loginfo("[SCAN] Bắt đầu quét chủ động...")
        paths = {"straight": False, "right": False, "left": False}
        if self.latest_image is not None:
            paths["straight"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        time.sleep(0.5)
        if self.latest_image is not None:
            paths["right"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(-180, update_main_direction=False)
        time.sleep(0.5)
        if self.latest_image is not None:
            paths["left"] = self._does_path_exist_in_frame(self.latest_image)
        self.turn_robot(90, update_main_direction=False)
        rospy.loginfo(f"[SCAN] Kết quả: {paths}")
        return paths

def main():
    rospy.init_node('jetbot_controller_node', anonymous=True)
    parser = argparse.ArgumentParser(description="JetBot Controller with map fetching")
    parser.add_argument("--token", type=str, required=True, help="Team token (required)")
    parser.add_argument("--map_type", type=str, default="map_z", choices=["map_a", "map_b", "map_z"],
                        help="Loại map: map_a, map_b, map_z (mặc định: map_z)")
    args = parser.parse_args()

    try:
        controller = JetBotController(token=args.token, map_type=args.map_type)
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node đã bị ngắt.")
    except Exception as e:
        rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)

if __name__ == '__main__':
    main()