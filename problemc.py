#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import json
from enum import Enum
import argparse
from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from opposite_detector import SimpleOppositeDetector
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
from ultralytics import YOLO

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
    def __init__(self):
        rospy.loginfo("Đang khởi tạo JetBot Event-Driven Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_mqtt()

        # Khởi tạo mô hình YOLOv8
        try:
            self.yolo_model = YOLO('best.pt')  # Đường dẫn tới file best.pt
            rospy.loginfo("Đã tải mô hình YOLOv8 thành công.")
        except Exception as e:
            rospy.logerr(f"Lỗi khi tải mô hình YOLOv8: {e}")
            self.yolo_model = None

        self.video_writer = None
        self.initialize_video_writer()

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        rospy.loginfo("Đã đăng ký vào các topic /scan và /csi_cam_0/image_raw.")
        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def initialize_video_writer(self):
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
        if image is None:
            return None
        
        debug_frame = image.copy()
        cv2.rectangle(debug_frame, (0, self.ROI_Y), (self.WIDTH-1, self.ROI_Y + self.ROI_H), (0, 255, 0), 1)
        cv2.rectangle(debug_frame, (0, self.LOOKAHEAD_ROI_Y), (self.WIDTH-1, self.LOOKAHEAD_ROI_Y + self.LOOKAHEAD_ROI_H), (0, 255, 255), 1)
        state_text = f"State: {self.current_state.name}"
        cv2.putText(debug_frame, state_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        if self.current_state == RobotState.DRIVING_STRAIGHT:
            line_center = self._get_line_center(image, self.ROI_Y, self.ROI_H)
            if line_center is not None:
                cv2.line(debug_frame, (line_center, self.ROI_Y), (line_center, self.ROI_Y + self.ROI_H), (0, 0, 255), 2)
        
        # Vẽ các hộp bao của YOLO khi xử lý giao lộ
        if self.yolo_model is not None and self.current_state == RobotState.HANDLING_EVENT:
            try:
                results = self.yolo_model(debug_frame)
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf)
                        if confidence > 0.5 and class_name in self.PRESCRIPTIVE_SIGNS | self.PROHIBITIVE_SIGNS:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            except Exception as e:
                rospy.logerr(f"Lỗi khi vẽ kết quả YOLO: {e}")
        
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
        self.LABEL_TO_DIRECTION_ENUM = {
            'N': Direction.NORTH,
            'E': Direction.EAST,
            'S': Direction.SOUTH,
            'W': Direction.WEST
        }
        # MQTT Parameters
        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        # Sign Definitions
        self.PRESCRIPTIVE_SIGNS = {'L', 'R', 'F'}  # Left, Right, Forward
        self.PROHIBITIVE_SIGNS = {'NL', 'NR', 'NF'}  # No Left, No Right, No Forward
        self.DATA_ITEMS = {'qr_code', 'math_problem'}
        self.ANGLE_TO_FACE_SIGN_MAP = {
            Direction.NORTH: 45,
            Direction.EAST: -45,
            Direction.SOUTH: -135,
            Direction.WEST: 135
        }
        # Ánh xạ tên lớp YOLO (nếu cần)
        self.YOLO_CLASS_MAPPING = {
            'left': 'L',
            'right': 'R',
            'forward': 'F',
            'no_left': 'NL',
            'no_right': 'NR',
            'no_forward': 'NF'
        }

    def initialize_hardware(self):
        try:
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot (động cơ) đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock object. Lỗi: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_mqtt(self):
        self.mqtt_client = mqtt.Client()
        def on_connect(client, userdata, flags, rc):
            rospy.loginfo(f"Kết nối MQTT: {'Thành công' if rc == 0 else 'Thất bại'}")
        self.mqtt_client.on_connect = on_connect
        try:
            self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            rospy.logerr(f"Không thể kết nối MQTT: {e}")

    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial:
                rospy.loginfo(f"Chuyển trạng thái: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) if image_msg.encoding.endswith('compressed') else np_arr.reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding.lower():
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
            
            # --- Tắt phần hiển thị GUI để chạy headless ---
            # debug_frame = self.draw_debug_info(self.latest_image)
            # if debug_frame is not None:
            #     cv2.imshow("JetBot Camera", debug_frame)
            #     cv2.waitKey(1)
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def publish_data(self, data):
        try:
            self.mqtt_client.publish(self.MQTT_DATA_TOPIC, json.dumps(data))
            rospy.loginfo(f"Đã publish dữ liệu: {data}")
        except Exception as e:
            rospy.logerr(f"Lỗi khi publish MQTT: {e}")

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
            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                if self.latest_image is None:
                    rospy.logwarn_throttle(5, "Đang chờ dữ liệu hình ảnh từ topic camera...")
                    self.robot.stop()
                    rate.sleep()
                    continue
                if self.detector.process_detection():
                    rospy.loginfo("SỰ KIỆN (LiDAR): Phát hiện giao lộ. Dừng ngay lập tức.")
                    self.robot.stop()
                    time.sleep(0.5)
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
        if hasattr(self, 'mqtt_client') and self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        cv2.destroyAllWindows()  # Đóng cửa sổ OpenCV
        rospy.loginfo("Đã giải phóng tài nguyên. Chương trình kết thúc.")

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
        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    def scan_for_signs(self):
        """Scan for QR codes and YOLO-based sign detections."""
        detections = []
        if self.latest_image is None:
            rospy.logwarn("No image available for sign scanning.")
            return detections

        # Scan for QR codes
        qr_codes = decode(self.latest_image)
        for qr in qr_codes:
            qr_data = qr.data.decode('utf-8')
            detections.append({'class_name': 'qr_code', 'value': qr_data})
            rospy.loginfo(f"Detected QR code: {qr_data}")

        # YOLO-based sign detection
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(self.latest_image)
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = self.yolo_model.names[class_id]
                        confidence = float(box.conf)
                        # Ánh xạ tên lớp nếu cần
                        class_name = self.YOLO_CLASS_MAPPING.get(class_name, class_name)
                        if confidence > 0.5 and class_name in self.PRESCRIPTIVE_SIGNS | self.PROHIBITIVE_SIGNS:
                            detections.append({'class_name': class_name})
                            rospy.loginfo(f"Detected sign: {class_name} (Confidence: {confidence:.2f})")
            except Exception as e:
                rospy.logerr(f"Lỗi khi chạy YOLOv8: {e}")
        else:
            rospy.logwarn("Mô hình YOLOv8 không được tải. Bỏ qua phát hiện biển báo.")

        return detections

    def handle_intersection(self):
        rospy.loginfo("\n[GIAO LỘ] Dừng lại và xử lý...")
        self.robot.stop()
        time.sleep(0.5)

        current_direction = self.DIRECTIONS[self.current_direction_index]
        angle_to_sign = self.ANGLE_TO_FACE_SIGN_MAP.get(current_direction, 0)
        self.turn_robot(angle_to_sign, False)
        detections = self.scan_for_signs()
        self.turn_robot(-angle_to_sign, False)

        prescriptive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PRESCRIPTIVE_SIGNS}
        prohibitive_cmds = {det['class_name'] for det in detections if det['class_name'] in self.PROHIBITIVE_SIGNS}
        data_items = [det for det in detections if det['class_name'] in self.DATA_ITEMS]

        # Process data items (QR codes, math problems)
        rospy.loginfo("[STEP 1] Processing data items...")
        for item in data_items:
            if item['class_name'] == 'qr_code':
                rospy.loginfo(f"Found QR Code: {item['value']}. Publishing data...")
                self.publish_data({'type': 'QR_CODE', 'value': item['value']})
            elif item['class_name'] == 'math_problem':
                rospy.loginfo("Found Math Problem. Solving and publishing...")
                self.publish_data({'type': 'MATH_PROBLEM', 'value': '2+2=4'})  # Placeholder

        # Scan for available paths
        rospy.loginfo("[STEP 2] Scanning for available paths...")
        available_paths = self.scan_for_available_paths_proactive()

        # Decide navigation based on signs
        rospy.loginfo("[STEP 3] Making navigation decision...")
        final_decision = None

        # Prioritize prescriptive signs
        if 'L' in prescriptive_cmds:
            final_decision = 'left'
            rospy.loginfo("Detected prescriptive sign: LEFT")
        elif 'R' in prescriptive_cmds:
            final_decision = 'right'
            rospy.loginfo("Detected prescriptive sign: RIGHT")
        elif 'F' in prescriptive_cmds:
            final_decision = 'straight'
            rospy.loginfo("Detected prescriptive sign: FORWARD")
        else:
            # No prescriptive signs, choose an available path
            if available_paths['straight']:
                final_decision = 'straight'
                rospy.loginfo("No prescriptive signs, choosing STRAIGHT (available)")
            elif available_paths['right']:
                final_decision = 'right'
                rospy.loginfo("No prescriptive signs, choosing RIGHT (available)")
            elif available_paths['left']:
                final_decision = 'left'
                rospy.loginfo("No prescriptive signs, choosing LEFT (available)")
            else:
                rospy.logerr("No available paths detected. Entering DEAD_END.")
                self._set_state(RobotState.DEAD_END)
                return

        # Check prohibitive signs
        is_prohibited = (final_decision == 'straight' and 'NF' in prohibitive_cmds) or \
                        (final_decision == 'right' and 'NR' in prohibitive_cmds) or \
                        (final_decision == 'left' and 'NL' in prohibitive_cmds)
        if is_prohibited:
            rospy.logwarn(f"Intended action '{final_decision}' is PROHIBITED. Choosing alternative...")
            # Try alternative paths
            alternative_paths = ['straight', 'right', 'left']
            alternative_paths.remove(final_decision)  # Remove prohibited action
            for alt_action in alternative_paths:
                if available_paths[alt_action]:
                    is_alt_prohibited = (alt_action == 'straight' and 'NF' in prohibitive_cmds) or \
                                       (alt_action == 'right' and 'NR' in prohibitive_cmds) or \
                                       (alt_action == 'left' and 'NL' in prohibitive_cmds)
                    if not is_alt_prohibited:
                        final_decision = alt_action
                        rospy.loginfo(f"Switching to alternative action: {final_decision}")
                        break
            else:
                rospy.logerr("No valid alternative paths. Entering DEAD_END.")
                self._set_state(RobotState.DEAD_END)
                return

        # Execute decision
        if final_decision == 'straight':
            rospy.loginfo("[FINAL] Decision: Go STRAIGHT.")
        elif final_decision == 'right':
            rospy.loginfo("[FINAL] Decision: Turn RIGHT.")
            self.turn_robot(90, True)
        elif final_decision == 'left':
            rospy.loginfo("[FINAL] Decision: Turn LEFT.")
            self.turn_robot(-90, True)
        else:
            rospy.logerr("[!!!] DEAD END! No valid decision.")
            self._set_state(RobotState.DEAD_END)
            return

        rospy.loginfo(f"==> Executing {final_decision} and moving to next segment")
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
        _img, contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    try:
        controller = JetBotController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node đã bị ngắt.")
    except Exception as e:
        rospy.logerr(f"Lỗi không xác định: {e}", exc_info=True)

if __name__ == '__main__':
    main()