#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
from enum import Enum
import requests
import argparse
import onnxruntime as ort
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
from sensor_msgs.msg import LaserScan, Image
import pytesseract  # OCR for math problems

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

    def __init__(self, token=None, map_type=None, use_mock=False):
        rospy.loginfo("Khởi tạo JetBot Controller...")
        self.api_token = token
        self.api_map_type = map_type
        self.use_mock = use_mock

        self.current_state = None
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_yolo()
        self.initialize_mqtt()
        self.video_writer = None
        self.initialize_video_writer()

        try:
            if not token or not map_type:
                raise ValueError("Token hoặc map_type không được cung cấp.")
            map_data = self.fetch_map(token, map_type)
            self.navigator = MapNavigator(map_data)
            self.current_node_id = self.navigator.start_node
            self.target_node_id = None
            self.planned_path = None
            self.banned_edges = []
            self.plan_initial_route()
            self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
            rospy.loginfo("Khởi tạo hoàn tất.")
        except ValueError as e:
            rospy.logfatal(f"KHỞI TẠO THẤT BẠI: {e}")
            self.navigator = None
            self._set_state(RobotState.DEAD_END, initial=True)

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.state_change_time = rospy.get_time()

    def setup_parameters(self):
        self.WIDTH, self.HEIGHT = 300, 300
        self.BASE_SPEED = 0.18
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
        self.MAX_CORRECTION_ADJ = 0.12

        self.YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/best.onnx")
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['L', 'R', 'S', 'NL', 'NR', 'NS', 'qr_code', 'math_problem']

        self.PRESCRIPTIVE_SIGNS = {'L', 'R', 'S'}
        self.PROHIBITIVE_SIGNS = {'NL', 'NR', 'NS'}
        self.DATA_ITEMS = {'qr_code', 'math_problem'}
        self.PRESCRIPTIVE_MAP = {'L': 'left', 'R': 'right', 'S': 'straight'}

        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/event_data"

        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1

        self.VIDEO_OUTPUT_FILENAME = 'jetbot_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

    def initialize_hardware(self):
        try:
            if self.use_mock:
                raise Exception("Sử dụng mock robot")
            from jetbot import Robot
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot đã khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock object: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_yolo(self):
        try:
            self.yolo_session = ort.InferenceSession(self.YOLO_MODEL_PATH,
                                                     providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            rospy.loginfo(f"Tải mô hình YOLO từ {os.path.basename(self.YOLO_MODEL_PATH)} thành công.")
        except Exception as e:
            rospy.logerr(f"LỖI: Không thể tải mô hình YOLO. {e}")
            self.yolo_session = None

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

    def initialize_video_writer(self):
        try:
            frame_size = (self.WIDTH, self.HEIGHT)
            self.video_writer = cv2.VideoWriter(self.VIDEO_OUTPUT_FILENAME, self.VIDEO_FOURCC, self.VIDEO_FPS, frame_size)
            if self.video_writer.isOpened():
                rospy.loginfo(f"Bắt đầu ghi video vào '{self.VIDEO_OUTPUT_FILENAME}'")
            else:
                rospy.logerr("Không thể mở file video để ghi.")
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi khi khởi tạo VideoWriter: {e}")
            self.video_writer = None

    def fetch_map(self, token, map_type, timeout=10):
        url = self.DOMAIN.rstrip("/") + self.ENDPOINT
        try:
            resp = requests.get(url, params={"token": token, "map_type": map_type}, timeout=timeout)
            resp.raise_for_status()
            rospy.loginfo("Tải map từ API thành công.")
            return resp.json()
        except Exception as e:
            raise ValueError(f"Lỗi khi lấy map từ API: {e}")

    def plan_initial_route(self):
        if not self.navigator:
            return
        rospy.loginfo(f"Lập kế hoạch từ node {self.navigator.start_node} đến {self.navigator.end_node}...")
        try:
            self.planned_path = self.navigator.find_path(self.navigator.start_node, self.navigator.end_node, self.banned_edges)
            if self.planned_path and len(self.planned_path) > 1:
                self.target_node_id = self.planned_path[1]
                rospy.loginfo(f"Đường đi: {self.planned_path}, đích đầu tiên: {self.target_node_id}")
            else:
                rospy.logerr("Không tìm thấy đường đi hợp lệ!")
                self._set_state(RobotState.DEAD_END)
        except Exception as e:
            rospy.logerr(f"Lỗi lập kế hoạch: {e}")
            self._set_state(RobotState.DEAD_END)

    def _set_state(self, new_state, initial=False):
        if not hasattr(self, 'current_state') or self.current_state != new_state:
            if not initial:
                rospy.loginfo(f"Chuyển trạng thái: {getattr(self, 'current_state', 'NONE').name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            if image_msg.encoding.endswith('compressed'):
                np_arr = np.frombuffer(image_msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def _get_line_center(self, image, roi_y, roi_h):
        if image is None:
            return None
        roi = image[roi_y:roi_y + roi_h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(mask)
        h, w = focus_mask.shape
        center_w = int(w * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (w - center_w) // 2
        end_x = start_x + center_w
        cv2.rectangle(focus_mask, (start_x, 0), (end_x, h), 255, -1)
        final_mask = cv2.bitwise_and(mask, focus_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        self.robot.set_motors(self.BASE_SPEED + adj, self.BASE_SPEED - adj)

    def draw_debug_info(self, image):
        if image is None:
            return None
        debug_frame = image.copy()
        cv2.rectangle(debug_frame, (0, self.ROI_Y), (self.WIDTH - 1, self.ROI_Y + self.ROI_H), (0, 255, 0), 1)
        cv2.rectangle(debug_frame, (0, self.LOOKAHEAD_ROI_Y), (self.WIDTH - 1, self.LOOKAHEAD_ROI_Y + self.LOOKAHEAD_ROI_H),
                      (0, 255, 255), 1)
        cv2.putText(debug_frame, f"State: {self.current_state.name}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return debug_frame

    def _record_frame(self):
        if self.video_writer is not None and self.latest_image is not None:
            debug_frame = self.draw_debug_info(self.latest_image)
            if debug_frame is not None:
                self.video_writer.write(debug_frame)

    def handle_intersection(self):
        if self.latest_image is None or self.yolo_session is None:
            rospy.logwarn("Không có ảnh hoặc YOLO session, bỏ qua xử lý giao lộ.")
            return

        img = cv2.resize(self.latest_image, self.YOLO_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        ort_inputs = {self.yolo_session.get_inputs()[0].name: img_input}
        ort_outs = self.yolo_session.run(None, ort_inputs)

        bboxes, confidences, class_ids = ort_outs
        detections = []
        for bbox, conf, cid in zip(bboxes, confidences, class_ids):
            if conf < self.YOLO_CONF_THRESHOLD:
                continue
            class_name = self.YOLO_CLASS_NAMES[int(cid)]
            detections.append((class_name, bbox, conf))

        for class_name, bbox, conf in detections:
            if class_name == "qr_code":
                qr_roi = self.latest_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                decoded_objs = decode(qr_roi)
                for obj in decoded_objs:
                    qr_data = obj.data.decode("utf-8")
                    rospy.loginfo(f"QR detected: {qr_data}")
                    self.mqtt_client.publish(self.MQTT_DATA_TOPIC, json.dumps({"type": "qr_code", "data": qr_data}))

            elif class_name == "math_problem":
                math_roi = self.latest_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                gray = cv2.cvtColor(math_roi, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 7').strip().replace(" ", "")
                rospy.loginfo(f"Math detected: {text}")
                try:
                    result = eval(text)
                except Exception:
                    result = None
                self.mqtt_client.publish(self.MQTT_DATA_TOPIC, json.dumps({"type": "math_problem", "expr": text, "result": result}))

        if self.planned_path and self.current_node_id in self.planned_path:
            idx = self.planned_path.index(self.current_node_id)
            if idx + 1 < len(self.planned_path):
                self.target_node_id = self.planned_path[idx + 1]
            else:
                self._set_state(RobotState.GOAL_REACHED)
                return

        move_direction = self.navigator.get_next_direction_label(self.current_node_id, self.planned_path)
        rospy.loginfo(f"Di chuyển tiếp theo: {move_direction}")
        if move_direction == "left":
            self.robot.set_motors(-self.TURN_SPEED, self.TURN_SPEED)
            time.sleep(self.TURN_DURATION_90_DEG)
        elif move_direction == "right":
            self.robot.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
            time.sleep(self.TURN_DURATION_90_DEG)
        elif move_direction == "straight":
            self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
            time.sleep(0.5)
        self.robot.stop()
        self._set_state(RobotState.LEAVING_INTERSECTION)

    def run(self):
        rospy.loginfo("Bắt đầu chạy...")
        time.sleep(3)

        if self.current_state == RobotState.DEAD_END:
            rospy.logfatal("Robot không thể chạy vì lỗi load map.")
            self.cleanup()
            return

        self.detector.start_scanning()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.current_state == RobotState.WAITING_FOR_LINE:
                self.robot.stop()
                if self.latest_image is None:
                    rate.sleep()
                    continue
                lookahead_line = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                execution_line = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if lookahead_line is not None and execution_line is not None:
                    self._set_state(RobotState.DRIVING_STRAIGHT)

            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                if self.latest_image is None:
                    self.robot.stop()
                    rate.sleep()
                    continue
                if self.detector.process_detection():
                    self.robot.stop()
                    time.sleep(0.5)
                    self.current_node_id = self.target_node_id
                    if self.current_node_id == self.navigator.end_node:
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()
                    continue
                lookahead_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                if lookahead_center is None:
                    self._set_state(RobotState.REACQUIRING_LINE)
                    continue
                self.correct_course(lookahead_center)

            elif self.current_state == RobotState.REACQUIRING_LINE:
                self.robot.set_motors(self.BASE_SPEED / 2, self.BASE_SPEED / 2)
                lookahead_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                if lookahead_center is not None:
                    self._set_state(RobotState.DRIVING_STRAIGHT)

            elif self.current_state == RobotState.GOAL_REACHED:
                self.robot.stop()
                rospy.loginfo("Mục tiêu đã đạt được!")
                self.cleanup()
                break

            self._record_frame()
            rate.sleep()

    def cleanup(self):
        rospy.loginfo("Dọn dẹp và tắt robot...")
        if self.robot is not None:
            self.robot.stop()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        rospy.loginfo("Kết thúc JetBot Controller.")

    def initialize_yolo(self):
        try:
            self.yolo_session = ort.InferenceSession(self.YOLO_MODEL_PATH, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            rospy.loginfo(f"Tải mô hình YOLO thành công.")
        except Exception as e:
            rospy.logerr(f"Lỗi khi tải YOLO: {e}")
            self.yolo_session = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--map_type", type=str, required=True)
    parser.add_argument("--mock", action="store_true", help="Dùng Mock robot để test")
    args = parser.parse_args()

    rospy.init_node('jetbot_controller', anonymous=True)
    controller = JetBotController(token=args.token, map_type=args.map_type, use_mock=args.mock)
    try:
        controller.run()
    except rospy.ROSInterruptException:
        controller.cleanup()
