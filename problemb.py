#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
from jetbot import Robot
import onnxruntime as ort
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
from sensor_msgs.msg import LaserScan, Image

from opposite_detector import SimpleOppositeDetector
from map_navigator import MapNavigator

# --- ENUMS ---
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

# --- LỚP CHÍNH ---
class JetBotController:
    DOMAIN = "https://hackathon2025-dev.fpt.edu.vn"
    ENDPOINT = "/api/maps/get_active_map/"

    def __init__(self, token=None, map_type=None):
        rospy.loginfo("Đang khởi tạo JetBot Event-Driven Controller...")
        self.api_token = token
        self.api_map_type = map_type

        self.current_state = None  # Khởi tạo để tránh lỗi

        self.setup_parameters()       # thiết lập tham số trước
        self.initialize_hardware()
        self.initialize_yolo()        # khởi tạo mô hình sau khi thiết lập tham số
        self.initialize_mqtt()
        self.video_writer = None
        self.initialize_video_writer()

        try:
            map_data = self.fetch_map(token, map_type) if token else self.load_local_map()
            self.navigator = MapNavigator(map_data)
            self.current_node_id = self.navigator.start_node
            self.target_node_id = None
            self.planned_path = None
            self.banned_edges = []
            self.plan_initial_route()
            self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
            rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")
        except ValueError as e:
            rospy.logfatal(f"KHỞI TẠO THẤT BẠI: Không thể load map. {e}")
            self.navigator = None
            self._set_state(RobotState.DEAD_END, initial=True)
            rospy.logwarn("Robot sẽ dừng do lỗi tải map.")

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.state_change_time = rospy.get_time()

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
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock object: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_yolo(self):
        try:
            self.yolo_session = ort.InferenceSession(
                self.YOLO_MODEL_PATH,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            rospy.loginfo("Tải mô hình YOLO thành công.")
        except Exception as e:
            rospy.logerr(f"Lỗi khi tải mô hình YOLO: {e}")
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
        session = self.create_session_with_retries()
        try:
            resp = session.get(url, params={"token": token, "map_type": map_type}, timeout=timeout)
            resp.raise_for_status()
            rospy.loginfo("Tải map từ API thành công.")
            return resp.json()
        except Exception as e:
            rospy.logerr(f"Lỗi fetch map: {e}. Sử dụng map cục bộ.")
            return self.load_local_map()

    def load_local_map(self):
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.json"), 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Không thể tải map từ file cục bộ: {e}")

    # Các phương thức khác cần thiết giữ nguyên hoặc bổ sung theo yêu cầu

# Main run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='API token')
    parser.add_argument('--map_type', type=str, help='Map type')
    args = parser.parse_args()

    rospy.init_node('jetbot_controller_node')
    controller = JetBotController(token=args.token, map_type=args.map_type)
    controller.run()
