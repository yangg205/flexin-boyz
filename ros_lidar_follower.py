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
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()
        self.video_writer = None
        self.initialize_video_writer()

        # Load map
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
            rospy.logwarn("Robot sẽ dừng ngay khi chạy do lỗi tải map.")

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.state_change_time = rospy.get_time()

    # ----------------- SETUP -----------------
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

        self.YOLO_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['L', 'R', 'S', 'NL', 'NR', 'NS', 'qr_code', 'math_problem']

        self.PRESCRIPTIVE_SIGNS = {'L', 'R', 'S'}
        self.PROHIBITIVE_SIGNS = {'NL', 'NR', 'NS'}
        self.DATA_ITEMS = {'qr_code', 'math_problem'}

        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/event_data"

        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.MAP_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.json")

        self.VIDEO_OUTPUT_FILENAME = 'jetbot_run.avi'
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')

    # ----------------- HARDWARE -----------------
    def initialize_hardware(self):
        try:
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot (động cơ) đã được khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, sử dụng Mock object. Lỗi: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_yolo(self):
        try:
            self.yolo_session = ort.InferenceSession(self.YOLO_MODEL_PATH, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
            rospy.loginfo(f"Tải mô hình YOLO từ {os.path.basename(self.YOLO_MODEL_PATH)} thành công.")
        except Exception as e:
            rospy.logerr(f"LỖI: Không thể tải mô hình YOLO. {e}")
            self.yolo_session = None

    def initialize_mqtt(self):
        self.mqtt_client = mqtt.Client()
        def on_connect(client, userdata, flags, rc):
            rospy.loginfo(f"Kết nối MQTT: {'Thành công' if rc == 0 else 'Thất bại (rc: '+str(rc)+')'}")
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
            if not self.video_writer.isOpened():
                rospy.logerr("Không thể mở file video để ghi.")
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi khi khởi tạo VideoWriter: {e}")
            self.video_writer = None

    # ----------------- MAP -----------------
    def create_session_with_retries(self, total_retries=3, backoff_factor=0.5, status_forcelist=(429,500,502,503,504)):
        s = requests.Session()
        retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def fetch_map(self, token, map_type, timeout=10):
        if not token or not map_type:
            return self.load_local_map()
        params = {"token": token, "map_type": map_type}
        url = self.DOMAIN.rstrip("/") + self.ENDPOINT
        session = self.create_session_with_retries()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            rospy.logerr(f"Lỗi fetch map: {e}. Sử dụng map cục bộ.")
            return self.load_local_map()

    def load_local_map(self):
        try:
            with open(self.MAP_FILE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Không thể tải map từ file cục bộ: {e}")

    def plan_initial_route(self):
        if self.navigator is None: return
        try:
            self.planned_path = self.navigator.find_path(self.navigator.start_node, self.navigator.end_node, self.banned_edges)
            if self.planned_path and len(self.planned_path) > 1:
                self.target_node_id = self.planned_path[1]
            else:
                self._set_state(RobotState.DEAD_END)
        except:
            self._set_state(RobotState.DEAD_END)

    def _set_state(self, new_state, initial=False):
        if not hasattr(self, 'current_state') or self.current_state != new_state:
            if not initial: rospy.loginfo(f"State: {getattr(self,'current_state','NONE').name} -> {new_state.name}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    # ----------------- CAMERA -----------------
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

    # ----------------- LINE FOLLOW -----------------
    def _get_line_center(self, image, roi_y, roi_h):
        if image is None: return None
        roi = image[roi_y:roi_y+roi_h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(mask)
        roi_height, roi_width = focus_mask.shape
        center_width = int(roi_width * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (roi_width - center_width) // 2
        end_x = start_x + center_width
        cv2.rectangle(focus_mask, (start_x,0),(end_x,roi_height),255,-1)
        final_mask = cv2.bitwise_and(mask, focus_mask)
        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD: return None
        M = cv2.moments(c)
        if M["m00"] > 0: return int(M["m10"]/M["m00"])
        return None

    def correct_course(self, line_center_x):
        error = line_center_x - (self.WIDTH/2)
        if abs(error) < (self.WIDTH/2)*self.SAFE_ZONE_PERCENT:
            self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
            return
        adj = (error/(self.WIDTH/2))*self.CORRECTION_GAIN
        adj = np.clip(adj, -self.MAX_CORRECTION_ADJ, self.MAX_CORRECTION_ADJ)
        self.robot.set_motors(self.BASE_SPEED+adj, self.BASE_SPEED-adj)

    def draw_debug_info(self, image):
        if image is None: return None
        debug_frame = image.copy()
        cv2.rectangle(debug_frame, (0,self.ROI_Y),(self.WIDTH-1,self.ROI_Y+self.ROI_H),(0,255,0),1)
        cv2.rectangle(debug_frame, (0,self.LOOKAHEAD_ROI_Y),(self.WIDTH-1,self.LOOKAHEAD_ROI_Y+self.LOOKAHEAD_ROI_H),(0,255,255),1)
        cv2.putText(debug_frame,f"State: {self.current_state.name}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        return debug_frame

    def _record_frame(self):
        if self.video_writer and self.latest_image is not None:
            debug_frame = self.draw_debug_info(self.latest_image)
            if debug_frame is not None: self.video_writer.write(debug_frame)

    # ----------------- YOLO + QR + Math -----------------
    def yolo_detect_objects(self, image):
        if self.yolo_session is None or image is None: return []
        img_resized = cv2.resize(image, self.YOLO_INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32)/255.0
        img_transposed = np.transpose(img_normalized,(2,0,1))
        img_input = np.expand_dims(img_transposed,0).astype(np.float32)
        input_name = self.yolo_session.get_inputs()[0].name
        outputs = self.yolo_session.run(None, {input_name: img_input})
        results=[]
        for det in outputs[0]:
            x1,y1,x2,y2,conf,cls_id = det
            if conf<self.YOLO_CONF_THRESHOLD: continue
            label = self.YOLO_CLASS_NAMES[int(cls_id)]
            scale_x = image.shape[1]/self.YOLO_INPUT_SIZE[0]
            scale_y = image.shape[0]/self.YOLO_INPUT_SIZE[1]
            bbox = (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))
            results.append({"label":label,"bbox":bbox,"confidence":conf})
        return results

    def solve_math_problem(self, image, bbox):
        try:
            x1,y1,x2,y2 = bbox
            roi = image[y1:y2,x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            import pytesseract
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            # Dọn chuỗi
            text = text.replace('\n','').replace(' ','')
            result = str(eval(text)) if text else None
            return result
        except:
            return None

    def process_detected_objects(self):
        if self.latest_image is None: return []
        detected = self.yolo_detect_objects(self.latest_image)
        results=[]
        for obj in detected:
            label = obj['label']
            bbox = obj['bbox']
            if label=='qr_code':
                x1,y1,x2,y2 = bbox
                roi = self.latest_image[y1:y2, x1:x2]
                qr_codes = decode(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                for qr in qr_codes:
                    results.append({"type":"qr","value":qr.data.decode("utf-8")})
            elif label=='math_problem':
                res = self.solve_math_problem(self.latest_image, bbox)
                if res: results.append({"type":"math","value":res})
        return results

    # ----------------- RUN -----------------
    def run(self):
        rospy.loginfo("Bắt đầu vòng lặp. Đợi 3 giây...")
        time.sleep(3)
        if self.current_state==RobotState.DEAD_END:
            rospy.logfatal("Robot không thể bắt đầu do lỗi Load Map. Dừng ngay lập tức.")
            self.cleanup()
            return
        rospy.loginfo("Hành trình bắt đầu!")
        self.detector.start_scanning()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.current_state==RobotState.WAITING_FOR_LINE:
                self.robot.stop()
                if self.latest_image is None:
                    rate.sleep()
                    continue
                lookahead_line = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                execution_line = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if lookahead_line is not None and execution_line is not None:
                    self._set_state(RobotState.DRIVING_STRAIGHT)

            elif self.current_state==RobotState.DRIVING_STRAIGHT:
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
                lookahead_line_center = self._get_line_center(self.latest_image, self.LOOKAHEAD_ROI_Y, self.LOOKAHEAD_ROI_H)
                if lookahead_line_center is None:
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                    continue
                execution_line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if execution_line_center is not None:
                    self.correct_course(execution_line_center)
                else:
                    self.robot.stop()

            elif self.current_state==RobotState.APPROACHING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
                if rospy.get_time()-self.state_change_time>self.INTERSECTION_APPROACH_DURATION:
                    self.robot.stop()
                    time.sleep(0.5)
                    self
