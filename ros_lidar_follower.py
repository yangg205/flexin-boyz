#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from enum import Enum
from jetbot import Robot
import torch
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
from sensor_msgs.msg import LaserScan, Image
from opposite_detector import SimpleOppositeDetector

class RobotState(Enum):
    WAITING_FOR_LINE = 0
    DRIVING_STRAIGHT = 1
    HANDLING_EVENT = 2
    DEAD_END = 3

class JetBotController:
    def __init__(self):
        rospy.loginfo("Khởi tạo JetBot Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_yolo()
        self.initialize_mqtt()
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)

    def setup_parameters(self):
        self.WIDTH, self.HEIGHT = 300, 300
        self.BASE_SPEED = 0.16
        self.TURN_SPEED = 0.2
        self.ROI_Y = int(self.HEIGHT * 0.85)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.ROI_CENTER_WIDTH_PERCENT = 0.5
        self.LINE_COLOR_LOWER = np.array([0, 0, 0])
        self.LINE_COLOR_UPPER = np.array([180, 255, 75])
        self.SAFE_ZONE_PERCENT = 0.3
        self.CORRECTION_GAIN = 0.5
        self.MAX_CORRECTION_ADJ = 0.12
        self.SCAN_PIXEL_THRESHOLD = 100
        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        self.YOLO_MODEL_PATH = "home/jetbot/flexin-boyz/model/best.pt"

    def initialize_hardware(self):
        try:
            self.robot = Robot()
        except:
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_yolo(self):
        rospy.loginfo(f"Tải YOLO model từ {self.YOLO_MODEL_PATH}...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.YOLO_MODEL_PATH)
        self.model.conf = 0.6  # threshold

    def initialize_mqtt(self):
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(self.MQTT_BROKER, self.MQTT_PORT, 60)
        self.mqtt_client.loop_start()

    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            self.current_state = new_state
            self.state_change_time = rospy.get_time()
            if not initial:
                rospy.loginfo(f"Chuyển trạng thái: {new_state.name}")

    def camera_callback(self, image_msg):
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển ảnh: {e}")

    def _get_line_center(self, image, roi_y, roi_h):
        roi = image[roi_y:roi_y+roi_h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(mask)
        center_width = int(mask.shape[1]*self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (mask.shape[1]-center_width)//2
        focus_mask[:, start_x:start_x+center_width] = 255
        final_mask = cv2.bitwise_and(mask, focus_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD: return None
        M = cv2.moments(c)
        if M["m00"] > 0: return int(M["m10"]/M["m00"])
        return None

    def correct_course(self, line_center_x):
        error = line_center_x - (self.WIDTH / 2)
        if abs(error) < (self.WIDTH / 2)*self.SAFE_ZONE_PERCENT:
            self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
            return
        adj = np.clip((error/(self.WIDTH/2))*self.CORRECTION_GAIN, -self.MAX_CORRECTION_ADJ, self.MAX_CORRECTION_ADJ)
        self.robot.set_motors(self.BASE_SPEED+adj, self.BASE_SPEED-adj)

    def detect_with_yolo(self, image):
        if image is None: return []
        results = self.model(image)
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = [int(b) for b in box]
            detections.append({
                'class_name': self.model.names[int(cls)],
                'confidence': float(conf),
                'box': [x1, y1, x2, y2]
            })
        return detections

    def handle_event(self):
        rospy.loginfo("Xử lý sự kiện...")
        detections = self.detect_with_yolo(self.latest_image)
        
        # Log tất cả các object mà camera nhìn thấy
        if detections:
            rospy.loginfo("Camera nhìn thấy các object sau:")
            for det in detections:
                rospy.loginfo(f" - {det['class_name']} (confidence: {det['confidence']:.2f}) "
                            f"box: {det['box']}")
        else:
            rospy.loginfo("Không phát hiện object nào.")
        
        # Vẫn giữ logic gửi MQTT cho QR / Math problem
        for det in detections:
            if det['class_name'] == 'qr_code':
                rospy.loginfo("Found QR. Publishing...")
                self.mqtt_client.publish(self.MQTT_DATA_TOPIC, '{"type":"QR","value":"simulated"}')
            elif det['class_name'] == 'math_problem':
                rospy.loginfo("Found Math problem. Publishing...")
                self.mqtt_client.publish(self.MQTT_DATA_TOPIC, '{"type":"MATH","value":"2+2=4"}')


    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.latest_image is None:
                self.robot.stop()
                rate.sleep()
                continue
            if self.current_state == RobotState.WAITING_FOR_LINE:
                line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if line_center is not None:
                    self._set_state(RobotState.DRIVING_STRAIGHT)
            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
                if line_center is not None:
                    self.correct_course(line_center)
                else:
                    self._set_state(RobotState.HANDLING_EVENT)
            elif self.current_state == RobotState.HANDLING_EVENT:
                self.robot.stop()
                self.handle_event()
                self._set_state(RobotState.DEAD_END)
            elif self.current_state == RobotState.DEAD_END:
                self.robot.stop()
                break
            rate.sleep()

def main():
    rospy.init_node('jetbot_controller_node', anonymous=True)
    try:
        controller = JetBotController()
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node bị ngắt.")

if __name__ == "__main__":
    main()
