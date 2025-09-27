#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import json
from enum import Enum
from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from opposite_detector import SimpleOppositeDetector
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt

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
        self.LABEL_TO_DIRECTION_ENUM = {'N': Direction.NORTH, 'E': Direction.EAST,
                                        'S': Direction.SOUTH, 'W': Direction.WEST}
        self.MQTT_BROKER = "localhost"
        self.MQTT_PORT = 1883
        self.MQTT_DATA_TOPIC = "jetbot/corrected_event_data"
        self.PRESCRIPTIVE_SIGNS = {'L', 'R', 'F'}
        self.PROHIBITIVE_SIGNS = {'NL', 'NR', 'NF'}
        self.DATA_ITEMS = {'qr_code', 'math_problem'}
        self.ANGLE_TO_FACE_SIGN_MAP = {Direction.NORTH: 45, Direction.EAST: -45,
                                       Direction.SOUTH: -135, Direction.WEST: 135}
        # Mapping các ký tự biển báo
        self.YOLO_CLASS_MAPPING = {'left': 'L', 'right': 'R', 'forward': 'F',
                                   'no_left': 'NL', 'no_right': 'NR', 'no_forward': 'NF'}

    def initialize_hardware(self):
        try:
            self.robot = Robot()
            rospy.loginfo("Phần cứng JetBot đã khởi tạo.")
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy phần cứng JetBot, dùng Mock. Lỗi: {e}")
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
                rospy.loginfo(f"Chuyển trạng thái: {self.current_state} -> {new_state}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) if image_msg.encoding.endswith('compressed') else np_arr.reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding.lower():
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH, self.HEIGHT))
        except Exception as e:
            rospy.logerr(f"Lỗi chuyển đổi ảnh: {e}")

    def publish_data(self, data):
        try:
            self.mqtt_client.publish(self.MQTT_DATA_TOPIC, json.dumps(data))
            rospy.loginfo(f"Đã publish dữ liệu: {data}")
        except Exception as e:
            rospy.logerr(f"Lỗi khi publish MQTT: {e}")

    def scan_for_signs(self):
        """Scan QR codes và log tất cả các class đã train"""
        detections = []
        if self.latest_image is None:
            return detections

        # QR code
        qr_codes = decode(self.latest_image)
        for qr in qr_codes:
            qr_data = qr.data.decode('utf-8')
            detections.append({'class_name': 'qr_code', 'value': qr_data})
            rospy.loginfo(f"Detected QR code: {qr_data}")

        # Log ra các class đã train
        trained_classes = list(self.YOLO_CLASS_MAPPING.values())
        rospy.loginfo(f"Classes available for detection: {trained_classes}")

        return detections

    def handle_intersection(self):
        rospy.loginfo("Xử lý giao lộ...")
        self.robot.stop()
        detections = self.scan_for_signs()
        # Log các QR code / dữ liệu
        for item in detections:
            rospy.loginfo(f"Item detected: {item}")
        # Quyết định đi thẳng, trái, phải (giữ đơn giản)
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
                rospy.sleep(1.0 / self.VIDEO_FPS)
        self.robot.stop()
        if update_main_direction and degrees % 90 == 0:
            num_turns = round(degrees / 90)
            self.current_direction_index = (self.current_direction_index + num_turns + 4) % 4

    def _get_line_center(self, image, roi_y, roi_h):
        roi = image[roi_y : roi_y + roi_h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(mask)
        roi_height, roi_width = mask.shape
        center_width = int(roi_width * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (roi_width - center_width) // 2
        end_x = start_x + center_width
        cv2.rectangle(focus_mask, (start_x, 0), (end_x, roi_height), 255, -1)
        final_mask = cv2.bitwise_and(mask, focus_mask)
        _, contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD:
            return None
        M = cv2.moments(c)
        return int(M["m10"] / M["m00"]) if M["m00"] > 0 else None

    def correct_course(self, line_center_x):
        error = line_center_x - (self.WIDTH / 2)
        if abs(error) < (self.WIDTH / 2) * self.SAFE_ZONE_PERCENT:
            self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
            return
        adj = (error / (self.WIDTH / 2)) * self.CORRECTION_GAIN
        adj = np.clip(adj, -self.MAX_CORRECTION_ADJ, self.MAX_CORRECTION_ADJ)
        self.robot.set_motors(self.BASE_SPEED + adj, self.BASE_SPEED - adj)

    def run(self):
        rospy.loginfo("Bắt đầu vòng lặp...")
        time.sleep(3)
        self.detector.start_scanning()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.latest_image is None:
                rate.sleep()
                continue
            line_center = self._get_line_center(self.latest_image, self.ROI_Y, self.ROI_H)
            if line_center is not None:
                self.correct_course(line_center)
            # Kiểm tra LiDAR để xử lý giao lộ
            if self.detector.process_detection():
                self.robot.stop()
                self._set_state(RobotState.HANDLING_EVENT)
                self.handle_intersection()
            rate.sleep()
        self.cleanup()

    def initialize_video_writer(self):
        try:
            frame_size = (self.WIDTH, self.HEIGHT)
            self.video_writer = cv2.VideoWriter(self.VIDEO_OUTPUT_FILENAME, 
                                                self.VIDEO_FOURCC, 
                                                self.VIDEO_FPS, frame_size)
        except Exception as e:
            rospy.logerr(f"Lỗi khởi tạo video writer: {e}")
            self.video_writer = None

    def cleanup(self):
        rospy.loginfo("Dừng robot và giải phóng tài nguyên...")
        if hasattr(self, 'robot') and self.robot is not None:
            self.robot.stop()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.stop_scanning()
        if hasattr(self, 'mqtt_client') and self.mqtt_client is not None:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        cv2.destroyAllWindows()

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
