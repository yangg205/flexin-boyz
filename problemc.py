#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import onnxruntime as ort
from enum import Enum
from jetbot import Robot
from sensor_msgs.msg import Image
import pytesseract
import time
import threading

class RobotState(Enum):
    WAITING = 0
    DRIVING = 1
    STOPPED = 2

class JetBotController:
    def __init__(self, onnx_model_path):
        rospy.init_node('jetbot_controller_node')
        self.robot = Robot()

        try:
            self.session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            rospy.loginfo("ONNX model loaded")
        except Exception as e:
            rospy.logerr(f"Error loading ONNX model: {e}")
            exit(1)

        self.latest_image = None
        self.lock = threading.Lock()

        self.WIDTH, self.HEIGHT = 300, 300
        self.ROI_Y = int(self.HEIGHT * 0.85)
        self.ROI_H = int(self.HEIGHT * 0.15)
        self.ROI_CENTER_WIDTH_PERCENT = 0.5
        self.LINE_COLOR_LOWER = np.array([0, 0, 0])
        self.LINE_COLOR_UPPER = np.array([180, 255, 75])
        self.BASE_SPEED = 0.16
        self.CORRECTION_GAIN = 0.5
        self.MAX_CORRECTION_ADJ = 0.12
        self.SAFE_ZONE_PERCENT = 0.3
        self.SCAN_PIXEL_THRESHOLD = 100

        self.last_infer_time = 0
        self.infer_rate = 3  # FPS
        self.min_infer_interval = 1.0 / self.infer_rate

        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)

    def camera_callback(self, data):
        try:
            img = np.frombuffer(data.data, np.uint8).reshape(data.height, data.width, -1)
            if 'rgb' in data.encoding.lower():
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_small = cv2.resize(img, (self.WIDTH, self.HEIGHT))
            with self.lock:
                self.latest_image = img_small
        except Exception as e:
            rospy.logerr(f"Camera callback error: {e}")

    def preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, image):
        input_tensor = self.preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        return outputs[0]

    def extract_text(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 7').strip()
        return text.upper()

    def map_class(self, cls):
        classes = ['X', 'NS', 'L']
        return classes[cls] if cls < len(classes) else "UNKNOWN"

    def yolo_detect(self, image):
        now = time.time()
        if now - self.last_infer_time < self.min_infer_interval:
            return []
        self.last_infer_time = now
        outputs = self.infer(image)
        detections = []
        for det in outputs:
            x1,y1,x2,y2,conf,cls = det
            if conf < 0.6:
                continue
            label = self.map_class(int(cls))
            bbox = (x1,y1,x2,y2)
            text = self.extract_text(image, bbox)
            detections.append({"label": label, "bbox": bbox, "text": text, "confidence": conf})
        return detections

    def get_line_center(self, image, roi_y, roi_h):
        roi = image[roi_y:roi_y+roi_h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(mask)
        h, w = focus_mask.shape
        center_width = int(w * self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (w - center_width) // 2
        end_x = start_x + center_width
        cv2.rectangle(focus_mask, (start_x,0), (end_x,h), 255, -1)
        final_mask = cv2.bitwise_and(mask, focus_mask)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.SCAN_PIXEL_THRESHOLD:
            return None
        M = cv2.moments(c)
        return int(M['m10'] / M['m00']) if M['m00'] > 0 else None

    def correct_course(self, line_center_x):
        error = line_center_x - (self.WIDTH / 2)
        if abs(error) < (self.WIDTH / 2) * self.SAFE_ZONE_PERCENT:
            self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
            return
        adj = (error / (self.WIDTH / 2)) * self.CORRECTION_GAIN
        adj = max(min(adj, self.MAX_CORRECTION_ADJ), -self.MAX_CORRECTION_ADJ)
        self.robot.set_motors(self.BASE_SPEED + adj, self.BASE_SPEED - adj)

    def move_robot(self, direction):
        directions = ['N', 'E', 'S', 'W']
        if direction not in directions:
            rospy.logwarn(f"Invalid direction {direction}, stopping robot.")
            self.robot.stop()
            return
        rospy.loginfo(f"Moving {direction}")
        self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.lock:
                image = self.latest_image.copy() if self.latest_image is not None else None
            if image is None:
                rospy.loginfo("Waiting for camera image...")
                rate.sleep()
                continue
            line_center = self.get_line_center(image, self.ROI_Y, self.ROI_H)
            if line_center is None:
                rospy.loginfo("No line detected, stopping robot.")
                self.robot.stop()
                rate.sleep()
                continue
            self.correct_course(line_center)
            detections = self.yolo_detect(image)
            stop_flag = False
            for det in detections:
                label, text = det['label'], det['text']
                rospy.loginfo(f"Detected sign: {label} with text: {text}")
                if label == 'NS':
                    rospy.loginfo(f"Direction {text} forbidden, stopping robot.")
                    self.robot.stop()
                    stop_flag = True
                    break
                elif label == 'X':
                    rospy.loginfo(f"Following direction {text}")
                    self.move_robot(text)
                elif label == 'L':
                    rospy.loginfo("Destination reached, stopping robot.")
                    self.robot.stop()
                    stop_flag = True
                    break
            if not stop_flag:
                self.robot.set_motors(self.BASE_SPEED, self.BASE_SPEED)
            rate.sleep()

if __name__ == "__main__":
    model_path = "/home/jetbot/flexin-boyz/model/best.onnx"  # Đường dẫn cứng
    controller = JetBotController(model_path)
    controller.run()
