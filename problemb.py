#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
import math
import requests
import re
from enum import Enum
from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from map_navigator import MapNavigator
from opposite_detector import SimpleOppositeDetector
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
import base64
import pytesseract
import onnxruntime as ort

class RobotState(Enum):
    WAITING_FOR_FLAG = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    HANDLING_EVENT = 3
    READING_INFORMATION = 4
    LEAVING_INTERSECTION = 5
    REACQUIRING_LINE = 6
    GOAL_REACHED = 7
    OUT_OF_BOUNDS = 8

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class ProblemBCompleteController:
    def __init__(self):
        rospy.loginfo("🚀 Khởi tạo Problem B Complete Controller...")
        self.setup_parameters()
        self.robot = Robot()
        self.bridge = CvBridge()

        # Navigation and map
        self.navigator = MapNavigator("/home/jetbot/hackathon/map.json")
        self.opposite_detector = SimpleOppositeDetector()

        # YOLO model initialization
        self.initialize_yolo()

        # State management
        self.current_state = RobotState.WAITING_FOR_FLAG
        self.current_node_id = None
        self.path = []
        self.load_nodes = []
        self.visited_loads = set()
        self.start_time = None
        self.goal_reached_time = None

        # ROS subscribers
        self.camera_sub = rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # Control variables
        self.latest_frame = None
        self.lidar_data = None
        self.flag_detected = False
        self.last_qr_node = None

        # MQTT for server communication
        self.setup_mqtt()

        # PID controller for line following
        self.prev_error = 0
        self.integral = 0

        rospy.loginfo("✅ Problem B Complete Controller initialized!")

    def setup_parameters(self):
        """Setup parameters for Problem B with YOLO"""
        # Line following parameters  
        self.base_speed = 0.18
        self.kp = 0.004
        self.ki = 0.0001
        self.kd = 0.001

        self.line_roi_height = 160
        self.line_roi_width = 224
        self.line_roi_top = 200

        # HSV thresholds
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

        # YOLO parameters
        self.YOLO_MODEL_PATH = "/home/jetbot/hackathon/models/best.onnx"
        self.YOLO_CONF_THRESHOLD = 0.6
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['N', 'E', 'W', 'S', 'NN', 'NE', 'NW', 'NS', 'math']

        # Information types
        self.INFORMATION_SIGNS = {'math'}  # Math problems need YOLO detection

        # Timing parameters
        self.intersection_threshold = 0.3
        self.stop_duration = 5.0
        self.max_time = 300  # 5 minutes
        self.reading_timeout = 15.0  # 15 seconds to read information

        # Recovery parameters
        self.max_line_lost = 10
        self.recovery_speed = 0.08

        # Server communication
        self.server_api_endpoint = "http://hackathon-server.local/api/information"
        self.mqtt_broker = "hackathon-server.local"
        self.mqtt_port = 1883
        self.mqtt_topic = "jetbot/information"

    def initialize_yolo(self):
        """Initialize YOLO model for information sign detection"""
        try:
            self.yolo_session = ort.InferenceSession(
                self.YOLO_MODEL_PATH, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            rospy.loginfo("✅ YOLO model loaded successfully")
        except Exception as e:
            rospy.logerr(f"❌ Cannot load YOLO model: {e}")
            self.yolo_session = None

    def setup_mqtt(self):
        """Setup MQTT client for server communication"""
        try:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_publish = self.on_mqtt_publish
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            rospy.loginfo("✅ MQTT client connected")
        except Exception as e:
            rospy.logerr(f"❌ MQTT connection failed: {e}")
            self.mqtt_client = None

    def on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            rospy.loginfo("📡 Connected to MQTT broker")
        else:
            rospy.logerr(f"❌ Failed to connect to MQTT broker: {rc}")

    def on_mqtt_publish(self, client, userdata, mid):
        rospy.loginfo(f"📤 Message published with id: {mid}")

    def camera_callback(self, msg):
        """Enhanced camera callback"""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"❌ Camera error: {e}")

    def lidar_callback(self, msg):
        """Enhanced lidar callback with flag detection"""
        self.lidar_data = msg

        # Flag detection
        if not self.flag_detected and self.current_state == RobotState.WAITING_FOR_FLAG:
            if self.latest_frame is not None:
                if self.detect_flag_color(self.latest_frame):
                    self.flag_detected = True
                    self.start_time = time.time()
                    rospy.loginfo("🚩 Flag detected! Starting mission...")

    def detect_flag_color(self, frame):
        """Detect red flag color"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_pixels = cv2.countNonZero(red_mask)
        threshold = frame.shape[0] * frame.shape[1] * 0.01
        return red_pixels > threshold

    def detect_line(self, frame):
        """Enhanced line detection with PID"""
        if frame is None:
            return 0, False

        height, width = frame.shape[:2]
        roi_top = self.line_roi_top
        roi_bottom = roi_top + self.line_roi_height
        roi_left = (width - self.line_roi_width) // 2
        roi_right = roi_left + self.line_roi_width

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)

        # Enhanced noise filtering
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find center using contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    line_center = cx - self.line_roi_width // 2
                    return line_center, True

        return 0, False

    def execute_line_following(self, line_center, line_detected):
        """Enhanced PID-based line following"""
        if not line_detected:
            self.robot.left_motor.value = self.recovery_speed
            self.robot.right_motor.value = self.recovery_speed
            return

        # PID controller
        error = -line_center
        self.integral += error
        derivative = error - self.prev_error

        pid_output = (self.kp * error + 
                     self.ki * self.integral + 
                     self.kd * derivative)

        left_speed = self.base_speed - pid_output
        right_speed = self.base_speed + pid_output

        left_speed = max(0.05, min(0.35, left_speed))
        right_speed = max(0.05, min(0.35, right_speed))

        self.robot.left_motor.value = left_speed
        self.robot.right_motor.value = right_speed

        self.prev_error = error

    def detect_intersection(self):
        """Multi-sensor intersection detection"""
        lidar_detect = False
        camera_detect = False

        if self.lidar_data is not None:
            lidar_detect = self.opposite_detector.detect_intersection(self.lidar_data)

        if self.latest_frame is not None:
            camera_detect = self.check_camera_intersection(self.latest_frame)

        return lidar_detect and camera_detect

    def check_camera_intersection(self, frame):
        """Camera-based intersection detection"""
        roi_top = self.line_roi_top - 50
        roi_bottom = roi_top + self.line_roi_height + 100
        roi_left = (frame.shape[1] - self.line_roi_width - 100) // 2  
        roi_right = roi_left + self.line_roi_width + 100

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)

        lines = cv2.HoughLinesP(mask, 1, np.pi/180, 
                               threshold=30, minLineLength=20, maxLineGap=10)

        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2-y1, x2-x1) * 180 / math.pi

                if abs(angle) < 30 or abs(angle) > 150:
                    horizontal_lines += 1
                elif 60 < abs(angle) < 120:
                    vertical_lines += 1

            return horizontal_lines >= 1 and vertical_lines >= 1

        return False

    def detect_with_yolo(self, image):
        """YOLO detection for information signs"""
        if self.yolo_session is None:
            return []

        try:
            original_height, original_width = image.shape[:2]
            img_resized = cv2.resize(image, self.YOLO_INPUT_SIZE)
            img_data = np.array(img_resized, dtype=np.float32) / 255.0
            img_data = np.transpose(img_data, (2, 0, 1))  # HWC to CHW
            input_tensor = np.expand_dims(img_data, axis=0)  # Add batch dimension

            input_name = self.yolo_session.get_inputs()[0].name
            outputs = self.yolo_session.run(None, {input_name: input_tensor})

            # Process YOLO outputs
            predictions = np.squeeze(outputs[0]).T
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > self.YOLO_CONF_THRESHOLD, :]
            scores = scores[scores > self.YOLO_CONF_THRESHOLD]

            if predictions.shape[0] == 0:
                return []

            class_ids = np.argmax(predictions[:, 4:], axis=1)
            x, y, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]

            # Convert to original image coordinates
            x_scale = original_width / self.YOLO_INPUT_SIZE[0]
            y_scale = original_height / self.YOLO_INPUT_SIZE[1]

            x1 = (x - w / 2) * x_scale
            y1 = (y - h / 2) * y_scale
            x2 = (x + w / 2) * x_scale
            y2 = (y + h / 2) * y_scale

            boxes = np.column_stack((x1, y1, x2, y2)).tolist()

            # NMS
            indices = self.numpy_nms(np.array(boxes), scores, 0.45)

            final_detections = []
            for i in indices.flatten():
                final_detections.append({
                    'class_name': self.YOLO_CLASS_NAMES[class_ids[i]],
                    'confidence': float(scores[i]),
                    'box': [int(coord) for coord in boxes[i]]
                })

            return final_detections

        except Exception as e:
            rospy.logerr(f"❌ YOLO detection error: {e}")
            return []

    def numpy_nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression implementation"""
        if len(boxes) == 0:
            return np.array([])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def read_qr_node(self, frame):
        """QR code reading with error handling"""
        try:
            decoded_objects = decode(frame)
            for obj in decoded_objects:
                node_id = int(obj.data.decode('utf-8'))
                if node_id in self.navigator.nodes_data:
                    return node_id
        except Exception as e:
            rospy.logwarn(f"⚠️ QR reading error: {e}")
        return None

    def update_current_node(self):
        """Update current node with QR validation"""
        if self.latest_frame is not None:
            qr_node = self.read_qr_node(self.latest_frame)
            if qr_node is not None and qr_node != self.last_qr_node:
                rospy.loginfo(f"📍 QR node detected: {qr_node}")
                self.current_node_id = qr_node
                self.last_qr_node = qr_node

    def find_load_nodes(self):
        """Find all Load nodes in the map"""
        load_nodes = []
        for node_id, node_data in self.navigator.nodes_data.items():
            if node_data.get('type') == 'Load':
                load_nodes.append(node_id)
        return load_nodes

    def plan_optimal_route(self):
        """Plan optimal route to visit all Load nodes and then End"""
        self.load_nodes = self.find_load_nodes()
        rospy.loginfo(f"🎯 Found Load nodes: {self.load_nodes}")

        if not self.load_nodes:
            rospy.logwarn("⚠️ No Load nodes found!")
            # Go directly to End
            self.path = self.navigator.find_path_from_to(
                self.navigator.start_node, 
                self.navigator.end_node
            )
            return

        # Simple approach: visit all Load nodes in order, then go to End
        current_pos = self.navigator.start_node
        full_path = [current_pos]

        for load_node in self.load_nodes:
            path_segment = self.navigator.find_path_from_to(current_pos, load_node)
            if path_segment:
                full_path.extend(path_segment[1:])  # Skip first node (duplicate)
                current_pos = load_node
            else:
                rospy.logerr(f"❌ Cannot find path to Load node {load_node}")

        # Finally, go to End
        end_path = self.navigator.find_path_from_to(current_pos, self.navigator.end_node)
        if end_path:
            full_path.extend(end_path[1:])

        self.path = full_path
        rospy.loginfo(f"🗺️ Complete route: {self.path}")

    def is_at_load_node(self):
        """Check if currently at a Load node"""
        if self.current_node_id is None:
            return False

        node_data = self.navigator.nodes_data.get(self.current_node_id)
        return node_data and node_data.get('type') == 'Load'

    def read_information_comprehensive(self, frame):
        """Comprehensive information reading using YOLO + OCR + QR"""
        if frame is None:
            return None

        information_data = {
            'type': None,
            'content': None,
            'confidence': 0.0
        }

        try:
            # Step 1: Use YOLO to detect information signs
            detections = self.detect_with_yolo(frame)

            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                box = detection['box']

                if class_name in self.INFORMATION_SIGNS:
                    # Extract ROI for the detected sign
                    x1, y1, x2, y2 = box
                    roi = frame[y1:y2, x1:x2]

                    if class_name == 'math':
                        # Handle math problems
                        math_content = self.solve_math_problem(roi)
                        if math_content:
                            information_data = {
                                'type': 'math_problem',
                                'content': math_content,
                                'confidence': confidence
                            }
                            rospy.loginfo(f"🧮 Math problem solved: {math_content}")
                            return information_data

            # Step 2: Try OCR on ES corner (fallback)
            height, width = frame.shape[:2]
            roi_x = int(width * 0.6)
            roi_y = int(height * 0.6)
            roi_w = int(width * 0.4)
            roi_h = int(height * 0.4)

            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

            # Try QR code first
            qr_content = self.read_qr_from_roi(roi)
            if qr_content:
                information_data = {
                    'type': 'qr_code',
                    'content': qr_content,
                    'confidence': 1.0
                }
                rospy.loginfo(f"📱 QR code read: {qr_content}")
                return information_data

            # Try OCR text
            text_content = self.read_text_from_roi(roi)
            if text_content:
                information_data = {
                    'type': 'text_information',
                    'content': text_content,
                    'confidence': 0.8
                }
                rospy.loginfo(f"📖 Text read: {text_content}")
                return information_data

        except Exception as e:
            rospy.logerr(f"❌ Information reading error: {e}")

        return None

    def solve_math_problem(self, roi_image):
        """Solve math problem from detected region"""
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR to extract math expression
            text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789+-*/=().')
            text = text.strip()

            if len(text) == 0:
                return None

            rospy.loginfo(f"🧮 Extracted math text: {text}")

            # Simple math solver (extend as needed)
            try:
                # Remove '=' and everything after it if present
                if '=' in text:
                    expression = text.split('=')[0].strip()
                else:
                    expression = text

                # Evaluate safe mathematical expression
                # Note: In production, use a safer math parser
                result = eval(expression)
                return f"{expression}={result}"
            except:
                # If eval fails, return the extracted text
                return text

        except Exception as e:
            rospy.logerr(f"❌ Math solving error: {e}")
            return None

    def read_qr_from_roi(self, roi_image):
        """Read QR code from ROI"""
        try:
            decoded_objects = decode(roi_image)
            if decoded_objects:
                return decoded_objects[0].data.decode('utf-8')
        except Exception as e:
            rospy.logdebug(f"QR reading failed: {e}")
        return None

    def read_text_from_roi(self, roi_image):
        """Read text from ROI using OCR"""
        try:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
            text = text.strip()

            if len(text) > 0:
                return text
        except Exception as e:
            rospy.logdebug(f"OCR reading failed: {e}")
        return None

    def send_information_to_server(self, node_id, information_data):
        """Send comprehensive information to server"""
        try:
            data = {
                "node_id": node_id,
                "information_type": information_data['type'],
                "information_content": information_data['content'],
                "confidence": information_data['confidence'],
                "timestamp": time.time(),
                "robot_id": "jetbot_001"
            }

            message = json.dumps(data)

            if self.mqtt_client:
                result = self.mqtt_client.publish(self.mqtt_topic, message)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    rospy.loginfo(f"📤 Information sent via MQTT: Node {node_id}")
                    return True

            # Fallback: HTTP POST
            response = requests.post(self.server_api_endpoint, json=data, timeout=5)
            if response.status_code == 200:
                rospy.loginfo(f"📤 Information sent via HTTP: Node {node_id}")
                return True

        except Exception as e:
            rospy.logerr(f"❌ Server communication error: {e}")

        return False

    def handle_intersection(self):
        """Handle intersection logic for Problem B"""
        self.update_current_node()

        if not self.path:
            rospy.logwarn("⚠️ No planned path!")
            return Direction.NORTH

        # Check if we're at a Load node that hasn't been visited
        if self.is_at_load_node() and self.current_node_id not in self.visited_loads:
            rospy.loginfo(f"📍 At Load node {self.current_node_id} - need to read information")
            self.current_state = RobotState.READING_INFORMATION
            return None

        # Standard intersection handling
        next_node_id = self.path[0]
        self.path.pop(0)

        if self.current_node_id in self.navigator.nodes_data:
            current_pos = self.navigator.nodes_data[self.current_node_id]
            next_pos = self.navigator.nodes_data[next_node_id]

            dx = next_pos['x'] - current_pos['x']
            dy = next_pos['y'] - current_pos['y']

            if dx > 0:
                direction = Direction.EAST
            elif dx < 0:
                direction = Direction.WEST  
            elif dy > 0:
                direction = Direction.NORTH
            else:
                direction = Direction.SOUTH
        else:
            direction = Direction.NORTH

        self.current_node_id = next_node_id
        rospy.loginfo(f"➡️ Moving to node {next_node_id}, direction {direction.name}")

        return direction

    def execute_turn(self, direction):
        """Enhanced turn execution"""
        base_speed = 0.2

        if direction == Direction.NORTH:
            self.robot.left_motor.value = base_speed
            self.robot.right_motor.value = base_speed
            turn_duration = 0.5
        elif direction == Direction.EAST:
            self.robot.left_motor.value = base_speed
            self.robot.right_motor.value = -base_speed * 0.6
            turn_duration = 1.2
        elif direction == Direction.WEST:
            self.robot.left_motor.value = -base_speed * 0.6  
            self.robot.right_motor.value = base_speed
            turn_duration = 1.2
        elif direction == Direction.SOUTH:
            self.robot.left_motor.value = base_speed
            self.robot.right_motor.value = -base_speed
            turn_duration = 2.2

        time.sleep(turn_duration)
        self.robot.stop()
        time.sleep(0.3)

    def check_goal_reached(self):
        """Check if reached End node"""
        self.update_current_node()

        # Check if all Load nodes visited and at End node
        all_loads_visited = len(self.visited_loads) == len(self.load_nodes)
        at_end_node = self.current_node_id == self.navigator.end_node

        if at_end_node and all_loads_visited:
            if self.goal_reached_time is None:
                self.goal_reached_time = time.time()
                rospy.loginfo(f"🎯 Mission completed! All {len(self.visited_loads)} Load nodes visited")

            elapsed = time.time() - self.goal_reached_time
            return elapsed >= self.stop_duration

        return False

    def run(self):
        """Main control loop for Problem B Complete"""
        rospy.loginfo("🚀 Starting Problem B Complete - Waiting for flag...")

        # Plan optimal route
        self.plan_optimal_route()
        if not self.path:
            rospy.logerr("❌ Cannot plan route!")
            return

        self.current_node_id = self.navigator.start_node

        rate = rospy.Rate(20)  # 20 Hz

        while not rospy.is_shutdown():
            current_time = time.time()

            # Timeout check
            if self.start_time and (current_time - self.start_time) > self.max_time:
                rospy.logwarn("⏰ Time limit reached!")
                break

            # State machine
            if self.current_state == RobotState.WAITING_FOR_FLAG:
                self.robot.stop()
                if self.flag_detected:
                    self.current_state = RobotState.DRIVING_STRAIGHT
                    rospy.loginfo("➡️ Flag detected - Starting mission")

            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                if self.latest_frame is not None:
                    line_center, line_detected = self.detect_line(self.latest_frame)

                    if self.check_goal_reached():
                        self.current_state = RobotState.GOAL_REACHED
                        continue

                    if self.detect_intersection():
                        self.current_state = RobotState.APPROACHING_INTERSECTION
                        rospy.loginfo("🔍 Intersection detected")
                        continue

                    self.execute_line_following(line_center, line_detected)

            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                self.robot.left_motor.value = 0.1
                self.robot.right_motor.value = 0.1
                time.sleep(0.8)
                self.current_state = RobotState.HANDLING_EVENT

            elif self.current_state == RobotState.HANDLING_EVENT:
                self.robot.stop()
                time.sleep(0.2)

                direction = self.handle_intersection()

                if direction is None:  # Need to read information
                    continue

                self.execute_turn(direction)
                self.current_state = RobotState.REACQUIRING_LINE

            elif self.current_state == RobotState.READING_INFORMATION:
                self.robot.stop()
                rospy.loginfo(f"📖 Reading comprehensive information at Load node {self.current_node_id}")

                # Try to read information with comprehensive approach
                read_start_time = time.time()
                information_read = False

                while time.time() - read_start_time < self.reading_timeout:
                    if self.latest_frame is not None:
                        info_data = self.read_information_comprehensive(self.latest_frame)
                        if info_data:
                            success = self.send_information_to_server(self.current_node_id, info_data)
                            if success:
                                self.visited_loads.add(self.current_node_id)
                                rospy.loginfo(f"✅ Information read and sent for node {self.current_node_id}")
                                information_read = True
                                break
                    time.sleep(0.1)

                if not information_read:
                    rospy.logwarn(f"⚠️ Failed to read information at node {self.current_node_id}")
                    # Mark as visited anyway to continue mission
                    self.visited_loads.add(self.current_node_id)

                # Continue with path
                if self.path:
                    next_node_id = self.path[0]
                    self.path.pop(0)
                    self.current_node_id = next_node_id

                self.current_state = RobotState.REACQUIRING_LINE

            elif self.current_state == RobotState.REACQUIRING_LINE:
                if self.latest_frame is not None:
                    line_center, line_detected = self.detect_line(self.latest_frame)

                    if line_detected:
                        self.current_state = RobotState.DRIVING_STRAIGHT
                        rospy.loginfo("✅ Line reacquired")
                    else:
                        self.robot.left_motor.value = 0.1
                        self.robot.right_motor.value = 0.1

            elif self.current_state == RobotState.GOAL_REACHED:
                self.robot.stop()
                elapsed = time.time() - self.goal_reached_time
                rospy.loginfo(f"🎯 At goal: {elapsed:.1f}s/{self.stop_duration}s")
                rospy.loginfo(f"📊 Visited {len(self.visited_loads)}/{len(self.load_nodes)} Load nodes")

                if elapsed >= self.stop_duration:
                    rospy.loginfo("🏆 PROBLEM B COMPLETED!")
                    break

            rate.sleep()

        # Cleanup
        self.robot.stop()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        rospy.loginfo("✅ Problem B Complete Controller stopped")

def main():
    rospy.init_node('problem_b_complete_controller')

    try:
        controller = ProblemBCompleteController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"❌ Error: {e}")
    finally:
        rospy.loginfo("👋 Problem B Complete Controller shutdown")

if __name__ == '__main__':
    main()
