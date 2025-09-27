#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
import math
import requests
from enum import Enum
from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from opposite_detector import SimpleOppositeDetector
from pyzbar.pyzbar import decode
import paho.mqtt.client as mqtt
import onnxruntime as ort
import pytesseract

class RobotState(Enum):
    WAITING_FOR_FLAG = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    HANDLING_EVENT = 3
    READING_INFORMATION = 4
    LEAVING_INTERSECTION = 5
    REACQUIRING_LINE = 6
    GOAL_REACHED = 7
    DEAD_END = 8

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class SignCommand:
    def __init__(self, command_type, direction, confidence=0.0):
        self.command_type = command_type  # 'MANDATORY', 'PROHIBITIVE', 'READ_INFO'
        self.direction = direction        # 'N', 'E', 'S', 'W' or None for READ_info
        self.confidence = confidence

class ProblemCController:
    def __init__(self):
        rospy.loginfo("🚀 Khởi tạo Problem C Controller - Sign-based Navigation...")
        self.setup_parameters()
        self.robot = Robot()
        self.bridge = CvBridge()

        # Intersection detector
        self.opposite_detector = SimpleOppositeDetector()

        # Initialize YOLO for sign detection
        self.initialize_yolo()

        # State management
        self.current_state = RobotState.WAITING_FOR_FLAG
        self.current_node_id = None
        self.current_direction = Direction.EAST  # Default start direction

        # Sign-based navigation
        self.detected_signs = []
        self.information_nodes_visited = set()
        self.must_read_information = False
        self.target_information_node = None

        # ROS subscribers
        self.camera_sub = rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # Control variables
        self.latest_frame = None
        self.lidar_data = None
        self.flag_detected = False
        self.start_time = None
        self.goal_reached_time = None

        # MQTT for server communication
        self.setup_mqtt()

        # PID controller for line following
        self.prev_error = 0
        self.integral = 0

        rospy.loginfo("✅ Problem C Controller khởi tạo thành công!")

    def setup_parameters(self):
        """Setup parameters for Problem C - Sign-based navigation"""
        # Line following parameters  
        self.base_speed = 0.15  # Slower for more precise control
        self.kp = 0.005
        self.ki = 0.0001
        self.kd = 0.002

        self.line_roi_height = 160
        self.line_roi_width = 224
        self.line_roi_top = 200

        # HSV thresholds
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])

        # YOLO parameters - enhanced for Problem C
        self.YOLO_MODEL_PATH = "/home/jetbot/hackathon/models/best.onnx"
        self.YOLO_CONF_THRESHOLD = 0.5  # Lower threshold for better detection
        self.YOLO_INPUT_SIZE = (640, 640)
        self.YOLO_CLASS_NAMES = ['N', 'E', 'W', 'S', 'NN', 'NE', 'NW', 'NS', 'math']

        # Sign types for Problem C
        self.MANDATORY_SIGNS = {'N', 'E', 'W', 'S'}      # Must go in this direction
        self.PROHIBITIVE_SIGNS = {'NN', 'NE', 'NW', 'NS'}  # Cannot go in this direction
        self.INFORMATION_SIGNS = {'math'}  # Must read information at next intersection

        # Timing parameters
        self.stop_duration = 5.0
        self.max_time = 300  # 5 minutes
        self.reading_timeout = 15.0
        self.sign_scanning_timeout = 5.0

        # Recovery parameters
        self.max_line_lost = 15
        self.recovery_speed = 0.08

        # Turn parameters
        self.turn_speed = 0.2
        self.turn_90_duration = 1.0
        self.turn_180_duration = 2.0

        # Server communication
        self.server_api_endpoint = "http://hackathon-server.local/api/information"
        self.mqtt_broker = "hackathon-server.local"
        self.mqtt_port = 1883
        self.mqtt_topic = "jetbot/information"

    def initialize_yolo(self):
        """Initialize YOLO model for sign detection"""
        try:
            self.yolo_session = ort.InferenceSession(
                self.YOLO_MODEL_PATH, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            rospy.loginfo("✅ YOLO model loaded for sign detection")
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
                    rospy.loginfo("🚩 Flag detected! Starting Problem C mission...")

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

        left_speed = max(0.05, min(0.3, left_speed))
        right_speed = max(0.05, min(0.3, right_speed))

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

    def detect_signs_with_yolo(self, image):
        """Enhanced YOLO detection for traffic signs"""
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
                return node_id
        except Exception as e:
            rospy.logwarn(f"⚠️ QR reading error: {e}")
        return None

    def update_current_node(self):
        """Update current node with QR validation"""
        if self.latest_frame is not None:
            qr_node = self.read_qr_node(self.latest_frame)
            if qr_node is not None:
                self.current_node_id = qr_node
                rospy.loginfo(f"📍 At intersection node: {qr_node}")

    def scan_for_signs(self):
        """Scan for traffic signs at WN corner"""
        rospy.loginfo("🔍 Scanning for traffic signs at WN corner...")

        # Turn to face WN corner (West-North) - typically -135° from East
        self.turn_robot_precise(-135, update_direction=False)
        time.sleep(0.5)  # Allow camera to stabilize

        signs = []
        if self.latest_frame is not None:
            detections = self.detect_signs_with_yolo(self.latest_frame)

            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']

                if class_name in self.MANDATORY_SIGNS:
                    signs.append(SignCommand('MANDATORY', class_name, confidence))
                    rospy.loginfo(f"🚦 Mandatory sign detected: GO {class_name} (conf: {confidence:.2f})")

                elif class_name in self.PROHIBITIVE_SIGNS:
                    # Convert from NN, NE, NW, NS to actual direction
                    prohibited_dir = class_name[1]  # Second character
                    signs.append(SignCommand('PROHIBITIVE', prohibited_dir, confidence))
                    rospy.loginfo(f"🚫 Prohibitive sign detected: NO {prohibited_dir} (conf: {confidence:.2f})")

                elif class_name in self.INFORMATION_SIGNS:
                    signs.append(SignCommand('READ_INFO', None, confidence))
                    rospy.loginfo(f"📖 Information sign detected: READ at next intersection (conf: {confidence:.2f})")

        # Turn back to original direction
        self.turn_robot_precise(135, update_direction=False)

        return signs

    def read_information_comprehensive(self, frame):
        """Read information at Load nodes using multi-modal approach"""
        if frame is None:
            return None

        information_data = {
            'type': None,
            'content': None,
            'confidence': 0.0
        }

        try:
            # Step 1: Use YOLO to detect information signs
            detections = self.detect_signs_with_yolo(frame)

            for detection in detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                box = detection['box']

                if class_name == 'math':
                    # Extract ROI for the detected math sign
                    x1, y1, x2, y2 = box
                    roi = frame[y1:y2, x1:x2]

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
            roi_h = int(width * 0.4)

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

            # Simple math solver
            try:
                if '=' in text:
                    expression = text.split('=')[0].strip()
                else:
                    expression = text

                result = eval(expression)
                return f"{expression}={result}"
            except:
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
        """Send information to server"""
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

    def make_navigation_decision(self, signs):
        """Make navigation decision based on detected signs"""
        rospy.loginfo("🤔 Making navigation decision based on signs...")

        # Extract different types of signs
        mandatory_signs = [s for s in signs if s.command_type == 'MANDATORY']
        prohibitive_signs = [s for s in signs if s.command_type == 'PROHIBITIVE']
        info_signs = [s for s in signs if s.command_type == 'READ_INFO']

        # Check if need to read information at next intersection
        if info_signs:
            self.must_read_information = True
            rospy.loginfo("📖 Must read information at next intersection")

        # Determine available directions (all directions robot can physically go)
        available_directions = self.get_available_directions()

        # Apply prohibitive constraints
        prohibited_directions = {s.direction for s in prohibitive_signs}
        valid_directions = [d for d in available_directions if d not in prohibited_directions]

        rospy.loginfo(f"🚧 Available directions: {available_directions}")
        rospy.loginfo(f"🚫 Prohibited directions: {prohibited_directions}")
        rospy.loginfo(f"✅ Valid directions: {valid_directions}")

        # Decision logic
        chosen_direction = None

        if mandatory_signs:
            # Priority 1: Follow mandatory signs
            mandatory_direction = mandatory_signs[0].direction  # Take highest confidence
            if mandatory_direction in valid_directions:
                chosen_direction = mandatory_direction
                rospy.loginfo(f"🚦 Following MANDATORY sign: {chosen_direction}")
            else:
                rospy.logerr(f"❌ Mandatory direction {mandatory_direction} is prohibited or unavailable!")
                return None
        else:
            # Priority 2: Continue straight if possible
            if 'N' in valid_directions and self.current_direction == Direction.NORTH:
                chosen_direction = 'N'
            elif 'E' in valid_directions and self.current_direction == Direction.EAST:
                chosen_direction = 'E'
            elif 'S' in valid_directions and self.current_direction == Direction.SOUTH:
                chosen_direction = 'S'
            elif 'W' in valid_directions and self.current_direction == Direction.WEST:
                chosen_direction = 'W'
            else:
                # Choose any valid direction
                if valid_directions:
                    chosen_direction = valid_directions[0]
                    rospy.loginfo(f"🔄 No straight path, choosing: {chosen_direction}")
                else:
                    rospy.logerr("❌ No valid directions available!")
                    return None

        rospy.loginfo(f"✅ Final decision: GO {chosen_direction}")
        return chosen_direction

    def get_available_directions(self):
        """Get available directions from current intersection"""
        # In real implementation, this would use sensors to detect available paths
        # For now, return all possible directions
        return ['N', 'E', 'S', 'W']

    def execute_direction_command(self, direction):
        """Execute turn to specified direction"""
        if direction is None:
            return False

        current_dir = self.current_direction
        target_dir_map = {'N': Direction.NORTH, 'E': Direction.EAST, 'S': Direction.SOUTH, 'W': Direction.WEST}
        target_dir = target_dir_map[direction]

        # Calculate turn angle
        current_idx = current_dir.value
        target_idx = target_dir.value
        turn_diff = (target_idx - current_idx + 4) % 4

        if turn_diff == 0:
            # Go straight
            rospy.loginfo("➡️ Going straight")
            return True
        elif turn_diff == 1:
            # Turn right (90°)
            rospy.loginfo("↗️ Turning RIGHT (90°)")
            self.turn_robot_precise(90, update_direction=True)
        elif turn_diff == 2:
            # Turn around (180°)
            rospy.loginfo("↩️ Turning AROUND (180°)")
            self.turn_robot_precise(180, update_direction=True)
        elif turn_diff == 3:
            # Turn left (90°)
            rospy.loginfo("↖️ Turning LEFT (90°)")
            self.turn_robot_precise(-90, update_direction=True)

        return True

    def turn_robot_precise(self, degrees, update_direction=True):
        """Execute precise robot turn"""
        duration = abs(degrees) / 90.0 * self.turn_90_duration

        if degrees > 0:
            # Turn right
            self.robot.left_motor.value = self.turn_speed
            self.robot.right_motor.value = -self.turn_speed
        elif degrees < 0:
            # Turn left
            self.robot.left_motor.value = -self.turn_speed
            self.robot.right_motor.value = self.turn_speed

        if degrees != 0:
            time.sleep(duration)
            self.robot.stop()
            time.sleep(0.3)  # Stabilization pause

        # Update current direction
        if update_direction and degrees != 0:
            turn_steps = round(degrees / 90)
            new_dir_idx = (self.current_direction.value + turn_steps + 4) % 4
            self.current_direction = Direction(new_dir_idx)
            rospy.loginfo(f"🧭 New direction: {self.current_direction.name}")

    def handle_intersection_sign_based(self):
        """Handle intersection using sign-based navigation"""
        rospy.loginfo("🚦 Handling intersection with sign-based navigation")
        self.robot.stop()
        time.sleep(0.5)

        # Update current node
        self.update_current_node()

        # Check if we need to read information at this node
        if self.must_read_information:
            rospy.loginfo("📖 This is an information node - reading information")
            self.current_state = RobotState.READING_INFORMATION
            self.must_read_information = False
            return

        # Scan for traffic signs
        detected_signs = self.scan_for_signs()

        if not detected_signs:
            rospy.logwarn("⚠️ No signs detected - continuing straight")
            self.current_state = RobotState.LEAVING_INTERSECTION
            return

        # Make navigation decision
        chosen_direction = self.make_navigation_decision(detected_signs)

        if chosen_direction is None:
            rospy.logerr("❌ Cannot determine valid direction")
            self.current_state = RobotState.DEAD_END
            return

        # Execute the direction command
        success = self.execute_direction_command(chosen_direction)

        if success:
            self.current_state = RobotState.LEAVING_INTERSECTION
        else:
            self.current_state = RobotState.DEAD_END

    def check_goal_reached(self):
        """Check if reached End node"""
        self.update_current_node()

        # For Problem C, we assume any node can be End
        # Goal is reached after successfully reading information and navigating
        if hasattr(self, 'information_read_successfully') and self.information_read_successfully:
            if self.current_node_id is not None:  # At any valid intersection
                if self.goal_reached_time is None:
                    self.goal_reached_time = time.time()
                    rospy.loginfo(f"🎯 Mission completed! At node {self.current_node_id}")

                elapsed = time.time() - self.goal_reached_time
                return elapsed >= self.stop_duration

        return False

    def run(self):
        """Main control loop for Problem C"""
        rospy.loginfo("🚀 Starting Problem C - Sign-based Navigation")
        rospy.loginfo("📍 Starting at intersection, facing EAST by default")

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
                    rospy.loginfo("➡️ Flag detected - Starting sign-based navigation")

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
                self.handle_intersection_sign_based()

            elif self.current_state == RobotState.READING_INFORMATION:
                self.robot.stop()
                rospy.loginfo(f"📖 Reading information at node {self.current_node_id}")

                # Try to read information
                read_start_time = time.time()
                information_read = False

                while time.time() - read_start_time < self.reading_timeout:
                    if self.latest_frame is not None:
                        info_data = self.read_information_comprehensive(self.latest_frame)
                        if info_data:
                            success = self.send_information_to_server(self.current_node_id, info_data)
                            if success:
                                self.information_nodes_visited.add(self.current_node_id)
                                self.information_read_successfully = True
                                rospy.loginfo(f"✅ Information read successfully at node {self.current_node_id}")
                                information_read = True
                                break
                    time.sleep(0.1)

                if not information_read:
                    rospy.logwarn(f"⚠️ Failed to read information at node {self.current_node_id}")
                    self.information_read_successfully = False

                self.current_state = RobotState.LEAVING_INTERSECTION

            elif self.current_state == RobotState.LEAVING_INTERSECTION:
                self.robot.left_motor.value = self.base_speed
                self.robot.right_motor.value = self.base_speed
                time.sleep(1.5)  # Clear intersection
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

                if elapsed >= self.stop_duration:
                    rospy.loginfo("🏆 PROBLEM C COMPLETED!")
                    break

            elif self.current_state == RobotState.DEAD_END:
                self.robot.stop()
                rospy.logerr("❌ Dead end reached - mission failed")
                break

            rate.sleep()

        # Cleanup
        self.robot.stop()
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        rospy.loginfo("✅ Problem C Controller stopped")

def main():
    rospy.init_node('problem_c_controller')

    try:
        controller = ProblemCController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"❌ Error: {e}")
    finally:
        rospy.loginfo("👋 Problem C Controller shutdown")

if __name__ == '__main__':
    main()
