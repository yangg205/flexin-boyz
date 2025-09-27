#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import os
import json
import math
from enum import Enum
from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from map_navigator import MapNavigator
from opposite_detector import SimpleOppositeDetector
from pyzbar.pyzbar import decode

class RobotState(Enum):
    WAITING_FOR_FLAG = 0
    DRIVING_STRAIGHT = 1
    APPROACHING_INTERSECTION = 2
    HANDLING_EVENT = 3
    LEAVING_INTERSECTION = 4
    REACQUIRING_LINE = 5
    GOAL_REACHED = 6
    OUT_OF_BOUNDS = 7

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class ProblemAImprovedController:
    def __init__(self):
        rospy.loginfo("Khởi tạo Problem A Improved Controller...")
        self.setup_parameters()
        self.robot = Robot()
        self.bridge = CvBridge()
        self.navigator = MapNavigator("/home/jetbot/hackathon/map.json")
        self.opposite_detector = SimpleOppositeDetector()
        self.current_state = RobotState.WAITING_FOR_FLAG
        self.target_node_id = None
        self.path = []
        self.current_node_id = None
        self.start_time = None
        self.goal_reached_time = None
        self.camera_sub = rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.latest_frame = None
        self.lidar_data = None
        self.flag_detected = False
        self.last_qr_node = None
        self.last_turn_time = None
        rospy.loginfo("Problem A Improved Controller khởi tạo thành công!")

    def setup_parameters(self):
        self.base_speed = 0.15
        self.turn_gain = 0.3
        self.line_roi_height = 160
        self.line_roi_width = 224
        self.line_roi_top = 200
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])
        self.intersection_threshold = 0.3
        self.stop_duration = 5.0
        self.max_time = 300
        self.boundary_check = True
    def camera_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Lỗi xử lý camera: {e}")
    def lidar_callback(self, msg):
        self.lidar_data = msg
        if not self.flag_detected and self.current_state == RobotState.WAITING_FOR_FLAG:
            # Flag detection cải tiến: dùng camera, mẫu cờ hiệu đỏ
            if self.latest_frame is not None:
                if self.detect_flag_color(self.latest_frame):
                    self.flag_detected = True
                    self.start_time = time.time()
                    rospy.loginfo("Phát hiện cờ hiệu đỏ! Bắt đầu di chuyển...")
    def detect_flag_color(self, frame):
        # Dò tìm màu đỏ trong frame cho cờ hiệu
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255]))
        mask2 = cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
        mask = cv2.bitwise_or(mask1, mask2)
        if cv2.countNonZero(mask) > 2000:
            return True
        return False
    def detect_line(self, frame):
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
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        moments = cv2.moments(mask)
        if moments['m00'] > 1000:
            cx = int(moments['m10'] / moments['m00'])
            line_center = cx - self.line_roi_width // 2
            return line_center, True
        return 0, False
    def detect_intersection(self):
        # Multi-sensor (LiDAR + camera)
        lidar_detect = self.opposite_detector.detect_intersection(self.lidar_data) if self.lidar_data is not None else False
        camera_detect = False
        if self.latest_frame is not None:
            camera_detect = self.check_camera_intersection(self.latest_frame)
        return lidar_detect or camera_detect
    def check_camera_intersection(self, frame):
        # Phát hiện giao lộ bằng camera (nhiều line, hình chữ T hoặc X)
        roi_top = self.line_roi_top
        roi_bottom = roi_top + self.line_roi_height
        roi_left = (frame.shape[1] - self.line_roi_width) // 2
        roi_right = roi_left + self.line_roi_width
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
        if lines is not None and len(lines) >= 2:
            return True
        return False
    def execute_line_following(self, line_center, line_detected):
        if not line_detected:
            # If line lost, slow down and try to reacquire
            self.robot.left_motor.value = 0.05
            self.robot.right_motor.value = 0.05
            return
        steering = -line_center * self.turn_gain / 100.0
        steering = max(-1.0, min(1.0, steering))
        left_speed = self.base_speed - steering * 0.1
        right_speed = self.base_speed + steering * 0.1
        left_speed = max(0, min(0.3, left_speed))
        right_speed = max(0, min(0.3, right_speed))
        self.robot.left_motor.value = left_speed
        self.robot.right_motor.value = right_speed
    def read_qr_node(self, frame):
        decoded_objects = decode(frame)
        for obj in decoded_objects:
            try:
                node_id = int(obj.data.decode('utf-8'))
                return node_id
            except:
                continue
        return None
    def update_current_node(self):
        # Đọc node bằng QR nếu available
        if self.latest_frame is not None:
            node_id = self.read_qr_node(self.latest_frame)
            if node_id in self.navigator.nodes_data:
                self.current_node_id = node_id
                self.last_qr_node = node_id
    def check_goal_reached(self):
        self.update_current_node()
        if self.current_node_id == self.navigator.end_node:
            if self.goal_reached_time is None:
                self.goal_reached_time = time.time()
                rospy.loginfo(f"Đã đến đích! Node {self.current_node_id}")
            elapsed = time.time() - self.goal_reached_time
            if elapsed >= self.stop_duration:
                return True
        return False
    def check_out_of_bounds(self):
        if not self.boundary_check or self.latest_frame is None:
            return False
        line_center, line_detected = self.detect_line(self.latest_frame)
        return not line_detected
    def handle_intersection(self):
        self.update_current_node()
        if not self.path:
            rospy.logwarn("Không có đường đi được lập kế hoạch!")
            return Direction.NORTH
        next_node_id = self.path[0]
        self.path.pop(0)
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
        self.current_node_id = next_node_id
        rospy.loginfo(f"Di chuyển đến node {next_node_id}, hướng {direction.name}")
        return direction
    def execute_turn(self, direction):
        # Dùng thời gian + encoder (nếu có) để điều chỉnh góc rẽ
        turn_duration = 1.0
        speed = self.base_speed
        if direction == Direction.NORTH:
            self.robot.left_motor.value = speed
            self.robot.right_motor.value = speed
        elif direction == Direction.EAST:
            self.robot.left_motor.value = speed
            self.robot.right_motor.value = -speed * 0.5
            turn_duration = 1.1
        elif direction == Direction.WEST:
            self.robot.left_motor.value = -speed * 0.5
            self.robot.right_motor.value = speed
            turn_duration = 1.1
        elif direction == Direction.SOUTH:
            self.robot.left_motor.value = speed
            self.robot.right_motor.value = -speed
            turn_duration = 2.0
        self.last_turn_time = time.time()
        time.sleep(turn_duration)
        self.robot.stop()
    def run(self):
        rospy.loginfo("Bắt đầu Problem A Improved - Chờ cờ hiệu...")
        self.path = self.navigator.find_path()
        if not self.path:
            rospy.logerr("Không thể tìm đường từ start đến end!")
            return
        self.update_current_node()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            current_time = time.time()
            if self.start_time and (current_time - self.start_time) > self.max_time:
                rospy.logwarn("Hết thời gian 5 phút!")
                break
            if self.current_state == RobotState.WAITING_FOR_FLAG:
                self.robot.stop()
                if self.flag_detected:
                    self.current_state = RobotState.DRIVING_STRAIGHT
            elif self.current_state == RobotState.DRIVING_STRAIGHT:
                line_center, line_detected = self.detect_line(self.latest_frame)
                if self.check_out_of_bounds():
                    self.current_state = RobotState.OUT_OF_BOUNDS
                    continue
                if self.detect_intersection():
                    self.current_state = RobotState.APPROACHING_INTERSECTION
                    continue
                if self.check_goal_reached():
                    self.current_state = RobotState.GOAL_REACHED
                    continue
                self.execute_line_following(line_center, line_detected)
            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                self.robot.left_motor.value = 0.1
                self.robot.right_motor.value = 0.1
                time.sleep(0.5)
                self.current_state = RobotState.HANDLING_EVENT
            elif self.current_state == RobotState.HANDLING_EVENT:
                self.robot.stop()
                direction = self.handle_intersection()
                self.execute_turn(direction)
                self.current_state = RobotState.REACQUIRING_LINE
            elif self.current_state == RobotState.REACQUIRING_LINE:
                line_center, line_detected = self.detect_line(self.latest_frame)
                if line_detected:
                    self.current_state = RobotState.DRIVING_STRAIGHT
                else:
                    self.robot.left_motor.value = 0.1
                    self.robot.right_motor.value = 0.1
            elif self.current_state == RobotState.GOAL_REACHED:
                self.robot.stop()
                elapsed = time.time() - self.goal_reached_time
                rospy.loginfo(f"Đã đến đích và dừng {elapsed:.1f}s/{self.stop_duration}s")
                if elapsed >= self.stop_duration:
                    rospy.loginfo("HOÀN THÀNH PROBLEM A!")
                    break
            elif self.current_state == RobotState.OUT_OF_BOUNDS:
                self.robot.stop()
                rospy.logwarn("Xe ra ngoài vạch kẻ! Quay về điểm start...")
                break
            rate.sleep()
        self.robot.stop()
        rospy.loginfo("Problem A Improved Controller đã dừng")
def main():
    rospy.init_node('problem_a_improved_controller')
    try:
        controller = ProblemAImprovedController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}")
    finally:
        rospy.loginfo("Tắt Problem A Improved Controller")
if __name__ == '__main__':
    main()
