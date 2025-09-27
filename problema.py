#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import time
from enum import Enum
from jetbot import Robot
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from map_navigator import MapNavigator

class RobotState(Enum):
    DRIVING_STRAIGHT = 0
    APPROACHING_INTERSECTION = 1
    HANDLING_EVENT = 2
    REACQUIRING_LINE = 3
    GOAL_REACHED = 4
    OUT_OF_BOUNDS = 5

class Direction(Enum):
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

class ProblemAController:
    def __init__(self):
        rospy.loginfo("Khởi tạo Problem A Controller...")
        self.setup_parameters()
        self.robot = Robot()
        self.bridge = CvBridge()
        self.navigator = MapNavigator("/home/jetbot/flexin-boyz/map.json")

        # Lấy đường đi ngắn nhất từ start → end
        self.current_node_id = self.navigator.start_node
        self.goal_node_id = self.navigator.end_node
        self.path = self.navigator.find_path(self.current_node_id, self.goal_node_id)

        if not self.path:
            rospy.logerr("Không tìm thấy đường đi từ start → end trong map!")
            return

        self.current_state = RobotState.DRIVING_STRAIGHT
        self.goal_reached_time = None

        # Camera
        self.latest_frame = None
        self.camera_sub = rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)

        rospy.loginfo("Problem A Controller khởi tạo thành công!")

    def setup_parameters(self):
        self.base_speed = 1.0   # tốc độ nhanh hơn (có thể tăng đến 0.3 tùy line)
        self.turn_gain = 0.3
        self.line_roi_height = 160
        self.line_roi_width = 224
        self.line_roi_top = 200
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 60])
        self.stop_duration = 3.0

    def camera_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Lỗi xử lý camera: {e}")

    def detect_line(self, frame):
        if frame is None:
            return 0, False
        height, width = frame.shape[:2]
        roi = frame[self.line_roi_top:self.line_roi_top+self.line_roi_height,
                    (width - self.line_roi_width)//2:(width + self.line_roi_width)//2]
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

    def execute_line_following(self, line_center, line_detected):
        if not line_detected:
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

    def handle_intersection(self):
        if not self.path:
            rospy.loginfo("Đã đến đích!")
            return None
        next_node_id = self.path.pop(0)
        edge = None
        for e in self.navigator.edges_data:
            if e['source'] == self.current_node_id and e['target'] == next_node_id:
                edge = e
                break
        if edge is None:
            rospy.logerr(f"Không tìm thấy cạnh từ {self.current_node_id} → {next_node_id}")
            return None
        label = edge['label']
        self.current_node_id = next_node_id
        rospy.loginfo(f"Đi tới node {next_node_id}, hướng {label}")
        if label == "N": return Direction.NORTH
        if label == "S": return Direction.SOUTH
        if label == "E": return Direction.EAST
        if label == "W": return Direction.WEST
        return None

    def execute_turn(self, direction):
        if direction is None:
            return
        speed = 0.2
        turn_duration = 1.0
        if direction == Direction.NORTH:
            self.robot.left_motor.value = speed
            self.robot.right_motor.value = speed
        elif direction == Direction.EAST:
            self.robot.left_motor.value = speed
            self.robot.right_motor.value = -speed
            turn_duration = 1.0
        elif direction == Direction.WEST:
            self.robot.left_motor.value = -speed
            self.robot.right_motor.value = speed
            turn_duration = 1.0
        elif direction == Direction.SOUTH:
            self.robot.left_motor.value = speed
            self.robot.right_motor.value = -speed
            turn_duration = 2.0
        time.sleep(turn_duration)
        self.robot.stop()

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.current_state == RobotState.DRIVING_STRAIGHT:
                line_center, line_detected = self.detect_line(self.latest_frame)
                if self.current_node_id == self.goal_node_id:
                    self.current_state = RobotState.GOAL_REACHED
                    continue
                # Giả định: tới gần giao lộ thì đổi state
                if not line_detected:  
                    self.current_state = RobotState.APPROACHING_INTERSECTION
                    continue
                self.execute_line_following(line_center, line_detected)

            elif self.current_state == RobotState.APPROACHING_INTERSECTION:
                self.robot.left_motor.value = 0.1
                self.robot.right_motor.value = 0.1
                time.sleep(0.3)
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
                if self.goal_reached_time is None:
                    self.goal_reached_time = time.time()
                    rospy.loginfo("Đã đến đích!")
                elapsed = time.time() - self.goal_reached_time
                if elapsed >= self.stop_duration:
                    rospy.loginfo("HOÀN THÀNH PROBLEM A!")
                    break

            rate.sleep()
        self.robot.stop()

def main():
    rospy.init_node('problem_a_controller')
    try:
        controller = ProblemAController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}")
    finally:
        rospy.loginfo("Tắt Problem A Controller")

if __name__ == '__main__':
    main()
