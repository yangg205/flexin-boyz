#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
from jetbot import Robot
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LineFollower:
    def __init__(self):
        rospy.init_node('line_follower_test')
        self.robot = Robot()
        self.bridge = CvBridge()

        self.base_speed = 0.15
        self.turn_gain = 0.4

        self.line_roi_top = 200
        self.line_roi_height = 120
        self.line_roi_width = 224

        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 70])

        self.latest_frame = None

        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)
        self.rate = rospy.Rate(20)

    def camera_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error processing camera image: {e}")

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

    def follow_line(self):
        if self.latest_frame is None:
            self.robot.stop()
            return

        line_center, detected = self.detect_line(self.latest_frame)

        if not detected:
            rospy.loginfo("Line lost, slowing to reacquire...")
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

    def run(self):
        rospy.loginfo("Line follower test node started")
        while not rospy.is_shutdown():
            self.follow_line()
            self.rate.sleep()
        self.robot.stop()

if __name__ == '__main__':
    try:
        follower = LineFollower()
        follower.run()
    except rospy.ROSInterruptException:
        pass
