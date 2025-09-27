#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import time
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import argparse
from enum import Enum

from jetbot import Robot
from sensor_msgs.msg import LaserScan, Image
from opposite_detector import SimpleOppositeDetector
from map_navigator import MapNavigator

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
    DOMAIN = "https://hackathon2025-dev.fpt.edu.vn"
    ENDPOINT = "/api/maps/get_active_map/"

    def __init__(self, token, map_type):
        rospy.loginfo("Khởi tạo JetBot Controller...")
        self.setup_parameters()
        self.initialize_hardware()
        self.initialize_video_writer()

        self.map_data = self.fetch_map(token, map_type)
        rospy.loginfo(f"Map data: {json.dumps(self.map_data, ensure_ascii=False, indent=2)}")
        self.navigator = MapNavigator(self.map_data)
        if self.navigator.start_node is None or self.navigator.end_node is None:
            raise ValueError("start_node hoặc end_node không hợp lệ.")

        self.current_node_id = self.navigator.start_node
        self.target_node_id = None
        self.planned_path = None
        self.banned_edges = []
        self.plan_initial_route()

        self.latest_scan = None
        self.latest_image = None
        self.detector = SimpleOppositeDetector()
        rospy.Subscriber('/scan', LaserScan, self.detector.callback)
        rospy.Subscriber('/csi_cam_0/image_raw', Image, self.camera_callback)

        self.state_change_time = rospy.get_time()
        self._set_state(RobotState.WAITING_FOR_LINE, initial=True)
        rospy.loginfo("Khởi tạo hoàn tất. Sẵn sàng hoạt động.")

    def create_session_with_retries(self, total_retries=3, backoff_factor=0.5, status_forcelist=(429,500,502,503,504)):
        s = requests.Session()
        try:
            retries = Retry(
                total=total_retries,
                backoff_factor=backoff_factor,
                status_forcelist=status_forcelist,
                allowed_methods=frozenset(["GET","POST"])
            )
        except TypeError:
            retries = Retry(
                total=total_retries,
                backoff_factor=backoff_factor,
                status_forcelist=status_forcelist,
                method_whitelist=frozenset(["GET","POST"])
            )
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def fetch_map(self, token, map_type, timeout=10):
        if not token:
            raise ValueError("Token is required")
        if map_type not in ["map_a","map_b","map_z"]:
            raise ValueError("map_type must be one of: map_a,map_b,map_z")
        params = {"token": token, "map_type": map_type}
        url = self.DOMAIN.rstrip("/") + self.ENDPOINT
        session = self.create_session_with_retries()
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            rospy.logerr(f"Lỗi khi fetch map: {e}")
            raise

    def plan_initial_route(self):
        rospy.loginfo(f"Lập kế hoạch từ {self.navigator.start_node} đến {self.navigator.end_node}...")
        try:
            self.planned_path = self.navigator.find_path(
                self.navigator.start_node,
                self.navigator.end_node,
                self.banned_edges
            )
            if self.planned_path and len(self.planned_path)>1:
                self.target_node_id = self.planned_path[1]
                rospy.loginfo(f"Đã lập kế hoạch: {self.planned_path}, mục tiêu đầu: {self.target_node_id}")
            else:
                rospy.logerr("Không tìm thấy đường đi hoặc quá ngắn!")
                self._set_state(RobotState.DEAD_END)
        except Exception as e:
            rospy.logerr(f"Lỗi lập kế hoạch: {e}")
            self._set_state(RobotState.DEAD_END)
            raise

    def setup_parameters(self):
        self.WIDTH, self.HEIGHT = 300,300
        self.BASE_SPEED = 0.16
        self.TURN_SPEED = 0.2
        self.TURN_DURATION_90_DEG = 0.8
        self.ROI_Y = int(self.HEIGHT*0.85)
        self.ROI_H = int(self.HEIGHT*0.15)
        self.ROI_CENTER_WIDTH_PERCENT = 0.5
        self.LOOKAHEAD_ROI_Y = int(self.HEIGHT*0.60)
        self.LOOKAHEAD_ROI_H = int(self.HEIGHT*0.15)
        self.CORRECTION_GAIN = 0.5
        self.SAFE_ZONE_PERCENT = 0.3
        self.LINE_COLOR_LOWER = np.array([0,0,0])
        self.LINE_COLOR_UPPER = np.array([180,255,75])
        self.INTERSECTION_CLEARANCE_DURATION = 1.5
        self.INTERSECTION_APPROACH_DURATION = 0.5
        self.LINE_REACQUIRE_TIMEOUT = 3.0
        self.SCAN_PIXEL_THRESHOLD = 100
        self.current_state = None
        self.DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        self.current_direction_index = 1
        self.MAX_CORRECTION_ADJ = 0.12
        self.VIDEO_OUTPUT_FILENAME = "jetbot_run.avi"
        self.VIDEO_FPS = 20
        self.VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')
        self.LABEL_TO_DIRECTION_ENUM = {'N':Direction.NORTH,'E':Direction.EAST,'S':Direction.SOUTH,'W':Direction.WEST}

    def initialize_hardware(self):
        try:
            self.robot = Robot()
            rospy.loginfo("JetBot hardware ready.")
        except Exception as e:
            rospy.logwarn(f"Sử dụng Mock Robot. Lỗi: {e}")
            from unittest.mock import Mock
            self.robot = Mock()

    def initialize_video_writer(self):
        try:
            self.video_writer = cv2.VideoWriter(
                self.VIDEO_OUTPUT_FILENAME,
                self.VIDEO_FOURCC,
                self.VIDEO_FPS,
                (self.WIDTH,self.HEIGHT)
            )
            if self.video_writer.isOpened():
                rospy.loginfo(f"Bắt đầu ghi video: {self.VIDEO_OUTPUT_FILENAME}")
            else:
                rospy.logerr("Không mở được file video.")
                self.video_writer = None
        except Exception as e:
            rospy.logerr(f"Lỗi VideoWriter: {e}")
            self.video_writer = None

    def _set_state(self, new_state, initial=False):
        if self.current_state != new_state:
            if not initial: rospy.loginfo(f"State: {self.current_state} -> {new_state}")
            self.current_state = new_state
            self.state_change_time = rospy.get_time()

    def camera_callback(self, image_msg):
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) if image_msg.encoding.endswith("compressed") else np_arr.reshape(image_msg.height, image_msg.width, -1)
            if 'rgb' in image_msg.encoding.lower():
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.latest_image = cv2.resize(cv_image, (self.WIDTH,self.HEIGHT))
        except Exception as e:
            rospy.logerr(f"Lỗi camera callback: {e}")

    def _get_line_center(self, image, roi_y, roi_h):
        if image is None: return None
        roi = image[roi_y:roi_y+roi_h,:]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LINE_COLOR_LOWER, self.LINE_COLOR_UPPER)
        focus_mask = np.zeros_like(mask)
        roi_height, roi_width = mask.shape
        center_width = int(roi_width*self.ROI_CENTER_WIDTH_PERCENT)
        start_x = (roi_width-center_width)//2
        end_x = start_x + center_width
        cv2.rectangle(focus_mask,(start_x,0),(end_x,roi_height),255,-1)
        final_mask = cv2.bitwise_and(mask, focus_mask)
        contours = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if not contours: return None
        c = max(contours,key=cv2.contourArea)
        if cv2.contourArea(c)<self.SCAN_PIXEL_THRESHOLD: return None
        M = cv2.moments(c)
        return int(M["m10"]/M["m00"]) if M["m00"]>0 else None

    def correct_course(self, line_center_x):
        error = line_center_x - self.WIDTH/2
        if abs(error)<self.WIDTH/2*self.SAFE_ZONE_PERCENT:
            self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
            return
        adj = (error/(self.WIDTH/2))*self.CORRECTION_GAIN
        adj = np.clip(adj,-self.MAX_CORRECTION_ADJ,self.MAX_CORRECTION_ADJ)
        self.robot.set_motors(self.BASE_SPEED+adj, self.BASE_SPEED-adj)

    def turn_robot(self, degrees, update_main_direction=True):
        if degrees==0: return
        duration = abs(degrees)/90*self.TURN_DURATION_90_DEG
        if degrees>0:
            self.robot.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
        else:
            self.robot.set_motors(-self.TURN_SPEED, self.TURN_SPEED)
        start_time = rospy.get_time()
        while rospy.get_time()-start_time<duration:
            self._record_frame()
            rospy.sleep(1.0/self.VIDEO_FPS)
        self.robot.stop()
        if update_main_direction:
            turns = round(degrees/90)
            self.current_direction_index = (self.current_direction_index+turns+4)%4
            rospy.loginfo(f"New direction: {self.DIRECTIONS[self.current_direction_index].name}")
        time.sleep(0.3)
        self._record_frame()

    def handle_intersection(self):
        self.robot.stop()
        time.sleep(0.3)
        while True:
            planned_dir_label = self.navigator.get_next_direction_label(self.current_node_id,self.planned_path)
            if not planned_dir_label:
                rospy.logerr("Không tìm thấy bước tiếp theo.")
                self._set_state(RobotState.DEAD_END)
                return
            action = self.map_absolute_to_relative(planned_dir_label, self.DIRECTIONS[self.current_direction_index])
            if action is None:
                self._set_state(RobotState.DEAD_END)
                return
            rospy.loginfo(f"[Intersection] Planned action: {action} -> {planned_dir_label}")
            break

        if action=="straight": self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
        elif action=="right": self.turn_robot(90)
        elif action=="left": self.turn_robot(-90)
        else:
            self._set_state(RobotState.DEAD_END)
            return

        next_idx = self.planned_path.index(self.current_node_id)+1
        self.target_node_id = self.planned_path[next_idx]
        rospy.loginfo(f"Next node: {self.target_node_id}")
        self._set_state(RobotState.LEAVING_INTERSECTION)

    def _record_frame(self):
        if self.video_writer is not None and self.latest_image is not None:
            frame = self.draw_debug_info(self.latest_image)
            if frame is not None:
                self.video_writer.write(frame)

    def draw_debug_info(self,image):
        if image is None: return None
        debug_frame = image.copy()
        cv2.rectangle(debug_frame,(0,self.ROI_Y),(self.WIDTH-1,self.ROI_Y+self.ROI_H),(0,255,0),1)
        cv2.rectangle(debug_frame,(0,self.LOOKAHEAD_ROI_Y),(self.WIDTH-1,self.LOOKAHEAD_ROI_Y+self.LOOKAHEAD_ROI_H),(0,255,255),1)
        cv2.putText(debug_frame,f"State: {self.current_state.name}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if self.current_state==RobotState.DRIVING_STRAIGHT:
            line_center = self._get_line_center(image,self.ROI_Y,self.ROI_H)
            if line_center is not None:
                cv2.line(debug_frame,(line_center,self.ROI_Y),(line_center,self.ROI_Y+self.ROI_H),(0,0,255),2)
        return debug_frame

    def run(self):
        rospy.loginfo("Bắt đầu vòng lặp...")
        time.sleep(3)
        self.detector.start_scanning()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.current_state==RobotState.WAITING_FOR_LINE:
                self.robot.stop()
                if self.latest_image is None: rate.sleep(); continue
                lookahead = self._get_line_center(self.latest_image,self.LOOKAHEAD_ROI_Y,self.LOOKAHEAD_ROI_H)
                execution = self._get_line_center(self.latest_image,self.ROI_Y,self.ROI_H)
                if lookahead is not None and execution is not None:
                    self._set_state(RobotState.DRIVING_STRAIGHT)

            elif self.current_state==RobotState.DRIVING_STRAIGHT:
                if self.latest_image is None: self.robot.stop(); rate.sleep(); continue
                if self.detector.process_detection():
                    self.robot.stop(); time.sleep(0.3)
                    self.current_node_id = self.target_node_id
                    if self.current_node_id==self.navigator.end_node:
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()
                    continue
                lookahead_center = self._get_line_center(self.latest_image,self.LOOKAHEAD_ROI_Y,self.LOOKAHEAD_ROI_H)
                execution_center = self._get_line_center(self.latest_image,self.ROI_Y,self.ROI_H)
                if execution_center is not None:
                    self.correct_course(execution_center)
                elif lookahead_center is None:
                    self._set_state(RobotState.APPROACHING_INTERSECTION)
                else:
                    self.robot.stop()

            elif self.current_state==RobotState.APPROACHING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
                if rospy.get_time()-self.state_change_time>self.INTERSECTION_APPROACH_DURATION:
                    self.robot.stop()
                    self.current_node_id=self.target_node_id
                    if self.current_node_id==self.navigator.end_node:
                        self._set_state(RobotState.GOAL_REACHED)
                    else:
                        self._set_state(RobotState.HANDLING_EVENT)
                        self.handle_intersection()

            elif self.current_state==RobotState.LEAVING_INTERSECTION:
                self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
                if rospy.get_time()-self.state_change_time>self.INTERSECTION_CLEARANCE_DURATION:
                    self._set_state(RobotState.REACQUIRING_LINE)

            elif self.current_state==RobotState.REACQUIRING_LINE:
                self.robot.set_motors(self.BASE_SPEED,self.BASE_SPEED)
                line_center = self._get_line_center(self.latest_image,self.ROI_Y,self.ROI_H)
                if line_center is not None: self._set_state(RobotState.DRIVING_STRAIGHT)
                elif rospy.get_time()-self.state_change_time>self.LINE_REACQUIRE_TIMEOUT: self._set_state(RobotState.DEAD_END)

            elif self.current_state==RobotState.DEAD_END:
                self.robot.stop(); break

            elif self.current_state==RobotState.GOAL_REACHED:
                self.robot.stop(); break

            self._record_frame()
            rate.sleep()

        self.cleanup()

    def cleanup(self):
        self.robot.stop()
        if self.video_writer: self.video_writer.release()
        if self.detector: self.detector.stop_scanning()
        rospy.loginfo("Đã dừng robot, giải phóng tài nguyên.")

    def map_absolute_to_relative(self,target_label,current_dir):
        target_dir=self.LABEL_TO_DIRECTION_ENUM.get(target_label)
        if target_dir is None: return None
        diff=(target_dir.value - current_dir.value + 4)%4
        return ['straight','right',None,'left','turn_around'][diff] if diff!=2 else 'turn_around'

def main():
    rospy.init_node('jetbot_controller_node', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--map_type", type=str, default="map_z", choices=["map_a","map_b","map_z"])
    args = parser.parse_args()
    try:
        controller = JetBotController(token=args.token,map_type=args.map_type)
        controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node bị ngắt.")
    except Exception as e:
        rospy.logerr(f"Lỗi: {e}", exc_info=True)

if __name__=='__main__':
    main()
