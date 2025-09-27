#!/usr/bin/env python3
import json
import rospy
import time
from collections import deque
from jetbot import Robot
from get_map import fetch_map
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import argparse

class LidarSensor:
    """Wrapper đơn giản cho LIDAR để đo khoảng cách phía trước."""
    def __init__(self, topic="/scan", forward_angle=10.0, min_distance=0.25):
        self.latest_scan = None
        self.forward_angle = forward_angle
        self.min_distance = min_distance
        self.sub = rospy.Subscriber(topic, LaserScan, self.callback)

    def callback(self, scan: LaserScan):
        self.latest_scan = scan

    def get_forward_distance(self):
        if self.latest_scan is None:
            return float('inf')
        scan = self.latest_scan
        ranges = np.array(scan.ranges)
        angle_increment_deg = math.degrees(scan.angle_increment)
        center_index = len(ranges) // 2
        offset = int(self.forward_angle / angle_increment_deg)
        start = max(0, center_index - offset)
        end = min(len(ranges), center_index + offset + 1)
        forward_ranges = ranges[start:end]
        valid_ranges = forward_ranges[np.isfinite(forward_ranges)]
        if len(valid_ranges) == 0:
            return float('inf')
        return np.min(valid_ranges)

class ProblemA:
    """Điều khiển robot từ Start -> End, tránh vật cản bằng LIDAR."""
    def __init__(self, token, map_type="map_z"):
        self.robot = Robot()
        self.speed = 0.25
        self.turn_time = 1
        self.lidar = LidarSensor()
        self.load_map_from_api(token, map_type)
        self.path = self.bfs_shortest_path()
        if self.path is None:
            raise ValueError("Không tìm được đường đi từ Start -> End")
        self.current_dir = "N"

    def load_map_from_api(self, token, map_type):
        data = fetch_map(token=token, map_type=map_type)
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]
        self.graph = {}
        for e in self.edges:
            src = self.nodes[e["source"]]
            tgt = self.nodes[e["target"]]
            dx, dy = tgt["x"] - src["x"], tgt["y"] - src["y"]
            if dx == 1: direction = "E"
            elif dx == -1: direction = "W"
            elif dy == 1: direction = "S"
            elif dy == -1: direction = "N"
            else: direction = "N"
            self.graph.setdefault(e["source"], []).append((e["target"], direction))
        self.start = next((n["id"] for n in data["nodes"] if n["type"].lower() == "start"), None)
        self.end = next((n["id"] for n in data["nodes"] if n["type"].lower() == "end"), None)
        if self.start is None or self.end is None:
            raise ValueError("Không tìm thấy Start hoặc End trong map")

    def bfs_shortest_path(self):
        queue = deque([(self.start,[self.start])])
        visited = set([self.start])
        while queue:
            cur, path = queue.popleft()
            if cur == self.end:
                return path
            for nxt,_ in self.graph.get(cur,[]):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path+[nxt]))
        return None

    def get_edge_label(self, u, v):
        for nxt, label in self.graph.get(u, []):
            if nxt == v:
                return label
        return None

    def relative_action(self, current_dir, target_dir):
        dirs = ["N","E","S","W"]
        ci, ti = dirs.index(current_dir), dirs.index(target_dir)
        diff = (ti-ci) % 4
        if diff == 0: return "straight"
        if diff == 1: return "right"
        if diff == 3: return "left"
        return "turn_around"

    def drive_action(self, action):
        if action == "straight":
            distance = self.lidar.get_forward_distance()
            if distance < 0.2:
                self.robot.stop()
                rospy.logwarn(f"Vật cản phía trước ({distance:.2f} m)! Dừng robot")
                return
            self.robot.set_motors(self.speed, self.speed)
            time.sleep(1.0)
        elif action == "right":
            self.robot.set_motors(self.speed, -self.speed)
            time.sleep(self.turn_time)
        elif action == "left":
            self.robot.set_motors(-self.speed, self.speed)
            time.sleep(self.turn_time)
        elif action == "turn_around":
            self.robot.set_motors(self.speed, -self.speed)
            time.sleep(self.turn_time*2)
        self.robot.stop()
        time.sleep(0.2)

    def run(self):
        rospy.loginfo(f"Đường đi: {self.path}")
        for i in range(len(self.path)-1):
            u, v = self.path[i], self.path[i+1]
            target_dir = self.get_edge_label(u,v)
            action = self.relative_action(self.current_dir, target_dir)
            rospy.loginfo(f"Đi {action} từ node {u} -> {v}")
            self.drive_action(action)
            self.current_dir = target_dir
        rospy.loginfo("ĐÃ ĐẾN ĐÍCH!")
        self.robot.stop()


def main():
    rospy.init_node("problem_a_node", anonymous=True)
    parser = argparse.ArgumentParser(description="Chạy robot Problem A")
    parser.add_argument("--token", required=True, help="Token của đội")
    parser.add_argument("--map_type", default="map_z", choices=["map_a","map_b","map_z"], help="Loại map")
    args = parser.parse_args()

    pa = ProblemA(token=args.token, map_type=args.map_type)
    pa.run()


if __name__=="__main__":
    main()
