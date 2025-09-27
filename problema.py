#!/usr/bin/env python3
import json
import rospy
import time
from collections import deque
from jetbot import Robot

class ProblemA:
    def __init__(self, map_file="map.json"):
        self.robot = Robot()
        self.speed = 2.0   # tăng tốc hơn bình thường
        self.turn_time = 0.3  # thời gian rẽ 90 độ (tối ưu hóa)
        self.load_map(map_file)
        self.path = self.bfs_shortest_path()
        self.current_dir = "N"  # giả sử ban đầu hướng Bắc

    def load_map(self, map_file):
        with open(map_file, "r") as f:
            data = json.load(f)
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]
        self.graph = {}
        for e in self.edges:
            self.graph.setdefault(e["source"], []).append((e["target"], e["label"]))
        self.start = next(n["id"] for n in data["nodes"] if n["type"]=="start")
        self.end = next(n["id"] for n in data["nodes"] if n["type"]=="end")

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
        for e in self.edges:
            if e["source"]==u and e["target"]==v:
                return e["label"]
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
        if action=="straight":
            self.robot.set_motors(self.speed, self.speed)
            time.sleep(1.0)  # chạy thẳng qua giao lộ
        elif action=="right":
            self.robot.set_motors(self.speed, -self.speed)
            time.sleep(self.turn_time)
        elif action=="left":
            self.robot.set_motors(-self.speed, self.speed)
            time.sleep(self.turn_time)
        elif action=="turn_around":
            self.robot.set_motors(self.speed, -self.speed)
            time.sleep(self.turn_time*2)
        self.robot.stop()
        time.sleep(0.2)

    def run(self):
        rospy.loginfo(f"Path: {self.path}")
        for i in range(len(self.path)-1):
            u, v = self.path[i], self.path[i+1]
            target_dir = self.get_edge_label(u,v)
            action = self.relative_action(self.current_dir, target_dir)
            rospy.loginfo(f"Đi {action} từ {u}->{v}")
            self.drive_action(action)
            self.current_dir = target_dir
        rospy.loginfo("ĐÃ ĐẾN ĐÍCH!")
        self.robot.stop()

def main():
    rospy.init_node("problem_a_node", anonymous=True)
    pa = ProblemA("map.json")
    pa.run()

if __name__=="__main__":
    main()
