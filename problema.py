#!/usr/bin/env python3
import json
import rospy
import time
import heapq
import numpy as np
import argparse
from jetbot import Robot
from sensor_msgs.msg import LaserScan
from get_map import fetch_map   # ⚡ import hàm lấy map


class ProblemA:
    def __init__(self, token, map_type):
        self.robot = Robot()
        self.speed = 0.25
        self.turn_time = 1

        # LiDAR
        self.latest_scan = None
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)

        # ⚡ luôn luôn load map từ get_map
        data = fetch_map(token=token, map_type=map_type)

        self.parse_map(data)
        self.path = self.a_star_shortest_path()

    # ---------------- MAP ----------------
    def parse_map(self, data):
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]

        # Graph
        self.graph = {}
        for e in self.edges:
            u, v = e["source"], e["target"]
            cost = e.get("cost", 1)
            dir_uv = self.compute_direction(self.nodes[u], self.nodes[v])
            self.graph.setdefault(u, []).append((v, dir_uv, cost))

            dir_vu = self.compute_direction(self.nodes[v], self.nodes[u])
            self.graph.setdefault(v, []).append((u, dir_vu, cost))

        # Start / End
        if data.get("startingPositions"):
            self.start = data["startingPositions"][0]
        else:
            self.start = next(n["id"] for n in data["nodes"] if n["type"].lower() == "start")

        if data.get("destinationPositions"):
            self.end = data["destinationPositions"][0]
        else:
            self.end = next(n["id"] for n in data["nodes"] if n["type"].lower() == "end")

        # Hướng ban đầu
        start_node = self.nodes[self.start]
        self.current_dir = start_node.get("dir", "E")

    # ---------------- PATHFIND ----------------
    def compute_direction(self, node_from, node_to):
        dx = node_to["x"] - node_from["x"]
        dy = node_to["y"] - node_from["y"]
        if dx == 1: return "E"
        if dx == -1: return "W"
        if dy == 1: return "S"
        if dy == -1: return "N"
        return None

    def heuristic(self, node_id):
        n1 = self.nodes[node_id]
        n2 = self.nodes[self.end]
        return abs(n1["x"] - n2["x"]) + abs(n1["y"] - n2["y"])

    def a_star_shortest_path(self):
        open_set = []
        heapq.heappush(open_set, (0, 0, self.start, [self.start]))
        g_score = {self.start: 0}
        visited = set()

        while open_set:
            f, g, cur, path = heapq.heappop(open_set)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == self.end:
                return path

            for nxt, _dir, cost in self.graph.get(cur, []):
                new_g = g + cost
                if new_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = new_g
                    f_score = new_g + self.heuristic(nxt)
                    heapq.heappush(open_set, (f_score, new_g, nxt, path + [nxt]))
        return None

    # ---------------- ACTION ----------------
    def relative_action(self, current_dir, target_dir):
        dirs = ["N", "E", "S", "W"]
        ci, ti = dirs.index(current_dir), dirs.index(target_dir)
        diff = (ti - ci) % 4
        if diff == 0: return "straight"
        if diff == 1: return "right"
        if diff == 3: return "left"
        return "turn_around"

    def drive_action(self, action):
        if action == "straight":
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
            time.sleep(self.turn_time * 2)
        self.robot.stop()
        time.sleep(0.2)

    # ---------------- LIDAR ----------------
    def lidar_callback(self, msg):
        self.latest_scan = msg

    def lidar_check_obstacle(self, threshold=0.4):
        if not self.latest_scan:
            return False
        ranges = np.array(self.latest_scan.ranges)
        front = np.concatenate([ranges[-15:], ranges[:15]])  # trước mặt ±15°
        front = front[np.isfinite(front)]
        if len(front) == 0:
            return False
        return np.min(front) < threshold

    # ---------------- RUN ----------------
    def run(self):
        rospy.loginfo(f"Path: {self.path}")
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            dir_uv = self.compute_direction(self.nodes[u], self.nodes[v])
            action = self.relative_action(self.current_dir, dir_uv)

            # nếu u là nút giao => check lidar
            if len(self.graph[u]) > 2:
                rospy.loginfo(f"Đang ở nút giao {u}, kiểm tra LiDAR...")
                if self.lidar_check_obstacle():
                    rospy.logwarn("Có vật cản tại nút giao! Dừng lại.")
                    self.robot.stop()
                    time.sleep(2.0)
                    continue

            rospy.loginfo(f"Đi {action} từ {u}->{v}")
            self.drive_action(action)
            self.current_dir = dir_uv
        rospy.loginfo("ĐÃ ĐẾN ĐÍCH!")
        self.robot.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Team token để lấy bản đồ")
    parser.add_argument("--map", required=True, help="Loại bản đồ, ví dụ: map_z")
    args = parser.parse_args()

    rospy.init_node("problem_a_node", anonymous=True)
    pa = ProblemA(token=args.token, map_type=args.map)
    pa.run()


if __name__ == "__main__":
    main()
