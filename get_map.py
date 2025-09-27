#!/usr/bin/env python3
import json
import rospy
import time
import requests
import heapq
from jetbot import Robot

DOMAIN = "https://hackathon2025-dev.fpt.edu.vn"
MAP_ENDPOINT = "/api/maps/get_active_map/"

# 🔧 Toàn bộ config di chuyển để tiện chỉnh sửa
MOTION_CONFIG = {
    "speed": 0.25,        # tốc độ cơ bản
    "forward_time": 4.0,  # thời gian chạy thẳng qua 1 ô
    "turn_time": 1.0,     # thời gian rẽ 90 độ
    "stop_delay": 0.2     # delay sau khi stop
}

class ProblemA:
    def __init__(self, token=None, map_type="map_z", local_file="map.json"):
        self.robot = Robot()
        self.speed = MOTION_CONFIG["speed"]
        self.forward_time = MOTION_CONFIG["forward_time"]
        self.turn_time = MOTION_CONFIG["turn_time"]
        self.stop_delay = MOTION_CONFIG["stop_delay"]

        # Load map từ server nếu có token, nếu không thì dùng file local
        if token:
            data = self.load_map_from_server(token, map_type)
        else:
            data = self.load_map_from_file(local_file)

        self.parse_map(data)
        self.path = self.a_star_shortest_path()

    def load_map_from_server(self, token, map_type):
        url = DOMAIN + MAP_ENDPOINT
        resp = requests.get(url, params={"token": token, "map_type": map_type})
        resp.raise_for_status()
        rospy.loginfo("Đã tải bản đồ từ server")
        return resp.json()

    def load_map_from_file(self, map_file):
        with open(map_file, "r") as f:
            data = json.load(f)
        rospy.loginfo("Đã tải bản đồ từ file local")
        return data

    def parse_map(self, data):
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]

        # Xây dựng graph kề với hướng
        self.graph = {}
        for e in self.edges:
            u, v = e["source"], e["target"]
            cost = e.get("cost", 1)
            dir_uv = self.compute_direction(self.nodes[u], self.nodes[v])
            self.graph.setdefault(u, []).append((v, dir_uv, cost))

            # nếu map vô hướng thì thêm cạnh ngược
            dir_vu = self.compute_direction(self.nodes[v], self.nodes[u])
            self.graph.setdefault(v, []).append((u, dir_vu, cost))

        # Lấy node Start
        if data.get("startingPositions"):
            self.start = data["startingPositions"][0]
        else:
            self.start = next(n["id"] for n in data["nodes"] if n["type"].lower() == "start")

        # Lấy node End
        if data.get("destinationPositions") and len(data["destinationPositions"]) > 0:
            self.end = data["destinationPositions"][0]
        else:
            self.end = next(n["id"] for n in data["nodes"] if n["type"].lower() == "end")

        # Hướng ban đầu
        start_node = self.nodes[self.start]
        self.current_dir = start_node.get("dir", "E")

    def compute_direction(self, node_from, node_to):
        dx = node_to["x"] - node_from["x"]
        dy = node_to["y"] - node_from["y"]
        if dx == 1: return "E"
        if dx == -1: return "W"
        if dy == 1: return "S"   # y tăng nghĩa là đi xuống
        if dy == -1: return "N"  # y giảm nghĩa là đi lên
        return None

    def heuristic(self, node_id):
        """Heuristic A*: khoảng cách Manhattan"""
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
            self.robot.set_motors(self.speed, self.speed)
            time.sleep(self.forward_time)
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
        time.sleep(self.stop_delay)

    def run(self):
        rospy.loginfo(f"Path: {self.path}")
        for i in range(len(self.path)-1):
            u, v = self.path[i], self.path[i+1]
            dir_uv = self.compute_direction(self.nodes[u], self.nodes[v])
            action = self.relative_action(self.current_dir, dir_uv)
            rospy.loginfo(f"Đi {action} từ {u}->{v}")
            self.drive_action(action)
            self.current_dir = dir_uv
        rospy.loginfo("ĐÃ ĐẾN ĐÍCH!")
        self.robot.stop()


def main():
    rospy.init_node("problem_a_node", anonymous=True)
    token = "c7f272d7cff632a6fd875954774ed7a7"  # thay bằng team token thật
    pa = ProblemA(token=token, map_type="map_z")
    pa.run()

if __name__ == "__main__":
    main()
