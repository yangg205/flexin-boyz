#!/usr/bin/env python3
import json
import rospy
import time
from collections import deque
from jetbot import Robot
from get_map import fetch_map  # tận dụng class get_map đã có

class ProblemA:
    """
    Class giải quyết Problem A: điều khiển robot đi từ Start -> End
    trên bản đồ tải từ API, tránh vật cản bằng LIDAR.
    """

    def __init__(self, token, map_type="map_z"):
        # Khởi tạo robot Jetbot
        self.robot = Robot()
        # TODO: Khởi tạo LIDAR
        # self.lidar = LidarSensor()  # Giả sử module LidarSensor đã cài đặt
        self.speed = 0.25   # tốc độ di chuyển
        self.turn_time = 1  # thời gian rẽ 90 độ

        # Load map từ API
        self.load_map_from_api(token, map_type)

        # Tính đường đi ngắn nhất
        self.path = self.bfs_shortest_path()

        # Giả sử hướng ban đầu là Bắc
        self.current_dir = "N"

    def load_map_from_api(self, token, map_type):
        """
        Lấy map từ API và chuyển thành cấu trúc nodes/edges/graph
        """
        data = fetch_map(token=token, map_type=map_type)
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.edges = data["edges"]
        self.graph = {}
        for e in self.edges:
            # Tính hướng từ source -> target dựa vào tọa độ
            src = self.nodes[e["source"]]
            tgt = self.nodes[e["target"]]
            dx, dy = tgt["x"] - src["x"], tgt["y"] - src["y"]
            if dx == 1: direction = "E"
            elif dx == -1: direction = "W"
            elif dy == 1: direction = "S"
            elif dy == -1: direction = "N"
            else: direction = "N"  # fallback
            self.graph.setdefault(e["source"], []).append((e["target"], direction))
        # Xác định start/end
        self.start = next(n["id"] for n in data["nodes"] if n["type"].lower()=="start")
        self.end = next(n["id"] for n in data["nodes"] if n["type"].lower()=="end")

    def bfs_shortest_path(self):
        """
        Tìm đường đi ngắn nhất từ start -> end
        """
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
        """
        Trả về hướng từ u -> v
        """
        for nxt, label in self.graph.get(u, []):
            if nxt == v:
                return label
        return None

    def relative_action(self, current_dir, target_dir):
        """
        Quyết định hành động (straight, left, right, turn_around)
        dựa trên hướng hiện tại và hướng cần đi
        """
        dirs = ["N","E","S","W"]
        ci, ti = dirs.index(current_dir), dirs.index(target_dir)
        diff = (ti-ci) % 4
        if diff == 0: return "straight"
        if diff == 1: return "right"
        if diff == 3: return "left"
        return "turn_around"

    def drive_action(self, action):
        """
        Di chuyển robot theo hành động
        - straight: đi thẳng (kiểm tra vật cản bằng LIDAR)
        - left/right/turn_around: rẽ
        """
        if action == "straight":
            # TODO: Kiểm tra vật cản trước khi đi thẳng
            # lidar_distance = self.lidar.get_distance()
            # if lidar_distance < 0.2:  # khoảng cách tối thiểu an toàn (m)
            #     self.robot.stop()
            #     rospy.logwarn("Vật cản phía trước! Dừng robot")
            #     return
            self.robot.set_motors(self.speed, self.speed)
            time.sleep(1.0)  # chạy qua giao lộ
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
    # TODO: Thay token của team bạn vào đây
    token = input("Nhập token của đội bạn: ").strip()
    if not token:
        print("Bạn phải nhập token!")
        return
    pa = ProblemA(token=token, map_type="map_z")
    pa.run()


if __name__=="__main__":
    main()
