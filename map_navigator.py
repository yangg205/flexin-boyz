# map_navigator.py
import rospy

import json
import networkx as nx
import math

class MapNavigator:
    def __init__(self, map_file_path):
        """
        Khởi tạo bộ điều hướng bản đồ.
        - Tải bản đồ từ file JSON.
        - Xây dựng một đồ thị có hướng (directed graph) bằng networkx.
        - Xác định điểm bắt đầu và kết thúc.
        """
        self.graph = nx.DiGraph()
        self.nodes_data = {}
        self.start_node = None
        self.end_node = None
        self._opposite_direction = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        self._load_map(map_file_path)

    def _load_map(self, map_file_path):
        """Tải và phân tích cú pháp file JSON của bản đồ."""
        with open(map_file_path, 'r') as f:
            data = json.load(f)

        for node in data['nodes']:
            self.nodes_data[node['id']] = node
            self.graph.add_node(node['id'], **node)
            if node['type'] == 'start':
                self.start_node = node['id']
            elif node['type'] == 'end':
                self.end_node = node['id']

        for edge in data['edges']:
            # Thêm cạnh xuôi và cạnh ngược để robot có thể đi hai chiều
            self.graph.add_edge(edge['source'], edge['target'], label=edge['label'])
            opposite_label = self._opposite_direction.get(edge['label'])
            if opposite_label:
                self.graph.add_edge(edge['target'], edge['source'], label=opposite_label)
    
    def _heuristic(self, node1_id, node2_id):
        """Hàm heuristic (khoảng cách Euclid) cho thuật toán A*."""
        pos1 = self.nodes_data[node1_id]
        pos2 = self.nodes_data[node2_id]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)

    def find_path(self, start_node_id, end_node_id, banned_edges=None):
        """
        Tìm đường đi ngắn nhất từ điểm bắt đầu đến điểm kết thúc bằng thuật toán A*.
        :param start_node_id: ID của node bắt đầu.
        :param end_node_id: ID của node kết thúc.
        :param banned_edges: List các cạnh (u, v) bị cấm, dùng để tìm đường lại.
        :return: List các ID node trên đường đi, hoặc None nếu không có đường.
        """
        graph_to_search = self.graph.copy()
        if banned_edges:
            graph_to_search.remove_edges_from(banned_edges)

        try:
            path = nx.astar_path(
                graph_to_search, 
                start_node_id, 
                end_node_id, 
                heuristic=self._heuristic
            )
            rospy.loginfo(f"Path found: {path}")
            return path
        except nx.NetworkXNoPath:
            return None

    def get_next_direction_label(self, current_node_id, path):
        """
        Từ đường đi đã cho, xác định hướng đi tiếp theo (N, E, S, W) từ node hiện tại.
        """
        if not path or current_node_id not in path:
            return None
        
        current_index = path.index(current_node_id)
        if current_index + 1 >= len(path):
            return None # Đã đến đích

        next_node_id = path[current_index + 1]
        edge_data = self.graph.get_edge_data(current_node_id, next_node_id)
        rospy.loginfo(f"Next direction from {current_node_id} to {next_node_id}: {edge_data.get('label', None)}")
        return edge_data.get('label', None)
    
    def get_neighbor_by_direction(self, current_node_id, direction_label):
        """
        Tìm ID của node hàng xóm từ node hiện tại theo một hướng cho trước.
        :param current_node_id: ID của node hiện tại.
        :param direction_label: Hướng đi ('N', 'E', 'S', 'W').
        :return: ID của node hàng xóm, hoặc None nếu không có.
        """
        for neighbor in self.graph.neighbors(current_node_id):
            edge_data = self.graph.get_edge_data(current_node_id, neighbor)
            if edge_data and edge_data.get('label') == direction_label:
                rospy.loginfo(f"Neighbor of {current_node_id} in direction {direction_label}: {neighbor}")
                return neighbor
        return None