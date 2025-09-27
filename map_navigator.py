import rospy
import json
import networkx as nx
import math

class MapNavigator:
    def __init__(self, map_data):
        """
        Initialize the map navigator.
        - Load map from JSON data (passed directly instead of a file).
        - Build a directed graph using networkx.
        - Identify start and end nodes.
        """
        self.graph = nx.DiGraph()
        self.nodes_data = {}
        self.start_node = None
        self.end_node = None
        self.load_nodes = []
        self._opposite_direction = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
        self._load_map(map_data)

    def _load_map(self, map_data):
        """Load and parse map data from JSON."""
        # Process nodes
        for node in map_data['nodes']:
            self.nodes_data[node['id']] = node
            self.graph.add_node(node['id'], **node)
            if node['type'] == 'Start':  # Case-sensitive to match JSON
                self.start_node = node['id']
            elif node['type'] == 'End':
                self.end_node = node['id']
            elif node['type'] == 'Load':
                self.load_nodes.append(node['id'])

        # Process edges
        for edge in map_data['edges']:
            # Add forward edge
            self.graph.add_edge(edge['source'], edge['target'], label=edge['label'])
            # Add reverse edge with opposite direction
            opposite_label = self._opposite_direction.get(edge['label'])
            if opposite_label:
                self.graph.add_edge(edge['target'], edge['source'], label=opposite_label)

        rospy.loginfo(f"Map loaded: {len(self.nodes_data)} nodes, {self.graph.number_of_edges()} edges")
        rospy.loginfo(f"Start node: {self.start_node}, End node: {self.end_node}, Load nodes: {self.load_nodes}")

    def _heuristic(self, node1_id, node2_id):
        """Euclidean distance heuristic for A* algorithm."""
        pos1 = self.nodes_data[node1_id]
        pos2 = self.nodes_data[node2_id]
        return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)

    def find_path(self, start_node_id=None, end_node_id=None, banned_edges=None, visit_load_nodes=False):
        """
        Find the shortest path from start to end using A*.
        :param start_node_id: ID of the start node (defaults to map's start node).
        :param end_node_id: ID of the end node (defaults to map's end node).
        :param banned_edges: List of edges (u, v) to avoid.
        :param visit_load_nodes: If True, path must include all load nodes before reaching end.
        :return: List of node IDs in the path, or None if no path exists.
        """
        if start_node_id is None:
            start_node_id = self.start_node
        if end_node_id is None:
            end_node_id = self.end_node

        if start_node_id is None or end_node_id is None:
            rospy.logerr("Start or end node not defined.")
            return None

        graph_to_search = self.graph.copy()
        if banned_edges:
            graph_to_search.remove_edges_from(banned_edges)

        try:
            if visit_load_nodes and self.load_nodes:
                # Create a path that visits all load nodes before reaching the end
                path = [start_node_id]
                current_node = start_node_id
                remaining_loads = self.load_nodes.copy()

                while remaining_loads:
                    # Find the shortest path to the closest unvisited load node
                    min_dist = float('inf')
                    next_load = None
                    temp_path = None
                    for load in remaining_loads:
                        try:
                            p = nx.astar_path(
                                graph_to_search,
                                current_node,
                                load,
                                heuristic=self._heuristic
                            )
                            dist = sum(self._heuristic(p[i], p[i+1]) for i in range(len(p)-1))
                            if dist < min_dist:
                                min_dist = dist
                                next_load = load
                                temp_path = p
                        except nx.NetworkXNoPath:
                            continue
                    if not temp_path:
                        rospy.logerr("No path to remaining load nodes.")
                        return None
                    # Append path to the next load node (excluding the start node already in path)
                    path.extend(temp_path[1:])
                    current_node = next_load
                    remaining_loads.remove(next_load)

                # After visiting all load nodes, find path to end node
                final_path = nx.astar_path(
                    graph_to_search,
                    current_node,
                    end_node_id,
                    heuristic=self._heuristic
                )
                path.extend(final_path[1:])  # Exclude the current node
            else:
                # Direct path from start to end
                path = nx.astar_path(
                    graph_to_search,
                    start_node_id,
                    end_node_id,
                    heuristic=self._heuristic
                )
            rospy.loginfo(f"Path found: {path}")
            return path
        except nx.NetworkXNoPath:
            rospy.logerr(f"No path from {start_node_id} to {end_node_id}")
            return None

    def get_next_direction_label(self, current_node_id, path):
        """
        Determine the next direction (N, E, S, W) from the current node along the path.
        :param current_node_id: Current node ID.
        :param path: List of node IDs in the path.
        :return: Direction label or None if at the end or invalid.
        """
        if not path or current_node_id not in path:
            rospy.logwarn(f"Invalid path or node {current_node_id} not in path.")
            return None

        current_index = path.index(current_node_id)
        if current_index + 1 >= len(path):
            rospy.loginfo("Reached destination.")
            return None

        next_node_id = path[current_index + 1]
        edge_data = self.graph.get_edge_data(current_node_id, next_node_id)
        if not edge_data:
            rospy.logerr(f"No edge from {current_node_id} to {next_node_id}")
            return None

        direction = edge_data.get('label')
        rospy.loginfo(f"Next direction from {current_node_id} to {next_node_id}: {direction}")
        return direction

    def get_neighbor_by_direction(self, current_node_id, direction_label):
        """
        Find the neighbor node ID from the current node in the given direction.
        :param current_node_id: Current node ID.
        :param direction_label: Direction ('N', 'E', 'S', 'W').
        :return: Neighbor node ID or None if none exists.
        """
        for neighbor in self.graph.neighbors(current_node_id):
            edge_data = self.graph.get_edge_data(current_node_id, neighbor)
            if edge_data and edge_data.get('label') == direction_label:
                rospy.loginfo(f"Neighbor of {current_node_id} in direction {direction_label}: {neighbor}")
                return neighbor
        rospy.logwarn(f"No neighbor of {current_node_id} in direction {direction_label}")
        return None