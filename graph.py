#######################################################################
# Name: env.py
# - Simple graph class with utility functions.
# - Adapted from https://gist.github.com/betandr/541a1f6466b6855471de5ca30b74cb31
#######################################################################

import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from parameter import *


class Edge:
    def __init__(self, to_node: Tuple[int, int], length: float) -> None:
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self) -> None:
        self.nodes: Set[Tuple[int, int]] = set()
        self.edges: Dict[Tuple[int, int], Dict[Tuple[int, int], Edge]] = dict()

    def add_node(self, node: Tuple[int, int]) -> None:
        """將節點加入圖中。

        Args:
            node (Tuple[int, int]): 節點標識（例如 tuple 座標）。

        Returns:
            None
        """
        self.nodes.add(node)

    def add_edge(
        self, from_node: Tuple[int, int], to_node: Tuple[int, int], length: float
    ) -> None:
        """在圖中加入一條有向邊（若必要會建立 from_node 的 edge dict）。

        Args:
            from_node (Tuple[int, int]): 起始節點標識。
            to_node (Tuple[int, int]): 目標節點標識。
            length (float): 此邊的距離或成本。

        Returns:
            None
        """
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]

        from_node_edges[to_node] = edge

    def clear_edge(
        self, from_node: Tuple[int, int], remove_bidirectional_edges: bool = False
    ) -> None:
        """清除指定節點的所有出邊；可選同時移除其他節點指向該節點的邊。

        Args:
            from_node (Tuple[int, int]): 要清除出邊的節點。
            remove_bidirectional_edges (bool): 若為 True，同時掃描並移除其他節點指向 from_node 的邊。

        Returns:
            None
        """
        if remove_bidirectional_edges:
            for to_node in list(self.edges.keys()):
                if from_node in self.edges[to_node]:
                    del self.edges[to_node][from_node]

        if from_node in self.edges:
            self.edges[from_node] = dict()

    def clear_node(
        self, node: Tuple[int, int], remove_bidirectional_edges: bool = False
    ) -> None:
        """從圖中移除節點（同時清除其出邊），可選同時移除其他節點指向該節點的邊。

        Args:
            node (Tuple[int, int]): 要移除的節點。
            remove_bidirectional_edges (bool): 若為 True，同時移除其他節點指向該節點的邊。

        Returns:
            None
        """
        self.clear_edge(node, remove_bidirectional_edges=remove_bidirectional_edges)

        # Remove the node from the set of nodes
        if node in self.nodes:
            self.nodes.remove(node)

    def is_connected_bfs(
        self,
        start_node: Tuple[int, int],
        criteria: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """使用廣度優先搜尋檢查圖的連通性。

        若 criteria 為 None，檢查整個圖是否連通（所有節點皆可到達）。
        若提供 criteria（節點集合），檢查 criteria 是否均能被從 start_node 到達。

        Args:
            start_node (Tuple[int, int]): 搜尋起點。
            criteria (Optional[Set[Tuple[int, int]]]): 若提供，為要檢查是否被覆蓋的節點集合（hashable 元素的集合）。

        Returns:
            Tuple[bool, List[Tuple[int, int]]]:
                is_connected (bool): 若符合連通性條件回傳 True，否則 False。
                visited_list (List[Tuple[int, int]]): 實際訪問到的節點清單（list）。
        """
        # An empty graph is considered connected
        if len(self.nodes) == 0:
            return True, []

        visited = set()
        queue = deque([start_node])

        while len(queue) != 0:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                if current_node in self.edges:
                    neighbor_nodes = set(self.edges[current_node].keys())
                    queue.extend(neighbor_nodes - visited)

        # To make sure edges cleared properly
        visited = set(map(tuple, visited)).intersection(set(map(tuple, self.nodes)))
        if criteria is not None:
            criteria_bounded = criteria.intersection(set(map(tuple, self.nodes)))

        if criteria is None:
            is_connected = len(visited) == len(self.nodes)
        else:
            is_connected = criteria_bounded.issubset(visited)
        return is_connected, list(visited)


def h(
    index: Union[Tuple[int, int], np.ndarray],
    destination: Union[Tuple[int, int], np.ndarray],
) -> float:
    """A* 使用的啟發式函式（歐式距離）。

    Args:
        index (Union[Tuple[int, int], np.ndarray]): 當前節點座標或索引。
        destination (Union[Tuple[int, int], np.ndarray]): 目標節點座標或索引。

    Returns:
        float: 當前節點到目標節點的估計距離（歐式距離）。
    """
    current = np.array(index)
    end = np.array(destination)
    h = np.linalg.norm(end - current)
    return h


def a_star(start: Tuple[int, int], destination: Tuple[int, int], graph: Graph) -> Tuple[
    Optional[List[Tuple[int, int]]],
    float,
    Set[Tuple[int, int]],
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]]],
]:
    """A* 最短路徑搜尋。

    Args:
        start (Tuple[int, int]): 起始節點（與 graph.nodes 的元素型態一致）。
        destination (Tuple[int, int]): 目標節點。
        graph (Graph): 圖物件，需包含 nodes 與 edges 結構。

    Returns:
        Tuple[Optional[List[Tuple[int, int]]], float, Set[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
            path (Optional[List[Tuple[int, int]]]): 若找到路徑回傳節點序列（包含 start 與 destination），否則 None。
            cost (float): 路徑成本或大數（代表失敗）。
            closed_list (Set[Tuple[int, int]]): 已展開的節點集合（供分析或視覺化）。
            edges_explored_list (Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]): 展開的邊（來源節點列表, 目標節點列表）。

    Raises:
        無：函式會在找不到路徑時回傳 None 並以大成本表示失敗。
    """
    if start == destination:
        return [], 0, set([]), ([], [])
    if (
        tuple(start) in graph.edges
        and tuple(destination) in graph.edges[tuple(start)].keys()
    ):
        cost = graph.edges[tuple(start)][tuple(destination)].length
        return [start, destination], cost, set([]), ([], [])
    open_list = {start}
    closed_list = set([])
    edges_explored_list = ([], [])

    g = {start: 0}
    parents = {start: start}

    while len(open_list) > 0:
        n = None  # current node with lowest f cost
        h_n = 1e5

        # Choose vertex with next lowest f cost
        for v in open_list:
            h_v = h(v, destination)
            if n is not None:
                h_n = h(n, destination)
            if n is None or g[v] + h_v < g[n] + h_n:
                n = v

        if n is None:
            # print('[1] Path does not exist!')
            return None, 1e5, closed_list, edges_explored_list

        # If found destination, backtrack to generate astar path
        if n == destination:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start)
            reconst_path.reverse()
            return reconst_path, g[destination], closed_list, edges_explored_list

        if tuple(n) in graph.edges:
            for edge in graph.edges[tuple(n)].values():

                m = tuple(edge.to_node)
                edges_explored_list[0].append(n)
                edges_explored_list[1].append(m)
                cost = edge.length

                if m in closed_list:
                    continue

                if m not in open_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost

                elif g[m] > g[n] + cost:
                    parents[m] = n
                    g[m] = g[n] + cost

        open_list.remove(n)
        closed_list.add(n)

    # print('[2] Path does not exist!')
    return None, 1e5, closed_list, edges_explored_list
