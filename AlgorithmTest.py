import heapq


def dijkstra_with_travel_time(graph, start, end, start_time, time_constraints):
    """
    改进后的Dijkstra算法，支持多个不同时段的通行限制。

    参数:
    - graph: 图的邻接表表示 {'A': {'B': travel_time, ...}, ...}
    - start: 起始节点
    - end: 终止节点
    - start_time: 起始时间
    - time_constraints: 时间窗口限制，格式 {('u', 'v'): [(start_time1, end_time1), (start_time2, end_time2), ...]}

    返回:
    - 最短路径和总通行时长，如果没有路径，返回 None。
    """
    # 初始化

    shortest_paths = {node: (float('inf'), None, None) for node in graph}  # (最短距离, 前驱节点, 到达时间)
    shortest_paths[start] = (0, None, start_time)  # 起点
    priority_queue = [(0, start, start_time)]  # (通行时间, 当前节点, 当前时间)

    while priority_queue:
        current_distance, current_node, current_time = heapq.heappop(priority_queue)

        # 如果当前节点是终点，返回路径
        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = shortest_paths[current_node][1]
            return path[::-1], current_distance

        # 遍历相邻节点
        for neighbor, travel_time in graph[current_node].items():
            next_time = current_time + travel_time  # 到达邻居节点的时间

            # 检查多个时间窗口限制
            if (current_node, neighbor) in time_constraints:
                restricted_windows = time_constraints[(current_node, neighbor)]
                is_restricted = False
                for restricted_start, restricted_end in restricted_windows:
                    # 检查进入和离开时间
                    if not (next_time <= restricted_start or current_time >= restricted_end):
                        is_restricted = True
                        break
                if is_restricted:
                    continue  # 如果受到限制，跳过该边

            # 计算新的通行时间
            distance = current_distance + travel_time
            if distance < shortest_paths[neighbor][0]:  # 找到更短路径
                shortest_paths[neighbor] = (distance, current_node, next_time)  # 更新路径信息
                heapq.heappush(priority_queue, (distance, neighbor, next_time))

    return None  # 如果没有路径满足条件


def dijkstra_with_stopover(graph, start, stopover, end, start_time, time_constraints):
    """
    支持起点、经停点和终点的路径规划，带时间窗口限制。

    参数:
    - graph: 图的邻接表表示，包含边的通行时长 {'A': {'B': travel_time, ...}, ...}
    - start: 起始节点
    - stopover: 经停点
    - end: 终止节点
    - start_time: 起始时间
    - time_constraints: 时间窗口限制，格式为 {('u', 'v'): (start_time, end_time)}

    返回:
    - 总路径和通行时长，如果没有可行路径，返回 None。
    """
    # 阶段1: 从起点到经停点
    path1, travel_time1 = dijkstra_with_travel_time(graph, start, stopover, start_time, time_constraints)
    if not path1:
        return None  # 如果第一阶段不可达，则直接返回

    # 阶段2: 从经停点到终点（起始时间为到达经停点的时间）
    arrival_time_at_stopover = start_time + travel_time1
    path2, travel_time2 = dijkstra_with_travel_time(graph, stopover, end, arrival_time_at_stopover, time_constraints)
    if not path2:
        return None  # 如果第二阶段不可达，则直接返回

    # 合并两段路径
    total_path = path1 + path2[1:]  # 避免重复经停点
    total_travel_time = travel_time1 + travel_time2

    return total_path, total_travel_time


# 定义图
graph = {
    'S': {'A': 2, 'D': 4},
    'A': {'S': 2, 'B': 3, 'D': 2},
    'B': {'A': 3, 'C': 4, 'F': 5},
    'C': {'B': 4, 'E': 6, 'G': 3},
    'D': {'S': 4, 'A': 2, 'F': 3, 'H': 5},
    'E': {'C': 6, 'G': 2},
    'F': {'B': 5, 'D': 3, 'G': 4},
    'G': {'C': 3, 'E': 2, 'F': 4, 'T': 6},
    'H': {'D': 5, 'T': 7},
    'T': {'G': 6, 'H': 7}
}

# 时间窗口限制
time_constraints = {
    ('A', 'B'): [(5, 10), (0, 2)],  # 边 A-B 有两个限制窗口
    # ('C', 'G'): [(7, 15)],            # 边 C-G 有一个限制窗口
    ('D', 'H'): [(3, 8)],             # 边 D-H 有一个限制窗口
    ('F', 'G'): [(6, 12)]             # 边 F-G 有一个限制窗口
}

# 输入参数
start_node = 'S'
stopover_node = 'C'
end_node = 'T'
start_time = 0

# 调用算法
path, travel_time = dijkstra_with_stopover(graph, start_node, stopover_node, end_node, start_time, time_constraints)

# 输出结果
if path:
    print(f"最短路径: {' -> '.join(path)}")
    print(f"总通行时长: {travel_time}")
else:
    print("没有可行路径满足条件。")


l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in l:
    print(l)
    print(i)
    l.pop(0)
