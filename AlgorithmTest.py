import heapq


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.index = 0

    def push(self, key, value):
        # Use a tuple (value, index, key) to ensure the queue is sorted by value
        heapq.heappush(self.queue, [value, key])
        self.index += 1

    def pop(self):
        # Pop the smallest value and return the entire tuple (value, index, key)
        if self.index > 0:
            self.index -= 1
            return heapq.heappop(self.queue)
        else:
            return None

    def is_empty(self):
        return len(self.queue) == 0

    def top(self):
        # Return the smallest value without removing it
        if self.index > 0:
            return self.queue[0]
        else:
            return None


# pq = PriorityQueue()
# pq.push('task1', 5)
# pq.push('task2', 1)
# pq.push('task3', 3)
#
# while not pq.is_empty():
#     print(pq.top())
#     print(pq.pop())


def dijkstra_with_travel_time(graph, start, end, start_time, time_constraints):
    """
    修改后的Dijkstra算法，接受时间点集合作为约束

    参数:
    - graph: 图的邻接表表示 {'A': {'B': travel_time, ...}, ...}
    - start: 起始节点
    - end: 终止节点
    - start_time: 起始时间
    - time_constraints: 格式 {('u', 'v'): [time_point1, time_point2, ...]}

    返回:
    - 最短路径和总通行时长，如果没有路径，返回 None
    """
    # 初始化
    shortest_paths = {node: (float('inf'), None, None) for node in graph}
    shortest_paths[start] = (0, None, start_time)
    priority_queue = [(0, start, start_time)]

    while priority_queue:
        current_distance, current_node, current_time = heapq.heappop(priority_queue)

        # 如果到达终点，返回路径
        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = shortest_paths[current_node][1]
            return path[::-1], current_distance

        # 遍历相邻节点
        for neighbor, travel_time in graph[current_node].items():
            # 计算到达下一节点的时间
            next_time = current_time + travel_time

            # 检查是否有约束时间点落在行驶区间内
            if (current_node, neighbor) in time_constraints:
                restricted_points = time_constraints[(current_node, neighbor)]
                can_traverse = True

                # 检查是否有时间点落在行驶时间段内（不含终点时间）
                # 如果车辆刚好在约束时间点离开，这是允许的
                # 如果车辆在约束时间点仍在道路上行驶，这是不允许的
                for point in restricted_points:
                    if current_time <= point < next_time:
                        # 车辆在point时刻还在道路上行驶，不允许
                        can_traverse = False
                        break
                    # next_time == point是允许的（车辆刚好在约束时间点离开）
                    # point < current_time或point > next_time也是允许的（车辆在约束前已进入或约束后才进入）

                if not can_traverse:
                    continue  # 如果受到限制，跳过该边

            # 更新路径
            distance = current_distance + travel_time
            if distance < shortest_paths[neighbor][0]:
                shortest_paths[neighbor] = (distance, current_node, next_time)
                heapq.heappush(priority_queue, (distance, neighbor, next_time))

    return None, float('inf')  # 没有找到有效路径


def dijkstra_plus(graph, start, stopover, end, start_time, time_constraints):
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
        return None, float('inf')  # 如果第一阶段不可达，则直接返回

    # 阶段2: 从经停点到终点（起始时间为到达经停点的时间）
    arrival_time_at_stopover = start_time + travel_time1
    path2, travel_time2 = dijkstra_with_travel_time(graph, stopover, end, arrival_time_at_stopover, time_constraints)
    if not path2:
        return None, float('inf')  # 如果第二阶段不可达，则直接返回

    # 合并两段路径
    total_path = path1 + path2[1:]  # 避免重复经停点
    total_travel_time = travel_time1 + travel_time2

    return total_path, total_travel_time


import heapq

# 简单测试图
graph = {
    0: {1: 2, 2: 5},
    1: {3: 3},
    2: {3: 2},
    3: {4: 1},
    4: {}
}

# 时间点约束：在边(0,2)上不能在时间点7行驶(但可以在时间点7离开)
time_constraints = {(0, 1): [2]}

# 测试从0到4的路径，起始时间为0
path, time = dijkstra_with_travel_time(graph, 0, 1, 0, time_constraints)
print(f"路径: {path}, 行驶时间: {time}")
# 预期结果: 路径: [0, 2, 3, 4], 行驶时间: 8
# 因为时间点7大于从0到2的行驶时间5，所以不受约束影响

# 另一个测试：添加约束让它必须选择另一条路径
path, time = dijkstra_with_travel_time(graph, 0, 4, 0, time_constraints)
print(f"新路径: {path}, 行驶时间: {time}")
# 预期结果: 路径: [0, 1, 3, 4], 行驶时间: 6
# 因为时间点3落在从0到2的行驶区间[0,5]内，所以选择了另一条路径