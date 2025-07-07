import random
import numpy as np
import math
import copy
import heapq
import time
from numba.cpython.randomimpl import permutation_impl
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from collections import defaultdict
from functools import lru_cache
import concurrent.futures

T = 3


def calculate_shortest_path(center, origin, destination, edge_weights):
    """
    使用Dijkstra算法计算最短路径，考虑交叉口延迟

    Args:
        center: 调度中心对象
        origin: 起点节点ID
        destination: 终点节点ID
        edge_weights: 边权重字典

    Returns:
        包含边ID的最短路径列表
    """

    # print(f"计算从 {origin} 到 {destination} 的最短路径")
    # 初始化距离和前驱节点
    distances = {node_id: float('inf') for node_id in center.nodes}
    distances[origin] = 0
    predecessors = {node_id: None for node_id in center.nodes}

    # 使用优先队列存储待处理节点和进入该节点的边ID
    # (距离, 节点ID, 进入该节点的边ID)
    queue = [(0, origin, None)]
    visited = set()

    while queue:
        current_distance, current_node, incoming_edge_id = heapq.heappop(queue)

        # 对于特定节点和进入边的组合，只处理一次
        if (current_node, incoming_edge_id) in visited:
            continue

        visited.add((current_node, incoming_edge_id))

        if current_node == destination:
            break

        # 遍历当前节点的所有出边
        for edge_id, edge in center.edges.items():
            if edge.origin == current_node:
                neighbor = edge.destination

                # 基本边权重
                weight = edge_weights.get(edge_id, float('inf'))

                # 计算交叉口延迟 - 使用边ID计算
                intersection_delay = 0
                if incoming_edge_id is not None and hasattr(center.nodes[current_node], 'signal'):
                    # 直接使用边ID计算交叉口延迟
                    signal_key = (incoming_edge_id, edge_id)
                    if hasattr(center.nodes[current_node], 'calculate_wait'):
                        intersection_delay = center.nodes[current_node].calculate_wait(incoming_edge_id, edge_id)
                    elif signal_key in center.nodes[current_node].signal:
                        intersection_delay = center.nodes[current_node].signal[signal_key]

                # 总权重 = 边权重 + 交叉口延迟
                total_weight = weight + intersection_delay
                distance = current_distance + total_weight

                # 检查是否找到更短路径
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = (current_node, edge_id)
                    heapq.heappush(queue, (distance, neighbor, edge_id))

    # 重建路径
    if predecessors[destination] is None:
        return []

    path = []
    current = destination

    while current != origin:
        prev_node, edge_id = predecessors[current]
        path.append(edge_id)
        current = prev_node

    path.reverse()
    return path


def update_flow(center, vehicle, path):
    """
    更新车辆路径上的流量

    Args:
        center: 调度中心对象
        vehicle: 车辆对象
        path: 车辆分配的路径（边ID列表）
    """
    # 获取当前时间
    current_time = getattr(center, 'current_time', 0)

    # 遍历路径上的每条边，更新流量
    for edge_id in path:
        edge = center.edges[edge_id]
        o, d = edge.origin, edge.destination

        # 确保edge_timely_estimated_load有相应的结构
        if (o, d) not in center.edge_timely_estimated_load:
            center.edge_timely_estimated_load[(o, d)] = {}

        if current_time not in center.edge_timely_estimated_load[(o, d)]:
            center.edge_timely_estimated_load[(o, d)][current_time] = [0]

        # 增加流量
        center.edge_timely_estimated_load[(o, d)][current_time][0] += 1


def diversified_path_assignment(center, vehicle_assignments, edge_weights, real_path_results):
    """
    为车辆分配多样化路径

    Args:
        center: 调度中心对象
        vehicle_assignments: 车辆分配信息字典
        edge_weights: 边权重字典
        real_path_results: 预计算的路径结果

    Returns:
        包含每辆车路径信息的字典
    """
    # 按OD对和车辆特性分组
    vehicle_groups = {}
    for v_id, assignment in vehicle_assignments.items():
        vehicle = center.vehicles[v_id]
        cs_id = assignment['cs']
        key = (vehicle.origin, vehicle.destination, cs_id, vehicle.anxiety)
        if key not in vehicle_groups:
            vehicle_groups[key] = []
        vehicle_groups[key].append(v_id)

    # 结果路径字典
    path_assignment = {}

    # 为每组车辆分配多样化路径
    for key, vehicles in vehicle_groups.items():
        origin, destination, cs_id, anxiety = key

        # 根据焦虑程度确定路径模式
        if anxiety == 1:  # 焦虑车辆
            # 生成路径：origin -> destination -> cs
            path1 = get_path_segments(center, origin, destination, edge_weights, real_path_results)
            path2 = get_path_segments(center, destination, cs_id, edge_weights, real_path_results)
            path_options = [(path1 + path2, sum(edge_weights.get(edge, float('inf')) for edge in path1 + path2))]
        else:  # 普通车辆
            # 生成路径：origin -> cs -> destination
            path1 = get_path_segments(center, origin, cs_id, edge_weights, real_path_results)
            path2 = get_path_segments(center, cs_id, destination, edge_weights, real_path_results)
            path_options = [(path1 + path2, sum(edge_weights.get(edge, float('inf')) for edge in path1 + path2))]

            # 尝试生成额外的K条路径
            alt_paths = generate_k_alternative_paths(center, origin, destination, cs_id, edge_weights, anxiety)
            path_options.extend(alt_paths)

        # 为该组内车辆分配路径
        if path_options[0][-1] == float('inf'):
            print(f"path1: {path1}, path2: {path2}")
            for edge in path1 + path2:
                if edge not in edge_weights:
                    print(f"警告: 边 {edge} 在边权重中未找到")
                print(f"边权重: {edge_weights.get(edge, float('inf'))}")
        path_costs = {i: cost for i, (path, cost) in enumerate(path_options)}
        print(f"为车辆组 {key} 生成路径选项: {path_options}")
        print(f"<UNK> {key} <UNK>: {path_costs}")

        # 计算路径选择概率（使用Logit模型，成本低的路径有更高概率）
        total_cost = sum(1.0 / cost for cost in path_costs.values())
        path_probs = {i: (1.0 / cost) / total_cost for i, cost in path_costs.items()}

        print(f"为车辆组 {key} 分配路径，路径概率: {path_probs}")
        for v_id in vehicles:
            # 路径随机选择
            paths = list(path_probs.keys())
            probs = list(path_probs.values())
            chosen_index = random.choices(paths, weights=probs, k=1)[0]
            chosen_path = path_options[chosen_index][0]

            path_assignment[v_id] = {'path': chosen_path}

    return path_assignment


def get_path_segments(center, origin, destination, edge_weights, real_path_results):
    """获取两点间的路径段，将节点路径转换为边路径"""
    if (origin, destination) in real_path_results and real_path_results[(origin, destination)]:
        # 从real_path_results获取节点序列
        node_path = real_path_results[(origin, destination)][0][0]

        # 将节点序列转换为边ID序列
        edge_path = []
        for i in range(len(node_path) - 1):
            from_node = node_path[i]
            to_node = node_path[i + 1]

            # 找到连接这两个节点的边
            for edge_id, edge in center.edges.items():
                if edge.origin == from_node and edge.destination == to_node:
                    edge_path.append(edge_id)
                    break

        return edge_path
    else:
        return calculate_shortest_path(center, origin, destination, edge_weights)


def generate_k_alternative_paths(center, origin, destination, cs_id, edge_weights, anxiety, k=3):
    """生成K条备选路径"""
    alt_paths = []

    # 使用改进的K最短路径算法
    if anxiety == 1:  # 焦虑车辆
        od_paths = improved_k_shortest_paths(center, origin, destination, edge_weights, k)
        for od_path in od_paths:
            cs_path = calculate_shortest_path(center, destination, cs_id, edge_weights)
            complete_path = od_path + cs_path
            path_cost = sum(edge_weights.get(edge, float('inf')) for edge in complete_path)
            alt_paths.append((complete_path, path_cost))
    else:  # 普通车辆
        ocs_paths = improved_k_shortest_paths(center, origin, cs_id, edge_weights, k)
        csd_paths = improved_k_shortest_paths(center, cs_id, destination, edge_weights, k)

        # 组合不同的路径段
        for ocs_path in ocs_paths:
            for csd_path in csd_paths:
                complete_path = ocs_path + csd_path
                path_cost = sum(edge_weights.get(edge, float('inf')) for edge in complete_path)
                alt_paths.append((complete_path, path_cost))

    return alt_paths



def dispatch_vehicles_with_frank_wolfe_update(center, sol, charge_v, OD_ratio, cs_for_choice, real_path_results,
                                              anxiety_OD_ratio=None, anxiety_cs_for_choice=None, max_iter=20,
                                              convergence_threshold=1e-4):
    """
    使用Frank-Wolfe算法进行交通分配优化的车辆调度函数

    Args:
        center: 调度中心对象
        sol: 求解结果
        charge_v: 需要充电的车辆ID列表
        OD_ratio: OD对的比例字典
        cs_for_choice: 每个OD对可选充电站字典
        real_path_results: 预计算的路径结果
        anxiety_OD_ratio: 焦虑车辆OD比例
        anxiety_cs_for_choice: 焦虑车辆可选充电站
        max_iter: Frank-Wolfe最大迭代次数
        convergence_threshold: 收敛阈值
    """
    print("开始执行Frank-Wolfe算法进行车辆调度...")

    # (1) 按center内的信息建图，并计算初始流量
    # 初始化边权重字典
    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[edge_id] = edge.free_time  # 初始权重使用自由流时间

    # 初始化边流量数组
    num_edges = max(edge_weights.keys()) + 1 if edge_weights else 0
    edge_flows = np.zeros(num_edges)

    # 添加背景流量 - 使用每条道路的k属性乘以容量
    for edge_id in edge_weights:
        edge = center.edges[edge_id]
        # 使用道路的k属性乘以容量作为背景流量
        if hasattr(edge, 'k') and edge.capacity and "all" in edge.capacity:
            background_flow = edge.k * edge.capacity["all"][0]
        else:
            background_flow = 0
        edge_flows[edge_id] = background_flow
    print(f"初始化背景流量完成")

    # (2) 接受sol然后按比例分配车辆至充电站
    vehicle_assignments = {}
    od_len = len(OD_ratio.keys()) if OD_ratio else 0

    print(f"为{len(charge_v)}辆车分配充电站...")
    for v_id in charge_v:
        vehicle = center.vehicles[v_id]

        if vehicle.anxiety == 1:  # 焦虑车辆
            if anxiety_OD_ratio and anxiety_cs_for_choice:
                od = (vehicle.origin, vehicle.destination)
                if od in anxiety_OD_ratio:
                    od_index = list(anxiety_OD_ratio.keys()).index(od)
                    cs_probs = sol[od_index + od_len]
                    chosen_cs = np.random.choice(anxiety_cs_for_choice[od], p=cs_probs)
                    vehicle_assignments[v_id] = {
                        'cs': chosen_cs,
                        'type': 'anxiety',
                        'route': [(vehicle.origin, vehicle.destination), (vehicle.destination, chosen_cs)]
                    }
        else:  # 普通车辆
            od = (vehicle.origin, vehicle.destination)
            if od in OD_ratio:
                od_index = list(OD_ratio.keys()).index(od)
                cs_probs = sol[od_index]
                chosen_cs = np.random.choice(cs_for_choice[od], p=cs_probs)
                vehicle_assignments[v_id] = {
                    'cs': chosen_cs,
                    'type': 'normal',
                    'route': [(vehicle.origin, chosen_cs), (chosen_cs, vehicle.destination)]
                }

    print(f"成功为{len(vehicle_assignments)}辆车分配充电站")

    # (3) 整理出每类（起点，终点，充电站）组的车数量（即流量）
    flow_groups = {}  # (origin, destination, cs_id, type) -> 车辆数量
    vehicle_groups = {}  # (origin, destination, cs_id, type) -> 车辆ID列表

    for v_id, assignment in vehicle_assignments.items():
        vehicle = center.vehicles[v_id]
        cs_id = assignment['cs']
        od_type = assignment['type']

        if od_type == 'anxiety':
            # 焦虑车辆：起点 -> 终点 -> 充电站
            key = (vehicle.origin, vehicle.destination, cs_id, 'anxiety')
        else:
            # 普通车辆：起点 -> 充电站 -> 终点
            key = (vehicle.origin, vehicle.destination, cs_id, 'normal')

        if key not in flow_groups:
            flow_groups[key] = 0
            vehicle_groups[key] = []

        flow_groups[key] += 1
        vehicle_groups[key].append(v_id)

    print(f"共有{len(flow_groups)}个OD-CS组合")

    # 初始化路径集合和路径流量
    group_paths = {}  # 每个组的当前主要路径
    group_paths_history = {}  # 每个组的历史路径列表
    path_flows = {}  # 路径流量字典 (group_key, path_tuple) -> 流量

    # 记录震荡检测
    previous_alphas = []
    previous_flows = []
    oscillation_detected = False

    # Frank-Wolfe 主迭代
    for iteration in range(max_iter):
        print(f"Frank-Wolfe算法迭代 {iteration + 1}")

        # 根据当前流量更新边权重
        for edge_id in edge_weights:
            if edge_id < len(edge_flows):
                # 获取边的信息
                edge = center.edges[edge_id]
                flow = edge_flows[edge_id]

                # 使用BPR函数更新边权重
                if hasattr(edge, 'capacity') and edge.capacity and "all" in edge.capacity:
                    capacity = edge.capacity["all"][0]
                    if capacity > 0:
                        # BPR函数: t = t0 * (1 + a * (v/c)^b)
                        a, b = 0.15, 4  # BPR函数参数
                        volume_capacity_ratio = flow / capacity
                        weight = edge.free_time * (1 + a * (volume_capacity_ratio ** b))
                    else:
                        weight = edge.free_time
                else:
                    weight = edge.free_time

                edge_weights[edge_id] = weight

        # (4) 全有全无分配（进行每次迭代的流量分配）
        # 重置辅助流量
        auxiliary_flows = np.zeros(num_edges)
        new_group_paths = {}

        # 为每个组计算最短路径
        for key, flow in flow_groups.items():
            origin, destination, cs_id, od_type = key

            if od_type == 'anxiety':
                # 焦虑车辆: 起点 -> 终点 -> 充电站
                path_o_to_d = calculate_shortest_path(center, origin, destination, edge_weights)
                path_d_to_cs = calculate_shortest_path(center, destination, cs_id, edge_weights)
                full_path = path_o_to_d + path_d_to_cs
            else:
                # 普通车辆: 起点 -> 充电站 -> 终点
                path_o_to_cs = calculate_shortest_path(center, origin, cs_id, edge_weights)
                path_cs_to_d = calculate_shortest_path(center, cs_id, destination, edge_weights)
                full_path = path_o_to_cs + path_cs_to_d

            new_group_paths[key] = full_path

            # 将流量添加到辅助流量数组
            for edge_id in full_path:
                if edge_id < len(auxiliary_flows):
                    auxiliary_flows[edge_id] += flow

            # 记录历史路径
            if key not in group_paths_history:
                group_paths_history[key] = []

            # 将新路径添加到历史路径中（如果不存在）
            path_tuple = tuple(full_path)
            if path_tuple not in [tuple(p) for p in group_paths_history[key]]:
                group_paths_history[key].append(full_path)

        # (5) 迭代得到流量结果
        # 如果是第一次迭代，直接使用全有全无分配结果
        if iteration == 0:
            edge_flows = auxiliary_flows.copy()
            group_paths = new_group_paths.copy()

            # 初始化路径流量
            for key, path in group_paths.items():
                path_tuple = tuple(path)
                path_flows[(key, path_tuple)] = flow_groups[key]

            # 保存当前流量用于震荡检测
            previous_flows.append(edge_flows.copy())
            continue

        # 检测震荡
        if len(previous_flows) >= 3:
            flow_diff1 = np.sum(np.abs(previous_flows[-1] - previous_flows[-2]))
            flow_diff2 = np.sum(np.abs(previous_flows[-2] - previous_flows[-3]))
            flow_diff_ratio = abs(flow_diff1 - flow_diff2) / (flow_diff1 + 1e-10)

            if flow_diff_ratio < 0.05 and iteration > 5:
                oscillation_detected = True
                print(f"检测到流量震荡，将使用MSA或平滑步长")

        # 计算最优步长 alpha
        def objective_function(alpha):
            # 计算当前流量分配下的系统总出行时间
            temp_flows = edge_flows + alpha * (auxiliary_flows - edge_flows)
            total_cost = 0
            for edge_id in edge_weights:
                if edge_id < len(temp_flows):
                    edge = center.edges[edge_id]
                    flow = temp_flows[edge_id]

                    # 使用积分形式的BPR函数
                    if hasattr(edge, 'capacity') and edge.capacity and "all" in edge.capacity:
                        capacity = edge.capacity["all"][0]
                        if capacity > 0:
                            a, b = 0.15, 4  # BPR函数参数
                            # 计算BPR函数的积分: t0*x + t0*a*x^(b+1)/((b+1)*c^b)
                            free_time = edge.free_time
                            total_cost += free_time * flow
                            total_cost += free_time * a * (flow ** (b + 1)) / ((b + 1) * (capacity ** b))
                        else:
                            total_cost += edge.free_time * flow
                    else:
                        total_cost += edge.free_time * flow

            # 添加探索项，防止步长过早变为0
            exploration_term = 0.01 * alpha * (1 - alpha)
            return total_cost - exploration_term

        # 使用scipy的minimize_scalar来找最优步长
        bounds = (0, 1)
        if oscillation_detected and iteration > 10:
            # 如果检测到震荡，使用MSA方法固定步长
            alpha = 1.0 / (iteration + 1)
            print(f"应用MSA步长: {alpha:.4f}")
        else:
            # 使用scipy的优化器找最优步长
            result = minimize_scalar(
                objective_function,
                bounds=bounds,
                method='bounded'
            )
            alpha = result.x

            # 记录步长历史用于震荡检测
            previous_alphas.append(alpha)
            if len(previous_alphas) > 3:
                previous_alphas.pop(0)

        # 应用最小步长防止收敛过早停止
        alpha = max(alpha, 0.01)
        print(f"迭代 {iteration + 1} 的最优步长: {alpha:.4f}")

        # 计算流量变化量
        flow_change = np.sum(np.abs(auxiliary_flows - edge_flows))
        relative_change = flow_change / (np.sum(edge_flows) + 1e-10)

        # 更新边流量
        edge_flows = edge_flows + alpha * (auxiliary_flows - edge_flows)

        # 保存当前流量用于震荡检测
        previous_flows.append(edge_flows.copy())
        if len(previous_flows) > 3:
            previous_flows.pop(0)

        # 更新路径流量
        for key in group_paths:
            old_path = group_paths[key]
            new_path = new_group_paths[key]
            old_path_tuple = tuple(old_path)
            new_path_tuple = tuple(new_path)

            # 获取该组的总流量
            group_flow = flow_groups[key]

            # 更新旧路径流量（减少alpha比例）
            if (key, old_path_tuple) in path_flows:
                path_flows[(key, old_path_tuple)] = path_flows[(key, old_path_tuple)] * (1 - alpha)

            # 更新新路径流量（增加alpha比例）
            if (key, new_path_tuple) not in path_flows:
                path_flows[(key, new_path_tuple)] = 0
            path_flows[(key, new_path_tuple)] = path_flows[(key, new_path_tuple)] + alpha * group_flow

            # 更新当前主要路径
            group_paths[key] = new_path

        # 检查收敛
        print(f"迭代 {iteration + 1} 流量相对变化: {relative_change:.6f}")

        # 如果震荡且迭代次数足够，或者相对变化很小，则认为已收敛
        if (oscillation_detected and iteration > 10) and (relative_change < convergence_threshold):
            print(f"Frank-Wolfe算法在第 {iteration + 1} 次迭代后收敛")
            break

    # (6) 将流量结果转化为路径结果并分配给对应车辆
    print("将优化结果分配给各车辆...")
    path_assignment = {}

    # 为每个组计算最终路径选择概率
    group_path_probs = {}

    for key, vehicle_ids in vehicle_groups.items():
        # 获取该组所有历史路径
        all_paths = group_paths_history[key]

        # 计算该组每条路径的流量
        path_probabilities = []
        for path in all_paths:
            path_tuple = tuple(path)
            flow = path_flows.get((key, path_tuple), 0)
            path_probabilities.append((path, flow))

        # 计算总流量
        total_flow = sum(flow for _, flow in path_probabilities)

        # 如果总流量为0，使用边权重计算概率
        if total_flow < 1e-6:
            path_costs = []
            for path in all_paths:
                path_cost = sum(edge_weights[edge_id] for edge_id in path if edge_id in edge_weights)
                path_costs.append(path_cost)

            # 使用路径成本的倒数作为选择概率
            inv_costs = [1.0 / max(0.1, cost) for cost in path_costs]
            total_inv_cost = sum(inv_costs)
            probabilities = [(path, inv_cost / total_inv_cost) for path, inv_cost in zip(all_paths, inv_costs)]
        else:
            # 使用流量比例作为概率
            probabilities = [(path, flow / total_flow) for path, flow in path_probabilities]

        # 保存路径选择概率
        group_path_probs[key] = {
            'paths': [p for p, _ in probabilities],
            'probs': [p for _, p in probabilities]
        }

        # 根据概率为每个车辆分配路径
        paths = [p for p, _ in probabilities]
        probs = [p for _, p in probabilities]

        for v_id in vehicle_ids:
            if not paths:  # 如果没有可用路径
                print(f"警告：车辆 {v_id} 没有可用路径")
                continue

            chosen_path_idx = np.random.choice(len(paths), p=probs)
            path_assignment[v_id] = {'path': paths[chosen_path_idx]}

    # 最终处理：更新车辆信息
    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        cs_id = vehicle_assignments[v_id]['cs']

        # 选择充电桩功率（选择第一个可用功率）
        available_powers = list(center.charge_stations[cs_id].pile.keys())
        chosen_power = available_powers[0]

        # 更新车辆信息
        vehicle.charge = (cs_id, chosen_power)
        vehicle.path = path_info['path']

        # 更新流量
        update_flow(center, vehicle, vehicle.path)

        # 更新车辆位置信息
        if vehicle.path:
            vehicle.road = vehicle.path[0]
            vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1

            # 更新容量计数
            center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                center.edges[vehicle.road].capacity["all"], 1)
            if vehicle.origin != vehicle.charge[0]:  # 如果起点不是充电站
                center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["charge"], 1)
            center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                center.edges[vehicle.road].capacity[vehicle.next_road], 1)

            vehicle.drive()

    # 输出路径分配统计信息
    print("\n----- 路径分配统计 -----")

    # 按OD对和焦虑等级分组统计
    od_anxiety_path_stats = {}

    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        origin = vehicle.origin
        destination = vehicle.destination
        anxiety = vehicle.anxiety
        cs_id = vehicle_assignments[v_id]['cs']

        # 路径元组（用于统计不同路径）
        path_tuple = tuple(path_info['path'])

        # 创建分组键
        key = (origin, destination, cs_id, anxiety)

        # 初始化统计信息
        if key not in od_anxiety_path_stats:
            od_anxiety_path_stats[key] = {
                'total': 0,
                'path_choices': {}
            }

        # 更新计数
        od_anxiety_path_stats[key]['total'] += 1

        # 更新路径选择
        if path_tuple not in od_anxiety_path_stats[key]['path_choices']:
            od_anxiety_path_stats[key]['path_choices'][path_tuple] = 0
        od_anxiety_path_stats[key]['path_choices'][path_tuple] += 1

    # 输出分组统计信息
    for (origin, destination, cs_id, anxiety), stats in od_anxiety_path_stats.items():
        anxiety_type = "焦虑车辆" if anxiety == 1 else "普通车辆"
        print(f"\nOD对 ({origin} → {destination} → {cs_id}), {anxiety_type}, 总数: {stats['total']}辆")

        # 输出路径选择概率
        for path_tuple, count in sorted(stats['path_choices'].items(), key=lambda x: x[1], reverse=True):
            probability = count / stats['total']
            print(f"  选择概率: {probability:.2%} ({count}/{stats['total']}辆)")

    return path_assignment




# def optimized_frank_wolfe(center, vehicle_assignments, edge_weights, real_path_results, max_iter=10, alpha=0.5):
#     """优化的Frank-Wolfe算法"""
#     # 使用NumPy数组存储流量，提高计算效率
#     num_edges = max(edge_weights.keys()) + 1
#     edge_flows = np.zeros(num_edges)
#
#     # 初始化边流量（背景流量）
#     for edge_id in edge_weights:
#         edge = center.edges[edge_id]
#         o, d = edge.origin, edge.destination
#         current_time = getattr(center, 'current_time', 0)
#
#         if (o, d) in center.edge_timely_estimated_load:
#             load_dict = center.edge_timely_estimated_load[(o, d)]
#             background_flow = load_dict.get(current_time, [0])[0]
#         else:
#             background_flow = 0
#
#         edge_flows[edge_id] = background_flow
#
#     # 缓存最短路径计算
#     path_cache = {}
#
#     # 初始路径分配
#     path_assignment = {}
#     for v_id, assignment in vehicle_assignments.items():
#         path_segments = []
#
#         for segment in assignment['route']:
#             o, d = segment
#             cache_key = (o, d, frozenset(edge_weights.items()))
#
#             if cache_key in path_cache:
#                 segment_path = path_cache[cache_key]
#             elif (o, d) in real_path_results and real_path_results[(o, d)]:
#                 segment_path = real_path_results[(o, d)][0][0]
#             else:
#                 segment_path = calculate_shortest_path(center, o, d, edge_weights)
#                 path_cache[cache_key] = segment_path
#
#             path_segments.extend(segment_path)
#
#         path_assignment[v_id] = {'path': path_segments}
#
#         # 更新边流量
#         for edge_id in path_segments:
#             edge_flows[edge_id] += 1
#
#     # 使用向量化操作优化FW迭代
#     for _ in range(max_iter):
#         # 更新边权重（向量化操作）
#         for edge_id, flow in enumerate(edge_flows):
#             if edge_id in center.edges:
#                 edge = center.edges[edge_id]
#                 capacity = max(1, edge.capacity["all"][0])
#                 edge_weights[edge_id] = edge.free_time * (1 + edge.b * ((flow / capacity) ** edge.power))
#
#         # 清除路径缓存
#         path_cache.clear()
#
#         # 计算新路径（可并行）
#         new_path_assignment = {}
#         new_edge_flows = np.zeros(num_edges)
#
#         def compute_new_path(v_id, assignment):
#             path_segments = []
#             for segment in assignment['route']:
#                 o, d = segment
#                 cache_key = (o, d, frozenset(edge_weights.items()))
#
#                 if cache_key in path_cache:
#                     segment_path = path_cache[cache_key]
#                 else:
#                     segment_path = calculate_shortest_path(center, o, d, edge_weights)
#                     path_cache[cache_key] = segment_path
#
#                 path_segments.extend(segment_path)
#
#             return v_id, {'path': path_segments}
#
#         # 使用并行计算新路径
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             for v_id, path_info in executor.map(lambda x: compute_new_path(*x),
#                                                 vehicle_assignments.items()):
#                 new_path_assignment[v_id] = path_info
#                 for edge_id in path_info['path']:
#                     new_edge_flows[edge_id] += 1
#
#         # 使用向量化操作更新流量
#         edge_flows = edge_flows + alpha * (new_edge_flows - edge_flows)
#
#         # 更新路径分配
#         for v_id in path_assignment:
#             if random.random() < alpha:
#                 path_assignment[v_id] = new_path_assignment[v_id]
#
#     return path_assignment


def improved_k_shortest_paths(center, origin, destination, edge_weights, k=3):
    """改进的K最短路径算法，使用更高效的数据结构，避免修改原始边权重"""
    # 创建边权重的深拷贝，避免修改原始边权重
    if origin == destination:
        return [[]] * k

    edge_weights_copy = edge_weights.copy()

    # 首先获取最短路径
    shortest_path = calculate_shortest_path(center, origin, destination, edge_weights_copy)
    if not shortest_path:
        return []

    # 使用集合存储路径，提高查找效率
    result_paths = [shortest_path]
    result_set = {tuple(shortest_path)}

    # 使用优先队列存储候选路径
    candidates = []

    # 优化路径分解和偏离点计算
    for i in range(k - 1):
        if i >= len(result_paths):
            break

        current_path = result_paths[i]

        # 遍历偏离点
        for j in range(len(current_path)):
            if j >= len(current_path):
                continue

            # 获取偏离节点
            if j == 0:
                spur_node = origin
            else:
                prev_edge = current_path[j - 1]
                spur_node = center.edges[prev_edge].destination

            # 路径前缀
            root_path = current_path[:j]

            # 为每个偏离点计算创建单独的边权重副本
            temp_weights = edge_weights_copy.copy()

            # 避免重复路径 - 修改临时副本
            for path in result_paths:
                if len(path) > j and path[:j] == root_path:
                    if j < len(path):
                        next_edge = path[j]
                        if next_edge in temp_weights:
                            temp_weights[next_edge] = float('inf')

            # 使用临时边权重副本计算偏离路径
            spur_path = calculate_shortest_path(center, spur_node, destination, temp_weights)

            # 如果找到有效路径
            if spur_path:
                candidate_path = root_path + spur_path
                candidate_tuple = tuple(candidate_path)

                if candidate_tuple not in result_set:
                    # 计算路径成本 - 使用原始边权重副本
                    path_cost = sum(edge_weights_copy.get(e, float('inf')) for e in candidate_path)
                    heapq.heappush(candidates, (path_cost, candidate_path))

        # 如果没有更多候选路径，退出
        if not candidates:
            break

        # 获取下一条最短路径
        _, next_path = heapq.heappop(candidates)
        next_tuple = tuple(next_path)

        # 避免重复路径
        while next_tuple in result_set and candidates:
            _, next_path = heapq.heappop(candidates)
            next_tuple = tuple(next_path)

        if next_tuple not in result_set:
            result_paths.append(next_path)
            result_set.add(next_tuple)

    return result_paths


#################################################################################################################

def dispatch_vehicles_with_projected_gradient(center, sol, charge_v, OD_ratio, cs_for_choice, real_path_results,
                                              anxiety_OD_ratio=None, anxiety_cs_for_choice=None, max_iter=50,
                                              convergence_threshold=1e-5):
    """
    使用投影梯度法进行交通分配优化的车辆调度函数，结合列生成法动态生成路径

    Args:
        center: 调度中心对象
        sol: 求解结果
        charge_v: 需要充电的车辆ID列表
        OD_ratio: OD对的比例字典
        cs_for_choice: 每个OD对可选充电站字典
        real_path_results: 预计算的路径结果
        anxiety_OD_ratio: 焦虑车辆OD比例
        anxiety_cs_for_choice: 焦虑车辆可选充电站
        max_iter: 最大迭代次数
        convergence_threshold: 收敛阈值
    """
    print("开始执行投影梯度法(列生成)进行车辆调度...")
    print(f"[检查点1] 输入的车辆数量: {len(charge_v)}")

    # 初始化边权重字典
    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[edge_id] = edge.free_time  # 初始权重使用自由流时间

    # 初始化边流量数组
    num_edges = max(edge_weights.keys()) + 1 if edge_weights else 0
    edge_flows = np.zeros(num_edges)

    # 添加背景流量
    for edge_id in range(num_edges):
        if edge_id in edge_weights:
            edge = center.edges[edge_id]
            if hasattr(edge, 'k') and edge.capacity and "all" in edge.capacity:
                edge_flows[edge_id] = edge.k * edge.capacity["all"][0]

    # 车辆分配到充电站
    vehicle_assignments = {}
    for v_id in charge_v:
        vehicle = center.vehicles[v_id]
        if vehicle.anxiety == 1:  # 焦虑车辆
            if anxiety_OD_ratio and anxiety_cs_for_choice:
                od = (vehicle.origin, vehicle.destination)
                if od in anxiety_OD_ratio:
                    od_index = list(anxiety_OD_ratio.keys()).index(od)
                    cs_probs = sol[od_index]
                    chosen_cs = np.random.choice(anxiety_cs_for_choice[od], p=cs_probs)
                    vehicle_assignments[v_id] = {
                        'cs': chosen_cs,
                        'type': 'anxiety'
                    }
        else:  # 普通车辆
            od = (vehicle.origin, vehicle.destination)
            if od in OD_ratio:
                od_index = list(OD_ratio.keys()).index(od)
                cs_probs = sol[od_index]
                chosen_cs = np.random.choice(cs_for_choice[od], p=cs_probs)
                vehicle_assignments[v_id] = {
                    'cs': chosen_cs,
                    'type': 'normal'
                }

    print(f"[检查点2] 成功分配充电站的车辆数量: {len(vehicle_assignments)}")

    # 按OD-CS-类型分组车辆
    flow_groups = {}  # (origin, destination, cs_id, type) -> 车辆数量
    vehicle_groups = {}  # (origin, destination, cs_id, type) -> 车辆ID列表

    for v_id, assignment in vehicle_assignments.items():
        vehicle = center.vehicles[v_id]
        cs_id = assignment['cs']
        od_type = assignment['type']

        if od_type == 'anxiety':
            # 焦虑车辆：起点 -> 终点 -> 充电站
            key = (vehicle.origin, vehicle.destination, cs_id, 1)
        else:
            # 普通车辆：起点 -> 充电站 -> 终点
            key = (vehicle.origin, vehicle.destination, cs_id, 0)

        if key not in flow_groups:
            flow_groups[key] = 0
            vehicle_groups[key] = []

        flow_groups[key] += 1
        vehicle_groups[key].append(v_id)

    print(f"[检查点3] OD-CS组合数量: {len(flow_groups)}")

    # 初始化路径集合和路径流量
    od_paths = {}  # 每个OD对的路径集合 key -> [path1, path2, ...]
    path_edge_incidence = {}  # 路径-边关联矩阵 (key, path_idx) -> [edge_ids]
    path_flows = {}  # 路径流量 (key, path_idx) -> flow

    # 为每个OD组生成初始路径（只生成一条最短路径）
    print("为每个OD组生成初始最短路径...")
    for key, flow in flow_groups.items():
        origin, destination, cs_id, anxiety = key
        od_paths[key] = []

        # 生成初始最短路径
        if anxiety == 1:
            # 焦虑车辆：起点 -> 终点 -> 充电站
            path1 = calculate_shortest_path(center, origin, destination, edge_weights)
            path2 = calculate_shortest_path(center, destination, cs_id, edge_weights)
            full_path = path1 + path2
        else:
            # 普通车辆：起点 -> 充电站 -> 终点
            path1 = calculate_shortest_path(center, origin, cs_id, edge_weights)
            path2 = calculate_shortest_path(center, cs_id, destination, edge_weights)
            full_path = path1 + path2

        # 添加最短路径到路径集
        if full_path:
            od_paths[key] = [full_path]
            path_key = (key, 0)
            path_edge_incidence[path_key] = full_path
            path_flows[path_key] = flow  # 初始时全部流量分配给最短路径

    # 检查点4：记录初始路径数量
    total_paths = sum(len(paths) for paths in od_paths.values())
    print(f"[检查点4] 初始生成的路径总数: {total_paths}")
    print(f"[检查点5] 初始分配的路径流量总和: {sum(path_flows.values())}")

    # 投影梯度法主循环
    step_size_init = 0.2  # 初始步长
    for iteration in range(max_iter):
        print(f"投影梯度法迭代 {iteration + 1}")

        # 记录迭代开始时的总流量
        if iteration % 10 == 0 or iteration == 0:
            total_edge_flow = np.sum(edge_flows)
            total_path_flow = sum(path_flows.values())
            print(f"[检查点6] 迭代 {iteration} 开始时的总流量: {total_edge_flow}")
            print(f"[检查点6] 迭代 {iteration} 开始时的路径流量总和: {total_path_flow}")

        # 根据当前路径流量计算边流量
        edge_flows = np.zeros(num_edges)
        # 添加背景流量
        for edge_id in range(num_edges):
            if edge_id in edge_weights:
                edge = center.edges[edge_id]
                if hasattr(edge, 'k') and edge.capacity and "all" in edge.capacity:
                    edge_flows[edge_id] = edge.k * edge.capacity["all"][0]

        # 加入路径分配的流量
        for (key, path_idx), flow in path_flows.items():
            path_key = (key, path_idx)
            if path_key in path_edge_incidence:
                for edge_id in path_edge_incidence[path_key]:
                    if edge_id < len(edge_flows):
                        edge_flows[edge_id] += flow

        # 计算每条边的延迟（梯度）
        edge_delays = np.zeros(num_edges)
        delay_dict = {}
        for edge_id in edge_weights:
            if edge_id < len(edge_flows):
                edge = center.edges[edge_id]
                capacity = edge.capacity["all"][0] if edge.capacity and "all" in edge.capacity else 1.0
                flow = edge_flows[edge_id]
                # BPR函数计算延迟
                edge_delays[edge_id] = edge.free_time * (1 + edge.b * ((flow / capacity) ** edge.power))
                delay_dict[edge_id] = edge_delays[edge_id]

        # 计算每条路径的成本梯度
        path_gradients = {}
        for (key, path_idx), flow in path_flows.items():
            path_key = (key, path_idx)
            if path_key in path_edge_incidence:
                path_cost = sum(edge_delays[edge_id] for edge_id in path_edge_incidence[path_key])
                path_gradients[path_key] = path_cost

        # 列生成：为每个OD组生成新路径
        for key, group_flow in flow_groups.items():
            origin, destination, cs_id, anxiety = key

            # 根据当前边延迟重新计算最短路径
            if anxiety == 1:
                # 焦虑车辆：起点 -> 终点 -> 充电站
                new_path1 = calculate_shortest_path(center, origin, destination, delay_dict)
                new_path2 = calculate_shortest_path(center, destination, cs_id, delay_dict)
                new_path = new_path1 + new_path2
            else:
                # 普通车辆：起点 -> 充电站 -> 终点
                new_path1 = calculate_shortest_path(center, origin, cs_id, delay_dict)
                new_path2 = calculate_shortest_path(center, cs_id, destination, delay_dict)
                new_path = new_path1 + new_path2

            # 如果新路径有效且不在现有路径集中，添加到路径集
            if new_path and all(new_path != path for path in od_paths.get(key, [])):
                if key not in od_paths:
                    od_paths[key] = []

                # 添加新路径到路径集
                path_idx = len(od_paths[key])
                od_paths[key].append(new_path)
                path_key = (key, path_idx)
                path_edge_incidence[path_key] = new_path

                # 计算新路径的成本
                path_cost = sum(edge_delays[edge_id] for edge_id in new_path)
                path_gradients[path_key] = path_cost

                # 初始化新路径的流量为0
                path_flows[path_key] = 0

        # 记录更新前的路径流量
        old_path_flows = path_flows.copy()

        # 根据梯度更新路径流量
        step_size = step_size_init / (1 + 0.1 * iteration)  # 递减步长
        new_path_flows = {}

        # 对每个OD组应用投影梯度更新
        for key, group_flow in flow_groups.items():
            if key not in od_paths or not od_paths[key]:
                continue

            paths = od_paths[key]
            path_costs = [(i, path_gradients.get((key, i), float('inf')))
                         for i in range(len(paths))]

            # 按成本排序
            path_costs.sort(key=lambda x: x[1])

            # 使用投影梯度法更新流量
            current_flows = {i: path_flows.get((key, i), 0) for i in range(len(paths))}
            total_current_flow = sum(current_flows.values())

            # 计算梯度方向
            gradient_direction = {}
            min_cost = path_costs[0][1]
            for i, cost in path_costs:
                # 对于成本最低的路径，增加流量；对于其他路径，减少流量
                if cost == min_cost:
                    gradient_direction[i] = 1.0
                else:
                    gradient_direction[i] = -1.0

            # 应用梯度更新
            updated_flows = {}
            for i in range(len(paths)):
                flow = current_flows.get(i, 0)
                direction = gradient_direction.get(i, 0)
                updated_flows[i] = max(0, flow + step_size * direction)

            # 投影到可行集上（保持总流量不变）
            total_updated_flow = sum(updated_flows.values())
            if total_updated_flow > 0:
                scaling_factor = group_flow / total_updated_flow
                for i, flow in updated_flows.items():
                    new_path_flows[(key, i)] = flow * scaling_factor
            else:
                # 如果所有路径流量都被设为0，将所有流量分配给成本最低的路径
                min_cost_path_idx = path_costs[0][0]
                new_path_flows[(key, min_cost_path_idx)] = group_flow

        # 计算流量变化量
        flow_change = sum(abs(new_path_flows.get(k, 0) - old_path_flows.get(k, 0))
                          for k in set(new_path_flows.keys()).union(old_path_flows.keys()))
        relative_change = flow_change / (sum(old_path_flows.values()) + 1e-10)

        # 更新路径流量
        path_flows = new_path_flows

        # 记录每次迭代后的路径流量总和
        total_path_flow = sum(path_flows.values())
        print(f"[检查点7] 迭代 {iteration + 1} 后的路径流量总和: {total_path_flow}")
        print(f"投影梯度法迭代 {iteration + 1} 的流量相对变化: {relative_change:.6f}")

        # 检查收敛
        if relative_change < convergence_threshold:
            print(f"投影梯度法在第 {iteration + 1} 次迭代后收敛")
            break

    # 为车辆分配路径
    path_assignment = {}
    for key, vehicle_ids in vehicle_groups.items():
        # 获取该组的路径和流量
        if key not in od_paths or not od_paths[key]:
            continue

        paths = od_paths[key]

        # 计算路径选择概率
        path_probs = []
        total_flow = 0

        for i, path in enumerate(paths):
            path_key = (key, i)
            flow = path_flows.get(path_key, 0)
            total_flow += flow
            path_probs.append((i, flow))

        # 归一化概率
        if total_flow > 0:
            path_probs = [(i, flow / total_flow) for i, flow in path_probs]
        else:
            # 如果总流量为0，使用均匀概率
            path_probs = [(i, 1.0 / len(paths)) for i in range(len(paths))]

        # 为每辆车分配路径
        for v_id in vehicle_ids:
            # 根据概率选择路径
            path_indices = [i for i, _ in path_probs]
            probs = [p for _, p in path_probs]

            if not path_indices:
                continue

            chosen_idx = np.random.choice(path_indices, p=probs)
            chosen_path = paths[chosen_idx]

            path_assignment[v_id] = {'path': chosen_path}

    # 更新车辆信息
    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        cs_id = vehicle_assignments[v_id]['cs']

        # 选择充电桩功率
        available_powers = list(center.charge_stations[cs_id].pile.keys())
        chosen_power = available_powers[0] if available_powers else 1

        # 更新车辆信息
        vehicle.charge = (cs_id, chosen_power)
        vehicle.path = path_info['path']

        # 更新流量
        update_flow(center, vehicle, vehicle.path)

        # 更新车辆位置信息
        if vehicle.path:
            vehicle.road = vehicle.path[0]
            vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1

            # 更新容量计数
            if vehicle.road in center.edges:
                center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                        center.edges[vehicle.road].capacity["all"], 1)

                if vehicle.origin != vehicle.charge[0]:
                    center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                        center.edges[vehicle.road].capacity["charge"], 1)

                center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                        center.edges[vehicle.road].capacity[vehicle.next_road], 1)

                # center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                #     center.edges[vehicle.road].capacity["all"], 1)
                # if vehicle.origin != vehicle.charge[0]:  # 如果起点不是充电站
                #     center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                #         center.edges[vehicle.road].capacity["charge"], 1)
                # center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                #     center.edges[vehicle.road].capacity[vehicle.next_road], 1)

            vehicle.drive()

    # 输出路径分配统计
    print("\n----- 路径分配统计 -----")
    od_anxiety_path_stats = {}

    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        origin = vehicle.origin
        destination = vehicle.destination
        anxiety = vehicle.anxiety
        cs_id = vehicle_assignments[v_id]['cs']

        # 路径元组
        path_tuple = tuple(path_info['path'])

        # 创建分组键
        key = (origin, destination, cs_id, anxiety)

        # 初始化统计信息
        if key not in od_anxiety_path_stats:
            od_anxiety_path_stats[key] = {'total': 0, 'path_choices': {}}

        # 更新计数
        od_anxiety_path_stats[key]['total'] += 1

        # 更新路径选择
        if path_tuple not in od_anxiety_path_stats[key]['path_choices']:
            od_anxiety_path_stats[key]['path_choices'][path_tuple] = 0
        od_anxiety_path_stats[key]['path_choices'][path_tuple] += 1

    # 输出分组统计信息
    for (origin, destination, cs_id, anxiety), stats in od_anxiety_path_stats.items():
        anxiety_type = "焦虑车辆" if anxiety == 1 else "普通车辆"
        print(f"\nOD对 ({origin} → {destination} → {cs_id}), {anxiety_type}, 总数: {stats['total']}辆")

        # 输出路径选择概率
        for path_tuple, count in sorted(stats['path_choices'].items(), key=lambda x: x[1], reverse=True):
            probability = count / stats['total']
            print(f"  选择概率: {probability:.2%} ({count}/{stats['total']}辆)")

    return path_assignment




def dispatch_vehicles_with_simplicial_decomposition(center, sol, charge_v, OD_ratio, cs_for_choice, real_path_results,
                                                    anxiety_OD_ratio=None, anxiety_cs_for_choice=None, max_iter=50,
                                                    convergence_threshold=1e-4, max_extreme_points=10):
    """
    使用单纯分解法和列生成进行交通分配优化的车辆调度函数

    Args:
        center: 调度中心对象
        sol: 求解结果
        charge_v: 需要充电的车辆ID列表
        OD_ratio: OD对的比例字典
        cs_for_choice: 每个OD对可选充电站字典
        real_path_results: 预计算的路径结果
        anxiety_OD_ratio: 焦虑车辆OD比例
        anxiety_cs_for_choice: 焦虑车辆可选充电站
        max_iter: 最大迭代次数
        convergence_threshold: 收敛阈值
        max_extreme_points: 每个OD对保留的最大极点数量
    """
    print("开始执行单纯分解法(列生成)进行车辆调度...")

    # 初始化边权重字典
    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[edge_id] = edge.free_time  # 初始权重使用自由流时间

    # 初始化边流量数组
    num_edges = max(edge_weights.keys()) + 1 if edge_weights else 0
    edge_flows = np.zeros(num_edges)

    # 添加背景流量 - 使用每条道路的k属性乘以容量
    for edge_id in edge_weights:
        edge = center.edges[edge_id]
        if hasattr(edge, 'k') and edge.capacity and "all" in edge.capacity:
            background_flow = edge.k * edge.capacity["all"][0]
        else:
            background_flow = 0
        edge_flows[edge_id] = background_flow
    print(f"初始化背景流量完成")

    # 接受sol然后按比例分配车辆至充电站
    vehicle_assignments = {}
    od_len = len(OD_ratio.keys()) if OD_ratio else 0

    print(f"为{len(charge_v)}辆车分配充电站...")
    for v_id in charge_v:
        vehicle = center.vehicles[v_id]

        if vehicle.anxiety == 1:  # 焦虑车辆
            if anxiety_OD_ratio and anxiety_cs_for_choice:
                od = (vehicle.origin, vehicle.destination)
                if od in anxiety_OD_ratio:
                    od_index = list(anxiety_OD_ratio.keys()).index(od) + od_len
                    cs_probs = sol[od_index]
                    chosen_cs = np.random.choice(anxiety_cs_for_choice[od], p=cs_probs)
                    vehicle_assignments[v_id] = {
                        'cs': chosen_cs,
                        'type': 'anxiety',
                        'route': [(vehicle.origin, vehicle.destination), (vehicle.destination, chosen_cs)]
                    }
        else:  # 普通车辆
            od = (vehicle.origin, vehicle.destination)
            if od in OD_ratio:
                od_index = list(OD_ratio.keys()).index(od)
                cs_probs = sol[od_index]
                chosen_cs = np.random.choice(cs_for_choice[od], p=cs_probs)
                vehicle_assignments[v_id] = {
                    'cs': chosen_cs,
                    'type': 'normal',
                    'route': [(vehicle.origin, chosen_cs), (chosen_cs, vehicle.destination)]
                }

    print(f"成功为{len(vehicle_assignments)}辆车分配充电站")

    # 整理出每类（起点，终点，充电站）组的车数量（即流量）
    flow_groups = {}  # (origin, destination, cs_id, type) -> 车辆数量
    vehicle_groups = {}  # (origin, destination, cs_id, type) -> 车辆ID列表

    for v_id, assignment in vehicle_assignments.items():
        vehicle = center.vehicles[v_id]
        cs_id = assignment['cs']
        od_type = assignment['type']

        if od_type == 'anxiety':
            # 焦虑车辆：起点 -> 终点 -> 充电站
            key = (vehicle.origin, vehicle.destination, cs_id, 'anxiety')
        else:
            # 普通车辆：起点 -> 充电站 -> 终点
            key = (vehicle.origin, vehicle.destination, cs_id, 'normal')

        if key not in flow_groups:
            flow_groups[key] = 0
            vehicle_groups[key] = []

        flow_groups[key] += 1
        vehicle_groups[key].append(v_id)

    print(f"共有{len(flow_groups)}个OD-CS组合")

    # 单纯分解算法主循环
    extreme_points = {}  # 存储每个OD对的极点集 key -> [(flow_vector, path)]
    extreme_point_costs = {}  # 极点成本 key -> [cost1, cost2, ...]
    lambda_values = {}  # 单纯形权重 key -> {extreme_point_idx: weight}
    current_paths = {}  # 当前每个OD对使用的路径 key -> path

    # 初始化：为每个OD对计算初始极点（全有全无分配结果）
    print("初始化每个OD对的极点...")
    for key, flow in flow_groups.items():
        origin, destination, cs_id, od_type = key

        # 生成初始路径
        if od_type == 'anxiety':
            # 焦虑车辆: 起点 -> 终点 -> 充电站
            path_o_to_d = calculate_shortest_path(center, origin, destination, edge_weights)
            path_d_to_cs = calculate_shortest_path(center, destination, cs_id, edge_weights)
            full_path = path_o_to_d + path_d_to_cs
        else:
            # 普通车辆: 起点 -> 充电站 -> 终点
            path_o_to_cs = calculate_shortest_path(center, origin, cs_id, edge_weights)
            path_cs_to_d = calculate_shortest_path(center, cs_id, destination, edge_weights)
            full_path = path_o_to_cs + path_cs_to_d

        # 创建流量向量（初始极点）
        initial_flow_vector = np.zeros(num_edges)
        for edge_id in full_path:
            if edge_id < len(initial_flow_vector):
                initial_flow_vector[edge_id] = flow

        # 初始化极点集和权重
        extreme_points[key] = [(initial_flow_vector, full_path)]
        extreme_point_costs[key] = [sum(edge_weights[e] for e in full_path)]
        lambda_values[key] = {0: 1.0}  # 初始只有一个极点，权重为1
        current_paths[key] = full_path

        # 更新全局流量
        edge_flows += initial_flow_vector

    # 主迭代循环
    for iteration in range(max_iter):
        print(f"单纯分解算法迭代 {iteration + 1}")

        # 根据当前流量更新边权重
        max_weight_change = 0
        for edge_id in edge_weights:
            if edge_id < len(edge_flows):
                old_weight = edge_weights[edge_id]

                # 获取边的信息
                edge = center.edges[edge_id]
                flow = edge_flows[edge_id]

                # 使用BPR函数更新边权重
                if hasattr(edge, 'capacity') and edge.capacity and "all" in edge.capacity:
                    capacity = edge.capacity["all"][0]
                    if capacity > 0:
                        weight = edge.free_time * (1 + edge.b * ((flow / capacity) ** edge.power))
                    else:
                        weight = edge.free_time
                else:
                    weight = edge.free_time

                edge_weights[edge_id] = weight
                max_weight_change = max(max_weight_change, abs(weight - old_weight) / max(0.1, old_weight))

        print(f"最大边权重相对变化: {max_weight_change:.6f}")

        # 列生成步骤：为每个OD对生成新的极点
        new_extreme_point_added = False
        auxiliary_flows = np.zeros(num_edges)

        for key, flow in flow_groups.items():
            origin, destination, cs_id, od_type = key

            # 使用当前边权重计算新的最短路径
            if od_type == 'anxiety':
                # 焦虑车辆: 起点 -> 终点 -> 充电站
                path_o_to_d = calculate_shortest_path(center, origin, destination, edge_weights)
                path_d_to_cs = calculate_shortest_path(center, destination, cs_id, edge_weights)
                new_path = path_o_to_d + path_d_to_cs
            else:
                # 普通车辆: 起点 -> 充电站 -> 终点
                path_o_to_cs = calculate_shortest_path(center, origin, cs_id, edge_weights)
                path_cs_to_d = calculate_shortest_path(center, cs_id, destination, edge_weights)
                new_path = path_o_to_cs + path_cs_to_d

            # 计算新路径成本
            new_path_cost = sum(edge_weights[e] for e in new_path)

            # 检查新路径是否与现有极点对应的路径重复
            is_duplicate = False
            for i, (_, existing_path) in enumerate(extreme_points[key]):
                if tuple(new_path) == tuple(existing_path):
                    is_duplicate = True
                    break

            # 如果不是重复的路径，添加为新极点
            if not is_duplicate:
                # 创建新极点流量向量
                new_flow_vector = np.zeros(num_edges)
                for edge_id in new_path:
                    if edge_id < len(new_flow_vector):
                        new_flow_vector[edge_id] = flow

                # 添加新极点
                extreme_points[key].append((new_flow_vector, new_path))
                extreme_point_costs[key].append(new_path_cost)

                # 标记已添加新极点
                new_extreme_point_added = True

                # 维持极点数量在限制范围内
                if len(extreme_points[key]) > max_extreme_points:
                    # 移除使用率最低的极点
                    min_weight_idx = min(lambda_values[key], key=lambda i: lambda_values[key].get(i, 0) if i < len(
                        extreme_points[key]) else float('inf'))
                    if lambda_values[key].get(min_weight_idx, 0) < 0.01:  # 只移除权重很小的极点
                        # 移除极点前调整其他极点的索引
                        new_lambda = {}
                        for idx, weight in lambda_values[key].items():
                            if idx < min_weight_idx:
                                new_lambda[idx] = weight
                            elif idx > min_weight_idx:
                                new_lambda[idx - 1] = weight

                        # 移除极点和对应成本
                        extreme_points[key].pop(min_weight_idx)
                        extreme_point_costs[key].pop(min_weight_idx)
                        lambda_values[key] = new_lambda

            # 将新路径的流量添加到辅助流量中
            for edge_id in new_path:
                if edge_id < len(auxiliary_flows):
                    auxiliary_flows[edge_id] += flow

        if not new_extreme_point_added and max_weight_change < convergence_threshold:
            print(f"单纯分解算法在第 {iteration + 1} 次迭代后收敛 - 没有新的极点生成且边权重变化小")
            break

        # 主问题：求解单纯形权重（二次规划问题）
        for key, extremes in extreme_points.items():
            flow = flow_groups[key]
            num_extremes = len(extremes)

            if num_extremes == 1:
                # 只有一个极点时，权重为1
                lambda_values[key] = {0: 1.0}
                continue

            # 构建二次规划问题
            # 目标：最小化总系统出行时间
            def objective_function(x):
                # 构建加权流量
                weighted_flow = np.zeros(num_edges)
                for i in range(num_extremes):
                    if i < len(extremes):
                        weighted_flow += x[i] * extremes[i][0]

                # 计算总系统出行时间
                total_cost = 0
                for edge_id in edge_weights:
                    if edge_id < len(weighted_flow):
                        flow = weighted_flow[edge_id]
                        edge = center.edges[edge_id]
                        if hasattr(edge, 'capacity') and edge.capacity and "all" in edge.capacity:
                            capacity = edge.capacity["all"][0]
                            if capacity > 0:
                                # 使用积分形式的BPR函数
                                total_cost += edge.free_time * (flow + edge.b * (capacity ** (1 - edge.power)) * (
                                            flow ** (1 + edge.power)) / (1 + edge.power))
                            else:
                                total_cost += edge.free_time * flow
                        else:
                            total_cost += edge.free_time * flow

                return total_cost

            # 初始解（使用当前的lambda值）
            x0 = np.zeros(num_extremes)
            for i in range(num_extremes):
                x0[i] = lambda_values[key].get(i, 0)

            # 约束条件：权重和为1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

            # 边界：所有权重在[0,1]之间
            bounds = [(0, 1) for _ in range(num_extremes)]

            # 使用优化器求解
            try:
                from scipy.optimize import minimize
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-6, 'maxiter': 100}
                )

                if result.success:
                    # 更新lambda值
                    new_lambda = {i: max(0, result.x[i]) for i in range(num_extremes) if result.x[i] > 1e-6}
                    # 归一化
                    total = sum(new_lambda.values())
                    if total > 0:
                        lambda_values[key] = {i: val / total for i, val in new_lambda.items()}
                    else:
                        # 如果所有权重都很小，设置第一个极点权重为1
                        lambda_values[key] = {0: 1.0}
                else:
                    print(f"警告：OD对 {key} 的单纯形权重优化失败，使用线性插值")
                    # 使用简单的线性方法计算权重
                    costs = [extreme_point_costs[key][i] for i in range(num_extremes)]
                    inv_costs = [1.0 / max(0.1, cost) for cost in costs]
                    total_inv_cost = sum(inv_costs)
                    lambda_values[key] = {i: inv_costs[i] / total_inv_cost for i in range(num_extremes)}
            except Exception as e:
                print(f"优化失败: {e}")
                # 使用简单的线性方法计算权重
                costs = [extreme_point_costs[key][i] for i in range(num_extremes)]
                inv_costs = [1.0 / max(0.1, cost) for cost in costs]
                total_inv_cost = sum(inv_costs)
                lambda_values[key] = {i: inv_costs[i] / total_inv_cost for i in range(num_extremes)}

        # 更新当前流量解
        new_edge_flows = np.zeros(num_edges)
        # 背景流量
        for edge_id in edge_weights:
            edge = center.edges[edge_id]
            if hasattr(edge, 'k') and edge.capacity and "all" in edge.capacity:
                new_edge_flows[edge_id] = edge.k * edge.capacity["all"][0]

        # 添加每个OD对按权重分配的流量
        for key, extremes in extreme_points.items():
            for i, weight in lambda_values[key].items():
                if i < len(extremes):
                    new_edge_flows += weight * extremes[i][0]

                    # 更新当前使用的路径（选择权重最大的极点对应的路径）
                    if weight > lambda_values[key].get(max(lambda_values[key], key=lambda_values[key].get), 0) - 1e-6:
                        current_paths[key] = extremes[i][1]

        # 计算流量变化
        flow_change = np.sum(np.abs(new_edge_flows - edge_flows))
        relative_flow_change = flow_change / (np.sum(edge_flows) + 1e-10)
        print(f"流量相对变化: {relative_flow_change:.6f}")

        # 更新流量
        edge_flows = new_edge_flows

        # 检查收敛
        if relative_flow_change < convergence_threshold:
            print(f"单纯分解算法在第 {iteration + 1} 次迭代后收敛")
            break

    # 分配路径给车辆
    path_assignment = {}

    for key, vehicle_ids in vehicle_groups.items():
        # 获取该OD组的极点和权重
        paths = [extreme_points[key][i][1] for i in lambda_values[key]]
        weights = list(lambda_values[key].values())

        # 为每辆车分配路径
        for v_id in vehicle_ids:
            chosen_path_idx = np.random.choice(len(paths), p=weights)
            path_assignment[v_id] = {'path': paths[chosen_path_idx]}

    # 最终处理：更新车辆信息
    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        cs_id = vehicle_assignments[v_id]['cs']

        # 选择充电桩功率（选择第一个可用功率）
        available_powers = list(center.charge_stations[cs_id].pile.keys())
        chosen_power = available_powers[0]

        # 更新车辆信息
        vehicle.charge = (cs_id, chosen_power)
        vehicle.path = path_info['path']

        # 更新流量
        update_flow(center, vehicle, vehicle.path)

        # 更新车辆位置信息
        if vehicle.path:
            vehicle.road = vehicle.path[0]
            vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1

            # 更新容量计数
            center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                center.edges[vehicle.road].capacity["all"], 1)
            if vehicle.origin != vehicle.charge[0]:  # 如果起点不是充电站
                center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["charge"], 1)
            center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                center.edges[vehicle.road].capacity[vehicle.next_road], 1)

            vehicle.drive()

    # 输出路径分配统计信息
    print("\n----- 路径分配统计 -----")
    od_anxiety_path_stats = {}

    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        origin = vehicle.origin
        destination = vehicle.destination
        anxiety = vehicle.anxiety
        cs_id = vehicle_assignments[v_id]['cs']

        # 路径元组（用于统计不同路径）
        path_tuple = tuple(path_info['path'])

        # 创建分组键
        key = (origin, destination, cs_id, anxiety)

        # 初始化统计信息
        if key not in od_anxiety_path_stats:
            od_anxiety_path_stats[key] = {
                'total': 0,
                'path_choices': {}
            }

        # 更新计数
        od_anxiety_path_stats[key]['total'] += 1

        # 更新路径选择
        if path_tuple not in od_anxiety_path_stats[key]['path_choices']:
            od_anxiety_path_stats[key]['path_choices'][path_tuple] = 0
        od_anxiety_path_stats[key]['path_choices'][path_tuple] += 1

    # 输出分组统计信息
    for (origin, destination, cs_id, anxiety), stats in od_anxiety_path_stats.items():
        anxiety_type = "焦虑车辆" if anxiety == 1 else "普通车辆"
        print(f"\nOD对 ({origin} → {destination} → {cs_id}), {anxiety_type}, 总数: {stats['total']}辆")

        # 输出路径选择概率
        for path_tuple, count in sorted(stats['path_choices'].items(), key=lambda x: x[1], reverse=True):
            probability = count / stats['total']
            print(f"  选择概率: {probability:.2%} ({count}/{stats['total']}辆)")

    return path_assignment


def dispatch_vehicles_with_disaggregated_simplicial_decomposition(center, sol, charge_v, OD_ratio, cs_for_choice,
                                                                  real_path_results,
                                                                  anxiety_OD_ratio=None, anxiety_cs_for_choice=None,
                                                                  max_iter=50,
                                                                  convergence_threshold=1e-4, max_extreme_points=10):
    """
    使用非聚合单纯分解法和列生成进行交通分配优化的车辆调度函数

    Args:
        center: 调度中心对象
        sol: 求解结果
        charge_v: 需要充电的车辆ID列表
        OD_ratio: OD对的比例字典
        cs_for_choice: 每个OD对可选充电站字典
        real_path_results: 预计算的路径结果
        anxiety_OD_ratio: 焦虑车辆OD比例
        anxiety_cs_for_choice: 焦虑车辆可选充电站
        max_iter: 最大迭代次数
        convergence_threshold: 收敛阈值
        max_extreme_points: 每个OD对保留的最大极点数量
    """
    print("开始执行非聚合单纯分解法(列生成)进行车辆调度...")

    # 初始化边权重字典
    edge_weights = defaultdict(float)
    for edge_id, edge in center.edges.items():
        edge_weights[edge_id] = edge.free_time  # 初始权重使用自由流时间

    # 初始化边流量数组
    num_edges = max(edge_weights.keys()) + 1 if edge_weights else 0
    edge_flows = np.zeros(num_edges)

    # 添加背景流量 - 使用每条道路的k属性乘以容量
    for edge_id in edge_weights:
        edge = center.edges[edge_id]
        if hasattr(edge, 'k') and edge.capacity and "all" in edge.capacity:
            background_flow = edge.k * edge.capacity["all"][0]
        else:
            background_flow = 0
        edge_flows[edge_id] = background_flow
    print(f"初始化背景流量完成")

    # 接受sol然后按比例分配车辆至充电站
    vehicle_assignments = {}
    od_len = len(OD_ratio.keys()) if OD_ratio else 0

    print(f"为{len(charge_v)}辆车分配充电站...")
    for v_id in charge_v:
        vehicle = center.vehicles[v_id]

        if vehicle.anxiety == 1:  # 焦虑车辆
            if anxiety_OD_ratio and anxiety_cs_for_choice:
                od = (vehicle.origin, vehicle.destination)
                if od in anxiety_OD_ratio:
                    od_index = list(anxiety_OD_ratio.keys()).index(od) + od_len
                    cs_probs = sol[od_index]
                    chosen_cs = np.random.choice(anxiety_cs_for_choice[od], p=cs_probs)
                    vehicle_assignments[v_id] = {
                        'cs': chosen_cs,
                        'type': 'anxiety',
                        'route': [(vehicle.origin, vehicle.destination), (vehicle.destination, chosen_cs)]
                    }
        else:  # 普通车辆
            od = (vehicle.origin, vehicle.destination)
            if od in OD_ratio:
                od_index = list(OD_ratio.keys()).index(od)
                cs_probs = sol[od_index]
                chosen_cs = np.random.choice(cs_for_choice[od], p=cs_probs)
                vehicle_assignments[v_id] = {
                    'cs': chosen_cs,
                    'type': 'normal',
                    'route': [(vehicle.origin, chosen_cs), (chosen_cs, vehicle.destination)]
                }

    print(f"成功为{len(vehicle_assignments)}辆车分配充电站")

    # 整理出每类（起点，终点，充电站）组的车数量（即流量）
    flow_groups = {}  # (origin, destination, cs_id, type) -> 车辆数量
    vehicle_groups = {}  # (origin, destination, cs_id, type) -> 车辆ID列表

    for v_id, assignment in vehicle_assignments.items():
        vehicle = center.vehicles[v_id]
        cs_id = assignment['cs']
        od_type = assignment['type']

        if od_type == 'anxiety':
            # 焦虑车辆：起点 -> 终点 -> 充电站
            key = (vehicle.origin, vehicle.destination, cs_id, 'anxiety')
        else:
            # 普通车辆：起点 -> 充电站 -> 终点
            key = (vehicle.origin, vehicle.destination, cs_id, 'normal')

        if key not in flow_groups:
            flow_groups[key] = 0
            vehicle_groups[key] = []

        flow_groups[key] += 1
        vehicle_groups[key].append(v_id)

    print(f"共有{len(flow_groups)}个OD-CS组合")

    # 非聚合单纯分解算法主循环
    extreme_points = {}  # 存储每个OD对的极点集 key -> [(flow_vector, path)]
    extreme_point_costs = {}  # 极点成本 key -> [cost1, cost2, ...]
    lambda_values = {}  # 单纯形权重 key -> {extreme_point_idx: weight}
    current_paths = {}  # 当前每个OD对使用的路径 key -> path
    od_specific_edge_flows = {}  # 每个OD对的边流量 key -> edge_flow_array

    # 初始化：为每个OD对计算初始极点
    print("初始化每个OD对的极点...")
    for key, flow in flow_groups.items():
        origin, destination, cs_id, od_type = key

        # 生成初始路径
        if od_type == 'anxiety':
            # 焦虑车辆: 起点 -> 终点 -> 充电站
            path_o_to_d = calculate_shortest_path(center, origin, destination, edge_weights)
            path_d_to_cs = calculate_shortest_path(center, destination, cs_id, edge_weights)
            full_path = path_o_to_d + path_d_to_cs
        else:
            # 普通车辆: 起点 -> 充电站 -> 终点
            path_o_to_cs = calculate_shortest_path(center, origin, cs_id, edge_weights)
            path_cs_to_d = calculate_shortest_path(center, cs_id, destination, edge_weights)
            full_path = path_o_to_cs + path_cs_to_d

        # 创建流量向量（初始极点）- 注意非聚合方法中每个OD对的流量向量是按单位流量计算的
        initial_flow_vector = np.zeros(num_edges)
        for edge_id in full_path:
            if edge_id < len(initial_flow_vector):
                initial_flow_vector[edge_id] = 1.0  # 单位流量

        # 初始化极点集和权重
        extreme_points[key] = [(initial_flow_vector, full_path)]
        extreme_point_costs[key] = [sum(edge_weights[e] for e in full_path)]
        lambda_values[key] = {0: 1.0}  # 初始只有一个极点，权重为1
        current_paths[key] = full_path

        # 初始化OD特定的边流量
        od_specific_edge_flows[key] = flow * initial_flow_vector

        # 更新全局流量
        edge_flows += flow * initial_flow_vector

    # 主迭代循环
    for iteration in range(max_iter):
        print(f"非聚合单纯分解算法迭代 {iteration + 1}")

        # 根据当前流量更新边权重
        max_weight_change = 0
        for edge_id in edge_weights:
            if edge_id < len(edge_flows):
                old_weight = edge_weights[edge_id]

                # 获取边的信息
                edge = center.edges[edge_id]
                flow = edge_flows[edge_id]

                # 使用BPR函数更新边权重
                if hasattr(edge, 'capacity') and edge.capacity and "all" in edge.capacity:
                    capacity = edge.capacity["all"][0]
                    if capacity > 0:
                        # BPR函数
                        weight = edge.free_time * (1 + edge.b * ((flow / capacity) ** edge.power))
                    else:
                        weight = edge.free_time
                else:
                    weight = edge.free_time

                edge_weights[edge_id] = weight
                max_weight_change = max(max_weight_change, abs(weight - old_weight) / max(0.1, old_weight))

        print(f"最大边权重相对变化: {max_weight_change:.6f}")

        # 列生成步骤：为每个OD对生成新的极点
        new_extreme_point_added = False
        max_flow_change = 0

        # 遍历每个OD对
        for key, flow in flow_groups.items():
            origin, destination, cs_id, od_type = key

            # 使用当前边权重计算新的最短路径
            if od_type == 'anxiety':
                # 焦虑车辆: 起点 -> 终点 -> 充电站
                path_o_to_d = calculate_shortest_path(center, origin, destination, edge_weights)
                path_d_to_cs = calculate_shortest_path(center, destination, cs_id, edge_weights)
                new_path = path_o_to_d + path_d_to_cs
            else:
                # 普通车辆: 起点 -> 充电站 -> 终点
                path_o_to_cs = calculate_shortest_path(center, origin, cs_id, edge_weights)
                path_cs_to_d = calculate_shortest_path(center, cs_id, destination, edge_weights)
                new_path = path_o_to_cs + path_cs_to_d

            # 计算新路径成本
            new_path_cost = sum(edge_weights[e] for e in new_path)

            # 检查新路径是否与现有极点对应的路径重复
            is_duplicate = False
            for i, (_, existing_path) in enumerate(extreme_points[key]):
                if tuple(new_path) == tuple(existing_path):
                    is_duplicate = True
                    break

            # 如果不是重复的路径，添加为新极点
            if not is_duplicate:
                # 创建新极点流量向量 - 非聚合方法中每个新极点是单位流量
                new_flow_vector = np.zeros(num_edges)
                for edge_id in new_path:
                    if edge_id < len(new_flow_vector):
                        new_flow_vector[edge_id] = 1.0

                # 添加新极点
                extreme_points[key].append((new_flow_vector, new_path))
                extreme_point_costs[key].append(new_path_cost)

                # 新极点的初始权重为0
                lambda_values[key][len(extreme_points[key]) - 1] = 0.0

                # 标记已添加新极点
                new_extreme_point_added = True

                # 维持极点数量在限制范围内
                if len(extreme_points[key]) > max_extreme_points:
                    # 移除使用率最低的极点
                    min_weight_idx = min(lambda_values[key], key=lambda i: lambda_values[key].get(i, 0) if i < len(
                        extreme_points[key]) else float('inf'))
                    if lambda_values[key].get(min_weight_idx, 0) < 0.01:  # 只移除权重很小的极点
                        # 删除极点
                        extreme_points[key].pop(min_weight_idx)
                        extreme_point_costs[key].pop(min_weight_idx)

                        # 更新lambda值
                        new_lambda = {}
                        for i, w in lambda_values[key].items():
                            if i < min_weight_idx:
                                new_lambda[i] = w
                            elif i > min_weight_idx:
                                new_lambda[i - 1] = w
                        lambda_values[key] = new_lambda

        # 非聚合法中，即使没有新极点但边权重仍有变化，也需要更新所有OD对的权重
        if max_weight_change < convergence_threshold and not new_extreme_point_added:
            print(f"非聚合单纯分解算法在第 {iteration + 1} 次迭代后收敛 - 没有新的极点生成且边权重变化小")
            break

        # 主问题：为每个OD对单独优化单纯形权重
        for key, extremes in extreme_points.items():
            flow = flow_groups[key]
            num_extremes = len(extremes)

            if num_extremes == 1:
                # 只有一个极点时，权重为1
                lambda_values[key] = {0: 1.0}
                continue

            # 构建二次规划问题
            # 目标：最小化该OD对的总系统出行时间
            def objective_function(x):
                # 构建该OD对的加权流量
                weighted_flow = np.zeros(num_edges)
                for i in range(num_extremes):
                    if i < len(extremes):
                        weighted_flow += x[i] * extremes[i][0]

                # 该OD对的总流量
                od_flow = weighted_flow * flow

                # 计算所有边的临时流量（替换该OD对的流量）
                temp_edge_flows = edge_flows - od_specific_edge_flows[key] + od_flow

                # 计算该流量分配下的总系统出行时间
                total_cost = 0
                for edge_id in edge_weights:
                    if edge_id < len(temp_edge_flows):
                        edge = center.edges[edge_id]
                        temp_flow = temp_edge_flows[edge_id]

                        if hasattr(edge, 'capacity') and edge.capacity and "all" in edge.capacity:
                            capacity = edge.capacity["all"][0]
                            if capacity > 0:
                                # 使用BPR函数计算出行时间
                                travel_time = edge.free_time * (1 + edge.b * ((temp_flow / capacity) ** edge.power))
                            else:
                                travel_time = edge.free_time
                        else:
                            travel_time = edge.free_time

                        # 系统总出行时间 = 流量 × 边出行时间
                        total_cost += temp_flow * travel_time

                return total_cost

            # 初始解（使用当前的lambda值）
            x0 = np.zeros(num_extremes)
            for i in range(num_extremes):
                x0[i] = lambda_values[key].get(i, 0)

            # 约束条件：权重和为1
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

            # 边界：所有权重在[0,1]之间
            bounds = [(0, 1) for _ in range(num_extremes)]

            # 使用优化器求解
            try:
                from scipy.optimize import minimize
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'ftol': 1e-6, 'maxiter': 100}
                )

                if result.success:
                    # 更新lambda值
                    new_lambda = {i: max(0, result.x[i]) for i in range(num_extremes) if result.x[i] > 1e-6}
                    # 归一化
                    total = sum(new_lambda.values())
                    if total > 0:
                        lambda_values[key] = {i: v / total for i, v in new_lambda.items()}
                    else:
                        lambda_values[key] = {0: 1.0}  # 如果所有权重都接近0，则使用第一个极点
                else:
                    print(f"警告：OD对 {key} 的单纯形权重优化失败，使用逆成本加权")
                    # 使用逆成本加权
                    costs = [extreme_point_costs[key][i] for i in range(num_extremes)]
                    inv_costs = [1.0 / max(0.1, cost) for cost in costs]
                    total_inv_cost = sum(inv_costs)
                    lambda_values[key] = {i: inv_costs[i] / total_inv_cost for i in range(num_extremes)}
            except Exception as e:
                print(f"优化失败: {e}")
                # 使用逆成本加权
                costs = [extreme_point_costs[key][i] for i in range(num_extremes)]
                inv_costs = [1.0 / max(0.1, cost) for cost in costs]
                total_inv_cost = sum(inv_costs)
                lambda_values[key] = {i: inv_costs[i] / total_inv_cost for i in range(num_extremes)}

            # 更新该OD对的流量分配
            old_od_flow = od_specific_edge_flows[key].copy()
            new_od_flow = np.zeros(num_edges)

            for i, weight in lambda_values[key].items():
                if i < len(extremes):
                    new_od_flow += weight * extremes[i][0] * flow

                    # 更新当前使用的路径（选择权重最大的极点对应的路径）
                    max_weight_idx = max(lambda_values[key], key=lambda_values[key].get)
                    if i == max_weight_idx:
                        current_paths[key] = extremes[i][1]

            # 计算该OD对流量变化
            od_flow_change = np.sum(np.abs(new_od_flow - old_od_flow))
            od_relative_flow_change = od_flow_change / (np.sum(old_od_flow) + 1e-10)
            max_flow_change = max(max_flow_change, od_relative_flow_change)

            # 更新全局流量和OD特定流量
            edge_flows = edge_flows - old_od_flow + new_od_flow
            od_specific_edge_flows[key] = new_od_flow

        print(f"最大OD流量相对变化: {max_flow_change:.6f}")

        # 检查收敛
        if max_flow_change < convergence_threshold:
            print(f"非聚合单纯分解算法在第 {iteration + 1} 次迭代后收敛")
            break

    # 分配路径给车辆
    path_assignment = {}

    for key, vehicle_ids in vehicle_groups.items():
        # 获取该OD组的极点和权重
        paths = [extreme_points[key][i][1] for i in lambda_values[key]]
        weights = list(lambda_values[key].values())

        # 为每辆车分配路径
        for v_id in vehicle_ids:
            chosen_path_idx = np.random.choice(len(paths), p=weights)
            path_assignment[v_id] = {'path': paths[chosen_path_idx]}

    # 最终处理：更新车辆信息
    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        cs_id = vehicle_assignments[v_id]['cs']

        # 选择充电桩功率（选择第一个可用功率）
        available_powers = list(center.charge_stations[cs_id].pile.keys())
        chosen_power = available_powers[0]

        # 更新车辆信息
        vehicle.charge = (cs_id, chosen_power)
        vehicle.path = path_info['path']

        # 更新流量
        update_flow(center, vehicle, vehicle.path)

        # 更新车辆位置信息
        if vehicle.path:
            vehicle.road = vehicle.path[0]
            vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1

            # 更新容量计数
            center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                center.edges[vehicle.road].capacity["all"], 1)
            if vehicle.origin != vehicle.charge[0]:  # 如果起点不是充电站
                center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["charge"], 1)
            center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                center.edges[vehicle.road].capacity[vehicle.next_road], 1)

            vehicle.drive()

    # 输出路径分配统计信息
    print("\n----- 路径分配统计 -----")
    od_anxiety_path_stats = {}

    for v_id, path_info in path_assignment.items():
        if v_id not in vehicle_assignments:
            continue

        vehicle = center.vehicles[v_id]
        origin = vehicle.origin
        destination = vehicle.destination
        anxiety = vehicle.anxiety
        cs_id = vehicle_assignments[v_id]['cs']

        # 路径元组（用于统计不同路径）
        path_tuple = tuple(path_info['path'])

        # 创建分组键
        key = (origin, destination, cs_id, anxiety)

        # 初始化统计信息
        if key not in od_anxiety_path_stats:
            od_anxiety_path_stats[key] = {
                'total': 0,
                'path_choices': {}
            }

        # 更新计数
        od_anxiety_path_stats[key]['total'] += 1

        # 更新路径选择
        if path_tuple not in od_anxiety_path_stats[key]['path_choices']:
            od_anxiety_path_stats[key]['path_choices'][path_tuple] = 0
        od_anxiety_path_stats[key]['path_choices'][path_tuple] += 1

    # 输出分组统计信息
    for (origin, destination, cs_id, anxiety), stats in od_anxiety_path_stats.items():
        anxiety_type = "焦虑车辆" if anxiety == 1 else "普通车辆"
        print(f"\nOD对 ({origin} → {destination} → {cs_id}), {anxiety_type}, 总数: {stats['total']}辆")

        # 输出路径选择概率
        for path_tuple, count in sorted(stats['path_choices'].items(), key=lambda x: x[1], reverse=True):
            probability = count / stats['total']
            print(f"  选择概率: {probability:.2%} ({count}/{stats['total']}辆)")

    return path_assignment