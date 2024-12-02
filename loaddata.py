import numpy as np
import pandas as pd
import networkx as nx
import random


# 读取 CSV 文件的指定列
def read_csv(file_path, cols):
    df = pd.read_csv(file_path, usecols=cols)
    return df


# 构建图
def build_graph(df, k_dict = None):
    G = nx.DiGraph()  # 有向图
    for index, row in df.iterrows():
        origin = row['init_node']
        destination = row['term_node']
        if k_dict is None:
            weight = row['length']
        else:
            weight = row['length'] * k_dict[(origin, destination)]
        G.add_edge(origin, destination, weight=weight)
    return G


# 使用 Dijkstra 算法计算路径
def shortest_path(G, origin, destination, k=1):
    if k == 1:
        paths = list(nx.all_shortest_paths(G, source=origin, target=destination, weight='weight'))
        path = nx.dijkstra_path(G, origin, destination)
        path_length = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        # return paths, path_length
    else:
        paths = list(nx.shortest_simple_paths(G, source=origin, target=destination, weight='weight'))
        result_paths = []
        for i, path in enumerate(paths):
            if i >= k:
                break
            result_paths.append(path)

        while len(result_paths) < k:
            try:
                next_path = next(paths)
                result_paths.append(next_path)
            except StopIteration:
                break

        paths = sorted(result_paths, key=lambda p: sum(G[u][v]['weight'] for u, v in zip(p[:-1], p[1:])))
        path_length = sum(G[u][v]['weight'] for u, v in zip(result_paths[0][:-1], result_paths[0][1:]))

    return paths, path_length


# 处理多个 OD 对
def process_od_pairs(G, od_pairs):
    results = {}
    for (origin, destination) in od_pairs:
        # print(origin, destination)
        try:
            path, path_length = shortest_path(G, origin, destination)
            results[(origin, destination)] = (path, path_length)
        except nx.NetworkXNoPath:
            results[(origin, destination)] = (None, None)
        except nx.NodeNotFound as e:
            results[(origin, destination)] = (str(e), None)
    return results


# 主函数
def main(file_path, od_pairs, k_list=None):
    df = read_csv(file_path, ['init_node', 'term_node', 'length'])
    G = build_graph(df, k_list)
    # print(G.edges(data=True))

    results = process_od_pairs(G, od_pairs)
    # print(results)
    return results
    # for (origin, destination), (path, path_length) in results.items():
    #     print(origin, destination, path, path_length)
    #     if path is None:
    #         print(f"从 {origin} 到 {destination} 没有可行的路径。")
    #     elif isinstance(path, str):
    #         print(f"节点错误: {path}")
    #     else:
    #         return path, path_length



def distribute_od_pairs(data_dict, elements_per_category=100):
    # Calculate total count and number of groups
    total_count = sum(data_dict.values())
    group_count = (total_count + elements_per_category - 1) // elements_per_category  # Round up

    # Initialize groups
    groups = [[] for _ in range(group_count)]
    od_used = [set() for _ in range(group_count)]

    # Shuffle OD pairs
    od_pairs = list(data_dict.items())
    random.shuffle(od_pairs)

    # Distribute OD pairs
    for (od, count) in od_pairs:
        while count > 0:
            placed = False
            for i in range(group_count):
                if od not in od_used[i] and len(groups[i]) < elements_per_category:
                    groups[i].append(od)
                    od_used[i].add(od)
                    count -= 1
                    placed = True
                    if count == 0:
                        break

            # If not placed, randomly select a group to place
            if not placed:
                attempts = 0
                while attempts < 10:  # Try up to 10 times
                    random_group = random.choice(range(group_count))
                    if len(groups[random_group]) < elements_per_category:
                        groups[random_group].append(od)
                        od_used[random_group].add(od)
                        count -= 1
                        break
                    attempts += 1

                # If 10 attempts fail, try sequentially
                if attempts >= 10:
                    for offset in range(1, group_count + 1):
                        random_group = (group_count - offset) % group_count
                        if len(groups[random_group]) < elements_per_category:
                            groups[random_group].append(od)
                            od_used[random_group].add(od)
                            count -= 1
                            break

                    # If all groups are full, add remaining elements to the last group
                    if count > 0:
                        for i in range(group_count):
                            if len(groups[i]) < elements_per_category:
                                groups[i].extend([od] * count)
                                od_used[i].add(od)
                                count = 0
                                break

    # Balance the number of elements in each sublist
    while True:
        max_len = max(len(group) for group in groups)
        min_len = min(len(group) for group in groups)
        if max_len - min_len <= 1:
            break
        for group in groups:
            if len(group) == max_len:
                for target_group in groups:
                    if len(target_group) == min_len:
                        target_group.append(group.pop())
                        break

    return groups



# 示例用法
# if __name__ == "__main__":
#     csv_file_path = 'data/SF/SiouxFalls_net.csv'
#     num_nodes = 24
#     max_distance = 0
#     sum_distance = 0
#     od = []
#     init_path = np.empty((num_nodes, num_nodes), dtype=object)
#     num_length = np.zeros(25, dtype=int)
#     for i in range(1, num_nodes + 1):
#         for j in range(1, num_nodes + 1):
#             if i != j:
#                 od.append((i, j))
#                 # init_path[i - 1, j - 1], distance = main(csv_file_path, [(i, j)])
#                 # # print(distance, end=' ')
#                 # num_length[distance] += 1
#                 # max_distance = max(max_distance, distance)
#                 # sum_distance += distance
#     # od_pairs = [(1, 2), (1, 5), (1, 20)]
#     # main(csv_file_path, od_pairs)
#     print(main(csv_file_path, od))
#     # cal = num_nodes * (num_nodes - 1)
#     # print(init_path)
#     # print(max_distance)
#     # index = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
#     # for i in range(1, num_nodes):
#     #     for j in range(1, num_nodes):
#     #         if init_path[i, j] is not None:
#     #             for k in range(1, len(init_path[i, j]) - 1):
#     #                 index[init_path[i, j][k], init_path[i, j][k + 1]] += 1
#     # print(index)