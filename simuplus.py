import numpy as np
import pandas as pd
from pathlib import Path
from pandapower.networks import create_synthetic_voltage_control_lv_network
import TNplus
import PDNplus
import loaddata as ld
import random
import csv
import ast
import matplotlib.pyplot as plt
import concurrent.futures
import numba
import heapq
from collections import defaultdict

standard_speed = 60  #km/h
t = 1  #min
T = 10  #min
T_pdn = 3 * T  #min
roadmap = 'SF'  #目前支持SF、EMA和PY
od_no = '2' #目前支持空字符串、1和2
csv_net_path = 'data/' + roadmap + '/' + roadmap + '_net.csv'
csv_od_path = 'data/' + roadmap + '/' + roadmap + '_od' + od_no +'.csv'
node = {'SF': 24, 'EMA': 76, 'PY': 167}
num_nodes = node[roadmap]
batch_size = 320000 / 20 #在等额分割时要与实际大小一致
all_log = False
OD_from_csv = False
dispatch_list = {}
k_list = {}
G = None
k = 2
rate = 0.5



def processplus(road_times, wait_times_list, k, center):
    def dijkstra(graph, start, k):
        pq = [(0, 0, [start], start)]
        shortest_paths = defaultdict(list)
        shortest_paths[start].append((0, 0, [start]))

        while pq:
            cost, wait_time, path, node = heapq.heappop(pq)
            if len(shortest_paths[node]) > k:
                continue
            for neighbor, travel_time in graph[node]:
                if neighbor in path:
                    continue  # Skip paths with duplicate nodes
                new_cost = cost + travel_time
                new_wait_time = wait_time + wait_times_list[node].get((node, neighbor), 0)
                new_path = path + [neighbor]
                total_time = new_cost + new_wait_time
                if len(shortest_paths[neighbor]) < k or total_time < shortest_paths[neighbor][-1][0] + shortest_paths[neighbor][-1][1]:
                    heapq.heappush(pq, (new_cost, new_wait_time, new_path, neighbor))
                    shortest_paths[neighbor].append((new_cost, new_wait_time, new_path))
                    shortest_paths[neighbor].sort(key=lambda x: x[0] + x[1])
                    if len(shortest_paths[neighbor]) > k:
                        shortest_paths[neighbor].pop()
        return shortest_paths

    graph = defaultdict(list)
    for road_id, (start, end, travel_time) in road_times.items():
        graph[start].append((end, travel_time))

    all_shortest_paths = {}
    nodes = set(start for start, _, _ in road_times.values()).union(set(end for _, end, _ in road_times.values()))
    for start in nodes:
        shortest_paths = dijkstra(graph, start, k)
        for end in nodes:
            if start != end and end in shortest_paths:
                unique_paths = list({tuple(path): (cost, wait_time) for cost, wait_time, path in shortest_paths[end]}.items())
                all_shortest_paths[(start, end)] = [[list(path), cost, wait_time] for path, (cost, wait_time) in unique_paths]

    for od, result in all_shortest_paths.items():
        for i in range(len(result)):
            path = result[i][0]
            sum = 0
            for j in range(1, len(path) - 1):
                node = center.nodes[path[j]]
                fr = to = -1
                for edge in node.enter:
                    if center.edges[edge].origin == path[j - 1]:
                        fr = center.edges[edge].id
                        break
                for edge in node.off:
                    if center.edges[edge].destination == path[j + 1]:
                        to = center.edges[edge].id
                        break
                sum += node.calculate_wait(fr, to)
            result[i][2] = sum

    return all_shortest_paths



class PathProcessor:
    def __init__(self, file_path, od_pairs):
        self.file_path = file_path
        self.od_pairs = od_pairs

    def process_paths(self, k_list=None, k=1):
        return ld.main(self.file_path, self.od_pairs, k_list, k)

    def get_shortest_path(self, G, origin, destination, k=1):
        return ld.shortest_path(G, origin, destination, k)

    def process_multiple_od_pairs(self, G):
        return ld.process_od_pairs(G, self.od_pairs)

    def build_graph(self, file_path, k_list=None):
        return ld.build_graph(ld.read_csv(self.file_path, ['init_node', 'term_node', 'length']), k_list)


class ODGenerator:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        df = ld.read_csv(self.file_path, ['O', 'D', 'Ton'])
        # df = df.dropna()
        data_dict = {(row['O'], row['D']): int(row['Ton']) for _, row in df.iterrows()}
        return data_dict

    def distribute_od_pairs(self, data_dict, elements_per_category):
        # 计算总个数和组数
        return ld.distribute_od_pairs(data_dict, elements_per_category)

def get_graph():
    return G


class ODGenerator2:
    def __init__(self, file_path, count = 100):
        self.file_path = file_path
        self.count = count

    def load(self):
        df = ld.read_csv(self.file_path, ['O', 'D', 'Ton'])
        data_dict = {(row['O'], row['D']): int(row['Ton']) for _, row in df.iterrows()}
        return data_dict

    def generate_od_pairs(self, data_dict):
        od_result = []
        for i in range(self.count):
            od_pairs = []
            for (O, D), ton in data_dict.items():
                count = int(ton / 10)
                od_pairs.extend([(O, D)] * count)
            random.shuffle(od_pairs)
            od_result.append(od_pairs)
        return od_result


class ODgenerator_equal:
    def __init__(self, file_path, count=20, c_rate=rate):
        self.file_path = file_path
        self.count = count
        self.c_rate = c_rate

    def load(self):
        df = pd.read_csv(self.file_path, usecols=['O', 'D', 'Ton'])
        return df

    def distribute_equal_od_pairs(self, df):
        od_result = []
        for _ in range(self.count):
            od_result.append([])
            for _, row in df.iterrows():
                O, D, Ton = row['O'], row['D'], row['Ton']
                for _ in range(int(Ton * self.c_rate / self.count)):
                    od_result[-1].append([O, D])
            for _, row in df.iterrows():
                O, D, Ton = row['O'], row['D'], row['Ton']
                for _ in range(int(Ton * (1 - self.c_rate) / self.count)):
                    od_result[-1].append([O, D])
        print("Equal OD pairs generated")
        # print("OD pairs: {}".format(od_result))
        return od_result

def update_edge_ratios(Edges):
    for e in Edges:
        e.update_ratio()

def update_node_ratios(Nodes):
    for no in Nodes:
        no.update_ratio()
        
        
        




if __name__ == "__main__":

    print(csv_net_path)
    print(csv_od_path)
    od_pairs = []
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            if i != j:
                od_pairs.append([i, j])
                k_list[(i, j)] = 1
    # print(od_pairs)
    processor = PathProcessor(csv_net_path, od_pairs)
    G = processor.build_graph(csv_net_path)
    print("建图完成")
    path_results = processor.process_paths()
    print(path_results)
    for (i, j), paths in path_results.items():
        print((i,j))


    # generator = ODGenerator(csv_od_path)
    # data = generator.load()
    # OD_results = generator.distribute_od_pairs(data, batch_size)
    # random.shuffle(OD_results)
    # for ODs in OD_results:
    #     random.shuffle(ODs)

    # generator2 = ODGenerator2(csv_od_path)
    # data = generator2.load()
    # OD_results = generator2.generate_od_pairs(data)
    # random.shuffle(OD_results)

    generator = ODgenerator_equal(csv_od_path, count=20)
    data = generator.load()
    print(data)
    OD_results = generator.distribute_equal_od_pairs(data)
    print("OD载入完成")
    print(len(OD_results), len(OD_results[0]), len(OD_results[1]), len(OD_results[2]))

    # Specify the file name
    file_path = 'OD_output.csv'
    if OD_from_csv:
        OD_results = []
        with open(file_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                row_data = [ast.literal_eval(cell) for cell in row]
                OD_results.append(row_data)
    else:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for sublist in OD_results:
                writer.writerow(sublist)

    # print(OD_results)
    # for ods in OD_results:
    #     d = {}
    #     for (O, D) in ods:
    #         if (O, D) in d:
    #             d[(O, D)] += 1
    #         else:
    #             d[(O, D)] = 1
    #     print(d)
    # print(len(OD_results), len(OD_results[0]), len(OD_results[1]), len(OD_results[2]))
    print("mother fucker")

    edge_data = pd.read_csv(csv_net_path, usecols=['init_node', 'term_node', 'capacity', 'length', 'free_flow_time'])
    # edge_data = edge_data.dropna()
    print("道路载入完成")

    center = TNplus.DispatchCenter([], {}, {}, {}, [], False)
    nodes = []
    edge_id = 1

    for index, row in edge_data.iterrows():
        origin = int(row['init_node'])
        destination = int(row['term_node'])
        capacity = int(row['capacity'] * 0.6)
        length = float(row['length'])
        free_flow_time = float(row['free_flow_time'])
        # print(f"edge_id: {edge_id}, origin: {origin}, destination: {destination}, length: {length}, free_flow_time: {free_flow_time}")

        if destination not in nodes:
            nodes.append(destination)
            center.nodes[destination] = TNplus.Node(destination, center, {}, [], 1, [], [], {})

        if origin not in nodes:
            nodes.append(origin)
            center.nodes[origin] = TNplus.Node(origin, center, {}, [], 1, [], [], {})

        center.nodes[origin].edge_num += 1
        center.nodes[destination].enter.append(edge_id)
        center.nodes[destination].edge_num += 1
        center.nodes[origin].off.append(edge_id)

        """
        从这里开始是对节点信号信息和容量及其分量的定义，这里没有写csv信息，而是假设其均匀然后平均分配
        实际情况下这里读入csv文件后直接赋值就行
        """

        # print(edge_id, center, origin, destination, length)
        k_list[(origin, destination)] = 1
        edge = TNplus.Edge(edge_id, center, origin, destination, length, {}, free_flow_time, 0.15, 4)
        edge.capacity["all"] = (capacity, 0)
        edge.capacity["charge"] = (capacity, 0)
        edge.capacity[-1] = (capacity, 0)
        center.edges[edge_id] = edge
        edge_id += 1

    for edge in center.edges.values():
        index = edge.destination
        destination = center.nodes[index]
        for off in destination.off:
            edge.capacity[off] = (edge.capacity['all'][0] / (destination.edge_num / 2 - 1), 0)

    for n in nodes:
        for enter in center.nodes[n].enter:
            for off in center.nodes[n].off:
                if enter != -1 and off != -1:
                    center.nodes[n].signal[(enter, off)] = (1.5 / (center.nodes[n].edge_num / 2), 1.5)

    if all_log:
        for node in center.nodes.values():
            print("测试信号信息")
            print(node.signal)

    print("路网建构完成")
    v_index = 0
    if roadmap == 'SF': pdn = PDNplus.create_ieee14()
    if roadmap == 'PY': pdn = PDNplus.create_ieee33()
    print("电网建构完成")
    pdn_result = []
    tn_result = []



    for edge in center.edges.values():
        print(edge.capacity)
        center.edge_timely_estimated_load[(edge.origin, edge.destination)] = []
        for i in range(1, 20 * 3 + 40):
            center.edge_timely_estimated_load[(edge.origin, edge.destination)].append([0, int(edge.capacity['all'][0] * (1 + 0))])

    for i in range(1, 20 * 3):
        # if all_log:
        print(f"主循环 {i}")
        center.current_time = i
        if i == 1:
            for edge in center.edges.values():
                edge.update_ratio()

            for node in center.nodes.values():
                node.update_ratio()
        else:
            edges = list(center.edges.values())
            nodes = list(center.nodes.values())

            # Split the edges and nodes into 8 chunks
            edge_chunks = [edges[i::8] for i in range(8)]
            node_chunks = [nodes[i::8] for i in range(8)]

            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                # Submit tasks for updating edge ratios
                edge_futures = [executor.submit(update_edge_ratios, chunk) for chunk in edge_chunks]

                # Wait for all futures to complete
                concurrent.futures.wait(edge_futures)

            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                # Submit tasks for updating node ratios
                node_futures = [executor.submit(update_node_ratios, chunk) for chunk in node_chunks]

                # Wait for all futures to complete
                concurrent.futures.wait(node_futures)

            for edge in center.edges.values():
                print(f"{edge.id} : {edge.capacity}")

            for cs in center.charge_stations.values():
                print(f"{cs.id} : {cs.capacity}")
                print(f"{cs.id} : {cs.dispatch}")
                print(f"{cs.id} : {len(cs.queue[300])}")
                print(f"{cs.id} : {cs.queue}")
                print(f"{cs.id} : {len(cs.charge[300])}")
                print(f"{cs.id} : {cs.charge}")
                print(' ')

        if center.delay_vehicles[i]:
            for vehicle_id in center.delay_vehicles[i]:
                vehicle = center.vehicles[vehicle_id]
                center.vehicles[vehicle_id].delay = False
                center.edges[vehicle.road].capacity["all"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["all"], 1)
                center.edges[vehicle.road].capacity["charge"] = center.solve_tuple(
                    center.edges[vehicle.road].capacity["charge"], 1)
                center.edges[vehicle.road].capacity[vehicle.next_road] = center.solve_tuple(
                    center.edges[vehicle.road].capacity[vehicle.next_road], 1)
                center.vehicles[vehicle_id].drive()
            center.delay_vehicles[i].clear()

        for vehicle in center.vehicles:
            if not vehicle.delay:
                if vehicle.charging == False and vehicle.is_wait > 0:
                    vehicle.wait(vehicle.road, vehicle.next_road)
                elif vehicle.charging:
                    continue
                elif vehicle.road != -1:
                    vehicle.drive(vehicle.road)

        """
        这里充电站的信息可改成用csv读入
        """
        if i == 1:
            if all_log:
                print("初始化充电站")
            for j in TNplus.cs:
                center.charge_stations[j] = TNplus.ChargeStation(j, center, {}, {150: [], 300: []}, {150: [], 300: []},
                                                                 25000, {300: 20000},
                                                                 {300: (0, 0)}, {300: 0},False)

        if i % 3 == 1 or i == 1:
            print("加入新车")
            if all_log:
                print(f"在循环i={i}时加入新OD")
            OD = OD_results[int(i / 3)]
            charge_num = int(batch_size * rate)
            charge_v = []
            charge_od = []


            total_charge_cost = {}
            for cs in center.charge_stations.values():
                total_charge_cost[f"EVCS {cs.id}"] = cs.cost / 60 / 1000
                cs.cost = 0
            # print(total_charge_cost)
            PDNplus.update_load(pdn, total_charge_cost, 3 * t / 60 * 6 * 2 * 2)
            PDNplus.run(pdn, 100)
            lmp_dict = pdn.res_bus['lam_p'].to_dict()
            # print(lmp_dict)
            pdn_loss = PDNplus.calculate_loss(pdn, 140)
            pdn_result.append(pdn_loss)
            # print(pdn_loss)
            tn_result.append(center.calculate_lost())
            # print(center.calculate_lost())

            for (O, D) in OD:
                #对于od2来说，设置车的电量，必须途中充电就4.2， 只是焦虑区间就6，直达就随便了（直达占比就是rate， 其他两个1:1分配或随机数得了）
                #od3等下再算，不过由于选的od距离平均下来反而更近了，其实这个数也可以（）
                choice = path_results[(O, D)][0]
                if len(choice) > 1:
                    path = choice[i % len(choice)]
                    true_path = []
                    for j in range(0, len(path) - 1):
                        for edge in center.edges.values():
                            if edge.origin == path[j] and edge.destination == path[j + 1]:
                                true_path.append(edge.id)
                    if len(true_path) > 1:
                        next = true_path[1]
                    else:
                        next = -1
                    new_vehicle = TNplus.Vehicle(v_index, center, O, D, center.edges[true_path[0]].length,
                                                 true_path[0], next,
                                                 true_path, 60, random.randint(12, 48), 0.05, 0.15, 0, {}, 1)  #电能单位为千瓦时
                    new_vehicle.start_time = center.current_time
                    center.vehicles.append(new_vehicle)

                    if charge_num == 0:
                        center.edges[true_path[0]].capacity['all'] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity["all"], 1)
                        if new_vehicle.log:
                            print(f'在车辆{new_vehicle.id}初始化中道路{true_path[0]}总流量+1')
                        center.edges[true_path[0]].capacity[next] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity[next], 1)
                        flow_ind = i
                        for path_ind in range(0, len(true_path)):
                            edge_id = true_path[path_ind]
                            edge_o = center.edges[edge_id].origin
                            edge_d = center.edges[edge_id].destination
                            time_interval = round(center.edges[edge_id].calculate_time())
                            # print(f"edge_id: {edge_id}, edge_o: {edge_o}, edge_d: {edge_d}, time_interval: {time_interval}")
                            while time_interval >= 1 and flow_ind <= 20 * 3:
                                center.edge_timely_estimated_load[(edge_o, edge_d)][flow_ind][0] += 1
                                time_interval -= 1
                                flow_ind += 1
                        new_vehicle.drive()

                    else:
                        charge_v.append(v_index)
                        rr = random.randint(0, 1)
                        # new_vehicle.E = 4.2 if rr == 0 else 6
                        # new_vehicle.anxiety = 1 if rr == 0 else 0
                        new_vehicle.E = 4.2 if charge_num % 2 == 0 else 6
                        new_vehicle.anxiety = 1 if charge_num % 2 == 0 else 0
                        charge_num -= 1
                        charge_od.append((O, D))
                        center.charge_id.append(v_index)


                elif len(choice) == 1:
                    path = choice[0]
                    true_path = []
                    for j in range(0, len(path) - 1):
                        for edge in center.edges.values():
                            if edge.origin == path[j] and edge.destination == path[j + 1]:
                                true_path.append(edge.id)
                    if len(true_path) > 1:
                        next = true_path[1]
                    else:
                        next = -1
                    new_vehicle = TNplus.Vehicle(v_index, center, O, D, center.edges[true_path[0]].length,
                                                 true_path[0], next,
                                                 true_path, 60, random.randint(48, 54), 0.05, 0.15, 0, {}, 1)  #电能单位为千瓦时
                    new_vehicle.start_time = center.current_time
                    if charge_num == 0:
                        center.edges[true_path[0]].capacity['all'] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity["all"], 1)
                        if new_vehicle.log:
                            print(f'在车辆{new_vehicle.id}初始化中道路{true_path[0]}总流量+1')
                        center.edges[true_path[0]].capacity[next] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity[next], 1)
                    center.vehicles.append(new_vehicle)

                    if charge_num == 0:
                        flow_ind = i
                        for path_ind in range(0, len(true_path)):
                            edge_id = true_path[path_ind]
                            edge_o = center.edges[edge_id].origin
                            edge_d = center.edges[edge_id].destination
                            time_interval = round(center.edges[edge_id].calculate_time())
                            # print(f"edge_id: {edge_id}, edge_o: {edge_o}, edge_d: {edge_d}, time_interval: {time_interval}")
                            while time_interval >= 1 and flow_ind <= 60:
                                center.edge_timely_estimated_load[(edge_o, edge_d)][flow_ind][0] += 1
                                time_interval -= 1
                                flow_ind += 1
                        new_vehicle.drive()

                    else:
                        charge_v.append(v_index)
                        rr = random.randint(0, 1)
                        # new_vehicle.E = 4.2 if rr == 0 else 6
                        # new_vehicle.anxiety = 1 if rr == 0 else 0
                        new_vehicle.E = 4.2 if charge_num % 2 == 0 else 6
                        new_vehicle.anxiety = 1 if charge_num % 2 == 0 else 0
                        charge_num -= 1
                        charge_od.append((O, D))
                        center.charge_id.append(v_index)

                v_index += 1

            road_index = 1
            kk_list = {}
            for index, row in edge_data.iterrows():
                road = center.edges[road_index]
                des_id = road.destination
                destination = center.nodes[des_id]
                # print((int(row['init_node']), int(row['term_node'])))
                kk_list[(int(row['init_node']), int(row['term_node']))] = (1 + road.b * (
                            road.k ** road.power))
                road_index += 1

            kkk_list = {}
            edge_index = 1
            # for (o, d), length in kk_list.items():
            #     kkk_list[edge_index] = (o, d, length)

            for edge in center.edges.values():
                kkk_list[edge.id] = (edge.origin, edge.destination, edge.length * kk_list[(edge.origin, edge.destination)])


            wait_times_list = [{}] * (num_nodes + 1)
            for node in center.nodes.values():
                for (enter, off) in node.signal.keys():
                    wait_times_list[node.id][(enter, off)] = node.calculate_wait(enter, off)


            G_k = processor.build_graph(csv_net_path, kk_list)
            path_results = processor.process_paths(kk_list)

            real_path_results = processplus(kkk_list, wait_times_list, k, center)
            for node in center.nodes.values():
                real_path_results[(node.id, node.id)] = [([], 0, 0)]
            for (o, d), result in path_results.items():
                result[0].clear()
                result[0].append(real_path_results[(o, d)][0][0])
                real_path_sum = real_path_results[(o, d)][0][1] + real_path_results[(o, d)][0][2]
                for iii in range(0, len(real_path_results[(o, d)])):
                    if iii > 0 and real_path_results[(o, d)][iii][1] + real_path_results[(o, d)][iii][2] - real_path_sum <= 1:
                        result[0].append(real_path_results[(o, d)][iii][0])
            # print(kk_list)
            # print(kkk_list)
            # print(' ')
            # print(wait_times_list)
            # print(1145141919810)
            # print(path_results)
            # print(11223344556677889900998877665544332211)
            # print(real_path_results)
            # print(666666666666666666666)


            # print(f"传进dispatch的参数{i}")
            # print(f"传进dispatch的参数{path_results}")
            # print(f"charge_v{charge_v}")
            print(f"传进dispatch的i：{i}")
            print(f"充电车数量为{len(charge_v)}")
            # if i / T <= 2:
            # center.dispatch(charge_v, path_results, i)
            # # else:
            # center.dispatch_plus(t, charge_v, center, batch_size, path_results, 1, 0)
            # for od, result in real_path_results.items():
            #     if len(result) == 1:
            #         result.append(result[0])
            print(len(charge_v))
            cnt_charge_od = {}
            cnt_anxiety_charge_od = {}
            for iiiiiiiiii in range(len(charge_v)):
                charge_vehicle = center.vehicles[charge_v[iiiiiiiiii]]
                if charge_vehicle.anxiety == 1:
                    if (charge_od[iiiiiiiiii][0], charge_od[iiiiiiiiii][1]) in cnt_anxiety_charge_od:
                        cnt_anxiety_charge_od[(charge_od[iiiiiiiiii][0], charge_od[iiiiiiiiii][1])] += 1
                    else:
                        cnt_anxiety_charge_od[(charge_od[iiiiiiiiii][0], charge_od[iiiiiiiiii][1])] = 1
                else:
                    if (charge_od[iiiiiiiiii][0], charge_od[iiiiiiiiii][1]) in cnt_charge_od:
                        cnt_charge_od[(charge_od[iiiiiiiiii][0], charge_od[iiiiiiiiii][1])] += 1
                    else:
                        cnt_charge_od[(charge_od[iiiiiiiiii][0], charge_od[iiiiiiiiii][1])] = 1

            # for (o, d) in charge_od:
            #     if (o, d) in cnt_charge_od:
            #         cnt_charge_od[(o, d)] += 1
            #     else:
            #         cnt_charge_od[(o, d)] = 1
            print(cnt_charge_od)
            print(cnt_anxiety_charge_od)
            center.dispatch_promax(i, center, real_path_results, charge_v, charge_od, 20, 4, 4, lmp_dict, 1, cnt_charge_od, cnt_anxiety_charge_od)
            #毕设用的种群大小50，进化代数250

        for cs in center.charge_stations.values():
            cs.process()

        # print(f'for {i}: {center.calculate_lost()}')


    print("dispatch list")
    print(TNplus.dispatch_list)
    # print("path results")
    # print(path_results)
    # if all_log:
    v = []
    for vehicle in center.vehicles:
        if vehicle.road == -1:
            v.append(vehicle.id)
    # print(v)

    sum1 = len(v)
    sum2 = 0
    for edge in center.edges.values():
        print(f"{edge.id} : {edge.capacity}")
        print(' ')
        sum2 += edge.capacity['all'][1]

    # for cs in center.charge_stations.values():
    #     print(f"{cs.id} : {cs.capacity}")
    #     print(f"{cs.id} : {cs.dispatch}")
    #     print(f"{cs.id} : {len(cs.queue[120])}")
    #     print(f"{cs.id} : {cs.queue}")
    #     print(f"{cs.id} : {len(cs.charge[120])}")
    #     print(f"{cs.id} : {cs.charge}")
    #     print(' ')

    print(pdn_result)
    print(tn_result)
    print(f"道路行驶车辆{sum2}")
    print(f"已到达车辆{sum1}")
    print(f"总共流统计{sum1 + sum2}")
    print(center.dispatch_time_cnt)

    sum_a = 0
    sum_d = 0
    sum_w = 0
    sum_v = 0
    sum_c = 0
    sum_us = 0
    sum_uv = 0
    min_start = 0
    sum_aa = 0
    sum_ad = 0
    sum_aw = 0
    sum_av = 0
    sum_ac = 0
    for final_vehicle in center.vehicles:
        if final_vehicle.anxiety != -1:
            if final_vehicle.arrive_time <= final_vehicle.start_time:
                final_vehicle.arrive_time = 60
                sum_us += 60 - final_vehicle.arrive_time
            sum_a += final_vehicle.arrive_time - final_vehicle.start_time
            sum_d += final_vehicle.arrive_time - final_vehicle.start_time - final_vehicle.total_wait - final_vehicle.total_charge
            sum_w += final_vehicle.total_wait
            sum_c += final_vehicle.total_charge
            sum_v += 1
            if final_vehicle.road == -1:
                min_start = max(min_start, final_vehicle.start_time)
                sum_aa += final_vehicle.arrive_time - final_vehicle.start_time
                sum_ad += final_vehicle.arrive_time - final_vehicle.start_time - final_vehicle.total_wait - final_vehicle.total_charge
                sum_aw += final_vehicle.total_wait
                sum_ac += final_vehicle.total_charge
                sum_av += 1
    print(f"交通侧总时间{sum_a}, 总行驶时间{sum_d}, 总等待时间{sum_w}，总充电时间{sum_c}，总统计车数{sum_v}")
    print(f"到达车辆：交通侧总时间{sum_aa}, 总行驶时间{sum_ad}, 总等待时间{sum_aw}，总充电时间{sum_ac}，总统计车数{sum_av},最晚开始行驶时间{min_start},未到达车辆总时间{sum_us}")



    # # 假设x轴为序号
    # x = list(i * 3 for i in range(1, len(pdn_result) + 1))
    #
    # # 创建一个新的图形
    # plt.figure()
    #
    # # 绘制 pdn_result 数据
    # plt.plot(x, pdn_result, label='pdn_result', marker='o')
    #
    #
    # # 添加图例
    # plt.legend()
    #
    # # 显示图形
    # plt.show()
    #
    # # 创建一个新的图形
    # plt.figure()
    #
    # # 绘制 tn_result 数据
    # plt.plot(x, tn_result, label='tn_result', marker='s')
    #
    # # 添加图例
    # plt.legend()
    #
    # # 显示图形
    # plt.show()

