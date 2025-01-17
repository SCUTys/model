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
import sys
import json

# import PDNplus


standard_speed = 60  #km/h
t = 1  #min
T = 10  #min
T_pdn = 3 * T  #min
roadmap = 'SF'  #目前支持SF和EMA
csv_net_path = 'data/' + roadmap + '/' + roadmap + '_net.csv'
csv_od_path = 'data/' + roadmap + '/' + roadmap + '_od.csv'
node = {'SF': 24, 'EMA': 76}
num_nodes = node[roadmap]
batch_size = 7212
all_log = False
OD_from_csv = True
dispatch_list = {}
k_list = {}
G = None
k = 3
rate = 0.5



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


if __name__ == "__main__":

    od_pairs = []
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            if i != j:
                od_pairs.append([i, j])
                k_list[(i, j)] = 1
    # print(od_pairs)
    processor = PathProcessor(csv_net_path, od_pairs)
    G = processor.build_graph(csv_net_path)
    path_results = processor.process_paths()
    # print(path_results)


    # generator = ODGenerator(csv_od_path)
    # data = generator.load()
    # OD_results = generator.distribute_od_pairs(data, batch_size)
    # random.shuffle(OD_results)
    # for ODs in OD_results:
    #     random.shuffle(ODs)

    generator2 = ODGenerator2(csv_od_path)
    data = generator2.load()
    OD_results = generator2.generate_od_pairs(data)
    random.shuffle(OD_results)
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
    # print(len(OD_results), len(OD_results[0]), len(OD_results[1]), len(OD_results[2]))
    # print("mother fucker")


    edge_data = pd.read_csv(csv_net_path, usecols=['init_node', 'term_node', 'capacity', 'length', 'free_flow_time'])
    edge_data = edge_data.dropna()

    center = TNplus.DispatchCenter([], {}, {}, {}, [], False)
    nodes = []
    edge_id = 1

    for index, row in edge_data.iterrows():
        origin = int(row['init_node'])
        destination = int(row['term_node'])
        capacity = int(row['capacity'] / 10)
        length = float(row['length'])
        free_flow_time = float(row['free_flow_time'])

        if destination not in nodes:
            nodes.append(destination)
            center.nodes[destination] = TNplus.Node(destination, center, {}, [], 1, [], [])

        if origin not in nodes:
            nodes.append(origin)
            center.nodes[origin] = TNplus.Node(origin, center, {}, [], 1, [], [])

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

    v_index = 0
    pdn = PDNplus.create_ieee14()
    pdn_result = []
    tn_result = []
    for i in range(1, 20 * 3):
        # if all_log:
        print(f"主循环 {i}")
        for vehicle in center.vehicles:
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
                center.charge_stations[j] = TNplus.ChargeStation(j, center, {}, {50: [], 120: []}, {50: [], 120: []},
                                                                 25000, {120: 20000},
                                                                 {120: (0, 0)}, {120: 0},
                                                                 False)  #规范充电桩功率为kw

        if i % 3 == 1 or i == 1:
            print("加入新车")
            if all_log:
                print(f"在循环i={i}时加入新OD")
            OD = OD_results[int(i / 3)]
            charge_num = int(batch_size * rate)
            charge_v = []

            if i != 1:
                total_charge_cost = {}
                for cs in center.charge_stations.values():
                    total_charge_cost[f"EVCS {cs.id}"] = cs.cost / 60 / 1000
                    cs.cost = 0
                print(total_charge_cost)
                print(66666)
                PDNplus.update_load(pdn, total_charge_cost, 3 * T / 60)
                PDNplus.run(pdn, 300)
                pdn_loss = PDNplus.calculate_loss(pdn, 140)
                pdn_result.append(pdn_loss)
                print(pdn_loss)
                print(555)
                tn_result.append(center.calculate_lost())
                print(center.calculate_lost())
                print(555)

            for (O, D) in OD:
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
                                                 true_path, 80, 64, 0.05, 0.15, 0, {}, 1)  #电能单位为千瓦时
                    if charge_num == 0:
                        center.edges[true_path[0]].capacity['all'] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity["all"], 1)
                        if new_vehicle.log:
                            print(f'在车辆{new_vehicle.id}初始化中道路{true_path[0]}总流量+1')
                        center.edges[true_path[0]].capacity[next] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity[next], 1)
                    center.vehicles.append(new_vehicle)
                    if charge_num == 0:
                        new_vehicle.drive()
                    else:
                        charge_num -= 1
                        charge_v.append(v_index)
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
                                                 true_path, 80, 64, 0.05, 0.15, 0, {}, 1)  #电能单位为千瓦时
                    if charge_num == 0:
                        center.edges[true_path[0]].capacity['all'] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity["all"], 1)
                        if new_vehicle.log:
                            print(f'在车辆{new_vehicle.id}初始化中道路{true_path[0]}总流量+1')
                        center.edges[true_path[0]].capacity[next] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity[next], 1)
                    center.vehicles.append(new_vehicle)
                    if charge_num == 0:
                        new_vehicle.drive()
                    else:
                        charge_num -= 1
                        charge_v.append(v_index)
                        center.charge_id.append(v_index)

                v_index += 1

            road_index = 1
            for index, row in edge_data.iterrows():
                road = center.edges[road_index]
                k_list[(int(row['init_node']), int(row['term_node']))] = 1 + road.b * (
                            (road.capacity['all'][1] / road.capacity['all'][0]) ** road.power)
                road_index += 1

            G_k = processor.build_graph(csv_net_path, k_list)
            path_results = processor.process_paths(k_list)


            # print(f"传进dispatch的参数{i}")
            # print(f"传进dispatch的参数{path_results}")
            # print(f"charge_v{charge_v}")
            print(f"传进dispatch的i：{i}")
            print(f"充电车数量为{len(charge_v)}")
            print(66666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666)
            # if i / T <= 2:
            center.dispatch(charge_v, path_results, i)
            # # else:
            # center.dispatch_plus(t, charge_v, center, batch_size, path_results, 1, 0)


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

    for cs in center.charge_stations.values():
        print(f"{cs.id} : {cs.capacity}")
        print(f"{cs.id} : {cs.dispatch}")
        print(f"{cs.id} : {len(cs.queue[120])}")
        print(f"{cs.id} : {cs.queue}")
        print(f"{cs.id} : {len(cs.charge[120])}")
        print(f"{cs.id} : {cs.charge}")
        print(' ')

    print(pdn_result)

    print(f"道路行驶车辆{sum2}")
    print(f"已到达车辆{sum1}")
    print(f"总共流统计{sum1 + sum2}")



    # 假设x轴为序号
    x = list(i * 3 for i in range(1, len(pdn_result) + 1))

    # 创建一个新的图形
    plt.figure()

    # 绘制 pdn_result 数据
    plt.plot(x, pdn_result, label='pdn_result', marker='o')


    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

    # 创建一个新的图形
    plt.figure()

    # 绘制 tn_result 数据
    plt.plot(x, tn_result, label='tn_result', marker='s')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

