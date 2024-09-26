import numpy as np
import pandas as pd
from pathlib import Path
import TNplus
import loaddata as ld
import random


standard_speed = 60 #km/h
t = 1 #min
T = 30 #min
csv_net_path = 'data/SF/SiouxFalls_net.csv'
csv_od_path = 'data/SF/SiouxFalls_od.csv'
num_nodes = 24



class PathProcessor:
    def __init__(self, file_path, od_pairs):
        self.file_path = file_path
        self.od_pairs = od_pairs

    def process_paths(self):
        return ld.main(self.file_path, self.od_pairs)

    def get_shortest_path(self, G, origin, destination):
        return ld.shortest_path(G, origin, destination)

    def process_multiple_od_pairs(self, G):
        return ld.process_od_pairs(G, self.od_pairs)

    def build_graph(self, file_path):
        return ld.build_graph(ld.read_csv(self.file_path, ['init_node', 'term_node', 'length']))


class ODGenerator:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        df =  ld.read_csv(self.file_path, ['O', 'D', 'Ton'])
        # df = df.dropna()
        data_dict = { (row['O'], row['D']): int(row['Ton']/100) for _, row in df.iterrows()}
        return data_dict

    def distribute_od_pairs(self, data_dict, elements_per_category=120):
        # 计算总个数和组数
        return ld.distribute_od_pairs(data_dict, elements_per_category)








if __name__ == "__main__":


    od_pairs = []
    for i in range(1, num_nodes + 1):
        for j in range(1, num_nodes + 1):
            if i != j:
                od_pairs.append([i, j])
    # print(od_pairs)
    processor = PathProcessor(csv_net_path, od_pairs)
    G = processor.build_graph(csv_net_path)
    path_results = processor.process_paths()

    generator = ODGenerator(csv_od_path)
    data = generator.load()
    OD_results = generator.distribute_od_pairs(data, 100)


    edge_data = pd.read_csv(csv_net_path, usecols=['init_node', 'term_node', 'capacity', 'length', 'free_flow_time'])
    edge_data = edge_data.dropna()


    center = TNplus.DispatchCenter([], {}, {}, {})
    nodes = []
    edge_id = 1

    for index, row in edge_data.iterrows():
        origin = int(row['init_node'])
        destination = int(row['term_node'])
        capacity = float(row['capacity'] / 1000)
        length = float(row['length'])
        free_flow_time = float(row['free_flow_time'])

        if destination not in nodes:
            nodes.append(destination)
            center.nodes[destination] = TNplus.Node(destination, center, {}, [], False, 1, [], [])

        if origin not in nodes:
            nodes.append(origin)
            center.nodes[origin] = TNplus.Node(origin, center, {}, [], False, 1, [], [])

        center.nodes[origin].edge_num += 1
        center.nodes[destination].enter.append(edge_id)
        center.nodes[destination].edge_num += 1
        center.nodes[origin].off.append(edge_id)

        for n in nodes:
            for enter in center.nodes[n].enter:
                for off in center.nodes[n].off:
                    if enter != -1 and off != -1:
                        center.nodes[n].signal[(enter, off)] = (1.5 / (center.nodes[n].edge_num / 2), 1.5)

        # print(edge_id, center, origin, destination, length)
        edge = TNplus.Edge(edge_id, center, origin, destination, length, {}, free_flow_time,0.15, 4)
        edge.capacity["all"] = (capacity, 0)
        edge.capacity[-1] = (capacity, 0)
        center.edges[edge_id] = edge
        edge_id += 1

    for edge in center.edges.values():
        index = edge.destination
        destination = center.nodes[index]
        for off in destination.off:
            edge.capacity[off] = (edge.capacity['all'][0] / (destination.edge_num / 2 - 1), 0)


    v_index = 0

    for i in range(1, 120):
        for vehicle in center.vehicles:
            if vehicle.charging == False and vehicle.is_wait > 0:
                vehicle.wait(vehicle.road, vehicle.next_road)
            elif vehicle.charging:
                continue
            elif vehicle.road != -1:
                vehicle.drive(vehicle.road)


        if i == 1:
            for i in TNplus.cs:
                center.charge_stations[i] = TNplus.ChargeStation(i, center, {}, {10: []}, {10: []}, 100, {10: 15} )

        if i % 10 == 1 or i == 1:
            OD = OD_results[int(i / 10)]
            charge_num = random.randint(10, 15)
            charge_v = []


            for (O,D) in OD:
                choice = path_results[(O, D)][0]
                if len(choice) > 1:
                    path = choice[random.randint(1, len(choice)) - 1]
                    true_path = []
                    for i in range(0, len(path) - 1):
                        for edge in center.edges.values():
                            if edge.origin == path[i] and edge.destination == path[i + 1]:
                                true_path.append(edge.id)
                    if len(true_path) > 1:
                        next = true_path[1]
                    else:
                        next = -1
                    new_vehicle = TNplus.Vehicle(v_index, center, O, D, center.edges[true_path[0]].length,
                                                    true_path[0], next,
                                                    true_path, 100, 80, 0.05, 0.15, 0, {}, 1)
                    if charge_num == 0:
                        center.edges[true_path[0]].capacity['all'] = new_vehicle.center.solve_tuple(
                            center.edges[true_path[0]].capacity["all"], 1)
                        print(f'在车辆{new_vehicle.id}初始化中道路{true_path[0]}总流量+1')
                        center.edges[true_path[0]].capacity[next] = new_vehicle.center.solve_tuple(
                        center.edges[true_path[0]].capacity[next], 1)
                    # print(center.edges[true_path[0]].capacity)
                    # print('mlgbdcl')
                    center.vehicles.append(new_vehicle)
                    if charge_num == 0:
                        new_vehicle.drive()
                    else:
                        charge_num -= 1
                        charge_v.append(v_index)


                elif len(choice) == 1:
                    path = choice[0]
                    true_path = []
                    for i in range(0, len(path) - 1):
                        for edge in center.edges.values():
                            if edge.origin == path[i] and edge.destination == path[i + 1]:
                                true_path.append(edge.id)
                    if len(true_path) > 1:
                        next = true_path[1]
                    else:
                        next = -1
                    new_vehicle = TNplus.Vehicle(v_index, center, O, D, center.edges[true_path[0]].length,
                                                    true_path[0], next,
                                                    true_path, 100, 80, 0.05, 0.15, 0, {}, 1)
                    if charge_num == 0:
                        center.edges[true_path[0]].capacity['all'] = new_vehicle.center.solve_tuple(center.edges[true_path[0]].capacity["all"], 1)
                        print(f'在车辆{new_vehicle.id}初始化中道路{true_path[0]}总流量+1')
                        center.edges[true_path[0]].capacity[next] = new_vehicle.center.solve_tuple(center.edges[true_path[0]].capacity[next], 1)
                    center.vehicles.append(new_vehicle)
                    if charge_num == 0:
                        new_vehicle.drive()
                    else:
                        charge_num -= 1
                        charge_v.append(v_index)


                v_index += 1
            center.dispatch(charge_v, path_results, i)



        for cs in center.charge_stations.values():
            cs.process()

        print(center.calculate_lost())
        print(i)
        print(999)


    # for edge in center.edges.values():
    #     print(f"{edge.id} : {edge.capacity}")

    for cs in center.charge_stations.values():
        print(f"{cs.id} : {cs.capacity}")
        print(f"{cs.id} : {cs.dispatch}")
        print(f"{cs.id} : {cs.queue}")
        print(f"{cs.id} : {cs.charge}")
