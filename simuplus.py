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
    print(G)
    path_results = processor.process_paths()
    print(path_results)

    generator = ODGenerator(csv_od_path)
    data = generator.load()
    OD_results = generator.distribute_od_pairs(data, 100)
    # print(OD_results)
    # for i in range(len(OD_results)):
    #     print(i, len(OD_results[i]))


    edge_data = pd.read_csv(csv_net_path, usecols=['init_node', 'term_node', 'capacity', 'length', 'free_flow_time'])
    edge_data = edge_data.dropna()
    nodes = []
    edge_id = 1





    center = TNplus.DispatchCenter([], {}, {}, [])


    for index, row in edge_data.iterrows():
        origin = int(row['init_node'])
        destination = int(row['term_node'])
        capacity = float(row['capacity'] / 1000)
        length = float(row['length'])
        free_flow_time = float(row['free_flow_time'])

        if destination not in nodes:
            nodes.append(destination)
            center.nodes[destination] = TNplus.Node(destination, center, {}, {}, False, 1, [], [])
            center.nodes[destination].enter.append(edge_id)
        elif origin not in nodes:
            nodes.append(origin)
            center.nodes[origin] = TNplus.Node(origin, center, {}, {}, False, 1, [], [])
            center.nodes[origin].off.append(edge_id)
        else:
            center.nodes[origin].edge_num += 1
            center.nodes[origin].signal = 1 / center.nodes[origin].edge_num
            center.nodes[destination].edge_num += 1
            center.nodes[destination].signal = 1 / center.nodes[destination].edge_num

        for n in nodes:
            for enter in center.nodes[n].enter:
                for off in center.nodes[n].off:
                    center.nodes[n].signal[(enter, off)] = (90 / (center.nodes[n].edge_num / 2 - 1), 90)

        # print(edge_id, center, origin, destination, length)
        edge = TNplus.Edge(edge_id, center, origin, destination, length, {}, free_flow_time,0.15, 4)
        edge_id += 1
        edge.capacity["all"] = (capacity, 0)
        for i in center.nodes[destination].off:
            if center.nodes[destination].edge_num > 2:
                edge.capacity[i] = (capacity / (center.nodes[destination].edge_num / 2 - 1), 0)
            else:
                edge.capacity[i] = (capacity, 0)
        center.edges[edge_id] = edge




    v_index = 0

    for i in range(1, 451):
        for vehicle in center.vehicles:
            if vehicle.charging == False and vehicle.is_wait > 0:
                vehicle.wait(vehicle.road, vehicle.road.id, vehicle.next_road.id)
            elif vehicle.charging:
                continue
            else:
                vehicle.drive(vehicle.road)

        if i % 10 == 1:
            OD = OD_results[int(i / 15)]

            for (O,D) in OD:
                choice = path_results[(O, D)][0]
                if len(choice) > 1:
                    path = choice[random.randint(1, len(choice)) - 1]
                    true_path = []
                    for i in range(len(path) - 2):
                        for edge in center.edges:
                            if edge.origin == path[i] and edge.destination == path[i + 1]:
                                true_path.append(edge.id)
                    new_vehicle = TNplus.Vehicle(v_index, center, O, D, center.edges[true_path[0]].length,
                                                    true_path[0], true_path[1],
                                                    path, 100, 80, 0.05, 0.15, 0, 0, 0)
                    center.vehicles.append(new_vehicle)
                    new_vehicle.drive(new_vehicle.road)
                    v_index += 1
                elif len(choice) == 1:
                    path = choice[0]
                    true_path = []
                    for i in range(len(path) - 2):
                        for edge in center.edges.values():
                            if edge.origin == path[i] and edge.destination == path[i + 1]:
                                true_path.append(edge.id)
                    new_vehicle = TNplus.Vehicle(v_index, center, O, D, center.edges[true_path[0]].length,
                                                      true_path[0], (),
                                                      path, 100, 80, 0.05, 0.15, 0, 0, 0)
                    center.vehicles.append(new_vehicle)
                    new_vehicle.drive()
                    v_index += 1

        for cs in center.charge_stations:
            cs.process()

        print(center.calculate_lost())
