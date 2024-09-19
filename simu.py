import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import TN
import loaddata as ld
import random

standard_speed = 60 #km/h
t = 10 #min
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



class DispatchCenter(TN.Vehicle, TN.Edge, TN.Node, TN.ChargeStation):
    def __init__(self, vehicles, edges, nodes, charge_stations):
        self.vehicles = vehicles
        self.edges = edges
        self.nodes = nodes
        self.charge_stations = charge_stations





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



    center = DispatchCenter([], [], [], [])



    edge_data = pd.read_csv(csv_net_path, usecols=['init_node', 'term_node', 'capacity', 'length', 'free_flow_time'])
    edge_data = edge_data.dropna()
    for index, row in edge_data.iterrows():
        origin = row['init_node']
        destination = row['term_node']
        capacity = row['capacity']
        length = row['length']
        free_flow_time = row['free_flow_time']
        center.edges.append({(origin, destination): TN.Edge(origin, destination, capacity, free_flow_time, length, 0.15, 4)})





    v_index = 0

    for i in range(1, 451):
        if i % 10 == 1:
            OD = OD_results[int(i / 15)]

            for (O,D) in OD:
                choice = path_results[(O, D)][0]
                if len(choice) > 1:
                    path = choice[random.randint(1, len(choice)) - 1]
                    center.vehicles.append(TN.Vehicle(v_index, O, D, G.get_edge_data(path[0], path[1])['weight'],
                                                      (path[0], path[1]), (path[1], path[2]),
                                                      path, 100, 0.05, 0.15, 0, 0, ))
                    v_index += 1
                elif len(choice) == 1:
                    path = choice[0]
                    center.vehicles.append(TN.Vehicle(v_index, O, D, G.get_edge_data(path[0], path[1])['weight'],
                                                      (path[0], path[1]), (),
                                                      path, 100, 0.05, 0.15, 0, 0, ))
                    v_index += 1
