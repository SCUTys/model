import random
import numpy as np
import math
import os
import csv
import ast
import sympy as sp
import EAalgorithm
import MOalgorithm
import time
from scipy.special import gammaln

t = 1 #min
T = 10 #min
cs_SF = [1, 5, 11, 13, 15, 20]
cs_SF_bus = [1, 3, 7, 6, 9, 13]
cs_EMA = [6, 10, 11, 17, 19, 22, 23, 25, 27, 29, 30, 33, 34, 38, 40, 42, 44, 47, 48, 49, 52, 57, 60, 63, 65, 69]
cs_PY = [167, 1, 3, 19, 20, 25, 29, 31, 33, 40, 46, 59, 62, 64, 69, 79, 81, 83, 94, 95, 96, 100, 103, 107, 111, 116, 136, 140, 141, 146, 155, 165]
cs = cs_SF
cs_bus = cs_SF_bus
output_for_demo = False
dispatch_list = {}
use_csv_input = False



class DispatchCenter:
    """
    调度中心，存储所有数据并调度车辆
    """
    def __init__(self, vehicles, edges, nodes, charge_stations, charge_id, log = False):
        """
        存储四大主体的集合，都是直接存的对象而非id
        :param vehicles: 车辆集合（对象）
        :param edges:  道路集合（对象）
        :param nodes:  交叉口集合（对象）
        :param charge_stations:  充电站集合（对象）
        """
        self.vehicles = vehicles
        self.edges = edges
        self.nodes = nodes
        self.charge_stations = charge_stations
        self.charge_id = charge_id
        self.edge_timely_estimated_load = {}
        self.edge_timely_estimated_load_bpr = {}
        self.log = log
        self.delay_vehicles = [[] for _ in range(100)]
        self.dispatch_time_cnt = []

    def calculate_path(self, path):
        return [edge.id for i in range(len(path) - 1) for edge in self.edges.values() if
                edge.origin == path[i] and edge.destination == path[i + 1]]

    def vehicle(self):
        return self.vehicles

    def edge(self):
        return self.edges

    def node(self):
        return self.nodes

    def charge_station(self):
        return self.charge_stations

    def solve_tuple(self, tu, n, i = 1):
        """
        更新元组值的工具函数
        :param i:
        :param tu: 元组
        :param n: 改变量（+）
        :return: 修改后元组
        """
        tu_list = list(tu)
        tu_list[i] += n
        return tuple(tu_list)

    def calculate_lost(self):
        """
        计算仍未到达终点的需充电车辆数
        :return: 未到达终点的充电车辆数量
        """
        not_arrived_count = 0

        # 只统计charge_id列表中的车辆，且它们还没有到达终点
        for vehicle_id in self.charge_id:
            vehicle = self.vehicles[vehicle_id]
            if vehicle.road != -1:  # 如果车辆仍在道路上（未到达终点）
                not_arrived_count += 1

        return not_arrived_count

    def dispatch_promax(self, t, center, real_path_results, charge_v, charge_od, num_population, num_cs, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio = None):

        # #NSGA2个体调度
        # cs_result, cs_for_choice, real_cs_ids = EAalgorithm.dispatch_cs_nsga2(center, real_path_results, charge_v, charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter)
        # # dispatched_path, _ , actual_path = EAalgorithm.dispatch_path_ga(cs_result, cs_for_choice, center, real_path_results, charge_v, charge_od, num_population, num_cs, max_iter)
        # _, actual_path = EAalgorithm.generate_shortest_actual_paths(cs_result, charge_od, real_path_results, cs_for_choice, charge_v)
        #
        # print("Dispatching finished")
        # # print(real_cs_ids)
        # # print(actual_path)
        #
        # for i, vehicle_id in enumerate(charge_v):
        #     vehicle = self.vehicles[vehicle_id]
        #     path = actual_path[i]
        #     print(path, vehicle.origin, vehicle.destination)
        #     vehicle.path = self.calculate_path(path)
        #     print(vehicle.path)
        #     vehicle.road = vehicle.path[0]
        #     vehicle.next_road = vehicle.path[1] if len(vehicle.path) > 1 else -1
        #     vehicle.charge = (real_cs_ids[i], list(self.charge_stations[real_cs_ids[i]].pile.keys())[0])  # Assuming the first pile for simplicity
        #
        #     # Update the flow on the roads
        #     time = sum(self.edges[road_id].calculate_drive() for road_id in vehicle.path)
        #     self.charge_stations[vehicle.charge[0]].dispatch[vehicle.id] = time
        #     self.edges[vehicle.road].capacity["all"] = self.solve_tuple(self.edges[vehicle.road].capacity["all"], 1)
        #     if vehicle.id in vehicle.center.charge_id:
        #         self.edges[vehicle.road].capacity["charge"] = self.solve_tuple(
        #             self.edges[vehicle.road].capacity["charge"], 1)
        #     self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple(
        #         self.edges[vehicle.road].capacity[vehicle.next_road], 1)
        #
        #     vehicle.drive()

        #CCRP、CCRPP
        # print(5198186941684986189)
        # print(center.edge_timely_estimated_load)
        # start = time.time()
        # dispatch_result, traffic_flow, anxiety_result = EAalgorithm.dispatch_CCRP(t, center, OD_ratio, cs, charge_v, anxiety_OD_ratio)
        # # dispatch_result, traffic_flow, anxiety_result = EAalgorithm.dispatch_CCRPP(t, center, OD_ratio, cs, charge_v, anxiety_OD_ratio)
        # end = time.time()
        # self.dispatch_time_cnt.append(end - start)
        # print(f"Dispatching finished in {end - start} seconds")
        # EAalgorithm.update_center_for_heuristic(center, dispatch_result, t, charge_v, anxiety_result)


        # MOPSO
        print(231342767)
        REP, cs_for_choice, anxiety_cs_for_choice = MOalgorithm.dispatch_cs_MOPSO(center, real_path_results, charge_v,
                        charge_od, num_population, num_cs, cs, cs_bus, lmp_dict, max_iter, OD_ratio, anxiety_OD_ratio)
        MOalgorithm.dispatch_vehicles_by_mopso(center, REP, charge_v, OD_ratio, cs_for_choice, real_path_results,
                                   anxiety_OD_ratio, anxiety_cs_for_choice)




    def dispatch(self, charging_vehicles, path_results, current_time):
        """
        根据某种方式调度要充电的车，这里由于没接算法写了个随机数
        :param charging_vehicles:需要调度的车辆id集合
        :param path_results:之前根据OD生成的路径（已途径节点表征）
        :param current_time:当前时间
        :return:无
        """


        #周期获取充电需求
        total_charge_cost = {}
        for charge_station in self.charge_stations.values():
            total_charge_cost[f"EVCS {charge_station.id}"] = charge_station.cost / 10 / 1000
            charge_station.cost = 0
        # print(total_charge_cost)
        # print("获取充电信息完成")

        #周期获取每个充电站该周期内到达车数（这里按功率直接分开）
        arrive_num = {}
        for charge_station in self.charge_stations.values():
            for pile in charge_station.pile.keys():
                # print(charge_station.v_arrive[pile])
                arrive_num[(charge_station.id, pile)] = charge_station.v_arrive[pile] / T
                charge_station.v_arrive[pile] = 0
        # print("获取到达车数完成")

        #周期获取每个充电站车辆平均充电时长
        charge_time = {}
        for charge_station in self.charge_stations.values():
            for pile in charge_station.pile.keys():
                if self.log:
                    print("Calculating charge_time")
                    print(charge_station.t_cost[pile])
                if charge_station.t_cost[pile][1] > 0 and charge_station.t_cost[pile][0] > 0:
                    charge_time[(charge_station.id, pile)] = charge_station.t_cost[pile][0] / charge_station.t_cost[pile][1]
                else:
                    charge_time[(charge_station.id, pile)] = 0 #待商榷
                charge_station.t_cost[pile] = self.solve_tuple(charge_station.t_cost[pile],
                                                                -charge_station.t_cost[pile][0], 0)
                charge_station.t_cost[pile] = self.solve_tuple(charge_station.t_cost[pile],
                                                                -charge_station.t_cost[pile][1], 1)
        # print("获取平均时长完成")




        #根据需求生成电价（也是示例）
        if 0 not in list(total_charge_cost.values()):
            cost = list(total_charge_cost.values())
            log_reversed_cost = [-np.log(x) for x in cost]
            exp_prob = np.exp(log_reversed_cost)
            softmax_prob = exp_prob / np.sum(exp_prob)
            cs_prob = softmax_prob / np.sum(softmax_prob)
        else:
            cs_prob = [1 / len(cs)] * len(cs)
        # print("生成电价完成")



        def calculate_TN_time_and_energy(path, eps):
            road_time = 0
            junction_time = 0
            for i in range(0, len(path) - 1):
                road_time += self.edges[path[i]].calculate_time()
                node_id = self.edges[path[i]].destination
                junction_time += self.nodes[node_id].calculate_wait(path[i], path[i + 1])
            return road_time * vehicle.Edrive + junction_time* vehicle.Ewait + eps


        def calculate_cs_wait_time(cs_id, cs_power):
            charge_s = self.charge_stations[cs_id]
            if charge_time[cs_id, cs_power] == 0:
                return 0
            else:
                cs_time = charge_s.calculate_wait_cs(charge_s.pile[cs_power],
                                                     charge_s.capacity * charge_s.pile[cs_power] / sum(charge_s.pile.values()),
                                                     arrive_num[cs_id, cs_power],
                                                     1 / charge_time[cs_id, cs_power])
                return cs_time

        wait_cs = {}
        print("预处理排队花费")
        for c in cs:
            for p in self.charge_stations[c].pile.keys():
                wait_cs[(c, p)] = calculate_cs_wait_time(c, p)
        print("预处理完成")


        def sort(var, var_cost): #按cost升序排序var中成分
            list_pairs = zip(var, var_cost)
            sorted_pairs = sorted(list_pairs, key=lambda x: x[1], reverse=True)
            sorted_var, sorted_var_cost = zip(*sorted_pairs)
            sorted_var = list(sorted_var)
            return sorted_var

        def get_vehicles_on_roads(self):
            """
            Returns a dictionary with road IDs as keys and a collection of vehicle information as values.
            Vehicle information includes id, distance, iswait, and next_road.
            """
            road_vehicles = {}
            for demo_v in self.vehicles:
                if demo_v.road != -1:
                    v_info = {
                        'id': demo_v.id,
                        'distance': demo_v.distance,
                        'iswait': demo_v.is_wait,
                        'next_road': demo_v.next_road
                    }
                    if demo_v.road not in road_vehicles:
                        road_vehicles[demo_v.road] = []
                    road_vehicles[demo_v.road].append(v_info)
            return road_vehicles



        #算法优化调度（这里还是写个随机数）
        def assign_cs(vehicle):
            if use_csv_input:
                print("直接使用输入的调度结果")
                with open('vehicle_info.csv', mode='r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if int(row['id']) == vehicle.id:
                            vehicle.charge = (row['cs'], row['power'])
                            vehicle.path = ast.literal_eval(row['path'])
                            vehicle.distance = self.edges[vehicle.path[0]].length
                            vehicle.road = vehicle.path[0]
                            if len(vehicle.path) > 1:
                                vehicle.next_road = vehicle.path[1]
                            else:
                                vehicle.next_road = -1
                            break

                    time = sum(self.edges[i].calculate_drive() for i in vehicle.path)
                    self.charge_stations[vehicle.charge[0]].dispatch[vehicle.id] = current_time + time
                    if self.log:
                        print(f"车辆{vehicle.id} 在 {current_time}影响交通流")
                    self.edges[vehicle.road].capacity["all"] = self.solve_tuple(self.edges[vehicle.road].capacity["all"], 1)
                    if vehicle.id in vehicle.center.charge_id:
                        self.edges[vehicle.road].capacity["charge"] = self.solve_tuple(self.edges[vehicle.road].capacity["charge"], 1)
                    self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple(
                        self.edges[vehicle.road].capacity[vehicle.next_road], 1)
                    if self.log:
                        print(f'在车辆 {vehicle.id} dispatch中道路{vehicle.road}总流量+1')
                    vehicle.drive()

            else:
                # print("随机分配充电站")
                if vehicle.origin in cs: #若起点可充电，直接在起点充电
                    # print("起点可充")
                    charge_id = vehicle.origin
                    vehicle.charge = (vehicle.origin, list(self.charge_stations[vehicle.origin].pile.keys())
                    [random.randint(0, len(list(self.charge_stations[vehicle.origin].pile.keys())) - 1)])
                elif vehicle.destination in cs: #若终点可充电，判断是否能到达终点，可以就分配至终点
                    # print("终点可充")
                    des_path_list = path_results[(vehicle.origin, vehicle.destination)][0]
                    # print(f"起点{vehicle.origin}, 终点{vehicle.destination}")
                    path_id_list = []
                    path_cost = []
                    min_cs_power = 120
                    # min_cs_power = min(list(self.charge_stations[vehicle.destination].pile.keys()), key=lambda cs_power: calculate_cs_wait_time(vehicle.destination, cs_power))
                    # print(f"vehicle id : {vehicle.id}, {calculate_cs_wait_time(vehicle.destination, min_cs_power)}")
                    # print("开始写去终点路径")
                    for path in des_path_list:
                        p_path = calculate_path(path)
                        path_id_list.append(p_path)
                        path_cost.append(calculate_TN_time_and_energy(p_path, 0))
                    # print("有完没完")
                    sorted_path = sort(path_id_list, path_cost)
                    # print("去终点路径写好了")
                    if vehicle.E >= wait_cs[(vehicle.destination, min_cs_power)] * vehicle.Ewait + calculate_TN_time_and_energy(sorted_path[0], 0):
                        charge_id = vehicle.destination
                        vehicle.charge = (vehicle.destination, min_cs_power)
                        vehicle.path = sorted_path[0]
                        vehicle.road = vehicle.path[0]
                        if len(vehicle.path) > 1:
                            vehicle.next_road = vehicle.path[1]
                        else:
                            vehicle.next_road = -1
                    else:
                        adjusted_prob = [p if cs[i] != vehicle.destination else 0 for i, p in enumerate(cs_prob)]
                        total_prob = sum(adjusted_prob)
                        normalized_prob = [p / total_prob for p in adjusted_prob]
                        charge_id = random.choices(cs, normalized_prob)[0]
                        vehicle.charge = (charge_id, list(self.charge_stations[charge_id].pile.keys())
                        [random.randint(0, len(list(self.charge_stations[charge_id].pile.keys())) - 1)])

                    # charge_id = vehicle.destination
                    # vehicle.charge = (vehicle.destination, list(self.charge_stations[vehicle.destination].pile.keys())
                    #     [random.randint(0, len(list(self.charge_stations[vehicle.destination].pile.keys())) - 1)])
                else:
                    # print("中间充电")
                    charge_id = random.choices(cs, cs_prob)[0]
                    vehicle.charge = (charge_id, list(self.charge_stations[charge_id].pile.keys())
                    [random.randint(0, len(list(self.charge_stations[charge_id].pile.keys())) - 1)])

                if self.log:
                    print(f"车辆 {vehicle.id} 分配到充电站{charge_id}, 充电功率为{vehicle.charge[1]}")

        def calculate_path(path):
            return [edge.id for i in range(len(path) - 1) for edge in self.edges.values() if
                    edge.origin == path[i] and edge.destination == path[i + 1]]

        def process_path(vehicle):
            des_path_list1 = path_results[(vehicle.origin, vehicle.charge[0])][0]
            path_id_list1 = []
            path_cost1 = []
            min_cs_power1 = min(list(self.charge_stations[vehicle.charge[0]].pile.keys()),
                               key=lambda cs_power: wait_cs[(vehicle.charge[0], cs_power)])
            for path in des_path_list1:
                p_path = calculate_path(path)
                path_id_list1.append(p_path)
                path_cost1.append(calculate_TN_time_and_energy(p_path, 0))
            sorted_path1 = sort(path_id_list1, path_cost1)
            if (vehicle.E >= wait_cs[(vehicle.charge[0], min_cs_power1)] * vehicle.Ewait
                                                            + calculate_TN_time_and_energy(sorted_path1[0], 0)):
                vehicle.charge = (vehicle.charge[0], min_cs_power1)
                path1 = sorted_path1[0]
            else:
                path1 = sorted_path1[0]

            des_path_list2 = path_results[(vehicle.charge[0], vehicle.destination)][0]
            path_id_list2 = []
            path_cost2 = []
            for path in des_path_list2:
                p_path = calculate_path(path)
                path_id_list2.append(p_path)
                path_cost2.append(calculate_TN_time_and_energy(p_path, 0))
            sorted_path2 = sort(path_id_list2, path_cost2)
            if vehicle.E >= calculate_TN_time_and_energy(sorted_path2[0], 0):
                path2 = sorted_path2[0]
            else:
                path2 = sorted_path2[0]

            vehicle.path = path1 + path2
            vehicle.road = vehicle.path[0]
            vehicle.next_road = vehicle.path[1]


        # #调度结果影响交通
        def update_flow(vehicle, path):
            time = sum(self.edges[i].calculate_drive() for i in path)
            self.charge_stations[vehicle.charge[0]].dispatch[vehicle.id] = current_time + time
            if self.log:
                print(f"车辆{vehicle.id} 在 {current_time}影响交通流")
            self.edges[vehicle.road].capacity["all"] = self.solve_tuple(self.edges[vehicle.road].capacity["all"], 1)
            if vehicle.id in vehicle.center.charge_id:
                self.edges[vehicle.road].capacity["charge"] = self.solve_tuple(self.edges[vehicle.road].capacity["charge"], 1)
            self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple(self.edges[vehicle.road].capacity[vehicle.next_road], 1)
            if self.log:
                print(f'在车辆 {vehicle.id} dispatch中道路{vehicle.road}总流量+1')

        def vehicle_process(vehicle):
            if vehicle.charge[0] == vehicle.origin:
                # print("充电站在脸上")
                vehicle.enter_charge()
            elif vehicle.charge[0] == vehicle.destination:
                # print("充电站在终点")
                update_flow(vehicle, vehicle.path)
                vehicle.drive()
            else:
                # print("充电站在中间")
                process_path(vehicle)
                update_flow(vehicle, vehicle.path)
                vehicle.drive()

        #执行主程序(加入demo生成)
        if output_for_demo:
            print("demo测试")
            print(get_vehicles_on_roads(self))
            for edge in self.edges.values():
                print(f"edge {edge.id} capacity: {edge.capacity}")

        # print("开始调度(主函数执行)")
        for v in charging_vehicles:
            vehicle = self.vehicles[v]
            if self.log:
                print(f"车辆 {vehicle.id} 原路径{vehicle.path},从{vehicle.origin}到{vehicle.destination}")
            # print("分配充电站")
            assign_cs(vehicle)
            if not use_csv_input:
                # print("处理路径")
                vehicle_process(vehicle)
            if output_for_demo:
                info = {
                            'id': vehicle.id,
                            'origin': vehicle.origin,
                            'cs': vehicle.charge[0],
                            'power': vehicle.charge[1],
                            'destination': vehicle.destination,
                            'path': vehicle.path
                        }
                # Specify the CSV file name
                csv_file = 'vehicle_info1.csv'

                # Check if the file exists and is not empty
                file_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0

                # Write the dictionary to the CSV file
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=info.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(info)
        return


class Vehicle:
    def __init__(self, id, center, origin, destination, distance, road, next_road, path, Emax, E, Ewait, Edrive, iswait, charge, index=0, log = False):
        """
        存储车辆信息
        :param id: 车辆id
        :param center: 所属调度中心（对象）
        :param origin: 起点id
        :param destination: 目的地id
        :param distance: 距下次转弯所需距离
        :param road: 当前道路id（已到达终点为-1）
        :param next_road: 下一道路id(无下一道路为-1)
        :param path: 车辆路径
        :param Emax: 最大电池储能
        :param E: 当前储能
        :param Ewait: 等待时平均耗能
        :param Edrive: 驾驶时平均耗能
        :param iswait: 是否在等待信号灯
        :param charge: 分配的充电站id与充电桩功率（id, power）
        :param index: 坐标符，在path中指向下一道路
        :param charging: 是否在充电
        :param log：是否打印相关行为信息
        :param speed: 车辆速度
        """
        self.id = id
        self.center = center
        self.origin = origin
        self.destination = destination #Node id
        self.distance = distance
        self.road = road   #id
        self.next_road = next_road
        self.path = path
        self.Emax = Emax  #电池最大容量
        self.E = E
        self.Ewait = Ewait  #等待时单位时间消耗电量
        self.Edrive = Edrive  #行驶时单位时间消耗电量
        self.is_wait = iswait
        self.charge = charge #充电站id, 充电功率， 默认和对应node的id相同（）
        self.index = index
        self.charging = False
        self.speed = -1
        self.wait_time = -1
        self.log = log
        self.delay = False
        self.anxiety = -1 #-1是直接到达终点，0是到达终点但在焦虑范围内（到终点后规划路径去充电），1是到达终点前就要充电

    def drive(self, rate=1):
        """
        车辆行为：驾驶
        若未到达路口，用bpr计算速度，车辆行进
        若到达路口，判断是否为充电站路口/终点
        :param rate: 时间比率，用于缓解误差
        :return:
        """

        if self.speed == -1:
            self.speed = self.center.edges[self.road].calculate_drive()

        if self.wait_time == -1:
            if self.next_road != -1:
                self.wait_time = self.center.nodes[self.center.edges[self.road].destination].calculate_wait(self.road, self.next_road)
            else:
                self.wait_time = 0

        if self.road != -1:
            road = self.center.edges[self.road]
            # print(road.id)
            if rate > 0 and rate < 1:
                drive_distance = self.speed * rate
            else:
                drive_distance = self.speed
            if drive_distance < self.distance:
                self.distance -= drive_distance
                self.E -= drive_distance * self.Edrive
                if self.log:
                    print(f"车辆 {self.id} 行驶了 {drive_distance} ")

            else:
                self.E -= self.distance * self.Edrive
                r = self.distance / drive_distance
                self.distance = 0

                if self.check_destination():
                    if len(self.charge) > 0:
                        if self.destination == self.charge[0]:
                            self.enter_charge()
                    else:
                        if self.log:
                            print(f'车辆 {self.id} 已到达终点{self.destination},不再行驶')
                        road = self.center.edges[self.road]
                        road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], -1)
                        if self.id in self.center.charge_id:
                            road.capacity["charge"] = self.center.solve_tuple(road.capacity["charge"], -1)
                            self.center.charge_id.remove(self.id)
                        road.capacity[-1] = self.center.solve_tuple(road.capacity[-1], -1)
                        if self.log:
                            print(f'在车辆 {self.id} destination中道路{self.road}流量-1')
                        self.road = -1
                elif self.check_charge():
                    self.enter_charge()
                elif not self.check_charge() and self.next_road != -1:
                    self.wait(self.road, self.next_road, 1 - r)
        else:
            print("车辆已到达终点, 不再驾驶")




    def wait(self, fr, to, rate=1):
        """
        车辆行为：等待
        时间到了就切换道路，没到就继续等
        :param fr: 当前道路
        :param to: 通过路口后后道路
        :param rate: 时间比率，用于缓解误差
        :return:
        """
        road = self.center.edges[self.road]
        junction = self.center.nodes[road.destination]
        if self.log:
            print(f"车辆 {self.id} 正在等待 ")
        if self.is_wait == 0 and self.distance == 0:
            self.is_wait = self.wait_time
            if self.is_wait > t * (1 - rate):
                self.is_wait -= t * (1 - rate)
                self.E -= t * self.Ewait * (1 - rate)
            else:
                self.is_wait = 0.001
                self.E -= self.is_wait * self.Ewait
            if self.log:
                print(f"车辆 {self.id} 需要的等待时间为{self.is_wait} ")
            junction.wait.append((self.id, self.is_wait))
        elif self.is_wait > 0:
            if self.is_wait <= t:
                self.E -= self.is_wait * self.Ewait
                r = self.is_wait / t
                self.is_wait = 0
                for tu in junction.wait:
                    if tu[0] == self.id:
                        junction.wait.remove(tu)
                self.change_road(1 - r)
            else:
                self.is_wait -= t
                self.E -= t * self.Ewait


    def change_road(self, rate = 0, from_charge = False):
        """
        车辆行为：切换道路
        在信号灯等待完毕以及离开充电站时进行
        :param from_charge: 是否从充电站出来
        :param rate: 比率，用于缓解误差
        :return:
        """
        if self.index < len(self.path) and self.next_road != -1:
            if not from_charge:
                road = self.center.edges[self.road]
                road.capacity[self.next_road] = self.center.solve_tuple(road.capacity[self.next_road], -1)
                road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], -1)
                if self.id in self.center.charge_id:
                    road.capacity["charge"] = self.center.solve_tuple(road.capacity["charge"], -1)
                if self.log:
                    print(f'在车辆 {self.id} change_road中道路{self.road}流量-1')

            self.road = self.next_road
            if self.index < len(self.path) - 1:
                self.index += 1
                self.next_road = self.path[self.index]
            else:
                self.index += 1
                self.next_road = -1

            road = self.center.edges[self.road]
            self.distance = road.length
            self.speed = road.calculate_drive()
            if self.next_road != -1:
                self.wait_time = self.center.nodes[road.destination].calculate_wait(self.road, self.next_road)
            else:
                self.wait_time = 0
            if self.distance > rate * self.speed:
                self.distance -= rate * self.speed
            else:
                self.distance = 0.001
            road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], 1)
            road.capacity[self.next_road] = self.center.solve_tuple(road.capacity[self.next_road], 1)
            if self.id in self.center.charge_id:
                road.capacity["charge"] = self.center.solve_tuple(road.capacity["charge"], 1)
            if self.log:
                print(f'在车辆 {self.id} change_road中道路{self.road}总流量+1')
            if self.next_road not in road.capacity.keys():
                print("草了")
                print(self.id)
                print(self.path)
                print(self.charge[0])
                print(self.road)
                print(self.next_road)

            if self.log:
                print(f"车辆 {self.id} 转到{self.road}")


    def check_charge(self):
        """
        检查是否处于分配的充电站所在节点
        :return: bool值
        """
        if self.charge == {}:
            return False
        if self.center.edges[self.road].destination == self.charge[0]:
            return True
        else:
            return False


    def enter_charge(self):
        """
        车辆行为：进入当前充电站
        删除调度集合中的对应元素，加入排队队列
        :return:
        """
        self.charging = True
        if self.origin == self.charge[0]:
            if self.log:
                print(f"车辆 {self.id} 进入充电站{self.charge[0]}, 充电功率为{self.charge[1]}")
        else:
            road = self.center.edges[self.road]
            road.capacity[self.next_road] = self.center.solve_tuple(road.capacity[self.next_road], -1)
            road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], -1)
            if self.id in self.center.charge_id:
                road.capacity["charge"] = self.center.solve_tuple(road.capacity["charge"], -1)
            if self.log:
                print(f"车辆 {self.id} 进入充电站{self.charge[0]}, 充电功率为{self.charge[1]}, 同时道路{self.road}流量-1")
        self.center.charge_stations[self.charge[0]].dispatch = {
            i: a for i, a in self.center.charge_stations[self.charge[0]].dispatch.items() if self.id != i
        }
        self.center.charge_stations[self.charge[0]].queue[self.charge[1]].append((self.id, 0))
        self.center.charge_stations[self.charge[0]].v_arrive[self.charge[1]] += 1

    def leave_charge(self, rate = 0):
        """
        车辆行为：离开充电站
        :param rate:比率，用于缓解误差
        :return:
        """
        if self.log:
            print(f'车辆 {self.id} 在{self.charge}充完电离开')
        self.charging = False
        self.E = self.Emax
        if self.destination != self.charge[0]:
            if self.origin == self.charge[0]:
                self.charge = {}
                self.center.edges[self.road].capacity['all'] = self.center.solve_tuple(self.center.edges[self.road].capacity['all'], 1)
                self.center.edges[self.road].capacity[self.next_road] = self.center.solve_tuple(self.center.edges[self.road].capacity[self.next_road], 1)
                if self.id in self.center.charge_id:
                    self.center.edges[self.road].capacity['charge'] = self.center.solve_tuple(self.center.edges[self.road].capacity['charge'], 1)
                self.drive(rate)
            else:
                self.charge = {}
                self.change_road(rate, True)
        else:
            if self.log:
                print(f'车辆{self.id}已到达终点{self.destination},不再行驶')
            self.road = -1
            self.charge = {}



    def check_destination(self):
        """
        检查是否为终点
        :return:
        """
        if self.distance == 0 and self.center.edges[self.road].destination == self.destination and self.next_road == -1:
            return True
        else:
            return False




class Edge:
    def __init__(self, id, center, origin, destination, length, capacity, free_time, b, power, k = 0, log = False):
        """
        道路对象类定义
        :param id: id
        :param center: 所属调度中心
        :param origin: 道路起点节点id
        :param destination: 道路终点节点id
        :param length: 道路长度
        :param capacity: {{to: cap, x}} 表示各方向车道组的容量(内为字典),to为下一条道路的id
        :param free_time: 无车流影响时正常行驶所需时间
        :param b: bpr公式参数
        :param power: bpr公式参数
        :param log:是否打印相关行为信息
        """
        self.id = id
        self.center = center
        self.origin = origin
        self.destination = destination #Node对象
        self.length = length
        self.capacity = capacity
        self.free_time = free_time
        self.b = b
        self.power = power
        self.k = k
        self.log = log

    def update_ratio(self):
        cap, load = self.capacity["all"]
        k = sp.symbols('k')
        equation = self.b * (k ** (self.power + 1)) + k - load / cap / self.free_time
        # print(f"道路{self.id}的方程为{equation}", end = " ")
        solutions = sp.solve(equation, k)
        real_non_negative_solutions = [sol for sol in solutions if sol.is_real and sol >= 0]
        numeric_solutions = [sp.N(sol) for sol in real_non_negative_solutions]
        # print(f"道路{self.id}的k值为{numeric_solutions[0]}")
        self.k = numeric_solutions[0]


    def calculate_drive(self):
        return self.length / self.free_time / (1 + self.b * (self.k ** self.power))


    def calculate_time(self):
        return self.free_time * (1 + self.b * (self.k ** self.power))



class Node:
    def __init__(self, id, center, signal, wait, edge_num, enter, off, ratio):
        """
        节点对象（即路口）
        :param id: 节点id
        :param center: 所属控制中心(对象)
        :param signal: {{fr, to}: {g, C}}, 表示每个方向信号灯所属绿灯时长与总周期时长
        :param wait: {{vid, t}} 表示正在等候的车辆的合集
        :param edge_num: 连接道路总数
        :param enter: 驶入该节点的道路id集合
        :param off: 驶出该节点的道路id集合
        """
        self.id = id
        self.center = center
        self.signal = signal
        self.wait = wait
        self.edge_num = edge_num
        self.enter = enter
        self.off = off
        self.ratio = ratio
        
        
    def update_ratio(self):
        if self.ratio == {}:
            for fr, to in self.signal.keys():
                self.ratio[(fr, to)] = 0
        else:
            for fr, to in self.signal.keys():
                g, c = self.signal[(fr, to)]
                cap, x = self.center.edges[fr].capacity[to]
                edge = self.center.edges[fr]
                k = sp.symbols('k')
                equation = edge.b * (k ** (edge.power + 1)) + k - c / cap / edge.free_time
                # print(f"节点{self.id}的方程为{equation}", end=" ")
                solutions = sp.solve(equation, k)
                real_non_negative_solutions = [sol for sol in solutions if sol.is_real and sol >= 0]
                numeric_solutions = [sp.N(sol) for sol in real_non_negative_solutions]
                self.ratio[(fr, to)] = numeric_solutions[0] * (len(self.off) - 1)
                # print(f"节点{self.id}的比率为{self.ratio[(fr, to)]}")


    def calculate_wait(self, fr, to, k = -1):
        g, c = self.signal[(fr, to)]
        if k < 0:
            return 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, self.ratio[(fr, to)]) * g / c))
        else:
            return 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, k) * g / c))







class ChargeStation:
    def __init__(self, id, center, dispatch, charge, queue, capacity, pile, t_cost, v_arrive, log = False, cost = 0):
        """
        :param id:充电站id，与道路id一致
        :param center:所属控制中心(对象)
        :param dispatch: {v: atime} 表示分配到该站点但仍未到达的车辆，理想情形下按预计到达时间排序
        :param charge: {power:{id, t}} 表示正在充电的车辆与剩余充电时长
        :param queue: {power: {id, t}} 表示正在站内排队等候充电的车辆与其已等待时间
        :param capacity: 最大容量
        :param pile: (power: num) 表示站内充电桩功率与数量
        :param log: 是否打印行为信息
        :param cost：统计上一T内充电用量
        :param t_cost：{power: (t, num)}统计上一时段内所有充电车辆的总数量与总时长
        :param v_arrive：统计上一时段内到达车辆数
        """

        # if v_arrive is None:
        #     v_arrive = {}
        # if t_cost is None:
        #     t_cost = {}
        self.id = id    #这里假设和路口一致
        self.center = center
        self.dispatch = dispatch
        self.charge = charge
        self.queue = queue
        self.capacity = capacity
        self.pile = pile
        self.log = log
        self.cost = 0
        self.t_cost = t_cost
        self.v_arrive = v_arrive

    def process(self):
        """
        充电站内活动：充电的充电，排队的排队，充电桩有空位就将队首车辆送去充电
        :return:
        """
        for p, n in self.pile.items():
            extra_time = []

            while len(self.charge[p]) < n and self.queue[p]:
                v_id = self.queue[p][0][0]
                e = self.center.vehicles[v_id].E
                e_max = self.center.vehicles[v_id].Emax
                v_t = (e_max - e) / p * 60 / 2
                self.charge[p].append((v_id, v_t))
                self.t_cost[p] = self.center.solve_tuple(self.t_cost[p], v_t, 0)
                self.t_cost[p] = self.center.solve_tuple(self.t_cost[p], 1, 1)
                self.queue[p].pop(0)

            if len(self.charge[p]) > 0:
                for index, tu in enumerate(self.charge[p]):
                    if tu[1] > t:
                        self.charge[p][index] = self.center.solve_tuple(tu, -t)
                        self.cost += p * t
                    else:
                        rate = 1 - tu[1] / t
                        if t != tu[1]:
                            extra_time.append(t - tu[1])
                        self.cost += p * tu[1]
                        v = self.center.vehicles[tu[0]]
                        self.charge[p].remove(tu)
                        v.leave_charge(rate)


            while len(self.charge[p]) < n and self.queue[p]:
                v_id = self.queue[p][0][0]
                e = self.center.vehicles[v_id].E
                e_max = self.center.vehicles[v_id].Emax
                v_t = (e_max * 0.9 - e) / p * 60
                if len(extra_time) > 0:
                    v_t -= extra_time[0]  #有个隐患，万一前车留下的空余时间足够后来相应车直接充满可能会有负数，但谁充电总共只充不到1分钟啊
                    self.cost += p / 60 * extra_time[0]
                    extra_time.pop(0)
                self.charge[p].append((v_id, v_t))
                self.t_cost[p] = self.center.solve_tuple(self.t_cost[p], v_t, 0)
                self.t_cost[p] = self.center.solve_tuple(self.t_cost[p], 1, 1)
                self.queue[p].pop(0)

            if len(self.queue[p]) > 0:
                for index, tu in enumerate(self.queue[p]):
                    self.queue[p][index] = self.center.solve_tuple(tu, t)

    def calculate_wait_cs(self, s, k, l, m):
        # s = self.pile[power]
        # k = self.capacity * self.pile[power]/ sum(self.pile.values())
        # l = self.v_arrive / T
        # if self.t_cost[1] != 0:
        #     m = self.t_cost[1] / self.t_cost[0]
        # else:
        #     m = 1

        def calculate_p_0(rou, s, k, rou_s):
            p_0 = 0

            # print("算个啥呢",end='')
            # print(rou, s, k, rou_s)
            for i in range(s):
                # print(i, end=" ")
                p_0 += math.exp(i * math.log(rou) - gammaln(i + 1))

            # print(f"循环完了")
            if rou_s == 1:
                log_term = math.log(k - s + 1) + s * math.log(rou) - math.log(math.factorial(s))
                p_0 += math.exp(log_term)
            else:
                term1 = s * math.log(rou) - math.log(math.factorial(s))
                term2_1 = (k - s + 1) * math.log(rou_s)
                if rou_s < 1:
                    term2 = math.log(1 - math.exp(term2_1)) - math.log(1 - rou_s)
                else:
                    term2 = math.log(math.exp(term2_1) - 1) - math.log(rou_s - 1)

                log_term = term1 + term2
                p_0 += math.exp(log_term)

            # print("p_0算完了")
            return 1 / p_0

        def calculate_p_k(p_0, rou, k, s):
            if k < s:
                print("k < s")
                return p_0 * (rou ** k) / math.exp(gammaln(k + 1))
            else:
                log_p_k = (math.log(p_0)
                           + k * math.log(rou)
                           - gammaln(s + 1)
                           - (k - s) * math.log(s))
                return math.exp(log_p_k)

        def calculate_L_q(p_0, rou, s, k, rou_s):
            if rou_s == 1:
                log_L_q = (math.log(p_0) + s * math.log(rou) + math.log(k - s) + math.log(k - s + 1)
                           - math.log(2) - gammaln(s + 1))
                L_q = math.exp(log_L_q)
            else:
                log_part1 = math.log(p_0) + s * math.log(rou) + math.log(rou_s)
                log_part2_1 = (k - s + 1) * math.log(rou_s) + math.log(k - s)
                log_part2_2 = (k - s) * math.log(rou_s) + math.log(k - s + 1)
                log_part2 = math.log(1 + math.exp(log_part2_1) - math.exp(log_part2_2))
                log_part3 = - gammaln(s + 1) - 2 * math.log(math.fabs(1 - rou_s))

                log_L_q = log_part1 + log_part2 + log_part3
                L_q = math.exp(log_L_q)

            return L_q

        # print("Before calculate")
        # print("s, k, l, m: ", end='')
        # print(s, k, l, m)
        if l == 0 or m == 0:
            return 0
        else:
            rou = l / m
            rou_s = rou / s
            p_0 = calculate_p_0(rou, s, k, rou_s)
            p_k = calculate_p_k(p_0, rou, k, s)
            # print("During calculate")
            # print("rou, rou ** k, math.factorial(s), s, k - s: ", end='')
            # print(rou, rou ** k, math.factorial(s), s, k - s)
            l_e = l * (1 - p_k)
            L_q = calculate_L_q(p_0, rou, s, k, rou_s)
            # print("After calculate")
            # print("p_0, p_k, L_q, l_e: ", end='')
            # print(p_0, p_k, L_q, l_e)
            if l_e == 0:
                return 0
            else:
                # print(L_q / l_e)
                return L_q / l_e


    def dispatch_rank(self, arrive_time):
        cnt = 0
        for v, t in self.dispatch.items():
            if t <= arrive_time:
                cnt += 1
            else:
                break
        return cnt

    def check(self):
        """
        检查充电站内车辆总数是否超过capacity(截至9.26仍未使用，因为调度暂时是随机数)
        :return: bool值
        """
        charge_sum = sum(len(v) for v in self.charge.values())
        queue_sum = sum(len(v) for v in self.queue.values())
        if charge_sum + queue_sum + len(self.dispatch) < self.capacity:
            return True
        else:
            return False