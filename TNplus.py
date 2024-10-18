import random
import numpy as np
import math

t = 1 #min
T = 10 #min
k = 1
cs = [1, 5, 11, 15, 20]


class DispatchCenter:
    """
    调度中心，存储所有数据并调度车辆
    """
    def __init__(self, vehicles, edges, nodes, charge_stations, log = False):
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
        self.log = log

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
        计算交通部分的总loss
        :return: 当前总loss
        """
        sum_road = 0
        for edge in self.edges.values():
            sum_road += edge.capacity["all"][1] * edge.calculate_drive()
        for node in self.nodes.values():
            for i, is_wait in node.wait:
                sum_road += is_wait
        for charge_station in self.charge_stations.values():
            if list(charge_station.charge.values()) != [[]]:
                for ipair in charge_station.charge.values():
                    for i in ipair:
                        sum_road += i[1]
            for p, n in charge_station.pile.items():
                length = len(charge_station.queue[p])
                if length > 0:
                    for i, time in charge_station.queue[p]:
                        sum_road += length * time
        return sum_road


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
        # print(222)

        #周期获取每个充电站该周期内到达车数（这里按功率直接分开）
        arrive_num = {}
        for charge_station in self.charge_stations.values():
            for pile in charge_station.pile.keys():
                print(charge_station.v_arrive[pile])
                arrive_num[(charge_station.id, pile)] = charge_station.v_arrive[pile] / T
                charge_station.v_arrive[pile] = 0

        #周期获取每个充电站车辆平均充电时长
        charge_time = {}
        for charge_station in self.charge_stations.values():
            for pile in charge_station.pile.keys():
                print("Calculating charge_time")
                print(charge_station.t_cost[pile])
                if charge_station.t_cost[pile][1] > 0:
                    charge_time[(charge_station.id, pile)] = charge_station.t_cost[pile][0] / charge_station.t_cost[pile][1]
                else:
                    charge_time[(charge_station.id, pile)] = 0 #待商榷
                charge_station.t_cost[pile] = self.solve_tuple(charge_station.t_cost[pile],
                                                                -charge_station.t_cost[pile][0], 0)
                charge_station.t_cost[pile] = self.solve_tuple(charge_station.t_cost[pile],
                                                                -charge_station.t_cost[pile][1], 1)


        #根据需求生成电价（也是示例）
        if 0 not in list(total_charge_cost.values()):
            cost = list(total_charge_cost.values())
            log_reversed_cost = [-np.log(x) for x in cost]
            exp_prob = np.exp(log_reversed_cost)
            softmax_prob = exp_prob / np.sum(exp_prob)
            cs_prob = softmax_prob / np.sum(softmax_prob)
        else:
            cs_prob = [1 / len(cs)] * len(cs)



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
            cs_time = charge_s.calculate_wait_cs(charge_s.pile[cs_power],
                                                 charge_s.capacity * charge_s.pile[cs_power] / sum(charge_s.pile.values()),
                                                 arrive_num[cs_id, cs_power],
                                                 charge_time[cs_id, cs_power])
            return cs_time


        def sort(var, var_cost): #按cost升序排序var中成分
            list_pairs = zip(var, var_cost)
            sorted_pairs = sorted(list_pairs, key=lambda x: x[1], reverse=True)
            sorted_var, sorted_var_cost = zip(*sorted_pairs)
            sorted_var = list(sorted_var)
            return sorted_var



        #算法优化调度（这里还是写个随机数）
        def assign_cs(vehicle):
            if vehicle.origin in cs: #若起点可充电，直接在起点充电
                charge_id = vehicle.origin
                vehicle.charge = (vehicle.origin, list(self.charge_stations[vehicle.origin].pile.keys())
                [random.randint(0, len(list(self.charge_stations[vehicle.origin].pile.keys())) - 1)])
            elif vehicle.destination in cs: #若终点可充电，判断是否能到达终点，可以就分配至终点
                des_path_list = path_results[(vehicle.origin, vehicle.destination)][0]
                # print(f"起点{vehicle.origin}, 终点{vehicle.destination}")
                path_id_list = []
                path_cost = []
                min_cs_power = min(list(self.charge_stations[vehicle.destination].pile.keys()), key=lambda cs_power: calculate_cs_wait_time(vehicle.destination, cs_power))
                print(f"vehicle id : {vehicle.id}, {calculate_cs_wait_time(vehicle.destination, min_cs_power)}")
                for path in des_path_list:
                    p_path = calculate_path(path)
                    path_id_list.append(p_path)
                    path_cost.append(calculate_TN_time_and_energy(p_path, 0))
                sorted_path = sort(path_id_list, path_cost)
                if vehicle.E >= calculate_cs_wait_time(vehicle.destination, min_cs_power) * vehicle.Ewait + calculate_TN_time_and_energy(sorted_path[0], 0):
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
                               key=lambda cs_power: calculate_cs_wait_time(vehicle.charge[0], cs_power))
            for path in des_path_list1:
                p_path = calculate_path(path)
                path_id_list1.append(p_path)
                path_cost1.append(calculate_TN_time_and_energy(p_path, 0))
            sorted_path1 = sort(path_id_list1, path_cost1)
            if (vehicle.E >= calculate_cs_wait_time(vehicle.charge[0], min_cs_power1) * vehicle.Ewait
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

            # path1 = random.choice(path_results[(vehicle.origin, vehicle.charge[0])][0])
            # path2 = random.choice(path_results[(vehicle.charge[0], vehicle.destination)][0])
            # print(1919810)
            # print(vehicle.id)
            # print(vehicle.origin)
            # print(vehicle.destination)
            # print(path1)
            # print(path2)
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
            if vehicle.next_road != -1:
                self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple(
                    self.edges[vehicle.road].capacity[vehicle.next_road], 1)
            if self.log:
                print(f'在车辆 {vehicle.id} dispatch中道路{vehicle.road}总流量+1')

        def vehicle_process(vehicle):
            if vehicle.charge[0] == vehicle.origin:
                vehicle.enter_charge()
            elif vehicle.charge[0] == vehicle.destination:
                update_flow(vehicle, vehicle.path)
                vehicle.drive()
            else:
                process_path(vehicle)
                update_flow(vehicle, vehicle.path)
                vehicle.drive()

        #执行主程序
        for v in charging_vehicles:
            vehicle = self.vehicles[v]
            if self.log:
                print(f"车辆 {vehicle.id} 原路径{vehicle.path},从{vehicle.origin}到{vehicle.destination}")
            assign_cs(vehicle)
            vehicle_process(vehicle)


        #备用版本（也就是老版本，可以对照检测下有无出错）
        # for v in charging_vehicles:
        #     vehicle = self.vehicles[v]
        #     if self.log:
        #         print(f"车辆 {vehicle.id} 原路径{vehicle.path},从{vehicle.origin}到{vehicle.destination}")
        #     if vehicle.origin in cs:
        #         vehicle.charge = (vehicle.origin, list(self.charge_stations[vehicle.origin].pile.keys())
        #                             [random.randint(0, len(list(self.charge_stations[vehicle.origin].pile.keys()))-1)])
        #         if self.log:
        #             print(f"车辆 {vehicle.id} 分配到充电站{vehicle.origin} (起点), 充电功率为{vehicle.charge[1]}")
        #         vehicle.enter_charge()
        #     elif vehicle.destination in cs:
        #         vehicle.charge = (vehicle.destination, list(self.charge_stations[vehicle.destination].pile.keys())
        #                             [random.randint(0, len(list(self.charge_stations[vehicle.destination].pile.keys()))-1)])
        #         if self.log:
        #            print(f"车辆 {vehicle.id} 分配到充电站{vehicle.destination} (终点), 充电功率为{vehicle.charge[1]}")
        #         time = 0
        #         for i in vehicle.path:
        #             time += self.edges[i].calculate_drive()
        #         self.charge_stations[vehicle.charge[0]].dispatch[v] = t + time
        #         self.edges[vehicle.road].capacity["all"] = self.solve_tuple(self.edges[vehicle.road].capacity["all"], 1)
        #         self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple(self.edges[vehicle.road].capacity[vehicle.next_road], 1)
        #         if self.log:
        #             print(f'在车辆 {vehicle.id} dispatch中道路{vehicle.road}总流量+1')
        #         vehicle.drive()
        #     else:
        #         c_index = cs[random.randint(0, len(cs) - 1)]  ##这里没并算法就写个随机数吧
        #         vehicle.charge = (c_index, list(self.charge_stations[c_index].pile.keys())
        #                             [random.randint(0, len(list(self.charge_stations[c_index].pile.keys()))-1)])
        #         if self.log:
        #             print(f"车辆 {vehicle.id} 分配到充电站{c_index}, 充电功率为{vehicle.charge[1]} ")
        #         if vehicle.origin != vehicle.charge[0]:
        #             path1 = path_results[(vehicle.origin, vehicle.charge[0])][0]
        #         else:
        #             path1 = []
        #         if vehicle.destination != vehicle.charge[0]:
        #             path2 = path_results[(vehicle.charge[0], vehicle.destination)][0]
        #         else:
        #             path2 = []
        #         if len(path1) >= 1:
        #             path11 = path1[random.randint(0, len(path1) - 1)]
        #         else:
        #             path11 = []
        #
        #         if len(path2) >= 1:
        #             path22 = path2[random.randint(0, len(path2) - 1)]
        #         else:
        #             path22 = []
        #
        #         true_path2 = []
        #         true_path1 = []
        #         for i in range(0, len(path11) - 1):
        #             for edge in self.edges.values():
        #                 if edge.origin == path11[i] and edge.destination == path11[i + 1]:
        #                     true_path1.append(edge.id)
        #         for i in range(0, len(path22) - 1):
        #             for edge in self.edges.values():
        #                 if edge.origin == path22[i] and edge.destination == path22[i + 1]:
        #                     true_path2.append(edge.id)
        #         vehicle.path = true_path1 + true_path2
        #         if self.log:
        #             print(f"重新分配的路径{vehicle.path}")
        #         if len(vehicle.path) > 0:
        #             vehicle.distance = self.edges[vehicle.path[0]].length
        #         vehicle.road = vehicle.path[0]
        #         self.edges[vehicle.road].capacity["all"] = self.solve_tuple(self.edges[vehicle.road].capacity["all"], 1)
        #         if len(vehicle.path) > 1:
        #             vehicle.next_road = vehicle.path[1]
        #         else:
        #             vehicle.next_road = -1
        #         self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple( self.edges[vehicle.road].capacity[vehicle.next_road], 1)
        #         time = 0
        #         for i in true_path1:
        #             time += self.edges[i].calculate_drive()
        #         self.charge_stations[vehicle.charge[0]].dispatch[v] = t + time
        #         if self.log:
        #             print(f'在车辆 {vehicle.id} dispatch中道路{vehicle.road}总流量+1')
        #         vehicle.drive()
        return



class Vehicle:
    def __init__(self, id, center, origin, destination, distance, road, next_road, path, Emax, E, Ewait, Edrive, iswait, charge, index=0, charging = False, log = False):
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
        self.log = log

    def drive(self, rate=1):
        """
        车辆行为：驾驶
        若未到达路口，用bpr计算速度，车辆行进
        若到达路口，判断是否为充电站路口/终点
        :param rate: 时间比率，用于缓解误差
        :return:
        """
        road = self.center.edges[self.road]
        # print(road.id)
        if rate > 0 and rate < 1:
            drive_distance = road.calculate_drive() * rate
        else:
            drive_distance = road.calculate_drive()
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
                    road.capacity[-1] = self.center.solve_tuple(road.capacity[-1], -1)
                    if self.log:
                        print(f'在车辆 {self.id} destination中道路{self.road}总流量-1')
                    self.road = -1
            elif self.check_charge():
                self.enter_charge()
            elif not self.check_charge() and self.next_road != -1:
                self.wait(self.road, self.next_road, 1 - r)




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
        g, c= junction.signal[(fr, to)]
        cap, x = road.capacity[to]
        if self.log:
            print(f"车辆 {self.id} 正在等待 ")
        if self.is_wait == 0 and self.distance == 0:
            if c != 0:
                self.is_wait = 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, x / cap) * g / c))
            else:
                self.change_road(1)
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
                if self.log:
                    print(f'在车辆 {self.id} change_road中道路{self.road}总流量-1')

            self.road = self.next_road

            if self.index < len(self.path) - 1:
                self.index += 1
                self.next_road = self.path[self.index]
            else:
                self.index += 1
                self.next_road = -1

            road = self.center.edges[self.road]
            self.distance = road.length
            if self.distance > rate * self.center.edges[self.road].calculate_drive():
                self.distance -= rate * self.center.edges[self.road].calculate_drive()
            else:
                self.distance = 0.001
            road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], 1)
            if self.log:
                print(f'在车辆 {self.id} change_road中道路{self.road}总流量+1')
            if self.next_road not in road.capacity.keys():
                print("草了")
                print(self.id)
                print(self.path)
                print(self.charge[0])
                print(self.road)
                print(self.next_road)
            road.capacity[self.next_road] = self.center.solve_tuple(road.capacity[self.next_road], 1)
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
                self.drive(rate)
            else:
                self.charge = {}
                self.change_road(rate, True)
        else:
            if self.log:
                print(f'车辆{self.id}已到达终点{self.destination},不再行驶')
            self.road = -1



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
    def __init__(self, id, center, origin, destination, length, capacity, free_time, b, power, log = False):
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
        self.log = log

    def calculate_drive(self):
        """
        BPR公式计算单位时间内可行驶距离
        :return:
        """
        cap, x = self.capacity["all"]
        return self.length / self.free_time / (1 + self.b * (x / cap)**self.power)


    def calculate_time(self):
        cap, x = self.capacity["all"]
        return self.free_time * (1 + self.b * (x / cap)**self.power)



class Node:
    def __init__(self, id, center, signal, wait, edge_num, enter, off):
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

    def calculate_wait(self, fr, to):
        g, c = self.signal[(fr, to)]
        cap, x = self.center.edges[fr].capacity[to]
        return 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, x / cap) * g / c))






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
                v_t = (e_max - e) / p * 60
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
                v_t = (e_max - e) / p * 60
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
        #     m = self.t_cost[0] / self.t_cost[1]
        # else:
        #     m = 1
        print("Before calculate")
        print(s, k, l, m)
        if l == 0 or m == 0:
            return 0
        else:
            rou = l / m
            rou_s = rou / s
            p_0 = 0
            for i in range(s):
                p_0 += rou**i / math.factorial(i)
            if rou_s == 1:
                p_0 += (k - s + 1) * (rou**s) / math.factorial(s)
            else:
                p_0 += (rou**s) * (1 - rou_s**(k - s + 1))/ math.factorial(s) / (1 - rou_s)
            p_0 = 1 / p_0
            p_k = p_0 * (rou**k) / math.factorial(s) / (s ** (k - s))
            print("During calculate")
            print(rou, k, rou**k, math.factorial(s), s, k - s)
            l_e = l * (1 - p_k)
            if rou_s == 1:
                L_q = p_0 * (rou**s) * (k - s) * (k - s + 1) / 2 / math.factorial(s)
            else:
                L_q = p_0 * (rou**s) * rou_s * (1 - (rou_s**(k - s + 1)) - (1 - rou_s) * (k - s + 1) * (rou_s**(k - s)))
            print("After calculate")
            print(p_0, p_k, l_e, L_q)
            if l_e == 0:
                return 0
            else:
                return L_q / l_e


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