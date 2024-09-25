import random


t = 1 #min
T = 1.5 #min
k = 1
cs = [10, 11, 16, 22]


class DispatchCenter:  #存储所有的数据，并且调度充电车辆
    def __init__(self, vehicles, edges, nodes, charge_stations):
        self.vehicles = vehicles
        self.edges = edges
        self.nodes = nodes
        self.charge_stations = charge_stations

    def vehicle(self):
        return self.vehicles

    def edge(self):
        return self.edges

    def node(self):
        return self.nodes

    def charge_station(self):
        return self.charge_stations

    def solve_tuple(self, tu, n):
        tu_list = list(tu)
        tu_list[1] += n
        return tuple(tu_list)

    def calculate_lost(self):
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


    def dispatch(self, charging_vehicles, path_results, t):
        for v in charging_vehicles:
            vehicle = self.vehicles[v]
            print(f"车辆{vehicle.id}原路径{vehicle.path},从{vehicle.origin}到{vehicle.destination}")
            c_index = cs[random.randint(0, len(cs) - 1)]     ##这里没并算法就写个随机数吧
            vehicle.charge = (c_index,  list(self.charge_stations[c_index].pile.keys())[0])
            print(f"车辆 {vehicle.id} 分配到充电站{c_index} ")
            if vehicle.origin != vehicle.charge[0]:
                path1 = path_results[(vehicle.origin, vehicle.charge[0])][0]
            else:
                path1 = []
            if vehicle.destination != vehicle.charge[0]:
                path2 = path_results[(vehicle.charge[0], vehicle.destination)][0]
            else:
                path2 = []
            if len(path1) >= 1:
                path11 = path1[random.randint(0, len(path1) - 1)]
                # path11.pop()
            else:
                path11 = []

            if len(path2) >= 1:
                path22 = path2[random.randint(0, len(path2) - 1)]
            else:
                path22 = []

            true_path2 = []
            true_path1 = []
            for i in range(0, len(path11) - 1):
                for edge in self.edges.values():
                    if edge.origin == path11[i] and edge.destination == path11[i + 1]:
                        true_path1.append(edge.id)
            for i in range(0, len(path22) - 1):
                for edge in self.edges.values():
                    if edge.origin == path22[i] and edge.destination == path22[i + 1]:
                        true_path2.append(edge.id)
            vehicle.path = true_path1 + true_path2
            print(f"重新分配的路径{vehicle.path}")
            if len(vehicle.path) > 0:
                vehicle.distance =self.edges[vehicle.path[0]].length
            vehicle.road =vehicle.path[0]
            self.edges[vehicle.road].capacity["all"] = self.solve_tuple(self.edges[vehicle.road].capacity["all"], 1)
            if len(vehicle.path) > 1:
                vehicle.next_road = vehicle.path[1]
            else:
                vehicle.next_road = -1
            self.edges[vehicle.road].capacity[vehicle.next_road] = self.solve_tuple(self.edges[vehicle.road].capacity[vehicle.next_road], 1)
            time = 0
            for i in true_path1:
                time += self.edges[i].calculate_drive()
            self.charge_stations[vehicle.charge[0]].dispatch[v] = t + time
            print(f'在车辆{vehicle.id}dispatch中道路{vehicle.road}总流量+1')
            vehicle.drive()
        return



class Vehicle:
    def __init__(self, id, center, origin, destination, distance, road, next_road, path, Emax, E, Ewait, Edrive, iswait, charge, index=0, charging = False):
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

    def drive(self, rate=1):
        road = self.center.edges[self.road]
        # print(road.id)
        if rate > 0 and rate < 1:
            drive_distance = road.calculate_drive() * rate
        else:
            drive_distance = road.calculate_drive()
        if drive_distance < self.distance:
            self.distance -= drive_distance
            self.E -= drive_distance * self.Edrive
            print(f"车辆 {self.id} 行驶了 {drive_distance} ")

        else:
            self.E -= self.distance * self.Edrive
            r = self.distance / drive_distance
            self.distance = 0

            if self.check_destination():
                print(f'车辆{self.id}已到达终点{self.destination},不再行驶')
                road = self.center.edges[self.road]
                road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], -1)
                road.capacity[-1] = self.center.solve_tuple(road.capacity[-1], -1)
                print(f'在车辆{self.id}destination中道路{self.road}总流量-1')
                self.road = -1
            elif self.check_charge():
                self.enter_charge()
            elif not self.check_charge() and self.next_road != -1:
                self.wait(self.road, self.next_road, r)




    def wait(self, fr, to, rate=1):
        road = self.center.edges[self.road]
        junction = self.center.nodes[road.destination]
        if fr == to:
            print(f"发现目标, id:{self.id}, path:{self.path}, road:{self.road}, next_road:{self.next_road}, {self.origin},{self.destination}")
        g, c= junction.signal[(fr, to)]
        cap, x = road.capacity[to]
        print(f"车辆 {self.id} 正在等待 ")
        if self.is_wait == 0 and self.distance == 0:
            if c != 0:
                self.is_wait = 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, x / cap) * g / c))
            else:
                self.change_road()
            if self.is_wait > t * (1 - rate):
                self.is_wait -= t * (1 - rate)
            else:
                self.is_wait = 0.001
            print(f"车辆 {self.id} 需要的等待时间为{self.is_wait} ")
            junction.wait.append((self.id, self.is_wait))
        elif self.is_wait > 0:
            if self.is_wait <= t:
                r = self.is_wait / t
                self.is_wait = 0
                for tu in junction.wait:
                    if tu[0] == self.id:
                        junction.wait.remove(tu)
                self.change_road(1 - r)
            else:
                self.is_wait -= t


    def change_road(self, rate = 0):
        if self.index < len(self.path) and self.next_road != -1:
            road = self.center.edges[self.road]
            next_road = self.center.edges[self.next_road]
            road.capacity[next_road.id] = self.center.solve_tuple(road.capacity[next_road.id], -1)
            # road.capacity[next_road.id][1] -= 1
            road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], -1)
            print(f'在车辆{self.id}change_road中道路{self.road}总流量-1')
            # road.capacity["all"][1] -= 1

            self.road = self.next_road

            if self.index < len(self.path) - 1:
                self.index += 1
                self.next_road = self.path[self.index]

                road = self.center.edges[self.road]
                next_road = self.center.edges[self.next_road]
                self.distance = road.length
                if self.distance > rate * self.center.edges[self.road].calculate_drive():
                    self.distance -= rate * self.center.edges[self.road].calculate_drive()
                else:
                    self.distance = 0.001
                road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], 1)
                print(f'在车辆{self.id}change_road中道路{self.road}总流量+1')
                # road.capacity["all"][1] += 1
                road.capacity[next_road.id] = self.center.solve_tuple(road.capacity[next_road.id], 1)
                # road.capacity[next_road.id][1] += 1
                print(f"车辆{self.id}转到{self.road}")
            elif self.index == len(self.path) - 1:
                self.index += 1
                self.next_road = -1
                road = self.center.edges[self.road]
                self.distance = road.length
                if self.distance > rate * self.center.edges[self.road].calculate_drive():
                    self.distance -= rate * self.center.edges[self.road].calculate_drive()
                else:
                    self.distance = 0.001
                road.capacity["all"] = self.center.solve_tuple(road.capacity["all"], 1)
                road.capacity[-1] = self.center.solve_tuple(road.capacity[-1], 1)
                print(f'在车辆{self.id}change_road中道路{self.road}总流量+1')
                print(f"车辆{self.id}转到{self.road}")


    def check_charge(self):
        # print(self.charge, end=' ')
        # print(self.id, end=' ')
        # print('新号别搞')
        if self.charge == {}:
            return False
        if self.center.edges[self.road].destination == self.charge[0]:
            return True
        else:
            return False


    def enter_charge(self):
        self.charging = True
        print(f"车辆{self.id}进入充电站{self.charge[0]}")
        del_list = []
        for (i, a) in self.center.charge_stations[self.charge[0]].dispatch.items():
            if self.id == i:
                del_list.append(i)

                self.center.charge_stations[self.charge[0]].queue[self.charge[1]].append((self.id, 0))
        for i in del_list:
            self.center.charge_stations[self.charge[0]].dispatch.pop(i)

    def leave_charge(self):
        print(f'车辆{self.id}在{self.charge}充完电离开')
        self.charging = False
        self.charge = {}
        self.change_road()


    def check_destination(self):
        if self.distance == 0 and self.center.edges[self.road].destination == self.destination and self.next_road == -1:
            return True
        else:
            return False




class Edge:
    def __init__(self, id, center, origin, destination, length, capacity, free_time, b, power):
        #capacity = {{to: cap, x}} 表示各方向车道组的容量(内为字典),to为下一条道路的id
        self.id = id
        self.center = center
        self.origin = origin
        self.destination = destination #Node对象
        self.length = length
        self.capacity = capacity
        self.free_time = free_time
        self.b = b
        self.power = power

    def calculate_drive(self):
        cap, x = self.capacity["all"]
        return self.length / self.free_time / (1 + self.b * (x / cap)**self.power) * t



class Node:
    def __init__(self, id, center, signal, wait, is_charge, edge_num, enter, off):
        #signal = {{fr, to}: {g, C}}, 表示每个方向信号灯所属绿灯时长与总周期时长
        #wait = {{vid, t}} 表示正在等候的车辆的合集
        self.id = id
        self.center = center
        self.signal = signal
        self.wait = wait
        self.is_charge = is_charge
        self.edge_num = edge_num
        self.enter = enter
        self.off = off




class ChargeStation:
    def __init__(self, id, center, dispatch, charge, queue, capacity, pile):
        #dispatch = {v: atime} 表示分配到该站点但仍未到达的车辆，按预计到达时间排序
        #charge = {power:{id, t}} 表示正在充电的车辆
        #queue = {power: id} 表示正在站内排队等候充电的队列
        #pile = (power:num) 表示站内充电桩功率与数量
        self.id = id    #这里假设和路口一致
        self.center = center
        self.dispatch = dispatch
        self.charge = charge
        self.queue = queue
        self.capacity = capacity
        self.pile = pile


    def process(self):
        for p, n in self.pile.items():
            # print(self.queue)
            # print(114514)
            if len(self.charge[p]) > 0:
                # print(self.charge[p])
                # print("我是神里绫华的狗")
                for index, tu in enumerate(self.charge[p]):
                    if tu[1] > t:
                        self.charge[p][index] = self.center.solve_tuple(tu, -t)
                        # print(self.charge[p])
                        # print("妮露宵宫的猫")
                    else:
                        v = self.center.vehicles[tu[0]]
                        self.charge[p].remove(tu)
                        v.leave_charge()


            while len(self.charge[p]) < n and self.queue[p]:
                v_id = self.queue[p][0][0]
                e = self.center.vehicles[v_id].E
                e_max = self.center.vehicles[v_id].Emax
                self.charge[p].append((v_id, (e_max - e) / p))
                self.queue[p].pop(0)




    def check(self):
        charge_sum = sum(len(v) for v in self.charge.values())
        queue_sum = sum(len(v) for v in self.queue.values())
        if charge_sum + queue_sum + len(self.dispatch) < self.capacity:
            return True
        else:
            return False