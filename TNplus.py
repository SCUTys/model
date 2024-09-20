import random


t = 1 #min
T = 30 #min
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

    def calculate_lost(self):
        sum_road = 0
        for edge in self.edges:
            sum_road += edge.capacity["all"] * edge.calculate_drive()
        for node in self.nodes:
            for i, is_wait in node.wait:
                sum_road += is_wait
        for charge_station in self.charge_stations:
            for i, time in charge_station.charge:
                sum_road += time
            for p, n in charge_station.pile:
                length = len(charge_station.queue[p])
                for i, time in charge_station.queue[p]:
                    sum_road += length * time
        return sum_road


    def dispatch(self, charging_vehicles, path_results):
        for v in charging_vehicles:
            v.charge = cs[random.randint(0, len(cs) - 1)]  ##这里没并算法就写个随机数吧
            path1 = path_results[(v.origin, v.charge)][0]
            path2 = path_results[(v.charge, v.destination)][0]
            if len(path1) > 1:
                path_1 = path1[random.randint(0, len(path1) - 1)].pop()
            else:
                path_1 = path1[0].pop()
            if len(path2) > 1:
                path_2 = path2[random.randint(0, len(path2) - 1)]
            else:
                path_2 = path2[0]
            v.path = path_1 + path_2
        return



class Vehicle:
    def __init__(self, id, center, origin, destination, distance, road, next_road, path, Emax, E, Ewait, Edrive, iswait, charge, index=0, charging = False):
        self.id = id
        self.center = center
        self.origin = origin
        self.destination = destination #Node id
        self.distance = distance
        self.road = road  #二元组，（O, D）
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
        print(road.id)
        drive_distance = road.calculate_drive() * rate
        print(drive_distance)
        if drive_distance < self.distance:
            self.distance -= drive_distance
            self.E -= drive_distance * self.Edrive
        else:
            self.E -= self.distance * self.Edrive
            self.distance = 0
            if self.check_destination():
                road = self.center.edges[self.road]
                next_road = self.center.edges[self.next_road]
                road.capacity[next_road.id][1] -= 1
                road.capacity["all"][1] -= 1
                self.center.vehicles.pop(self.id)
            if not self.check_charge():
                self.wait(road.id, self.center.edges[self.next_road].id, self.distance / drive_distance)
            else:
                self.enter_charge()


    def wait(self, fr, to, rate=1):
        road = self.center.edges[self.road]
        junction = self.center.nodes[road.destination]
        g, c= junction.signal[(fr, to)]
        cap, x = road.capacity[to]
        if self.is_wait == 0 and self.distance == 0:
            self.is_wait = 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, cap / x) * g / c))
            self.is_wait -= t * (1 - rate)
            junction.wait.append((self.id, self.is_wait))
        elif self.is_wait > 0:
            if self.is_wait <= t:
                self.is_wait = 0
                self.change_road()
                junction.wait.remove((self.id, self.is_wait))
            else:
                self.is_wait -= t * rate


    def change_road(self):
        road = self.center.edges[self.road]
        next_road = self.center.edges[self.next_road]
        road.capacity[next_road.id][1] -= 1
        road.capacity["all"][1] -= 1

        self.road = self.next_road
        if self.index < len(self.path):
            self.index += 1
        self.next_road = self.path[self.index]

        road = self.center.edges[self.road]
        next_road = self.center.edges[self.next_road]
        self.distance = road.length
        road.capacity["all"][1] -= 1
        road.capacity[next_road.id][1] += 1


    def check_charge(self):
        if self.distance == 0 and self.center.edges[self.road].destination == self.charge:
            return True
        else:
            return False


    def enter_charge(self):
        charge = self.center.charge_stations[self.charge[0]]
        self.charging = True
        for (i, a) in self.charge.dispatch:
            if self.id == i:
                charge.dispatch.remove((i, a))
                charge.q_length += 1
                charge.queue[self.charge[1]].append((self.id, 0))


    def leave_charge(self):
        self.charging = False
        self.charge = 0
        self.change_road()


    def check_destination(self):
        if self.distance == 0 and self.center.edges[self.road].destination == self.destination:
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
        return self.length / self.free_time / (1 + self.b * (x / cap)**self.power) * T



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
        #dispatch = {{v, atime}} 表示分配到该站点但仍未到达的车辆，按预计到达时间排序
        #charge = {power:{id, t}} 表示正在充电的车辆
        #queue = {power:{id, t}} 表示正在站内排队等候充电的队列
        #pile = ({power, num}) 表示站内充电桩功率与数量
        self.id = id
        self.center = center
        self.dispatch = dispatch
        self.charge = charge
        self.queue = queue
        self.capacity = capacity
        self.pile = pile


    def process(self):
        for p, n in self.pile.items():
            for id, time in self.charge[p]:
                if time > t:
                    time -= t
                else:
                    time -= time
                    v = self.center.vehicles[id]
                    self.charge[p].remove((v.id, time))
                    v.leave_charge()

            while len(self.charge[p]) < n and self.queue[p]:
                v_id = self.queue[p][0]
                e = self.center.vehicles[v_id].E
                e_max = self.center.vehicles[v_id].Emax
                self.charge[p].append((v_id, (e_max - e) / p))
                self.queue[p].pop(0)

            for i, t in self.queue.items():
                t += 1


    def check(self):
        charge_sum = sum(len(v) for v in self.charge.values())
        queue_sum = sum(len(v) for v in self.queue.values())
        if charge_sum + queue_sum + len(self.dispatch) < self.capacity:
            return True
        else:
            return False