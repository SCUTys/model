


t = 1 #min
T = 30 #min


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

    def dispatch(self):
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
        drive_distance = road.calculate_drive() * rate
        if drive_distance < self.distance:
            self.distance -= drive_distance
            self.E -= drive_distance * self.Edrive
        else:
            self.E -= self.distance * self.Edrive
            self.distance = 0
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
                charge.queue[self.charge[1]].append(self.id)


    def leave_charge(self):
        self.charging = False
        self.change_road()







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
    def __init__(self, id, signal, wait, is_charge, edge_num, enter, off):
        #signal = {{fr, to}: {g, C}}, 表示每个方向信号灯所属绿灯时长与总周期时长
        #wait = {{vid, t}} 表示正在等候的车辆的合集
        self.id = id
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
        #queue = {power:{id}} 表示正在站内排队等候充电的队列
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

    def check(self):
        charge_sum = sum(len(v) for v in self.charge.values())
        queue_sum = sum(len(v) for v in self.queue.values())
        if charge_sum + queue_sum + len(self.dispatch) < self.capacity:
            return True
        else:
            return False