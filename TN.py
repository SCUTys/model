from operator import truediv

T = 0.25
t = 1/60

class Vehicle:
    def __init__(self, id, origin, destination, distance, road, next_road, path, Emax, Ewait, Edrive, iswait, charge, index=1):
        self.id = id
        self.origin = origin
        self.destination = destination #Node对象
        self.distance = distance
        self.road = road
        self.next_road = next_road
        self.path = path
        self.Emax = Emax  #电池最大容量
        self.Ewait = Ewait  #等待时单位时间消耗电量
        self.Edrive = Edrive  #行驶时单位时间消耗电量
        self.iswait = iswait
        self.charge = charge #ChargeStation对象
        self.index = index

    def drive(self, road):
        drive_distance = road.calculate_drive()
        if drive_distance <= self.distance:
            self.distance -= drive_distance
        else:
            self.distance = 0
            self.wait(road, road.id, self.next_road.id)
            self.iswait -= T * self.distance / drive_distance

    def wait(self, road, fr, to):
        self.check_charge()
        junction = road.destination
        g, c= junction.signal[(fr, to)]
        cap, x = road.capacity[to]
        if self.iswait == 0 and self.distance == 0:
            self.iswait = 0.5 * c * ((1 - g/c)**2 / (1 - min(1, cap/x)*g/c))
            junction.wait.append((self.id, self.iswait))
        elif self.iswait > 0:
            if self.iswait <= t:
                self.iswait = 0
                self.change_road()
                junction.wait.remove((self.id, self.iswait))
            else:
                self.iswait -= t


    def change_road(self):
        self.road.capacity[self.next_road.id][1] -= 1
        self.road.capacity["all"][1] -= 1
        self.road = self.next_road
        if self.index < len(self.path):
            self.index += 1
        self.next_road = self.path[self.index]
        self.distance = self.next_road.length
        self.road.capacity[self.next_road.id][1] += 1

    def check_charge(self):
        if self.distance == 0 and self.road.destination.id == self.charge.id:
            self.charge.queue.append(self.id, self.Ewait)
            self.charge.q_length += 1
            self.road.capacity[self.road.id][1] -= 1
            for (i, a) in self.charge.dispatch:
                if self.id == i:
                    self.charge.dispatch.remove((i, a))
        else:
            return




class Edge:
    def __init__(self, id, origin, destination, length, capacity, free_time, b, power):
        #capacity = {{to: cap, x}} 表示各方向车道组的容量(内为字典),to为下一条道路的id
        self.id = id
        self.origin = origin
        self.destination = destination #Node对象
        self.length = length
        self.capacity = capacity
        self.free_time = free_time
        self.b = b
        self.power = power

    def calculate_drive(self):
        cap, x = self.capacity
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
    def __init__(self, id, dispatch, vehicle, queue, q_length, capacity, pile):
        #dispatch = {{id, atime}} 表示分配到该站点但仍未到达的车辆，按预计到达时间排序
        #vehicle = {{id, wtime, ctime, ischarge}} 表示到达该站点且未充完电离开的车辆
        #queue = {id} 表示正在站内排队等候充电的队列
        #pile = {{power, num}} 表示站内充电桩功率与数量
        self.id = id
        self.dispatch = dispatch
        self.vehicle = vehicle
        self.queue = queue
        self.q_length = q_length
        self.capacity = capacity
        self.pile = pile

        def process(self):
            for (id, w, c, i) in self.vehicle:
                if w > t:
                    w = w - t
                elif t >= w > 0:
                    w = 0
                    i = True
                    self.q_length -= 1
                    c -= (t - w)
                elif w == 0 and c > t and i:
                    c -= t
                elif w == 0 and c <= t and i:
                    self.vehicle.remove((id, w, c, i))






class DispatchCenter:
    def __init__(self, vehicles, edges, nodes, charge_stations):
        self.vehicles = vehicles
        self.edges = edges
        self.nodes = nodes
        self.charge_stations = charge_stations

