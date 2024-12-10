# The `pymoo.algorithms.moo` module in the `pymoo` library includes several multi-objective optimization algorithms. Here are some of the algorithms available:
#
# - `NSGA2`: Non-dominated Sorting Genetic Algorithm II
# - `NSGA3`: Non-dominated Sorting Genetic Algorithm III
# - `SPEA2`: Strength Pareto Evolutionary Algorithm 2
# - `MOEAD`: Multi-Objective Evolutionary Algorithm based on Decomposition
# - `RVEA`: Reference Vector Guided Evolutionary Algorithm
# - `CMAES`: Covariance Matrix Adaptation Evolution Strategy for multi-objective optimization
# - `UNSGA3`: Unified NSGA-III
#
# These algorithms are designed to handle multi-objective optimization problems and can be used to find a set of Pareto-optimal solutions.


'''
调度目标：充电车辆是否充电、充电站的选择、充电功率的选择、充电车辆的路径规划（起点-充电站-终点）
调度状态：路网形状、每条路车数、扩展的话加上车辆位置（road, next_road, distance, iswait）
G, road.capacity['all'], vehicle.road, vehicle.next_road, vehicle.distance, vehicle.iswait
调度约束：电量要够到充电、充电站容量不能超、一车至多对应一桩（反之亦然）、分配时需保证充电站仍足够满足该车充电需求
调度模板算法：多目标优化算法（NSGA2、SPEA2等）
'''

import numpy as np
import simuplus
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultTermination
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from TNplus import ChargeStation, Vehicle, DispatchCenter, Edge, Node


cs_SF = [1, 5, 11, 13, 15, 20]
cs_EMA = [6, 10, 11, 17, 19, 22, 23, 25, 27, 29, 30, 33, 34, 38, 40, 42, 44, 47, 48, 49, 52, 57, 60, 63, 65, 69]
cs = cs_SF
T = 10


class ChargingStationProblem(Problem):
    '''
    调度目标：充电车辆是否充电、充电站的选择、充电功率的选择、充电车辆的路径规划（起点-充电站-终点）
    调度状态：路网形状、每条路车数、扩展的话加上车辆位置（road, next_road, distance, iswait）
    G, road.capacity['all'], vehicle.road, vehicle.next_road, vehicle.distance, vehicle.iswait
    调度约束：电量要够到充电、充电站容量不能超、一车至多对应一桩（反之亦然）、分配时需保证充电站仍足够满足该车充电需求
    调度模板算法：多目标优化算法（NSGA2、SPEA2等）
    '''
    def __init__(self, center, charge_vehicles, path_results, eps = 0, path_detail=None):
        super().__init__(n_var=len(cs) * len(charge_vehicles), n_obj=2, n_constr=1, xl=0, xu=1, type_var=int)
        self.center = center
        self.charge_vehicles = charge_vehicles #需充电车辆id集合
        self.path_results = path_results
        self.cs = cs
        self.eps = eps
        self.Graph = simuplus.get_graph()
        self.path_detail = path_detail if path_detail is not None else {}
        self.vehicle_start_end = {
            vehicle_id: (center.vehicles[vehicle_id].origin, center.vehicles[vehicle_id].destination) for vehicle_id in
            charge_vehicles}

    def _evaluate(self, x, out, *args, **kwargs): ##这里的x希望能传[[vehicle_id, path, cs_id, power]]
        cost_result = {}
        charge_cnt = {}
        dispatch_cnt = {}
        alpha = 0.75
        for [vehicle_id, path, cs_id, power] in x:
            cost_result[vehicle_id] = self.calculate_cost(vehicle_id, path, cs_id, power)
            charge_cnt[(cs_id, power)] = charge_cnt.get((cs_id, power), 0) + 1

        f1 = self.center.calculate_lost()
        f2 = sum([alpha * (cost[0] + cost[1] + cost[2] + cost[3]) + (1 - alpha) * cost[4] for cost in cost_result.values()])
        out["F"] = [f1, f2]

        # Example constraint: sum of variables should be less than or equal to a threshold
        g1 = []
        g2 = []
        for cost in cost_result.values():
            g1.append(self.eps - cost[4]) #能够到达充电站

        for c in self.cs:
            for vid, atime in self.center.charge_stations[c].dispatch.items():
                dispatch_cnt[(c, self.center.vehicles[vid].charge[1])] = dispatch_cnt.get((c, self.center.vehicles[vid].charge[1]), 0) + 1
            for power in self.center.charge_stations[c].pile.keys():
                g2.append(charge_cnt.get((c, power), 0) + dispatch_cnt.get((c, power), 0)
                          + sum(self.center.charge_stations[c].pile.values()) / len(self.center.charge_stations[c].pile)
                          - self.center.charge_stations[c].capacity / len(self.center.charge_stations[c].pile))

        out["G"] = [g1, g2]

    def calculate_cost(self, vehicle_id, path, cs_id, power):
        def calculate_wait(node_id, fr, to):
            g, c = self.center.nodes[node_id].signal[(fr, to)]
            cap, x = self.center.edges[fr].capacity[to]
            return 0.5 * c * ((1 - g / c) ** 2 / (1 - min(1, x / cap) * g / c))

        def calculate_drive(fr, to):
            cap, x = self.center.edges[(fr, to)].capacity["all"]
            return (self.center.edges[(fr, to)].length / self.center.edges[(fr, to)].free_time
                    / (1 + self.center.edges[(fr, to)].b * (x / cap) ** self.center.edges[(fr, to)].power))


        cost_drive_to_cs = cost_wait_to_cs = cost_drive = cost_wait = 0
        if path[0] != self.center.vehicles[vehicle_id].origin or path[-1] != self.center.vehicles[vehicle_id].destination or cs_id not in path:
            return [100000, 100000, 100000, 100000, 100000 * power]
        path_r = self.center.calculate_path(path)
        for index in range(len(path_r) - 1):
            if (path[index], path[index + 1]) not in self.Graph.edges:
                return [100000, 100000, 100000, 100000, 100000 * power]
            cost_drive += calculate_drive(path[index], path[index + 1])
            if path[index + 1] == cs_id:
                cost_drive_to_cs = cost_drive
                cost_wait_to_cs = cost_wait
            if index < len(path_r) - 1:
                cost_wait += calculate_wait(path[index + 1], path_r[index], path_r[index + 1])

        charge_s = self.center.charge_stations[cs_id]
        if charge_s.t_cost[power][1] > 0 and charge_s.t_cost[power][0] > 0:
            aver_charge = charge_s.t_cost[power][1] / charge_s.t_cost[power][0]
        else:
            aver_charge = 1
        cost_queue = charge_s.calculate_wait_cs(charge_s.pile[power],
                                                charge_s.capacity * charge_s.pile[power]/ sum(charge_s.pile.values()),
                                                charge_s.v_arrive / T,
                                                aver_charge)

        cost_charge = (self.center.vehicles[vehicle_id].Emax
                       - cost_drive * self.center.vehicles[vehicle_id].Edrive
                       - (cost_wait + cost_queue) * self.center.vehicles[vehicle_id].Ewait) / power

        return [cost_drive, cost_wait, cost_queue, cost_charge, cost_charge * power]