import numpy as np
import simuplus
import math
import random
import multiprocessing


cs_SF = [1, 5, 11, 13, 15, 20]
cs_EMA = [6, 10, 11, 17, 19, 22, 23, 25, 27, 29, 30, 33, 34, 38, 40, 42, 44, 47, 48, 49, 52, 57, 60, 63, 65, 69]
cs = cs_SF
T = 10

#dijkstra算法的优势在于速度快，并且道路路径短
class compareDJ:
    def __init__(self, v_charge, center, batch_size, path_result, cal_t):
        self.v_charge = v_charge
        self.center = center
        self.batch_size = batch_size
        self.path_result = path_result
        self.cal_t = cal_t
    def generate_result(self):
        print("Initializing")
        result = []
        for c in cs:
            self.path_result[(c, c)] = [[], 0]
        for i in self.v_charge:
            vehicle = self.center.vehicles[i]
            O = vehicle.origin
            D = vehicle.destination
            count = 100000
            best_cs = []
            for i in range(len(cs)):
                c = cs[i]
                if self.path_result[(O, c)][1] + self.path_result[(c, D)][1] < count:
                    count = self.path_result[(O, c)][1] + self.path_result[(c, D)][1]
                    best_cs = [i]
                elif self.path_result[(O, c)][1] + self.path_result[(c, D)][1] == count:
                    best_cs.append(i)
            cs_index = random.choice(best_cs)
            path1 = 0 if O == cs[cs_index] else 1
            path2 = 0 if D == cs[cs_index] else 1
            v_result = [path1, path2, cs_index + 1, 1]
            result += v_result
        for i in range(min(self.batch_size, len(self.v_charge))):
            print(f"车辆{i}的部分解为{result[4 * i: 4 * i + 4]}")
        return result

    def evaluate_individual(self, individual):
        road_counts = {(O, D): [0] * self.cal_t for O in range(1, simuplus.num_nodes + 1) for D in
                       range(1, simuplus.num_nodes + 1)}
        cs_qcounts = {c_s: [0] * self.cal_t for c_s in cs}
        cs_pcounts = {c_s: [0] * self.cal_t for c_s in cs}
        node_counts = {node: [0] * self.cal_t for node in range(1, simuplus.num_nodes + 1)}
        csum = 0
        psum = 0
        all_fit = 1
        for i in range(min(self.batch_size, len(self.v_charge))):
            valid = 1
            path1 = individual[4 * i]
            path2 = individual[4 * i + 1]
            c = individual[4 * i + 2]
            power = individual[4 * i + 3]
            vehicle_id = self.v_charge[i]
            vehicle = self.center.vehicles[vehicle_id]
            O = vehicle.origin
            D = vehicle.destination

            if O == cs[c - 1]:
                path1 = 0
            elif path1 == 0:
                path1 = 1

            if D == cs[c - 1]:
                path2 = 0
            elif path2 == 0:
                path2 = 1

            if O == cs[c - 1]:
                if path1 != 0:
                    csum += 1000
                    continue
                else:
                    path1_result = []
            elif path1 == 0:
                csum += 1000
                all_fit = 6
                continue
            else:
                path1_result = self.path_result[(O, cs[c - 1])][0][path1 - 1]

            if D == cs[c - 1]:
                if path2 != 0:
                    csum += 1500
                    all_fit = 2
                    continue
                else:
                    path2_result = []
            elif path2 == 0:
                csum += 1500
                all_fit = 3
                continue
            else:
                path2_result = self.path_result[(cs[c - 1], D)][0][path2 - 1]

            if path1_result and path2_result and path1_result[-1] == path2_result[0]:
                path = path1_result + path2_result[1:]
            else:
                path = path1_result + path2_result

            c = cs[c - 1]
            power_result = list(self.center.charge_stations[c].pile.keys())[power - 1]
            path_id = self.center.calculate_path(path)

            for id in path_id:
                psum += self.center.edges[id].calculate_time()

            charge_s = self.center.charge_stations[c]
            if charge_s.t_cost[power_result][0] != 0 and charge_s.t_cost[power_result][1] != 0:
                avg_charge = 1 / (charge_s.t_cost[power_result][0] / charge_s.t_cost[power_result][1])
            else:
                avg_charge = 1
            charge_t1 = charge_s.calculate_wait_cs(charge_s.pile[power_result],
                                                   charge_s.capacity * charge_s.pile[power_result] / sum(
                                                       charge_s.pile.values()),
                                                   charge_s.v_arrive[power_result] / T, avg_charge)
            Ecost = charge_t1 * vehicle.Ewait
            for idindex in range(len(path_id)):
                Ecost += self.center.edges[path_id[idindex]].calculate_time() * vehicle.Edrive
                if Ecost > vehicle.E:
                    valid = 0
                    break
                elif idindex < len(path_id) - 1:
                    Ecost += self.center.nodes[path[idindex + 1]].calculate_wait(path_id[idindex], path_id[idindex + 1])
                    if Ecost > vehicle.E:
                        valid = 0
                        break

            if valid == 0:
                all_fit = 4
                csum += 50000
                continue

            ssum = 0
            for index in range(0, len(path) - 1):
                if ssum >= self.cal_t:
                    break
                if path[index] != c:
                    node = self.center.nodes[path[index + 1]]
                    edge = self.center.edges[path_id[index]]
                    if index <= len(path_id) - 2:
                        for j in range(math.ceil(ssum), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                self.cal_t)):
                            road_counts[(path[index], path[index + 1])][j] += 1

                        for q in range(math.ceil(ssum + edge.calculate_time()), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                self.cal_t)):
                            node_counts[path[index + 1]][q] += 1
                    ssum += edge.calculate_time()
                else:
                    charge_t2 = (vehicle.E - ssum * vehicle.Edrive) / power_result
                    for k in range(math.ceil(ssum), min(math.floor(ssum + charge_t1), self.cal_t)):
                        cs_qcounts[c][k] += 1
                    for k in range(math.ceil(ssum + charge_t1),
                                   min(math.floor(ssum + charge_t1 + charge_t2), self.cal_t)):
                        cs_pcounts[c][k] += 1
                    ssum += charge_t1 + charge_t2

        for m in range(self.cal_t):
            for c_s in cs:
                c_station = self.center.charge_stations[c_s]
                charge_sum = sum(len(v) for v in c_station.charge.values())
                queue_sum = sum(len(v) for v in c_station.queue.values())
                if cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum > self.center.charge_stations[
                    c_s].capacity:
                    csum += 100 * (cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum -
                                   self.center.charge_stations[c_s].capacity)
                    all_fit = 5
                    break


        eval_s = 0
        for m in range(self.cal_t):
            for edge in self.center.edges.values():
                cap, x = edge.capacity["all"]
                eval_s += ((edge.capacity["all"][1] + road_counts[(edge.origin, edge.destination)][
                    m]) * edge.free_time * (1 + edge.b * (
                        (x + road_counts[(edge.origin, edge.destination)][m]) / cap) ** edge.power))
            for node in self.center.nodes.values():
                for p, is_wait in node.wait:
                    eval_s += is_wait + node_counts[node.id][m]
            for charge_station in self.center.charge_stations.values():
                if list(charge_station.charge.values()) != [[]]:
                    for ipair in charge_station.charge.values():
                        for i in ipair:
                            eval_s += i[1] + cs_pcounts[charge_station.id][m]
                for p, n in charge_station.pile.items():
                    length = len(charge_station.queue[p])
                    if length > 0:
                        for i, time in charge_station.queue[p]:
                            eval_s += (length + cs_qcounts[charge_station.id][m]) * time
        csum += eval_s
        return (csum + 0.5 * psum) / len(self.v_charge)

    def run(self):
        result = self.generate_result()
        print(f"compare result: {self.evaluate_individual(result)}")
        print("这B玩意到底咋收敛啊")
        return result





#进化算法的优势区间在于高负载场景下，将同样的OD分配到不同路径以减小拥堵
class NSGA2:
    def __init__(self, v_charge, center, batch_size, path_result, pop_size, n_gen, n_path1, n_path2, n_cs, n_power, eps = 0, crossover_prob=0.9, mutation_prob=0.3, cal_t=10):
        self.cal_t = cal_t
        self.v_charge = v_charge
        self.center = center
        self.path_result = path_result
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_path1 = n_path1
        self.n_path2 = n_path2
        self.n_cs = n_cs
        self.n_power = n_power
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eps = eps

    def initialize_population(self):
        print("Initializing")
        population = np.zeros((self.pop_size, 4 * self.batch_size), dtype=int)
        for i in range(self.batch_size):
            population[:, 4 * i] = np.random.randint(0, self.n_path1 + 1, self.pop_size)
            population[:, 4 * i + 1] = np.random.randint(0, self.n_path2 + 1, self.pop_size)
            population[:, 4 * i + 2] = np.random.randint(1, self.n_cs + 1, self.pop_size)
            population[:, 4 * i + 3] = np.random.randint(1, self.n_power + 1, self.pop_size)
        print("沈哥哥太强了")

        return population

    def generate_new_individuals(self, count):
        new_individuals = np.zeros((count, 4 * min(self.batch_size, len(self.v_charge))), dtype=int)
        for i in range(count):
            for j in range(self.batch_size):
                new_individuals[i, 4 * j] = np.random.randint(0, self.n_path1 + 1)
                new_individuals[i, 4 * j + 1] = np.random.randint(0, self.n_path2 + 1)
                new_individuals[i, 4 * j + 2] = np.random.randint(1, self.n_cs + 1)
                new_individuals[i, 4 * j + 3] = np.random.randint(1, self.n_power + 1)
        return new_individuals


    # def evaluate_individual(self, individual):
    #     road_counts = {(O, D): [0] * self.cal_t for O in range(1, simuplus.num_nodes + 1) for D in
    #                    range(1, simuplus.num_nodes + 1)}
    #     cs_qcounts = {c_s: [0] * self.cal_t for c_s in cs}
    #     cs_pcounts = {c_s: [0] * self.cal_t for c_s in cs}
    #     node_counts = {node: [0] * self.cal_t for node in range(1, simuplus.num_nodes + 1)}
    #     csum = 0
    #     psum = 0
    #     all_fit = 1
    #     for i in range(min(self.batch_size, len(self.v_charge))):
    #         valid = 1
    #         path1 = individual[4 * i]
    #         path2 = individual[4 * i + 1]
    #         c = individual[4 * i + 2]
    #         power = individual[4 * i + 3]
    #         vehicle_id = self.v_charge[i]
    #         vehicle = self.center.vehicles[vehicle_id]
    #         O = vehicle.origin
    #         D = vehicle.destination
    #
    #         if O == cs[c - 1]:
    #             path1 = 0
    #         elif path1 == 0:
    #             path1 = random.randint(1, self.n_path1)
    #
    #         if D == cs[c - 1]:
    #             path2 = 0
    #         elif path2 == 0:
    #             path2 = random.randint(1, self.n_path2)
    #
    #         if O == cs[c - 1]:
    #             if path1 != 0:
    #                 csum += 1000
    #                 continue
    #             else:
    #                 path1_result = []
    #         elif path1 == 0:
    #             csum += 1000
    #             all_fit = 6
    #             continue
    #         else:
    #             path1_result = self.path_result[(O, cs[c - 1])][0][path1 - 1]
    #
    #         if D == cs[c - 1]:
    #             if path2 != 0:
    #                 csum += 1500
    #                 all_fit = 2
    #                 continue
    #             else:
    #                 path2_result = []
    #         elif path2 == 0:
    #             csum += 1500
    #             all_fit = 3
    #             continue
    #         else:
    #             path2_result = self.path_result[(cs[c - 1], D)][0][path2 - 1]
    #
    #         if path1_result and path2_result and path1_result[-1] == path2_result[0]:
    #             path = path1_result + path2_result[1:]
    #         else:
    #             path = path1_result + path2_result
    #
    #         c = cs[c - 1]
    #         power_result = list(self.center.charge_stations[c].pile.keys())[power - 1]
    #         path_id = self.center.calculate_path(path)
    #
    #         for id in path_id:
    #             psum += self.center.edges[id].calculate_time()
    #
    #         charge_s = self.center.charge_stations[c]
    #         if charge_s.t_cost[power_result][0] != 0 and charge_s.t_cost[power_result][1] != 0:
    #             avg_charge = 1 / (charge_s.t_cost[power_result][0] / charge_s.t_cost[power_result][1])
    #         else:
    #             avg_charge = 1
    #         charge_t1 = charge_s.calculate_wait_cs(charge_s.pile[power_result],
    #                                                charge_s.capacity * charge_s.pile[power_result] / sum(
    #                                                    charge_s.pile.values()),
    #                                                charge_s.v_arrive[power_result] / T, avg_charge)
    #         Ecost = charge_t1 * vehicle.Ewait
    #         for idindex in range(len(path_id)):
    #             Ecost += self.center.edges[path_id[idindex]].calculate_time() * vehicle.Edrive
    #             if Ecost > vehicle.E:
    #                 valid = 0
    #                 break
    #             elif idindex < len(path_id) - 1:
    #                 Ecost += self.center.nodes[path[idindex + 1]].calculate_wait(path_id[idindex], path_id[idindex + 1])
    #                 if Ecost > vehicle.E:
    #                     valid = 0
    #                     break
    #
    #         if valid == 0:
    #             all_fit = 4
    #             csum += 50000
    #             continue
    #
    #         ssum = 0
    #         for index in range(0, len(path) - 1):
    #             if ssum >= self.cal_t:
    #                 break
    #             if path[index] != c:
    #                 node = self.center.nodes[path[index + 1]]
    #                 edge = self.center.edges[path_id[index]]
    #                 if index <= len(path_id) - 2:
    #                     for j in range(math.ceil(ssum), min(math.floor(
    #                             ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
    #                                                         self.cal_t)):
    #                         road_counts[(path[index], path[index + 1])][j] += 1
    #
    #                     for q in range(math.ceil(ssum + edge.calculate_time()), min(math.floor(
    #                             ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
    #                                                                                 self.cal_t)):
    #                         node_counts[path[index + 1]][q] += 1
    #                 ssum += edge.calculate_time()
    #             else:
    #                 charge_t2 = (vehicle.E - ssum * vehicle.Edrive) / power_result
    #                 for k in range(math.ceil(ssum), min(math.floor(ssum + charge_t1), self.cal_t)):
    #                     cs_qcounts[c][k] += 1
    #                 for k in range(math.ceil(ssum + charge_t1),
    #                                min(math.floor(ssum + charge_t1 + charge_t2), self.cal_t)):
    #                     cs_pcounts[c][k] += 1
    #                 ssum += charge_t1 + charge_t2
    #
    #         # print(f"在此解中，车辆{self.v_charge[i]}的起点为{O}， 终点为{D}，节点路径为{path}，道路路径为{path_id}")
    #
    #     for m in range(self.cal_t):
    #         for c_s in cs:
    #             c_station = self.center.charge_stations[c_s]
    #             charge_sum = sum(len(v) for v in c_station.charge.values())
    #             queue_sum = sum(len(v) for v in c_station.queue.values())
    #             if cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum > self.center.charge_stations[
    #                 c_s].capacity:
    #                 csum += 100 * (cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum -
    #                                self.center.charge_stations[c_s].capacity)
    #                 all_fit = 5
    #                 break
    #
    #     eval_s = 0
    #     for m in range(self.cal_t):
    #         for edge in self.center.edges.values():
    #             cap, x = edge.capacity["all"]
    #             eval_s += ((edge.capacity["all"][1] + road_counts[(edge.origin, edge.destination)][
    #                 m]) * edge.free_time * (1 + edge.b * (
    #                         (x + road_counts[(edge.origin, edge.destination)][m]) / cap) ** edge.power))
    #         for node in self.center.nodes.values():
    #             for p, is_wait in node.wait:
    #                 eval_s += is_wait + node_counts[node.id][m]
    #         for charge_station in self.center.charge_stations.values():
    #             if list(charge_station.charge.values()) != [[]]:
    #                 for ipair in charge_station.charge.values():
    #                     for i in ipair:
    #                         eval_s += i[1] + cs_pcounts[charge_station.id][m]
    #             for p, n in charge_station.pile.items():
    #                 length = len(charge_station.queue[p])
    #                 if length > 0:
    #                     for i, time in charge_station.queue[p]:
    #                         eval_s += (length + cs_qcounts[charge_station.id][m]) * time
    #     csum += eval_s
    #     return (csum + 0.5 * psum) / len(self.v_charge)
    #
    # def evaluate(self, population):
    #     with multiprocessing.Pool() as pool:
    #         eval = pool.map(self.evaluate_individual, population)
    #     return eval

    def evaluate(self, population):
        n_individual = 0
        eval = []
        for individual in population:
            print("evaluate中")
            n_individual += 1
            road_counts = {(O, D): [0] * self.cal_t for O in range(1, simuplus.num_nodes + 1) for D in
                           range(1, simuplus.num_nodes + 1)}
            cs_qcounts = {c_s: [0] * self.cal_t for c_s in cs}
            cs_pcounts = {c_s: [0] * self.cal_t for c_s in cs}
            node_counts = {node: [0] * self.cal_t for node in range(1, simuplus.num_nodes + 1)}
            csum = 0
            psum = 0
            all_fit = 1
            for i in range(min(self.batch_size, len(self.v_charge))):
                valid = 1
                path1 = individual[4 * i]
                path2 = individual[4 * i + 1]
                c = individual[4 * i + 2]
                power = individual[4 * i + 3]
                # print(f"i {i}, len{len(self.v_charge), batch_size}")
                vehicle_id = self.v_charge[i]
                vehicle = self.center.vehicles[vehicle_id]
                O = vehicle.origin
                D = vehicle.destination

                if O == cs[c - 1]:
                    path1 = 0
                elif path1 == 0:
                    path1 = random.randint(1, self.n_path1)

                if D == cs[c - 1]:
                    path2 = 0
                elif path2 == 0:
                    path2 = random.randint(1, self.n_path2)

                if O == cs[c - 1]:
                    if path1 != 0:
                        csum += 1000
                        continue
                    else:
                        path1_result = []
                elif path1 == 0:
                    csum += 1000
                    all_fit = 6
                    continue
                else:
                    path1_result = self.path_result[(O, cs[c - 1])][0][path1 - 1]


                if D == cs[c - 1]:
                    if path2 != 0:
                        csum += 1500
                        all_fit = 2
                        continue
                    else:
                        path2_result = []
                elif path2 == 0:
                    csum += 1500
                    all_fit = 3
                    continue
                else:
                    path2_result = self.path_result[(cs[c - 1], D)][0][path2 - 1]


                if path1_result and path2_result and path1_result[-1] == path2_result[0]:
                    path = path1_result + path2_result[1:]
                else:
                    path = path1_result + path2_result

                c = cs[c - 1]
                power_result = list(self.center.charge_stations[c].pile.keys())[power - 1]
                path_id = self.center.calculate_path(path)

                for id in path_id:
                    psum += self.center.edges[id].calculate_time()

                charge_s = self.center.charge_stations[c]
                if charge_s.t_cost[power_result][0] != 0 and charge_s.t_cost[power_result][1] != 0:
                    avg_charge = 1 / (charge_s.t_cost[power_result][0] / charge_s.t_cost[power_result][1])
                else:
                    avg_charge = 1
                charge_t1 = charge_s.calculate_wait_cs(charge_s.pile[power_result],
                                                       charge_s.capacity * charge_s.pile[power_result] / sum(
                                                           charge_s.pile.values()),
                                                       charge_s.v_arrive[power_result] / T, avg_charge)
                Ecost = charge_t1 * vehicle.Ewait
                for idindex in range(len(path_id)):
                    Ecost += self.center.edges[path_id[idindex]].calculate_time() * vehicle.Edrive
                    if Ecost > vehicle.E:
                        valid = 0
                        break
                    elif idindex < len(path_id) - 1:
                        Ecost += self.center.nodes[path[idindex + 1]].calculate_wait(path_id[idindex],
                                                                                     path_id[idindex + 1])
                        if Ecost > vehicle.E:
                            valid = 0
                            break

                if valid == 0:
                    all_fit = 4
                    csum += 50000
                    continue

                ssum = 0
                for index in range(0, len(path) - 1):
                    if ssum >= self.cal_t:
                        break
                    if path[index] != c:

                        node = self.center.nodes[path[index + 1]]
                        edge = self.center.edges[path_id[index]]
                        if index <= len(path_id) - 2:
                            for j in range(math.ceil(ssum), min(math.floor(
                                    ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                                                self.cal_t)):
                                road_counts[(path[index], path[index + 1])][j] += 1

                            for q in range(math.ceil(ssum + edge.calculate_time()), min(math.floor(
                                    ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                                                                        self.cal_t)):
                                node_counts[path[index + 1]][q] += 1
                        ssum += edge.calculate_time()
                    else:
                        charge_t2 = (vehicle.E - ssum * vehicle.Edrive) / power_result
                        for k in range(math.ceil(ssum), min(math.floor(ssum + charge_t1), self.cal_t)):
                            cs_qcounts[c][k] += 1
                        for k in range(math.ceil(ssum + charge_t1),
                                       min(math.floor(ssum + charge_t1 + charge_t2), self.cal_t)):
                            cs_pcounts[c][k] += 1
                        ssum += charge_t1 + charge_t2

                print(f"在此解{n_individual}中，车辆{self.v_charge[i]}的起点为{O}， 终点为{D}，充电站节点号为{c}，节点路径为{path}，道路路径为{path_id}")


            for m in range(self.cal_t):
                for c_s in cs:
                    c_station = self.center.charge_stations[c_s]
                    charge_sum = sum(len(v) for v in c_station.charge.values())
                    queue_sum = sum(len(v) for v in c_station.queue.values())
                    if cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum > self.center.charge_stations[
                        c_s].capacity:
                        csum += 100 * (cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum - self.center.charge_stations[
                        c_s].capacity)
                        all_fit = 5
                        break



            eval_s = 0
            for m in range(self.cal_t):
                for edge in self.center.edges.values():
                    cap, x = edge.capacity["all"]
                    eval_s += ((edge.capacity["all"][1] + road_counts[(edge.origin, edge.destination)][m])
                               * edge.free_time
                               * (1 + edge.b * ((x + road_counts[(edge.origin, edge.destination)][m])/ cap)**edge.power))
                for node in self.center.nodes.values():
                    for p, is_wait in node.wait:
                        eval_s += is_wait + node_counts[node.id][m]
                for charge_station in self.center.charge_stations.values():
                    if list(charge_station.charge.values()) != [[]]:
                        for ipair in charge_station.charge.values():
                            for i in ipair:
                                eval_s += i[1] + cs_pcounts[charge_station.id][m]
                    for p, n in charge_station.pile.items():
                        length = len(charge_station.queue[p])
                        if length > 0:
                            for i, time in charge_station.queue[p]:
                                eval_s += (length + cs_qcounts[charge_station.id][m]) * time
            csum += eval_s
            eval.append((csum + 0.5 * psum) / len(self.v_charge))
            print(eval[-1])
        print(eval)
        return eval

    def non_dominated_sorting(self, objectives):
        num_individuals = len(objectives)
        domination_count = np.zeros(num_individuals)
        dominated_solutions = [[] for _ in range(num_individuals)]
        rank = np.zeros(num_individuals)

        for i in range(num_individuals):
            for j in range(num_individuals):
                if i != j:
                    if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                        dominated_solutions[i].append(j)
                    elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                        domination_count[i] += 1
            if domination_count[i] == 0:
                rank[i] = 0

        fronts = [[]]
        for i in range(num_individuals):
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = current_front + 1
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(self, objectives, front):
        distance = np.zeros(len(front))
        for m in range(objectives.shape[1]):
            sorted_indices = np.argsort(objectives[front, m])
            distance[sorted_indices[0]] = distance[sorted_indices[-1]] = np.inf
            for k in range(1, len(front) - 1):
                distance[sorted_indices[k]] += (objectives[front[sorted_indices[k + 1]], m] - objectives[front[sorted_indices[k - 1]], m])
        return distance

    def selection(self, population, fronts, objectives):
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) > self.pop_size:
                distances = self.crowding_distance(objectives, front)
                sorted_indices = np.argsort(distances)[::-1]
                for i in sorted_indices:
                    if len(new_population) < self.pop_size:
                        new_population.append(population[front[i]])
            else:
                new_population.extend(population[front])

        # Sort the population based on objective values
        sorted_indices = np.argsort(objectives)
        worst_indices = sorted_indices[-int(0.1 * self.pop_size):]  # Replace the worst 10%

        # Pre-generate new individuals for diversity maintenance
        diversity_maintenance_count = len(worst_indices)
        new_individuals = self.generate_new_individuals(diversity_maintenance_count)

        # Replace the worst individuals with new individuals
        for i, index in enumerate(worst_indices):
            new_population[index] = new_individuals[i]

        return np.array(new_population)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutation(self, individual):
        for i in range(min(len(self.v_charge), self.batch_size)):
            O = self.center.vehicles[self.v_charge[i]].origin
            D = self.center.vehicles[self.v_charge[i]].destination
            c = individual[4 * i + 2]


            if np.random.rand() < self.mutation_prob:
                path1 = individual[4 * i]
                path2 = individual[4 * i + 1]

                if O == cs[c - 1]:
                    path1 = 0
                elif path1 == 0:
                    path1 = random.randint(1, self.n_path1)

                if D == cs[c - 1]:
                    path2 = 0
                elif path2 == 0:
                    path2 = random.randint(1, self.n_path2)

                individual[4 * i] = path1
                individual[4 * i + 1] = path2
                individual[4 * i + 2] = np.random.randint(1, self.n_cs + 1)
                individual[4 * i + 3] = np.random.randint(1, self.n_power + 1)
        return individual

    def run(self):
        print("running")
        population = self.initialize_population()
        best_solution = None
        best_objective_value = float('inf')
        best_insert_index = 0

        for gen in range(self.n_gen):
            print(f"Generation {gen + 1}/{self.n_gen}")
            objectives = self.evaluate(population)
            print("Objective values: ", objectives)
            print("为啥全缩到45左右了？明明23~45都可能有值啊")
            fronts = self.non_dominated_sorting(objectives)
            selected_population = self.selection(population, fronts, objectives)
            new_population = []
            while len(new_population) < self.pop_size:
                parents = selected_population[np.random.choice(len(selected_population), 2, replace=False)]
                child1, child2 = self.crossover(parents[0], parents[1])
                new_population.append(self.mutation(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutation(child2))
            population = np.array(new_population)

            # Update the best solution in each generation
            current_best_index = np.argmin(objectives)
            current_best_value = objectives[current_best_index]
            if current_best_value < best_objective_value:
                best_objective_value = current_best_value
                best_solution = population[current_best_index]
            random_index1 = np.random.randint(0, self.pop_size * 0.25)
            random_index2 = np.random.randint(self.pop_size * 0.25, self.pop_size * 0.5)
            if gen % 2 == 1:
                population[random_index1] = population[random_index2] = best_solution
            print(f"目前的最优解值：{best_objective_value}")
            print(f"当前批次最优解值：{current_best_value}")
            print(f"当前批次平均值： {np.mean(objectives)}")

            diversity = np.mean(np.std(population, axis=0))
            self.crossover_prob = 0.9 - 0.5 * (gen / self.n_gen) * (1 - diversity)
            self.mutation_prob = 0.3 + 0.5 * (gen / self.n_gen) * diversity

        print(f"Best solution: {best_solution}")
        return best_solution




class SPEA2:
    def __init__(self, v_charge, center, batch_size, path_result, pop_size, n_gen, n_path1, n_path2, n_cs, n_power, eps=0, crossover_prob=0.9, mutation_prob=0.3, cal_t=10):
        self.cal_t = cal_t
        self.v_charge = v_charge
        self.center = center
        self.path_result = path_result
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_path1 = n_path1
        self.n_path2 = n_path2
        self.n_cs = n_cs
        self.n_power = n_power
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eps = eps

    def initialize_population(self):
        print("Initializing")
        population = np.zeros((self.pop_size, 4 * self.batch_size), dtype=int)
        for i in range(self.batch_size):
            population[:, 4 * i] = np.random.randint(0, self.n_path1 + 1, self.pop_size)
            population[:, 4 * i + 1] = np.random.randint(0, self.n_path2 + 1, self.pop_size)
            population[:, 4 * i + 2] = np.random.randint(1, self.n_cs + 1, self.pop_size)
            population[:, 4 * i + 3] = np.random.randint(1, self.n_power + 1, self.pop_size)
        return population

    def generate_new_individuals(self, count):
        new_individuals = np.zeros((count, 4 * self.batch_size), dtype=int)
        for i in range(count):
            for j in range(self.batch_size):
                new_individuals[i, 4 * j] = np.random.randint(0, self.n_path1 + 1)
                new_individuals[i, 4 * j + 1] = np.random.randint(0, self.n_path2 + 1)
                new_individuals[i, 4 * j + 2] = np.random.randint(1, self.n_cs + 1)
                new_individuals[i, 4 * j + 3] = np.random.randint(1, self.n_power + 1)
        return new_individuals

    def evaluate_individual(self, individual):
        road_counts = {(O, D): [0] * self.cal_t for O in range(1, simuplus.num_nodes + 1) for D in
                       range(1, simuplus.num_nodes + 1)}
        cs_qcounts = {c_s: [0] * self.cal_t for c_s in cs}
        cs_pcounts = {c_s: [0] * self.cal_t for c_s in cs}
        node_counts = {node: [0] * self.cal_t for node in range(1, simuplus.num_nodes + 1)}
        csum = 0
        psum = 0
        all_fit = 1
        for i in range(min(self.batch_size, len(self.v_charge))):
            valid = 1
            path1 = individual[4 * i]
            path2 = individual[4 * i + 1]
            c = individual[4 * i + 2]
            power = individual[4 * i + 3]
            vehicle_id = self.v_charge[i]
            vehicle = self.center.vehicles[vehicle_id]
            O = vehicle.origin
            D = vehicle.destination

            if O == cs[c - 1]:
                path1 = 0
            elif path1 == 0:
                path1 = random.randint(1, self.n_path1)

            if D == cs[c - 1]:
                path2 = 0
            elif path2 == 0:
                path2 = random.randint(1, self.n_path2)

            if O == cs[c - 1]:
                if path1 != 0:
                    csum += 1000
                    continue
                else:
                    path1_result = []
            elif path1 == 0:
                csum += 1000
                all_fit = 6
                continue
            else:
                path1_result = self.path_result[(O, cs[c - 1])][0][path1 - 1]

            if D == cs[c - 1]:
                if path2 != 0:
                    csum += 1500
                    all_fit = 2
                    continue
                else:
                    path2_result = []
            elif path2 == 0:
                csum += 1500
                all_fit = 3
                continue
            else:
                path2_result = self.path_result[(cs[c - 1], D)][0][path2 - 1]

            if path1_result and path2_result and path1_result[-1] == path2_result[0]:
                path = path1_result + path2_result[1:]
            else:
                path = path1_result + path2_result

            c = cs[c - 1]
            power_result = list(self.center.charge_stations[c].pile.keys())[power - 1]
            path_id = self.center.calculate_path(path)

            for id in path_id:
                psum += self.center.edges[id].calculate_time()

            charge_s = self.center.charge_stations[c]
            if charge_s.t_cost[power_result][0] != 0 and charge_s.t_cost[power_result][1] != 0:
                avg_charge = 1 / (charge_s.t_cost[power_result][0] / charge_s.t_cost[power_result][1])
            else:
                avg_charge = 1
            charge_t1 = charge_s.calculate_wait_cs(charge_s.pile[power_result],
                                                   charge_s.capacity * charge_s.pile[power_result] / sum(
                                                       charge_s.pile.values()),
                                                   charge_s.v_arrive[power_result] / T, avg_charge)
            Ecost = charge_t1 * vehicle.Ewait
            for idindex in range(len(path_id)):
                Ecost += self.center.edges[path_id[idindex]].calculate_time() * vehicle.Edrive
                if Ecost > vehicle.E:
                    valid = 0
                    break
                elif idindex < len(path_id) - 1:
                    Ecost += self.center.nodes[path[idindex + 1]].calculate_wait(path_id[idindex], path_id[idindex + 1])
                    if Ecost > vehicle.E:
                        valid = 0
                        break

            if valid == 0:
                all_fit = 4
                csum += 50000
                continue

            ssum = 0
            for index in range(0, len(path) - 1):
                if ssum >= self.cal_t:
                    break
                if path[index] != c:
                    node = self.center.nodes[path[index + 1]]
                    edge = self.center.edges[path_id[index]]
                    if index <= len(path_id) - 2:
                        for j in range(math.ceil(ssum), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                self.cal_t)):
                            road_counts[(path[index], path[index + 1])][j] += 1

                        for q in range(math.ceil(ssum + edge.calculate_time()), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                self.cal_t)):
                            node_counts[path[index + 1]][q] += 1
                    ssum += edge.calculate_time()
                else:
                    charge_t2 = (vehicle.E - ssum * vehicle.Edrive) / power_result
                    for k in range(math.ceil(ssum), min(math.floor(ssum + charge_t1), self.cal_t)):
                        cs_qcounts[c][k] += 1
                    for k in range(math.ceil(ssum + charge_t1),
                                   min(math.floor(ssum + charge_t1 + charge_t2), self.cal_t)):
                        cs_pcounts[c][k] += 1
                    ssum += charge_t1 + charge_t2

        for m in range(self.cal_t):
            for c_s in cs:
                c_station = self.center.charge_stations[c_s]
                charge_sum = sum(len(v) for v in c_station.charge.values())
                queue_sum = sum(len(v) for v in c_station.queue.values())
                if cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum > self.center.charge_stations[
                    c_s].capacity:
                    csum += 100 * (cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum -
                                   self.center.charge_stations[c_s].capacity)
                    all_fit = 5
                    break

        eval_s = 0
        for m in range(self.cal_t):
            for edge in self.center.edges.values():
                cap, x = edge.capacity["all"]
                eval_s += ((edge.capacity["all"][1] + road_counts[(edge.origin, edge.destination)][
                    m]) * edge.free_time * (1 + edge.b * (
                        (x + road_counts[(edge.origin, edge.destination)][m]) / cap) ** edge.power))
            for node in self.center.nodes.values():
                for p, is_wait in node.wait:
                    eval_s += is_wait + node_counts[node.id][m]
            for charge_station in self.center.charge_stations.values():
                if list(charge_station.charge.values()) != [[]]:
                    for ipair in charge_station.charge.values():
                        for i in ipair:
                            eval_s += i[1] + cs_pcounts[charge_station.id][m]
                for p, n in charge_station.pile.items():
                    length = len(charge_station.queue[p])
                    if length > 0:
                        for i, time in charge_station.queue[p]:
                            eval_s += (length + cs_qcounts[charge_station.id][m]) * time
        csum += eval_s
        return (csum + 0.5 * psum) / len(self.v_charge)

    def evaluate(self, population):
        with multiprocessing.Pool() as pool:
            eval = pool.map(self.evaluate_individual, population)
        return eval

    def fitness_assignment(self, population, objectives):
        num_individuals = len(population)
        strength = np.zeros(num_individuals)
        raw_fitness = np.zeros(num_individuals)
        for i in range(num_individuals):
            for j in range(num_individuals):
                if i != j and np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    strength[i] += 1
        for i in range(num_individuals):
            for j in range(num_individuals):
                if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    raw_fitness[i] += strength[j]
        return raw_fitness

    def environmental_selection(self, population, objectives, k):
        fitness = self.fitness_assignment(population, objectives)
        selected_indices = np.argsort(fitness)[:k]
        return population[selected_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutation(self, individual):
        for i in range(min(len(self.v_charge), self.batch_size)):
            O = self.center.vehicles[self.v_charge[i]].origin
            D = self.center.vehicles[self.v_charge[i]].destination
            c = individual[4 * i + 2]

            if np.random.rand() < self.mutation_prob:
                path1 = individual[4 * i]
                path2 = individual[4 * i + 1]

                if O == cs[c - 1]:
                    path1 = 0
                elif path1 == 0:
                    path1 = random.randint(1, self.n_path1)

                if D == cs[c - 1]:
                    path2 = 0
                elif path2 == 0:
                    path2 = random.randint(1, self.n_path2)

                individual[4 * i] = path1
                individual[4 * i + 1] = path2
                individual[4 * i + 2] = np.random.randint(1, self.n_cs + 1)
                individual[4 * i + 3] = np.random.randint(1, self.n_power + 1)
        return individual

    def run(self):
        print("running")
        population = self.initialize_population()
        archive = np.empty((0, population.shape[1]), dtype=int)
        best_solution = None
        best_objective_value = float('inf')

        for gen in range(self.n_gen):
            print(f"Generation {gen + 1}/{self.n_gen}")
            objectives = self.evaluate(population)
            combined_population = np.vstack((population, archive))
            combined_objectives = np.hstack((objectives, self.evaluate(archive)))
            archive = self.environmental_selection(combined_population, combined_objectives, self.pop_size)

            new_population = []
            while len(new_population) < self.pop_size:
                parents = archive[np.random.choice(len(archive), 2, replace=False)]
                child1, child2 = self.crossover(parents[0], parents[1])
                new_population.append(self.mutation(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutation(child2))
            population = np.array(new_population)

            current_best_index = np.argmin(objectives)
            current_best_value = objectives[current_best_index]
            if current_best_value < best_objective_value:
                best_objective_value = current_best_value
                best_solution = population[current_best_index]
            print(f"目前的最优解值：{best_objective_value}")
            print(f"当前批次最优解值：{current_best_value}")
            print(f"当前批次平均值： {np.mean(objectives)}")

        print(f"Best solution: {best_solution}")
        return best_solution
    


class MOPSO:
    def __init__(self, v_charge, center, batch_size, path_result, pop_size, n_gen, n_path1, n_path2, n_cs, n_power, eps=0, w=0.5, c1=1.5, c2=1.5, cal_t=10):
        self.cal_t = cal_t
        self.v_charge = v_charge
        self.center = center
        self.path_result = path_result
        self.batch_size = batch_size
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.n_path1 = n_path1
        self.n_path2 = n_path2
        self.n_cs = n_cs
        self.n_power = n_power
        self.eps = eps
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.population = self.initialize_population()
        self.velocities = np.zeros_like(self.population)
        self.pbest = self.population.copy()
        self.gbest = self.population[np.argmin(self.evaluate(self.population))]

    def initialize_population(self):
        population = np.zeros((self.pop_size, 4 * self.batch_size), dtype=int)
        for i in range(self.batch_size):
            population[:, 4 * i] = np.random.randint(0, self.n_path1 + 1, self.pop_size)
            population[:, 4 * i + 1] = np.random.randint(0, self.n_path2 + 1, self.pop_size)
            population[:, 4 * i + 2] = np.random.randint(1, self.n_cs + 1, self.pop_size)
            population[:, 4 * i + 3] = np.random.randint(1, self.n_power + 1, self.pop_size)
        return population

    def evaluate_individual(self, individual):
        road_counts = {(O, D): [0] * self.cal_t for O in range(1, simuplus.num_nodes + 1) for D in
                       range(1, simuplus.num_nodes + 1)}
        cs_qcounts = {c_s: [0] * self.cal_t for c_s in cs}
        cs_pcounts = {c_s: [0] * self.cal_t for c_s in cs}
        node_counts = {node: [0] * self.cal_t for node in range(1, simuplus.num_nodes + 1)}
        csum = 0
        psum = 0
        all_fit = 1
        for i in range(min(self.batch_size, len(self.v_charge))):
            valid = 1
            path1 = individual[4 * i]
            path2 = individual[4 * i + 1]
            c = individual[4 * i + 2]
            power = individual[4 * i + 3]
            vehicle_id = self.v_charge[i]
            vehicle = self.center.vehicles[vehicle_id]
            O = vehicle.origin
            D = vehicle.destination

            if O == cs[c - 1]:
                path1 = 0
            elif path1 == 0:
                path1 = random.randint(1, self.n_path1)

            if D == cs[c - 1]:
                path2 = 0
            elif path2 == 0:
                path2 = random.randint(1, self.n_path2)

            if O == cs[c - 1]:
                if path1 != 0:
                    csum += 1000
                    continue
                else:
                    path1_result = []
            elif path1 == 0:
                csum += 1000
                all_fit = 6
                continue
            else:
                path1_result = self.path_result[(O, cs[c - 1])][0][path1 - 1]

            if D == cs[c - 1]:
                if path2 != 0:
                    csum += 1500
                    all_fit = 2
                    continue
                else:
                    path2_result = []
            elif path2 == 0:
                csum += 1500
                all_fit = 3
                continue
            else:
                path2_result = self.path_result[(cs[c - 1], D)][0][path2 - 1]

            if path1_result and path2_result and path1_result[-1] == path2_result[0]:
                path = path1_result + path2_result[1:]
            else:
                path = path1_result + path2_result

            c = cs[c - 1]
            power_result = list(self.center.charge_stations[c].pile.keys())[power - 1]
            path_id = self.center.calculate_path(path)

            for id in path_id:
                psum += self.center.edges[id].calculate_time()

            charge_s = self.center.charge_stations[c]
            if charge_s.t_cost[power_result][0] != 0 and charge_s.t_cost[power_result][1] != 0:
                avg_charge = 1 / (charge_s.t_cost[power_result][0] / charge_s.t_cost[power_result][1])
            else:
                avg_charge = 1
            charge_t1 = charge_s.calculate_wait_cs(charge_s.pile[power_result],
                                                   charge_s.capacity * charge_s.pile[power_result] / sum(
                                                       charge_s.pile.values()),
                                                   charge_s.v_arrive[power_result] / T, avg_charge)
            Ecost = charge_t1 * vehicle.Ewait
            for idindex in range(len(path_id)):
                Ecost += self.center.edges[path_id[idindex]].calculate_time() * vehicle.Edrive
                if Ecost > vehicle.E:
                    valid = 0
                    break
                elif idindex < len(path_id) - 1:
                    Ecost += self.center.nodes[path[idindex + 1]].calculate_wait(path_id[idindex], path_id[idindex + 1])
                    if Ecost > vehicle.E:
                        valid = 0
                        break

            if valid == 0:
                all_fit = 4
                csum += 50000
                continue

            ssum = 0
            for index in range(0, len(path) - 1):
                if ssum >= self.cal_t:
                    break
                if path[index] != c:
                    node = self.center.nodes[path[index + 1]]
                    edge = self.center.edges[path_id[index]]
                    if index <= len(path_id) - 2:
                        for j in range(math.ceil(ssum), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                self.cal_t)):
                            road_counts[(path[index], path[index + 1])][j] += 1

                        for q in range(math.ceil(ssum + edge.calculate_time()), min(math.floor(
                                ssum + edge.calculate_time() + node.calculate_wait(path_id[index], path_id[index + 1])),
                                self.cal_t)):
                            node_counts[path[index + 1]][q] += 1
                    ssum += edge.calculate_time()
                else:
                    charge_t2 = (vehicle.E - ssum * vehicle.Edrive) / power_result
                    for k in range(math.ceil(ssum), min(math.floor(ssum + charge_t1), self.cal_t)):
                        cs_qcounts[c][k] += 1
                    for k in range(math.ceil(ssum + charge_t1),
                                   min(math.floor(ssum + charge_t1 + charge_t2), self.cal_t)):
                        cs_pcounts[c][k] += 1
                    ssum += charge_t1 + charge_t2

        for m in range(self.cal_t):
            for c_s in cs:
                c_station = self.center.charge_stations[c_s]
                charge_sum = sum(len(v) for v in c_station.charge.values())
                queue_sum = sum(len(v) for v in c_station.queue.values())
                if cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum > self.center.charge_stations[
                    c_s].capacity:
                    csum += 100 * (cs_qcounts[c_s][m] + cs_pcounts[c_s][m] + charge_sum + queue_sum -
                                   self.center.charge_stations[c_s].capacity)
                    all_fit = 5
                    break

        eval_s = 0
        for m in range(self.cal_t):
            for edge in self.center.edges.values():
                cap, x = edge.capacity["all"]
                eval_s += ((edge.capacity["all"][1] + road_counts[(edge.origin, edge.destination)][
                    m]) * edge.free_time * (1 + edge.b * (
                        (x + road_counts[(edge.origin, edge.destination)][m]) / cap) ** edge.power))
            for node in self.center.nodes.values():
                for p, is_wait in node.wait:
                    eval_s += is_wait + node_counts[node.id][m]
            for charge_station in self.center.charge_stations.values():
                if list(charge_station.charge.values()) != [[]]:
                    for ipair in charge_station.charge.values():
                        for i in ipair:
                            eval_s += i[1] + cs_pcounts[charge_station.id][m]
                for p, n in charge_station.pile.items():
                    length = len(charge_station.queue[p])
                    if length > 0:
                        for i, time in charge_station.queue[p]:
                            eval_s += (length + cs_qcounts[charge_station.id][m]) * time
        csum += eval_s
        return (csum + 0.5 * psum) / len(self.v_charge)

    def evaluate(self, population):
        with multiprocessing.Pool() as pool:
            eval = pool.map(self.evaluate_individual, population)
        return eval

    def update_velocity(self, particle, velocity, pbest, gbest):
        r1 = np.random.rand(len(particle))
        r2 = np.random.rand(len(particle))
        new_velocity = self.w * velocity + self.c1 * r1 * (pbest - particle) + self.c2 * r2 * (gbest - particle)
        return new_velocity

    def update_position(self, particle, velocity):
        new_position = particle + velocity
        new_position = np.clip(new_position, 0, [self.n_path1, self.n_path2, self.n_cs, self.n_power] * self.batch_size)
        return new_position

    def run(self):
        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                self.velocities[i] = self.update_velocity(self.population[i], self.velocities[i], self.pbest[i], self.gbest)
                self.population[i] = self.update_position(self.population[i], self.velocities[i])
                if self.evaluate_individual(self.population[i]) < self.evaluate_individual(self.pbest[i]):
                    self.pbest[i] = self.population[i]
            self.gbest = self.pbest[np.argmin(self.evaluate(self.pbest))]
        return self.gbest


















# 除了NSGA2和SPEA2之外，还有以下几种常见的多目标优化算法适用于你的问题：
# MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)：
# 该算法将多目标优化问题分解为若干个单目标优化问题，并同时求解这些子问题。
# PAES (Pareto Archived Evolution Strategy)：
# 该算法使用一个存档来保存非支配解，并通过局部搜索来生成新解。
# MOPSO (Multi-Objective Particle Swarm Optimization)：
# 该算法是粒子群优化算法的多目标版本，使用粒子群的概念来搜索解空间。
# IBEA (Indicator-Based Evolutionary Algorithm)：
# 该算法使用性能指标（如Hypervolume）来指导选择过程。
# GDE3 (Generalized Differential Evolution 3)：
# 该算法是差分进化算法的多目标版本，适用于连续优化问题。
# 这些算法各有优缺点，可以根据具体问题的需求选择合适的算法