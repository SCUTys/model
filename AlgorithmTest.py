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


cs_SF = [1, 5, 11, 13, 15, 20]
cs_EMA = [6, 10, 11, 17, 19, 22, 23, 25, 27, 29, 30, 33, 34, 38, 40, 42, 44, 47, 48, 49, 52, 57, 60, 63, 65, 69]
cs = cs_SF
T = 10


class HighQualityRandomGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def randint(self, low, high, size=None):
        return self.rng.integers(low, high, size)

    def random(self, size=None):
        return self.rng.random(size)



