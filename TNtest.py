import math


def calculate_wait_cs(s, k, l, m):
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
            p_0 += rou ** i / math.factorial(i)
        if rou_s == 1:
            p_0 += (k - s + 1) * (rou ** s) / math.factorial(s)
        else:
            p_0 += (rou ** s) * (1 - rou_s ** (k - s + 1)) / math.factorial(s) / (1 - rou_s)
        p_0 = 1 / p_0
        p_k = p_0 * (rou ** k) / math.factorial(s) / (s ** (k - s))
        print("During calculate")
        print(rou, k, rou ** k, math.factorial(s), s, k - s)
        l_e = l * (1 - p_k)
        if rou_s == 1:
            L_q = p_0 * (rou ** s) * (k - s) * (k - s + 1) / 2 / math.factorial(s)
        else:
            L_q = p_0 * (rou ** s) * rou_s * (
                        1 - (rou_s ** (k - s + 1)) - (1 - rou_s) * (k - s + 1) * (rou_s ** (k - s)))
        print("After calculate")
        print(p_0, p_k, l_e, L_q)
        if l_e == 0:
            return 0
        else:
            return L_q / l_e


calculate_wait_cs(2, 5, 2, 0.5)