import math
import csv
import ast
import pandas as pd
import numpy as np

def calculate_wait_cs(s, k, l, m):
    # s = self.pile[power]
    # k = self.capacity * self.pile[power]/ sum(self.pile.values())
    # l = self.v_arrive / T
    # if self.t_cost[1] != 0:
    #     m = self.t_cost[1] / self.t_cost[0]
    # else:
    #     m = 1


    def calculate_p_0(rou, s, k, rou_s):
        p_0 = 0

        for i in range(s):
            p_0 += math.exp(i * math.log(rou) - math.log(math.factorial(i)))

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

        return 1 / p_0


    def calculate_p_k(p_0, rou, k, s):
        if k < s:
            print("k < s")
            return p_0 * (rou ** k) / math.factorial(k)
        else:
            # 使用对数计算避免溢出
            log_p_k = (math.log(p_0)
                       + k * math.log(rou)
                       - math.log(math.factorial(s))
                       - (k - s) * math.log(s))
            return math.exp(log_p_k)

    def calculate_L_q(p_0, rou, s, k, rou_s):
        if rou_s == 1:
            log_L_q = math.log(p_0) + s * math.log(rou) + math.log(k - s) + math.log(k - s + 1) - math.log(2) - math.log(math.factorial(s))
            L_q = math.exp(log_L_q)
        else:
            # 使用对数计算避免溢出
            log_part1 = math.log(p_0) + s * math.log(rou) + math.log(rou_s)
            log_part2_1 = (k - s + 1) * math.log(rou_s) + math.log(k - s)
            log_part2_2 = (k - s) * math.log(rou_s) + math.log(k - s + 1)
            log_part2 = math.log(1 + math.exp(log_part2_1) - math.exp(log_part2_2))
            log_part3 = - math.log(math.factorial(s)) - math.log(math.fabs(1 - rou_s)) * 2

            log_L_q = log_part1 + log_part2 + log_part3
            L_q = math.exp(log_L_q)

        return L_q

    print("Before calculate")
    print("s, k, l, m: ", end='')
    print(s, k, l, m)
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
        print("After calculate")
        print("p_0, p_k, L_q, l_e: ", end='')
        print(p_0, p_k, L_q, l_e)
        if l_e == 0:
            return 0
        else:
            print(L_q / l_e)
            return L_q / l_e


def read_csv_to_list(file_path):
    all_data = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row_data = [ast.literal_eval(cell) for cell in row]
            all_data.append(row_data)
    return all_data


def generate_od():
    # Initialize lists to store the data
    origins = []
    destinations = []
    flows = []

    # Read the file
    with open('data/EMA/EMA_trips.tntp', 'r') as file:
        lines = file.readlines()

    # Parse the file
    current_origin = None
    for line in lines:
        line = line.strip()
        if line.startswith('Origin'):
            current_origin = int(line.split()[1])
        elif ':' in line:
            parts = line.split(';')
            for part in parts:
                if ':' in part:
                    destination, flow = part.split(':')
                    destination = int(destination.strip())
                    flow = float(flow.strip())
                    origins.append(current_origin)
                    destinations.append(destination)
                    # if flow == 0 : flow = 1
                    flows.append(np.ceil(flow))

    # Create a DataFrame
    df = pd.DataFrame({
        'O': origins,
        'D': destinations,
        'Ton': flows
    })

    # Save to CSV
    df.to_csv('data/EMA/EMA_od.csv', index=False)


generate_od()

# file_name = 'OD_output.csv'
# data = read_csv_to_list(file_name)
# print(data)

# print("result testing1")
# calculate_wait_cs(2, 5, 2, 0.5)
# print('\n')
# print("result testing2")
# calculate_wait_cs(20, 50, 20, 0.5)
# print('\n')
# print("result testing3")
# calculate_wait_cs(100, 125.0, 20.0, 1 / 13.553238105736334)
# print('\n')
# print("result testing4")
# calculate_wait_cs(100, 125.0, 40.0, 1 / 13.553238105736334)
# print('\n')
# print("result testing5")
# calculate_wait_cs(100, 125.0, 2.0, 1 / 13.553238105736334)
# print('\n')
# print("result testing6")
# calculate_wait_cs(100, 125.0, 4.0, 1 / 13.553238105736334)
# print('\n')
# print("result testing7")
# calculate_wait_cs(100, 125.0, 100.0, 1 / 13.553238105736334)
# print('\n')
# print("result testing8")
# calculate_wait_cs(100, 125.0, 120.0, 1 / 13.553238105736334)
# print('\n')
# print("result testing9")
# calculate_wait_cs(100, 125.0, 1.4, 0.050242593025730656)
# print('\n')
