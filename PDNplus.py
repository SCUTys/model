import pandapower as pp
from pandapower import networks as pn
from pandapower.networks import case14



def create_ieee14():
    net = pn.case14()
    # print(net)

    # 在节点2添加充电站
    pp.create_load(net, bus=1, p_mw=17, q_mvar=3.4, min_p_mw=0.1, min_q_mvar=0.002, max_p_mw=17, max_q_mvar=3.4, name="EVCS 1", controllable=True)
    pp.create_load(net, bus=1, p_mw=0.03, q_mvar=0.006, name="EVCS 1 Auxiliary Load")

    # 在节点4添加充电站
    pp.create_load(net, bus=3, p_mw=17, q_mvar=3.4, min_p_mw=0.1, min_q_mvar=0.002, max_p_mw=1., max_q_mvar=3.4, name="EVCS 5", controllable=True)
    pp.create_load(net, bus=3, p_mw=0.03, q_mvar=0.006, name="EVCS 5 Auxiliary Load")
    #
    # 在节点7添加充电站
    pp.create_load(net, bus=6, p_mw=17, q_mvar=3.4, min_p_mw=0.1, min_q_mvar=0.002, max_p_mw=17, max_q_mvar=3.4, name="EVCS 16", controllable=True)
    pp.create_load(net, bus=6, p_mw=0.03, q_mvar=0.006, name="EVCS 16 Auxiliary Load")
    #
    # 在节点8添加充电站
    pp.create_load(net, bus=7, p_mw=17, q_mvar=3.4, min_p_mw=0.1, min_q_mvar=0.002, max_p_mw=17, max_q_mvar=3.4, name="EVCS 11", controllable=True)
    pp.create_load(net, bus=7, p_mw=0.03, q_mvar=0.006, name="EVCS 11 Auxiliary Load")
    #
    # 在节点10添加充电站
    pp.create_load(net, bus=9, p_mw=17, q_mvar=3.4, min_p_mw=0.1, min_q_mvar=0.002, max_p_mw=17, max_q_mvar=3.4, name="EVCS 15", controllable=True)
    pp.create_load(net, bus=9, p_mw=0.03, q_mvar=0.006, name="EVCS 15 Auxiliary Load")
    #
    # 在节点14添加充电站
    pp.create_load(net, bus=13, p_mw=17, q_mvar=3.4, min_p_mw=0.1, min_q_mvar=0.002, max_p_mw=17, max_q_mvar=3.4, name="EVCS 20", controllable=True)
    pp.create_load(net, bus=13, p_mw=0.03, q_mvar=0.006, name="EVCS 20 Auxiliary Load")
    #
    for gen_idx in net.gen.index:
        net.poly_cost.drop(net.poly_cost[net.poly_cost.element == gen_idx].index, inplace=True)
        pp.create_poly_cost(net, element=gen_idx, et="gen", cp1_eur_per_mw=15, cp2_eur_per_mw2=0.03, cp0_eur=0)

    net.bus['max_vm_pu'] = 1.2
    #
    # for line in [7, 8, 9, 10, 11, 12, 13, 14]:
    #      if net.line.loc[line, 'r_ohm_per_km'] <= 0.001:
    #          net.line.loc[line, 'r_ohm_per_km'] = 0.002  # 设置为合理的最小值
    #      if net.line.loc[line, 'x_ohm_per_km'] <= 0.001:
    #          net.line.loc[line, 'x_ohm_per_km'] = 0.002  # 设置为合理的最小值

    return net



def test_load_increase(net):
    # 初始负载缩放比例
    scale_factor = 0.001

    # 逐步增加负载
    for i in range(1, 1001):
        net.load['p_mw'] *= scale_factor
        net.load['q_mvar'] *= scale_factor
        try:
            pp.runpp(net)
            print(f"Power flow converges with load scaled to {scale_factor * 100}%")
            scale_factor += 0.001  # 增加负载比例
        except pp.LoadflowNotConverged:
            print(f"Power flow does not converge with load scaled to {scale_factor * 100}%")
            break



def run(net, max_iter = 10):
    print("114514")
    pp.diagnostic(net)
    pp.runopp(net, max_iteration=max_iter)


def update_load(net, total_load, time_slot):
    # 更新充电站的负载
    for load in net.load.itertuples():
        if load.name in total_load:
            # 计算平均功率
            avg_power_mw = total_load[load.name] / time_slot
            # 更新负载
            net.load.at[load.Index, 'p_mw'] = avg_power_mw
            net.load.at[load.Index, 'q_mvar'] = avg_power_mw * 0.2
            print('p_mw')
            print(load.Index, avg_power_mw)


def calculate_loss(net, rou):
# 获取平衡节点（1号节点）输送的有功功率
    slack_bus = net.ext_grid.bus.values[0]

# 计算平衡节点输送的有功功率之和
    slack_power = net.res_gen[net.gen.bus == slack_bus]['p_mw'].sum()

    # 计算所有发电机的代价函数之和
    total_cost = 0
    for gen_idx in net.gen.index:
        p_mw = net.res_gen.at[gen_idx, 'p_mw']
        cost_params = net.poly_cost[net.poly_cost.element == gen_idx]
        cp0 = cost_params.cp0_eur.values[0]
        cp1 = cost_params.cp1_eur_per_mw.values[0]
        cp2 = cost_params.cp2_eur_per_mw2.values[0]
        total_cost += cp0 + cp1 * p_mw + cp2 * p_mw ** 2

    return rou * slack_power + total_cost