import pandapower as pp
import PDNplus
from pandapower import networks as pn, diagnostic
from pandapower.networks import case14
import logging
import numpy as np
import networkx as nx



def t_ieee33():
    net = pn.case33bw()
    net.load['controllable'] = True
    net.load['p_mw'] = 0
    net.load['q_mvar'] = 0
    net.load['min_p_mw'] = 0
    net.load['min_q_mvar'] = 0
    net.load['max_p_mw'] = 20
    net.load['max_q_mvar'] = 20

    # Run the optimal power flow calculation
    pp.runopp(net)
    print(net.res_bus)


def create_ieee33():

    net = pn.case33bw()
    # print(net.load)
    # print(net.line)
    # print(net.ext_grid.loc[0, 'max_p_mw'])
    # print(666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666)
    net.load.drop(net.load.index, inplace=True)
    net.ext_grid.loc[0, 'max_p_mw'] = 100

    pp.create_load(net, bus=1, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 155", controllable=True)


    pp.create_load(net, bus=2, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 165", controllable=True)


    pp.create_load(net, bus=3, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 3", controllable=True)


    pp.create_load(net, bus=4, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 40", controllable=True)


    pp.create_load(net, bus=5, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 1", controllable=True)


    pp.create_load(net, bus=6, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 33", controllable=True)


    pp.create_load(net, bus=7, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 81", controllable=True)


    pp.create_load(net, bus=8, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 83", controllable=True)


    pp.create_load(net, bus=9, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 111", controllable=True)


    pp.create_load(net, bus=10, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 103", controllable=True)


    pp.create_load(net, bus=11, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 100", controllable=True)


    pp.create_load(net, bus=12, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 94", controllable=True)


    pp.create_load(net, bus=13, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 95", controllable=True)


    pp.create_load(net, bus=14, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 96", controllable=True)


    pp.create_load(net, bus=15, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 79", controllable=True)


    pp.create_load(net, bus=16, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 69", controllable=True)


    pp.create_load(net, bus=17, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 64", controllable=True)


    pp.create_load(net, bus=18, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 141", controllable=True)


    pp.create_load(net, bus=19, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 146", controllable=True)


    pp.create_load(net, bus=20, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 136", controllable=True)


    pp.create_load(net, bus=21, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 140", controllable=True)


    pp.create_load(net, bus=22, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 59", controllable=True)


    pp.create_load(net, bus=23, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 46", controllable=True)


    pp.create_load(net, bus=24, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 62", controllable=True)


    pp.create_load(net, bus=25, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 167", controllable=True)


    pp.create_load(net, bus=26, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 116", controllable=True)


    pp.create_load(net, bus=27, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 31", controllable=True)


    pp.create_load(net, bus=28, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 29", controllable=True)


    pp.create_load(net, bus=29, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 107", controllable=True)


    pp.create_load(net, bus=30, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 19", controllable=True)


    pp.create_load(net, bus=31, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 20", controllable=True)


    pp.create_load(net, bus=32, p_mw=0.1, q_mvar=0.02, min_p_mw=0.1, min_q_mvar=0.02, max_p_mw=10, max_q_mvar=2,
                   name="EVCS 15", controllable=True)

    pp.create_gen(net, bus=1, p_mw=100.0, vm_pu=1.0, max_q_mvar=100.0, min_q_mvar=-100.0, min_p_mw=0.0, max_p_mw=100.0,
                  min_vm_pu=0.9, max_vm_pu=1.1, name="Generator 1", controllable=True)
    pp.create_gen(net, bus=2, p_mw=1.0, vm_pu=1.0, max_q_mvar=100.0, min_q_mvar=-100.0, min_p_mw=0.0, max_p_mw=100.0,
                  min_vm_pu=0.9, max_vm_pu=1.1, name="Generator 2", controllable=True)
    pp.create_gen(net, bus=5, p_mw=1.0, vm_pu=1.0, max_q_mvar=100.0, min_q_mvar=-100.0, min_p_mw=0.0, max_p_mw=100.0,
                  min_vm_pu=0.9, max_vm_pu=1.1, name="Generator 3", controllable=True)
    pp.create_gen(net, bus=8, p_mw=1.0, vm_pu=1.0, max_q_mvar=100.0, min_q_mvar=-100.0, min_p_mw=0.0, max_p_mw=100.0,
                  min_vm_pu=0.9, max_vm_pu=1.1, name="Generator 4", controllable=True)

    for ext_grid_idx in net.ext_grid.index:
        net.poly_cost.drop(net.poly_cost[net.poly_cost.element == ext_grid_idx].index, inplace=True)
        pp.create_poly_cost(net, element=ext_grid_idx, et="ext_grid", cp1_eur_per_mw=15, cp2_eur_per_mw2=0.03, cp0_eur=0)

    return net

def update_load(net, total_load, time_slot):
    # 更新充电站的负载
    min_p = total_load[min((k for k in total_load if total_load[k] > 0), key=total_load.get)]
    max_p= total_load[max((k for k in total_load if total_load[k] > 0), key=total_load.get)]
    min_avg_power = min_p / time_slot
    max_avg_power = max_p / time_slot
    for load in net.load.itertuples():
        if load.name in total_load:
            # 计算平均功率
            avg_power_mw = total_load[load.name] / time_slot
            # 更新负载
            if avg_power_mw > 0:
                net.load.at[load.Index, 'p_mw'] = np.float64(avg_power_mw)
                net.load.at[load.Index, 'q_mvar'] = np.float64(avg_power_mw * 0.2)
                net.load.at[load.Index, 'min_p_mw'] = np.float64(avg_power_mw)
                net.load.at[load.Index, 'max_p_mw'] = np.float64(max_avg_power * 2)
                net.load.at[load.Index, 'min_q_mvar'] = np.float64(avg_power_mw * 0.2)
                net.load.at[load.Index, 'max_q_mvar'] = np.float64(max_avg_power * 0.4)
            else:
                net.load.at[load.Index, 'p_mw'] = np.float64(min_avg_power)
                net.load.at[load.Index, 'q_mvar'] = np.float64(min_avg_power * 0.2)
                net.load.at[load.Index, 'min_p_mw'] = np.float64(min_avg_power)
                net.load.at[load.Index, 'max_p_mw'] = np.float64(max_avg_power * 2)
                net.load.at[load.Index, 'min_q_mvar'] = np.float64(min_avg_power * 0.2)
                net.load.at[load.Index, 'max_q_mvar'] = np.float64(max_avg_power * 0.4)



# def calculate_impedance_to_slack_and_generators(net):
#     # Create the network graph
#     graph = pp.topology.create_nxgraph(net, respect_switches=True)
#
#     # Get the index of the slack bus
#     slack_bus = net.ext_grid.bus.values[0]
#
#     # Calculate impedance to the slack bus
#     impedance_to_slack = {}
#     for bus in net.bus.index:
#         path = pp.topology.dijkstra(graph, slack_bus, bus, weight='r_ohm')
#         impedance_to_slack[bus] = sum(path.values())
#
#     # Calculate impedance to each generator
#     impedance_to_generators = {}
#     for gen_idx in net.gen.index:
#         gen_bus = net.gen.at[gen_idx, 'bus']
#         impedance_to_generators[gen_bus] = {}
#         for bus in net.bus.index:
#             path = pp.topology.dijkstra(graph, gen_bus, bus, weight='r_ohm')
#             impedance_to_generators[gen_bus][bus] = sum(path.values())
#
#     return impedance_to_slack, impedance_to_generators
#
# # Example usage
# net = pn.case33bw()
# impedance_to_slack, impedance_to_generators = calculate_impedance_to_slack_and_generators(net)
#
# print("Impedance to slack bus (0):")
# print(impedance_to_slack)
#
# print("\nImpedance to generators:")
# for gen_bus, impedance in impedance_to_generators.items():
#     print(f"Generator at bus {gen_bus}:")
#     print(impedance)






if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    t_ieee33()
    # print(333333333333333333333333333333333333333333333333333333333333333333333)



    net = create_ieee33()
    total_charge_cost = {'EVCS 167': 0.024, 'EVCS 1': 0.436, 'EVCS 3': 0.12, 'EVCS 19': 0.652, 'EVCS 20': 1.274, 'EVCS 25': 0.036, 'EVCS 29': 0.052, 'EVCS 31': 0.0, 'EVCS 33': 0.0, 'EVCS 40': 0.0, 'EVCS 46': 0.0, 'EVCS 59': 0.0, 'EVCS 62': 0.0, 'EVCS 64': 0.0, 'EVCS 69': 0.0, 'EVCS 79': 0.0, 'EVCS 81': 0.0, 'EVCS 83': 0.0, 'EVCS 94': 0.0, 'EVCS 95': 0.0, 'EVCS 96': 0.0, 'EVCS 100': 0.0, 'EVCS 103': 0.0, 'EVCS 107': 0.028, 'EVCS 111': 0.0, 'EVCS 116': 0.008, 'EVCS 136': 0.0, 'EVCS 140': 0.0, 'EVCS 141': 0.0, 'EVCS 146': 0.0, 'EVCS 155': 0.0, 'EVCS 165': 0.0}
    update_load(net, total_charge_cost, 3 * 10 / 60)
    pp.diagnostic(net)


    # net = pn.case14()
    # net.load['controllable'] = True
    #
    # # Run the optimal power flow calculation
    # pp.runopp(net)
    # print(net.res_bus)

    # # Extract LMP values from the OPF results
    # lmp = net.res_bus['vm_pu'] * net.res_bus['va_degree']  # Example calculation, adjust as needed
    #
    # print("LMP values:")
    # print(lmp)

    # print("net")
    # print(net)
    # print('')
    print("load")
    print(net.load)
    # print('')
    # print("line")
    # print(net.line)
    # print('')
    # print("ext_grid")
    # print(net.ext_grid)
    # print('')
    # print("gen")
    # print(net.gen)
    # print('')
    # print("bus")
    # print(net.bus)
    # print('')

    # total_load = net.load.p_mw.sum()
    # total_gen = net.ext_grid['max_p_mw'].sum()
    #
    # print("Total load demand:", total_load, "MW")
    # print("Total generation capacity:", total_gen, "MW")

    # net.ext_grid['max_p_mw'] = 10000
    # net.ext_grid['min_p_mw'] = 0
    # net.ext_grid['max_q_mvar'] = 10000
    # net.ext_grid['min_q_mvar'] = -10000
    # total_gen = net.ext_grid['max_p_mw'].sum()
    # print("ext_grid")
    # print(net.ext_grid)
    # print("Total generation capacity:", total_gen, "MW")

    pp.runopp(net,max_iter = 300)
    # try:
    #     pp.runopp(net, verbose=True)
    # except pp.optimal_powerflow.OPFNotConverged as e:
    #     logging.error("Optimal Power Flow did not converge!")
    #     logging.error(e)
    # except Exception as e:
    #     logging.error("An error occurred during OPF calculation!")
    #     logging.error(e)

    # net1 = pn.case33bw()
    # pp.runopp(net1)

    # # pp.diagnostic(net)
    # print("generator")
    # print(net.gen)
    #
    #
    # print("bus")
    # print(net.bus)
    #
    #
    # pp.runopp(net)
    # print("result")
    # print(111)
    #
    # print(net.res_bus)
    # print(net.res_gen)
    # print(net.res_load)

    # loss = PDNplus.calculate_loss(net, 140)
    # print(loss)

