import pandapower as pp
import PDNplus
from pandapower import networks as pn
from pandapower.networks import case14






if __name__ == "__main__":
    net = PDNplus.create_ieee14()
    print("net")
    print(net)
    # pp.diagnostic(net)
    print("generator")
    print(net.gen)

    print("bus")
    print(net.bus)


    pp.runopp(net)
    print("result")
    print(111)

    print(net.res_bus)
    print(net.res_gen)
    print(net.res_load)

    # loss = PDNplus.calculate_loss(net, 140)
    # print(loss)

