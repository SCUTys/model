import pandapower as pp
import PDNplus
from pandapower import networks as pn
from pandapower.networks import case14






if __name__ == "__main__":
    net = PDNplus.create_ieee14()
    # pp.diagnostic(net)
    pp.runpp(net)
    print(net.res_gen)

    loss = PDNplus.calculate_loss(net, 140)
    print(loss)

