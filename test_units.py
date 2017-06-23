import data as dl
from sim import *
from matplotlib import pyplot as plt

def test_sim ():
    # Generate 10 point test dataset
    od, d = dl.test_data_lin(10)

    sim = Simulator(od, d)

    # First buy 90% crypt
    sim.step(0, .9)

    # Hold 4 steps
    for i in range(4):
        sim.step(2, .5)

    # Sell 50%
    sim.step(1, .5)

    # Hold 4 steps
    for i in range(3):
        sim.step(2, .5)

    # Asset value should be 100 + 2700 + 4050 = 6850
    assert sim.assets == 6850.
