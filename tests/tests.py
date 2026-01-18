from kmc_tools.sim_engine import (
    KMCSimulation,
    figure_dump_simulation,
    run,
)
from kmc_tools.lattices import ZBLatticeState
import os

if __name__ == "__main__":
    print(os.getcwd())
    # lat = ZBLatticeState((5, 5, 5))
    # sim = KMCSimulation(lattice_state=lat, temperature=850)
    # run(sim, 2000, 1000)
    # figure_dump_simulation(sim, "test_figs1.png")
