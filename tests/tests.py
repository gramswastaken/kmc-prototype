from kmc_tools.sim_engine import (
    KMCSimulation,
    figure_dump_simulation,
    run,
)
from kmc_tools.lattices import ZBLatticeState, dump_zblattice_spparksfmt
from kmc_tools.logging import LogParams


def basic():
    params = LogParams("data", "simulation-test")
    lat = ZBLatticeState((5, 5, 5))
    sim = KMCSimulation(lattice_state=lat, temperature=850)
    run(sim, 20000, 1000)
    figure_dump_simulation(sim, params.log_path + "/test_figs1.png")
    dump_zblattice_spparksfmt(lat, params)


def big_lattice():
    params = LogParams("data", "biglat-test")
    lat = ZBLatticeState((15, 15, 10))
    sim = KMCSimulation(lattice_state=lat, temperature=850)
    run(sim, 10000, 1000)
    figure_dump_simulation(sim, params.log_path + "/big_lat.png")
    dump_zblattice_spparksfmt(lat, params)


def big_lattice_2():
    params = LogParams("data", "long-test")
    lat = ZBLatticeState((20, 20, 20))
    sim = KMCSimulation(lattice_state=lat, temperature=850)
    run(sim, 1000000, 1000)
    figure_dump_simulation(sim, params.log_path + "/long.png")
    dump_zblattice_spparksfmt(lat, params)


if __name__ == "__main__":
    big_lattice_2()
