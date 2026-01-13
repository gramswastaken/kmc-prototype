from kmcproto import (
    ZBLatticeState,
    debug_fnn_distance,
    #    build_gaas_superlattice,
    #    build_fnn_lists,
    dump_zblattice,
)


if __name__ == "__main__":
    dimensions = (5, 5, 5)
    lat = ZBLatticeState(dimensions)
    dump_zblattice(lat)
    debug_fnn_distance(lat.sites, 531, 0.44)
