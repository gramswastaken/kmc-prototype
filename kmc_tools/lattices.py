from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple, Dict, List
import itertools
import numpy as np
import scipy as sp

from kmc_tools.logging import LogParams


class ZBUCell(Enum):
    """Const container for Zincblende unit cell locations"""

    # If it has two (or none) 0.75 its an fcc position, if it has one (or three) then it is an interstitial
    # The positions are not neccarrily logically ordered, the neighbor logic has to be computed later anyway
    FCC_1 = (
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
    )
    FCC_2 = (
        (0.25, 0.25, 0.25),
        (0.75, 0.75, 0.25),
        (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75),
    )
    TET_1 = (
        (0.75, 0.75, 0.75),
        (0.25, 0.25, 0.75),
        (0.25, 0.75, 0.25),
        (0.75, 0.25, 0.25),
    )
    TET_2 = ((0.5, 0.5, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5))


class RocksaltCell(Enum):
    ONE = 1
    TWO = 2


class OccupationType(IntEnum):
    EMPTY = 0
    GA = 1
    AS = 2


@dataclass
class LatticeSite:
    id: int
    cell: Tuple[int, int, int]
    sublattice: ZBUCell
    occupation_type: OccupationType
    location: np.ndarray


@dataclass
class ZBLatticeState:
    """
    Every lattice is thought of as a superlattice of some unit cell.
    Only works for GaAs atm
    """

    def __init__(self, dimensions: Tuple[int, int, int] = (1, 1, 1)) -> None:
        """
        Frontloading computational work by precomputing the neighbor lists for all sites in the simulation.
        """
        self.superlattice_dimensions: Tuple[int, int, int] = dimensions
        self.sitelist = build_gaas_superlattice(dimensions=dimensions, inter=False)
        self.sites: Dict[int, LatticeSite] = {s.id: s for s in self.sitelist}
        self.fnn_lists: Dict[int, List[int]] = build_fnn_lists_tree_search(
            self.sites, 0.44, dimensions
        )
        self.snn_lists: Dict[int, List[int]] = build_snn_lists(
            self.sites, self.fnn_lists
        )


def build_fnn_lists_tree_search(
    sites: Dict[int, LatticeSite], cutoff: float, dimensions: Tuple[int, int, int]
) -> Dict[int, List[int]]:
    site_ids = list(sites.keys())
    neighbor_lists = {id: [] for id in site_ids}

    dim = np.array(dimensions, dtype=float)
    positions = np.array([sites[sid].location for sid in site_ids])

    tree = sp.spatial.cKDTree(positions, boxsize=dim)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")

    for i, j in pairs:
        neighbor_lists[site_ids[i]].append(site_ids[j])
        neighbor_lists[site_ids[j]].append(site_ids[i])

    return neighbor_lists


def build_fnn_lists(
    sites: Dict[int, LatticeSite], cutoff: float, dimensions: Tuple[int, int, int]
) -> Dict[int, List[int]]:
    site_ids = sites.keys()
    neighbor_lists = {id: [] for id in site_ids}
    dim = np.array(dimensions)
    keylist = list(sites.keys())

    for i_idx, i in enumerate(keylist):
        si = sites[i]
        for j in keylist[i_idx + 1 :]:
            sj = sites[j]

            if i == j:
                continue

            # next 2 lines periodic boundaries
            delta = si.location - sj.location
            delta = delta - dim * np.round(delta / dim)

            # symmetrically add as neighbors
            if np.linalg.norm(delta) <= cutoff:
                neighbor_lists[i].append(j)
                neighbor_lists[j].append(i)

    return neighbor_lists


def build_snn_lists(
    sites: Dict[int, LatticeSite],
    fnn_lists: Dict[int, List[int]],
) -> Dict[int, List[int]]:
    site_ids = sites.keys()
    snn_lists = {id: [] for id in site_ids}
    for id, neighbors in fnn_lists.items():
        for fnid in neighbors:
            snids: List[int] = fnn_lists[fnid].copy()
            snids.remove(id)
            snn_lists[id] += snids

    return snn_lists


def build_gaas_superlattice(
    dimensions: Tuple[int, int, int], inter: bool
) -> List[LatticeSite]:
    nx, ny, nz = dimensions
    sites: List[LatticeSite] = []
    site_id: int = 0

    for ix, iy, iz in itertools.product(range(nx), range(ny), range(nz)):
        uc_origin = np.array((ix, iy, iz), dtype=float)
        for position in ZBUCell.FCC_1.value:
            sites.append(
                LatticeSite(
                    id=site_id,
                    cell=(ix, iy, iz),
                    sublattice=ZBUCell.FCC_1,
                    occupation_type=OccupationType.EMPTY,
                    location=np.add(np.array(position), uc_origin),
                )
            )
            site_id += 1
        for position in ZBUCell.FCC_2.value:
            sites.append(
                LatticeSite(
                    id=site_id,
                    cell=(ix, iy, iz),
                    sublattice=ZBUCell.FCC_2,
                    occupation_type=OccupationType.EMPTY,
                    location=np.add(np.array(position), uc_origin),
                )
            )
            site_id += 1
        if inter:
            for position in ZBUCell.TET_1.value:
                sites.append(
                    LatticeSite(
                        id=site_id,
                        cell=(ix, iy, iz),
                        sublattice=ZBUCell.TET_1,
                        occupation_type=OccupationType.EMPTY,
                        location=np.add(np.array(position), uc_origin),
                    )
                )
            site_id += 1
            for position in ZBUCell.TET_2.value:
                sites.append(
                    LatticeSite(
                        id=site_id,
                        cell=(ix, iy, iz),
                        sublattice=ZBUCell.TET_2,
                        occupation_type=OccupationType.EMPTY,
                        location=np.add(np.array(position), uc_origin),
                    )
                )
                site_id += 1

    return sites


def dump_zblattice_spparksfmt(lat: ZBLatticeState, params: LogParams) -> None:
    sites_def: List = []
    neighbors_def: List[int] = []
    for site in lat.sitelist:
        sitestr = [
            str(site.id),
            str(site.location[0]),
            str(site.location[1]),
            str(site.location[2]),
        ]
        nlist = [str(site.id)] + [str(nid) for nid in lat.fnn_lists[site.id]]
        sites_def.append(sitestr)
        neighbors_def.append(nlist)

    # this line of code is why python is both great and cursed
    sdef_column_widths: List[int] = [max(map(len, col)) for col in zip(*sites_def)]
    ndef_col_widths: List[int] = [max(map(len, col)) for col in zip(*neighbors_def)]

    logfile = params.log_path + f"/{params.name}-spparksfmt-lat"

    with open(logfile, "w") as log:
        heading: str = (
            "simulation \n",
            "3 dimension \n",
            f"{len(lat.sitelist)} sites \n",
            f"{len(lat.fnn_lists[0])} max neighbors \n",
            f"0 {lat.superlattice_dimensions[0]} xlo xhi \n",
            f"0 {lat.superlattice_dimensions[1]} ylo yhi \n",
            f"0 {lat.superlattice_dimensions[2]} zlo zhi \n",
        )
        for line in heading:
            log.write(line)

        log.write("\nSites\n\n")
        for sitestr in sites_def:
            log.write(
                "\t".join(
                    sdef.ljust(colwidth)
                    for sdef, colwidth in zip(sitestr, sdef_column_widths)
                )
            )
            log.write("\n")

        log.write("\nNeighbors\n\n")
        for nlist in neighbors_def:
            log.write(
                "\t".join(
                    nid.ljust(colwidth) for nid, colwidth in zip(nlist, ndef_col_widths)
                )
            )
            log.write("\n")

        # log.write("\n Values \n")


def dump_zblattice(lat: ZBLatticeState) -> None:
    print("=" * 50)
    print("Dump of Simulation Sate -- Lattice Only")
    print("=" * 50)
    print(
        f"dimensions={lat.superlattice_dimensions}, num_lattice_sites={len(lat.sites)}"
    )
    for site_id, _ in lat.sites.items():
        fnn_list = lat.fnn_lists[site_id]
        snn_list = lat.snn_lists[site_id]
        print(
            f"\nsite-id: {site_id}, cell: {lat.sites[site_id].cell}, sublattice: {lat.sites[site_id].sublattice.name}"
        )
        print(f"location: {lat.sites[site_id].location}")
        print(f"fnns: {fnn_list}")
        print(f"snns: {snn_list}")

    return


def debug_fnn_distance(
    sites: Dict[int, LatticeSite], test_site_id: int, cutoff: float
) -> None:
    distances: List = []
    si = sites[test_site_id]

    for j in sites.keys():
        if j == test_site_id:
            continue
        sj = sites[j]
        dist = np.linalg.norm(si.location - sj.location)
        if dist <= cutoff:
            distances.append((dist, sj.id, sj.sublattice, sj.location))

    distances.sort(key=lambda x: x[0])

    print(f"{len(distances)} neighbors with cutoff {cutoff}")
    for dist, id, sublattice, loc in distances[:30]:
        print(f"dist: {dist}, id: {id}, loc: {loc}, sublattice: {sublattice}")


def set_gaas_001_substrate(lat: ZBLatticeState, layers: int) -> bool:
    """Only does one layer but will need to do more"""
    if layers < 1:
        return False
    layer: int = layers
    while layer > 0:
        for _, s in lat.sites.items():
            if s.location[2] == layer:
                s.occupation_type = OccupationType.GA
                continue
            elif s.location[2] == layer + 0.25:
                s.occupation_type = OccupationType.AS
                continue
        layer -= 1

    return True


def get_site_coordination(lat: ZBLatticeState, site: LatticeSite) -> int:
    neighbor_ids = lat.fnn_lists[site.id]
    coordination: int = 0
    for nid in neighbor_ids:
        if lat.sites[nid].occupation_type == OccupationType.EMPTY:
            coordination += 1
    return coordination
