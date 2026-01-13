from enum import Enum, IntEnum
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import scipy as sp
import itertools
import random

kb = sp.constants.k
default_prefactor = 1e13


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


class OccupationType(IntEnum):
    EMPTY = 0
    GA = 1
    AS = 2


@dataclass
class Process:
    """Single process with Arrheius rate"""

    name: str


@dataclass
class DiffusionProcess(Process):
    barrier: float
    prefactor: float = default_prefactor

    def rate(self, temperature: float) -> float:
        return self.prefactor * np.exp(-self.barrier / (kb * temperature))


@dataclass
class DepositionProcess(Process):
    gr: float = 0
    area: float = 1

    def rate(self) -> float:
        return self.gr / (1 / self.area)


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
    The default state is GaAs on a zincblende lattice with tetragonal interstitial positions
    """

    def __init__(self, dimensions: Tuple[int, int, int] = (1, 1, 1)) -> None:
        """
        Frontloading computational work by precomputing the neighbor lists for all sites in the simulation.
        """
        self.superlattice_dimensions: Tuple[int, int, int] = dimensions
        self.sitelist = build_gaas_superlattice(dimensions=dimensions, inter=False)
        self.sites: Dict[int, LatticeSite] = {s.id: s for s in self.sitelist}
        self.fnn_lists: Dict[int, List[int]] = build_fnn_lists(
            self.sites, 0.44, dimensions
        )
        self.snn_lists: Dict[int, List[int]] = build_snn_lists(
            self.sites, self.fnn_lists
        )


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
    # layer: int = layers
    for _, s in lat.sites.items():
        if s.location[2] == 0:
            s.occupation_type = OccupationType.GA
            continue
        elif s.location == 0.25:
            s.occupation_type = OccupationType.AS
            continue
    return True


def get_site_coordination(lat: ZBLatticeState, site: LatticeSite) -> int:
    neighbor_ids = lat.fnn_lists[site.id]
    coordination: int = 0
    for nid in neighbor_ids:
        if lat.sites[nid].occupation_type == OccupationType.EMPTY:
            coordination += 1
    return coordination


class EventType(IntEnum):
    DEPOSITION = 0
    DIFFUSION = 1


@dataclass
class SimulationEvent:
    # Diffusion for example would be from site_a to site_b
    process: Process
    rate: float
    type: EventType


@dataclass
class DiffusionEvent(SimulationEvent):
    site_a: LatticeSite
    site_b: LatticeSite


@dataclass
class DepositionEvent(SimulationEvent):
    site: LatticeSite
    species: ZBUCell


@dataclass
class KMCSimulation:
    """
    The siulation engine.
    This is the final meeting point for all of the data and logic needed to run the sim.
    Only III-V compounds on a ZB lattice are supported
    """

    flux_def = {"III": 0.1, "V": 0.8}

    def __init__(
        self,
        temperature: float,
        lattice_state: ZBLatticeState,
        # process_list: Dict[str, Process],
        # flux: Dict[str, float] = flux_def,
    ) -> None:
        self.temperature: float = temperature
        self.lattice_state: ZBLatticeState = lattice_state
        self.simulation_time: float = 0.0
        self.event_count: int = 0
        self.bond_energy = {
            (OccupationType.GA, OccupationType.AS): 0.8,
            (OccupationType.AS, OccupationType.GA): 0.8,
            (OccupationType.GA, OccupationType.GA): 0.16,
            (OccupationType.AS, OccupationType.AS): 0.2,
        }
        nx, ny, _ = self.lattice_state.superlattice_dimensions
        self.deposition_area = nx * ny * 2
        self.deposition_rates = {
            OccupationType.GA: 0.1,
            OccupationType.AS: 0.1,
        }
        # self.process_list = process_list
        # self.rates = {
        #    proc.name: proc.rate(self.temperature) for proc in self.process_list.items()
        # }
        # self.flux = flux


def _execute_event(sim: KMCSimulation, event: SimulationEvent) -> bool:
    # The referencing sites through the simulation is un necessary but I wanted to avoid implicitly mutating it
    lat = sim.lattice_state
    if isinstance(event, DiffusionEvent):
        site_a = lat.sites[event.site_a.id]
        site_b = lat.sites[event.site_b.id]
        site_b.occupation_type = site_a.occupation_type
        site_a.occupation_type = OccupationType.EMPTY
        return True

    if isinstance(event, DepositionEvent):
        site = lat.sites[event.site.id]
        if event.species == ZBUCell.FCC_1:
            site.occupation_type = OccupationType.GA
            return True
        if event.species == ZBUCell.FCC_2:
            site.occupation_type = OccupationType.AS
            return True

    return False


def get_available_events(sim: KMCSimulation) -> List[SimulationEvent]:
    """
    Builds the total possible event list for every site on the lattice
    """
    eventlist: List[SimulationEvent] = []
    lat = sim.lattice_state
    for site in lat.sitelist:
        diff_events: List[SimulationEvent] = get_diffusion_events(sim, site)
        dep_events: List[SimulationEvent] = get_deposition_events(sim, site)
        eventlist += diff_events
        eventlist += dep_events
    return eventlist


def get_deposition_events(
    sim: KMCSimulation, site: LatticeSite
) -> List[SimulationEvent]:
    """
    NOTE
    Event handling is extremely janky,
    the process and event model is not properly designed to accomodate depsoition
    """

    lat = sim.lattice_state
    coord = get_site_coordination(lat, site)
    if coord in (4, 3, 0):
        return []
    if site.sublattice == ZBUCell.FCC_1:
        process = DepositionProcess(
            "dep",
            gr=sim.deposition_rates[OccupationType.GA],
            area=sim.deposition_area,
        )
        return [
            DepositionEvent(
                process=process,
                rate=process.rate(),
                site=site,
                species=ZBUCell.FCC_1,
                type=EventType.DEPOSITION,
            )
        ]
    if site.sublattice == ZBUCell.FCC_2:
        process = DepositionProcess(
            "dep",
            gr=sim.deposition_rates[OccupationType.AS],
            area=sim.deposition_area,
        )
        return [
            DepositionEvent(
                process=process,
                rate=process.rate(),
                site=site,
                species=ZBUCell.FCC_2,
                type=EventType.DEPOSITION,
            )
        ]
    return []


def get_diffusion_events(
    sim: KMCSimulation, site: LatticeSite
) -> List[SimulationEvent]:
    """
    Computes diffusion barriers for a given lattice site based on the amrani model.
    An atom must break all bonds except for one fnn bond to diffuse.
    """
    # I think later on these checks should go everywhere and include a result type
    if site.occupation_type == OccupationType.EMPTY:
        return []

    lat = sim.lattice_state
    neighbors_id: List[int] = lat.fnn_lists[site.id]
    events: List[SimulationEvent] = []
    site_energy: float = compute_site_binding_energy(sim, site)

    for nid in neighbors_id:
        neighbor = lat.sites[nid]
        neighbor_coord = get_site_coordination(lat, neighbor)
        if neighbor_coord == 0 or neighbor == 1:
            continue
        if neighbor.occupation_type == OccupationType.EMPTY:
            barrier: float = (
                site_energy - sim.bond_energy[(OccupationType.GA, OccupationType.AS)]
            )
            process = DiffusionProcess(
                name=f"{site.occupation_type.name}-{neighbor.occupation_type.name}-hop",
                barrier=barrier,
            )
            rate: float = process.rate(sim.temperature)
            events.append(
                DiffusionEvent(
                    process=process,
                    rate=rate,
                    site_a=site,
                    site_b=neighbor,
                    type=EventType.DIFFUSION,
                )
            )
    return events


def compute_site_binding_energy(sim: KMCSimulation, site: LatticeSite) -> float:
    lat: ZBLatticeState = sim.lattice_state
    fnn_ids: List[int] = lat.fnn_lists[site.id]
    snn_ids: List[int] = lat.snn_lists[site.id]
    neighbor_ids: List[int] = fnn_ids + snn_ids
    energy: float = 0
    for nid in neighbor_ids:
        neighbor = lat.sites[nid]
        energy += sim.bond_energy[(site.occupation_type, neighbor.occupation_type)]

    return energy


def kmc_step(sim: KMCSimulation) -> bool:
    """
    Select and execute a kmc event via the Gilespie algo.
    This is the same method for choosing an event in spparks.
    Returns: boolean success flag
    """

    events = get_available_events(sim)
    if not events:
        return False

    # pull a random number in [0,1)
    r = random.random()

    # add all rates and compute a target in [0,R)
    # rates = {event.rate: event for event in events}
    rate_list = sum(event.rate for event in events)
    total_rate = np.sum(rate_list)
    cdf = np.cumulative_sum(rate_list)
    target = total_rate * r

    # Pick an event out and execute it
    event_ind = np.searchsorted(cdf, target)
    _execute_event(sim, events[event_ind])

    ## update simulation time and event counter
    sim.simulation_time += -np.log(r) / total_rate
    sim.event_count += 1

    return True


def get_sim_stats(sim: KMCSimulation) -> Dict[str, Any]:
    num_antisites: int = 0
    for site in sim.lattice_state.sitelist:
        sublattice: ZBUCell = site.sublattice
        if sublattice == ZBUCell.FCC_1:
            if site.occupation_type == OccupationType.AS:
                num_antisites += 1
        if sublattice == ZBUCell.FCC_2:
            if site.occupation_type == OccupationType.GA:
                num_antisites += 1

    height_map = calculate_height_map(sim)
    mean_height = np.mean(height_map)
    rms_roughness = np.sqrt(np.mean((np.max(height_map) - mean_height) ** 2))

    stats: Dict[str, Any] = {
        "num_antisites": num_antisites,
        "mean heist": mean_height,
        "rms roughness": rms_roughness,
    }
    return stats


def calculate_height_map(sim: KMCSimulation):
    """
    Tosses out the current height map and rebuilds it based on the current lattice state
    """
    lat = sim.lattice_state
    cols: Dict[Tuple[int, int], List[int]] = {}
    height_map: List[List[int]] = []
    for site in lat.sitelist:
        col_x, col_y = _get_col_inx(sim, site)
        cols[(col_x, col_y)].append(site.id)

    # Sort the site ids in each column by z coordinate
    for (col_x, col_y), site_ids in cols.items():
        sorted_ids = sorted(site_ids, key=lambda sid: lat.sites[sid].location[2])

        # Height goes by number of occupied sites, allows for vacancies and terraces
        occupied = sum(
            1 for sid in sorted_ids if lat.sites[sid].occupation_type.value != 0
        )
        height_map[col_x][col_y] = occupied

    return height_map


def _get_col_inx(sim: KMCSimulation, site: LatticeSite):
    """
    This explicitly only works for a ZB lattice.
    Returns the column of a lattice site relevant for the height map.
    """
    ix, iy, _ = site.cell
    x, y, _ = site.location

    sub_col_x = (x % 1) * 4
    col_x = ix * 4 + sub_col_x

    sub_col_y = (y % 1) * 4
    col_y = iy * 4 + sub_col_y

    return col_x, col_y


def run(sim: KMCSimulation, max_events: int, stats_interval: int):
    for i in range(max_events):
        if not kmc_step(sim):
            print(f"Stopped after {i} steps/events")

        if i % stats_interval == 0:
            statistics: Dict[str, Any] = get_sim_stats(sim)
            for name, value in statistics.items():
                print(f"{name}, {value}")

    pass


def figure_dump_simulation(sim: KMCSimulation):
    pass


def main():
    pass
    # process_list = (
    #    Process(default_prefactor, 0.0, "ga_adsorption"),
    #    Process(default_prefactor, 1.1, "ga_diffusion_110"),
    #    Process(default_prefactor, 0.5, "ga_diffusion_011"),
    # )
