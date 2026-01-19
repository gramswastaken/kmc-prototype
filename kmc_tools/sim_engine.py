from enum import IntEnum
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import random
import warnings

from kmc_tools.lattices import (
    LatticeSite,
    ZBLatticeState,
    OccupationType,
    ZBUCell,
    get_site_coordination,
    set_gaas_001_substrate,
)

# kb = sp.constants.k
kb = 8.617e-5
default_prefactor = 1e13


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
            (OccupationType.GA, OccupationType.EMPTY): 0.0,
            (OccupationType.AS, OccupationType.EMPTY): 0.0,
        }
        nx, ny, _ = self.lattice_state.superlattice_dimensions
        self.deposition_area = nx * ny * 2
        self.deposition_rates = {
            OccupationType.GA: 55555555.8,
            OccupationType.AS: 99995555.8,
        }


@dataclass
class Process:
    """Single process with Arrheius rate"""

    name: str


@dataclass
class DiffusionProcess(Process):
    barrier: float
    prefactor: float = default_prefactor

    def rate(self, temperature: float) -> float:
        rate = self.prefactor * np.exp(-self.barrier / (kb * temperature))
        return rate

    # def rate(self, temperature: float) -> float:
    #    with np.errstate(over="raise", divide="raise", invalid="raise"):
    #        try:
    #            rate = self.prefactor * np.exp(-self.barrier / (kb * temperature))
    #            return rate
    #        except FloatingPointError as e:
    #            print(f"{e}:")
    #            print(f"barrier: {self.barrier}")
    #            print(f"prefactor: {self.prefactor}")
    #            print(f"temp: {temperature}")
    #            print(f"value: {-self.barrier / (kb * temperature)}")


@dataclass
class DepositionProcess(Process):
    gr: float = 0
    area: float = 1

    def rate(self) -> float:
        return self.gr / (1 / self.area)


class EventType(IntEnum):
    DEPOSITION = 0
    DIFFUSION = 1


@dataclass
class SimulationEvent:
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


def execute_event(sim: KMCSimulation, event: SimulationEvent) -> bool:
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


def get_available_events_site(
    sim: KMCSimulation, site: LatticeSite
) -> List[SimulationEvent]:
    """
    Builds the total possible event list for every site on the lattice
    """
    eventlist: List[SimulationEvent] = []
    diff_events: List[SimulationEvent] = get_diffusion_events(sim, site)
    dep_events: List[SimulationEvent] = get_deposition_events(sim, site)
    eventlist += diff_events
    eventlist += dep_events
    return eventlist


def get_deposition_events(
    sim: KMCSimulation, site: LatticeSite
) -> List[SimulationEvent]:
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


def kmc_step_global(sim: KMCSimulation) -> bool:
    """
    Select and execute a kmc event via the Gilespie algo.
    This is the same method for choosing an event in spparks.
    Returns: boolean success flag
    """

    events = get_available_events(sim)
    if not events:
        print("events empty")
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
    execute_event(sim, events[event_ind])

    __import__("pprint").pprint(f"events: {events} \n selected: {events[event_ind]}")
    ## update simulation time and event counter
    sim.simulation_time += -np.log(r) / total_rate
    sim.event_count += 1

    return True


def kmc_step_single(sim: KMCSimulation) -> bool:
    """
    Select and execute a kmc event via the Gilespie algo.
    This is the same method for choosing an event in spparks.
    Returns: boolean success flag
    """
    site = random.choice(sim.lattice_state.sitelist)
    events = get_available_events_site(sim, site)
    if not events:
        print("events empty")
        return True

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
    execute_event(sim, events[event_ind])

    __import__("pprint").pprint(f"events: {events} \n selected: {events[event_ind]}")
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
        "mean height": mean_height,
        "rms roughness": rms_roughness,
    }
    return stats


def calculate_height_map(sim: KMCSimulation) -> List[List[int]]:
    """
    Tosses out the current height map and rebuilds it based on the current lattice state
    """
    lat = sim.lattice_state
    cols: Dict[Tuple[int, int], List[int]] = {}
    height_map: List[List[int]] = [
        [0] * (lat.superlattice_dimensions[0] * 4)
        for _ in range(lat.superlattice_dimensions[1] * 4)
    ]

    for site in lat.sitelist:
        col_x, col_y = _get_col_inx(sim, site)
        col_x = int(col_x)
        col_y = int(col_y)

        if (col_x, col_y) not in cols:
            cols[(col_x, col_y)] = []

        cols[(col_x, col_y)].append(site.id)

    for (col_x, col_y), site_ids in cols.items():
        sorted_ids = sorted(site_ids, key=lambda sid: lat.sites[sid].location[2])

        occupied: int = sum(
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
    set_gaas_001_substrate(sim.lattice_state, 2)
    print("run function called")
    for i in range(max_events):
        if not kmc_step_single(sim):
            print(f"Stopped after {i} steps/events")
            break

        if i % stats_interval == 0:
            statistics: Dict[str, Any] = get_sim_stats(sim)
            for name, value in statistics.items():
                print(f"{name}, {value}")
    return


def log_event(event: SimulationEvent, filename: str):
    etype: EventType = event.type

    if etype == EventType.DIFFUSION:
        # idk if this typecase is helpful
        event: DiffusionEvent = event
        rate: float = event.rate
        process: DiffusionProcess = event.process
        name = process.name
        site_a: LatticeSite = event.site_a
        site_b: LatticeSite = event.site_b
        print(f"thing: {name}, from: {site_a}, to: {site_b}, rate: {rate}")
        pass


def figure_dump_simulation(sim: KMCSimulation, filename: str):
    # What info do I want from the figures?
    fig = plt.figure(figsize=(18, 22))

    ax1 = fig.add_subplot(111, projection="3d")
    height_map: List[List[int]] = calculate_height_map(sim)

    numrows = len(height_map)
    numcols = len(height_map[0])
    hx, hy = np.meshgrid(np.arange(0, numrows), np.arange(0, numcols))
    hx = hx.flatten()
    hy = hy.flatten()
    hz = np.zeros(numrows * numcols)
    dhx = np.ones(numrows * numcols)
    dhy = np.ones(numrows * numcols)
    dhz = np.array(height_map).ravel()

    ax1.bar3d(hx, hy, hz, dhx, dhy, dhz)

    plt.savefig(filename)
    return
