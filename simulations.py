
from typing import Callable, Tuple, List, Sequence, Dict

from main.compiling.noise.models.CircuitLevelNoise import CircuitLevelNoise
from main.compiling.noise.models.NoiseModel import NoiseModel

from twisted_fault_equivalent_floquetified_colour_code import twisted_fault_equivalent_memory_experiment
from naive_floquetified_colour_code import naive_memory_experiment
from fault_equivalent_floquetified_colour_code import fault_equivalent_memory_experiment
from fault_equivalent_floquetified_colour_code_TEST import (
    fault_equivalent_memory_experiment_TEST,
    fault_equivalent_measurement_metadata,
    SingleDepolarizingNoise,
    MeasurementInfo,
)
from vanilla_colour_code import build_vanilla_colour_code_circuit
from datetime import datetime
from pathlib import Path
import chromobius
import stim
import matplotlib.pyplot as plt
import numpy as np
import json
import cProfile
import pstats
import io
from ldpc import BpOsdDecoder
import stim

def extract_index_from_target(target: stim.DemTarget):
    s = str(target)
    if s == "^":
        return None, None
    if s.startswith("D"):
        return "D", int(s[1:])
    elif s.startswith("L"):
        return "L", int(s[1:])
    else:
        raise ValueError(f"Unknown target: {s}")


def dem_to_parity_check_matrix(dem: stim.DetectorErrorModel):
    # Returns one parity check matrix for the detectors and one for the observables.
    detector_data = []
    observable_data = []
    for instruction in dem:
        if instruction.type == 'error':
            detectors = []
            observables = []

            for target in instruction.targets_copy():
                kind, idx = extract_index_from_target(target)
                if kind == "D":
                    detectors.append(idx)
                elif kind == "L":
                    observables.append(idx)
            detector_data.append(detectors)
            observable_data.append(observables)

    max_det = max((max(dets, default=-1) for dets in detector_data), default=-1)
    max_obs = max((max(obs, default=-1) for obs in observable_data), default=-1)

    num_det_cols = max_det + 1
    num_obs_cols = max_obs + 1
    detector_matrix = np.zeros((len(detector_data), num_det_cols), dtype=np.uint8)
    observable_matrix = np.zeros((len(observable_data), num_obs_cols), dtype=np.uint8)

    for i, detectors in enumerate(detector_data):
        for d in detectors:
            detector_matrix[i, d] = 1
    for i, observables in enumerate(observable_data):
        for o in observables:
            observable_matrix[i, o] = 1

    return detector_matrix, observable_matrix


def _extract_fault_equivalent_measurement_suggestions(
        metadata: List[MeasurementInfo],
        max_items: int = 5) -> List[MeasurementInfo]:
    """Heuristic selection of interesting measurement locations to probe.

    Args:
        metadata: Ordered list of measurement metadata.
        max_items: Maximum number of suggestions to return.
    """
    if not metadata:
        return []

    def first_with_basis(basis: str) -> MeasurementInfo:
        for info in metadata:
            if info.basis == basis:
                return info
        return metadata[0]

    def last_with_basis(basis: str) -> MeasurementInfo:
        for info in reversed(metadata):
            if info.basis == basis:
                return info
        return metadata[-1]

    suggestions = [
        first_with_basis("Z"),
        first_with_basis("X"),
        last_with_basis("Z"),
    ]

    # Deduplicate while preserving order.
    unique: Dict[int, MeasurementInfo] = {}
    for info in suggestions:
        unique.setdefault(info.index, info)
    ordered = list(unique.values())
    return ordered[:max_items]


def run_single_fault_fault_equivalent_analysis(
        indices: Sequence[int] | None = None,
        max_suggestions: int = 5):
    """Inject single depolarising faults at selected measurements and report logical rates."""
    distance = 4
    tiles_width = 3 * distance
    tiles_height = 3 * distance
    total_rounds = 48
    shots = 200

    metadata = fault_equivalent_measurement_metadata(tiles_width, tiles_height, total_rounds)
    print(f"[Single-fault analysis] total measurements: {len(metadata)}")

    if indices is None:
        targets = _extract_fault_equivalent_measurement_suggestions(metadata, max_items=max_suggestions)
    else:
        targets = []
        for idx in indices:
            if 0 <= idx < len(metadata):
                targets.append(metadata[idx])
            else:
                print(f"  ! Skipping out-of-range measurement index {idx}.")

    if not targets:
        print("No measurement locations selected for analysis.")
        return

    for info in targets:
        config = SingleDepolarizingNoise(measurement_index=info.index, probability=0.05)
        print(f"  -> Injecting at {config.describe(metadata)}")

        circuit = fault_equivalent_memory_experiment_TEST(
            tiles_width,
            tiles_height,
            total_rounds,
            config)

        dem = circuit.detector_error_model(
            decompose_errors=True,
            ignore_decomposition_failures=True,
            approximate_disjoint_errors=True)
        decoder = chromobius.compile_decoder_for_dem(dem)
        sampler = circuit.compile_detector_sampler()
        dets, obs = sampler.sample(
            shots=shots,
            separate_observables=True,
            bit_packed=True)

        # unpack observable samples
        num_observables = circuit.num_observables
        actual_obs = np.unpackbits(
            obs,
            axis=1,
            bitorder='little')[:, :num_observables]

        predicted_obs = decoder.predict_obs_flips_from_dets_bit_packed(dets)
        if predicted_obs.ndim == 1:
            predicted_obs = predicted_obs[np.newaxis, :]
        predicted_obs = np.unpackbits(
            predicted_obs,
            axis=1,
            bitorder='little')[:, :num_observables]

        logical_failures = np.count_nonzero(np.any(actual_obs != predicted_obs, axis=1))
        logical_rate = logical_failures / shots
        print(f"     Logical failures: {logical_failures}/{shots}  (rate={logical_rate:.4g})")


def _strip_initial_round_noise(
        circuit: stim.Circuit,
        measurements_to_skip: int) -> stim.Circuit:
    """Remove noise instructions affecting the initial measurement round."""
    noise_prefixes = (
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "CORRELATED_ERROR",
    )
    measurement_ops = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRZ", "MPP"}

    def measurement_groups(inst: stim.CircuitInstruction) -> int:
        if inst.name != "MPP":
            return len(inst.targets_copy())
        targets = inst.targets_copy()
        groups = 0
        for i, target in enumerate(targets):
            if target.is_combiner:
                continue
            if i == 0 or not targets[i - 1].is_combiner:
                groups += 1
        return groups

    measurement_count = 0
    new_circuit = stim.Circuit()
    for inst in circuit:
        if inst.name in measurement_ops:
            if inst.name == "MPP":
                measurement_count += measurement_groups(inst)
            else:
                measurement_count += len(inst.targets_copy())
            new_circuit.append(inst)
        elif inst.name.startswith(noise_prefixes) and measurement_count <= measurements_to_skip:
            continue
        else:
            new_circuit.append(inst)
    return new_circuit


def build_fault_equivalent_circuit_skip_initial_noise(
        tiles_width: int,
        tiles_height: int,
        total_rounds: int,
        noise_model: NoiseModel,
        rounds_to_skip: int = 1) -> stim.Circuit:
    """Build the fault-equivalent circuit, stripping noise from the first round."""
    circuit = fault_equivalent_memory_experiment(
        tiles_width,
        tiles_height,
        total_rounds,
        noise_model)
    metadata = fault_equivalent_measurement_metadata(tiles_width, tiles_height, total_rounds)
    if total_rounds == 0:
        return circuit
    measurements_per_round = len(metadata) // total_rounds
    return _strip_initial_round_noise(circuit, rounds_to_skip * measurements_per_round)



def simulate(
        create_circuit: Callable[[int, int, int, NoiseModel], stim.Circuit],
        get_tiles_width: Callable[[int], int],
        get_tiles_height: Callable[[int], int],
        get_total_rounds: Callable[[int], int],
        get_noise_model_args: Callable[[float], Tuple[float|None, float|None, float|None, float|None, float|None]],
        shots: int,
        output_filename: str,    
        plot_title: str,
        sizes: Tuple[int, ...] = (1,3),
    ):
    # Decode with BP+OSD
    # DemMatrices = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    # sampler = circuit.compile_detector_sampler()
    # samples,b= sampler.sample(shots=1, separate_observables=True)
    # altpcm = dem_to_parity_check_matrix(dem, include_observables=False)[:-20*L_x*L_y]
    # bp_osd = BpOsdDecoder(
    #         altpcm.T,
    #         error_rate = 0.,
    #         bp_method = 'product_sum',
    #         max_iter = 1,
    #         schedule = 'serial',
    #         osd_method = 'osd_0', #set to OSD_0 for fast solve
    #         osd_order = 0
    #     )
    
    # decoding = bp_osd.decode(samples[0])
    # print("decoding worked?",np.array_equal(altpcm.T @ decoding %2, samples[0].astype(int)))

    # Decode with Chromobius.
    # shots = 1
    # dets, actual_obs_flips = circuit.compile_detector_sampler().sample(
    #     shots=shots,
    #     separate_observables=True,
    #     bit_packed=True,
    # )
    # decoder = chromobius.compile_decoder_for_dem(circuit.detector_error_model())
    # predicted_obs_flips = decoder.predict_obs_flips_from_dets_bit_packed(dets)
    # # count logical errors
    # print(np.count_nonzero(np.any(predicted_obs_flips != actual_obs_flips, axis=1))/shots)   

    simulation_data = {}
    for size in sizes:
        tiles_width = get_tiles_width(size)
        tiles_height = get_tiles_height(size)
        total_rounds = get_total_rounds(size)
        code_data = {
            "tiles_width": tiles_width,
            "tiles_height": tiles_height,
            "total_rounds": total_rounds,
            "simulations": {}
        }
        code_key = str((tiles_width, tiles_height, total_rounds))
        simulation_data[code_key] = code_data

        for physical_error_rate in np.logspace(-5,-1,5):
        # for a in [1]:
            noise_model_args = get_noise_model_args(physical_error_rate)
            noise_model = CircuitLevelNoise(*noise_model_args)
            noise_model_data = {
                "initialisation": noise_model_args[0],
                "idling": noise_model_args[1],
                "one_qubit_gate": noise_model_args[2],
                "two_qubit_gate": noise_model_args[3],
                "measurement": noise_model_args[4],
                "shots": shots,
            }
            code_data["simulations"][physical_error_rate] = noise_model_data

            print(f"Creating code...")
            circuit = create_circuit(tiles_width, tiles_height, total_rounds, noise_model)
            # circuit = stim.Circuit(Path(f"fault_equivalent_floquetified_colour_code_circuit{size}x{size}.txt").read_text())

            try:
                dem = circuit.detector_error_model(
                    decompose_errors=True,
                    ignore_decomposition_failures=True,
                    approximate_disjoint_errors=True)
            except ValueError as dem_error:
                print(f"Warning: {dem_error}. Retrying without decomposition.")
                dem = circuit.detector_error_model(
                    decompose_errors=False,
                    ignore_decomposition_failures=True,
                    approximate_disjoint_errors=True)
            sampler = circuit.compile_detector_sampler()
            print(f"Sampling...")
            detector_samples, observable_samples = sampler.sample(
                shots=shots,
                separate_observables=True,
                bit_packed=True)

            num_observables = circuit.num_observables

            if detector_samples.ndim == 1:
                detector_samples = detector_samples[np.newaxis, :]
            if observable_samples.ndim == 1:
                observable_samples = observable_samples[np.newaxis, :]
            actual_obs = np.unpackbits(observable_samples, axis=1, bitorder='little')[:, :num_observables]
            shots_recorded = actual_obs.shape[0]
            noise_model_data["shots"] = shots_recorded

            try:
                print(f"Decoding with Chromobius...")
                decoder = chromobius.compile_decoder_for_dem(dem)
                predicted_observables = decoder.predict_obs_flips_from_dets_bit_packed(detector_samples)
                if predicted_observables.ndim == 1:
                    predicted_observables = predicted_observables[np.newaxis, :]
                predicted_obs = np.unpackbits(predicted_observables, axis=1, bitorder='little')[:, :num_observables]
                logical_differences = actual_obs != predicted_obs
            except Exception as chromobius_error:
                print(f"Chromobius decoding failed ({chromobius_error}). Falling back to BP+OSD.")
                detector_pcm, observable_pcm = dem_to_parity_check_matrix(dem)
                num_detectors = detector_pcm.shape[1]
                detector_samples_unpacked = np.unpackbits(
                    detector_samples, axis=1, bitorder='little')[:, :num_detectors]
                predicted_obs = np.zeros_like(actual_obs, dtype=np.uint8)
                bp_osd_decoder = BpOsdDecoder(
                    detector_pcm.T,
                    error_rate=0.,
                    bp_method='product_sum',
                    max_iter=1,
                    schedule='serial',
                    osd_method='osd_0',
                    osd_order=0
                )
                for i in range(shots_recorded):
                    error_guess = bp_osd_decoder.decode(detector_samples_unpacked[i].astype(int))
                    error_guess = np.array(error_guess).astype(int)
                    for j in range(num_observables):
                        obs_guess = int(np.dot(observable_pcm[:, j], error_guess) % 2)
                        predicted_obs[i, j] = obs_guess
                logical_differences = actual_obs != predicted_obs

            word_errors = logical_differences.sum(axis=0).astype(int).tolist()
            any_errors = int(np.count_nonzero(np.any(logical_differences, axis=1)))
            
            word_error_rates = [errors/shots_recorded for errors in word_errors]
            any_error_rate = any_errors/shots_recorded

            noise_model_data["word_errors"] = word_errors
            noise_model_data["any_errors"] = any_errors
            noise_model_data["word_error_rates"] = word_error_rates
            noise_model_data["any_error_rate"] = any_error_rate

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / f"{output_filename}_simulations_{now}.json", 'w') as file:
        json.dump(simulation_data, file, indent=4)
    
    for code_key, code_data in simulation_data.items():
        simulations_data = code_data["simulations"]
        physical_error_rates = sorted(simulations_data.keys())
        logical_error_rates = [simulations_data[p]["any_error_rate"] for p in physical_error_rates]
        plt.plot(physical_error_rates, logical_error_rates, marker='o', label=f'{code_key}')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical error rate')
    plt.title(plot_title)
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(data_dir / f"{output_filename}_shots_{shots}_{now}.png")


def toric_colour_code_from_stim(
        tiles_width: int,
        tiles_height: int,
        total_rounds: int,
        _noise_model: NoiseModel) -> stim.Circuit:
    """Load the handcrafted L=3 toric colour code circuit from disk."""
    expected = (3, 3, 6)
    if (tiles_width, tiles_height, total_rounds) != expected:
        raise ValueError(
            f"Stim file only supports tiles={expected[:2]}, rounds={expected[2]}, "
            f"but received {(tiles_width, tiles_height, total_rounds)}")
    stim_path = Path(__file__).resolve().parent / "data" / "toric_color_L3_L3_rounds6.stim"
    return stim.Circuit.from_file(str(stim_path))

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()

    # simulate(
    #     naive_memory_experiment, 
    #     lambda size: 3 * (2 * size - 1),
    #     lambda size: 2 * size - 1,
    #     lambda size: 13 * size,
    #     lambda p: (p, p, p, p, p),
    #     1000, 
    #     "naive_floquetified_colour_code", 
    #     "Naive Floquetified Colour Code")

    simulate( # DER HIER IST LOWKEY BESSER, NOCHMAL ÜBERPRÜFEN 
        fault_equivalent_memory_experiment, 
        lambda size: 3 * size,
        lambda size: 3 * size,
        lambda size: 48 * size,
        lambda p: (p, None, p, p, p),
        1000, 
        "fault_equivalent_floquetified_colour_code_no_idling", 
        "Fault Equivalent Floquetified Colour Code - No Idling",
        (1,))

    # vanilla_round_schedule = {
    #     4: 12,
    #     8: 18,
    #     12: 24,
    #     16: 30,
    #     20: 36
    # }

    # simulate(
    #     lambda tiles_width, tiles_height, total_rounds, noise_model: build_fault_equivalent_circuit_skip_initial_noise(
    #         tiles_width,
    #         tiles_height,
    #         total_rounds,
    #         noise_model,
    #         rounds_to_skip=3),
    #     lambda size: 3 * size,
    #     lambda size: 3 * size,
    #     lambda _: 48,
    #     lambda p: (p, None, p, p, p),
    #     500,
    #     "fault_equivalent_floquetified_colour_code_skip_initial_noise",
    #     "Fault Equivalent Floquetified Colour Code (noise from round 1)",
    #     sizes=(1, 2))

    # run_single_fault_fault_equivalent_analysis()

    # toric_round_schedule = {
    #     # distance: total rounds
    #     4: 36,
    #     8: 48,
    #     12: 60,
    # }

    # simulate(
    #     lambda distance, _, total_rounds, noise_model: build_toric_colour_code_circuit(distance, total_rounds, noise_model),
    #     lambda distance: distance,
    #     lambda distance: distance,
    #     lambda distance: toric_round_schedule[distance],
    #     lambda p: (p, p, p, p, p),
    #     1000,
    #     "toric_colour_code_manual",
    #     "Toric Colour Code (manual builder)",
    #     sizes=tuple(sorted(toric_round_schedule.keys())))

    # simulate(
    #     toric_colour_code_from_stim,
    #     lambda size: size,
    #     lambda size: size,
    #     lambda _: 6,
    #     lambda p: (p, p, p, p, p),
    #     1000,
    #     "toric_colour_code_stim_L3",
    #     "Toric Colour Code (Stim L=3)",
    #     sizes=())

    # simulate(
    #     twisted_fault_equivalent_memory_experiment, 
    #     lambda size: 3 * size,
    #     lambda size: 2,
    #     lambda size: 48,
    #     lambda p: (p, None, p, p, p),
    #     1000, 
    #     "twisted_fault_equivalent_floquetified_colour_code_no_idling", 
    #     "Twisted Fault Equivalent Floquetified Colour Code - No Idling")

    # pr.disable()
    # s = io.StringIO()
    # sortby = pstats.SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.dump_stats("profile_data.prof")
