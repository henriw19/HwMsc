
from typing import Callable

from kandel_naive_floquetified_colour_code import naive_memory_experiment
from datetime import datetime
from ldpc import BpOsdDecoder
import stim
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm

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


def simulate(
        create_circuit: Callable[[int, int, int, float], stim.Circuit],
        shots: int,
        output_filename: str,    
        plot_title: str,
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

    
    # threshold generator:
    min_size, max_size = 1, 4
    simulation_data = {}

    for size in range(min_size, max_size):
        tiles_width = 3 * size
        tiles_height = size
        total_rounds = 13 * size
        code_data = {
            "tiles_width": tiles_width,
            "tiles_height": tiles_height,
            "total_rounds": total_rounds,
            "simulations": {}
        }
        code_key = str((tiles_width, tiles_height, total_rounds))
        simulation_data[code_key] = code_data

        for physical_error_rate in np.logspace(-5,-1,5):
            physical_error_rate_data = {
                "physical_error_rate": physical_error_rate,
                "shots": shots,
            }
            code_data["simulations"][physical_error_rate] = physical_error_rate_data

            print(f"Creating code...")
            circuit = create_circuit(tiles_width, tiles_height, total_rounds, physical_error_rate)
            dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
            detector_pcm, observable_pcm = dem_to_parity_check_matrix(dem)
            sampler = circuit.compile_detector_sampler()
            print(f"Sampling...")
            detector_samples, observable_samples = sampler.sample(shots=shots, separate_observables=True)

            num_observables = observable_pcm.shape[1]
            word_errors = [0 for _ in range(num_observables)]
            any_errors = 0

            print(f"Decoding...")
            for i in tqdm(range(shots)):
                bp_osd_decoder = BpOsdDecoder(
                    detector_pcm.T,
                    error_rate = 0.,
                    bp_method = 'product_sum',
                    max_iter = 1,
                    schedule = 'serial',
                    osd_method = 'osd_0', #set to OSD_0 for fast solve
                    osd_order = 0
                )
                error_guess = bp_osd_decoder.decode(detector_samples[i])
                any_logical_error = False
                for j in range(num_observables):
                    observables_flipped_guess = sum(observable_pcm[:, j].T * np.array(error_guess).astype(int)) % 2
                    observables_flipped_actual = observable_samples[i][j]
                    logical_error = observables_flipped_actual != observables_flipped_guess
                    word_errors[j] += int(logical_error)
                    any_logical_error = any_logical_error or logical_error
                any_errors += int(any_logical_error)
            
            word_error_rates = [errors/shots for errors in word_errors]
            any_error_rate = any_errors/shots

            physical_error_rate_data["word_errors"] = word_errors
            physical_error_rate_data["any_errors"] = any_errors
            physical_error_rate_data["word_error_rates"] = word_error_rates
            physical_error_rate_data["any_error_rate"] = any_error_rate

    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"data/{output_filename}_simulations_{now}.json", 'w') as file:
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
    plt.savefig(f"data/{output_filename}_shots_{shots}_{now}.png")

simulate(
    naive_memory_experiment, 
    1000, 
    "naive_floquetified_colour_code", 
    "Naive Floquetified Colour Code")