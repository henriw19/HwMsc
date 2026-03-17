from __future__ import annotations

"""
Memory experiment for the toric 6.6.6 colour code:
 - build the stabiliser-measurement circuit
 - insert single-qubit depolarising noise after each measurement
 - run a few rounds at fixed time steps
 - estimate logical error rate from observable flips
 - plot logical error vs. code distance
"""

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import stim

from vanilla_colour_code import build_vanilla_colour_code_circuit
from ldpc import BpOsdDecoder


def dem_to_parity_check_matrix(dem: stim.DetectorErrorModel):
    """Return detector and observable parity-check matrices from a DEM."""
    detector_data = []
    observable_data = []
    for instruction in dem:
        if instruction.type == 'error':
            detectors = []
            observables = []
            for target in instruction.targets_copy():
                s = str(target)
                if s.startswith("D"):
                    detectors.append(int(s[1:]))
                elif s.startswith("L"):
                    observables.append(int(s[1:]))
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


def add_measurement_depolarising_noise(
    circuit: stim.Circuit, p: float
) -> stim.Circuit:
    """Return a copy of `circuit` with DEPOLARIZE1(p) inserted before every measurement."""
    new_circuit = stim.Circuit()
    for inst in circuit:
        targets = inst.targets_copy()
        if inst.name == "M" and targets:
            new_circuit.append("DEPOLARIZE1", targets, [p])
            new_circuit.append(inst.name, targets, inst.gate_args_copy())
        else:
            new_circuit.append(inst.name, targets, inst.gate_args_copy())
    return new_circuit


def estimate_logical_error_rate(
    circuit: stim.Circuit, shots: int
) -> float:
    """Estimate logical error rate from observable flips."""
    dem = circuit.detector_error_model(
        decompose_errors=True,
        ignore_decomposition_failures=True,
        approximate_disjoint_errors=True,
    )
    det_pcm, obs_pcm = dem_to_parity_check_matrix(dem)
    if obs_pcm.shape[1] == 0:
        raise ValueError("Circuit has no observables to track logical error.")
    sampler = circuit.compile_detector_sampler()
    det_samp, obs_samp = sampler.sample(shots=shots, separate_observables=True)
    decoder = BpOsdDecoder(
        det_pcm.T,
        error_rate=0.0,
        bp_method="product_sum",
        max_iter=5,
        schedule="serial",
        osd_method="osd_0",
        osd_order=0,
    )
    logical_errors = 0
    for i in range(shots):
        guess = decoder.decode(det_samp[i])
        pred_obs = (obs_pcm.T @ np.array(guess, dtype=np.uint8)) % 2
        logical_errors += int(np.any(pred_obs != obs_samp[i]))
    return logical_errors / shots


def run_experiment(
    distances: Iterable[int],
    rounds: int,
    p_values: Iterable[float],
    shots: int,
):
    curves = {}
    for d in distances:
        y = []
        for p in p_values:
            base = build_vanilla_colour_code_circuit(distance=d, rounds=rounds)
            noisy = add_measurement_depolarising_noise(base, p)
            p_log = estimate_logical_error_rate(noisy, shots)
            y.append(p_log)
        curves[d] = y
    return curves


def main():
    parser = argparse.ArgumentParser(
        description="Memory experiment with depolarising noise after measurements."
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=[4, 8, 12],
        help="List of code distances.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of stabiliser-measurement rounds (time steps).",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=1e-3,
        help="Minimum depolarising probability applied before each measurement.",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=1e-1,
        help="Maximum depolarising probability applied before each measurement.",
    )
    parser.add_argument(
        "--p-points",
        type=int,
        default=7,
        help="Number of log-spaced points between p-min and p-max.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2000,
        help="Number of Monte Carlo samples per distance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("memory_experiment.png"),
        help="Path to save the plot.",
    )
    args = parser.parse_args()

    distances = args.distances
    p_vals = np.logspace(np.log10(args.p_min), np.log10(args.p_max), args.p_points)
    curves = run_experiment(distances, args.rounds, p_vals, args.shots)

    plt.figure(figsize=(6, 4))
    eps = 1e-6
    for d, ys in curves.items():
        ys_arr = np.array(ys, dtype=float)
        ys_arr = np.where(ys_arr > 0, ys_arr, eps)
        plt.plot(p_vals, ys_arr, "o-", label=f"d={d}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Physical depolarising rate per measurement (p)")
    plt.ylabel("Logical error rate")
    plt.title(f"Memory experiment ({args.rounds} rounds)")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
