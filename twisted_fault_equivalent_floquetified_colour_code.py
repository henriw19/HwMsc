from collections import Counter
from math import gcd
from typing import Tuple, List, Dict
from pathlib import Path
from datetime import datetime

import numpy as np
from main.building_blocks.Check import Check
from main.building_blocks.Qubit import Qubit
from main.building_blocks.detectors.Detector import Detector
from main.building_blocks.logical.DynamicLogicalOperator import DynamicLogicalOperator
from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.Code import Code, LogicalQubit
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.compilers.Compiler import Compiler
from main.compiling.compilers.NativePauliProductMeasurementsCompiler import NativePauliProductMeasurementsCompiler
from main.compiling.noise.models.NoNoise import NoNoise
from main.compiling.noise.models.NoiseModel import NoiseModel
from main.compiling.syndrome_extraction.extractors.NativePauliProductMeasurementsExtractor import \
    NativePauliProductMeasurementsExtractor
from main.compiling.noise.models.CircuitLevelNoise import CircuitLevelNoise
from main.utils.enums import State
from main.printing.Printer2D import Printer2D
from fault_equivalent_floquetified_colour_code import FaultEquivalentFloquetifiedColourCode

class TwistedFaultEquivalentFloquetifiedColourCode(FaultEquivalentFloquetifiedColourCode):
    def __init__(self, tiles_width: int, tiles_height: int):
        assert tiles_height % 2 == 0
        self.logical_tile_intercept = (tiles_height // 2) % tiles_width
        self.wonky_tile_width = 32
        super().__init__(tiles_width, tiles_height)

    def wrap_wonky_coords(self, wonky_coords: Tuple[int, int]):
        x, y = wonky_coords
        z = y // self.wonky_y_max
        if z != 0:
            test = 0
        x_shifted = x - z * self.logical_tile_intercept * self.wonky_tile_width
        return (x_shifted % self.wonky_x_max, y % self.wonky_y_max)
    
    def get_logical_z_0(self):
        initial_paulis_two_by_two_tile_coordss = [
            (4, 4),
            (4, 8),
            (4, 12),
            (2, 12),
            (4, 14),
            (2, 16),
            (4, 18),
            (2, 20),
            (2, 22),
            (4, 22),
            (2, 26),
            (2, 30),
            (0, 34),
            (2, 34),
            (2, 36),
            (0, 38),
            (0, 40),
            (2, 40),
            (0, 44),
            (0, 48),
            (0, 52),
            (0, 56),
            (-2, 56),
            (0, 58),
            (-2, 60),
            (-2, 62),
            (0, 62)
        ]
        two_by_two_tiles_needed = self._tiles_height // 2
        # Write down the vector by which to shift each two by two tile 
        # to get the next one. 
        two_by_two_tiles_shift = self._tile_bottom_vector + 2 * self._tile_side_vector
        initial_paulis_coordss = [
            self.wrap_straight_coords(tuple(i * two_by_two_tiles_shift + coords))
            for coords in initial_paulis_two_by_two_tile_coordss
            for i in range(two_by_two_tiles_needed)
        ]
        initial_paulis = [
            Pauli(self.data_qubits[coords], PauliLetter("Z"))
            for coords in initial_paulis_coordss
        ]

        def update(round: int) -> List[Check]:
            # Checks to multiply in depends on the parity of the round!
            if round % 2 == 0:
                tile_check_anchors = [
                    (2, 19), 
                    (4, 22)]
            else:
                tile_check_anchors = [
                    (6, -7), 
                    (4, -10)]
            # This 2-round pattern then shifts every two rounds.
            two_round_shift = (round // 2) * (self._tile_bottom_vector - 2 * self.single_round_shift)
            anchors = [
                self.wrap_straight_coords(
                    tuple(two_round_shift + anchor + i * two_by_two_tiles_shift)
                )
                for anchor in tile_check_anchors
                for i in range(two_by_two_tiles_needed)
            ]
            relative_round = round % self.schedule_length
            checks_to_multiply_in = [
                self._dict_based_check_schedule[relative_round][anchor]
                for anchor in anchors
            ]
            return checks_to_multiply_in

        logical = DynamicLogicalOperator(initial_paulis, update)
        return logical


def twisted_fault_equivalent_memory_experiment(tiles_width: int, tiles_height: int, total_rounds: int, noise_model: NoiseModel):
    code = TwistedFaultEquivalentFloquetifiedColourCode(tiles_width, tiles_height)
    syndrome_extractor = NativePauliProductMeasurementsExtractor()
    compiler = NativePauliProductMeasurementsCompiler(noise_model, syndrome_extractor)

    initial_state = State.Zero
    # initial_state = State.Plus
    initial_states = {
        qubit: initial_state
        for qubit in code.data_qubits.values()}
    # initial_detector_schedule = code.get_initial_detector_schedule(initial_state)
    initial_detector_schedule = code._get_simpler_initial_detector_schedule(initial_state)
    # initial_detector_schedule = [[] for _ in range(29)]
    # initial_detector_schedule = None

    final_measurement_basis = PauliLetter('Z')
    final_checks = {
        qubit: Check([Pauli(qubit, final_measurement_basis)])
        for qubit in code.data_qubits.values()}
    # final_detectors = code.get_final_detectors(
    #     final_measurement_basis,
    #     final_checks,
    #     total_rounds)
    final_detectors = code.get_simpler_final_detectors(
        final_measurement_basis,
        final_checks,
        total_rounds)
    # final_detectors = None

    observables = [code.get_logical_z_0()]
    # observables = None

    circuit = compiler.compile_to_stim(
        code,
        total_rounds=total_rounds,
        initial_states=initial_states,
        initial_detector_schedule=initial_detector_schedule,
        final_measurements=final_checks,
        final_detectors=final_detectors,
        observables=observables)
    return circuit

def print_check_schedules():
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(project_root / f'printouts/{now}/TwistedFaultEquivalent')
    code = TwistedFaultEquivalentFloquetifiedColourCode(3, 2)
    logical_qubit = LogicalQubit(z=code.get_logical_z_0())
    code.logical_qubits = [logical_qubit]
    printer = Printer2D()
    printer.print_code(code, output_path, print_logicals=True)

# project_root = Path('/Users/teague/Coding/Research/Quantum/HwMsc')
# circuit = twisted_fault_equivalent_memory_experiment(12, 2, 48, 0.1)
# print(circuit.num_qubits)
# print(len(circuit.shortest_graphlike_error()))