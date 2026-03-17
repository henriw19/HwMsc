from collections import Counter
from math import gcd
from typing import Tuple, List, Dict
from pathlib import Path
from datetime import datetime

import numpy as np
import stim
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

from utils import flatten_dicts


class FaultEquivalentFloquetifiedColourCode(Code):
    def __init__(self, tiles_width: int, tiles_height: int):
        # if tiles_width <= 0 or tiles_width % 3 != 0:
        #     raise ValueError(
        #         "Width in terms of tiles must be a positive multiple of 3. " +
        #         f"Instead, got {tiles_width}")
        # if tiles_height <= 0:
        #     raise ValueError(
        #         "Height in terms of tiles must be a positive integer. " +
        #         f"Instead, got {tiles_height}")

        # Code is made out of tiles.
        # Unlike naive case, tile can sort of be seen as three third-tiles, 
        # rather than two half-tiles.
        # But in naive case there was a clear symmetry between half-tiles - 
        # not so much in this new case.
        # Each third-tile is a 'wonky' rectangle within a 4x6 grid of qubits.
        # Throughout, we double the values of all straight coordinates,
        # so that they're always even, and then we can also refer to 
        # integer coordinates of midpoints of edges between points. 
        
        # Set some initial values about all this geometry.
        self._tiles_width = tiles_width
        self._tiles_height = tiles_height

        self._third_tile_bottom_vector = np.array([6, 2])
        self._third_tile_side_vector = np.array([-2, 10])
        self._tile_bottom_vector = self._third_tile_bottom_vector
        self._tile_side_vector = 3 * self._third_tile_side_vector

        # print(self.to_wonky_coords(tuple(self._tile_bottom_vector)))

        bottom_right = tiles_width * np.array(self._tile_bottom_vector)
        top_left = tiles_height * np.array(self._tile_side_vector)
        top_right = bottom_right + top_left
        wonky_top_right = self.to_wonky_coords(tuple(top_right))
        self.wonky_x_max, self.wonky_y_max = wonky_top_right[0], wonky_top_right[1]

        # The check that qubit (x, y) undergoes in round t+1 is 
        # the check undergone by qubit (x+2, y+12) in round t,
        # but with the roles of X and Z exchanged.
        self.single_round_shift = np.array([2, 12])

        # Set the data qubits now so we can refer to them in a sec.
        super().__init__(data_qubits=self._get_data_qubits())

        # Figure out the check and detector schedules (fiddly!)
        self._dict_based_check_schedule = self._get_check_schedule()
        detector_schedule = self._get_detector_schedule()
        # Now convert check_schedule to a list of lists rather than list of dicts.
        check_schedule = [
            list(checks_dict.values())
            for checks_dict in self._dict_based_check_schedule]
        self.set_schedules(check_schedule, detector_schedule)

    @staticmethod
    def to_wonky_coords(straight_coords: Tuple[int, int]):
        # Convert from 'straight' coordinates to 'wonky' coordinates
        x, y = straight_coords
        return (5 * x + y, 3 * y - x)

    @staticmethod
    def to_straight_coords(wonky_coords: Tuple[int, int]):
        # Convert from 'wonky' coordinates to 'straight' coordinates
        x, y = wonky_coords
        return ((3 * x - y) // 16, (5 * y + x) // 16)

    def wrap_wonky_coords(self, wonky_coords: Tuple[int, int]):
        x, y = wonky_coords
        return (x % self.wonky_x_max, y % self.wonky_y_max)

    def wrap_straight_coords(self, straight_coords: Tuple[int, int]):
        # No way around it - gotta convert to wonky coords and back.
        wonky_coords = self.to_wonky_coords(straight_coords)
        wrapped_coords = self.wrap_wonky_coords(wonky_coords)
        return self.to_straight_coords(wrapped_coords)

    @staticmethod
    def _relative_coords_in_third_tile():
        # 'Straight' coordinates of the 16 qubits in each third-tile,
        # assuming the bottommost one is (0,0)
        return [(0, 0)] + [(x, y) for x in range(0, 6, 2) for y in range(2, 12, 2)]

    def _get_data_qubits(self):
        # Figure out the (straight) coordinates of the data qubits.
        # Take the 16 qubit coords in each half tile and shift them around.
        data_qubit_coordss = [
            tuple(coords +
                  third_tile_x * self._third_tile_bottom_vector +
                  third_tile_y * self._third_tile_side_vector)
            for third_tile_y in range(3 * self._tiles_height)
            for third_tile_x in range(self._tiles_width)
            for coords in self._relative_coords_in_third_tile()]
        data_qubits = {coords: Qubit(coords) for coords in data_qubit_coordss}
        return data_qubits

    def _get_check_schedule(self):
        # Initially, let check_schedule be a list of dicts.
        # Keys for each dict will be the anchors of the corresponding checks.
        check_schedule = []
        # Get all the checks that occur in round 0, one tile at a time.
        check_schedule.append(flatten_dicts([
            self._tile_checks_round_0((tile_x, tile_y))
            for tile_x in range(self._tiles_width)
            for tile_y in range(self._tiles_height)]))
        # Now shift this pattern by the required amount at each timestep,
        # swapping the roles of X and Z each time.
        # The code has period 48 altogether.
        shift = -self.single_round_shift
        for t in range(1, 48):
            shifted_checks = {}
            for check in check_schedule[t-1].values():
                # Shift the anchor.
                shifted_anchor = self.wrap_straight_coords(tuple(shift + check.anchor))
                # Then shift all of the Paulis.
                shifted_paulis = {}
                for key, pauli in check.paulis.items():
                    # `key` here is the vector from the anchor to the qubit.
                    # This won't change, because we shift both the anchor and the qubit coords.
                    coords = pauli.qubit.coords
                    shifted_coords = self.wrap_straight_coords(tuple(shift + coords))
                    new_letter = PauliLetter('X') if pauli.letter.letter == 'Z' else PauliLetter('Z')
                    pauli = Pauli(self.data_qubits[shifted_coords], new_letter)
                    shifted_paulis[key] = pauli
                shifted_check = Check(shifted_paulis, shifted_anchor)
                shifted_checks[shifted_anchor] = shifted_check
            check_schedule.append(shifted_checks)
        return check_schedule

    def _tile_checks_round_0(self, tile_coords: Tuple[int, int]) -> Dict[Tuple[int, int], Check]:
        # Get all checks in round 0 for the tile with the given coords.
        tile_x, tile_y = tile_coords
        # First just collect all the actual information needed to create the Check objects.
        # Call these 'raw checks'.
        raw_checks = [
            ([(0, 0)], PauliLetter('X')),
            ([(2, 2), (2, 4)], PauliLetter('X')),
            ([(2, 6), (4, 6)], PauliLetter('X')),
            ([(0, 6), (2, 8)], PauliLetter('X')),
            ([(0, 8), (2, 10)], PauliLetter('Z')),
            ([(2, 12), (4, 14)], PauliLetter('X')),
            ([(2, 14), (4, 16)], PauliLetter('Z')),
            ([(0, 16), (2, 16)], PauliLetter('Z')),
            ([(2, 18), (2, 20)], PauliLetter('Z')),
            ([(-2, 20)], PauliLetter('Z')),
            ([(-2, 22), (0, 22)], PauliLetter('Z')),
            ([(-6, 24), (-2, 24)], PauliLetter('Z')),
            ([(-6, 26), (-2, 26)], PauliLetter('X')),
            ([(-2, 30), (0, 30)], PauliLetter('X')),
        ]
        shift = \
            tile_x * self._tile_bottom_vector + \
            tile_y * self._tile_side_vector

        checks_dict = {}
        for raw_check in raw_checks:
            coordss, pauli_letter = raw_check
            # Set the check's anchor to be the midpoint of the two data qubits
            # (or just the data qubit itself, for the single qubit measurement).
            # (Anchor only relevant for nicely printing the toric geometry.
            # Has no effect on what's actually measured in the code.
            # We also use it temporarily as a key for the check,
            # for use in determining the detector schedule shortly.)
            anchor = tuple(np.mean(coordss, axis=0).astype(int) + shift)
            paulis = {}
            for coords in coordss:
                qubit_coords = self.wrap_straight_coords(shift + coords)
                qubit = self.data_qubits[qubit_coords]
                pauli = Pauli(qubit, pauli_letter)
                # Calculate the vector from the anchor to the qubit coords.
                # Again, just relevant for printing what's happening.
                from_anchor = tuple(np.array(coords) + shift - anchor)
                paulis[from_anchor] = pauli
            wrapped_anchor = self.wrap_straight_coords(anchor)
            checks_dict[wrapped_anchor] = Check(paulis, wrapped_anchor)
        return checks_dict

    def _get_detector_schedule(self) -> List[List[Detector]]:
        # Code has period 48.
        # There is one of each type of detector per tile per round -
        # the big one derived from the colour code, 
        # then seven types of small ones derived from Ben's fault-equivalent rewrites.
        
        anchored_raw_detectors = [
            self._get_big_raw_detector(),
            self._get_single_qubit_raw_detector(),
            self._get_flat_bottom_SE_raw_detector(),
            self._get_flat_bottom_NW_raw_detector(),
            self._get_flat_bottom_wide_raw_detector(),
            self._get_flat_top_SE_raw_detector(),
            self._get_flat_top_NW_raw_detector(),
            self._get_flat_top_wide_raw_detector(),
        ]

        detector_schedule = [
            [
                self._realise_raw_detector(raw_detector, round_0_anchor, (x, y), round)
                for x in range(self._tiles_width)
                for y in range(self._tiles_height)
                for raw_detector, round_0_anchor in anchored_raw_detectors
            ]
            for round in range(48)
        ]

        return detector_schedule

    def _get_big_raw_detector(self):
        raw_detector_dict = {
            0: [(-3, 0)],
            -1: [(0, -1), (1, 2)],
            -7: [(-3, -2)],
            -8: [(2, -1), (4, 2)],
            -12: [(-3, 1)],
            -17: [(3, -1)],
            -21: [(-4, -2), (-2, 1)],
            -22: [(3, 2)],
            -28: [(-1, -2), (0, 1)],
            -29: [(3, 0)],
        }
        detector_span = 30
        raw_detector = [[] for _ in range(detector_span)]
        for t, anchors in raw_detector_dict.items():
            raw_detector[-t] = anchors
        round_0_anchor = (4, 16)
        return raw_detector, round_0_anchor
        
    def _get_single_qubit_raw_detector(self):
        raw_detector = [
            [(0, 0)],
            [(0, 0)]
        ]
        round_0_anchor = (0, 0)
        return raw_detector, round_0_anchor
        
    def _get_flat_bottom_SE_raw_detector(self):
        raw_detector = [
            [(1, 0)],
            [(0, -1)],
            [(0, 0)],
        ]
        round_0_anchor = (1, 3)
        return raw_detector, round_0_anchor

    def _get_flat_bottom_NW_raw_detector(self):
        raw_detector = [
            [(0, 1)],
            [(-1, 0)],
            [(0, 0)],
        ]
        round_0_anchor = (-1, 29)
        return raw_detector, round_0_anchor
    
    def _get_flat_bottom_wide_raw_detector(self):
        raw_detector = [
            [(1, 0)],
            [(-1, 0)],
            [(0, 0)],
        ]
        round_0_anchor = (0, 16)
        return raw_detector, round_0_anchor

    def _get_flat_top_SE_raw_detector(self):
        raw_detector = [
            [(0, 0)],
            [(1, 0)],
            [(0, -1)],
        ]
        round_0_anchor = (3, 15)
        return raw_detector, round_0_anchor

    def _get_flat_top_NW_raw_detector(self):
        raw_detector = [
            [(0, 0)],
            [(0, 1)],
            [(-1, 0)],
        ]
        round_0_anchor = (1, 9)
        return raw_detector, round_0_anchor

    def _get_flat_top_wide_raw_detector(self):
        raw_detector = [
            [(0, 0)],
            [(1, 0)],
            [(-1, 0)],
        ]
        round_0_anchor = (-4, 26)
        return raw_detector, round_0_anchor
        

    def _realise_raw_detector(
            self,
            raw_detector: List[List[Tuple[int, int]]],
            raw_detector_anchor: Tuple[int, int],
            tile_coords: Tuple[int, int],
            round: int):
        tile_x, tile_y = tile_coords
        # Now need to shift the whole thing -
        # One shift accounts for which tile we're in
        tile_shift = \
            tile_x * self._tile_bottom_vector + \
            tile_y * self._tile_side_vector
        # Another shift accounts for the round we're in
        round_shift = round * -self.single_round_shift
        shift = tile_shift + round_shift

        shifted_anchor = self.wrap_straight_coords(tuple(shift + raw_detector_anchor))
        shifted_raw_detector = [[
                self.wrap_straight_coords(tuple(np.array(shifted_anchor) + check_anchor)) 
                for check_anchor in round_check_anchors]
            for round_check_anchors in raw_detector]

        detector_checks = [
            (-t, self._dict_based_check_schedule[round-t][shifted_check_anchor])
            for t, shifted_check_anchors in enumerate(shifted_raw_detector)
            for shifted_check_anchor in shifted_check_anchors]
        detector = Detector(detector_checks, round, shifted_anchor)
        return detector

    def get_initial_detector_schedule(self, initial_state: State):
        pass
    
    def _get_simpler_initial_detector_schedule(self, initial_state: State):
        allowed_states = [State.Zero, State.Plus]
        if initial_state not in allowed_states:
            raise ValueError(
                f"Can't handle initial state {initial_state}. "
                f"Must be in {allowed_states}.")
        
        # For simpler version, just use the simple cut-off detectors - 
        # no optimising to find extra detectors.
        
        # Big detector spans 30 rounds. 
        # So round 29 is when we can stop and let the usual schedule kick in again.
        tile_coordss = [
            (x, y) 
            for x in range(self._tiles_width) 
            for y in range(self._tiles_height)]
        initial_detector_schedule = [[] for round in range(29)]

        # Populate the initial schedule one detector type at a time.
        # Start with the single-qubit measurements - since these only span 2 rounds,
        # these are only cut-off in round 0, and only form a detector if initialised in |+>.
        round = 0
        raw_detector, round_0_anchor = self._get_single_qubit_raw_detector()
        if initial_state == State.Plus:
            cutoff_raw_detector = raw_detector[:round + 1]
            for tile_coords in tile_coordss:
                detector = self._realise_raw_detector(
                    cutoff_raw_detector, round_0_anchor, tile_coords, round)
                initial_detector_schedule[round].append(detector)
        # For the remaining rounds, add a non-cutoff detector per tile per round.
        for round in range(1, 29):
            for tile_coords in tile_coordss:
                detector = self._realise_raw_detector(
                    raw_detector, round_0_anchor, tile_coords, round)
                initial_detector_schedule[round].append(detector)
        
        # Next handle all of the small types of detectors. 
        # All of these only span three rounds, so can only be cut off in rounds 0 and 1.
        # Some of them form green detectors in even rounds and red detectors in odd rounds, 
        # while the rest form green detectors in odd rounds and red detectors in even rounds.
        even_round_green_anchored_raw_detectors = [
                self._get_flat_top_SE_raw_detector(),
                self._get_flat_top_NW_raw_detector(),
                self._get_flat_bottom_wide_raw_detector(),
            ]
        odd_round_green_anchored_raw_detectors = [
                self._get_flat_bottom_SE_raw_detector(),
                self._get_flat_bottom_NW_raw_detector(),
                self._get_flat_top_wide_raw_detector(),
            ]
        even_round_red_anchored_raw_detectors = odd_round_green_anchored_raw_detectors
        odd_round_red_anchored_raw_detectors = even_round_green_anchored_raw_detectors

        for round in range(0, 2):
            if initial_state == State.Zero:
                if round % 2 == 0:
                    anchored_raw_detectors = even_round_green_anchored_raw_detectors
                else:
                    anchored_raw_detectors = odd_round_green_anchored_raw_detectors
            elif initial_state == State.Plus:
                if round % 2 == 0:
                    anchored_raw_detectors = even_round_red_anchored_raw_detectors
                else:
                    anchored_raw_detectors = odd_round_red_anchored_raw_detectors
            for raw_detector, round_0_anchor in anchored_raw_detectors:
                cutoff_raw_detector = raw_detector[:round + 1]
                for tile_coords in tile_coordss:
                    detector = self._realise_raw_detector(
                        cutoff_raw_detector, round_0_anchor, tile_coords, round)
                    initial_detector_schedule[round].append(detector)
        # For the remaining rounds, for each detector type, 
        # add a non-cutoff detector per tile per round.
        anchored_raw_detectors = \
            even_round_green_anchored_raw_detectors + \
            odd_round_green_anchored_raw_detectors
        for round in range(2, 29):
            for tile_coords in tile_coordss:
                for raw_detector, round_0_anchor in anchored_raw_detectors:
                    detector = self._realise_raw_detector(
                        raw_detector, round_0_anchor, tile_coords, round)
                    initial_detector_schedule[round].append(detector)

        # Finally, handle the big detectors.
        # Big detectors are green in even rounds, red in odd rounds.
        # So if initialised in zero state, we get a cutoff big detector every even round.
        # Else if initialised in plus state, we get a cutoff big detector every odd round.
        start_round = 0 if initial_state == State.Zero else 1
        raw_detector, round_0_anchor = self._get_big_raw_detector()
        for round in range(start_round, 29, 2):
            cutoff_raw_detector = raw_detector[:round + 1]
            for tile_coords in tile_coordss:
                detector = self._realise_raw_detector(
                    cutoff_raw_detector, round_0_anchor, tile_coords, round)
                initial_detector_schedule[round].append(detector)
    
        return initial_detector_schedule

        
    def _get_initial_raw_detectors(self):
        pass

    def get_simpler_final_detectors(
            self, 
            final_measurement_basis: PauliLetter,
            final_checks: Dict[Qubit, Check],
            total_rounds: int
    ) -> List[List[Detector]]:
        allowed_final_measurement_bases = [PauliLetter('X'), PauliLetter('Z')]
        if final_measurement_basis not in allowed_final_measurement_bases:
            raise ValueError(
                f"Can't handle final measurement basis {final_measurement_basis}. "
                f"Must be in {allowed_final_measurement_bases}.")

        final_detectors = []
        tile_coordss = [
            (x, y) 
            for x in range(self._tiles_width) 
            for y in range(self._tiles_height)]

        # Populate the final detectors one detector type at a time.
        # Start with the single-qubit measurements - since these only span 2 rounds,
        # these are only cut-off in the final round.
        # Single-qubit measurement detectors are green if they end in odd rounds 
        # and red if they end in even rounds.
        # So cutoff versions only form a detector if measuring in Z and final regular round is even
        # (so very final round consisting of single qubit measurements is odd)
        # or if measuring in X and final regular round is odd
        # (so very final round consisting of single qubit measurements is even)
        raw_detector, round_0_anchor = self._get_single_qubit_raw_detector()
        make_green_detector = final_measurement_basis == PauliLetter('Z') and total_rounds % 2 == 1
        make_red_detector = final_measurement_basis == PauliLetter('X') and total_rounds % 2 == 0
        if make_green_detector or make_red_detector:
            _, round_0_anchor = self._get_single_qubit_raw_detector()
            rounds_shift = total_rounds * -self.single_round_shift
            final_round_anchor = rounds_shift + round_0_anchor
            for tile_coords in tile_coordss:
                anchor = self.wrap_straight_coords(tuple(final_round_anchor + tile_coords))
                qubit = self.data_qubits[anchor]
                final_regular_check = self._dict_based_check_schedule[total_rounds - 1][anchor]
                very_final_check = final_checks[qubit]
                timed_checks = [
                    (0, very_final_check),
                    (-1, final_regular_check)]
                detector = Detector(timed_checks, 0, anchor)
                final_detectors.append(detector)
                
        # Next handle all of the small types of detectors. 
        # All of these only span three rounds, so can only be cut off in the final and penultimate round.
        # Some of them form green detectors in even rounds and red detectors in odd rounds, 
        # while the rest form green detectors in odd rounds and red detectors in even rounds.
        even_round_green_anchored_raw_detectors = [
                self._get_flat_top_SE_raw_detector(),
                self._get_flat_top_NW_raw_detector(),
                self._get_flat_bottom_wide_raw_detector(),
            ]
        odd_round_green_anchored_raw_detectors = [
                self._get_flat_bottom_SE_raw_detector(),
                self._get_flat_bottom_NW_raw_detector(),
                self._get_flat_top_wide_raw_detector(),
            ]
        even_round_red_anchored_raw_detectors = odd_round_green_anchored_raw_detectors
        odd_round_red_anchored_raw_detectors = even_round_green_anchored_raw_detectors

        for t in range(0, 2):
            round_detector_would_have_ended_in = total_rounds + t
            rounds_cut_off = t + 1
            rounds_shift = round_detector_would_have_ended_in * -self.single_round_shift
            if final_measurement_basis == PauliLetter('Z'):
                if round_detector_would_have_ended_in % 2 == 0:
                    anchored_raw_detectors = even_round_green_anchored_raw_detectors
                else:
                    anchored_raw_detectors = odd_round_green_anchored_raw_detectors
            elif final_measurement_basis == PauliLetter('X'):
                if round_detector_would_have_ended_in % 2 == 0:
                    anchored_raw_detectors = even_round_red_anchored_raw_detectors
                else:
                    anchored_raw_detectors = odd_round_red_anchored_raw_detectors
            for tile_x, tile_y in tile_coordss:    
                for raw_detector, round_0_anchor in anchored_raw_detectors:
                    tile_shift = tile_x * self._tile_bottom_vector + tile_y * self._tile_side_vector
                    final_round_anchor = self.wrap_straight_coords(
                        tuple(rounds_shift + tile_shift + round_0_anchor))
                    cutoff_detector = self._realise_cutoff_detector(
                        raw_detector, final_round_anchor, rounds_cut_off, total_rounds, final_checks)
                    final_detectors.append(cutoff_detector)
        
        # Finally, handle the big detectors. 
        # Big detectors are green if they end in even rounds, red in odd rounds, and span 30 rounds
        # (e.g. if one starts in round t it ends in round t + 29).
        # So if measuring in Z basis, we get a cutoff green detector if 
        # the detector would have ended in an even round, i.e. if it starts in an odd round.
        # Vice versa if measuring in X basis we get a cutoff red detector if 
        # the detector would have ended in an odd round, i.e. if it starts in an even round.
        # Since total_rounds - 1 denotes the final regular round,
        # then cutoff big detectors start in round total_rounds - 1 - 28.
        raw_detector, round_0_anchor = self._get_big_raw_detector()
        potential_cutoff_start_round = total_rounds - 1 - 28
        if final_measurement_basis == PauliLetter('Z'):
            if potential_cutoff_start_round % 2 == 0:
                cutoff_start_round = potential_cutoff_start_round + 1
            else:
                cutoff_start_round = potential_cutoff_start_round
        elif final_measurement_basis == PauliLetter('X'):
            if potential_cutoff_start_round % 2 == 0:
                cutoff_start_round = potential_cutoff_start_round
            else:
                cutoff_start_round = potential_cutoff_start_round + 1
        for round in range(cutoff_start_round, total_rounds, 2):
            # potential_cutoff_start_round is the round in which detectors would have 1 round cut off.
            # So to figure out how many rounds have been cut off in this particular round,
            # we add 1 to the difference between the round and the potential cutoff start round.
            rounds_cut_off = 1 + round - potential_cutoff_start_round
            round_detector_would_have_ended_in = round + 29
            rounds_shift = round_detector_would_have_ended_in * -self.single_round_shift
            for tile_x, tile_y in tile_coordss:    
                tile_shift = tile_x * self._tile_bottom_vector + tile_y * self._tile_side_vector
                final_round_anchor = self.wrap_straight_coords(
                    tuple(rounds_shift + tile_shift + round_0_anchor))
                cutoff_detector = self._realise_cutoff_detector(
                    raw_detector, final_round_anchor, rounds_cut_off, total_rounds, final_checks)
                final_detectors.append(cutoff_detector)
        
        return final_detectors


    def _realise_cutoff_detector(
            self, 
            raw_detector: List[List[Tuple[int, int]]], 
            final_round_anchor: Tuple[int, int], 
            rounds_cut_off: int,
            total_rounds: int,
            final_checks: Dict[Qubit, Check]):
        cutoff_raw_detector = raw_detector[rounds_cut_off:]
        timed_checks = []
        qubits_involved = Counter()
        for rounds_from_end, raw_anchors in enumerate(cutoff_raw_detector):
            round = total_rounds - 1 - rounds_from_end
            relative_round = round % self.schedule_length
            for raw_anchor in raw_anchors:
                anchor = self.wrap_straight_coords(tuple(np.array(final_round_anchor) + raw_anchor))
                check = self._dict_based_check_schedule[relative_round][anchor]
                timed_checks.append((-(rounds_from_end + 1), check))
                check_qubits = [pauli.qubit for pauli in check.paulis.values()]
                qubits_involved.update(check_qubits)
        very_final_qubits = [
            qubit
            for qubit, count in qubits_involved.items()
            if count % 2 == 1 
        ]
        very_final_checks = [final_checks[qubit] for qubit in very_final_qubits]
        timed_checks.extend([(0, check) for check in very_final_checks])
        relative_very_final_round = total_rounds % self.schedule_length
        cutoff_detector = Detector(timed_checks, relative_very_final_round, final_round_anchor)
        return cutoff_detector

    def get_final_detectors(
            self, 
            final_measurement_basis: PauliLetter,
            final_checks: Dict[Qubit, Check],
            total_rounds: int
    ) -> List[Detector]:
        pass

    def _get_final_raw_detectors(self):
        pass

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
        # Logical wraps around the torus in a wonky fashion - 
        # So figure out how many tiles this is!
        covering_space_width = self._tiles_height // (gcd(self._tiles_height, 2 * self._tiles_width))
        covering_space_height = (covering_space_width * (2 * self._tiles_width)) // self._tiles_height
        two_by_two_tiles_needed = (covering_space_height * self._tiles_height) // 2
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

def fault_equivalent_memory_experiment(tiles_width: int, tiles_height: int, total_rounds: int, noise_model: NoiseModel):
    code = FaultEquivalentFloquetifiedColourCode(tiles_width, tiles_height)
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
    output_path = Path(project_root / f'printouts/{now}/FaultEquivalent')
    code = FaultEquivalentFloquetifiedColourCode(6, 4)
    logical_qubit = LogicalQubit(z=code.get_logical_z_0())
    code.logical_qubits = [logical_qubit]
    printer = Printer2D()
    printer.print_code(code, output_path, print_logicals=True)

project_root = Path('/Users/teague/Coding/Research/Quantum/HwMsc')

noise_model = CircuitLevelNoise(0.1, None, 0, 0, 0.1)
sizes = [3,3]
circuit = fault_equivalent_memory_experiment(sizes[0], sizes[1], 48, noise_model)
# print(circuit)
# output_path = Path.cwd() / f"fault_equivalent_floquetified_colour_code_circuit{sizes[0]}x{sizes[1]}.txt"
# output_path.write_text(str(circuit))
# print(f"Wrote circuit to {output_path}")

# circuit = stim.Circuit(Path("fault_equivalent_floquetified_colour_code_circuit6x6.txt").read_text())

# print("graph-like-distance:",len(circuit.shortest_graphlike_error()))

# logical_errors = circuit.search_for_undetectable_logical_errors(
#     dont_explore_detection_event_sets_with_size_above=5,
#     dont_explore_edges_with_degree_above=5,
#     dont_explore_edges_increasing_symptom_degree=False,
#     canonicalize_circuit_errors=False)
# print("code distance upper bound",len(logical_errors))

# for logical_error in logical_errors:
#     print(logical_error)
dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
# with open(project_root / 'fault_equivalent_floquetified_colour_code_dem.txt', 'w') as f:
#     f.write(str(dem))
print(dem)

#3x3:
# graph-like-distance: 39
# code distance upper bound 36

#3x6:
# graph-like-distance: 78
# code distance upper bound 4 #mit max parametern konstant

#6x3:
# error
# code distance upper bound 36

#6x6:
# graph-like-distance: 78

#9x3:

#9x6:
# graph-like-distance: 234
# code distance upper bound 12
