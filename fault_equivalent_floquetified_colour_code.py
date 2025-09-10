from typing import Tuple, List, Dict
from pathlib import Path
from datetime import datetime

from main.utils.utils import output_path
import numpy as np
from main.building_blocks.Check import Check
from main.building_blocks.Qubit import Qubit
from main.building_blocks.detectors.Detector import Detector
from main.building_blocks.logical.DynamicLogicalOperator import DynamicLogicalOperator
from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.Code import Code
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.compilers.Compiler import Compiler
from main.compiling.compilers.NativePauliProductMeasurementsCompiler import NativePauliProductMeasurementsCompiler
from main.compiling.noise.models.NoNoise import NoNoise
from main.compiling.syndrome_extraction.extractors.NativePauliProductMeasurementsExtractor import \
    NativePauliProductMeasurementsExtractor
from main.compiling.noise.models.CircuitLevelNoise import CircuitLevelNoise
from main.utils.enums import State
from main.printing.Printer2D import Printer2D

from utils import flatten_dicts


class FaultEquivalentFloquetifiedColourCode(Code):
    def __init__(self, tiles_width: int, tiles_height: int):
        if tiles_width <= 0 or tiles_width % 3 != 0:
            raise ValueError(
                "Width in terms of tiles must be a positive multiple of 3. " +
                f"Instead, got {tiles_width}")
        if tiles_height <= 0:
            raise ValueError(
                "Height in terms of tiles must be a positive number. " +
                f"Instead, got {tiles_height}")
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
        if initial_state not in {State.Zero, State.Plus}:
            raise ValueError(
                f"Can't handle initial state {initial_state}. "
                f"Must be State.Zero or State.Plus.")

        initial_raw_detectors = self._get_initial_raw_detectors()
        for i, raw_detector in enumerate(initial_raw_detectors):
            # Round 0 already handled so rounds here are 1 more than the list index.
            round = i + 1
            round_detectors = [
                self._realise_raw_detector(
                    raw_detector,
                    (0, 6),
                    half_tile_coords,
                    round)
                for half_tile_coords in half_tile_coordss]
            initial_detector_schedule.append(round_detectors)
        return initial_detector_schedule
    
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
        # All of these only span two rounds, so can only be cut off in rounds 0 and 1.
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
                anchored_raw_detectors = even_round_red_anchored_raw_detectors
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
        # Detectors in the first few rounds follow an irregular pattern,
        # due to initialisation of the qubits cutting up the detectors, effectively.
        round_0_raw_detector = [
            [(3, 0)]
        ]
        round_1_raw_detector = [
            [(-3, 0), (0, -1), (2, -1)],
            []]        
        round_2_raw_detector = [
            [(-3, 0), (0, -1), (2, -1)],
            [(-3, -2), (1, 2)],
            [(3, -2), (-2, 0)]]
        
        # Rounds 3 and 4 just use the regular detector but cut off at the required round.
        round_3_raw_detector = self._get_raw_big_detector()[:3 + 1]
        round_4_raw_detector = self._get_raw_big_detector()[:4 + 1]

        # But annoyingly round 5 is irregular!  
        round_5_raw_detector = self._get_raw_big_detector()[:5 + 1]
        round_5_raw_detector[-1].append((2, 0))

        # Round 6 is regular again.
        round_6_raw_detector = self._get_raw_big_detector()[:6 + 1]

        # Important that we stop at round 6! 
        # The first "full" (not cut off) red detectors end at round 7.
        # So we want the usual detector schedule to take over at this point. 
        return [
            round_1_raw_detector,
            round_2_raw_detector,
            round_3_raw_detector,
            round_4_raw_detector,
            round_5_raw_detector,
            round_6_raw_detector]

    def get_final_detectors(
            self, 
            final_measurement_basis: PauliLetter,
            final_checks: Dict[Qubit, Check],
            total_rounds: int
    ) -> List[List[Detector]]:
        if final_measurement_basis not in {PauliLetter('X'), PauliLetter('Z')}:
            raise ValueError(
                f"Can't handle final measurement basis {final_measurement_basis}. "
                f"Must be PauliLetter('X') or PauliLetter('Z').")
        
        # Start by creating the detectors that just consist of a measurement in the final regular round,
        # plus single-qubit measurements in the very final round.

        # Figure out which measurements can form a detector - e.g. if measuring out in Z-basis,
        # then we can form detectors from the ZZ and Z measurements in the final regular round.
        # Vice versa for X-basis.
        if final_measurement_basis == PauliLetter('Z'):
            half_tile_shift = np.array([0, 0])
        else:
            half_tile_shift = self._half_tile_side_vector
        rounds_shift = (total_rounds - 1) * -self.single_round_shift
        shift = half_tile_shift + rounds_shift
        # These are the checks that we can form detectors from.
        anchored_tile_checks = [
            (tuple(shift + (1, 0)), [(-1, 0), (1, 0)]), 
            (tuple(shift + (-1, 2)), [(-1, 0), (1, 0)]),
            (tuple(shift + (0, 5)), [(0, -1), (0, 1)]),
            (tuple(shift + (2, 5)), [(0, -1), (0, 1)]),
            (tuple(shift + (3, 8)), [(-1, 0), (1, 0)]),
            (tuple(shift + (1, 10)), [(-1, 0), (1, 0)]),
            (tuple(shift + (2, 14)), [(0, 0)])] # Single-qubit measurement!
        # Now turn this into check anchors that form detectors in the final round across the whole code.
        tile_coordss = [
            (x, y)
            for x in range(self._tiles_width)
            for y in range(self._tiles_height)]        
        wrapped_anchored_checks = [
            (self.wrap_straight_coords(tuple(x * self._tile_bottom_vector + y * self._tile_side_vector + anchor)), endpoints)
            for (x, y) in tile_coordss
            for anchor, endpoints in anchored_tile_checks]
        
        # Finally, convert these check anchors into measurement detectors.
        final_detectors = []
        # Make a distinction between the very final round (in which we do single-qubit measurements)
        # and the final regular round (in which we measure checks from the usual check schedule).
        relative_final_regular_round = (total_rounds - 1) % self.schedule_length
        relative_final_round = total_rounds % self.schedule_length
        for anchor, endpoints in wrapped_anchored_checks:
            # Get the check that actually will be measured in the final regular round.
            check = self._dict_based_check_schedule[relative_final_regular_round][anchor]
            timed_checks = [(-1, check)]
            # Now add the single-qubit measurement(s) from the very final round.
            for endpoint in endpoints:
                wrapped_endpoint = self.wrap_straight_coords(np.array(anchor) + endpoint)
                data_qubit = self.data_qubits[wrapped_endpoint]
                data_qubit_check = final_checks[data_qubit]
                timed_checks.append((0, data_qubit_check))
            detector = Detector(timed_checks, relative_final_round, anchor)
            final_detectors.append(detector)
        
        # Now we need to create the detectors that start in the remaining final regular rounds.
        # If measuring out in the Z-basis,
        # then the cut-off green detectors will still be detecting webs,
        # but the cut-off red detectors won't be valid Pauli webs.
        # Vice versa if measuring out in the X-basis.
        
        # In the bulk, there is one detector per half-tile per round.
        # So at this final boundary there is one detector per WHOLE tile.
        # The half-tiles with even y-coord are the ones that contain a single-qubit X measurement.
        # Each of these half-tiles is where a cut-off green detector starts each round.
        # The half-tiles with odd y-coord are the ones that contain a single-qubit Z measurement.
        # Each of these half-tiles is where a cut-off red detector starts each round.
        y_offset = 0 if final_measurement_basis == PauliLetter('Z') else 1
        half_tile_coordss = [
            (x, 2*y + y_offset)
            for x in range(self._tiles_width)
            for y in range(self._tiles_height)]
        final_raw_detectors = self._get_final_raw_detectors()
        raw_detector_anchor = (2, 4)
        for i, raw_detector in enumerate(final_raw_detectors):
            detector_start_relative_to_final_round = -(i + 2)
            for x, y in half_tile_coordss:
                half_tile_shift = x * self._tile_bottom_vector + y * self._tile_side_vector
                detector_start = total_rounds + detector_start_relative_to_final_round
                rounds_shift = detector_start * -self.single_round_shift
                shift = half_tile_shift + rounds_shift

                shifted_detector_anchor = self.wrap_straight_coords(tuple(shift + raw_detector_anchor))
                shifted_raw_detector = [[
                        self.wrap_straight_coords(tuple(np.array(shifted_detector_anchor) + check_anchor)) 
                        for check_anchor in round_check_anchors]
                    for round_check_anchors in raw_detector]

                single_qubit_measurement_coordss = shifted_raw_detector[0]
                single_qubit_measurement_timed_checks = [
                    (0, final_checks[self.data_qubits[coords]])
                    for coords in single_qubit_measurement_coordss]
                timed_checks = single_qubit_measurement_timed_checks
                
                remaining_shifted_check_anchors = shifted_raw_detector[1:]
                for j, shifted_check_anchors in enumerate(remaining_shifted_check_anchors):
                    round_relative_to_detector_end = -(j + 1)
                    relative_round = (relative_final_round + round_relative_to_detector_end) % self.schedule_length
                    timed_checks += [
                        (round_relative_to_detector_end, self._dict_based_check_schedule[relative_round][check_anchor])
                        for check_anchor in shifted_check_anchors]

                detector = Detector(timed_checks, relative_final_round, shifted_detector_anchor)
                final_detectors.append(detector)

        return final_detectors

    def _get_final_raw_detectors(self):
        # Completely symmetric to _get_initial_raw_detectors!
        # BUT with the extra complication that we also have to include 
        # the final single-qubit measurements in the detector.
        # Wasn't the case for initial detectors - no need to explicitly 
        # include the initialistion of qubits in the detectors.

        # Detectors in the last few rounds follow an irregular pattern,
        # due to measurements of the qubits cutting up the detectors, effectively.
        # We've handled the last round of the regular check schedule already 
        # (i.e. the one right before we measure the data qubits out), 
        # so start at the penultimate round of the regular check schedule.
        round_minus_2_single_qubit_measurements = [
            (-2, 0), (-2, 2), (0, 0), (0, 2), (2, 0), (4, 0)]
        round_minus_2_raw_detector = [
            round_minus_2_single_qubit_measurements,
            [],
            [(-2, 1), (0, 1), (3, 0)]]

        round_minus_3_single_qubit_measurements = [
            (-4, 2), (-2, -2), (-2, 0), (0, -2), (0, 0), (0, 2), (2, 2), (4, 0), (4, 2)]
        round_minus_3_raw_detector = [
            round_minus_3_single_qubit_measurements,
            [(-3, 2), (2, 0)],
            [(-1, -2), (3, 2)],
            [(-2, 1), (0, 1), (3, 0)]]
            
        # As in _get_initial_raw_detectors, the next two rounds' detectors are "regular"
        # in the sense that we just cut off the regular detector at the required round.
        # But we still need to add the single-qubit measurements to the detectors.
        round_minus_4_single_qubit_measurements = [
            (-4, -2), (-4, 0), 
            (-2, -2), (-2, 0), 
            (0, -2), (0, 0), (0, 2), 
            (2, 0), (2, 2), 
            (4, 0), (4, 2)]
        round_minus_4_raw_detector = [
            round_minus_4_single_qubit_measurements,
            *self._get_raw_big_detector()[-4:]]

        round_minus_5_single_qubit_measurements = [
            (-4, -2), (-4, 0), 
            (-2, -2), (-2, 0), 
            (0, -2), (0, 0), (0, 2), 
            (2, 0), (2, 2), 
            (4, -2), (4, 2)]
        round_minus_5_raw_detector = [
            round_minus_5_single_qubit_measurements,
            *self._get_raw_big_detector()[-5:]]

        # Next round is a tiny bit irregular - have to add an extra single-qubit measurement.
        round_minus_6_single_qubit_measurements = [
            (-4, -2), (-4, 0),
            (-2, -2), 
            (0, -2), (0, 0), (0, 2),
            (2, -2), (2, 0), (2, 2)]
        round_minus_6_raw_detector = [
            round_minus_6_single_qubit_measurements,
            *self._get_raw_big_detector()[-6:]]
        round_minus_6_raw_detector[1].append((-2, 0))

        # As regular as can be again.
        round_minus_7_single_qubit_measurements = [
            (-4, 0),
            (-2, 0), 
            (0, -2), (0, 0),
            (2, -2), (2, 0)]
        round_minus_7_raw_detector = [
            round_minus_7_single_qubit_measurements,
            *self._get_raw_big_detector()[-7:]]
        
        # Important that we stop at round -7! (where round 0 means 
        # the final round in which we do the single qubit measurements).
        # The first "full" (not cut off) red detectors start at round -8.
        # So we want the usual detector schedule to take over at this point. 
        return [
            round_minus_2_raw_detector,
            round_minus_3_raw_detector,
            round_minus_4_raw_detector,
            round_minus_5_raw_detector,
            round_minus_6_raw_detector,
            round_minus_7_raw_detector]

    def get_logical_z_0(self):
        initial_paulis_tile_coordss = [
            (2, 2),
            (4, 2),
            (4, 4),
            (2, 6),
            (2, 8),
            (2, 10),
            (0, 12),
            (0, 14),
            (2, 14),
        ]
        initial_paulis_coordss = [
            tuple(y * self._tile_side_vector + coords)
            for coords in initial_paulis_tile_coordss
            for y in range(self._tiles_height)
        ]
        # Note no wrapping of coordinates needed above. 
        initial_paulis = [
            Pauli(self.data_qubits[coords], PauliLetter("Z"))
            for coords in initial_paulis_coordss
        ]

        def update(round: int) -> List[Check]:
            # Checks to multiply in depends on the parity of the round!
            if round % 2 == 0:
                tile_check_anchors = [
                    (2, 5), 
                    (2, 14)]
            else:
                tile_check_anchors = [
                    tuple(-self.single_round_shift + (6, 7)), 
                    tuple(-self.single_round_shift + (2, 14))]
            # This 2-round pattern then shifts every two rounds.
            shift = (round // 2) * (self._tile_bottom_vector - 2 * self.single_round_shift)
            anchors = [
                self.wrap_straight_coords(
                    tuple(shift + anchor + y * self._tile_side_vector)
                )
                for anchor in tile_check_anchors
                for y in range(self._tiles_height)
            ]
            relative_round = round % self.schedule_length
            checks_to_multiply_in = [
                self._dict_based_check_schedule[relative_round][anchor]
                for anchor in anchors
            ]
            return checks_to_multiply_in

        logical = DynamicLogicalOperator(initial_paulis, update)
        return logical

def fault_equivalent_memory_experiment(tiles_width: int, tiles_height: int, total_rounds: int, physical_error_rate: float):
    code = FaultEquivalentFloquetifiedColourCode(tiles_width, tiles_height)
    p = physical_error_rate
    # noise_model = CircuitLevelNoise(p, p, p, p, p)
    noise_model = NoNoise()
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
    final_detectors = None

    # observables = [code.get_logical_z_0()]
    observables = None

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
    project_root = Path('/Users/teague/Coding/Research/Quantum/HwMsc')
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(project_root / f'printouts/FaultEquivalent/{now}')
    code = FaultEquivalentFloquetifiedColourCode(3, 2)
    printer = Printer2D()
    printer.print_code(code, output_path, print_logicals=False)

# print_check_schedules()
circuit = fault_equivalent_memory_experiment(3, 1, 48, 0.1)
dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
# print(dem)


