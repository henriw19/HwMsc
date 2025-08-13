from typing import Tuple, List, Dict

import numpy as np
from main.building_blocks.Check import Check
from main.building_blocks.Qubit import Qubit
from main.building_blocks.detectors.Detector import Detector
from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.Code import Code
from main.compiling.compilers.AncillaPerCheckCompiler import AncillaPerCheckCompiler
from main.compiling.compilers.Compiler import Compiler
from main.compiling.compilers.NativePauliProductMeasurementsCompiler import NativePauliProductMeasurementsCompiler
from main.compiling.noise.models.NoNoise import NoNoise
from main.compiling.syndrome_extraction.extractors.NativePauliProductMeasurementsExtractor import \
    NativePauliProductMeasurementsExtractor
from main.utils.enums import State

from utils import flatten_dicts


class NaiveFloquetifiedColourCode(Code):
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
        # Each tile can be seen as two half-tiles vertically stacked,
        # with X and Z swapped between the two half tiles.
        # A half tile is a 'wonky' rectangle within a 6x5 grid of qubits.
        # Set some initial values about all this geometry.
        self._tiles_width = tiles_width
        self._tiles_height = tiles_height
        self._half_tile_bottom_vector = np.array([6, 2])
        self._half_tile_side_vector = np.array([-2, 8])
        self._tile_bottom_vector = self._half_tile_bottom_vector
        self._tile_side_vector = 2 * self._half_tile_side_vector
        bottom_right = tiles_width * np.array(self._tile_bottom_vector)
        top_left = tiles_height * np.array(self._tile_side_vector)
        top_right = bottom_right + top_left
        wonky_top_right = self.to_wonky_coords(tuple(top_right))
        self.wonky_x_max, self.wonky_y_max = wonky_top_right[0], wonky_top_right[1]

        # The check that qubit (x, y) undergoes in round t+1 is 
        # the check undergone by qubit (x+2, y+6) in round t.
        self.single_round_shift = np.array([2, 6])

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
        return (4 * x + y, 3 * y - x)

    @staticmethod
    def to_straight_coords(wonky_coords: Tuple[int, int]):
        # Convert from 'wonky' coordinates to 'straight' coordinates
        x, y = wonky_coords
        return ((3 * x - y) // 13, (4 * y + x) // 13)

    def wrap_wonky_coords(self, wonky_coords: Tuple[int, int]):
        x, y = wonky_coords
        return (x % self.wonky_x_max, y % self.wonky_y_max)

    def wrap_straight_coords(self, straight_coords: Tuple[int, int]):
        # No way around it - gotta convert to wonky coords and back.
        wonky_coords = self.to_wonky_coords(straight_coords)
        wrapped_coords = self.wrap_wonky_coords(wonky_coords)
        return self.to_straight_coords(wrapped_coords)

    @staticmethod
    def _relative_coords_in_half_tile():
        # 'Straight' coordinates of the 13 qubits in each half-tile,
        # assuming the bottommost one is (0,0)
        return [(0, 0)] + [(x, y) for x in range(0, 6, 2) for y in range(2, 10, 2)]

    def _get_data_qubits(self):
        # Figure out the (straight) coordinates of the data qubits.
        # Take the 13 qubit coords in each half tile and shift them around.
        data_qubit_coordss = [
            tuple(coords +
                  half_tile_x * self._half_tile_bottom_vector +
                  half_tile_y * self._half_tile_side_vector)
            for half_tile_y in range(2 * self._tiles_height)
            for half_tile_x in range(self._tiles_width)
            for coords in self._relative_coords_in_half_tile()]
        data_qubits = {coords: Qubit(coords) for coords in data_qubit_coordss}
        return data_qubits

    def _get_check_schedule(self):
        # Initially, let check_schedule be a list of dicts.
        # Keys for each dict will be the anchors of the corresponding checks.
        check_schedule = []
        # Get all the checks that occur in round 0,
        # one half-tile at a time.
        check_schedule.append(flatten_dicts([
            self._half_tile_checks_round_0((half_tile_x, half_tile_y))
            for half_tile_x in range(self._tiles_width)
            for half_tile_y in range(2 * self._tiles_height)]))
        # Now shift this pattern by (-2, -6) at each timestep.
        # The code has period 13 altogether.
        shift = -self.single_round_shift
        for t in range(1, 13):
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
                    pauli = Pauli(self.data_qubits[shifted_coords], pauli.letter)
                    shifted_paulis[key] = pauli
                shifted_check = Check(shifted_paulis, shifted_anchor)
                shifted_checks[shifted_anchor] = shifted_check
            check_schedule.append(shifted_checks)
        return check_schedule

    def _half_tile_checks_round_0(self, half_tile_coords: Tuple[int, int]) -> Dict[Tuple[int, int], Check]:
        # Get all checks in round 0 for the half-tile with the given coords.
        half_tile_x, half_tile_y = half_tile_coords
        even_half_tile_row = half_tile_y % 2 == 0
        P = PauliLetter('Z') if even_half_tile_row else PauliLetter('X')
        Q = PauliLetter('X') if even_half_tile_row else PauliLetter('Z')

        # First just collect all the actual information needed to create the Check objects.
        # Call these 'raw checks'.
        raw_checks = [
            ([(0, 0), (2, 0)], P),
            ([(0, 2), (-2, 2)], P),
            ([(0, 4), (0, 6)], P),
            ([(2, 2), (4, 2)], Q),
            ([(2, 4), (2, 6)], P),
            ([(2, 8), (4, 8)], P),
            ([(4, 6)], Q)]
        shift = \
            half_tile_x * self._half_tile_bottom_vector + \
            half_tile_y * self._half_tile_side_vector

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
        # Code has period 13.
        # There is one detector per half-tile per round.
        detector_schedule = [[
                self._realise_raw_detector(
                    self._get_raw_detector(),
                    (0, 6),
                    (x, y),
                    round)
                for x in range(self._tiles_width)
                for y in range(2 * self._tiles_height)]
            for round in range(13)]
        return detector_schedule

    def _realise_raw_detector(
            self,
            raw_detector: List[List[Tuple[int, int]]],
            raw_detector_anchor: Tuple[int, int],
            half_tile_coords: Tuple[int, int],
            round: int):
        half_tile_x, half_tile_y = half_tile_coords
        # Now need to shift the whole thing -
        # One shift accounts for which half-tile we're in
        half_tile_shift = \
            half_tile_x * self._half_tile_bottom_vector + \
            half_tile_y * self._half_tile_side_vector
        # Another shift accounts for the round we're in
        round_shift = round * -self.single_round_shift
        shift = half_tile_shift + round_shift

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
        # In the zeroth round (immediately after initialisation),
        # if we initialised in |0> (+1 eigenstate of PauliLetter('Z')),
        # then every ZZ or Z measurement is a detector.
        # Vice versa for initialisation in |+> with XX and X measurements.
        if initial_state == State.Zero:
            half_tile_offset = np.array([0, 0])
        else:
            half_tile_offset = self._half_tile_side_vector
        # Write down anchors of all the checks in one tile that form detectors in round 0.
        tile_check_anchors = [
            tuple(half_tile_offset + (1, 0)), 
            tuple(half_tile_offset + (-1, 2)),
            tuple(half_tile_offset + (0, 5)),
            tuple(half_tile_offset + (2, 5)),
            tuple(half_tile_offset + (3, 8)),
            tuple(half_tile_offset + (2, 14))]
        # Now turn this into check anchors that form detectors in round 0 across the whole code.
        tile_coordss = [
            (x, y)
            for x in range(self._tiles_width)
            for y in range(self._tiles_height)]        
        check_anchors = [
            self.wrap_straight_coords(tuple(
                x * self._tile_bottom_vector + y * self._tile_side_vector + anchor))
            for (x, y) in tile_coordss
            for anchor in tile_check_anchors]

        # Finally, convert these check anchors into single measurement detectors.
        detectors = [
            Detector([(0, self._dict_based_check_schedule[0][check_anchor])], 0, check_anchor)
            for check_anchor in check_anchors]
        initial_detector_schedule = [detectors]

        # In subsequent initial rounds, if we initialised in |0>,
        # then the cut-off green detectors will still be detecting webs,
        # but the cut-off red detectors won't be valid Pauli webs.
        # Vice versa for |+>.
        
        # In the bulk, there is one detector per half-tile per round.
        # So at this initial boundary there is one detector per WHOLE tile.
        # The half-tiles with even y-coord are the ones that contain a single-qubit X measurement.
        # Each of these half-tiles is where a cut-off green detector ends each round.
        # The half-tiles with odd y-coord are the ones that contain a single-qubit Z measurement.
        # Each of these half-tiles is where a cut-off red detector ends each round.
        y_offset = 0 if initial_state == State.Zero else 1
        half_tile_coordss = [
            (x, 2*y + y_offset)
            for x in range(self._tiles_width)
            for y in range(self._tiles_height)
        ]
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
        
    def _get_initial_raw_detectors(self):
        # Detectors in the first three round follow an irregular pattern,
        # due to initialisation of the qubits cutting up the detectors, effectively.
        # We've handled round 0 already, so start at round 1.
        round_1_raw_detector = [
            [(-3, 0), (0, -1), (2, -1)],
            []
        ]        
        round_2_raw_detector = [
            [(-3, 0), (0, -1), (2, -1)],
            [(-3, -2), (1, 2)],
            [(3, -2)],
        ]
        # At this point, there are no more "irregular" detectors - 
        # we just cut off the regular detector at the required round.
        # Important that we stop at round 6! 
        # The first "full" (not cut off) red detectors end at round 7.
        # So we want the usual detector schedule to take over at this point. 
        remaining_rounds_raw_detectors = [
            self._get_raw_detector()[:round + 1]
            for round in range(3, 7)
        ]
        return [
            round_1_raw_detector,
            round_2_raw_detector,
            *remaining_rounds_raw_detectors
        ]

    def _get_raw_detector(self):
        return [
            [(-3, 0), (0, -1), (2, -1)],
            [(-3, -2), (1, 2)],
            [(3, -2), (4, 2)],
            [(4, -1)],
            [(-4, 1)],
            [(-4, -2), (-3, 2)],
            [(-1, -2), (3, 2)],
            [(-2, 1), (0, 1), (3, 0)]]
        

code = NaiveFloquetifiedColourCode(3, 1)
noise_model = NoNoise()
syndrome_extractor = NativePauliProductMeasurementsExtractor()
compiler = NativePauliProductMeasurementsCompiler(noise_model, syndrome_extractor)
initial_state = State.Zero
initial_states = {
    qubit: initial_state
    for qubit in code.data_qubits.values()}
initial_detector_schedule = code.get_initial_detector_schedule(initial_state)
final_measurements = [
    Pauli(qubit, PauliLetter('Z'))
    for qubit in code.data_qubits.values()]
circuit = compiler.compile_to_stim(
    code,
    total_rounds=13,
    initial_states=initial_states,
    initial_detector_schedule=initial_detector_schedule,
    final_measurements=final_measurements)
print(circuit)