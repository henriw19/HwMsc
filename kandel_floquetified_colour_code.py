from typing import Tuple, List, Dict

import numpy as np
from main.building_blocks.Check import Check
from main.building_blocks.Qubit import Qubit
from main.building_blocks.detectors.Detector import Detector
from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.Code import Code
from main.printing.Printer2D import Printer2D

from utils import flatten, flatten_dicts


class NaiveFloquetifiedColourCode(Code):
    def __init__(self, tiles_width: int, tiles_height: int):
        if tiles_width <= 0 or tiles_width % 3 == 0:
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

        # Set the data qubits now so we can refer to them in a sec.
        super().__init__(data_qubits=self._get_data_qubits())

        # Figure out the check and detector schedules (fiddly!)
        check_schedule = self._get_check_schedule()
        detector_schedule = self._get_detector_schedule(check_schedule)

        # Now convert check_schedule to a list of lists rather than list of dicts.
        check_schedule = [
            list(checks_dict.values())
            for checks_dict in check_schedule]
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
        shift = np.array([-2, -6])
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

    def _get_detector_schedule(self, check_schedule: List[Dict[Tuple[int, int], Check]]) -> List[List[Detector]]:
        # Code has period 13.
        # There is one detector per half-tile per round.
        detector_schedule = [[
                self._half_tile_detector((x, y), t, check_schedule)
                for x in range(self._tiles_width)
                for y in range(2 * self._tiles_height)]
            for t in range(13)]
        return detector_schedule

    def _half_tile_detector(
            self,
            half_tile_coords: Tuple[int, int],
            round: int,
            check_schedule: List[Dict[Tuple[int, int], Check]]):
        half_tile_x, half_tile_y = half_tile_coords
        # As with the check schedule, start with a sort of 'raw detector',
        # containing the minimum information required to create a detector.
        # In this case this is just the anchors of the checks involved
        # and the timesteps at which they occur.
        raw_detector = [
            [(-3, 0), (0, -1), (2, -1)],
            [(-3, -2), (1, 2)],
            [(3, -2), (4, 2)],
            [(4, -1)],
            [(-4, 1)],
            [(-4, -2), (-3, 2)],
            [(-1, -2), (3, 2)],
            [(-2, 1), (0, 1), (3, 0)]]
        detector_anchor = (0, 6)
        # One shift accounts for which half-tile we're in
        shift_a = \
            half_tile_x * self._half_tile_bottom_vector + \
            half_tile_y * self._half_tile_side_vector
        # Another shift accounts for the round we're in
        shift_b = round * np.array([-2, -6])
        shift = shift_a + shift_b
        shifted_anchor = self.wrap_straight_coords(tuple(shift + detector_anchor))
        detector_checks = [
            (-t, check_schedule[round-t][self.wrap_straight_coords(tuple(np.array(shifted_anchor) + check_anchor))])
            for t, check_anchors in enumerate(raw_detector)
            for check_anchor in check_anchors]
        detector = Detector(detector_checks, round, shifted_anchor)
        return detector



code = NaiveFloquetifiedColourCode(6, 2)
printer = Printer2D()
printer.print_code(code, 'naive_floquetified_colour_code', )

