from typing import Tuple, List

import numpy as np
from main.building_blocks.Check import Check
from main.building_blocks.Qubit import Qubit
from main.building_blocks.pauli import Pauli
from main.building_blocks.pauli.PauliLetter import PauliLetter
from main.codes.Code import Code
from main.printing.Printer2D import Printer2D

from utils import flatten


class NaiveFloquetifiedColourCode(Code):
    def __init__(self, tiles_width: int, tiles_height: int):
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
        super().__init__(data_qubits=self.get_data_qubits())

        # Figure out the check schedule (fiddly!)
        self.set_schedules(
            self.get_check_schedule(),
            self.get_detector_schedule())

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

    def get_data_qubits(self):
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

    def get_check_schedule(self):
        check_schedule = []
        # Get all the checks that occur in round 0,
        # one half-tile at a time.
        check_schedule.append(flatten([
            self._half_tile_checks_round_0((half_tile_x, half_tile_y))
            for half_tile_x in range(self._tiles_width)
            for half_tile_y in range(2 * self._tiles_height)]))
        # Now shift this pattern by (-2, -6) at each timestep.
        # The code has period 13 altogether.
        shift = np.array([-2, -6])
        for t in range(1, 13):
            shifted_checks = []
            for check in check_schedule[t-1]:
                # Shift the anchor (anchor only relevant for printing the code).
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
                shifted_checks.append(shifted_check)
            check_schedule.append(shifted_checks)
        return check_schedule

    def _half_tile_checks_round_0(self, half_tile_coords: Tuple[int, int]) -> List[Check]:
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

        checks = []
        for raw_check in raw_checks:
            coordss, pauli_letter = raw_check
            # Set the check's anchor to be the midpoint of the two data qubits
            # (or just the data qubit itself, for the single qubit measurement).
            # (Anchor only relevant for nicely printing the toric geometry.
            # Has no effect on what's actually measured in the code.)
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
            checks.append(Check(paulis, anchor))
        return checks

    def get_detector_schedule(self):
        # Code has period 13
        detector_schedule = []
        for t in range(13):
            # TODO!
            detector_schedule.append([])
        return detector_schedule


code = NaiveFloquetifiedColourCode(6, 2)
printer = Printer2D()
printer.print_code(code, 'naive_floquetified_colour_code', )

