import stim
from stim import Circuit
import numpy as np
from matplotlib import pyplot as plt 
import networkx as nx
import re

def floquetcolour(L_x, L_y, L_t):
    # generate floquetified colour code with periodic boundary conditions (torus)
    
    # map lattice coordinates to numbers 
    rows = L_x * 13
    cols = L_y * 26

    mapping = {}
    vertex_number = 0

    for row in range(rows):
        for col in range(cols):
            mapping[(row, col)] = vertex_number
            vertex_number += 1
    
    measurements = []

    # ancillary qubit:
    a = L_x * 13 * L_y * 26 + 1

    circuit = Circuit()
    def measurex(target_qubits, ancilla):
        if all(isinstance(item, tuple) for item in target_qubits):
            mapped_qubits = [mapping[i] for i in target_qubits]    
        else:
            mapped_qubits = [mapping[target_qubits]] 
        measurements.append([target_qubits,"red"]) 
        return Circuit(f'''
                            H {ancilla}
                            CX {' '.join(f'{ancilla} {x}' for x in mapped_qubits)}
                            H {ancilla}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    def measurez(target_qubits, ancilla):
        if all(isinstance(item, tuple) for item in target_qubits):
            mapped_qubits = [mapping[i] for i in target_qubits]    
        else:
            mapped_qubits = [mapping[target_qubits]]
        measurements.append([target_qubits,"green"])
        return Circuit(f'''
                            CX {' '.join(f'{x} {ancilla}' for x in mapped_qubits)}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    # basis: 0 == z, 1 == x
    measurement_schedule = list()
    for t in range(L_t):
        for y in range(L_y):
            for x in range(L_x):
                
                # variable origin, depending on timestep
                ox = (- 7 * t) % (L_x * 13)
                oy = (- 8 * t) % (L_y * 26)

                measurement_schedule = ([((0,3),(4,2),0),
                                        ((8,1),(12,0),1),

                                        ((1,6),0),
                                        ((5,5),(6,8),0),
                                        ((9,4),(10,7),0),

                                        ((3,12),(7,11),1),
                                        ((11,10),(15,9),0),

                                        ((0,16),(4,15),1),
                                        ((8,14),(12,13),0),

                                        ((1,19),1),
                                        ((5,18),(6,21),1),
                                        ((9,17),(10,20),1),

                                        ((3,25),(7,24),0),
                                        ((11,23),(15,22),1)
                                        ])

                for i in measurement_schedule:
                    if len(i) > 2:
                        target_qubits = tuple(((ox + a + x * 13) % (L_x * 13), (oy + b + y * 26) % (L_y * 26)) for a, b in i[:2])
                    else:
                        target_qubits = ((ox + i[0][0] + x * 13) % (L_x * 13), (oy + i[0][1] + y * 26) % (L_y * 26)) 
    
                    circuit += locals()[f"measure{"x" if i[-1] == 1 else "z"}"](target_qubits, a)

    return circuit,measurements


def detector_finder(measurements, num_periods, L_x, L_y):
    # find detectors in a floquetified colour code

    num_meas_per_timestep = len(measurements) / num_periods
    num_x = 13 * L_x
    num_y = 26 * L_y
    detector_foundation = set()


    for p in range(num_periods):
        current_meas = measurements[int(num_meas_per_timestep * p): int(num_meas_per_timestep * (p + 1))]
        sorted_meas = sorted(current_meas, key=lambda x: (x[0][0][0], x[0][0][1]) if isinstance(x[0][0], tuple) else (x[0][0], x[0][1]))
        
        if p == num_periods-1:
            
            for m in sorted_meas:
                item = m
                if isinstance(item[0], tuple) and isinstance(item[0][0], int):  # Single tuple case
                    x = item[0][0]
                    y = item[0][1]
                elif isinstance(item[0], tuple) and isinstance(item[0][0], tuple):  # Nested tuple case
                    x = item[0][0][0]
                    y = item[0][0][1]
                
                detecting_region_vertices = [(x % num_x,y % num_y),((x+4) % num_x,(y-1) % num_y),((x+8) % num_x,(y-2) % num_y),((x+12) % num_x,(y-3) % num_y),((x+16) % num_x,(y-4) % num_y),  ((x+1) % num_x,(y+3) % num_y),((x+5) % num_x,(y+2) % num_y),((x+9) % num_x,(y+1) % num_y),((x+13) % num_x,y % num_y),((x+17) % num_x,(y-1) % num_y),   ((x+2) % num_x,(y+6) % num_y),((x+6) % num_x,(y+5) % num_y),((x+10) % num_x,(y+4) % num_y),((x+14) % num_x,(y+3) % num_y),((x+18) % num_x,(y+2) % num_y)]
                detecting_region_edges = []
                # add  all edges within the slice:
                for edge in current_meas:
                    if isinstance(edge[0][0], tuple):
                        for vertex1 in detecting_region_vertices:
                            for vertex2 in detecting_region_vertices:
                                if edge[0] == (vertex1,vertex2):
                                    detecting_region_edges.append(edge[0])

                    elif isinstance(edge[0], tuple):
                        for vertex in detecting_region_vertices:
                            if edge[0] == vertex:
                                detecting_region_edges.append(edge[0])


                
                # Iterate over the list of edges and check for the pattern
                for edge in detecting_region_edges:
                    if not isinstance(edge[0], int):
                    # For each edge, check if it matches the pattern
                        x, y = edge[0]  # Unpack the first point
                        if isinstance(edge[1], tuple): 
                            # Check if the pattern matches
                            pattern1 = ((x % num_x, y % num_y), ((x + 1) % num_x, (y + 3) % num_y))
                            pattern2 = (((x + 4) % num_x, (y - 1) % num_y), ((x + 5) % num_x, (y + 2) % num_y))
                            pattern3 = (((x + 8) % num_x, (y - 2) % num_y), ((x + 12) % num_x, (y - 3) % num_y))
                            # If these three edges are in the list, return True
                            if pattern1 in detecting_region_edges and pattern2 in detecting_region_edges and pattern3 in detecting_region_edges and tuple(detecting_region_vertices) not in detector_foundation:
                                detector_foundation.add(tuple(detecting_region_vertices))

                
                # ....

    return 0

def plot_lattice(list):
    # plots a single timestep of the floquet colour code
    
    coordinate_list = [entry[0] for entry in list if isinstance(entry[0], tuple)]
    colors = [entry[1] for entry in list if isinstance(entry[1], str)]
    cols = []
    vertices = set()
    edges = []

    # Extract vertices and edges
    for i,item in enumerate(coordinate_list):
        if isinstance(item, tuple):
            if isinstance(item[0], tuple) and isinstance(item[1], tuple):  # Edge
                edges.append(item)
                cols.append(colors[i])
                vertices.update(item)  # Add both endpoints of the edge
            else:  # Single vertex
                vertices.add(item)

    # Separate x and y coordinates for vertices
    vertex_x = [v[0] for v in vertices]
    vertex_y = [v[1] for v in vertices]

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot edges
    for i,edge in enumerate(edges):
        x_coords = [edge[0][0], edge[1][0]]
        y_coords = [edge[0][1], edge[1][1]]
        plt.plot(x_coords, y_coords, color=cols[i], linewidth=1)  # Edges as blue lines

    # Plot vertices
    plt.scatter(vertex_x, vertex_y, color='0', s=50, label="Vertices")  # Vertices as red dots

    # Add grid for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    # Labels and legend
    plt.title("Lattice with Highlighted Vertices and Edges")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig('lattice')

def addnoise(circuit, faulty_gates = {"H","M","CZ","CX","R"}, single_qubit_noise = 0, multi_qubit_noise = 0):
    noisy_circuit = stim.Circuit()
    for operation in circuit:
        operation = str(operation).split(' ')
        gate = operation[0]
        target_qubits = operation[1:]
        if gate in faulty_gates:
            target_qubits = np.array(target_qubits).astype(int)
            # Apply original gate
            noisy_circuit.append_operation(gate, target_qubits)

            # Apply depolarizing noise after each faulty gate
            if len(target_qubits) > 1:
                for i,qubit in enumerate(target_qubits[::2]):
                    noisy_circuit.append_operation("DEPOLARIZE2", [target_qubits[i * 2], target_qubits[(i * 2 + 1)]], multi_qubit_noise)
            else:
                for target in target_qubits:
                    noisy_circuit.append_operation("DEPOLARIZE1", [target], single_qubit_noise)
        else:
            noisy_circuit += Circuit(f'''{gate} {" ".join(target_qubits)}''')
    return noisy_circuit


def main():
    L_x = 3     # number of horizontal "tiles"
    L_y = 1     # number of vertical "tiles"
    L_t = 7     # number of timesteps

    circuit, measurements = floquetcolour(3,1,1)

    plot_lattice(measurements)

    # detector_finder(measurements, 7)
    # print(repr(circuit))    

if __name__ == "__main__":
    main()