import stim
import chromobius
from stim import Circuit
import numpy as np
from matplotlib import pyplot as plt 



def floquetcolour(L_x, L_y, L_t,g):
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
    final_measurements = dict()

    # ancillary qubit:
    a = L_x * 13 * L_y * 26 + 1
    q = int(14 * L_x * L_y * L_t)
    p = 26 * L_x * L_y
    circuit = Circuit()
    measurement_index = -1

    def measurex(target_qubits, ancilla,x,y,final:bool,p):
        nonlocal measurement_index
        measurement_index += 1

        if all(isinstance(item, tuple) for item in target_qubits):
            mapped_qubits = [mapping[i] for i in target_qubits] 
            
        else:
            mapped_qubits = [mapping[target_qubits]] 

        if final:
            final_measurements.setdefault((x, y), []).append(-p)
        else:
            measurements.append([-q,target_qubits,"red"]) 

  
        return Circuit(f'''
                            H {ancilla}
                            CX {' '.join(f'{ancilla} {x}' for x in mapped_qubits)}
                            H {ancilla}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    def measurez(target_qubits, ancilla,x,y,final:bool,p):
        nonlocal measurement_index
        measurement_index += 1
        
        if all(isinstance(item, tuple) for item in target_qubits):
            mapped_qubits = [mapping[i] for i in target_qubits]    
        else:
            mapped_qubits = [mapping[target_qubits]]
        
        if final:
            final_measurements.setdefault((x, y), []).append({target_qubits:-p})
        else:
            measurements.append([-q,target_qubits,"green"])
 
        return Circuit(f'''
                            CX {' '.join(f'{x} {ancilla}' for x in mapped_qubits)}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    # basis: 0 == z, 1 == x
    measurement_schedule = list()
    for t in range(g,L_t):
        # variable origin, depending on timestep         
        ox = (- 7 * t) % (L_x * 13)
        oy = (- 8 * t) % (L_y * 26)
        for y in range(L_y):
            for x in range(L_x):
                
                measurement_schedule = ([((0,3),(4,2),0),
                                        ((8,1),(12,0),1),

                                        ((1,6),1),
                                        ((5,5),(6,8),0),
                                        ((9,4),(10,7),0),

                                        ((3,12),(7,11),1),
                                        ((11,10),(15,9),0),

                                        ((0,16),(4,15),1),
                                        ((8,14),(12,13),0),

                                        ((1,19),0),
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
    
                    circuit += locals()[f"measure{'x' if i[-1] == 1 else 'z'}"](target_qubits, a,x,y,0,p)
                    q -= 1

    #final single qubit z measurements:
    qubit_positions = [(0, 3), (4, 2), (8, 1), (12, 0), (1, 6), (5, 5), (6, 8), (9, 4), (10, 7), (3, 12), (7, 11), (11, 10), (2, 9), (0, 16), (4, 15), (8, 14), (12, 13), (1, 19), (5, 18), (6, 21), (9, 17), (10, 20), (3, 25), (7, 24), (11, 23), (2, 22)]

    for y in range(L_y):
        for x in range(L_x):
            for pos in qubit_positions:
                circuit += measurez(((pos[0] + x * 13) % (L_x * 13), (pos[1] + y * 26) % (L_y * 26)),a,x,y,1,p)
                p -= 1

    return circuit,measurements, final_measurements

def plot_lattice(list, filename,key):
    num_meas = len(list)
    # plots a single timestep of the floquet colour code
    
    coordinate_list = [entry[1] for entry in list if isinstance(entry[1], tuple)]
    colors = [entry[2] for entry in list if isinstance(entry[2], str)]
    cols = []
    vertices = set()
    edges = []
    meas = []

    # Extract vertices and edges
    for i,item in enumerate(coordinate_list):
        if isinstance(item, tuple):
            if isinstance(item[0], tuple) and isinstance(item[1], tuple):  # Edge
                edges.append(item)
                meas.append(item)
                cols.append(colors[i])
                vertices.update(item)  # Add both endpoints of the edge
            else:  # Single vertex
                meas.append(item)
                cols.append(colors[i])
                vertices.add(item)

    # Separate x and y coordinates for vertices
    vertex_x = [v[0] for v in vertices]
    vertex_y = [v[1] for v in vertices]

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot edges
    for i,edge in enumerate(meas):
        if isinstance(edge[0],tuple):
            x_coords = [edge[0][0], edge[1][0]]
            y_coords = [edge[0][1], edge[1][1]]
            plt.plot(x_coords, y_coords, color=cols[i], linewidth=1)  # Edges as blue lines
            # Calculate the midpoint of the edge for labeling
            mid_x = (edge[0][0] + edge[1][0]) / (2)# *(i+1)/10)
            mid_y = (edge[0][1] + edge[1][1]) / 2
            
            # Add the label at the midpoint
            if key == "sorted":
                plt.text(mid_x, mid_y, str(i), color='black', fontsize=8, ha='center', va='center')

            if key == "unsorted":
                plt.text(mid_x, mid_y, str(- num_meas + i), color='black', fontsize=8, ha='center', va='center')
    
    # Plot vertices
    plt.scatter(vertex_x, vertex_y, color='0', s=5, label="Vertices")  # Vertices as red dots

    # Add grid for better visualization
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    # Labels and legend
    plt.title("Lattice with Highlighted Vertices and Edges")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig(filename)
    # plt.show()

def addnoise(circuit, faulty_gates = {"H","M","CZ","CX","R"}, single_qubit_noise = 0, multi_qubit_noise = 0, measurement_noise = 0):
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

def addzdetectors(circuit, measurements,final_measurements, L_x, L_y, L_t):
    # add detectors and observable to the toric floquet colour code
    sliced_measurements = {}

    for t in range(L_t-1, -1, -1): # slice measurement schedule into timesteps
        if t == L_t-1:
            sliced_measurements[t] = (measurements[-14 * L_x * L_y:])
        else: 
            sliced_measurements[t] = (measurements[-14*((L_t - t ))*L_x*L_y:(-(L_t - t - 1)*14*L_x*L_y)] )

    def find_corresponding_meas(dict1, dict2, check = False):   # find measurement number ("-x") corresponding to specific measurement at t
        result = set()
        for key in dict1:  
            if key in dict2:  
                for tuple_list in dict1[key]:
                    if isinstance(tuple_list[0][0], int):
                        anchor = tuple_list[0]
                        for measurements in dict2[key]:
                                number, coords, _ = measurements                      
                                if isinstance(coords[0], tuple):
                                    if anchor in coords:  # Check if the target_tuple exists inside                          
                                        result.add(number)       
                                else:
                                    if anchor == coords:  # Direct comparison
                                        result.add(number)
                    else:
                        for tup in tuple_list:
                            if isinstance(tup[0], tuple):
                                anchor = tup[0]
                            else:
                                anchor = tup
                            for measurements in dict2[key]:
                                number, coords, _ = measurements                      
                                if isinstance(coords[0], tuple):
                                    if anchor in coords:  # Check if the target_tuple exists inside
                                        result.add(number)       
                                else:
                                    if anchor == coords:  # Direct comparison
                                        result.add(number)
        return result
    

    # initialize detectors:
    detectors = [[[{key: set() for key in range(L_t , 0, -1)} for _ in range(L_t+6)] for _ in range(L_y)] for _ in range(L_x)]
    cutoffdetectors = [[[{key: set() for key in range(L_t-9 , L_t+1)} for _ in range(20)] for _ in range(L_y)] for _ in range(L_x)]
    
    # cutoff detectors:
    for t in range(L_t - 1, L_t): 
        for y in range(L_y):
            for x in range(L_x):
                dx = 13 * x
                dy = 26 * y
                Dx = 13 * L_x
                Dy = 26 * L_y
                dtx = 7 * (L_t - 1 - t)
                dty = 8 * (L_t - 1 - t)

                a = t - 7
                if t >= L_t - 7:
                    cutoffdetectors[x][y][a][L_t if t == L_t-1 else t].add(((((-47 + dx + dtx) % Dx, (-47 + dy + dty) % Dy),((-46 + dx + dtx) % Dx, (-44 + dy + dty) % Dy)),
                                                    (((-43 + dx + dtx) % Dx, (-48 + dy + dty) % Dy),((-42 + dx + dtx) % Dx, (-45 + dy + dty) % Dy)),
                                                    (((-39 + dx + dtx) % Dx, (-49 + dy + dty) % Dy),((-35 + dx + dtx) % Dx, (-50 + dy + dty) % Dy))))
                if t >= L_t - 6:
                    cutoffdetectors[x][y][a-1][L_t if t == L_t-1 else t].add(((((-31 + dx + dtx) % Dx, (-38 + dy + dty) % Dy),((-27 + dx + dtx) % Dx, (-39 + dy + dty) % Dy)),
                                                    (((-41 + dx + dtx) % Dx, (-42 + dy + dty) % Dy),((-37 + dx + dtx) % Dx, (-43 + dy + dty) % Dy))))
                if t >= L_t -5:
                    cutoffdetectors[x][y][a-2][L_t if t == L_t-1 else t].add(((((-36 + dx + dtx) % Dx, (-27 + dy + dty) % Dy),((-32 + dx + dtx) % Dx, (-28 + dy + dty) % Dy)),
                                            ((-38+ dx + dtx) % Dx, (-33+ dy + dty) % Dy)))
                if t>= L_t -4: 
                    cutoffdetectors[x][y][a-3][L_t if t == L_t-1 else t].add(((((-29 + dx + dtx) % Dx, (-19 + dy + dty) % Dy),((-30 + dx + dtx) % Dx, (-22 + dy + dty) % Dy))))   
                if t>= L_t -3:
                    cutoffdetectors[x][y][a-4][L_t if t == L_t-1 else t].add(((((-8 + dx + dtx) % Dx, (-21 + dy + dty) % Dy),((-7 + dx + dtx) % Dx, (-18 + dy + dty) % Dy))))
                if t>= L_t -2:
                    cutoffdetectors[x][y][a-5][L_t if t == L_t-1 else t].add(((((-1+ dx + dtx) % Dx, (-13+ dy + dty) % Dy),((-5+ dx + dtx) % Dx, (-12+ dy + dty) % Dy)),
                                           ((1+ dx + dtx) % Dx, (-7+ dy + dty) % Dy)))
                if t>= L_t -1:
                    cutoffdetectors[x][y][a-6][L_t if t == L_t-1 else t].add(((((0  + dx + dtx) % Dx, (3 + dy + dty) % Dy),((4 + dx + dtx) % Dx, (2 + dy + dty) % Dy)),
                                    (((-10 + dx + dtx) % Dx, (-1 + dy + dty) % Dy), ((-6 + dx + dtx) % Dx, (-2 + dy + dty) % Dy))))

    # fully contained & top cutoff detectors:
    for t in range(L_t, 0,-1): 
        for y in range(L_y):
            for x in range(L_x):
                dx = 13 * x
                dy = 26 * y
                Dx = 13 * L_x
                Dy = 26 * L_y
                dtx = 7 * (L_t - 1 - t)
                dty = 8 * (L_t - 1 - t)

                a = L_t - 1 - t # a is detector label

                if t >= L_t:
                    detectors[x][y][a-19][t].add(((((5  + dx ) % Dx, (5  + dy ) % Dy),((6  + dx ) % Dx, (8 + dy ) % Dy)),
                                                    (((9  + dx ) % Dx, (4   + dy ) % Dy),((10  + dx ) % Dx, (7  + dy ) % Dy)),
                                                    (((13  + dx ) % Dx, (3  + dy ) % Dy),((17  + dx ) % Dx, (2  + dy ) % Dy))))
                    
                    detectors[x][y][a-18][t].add((
                                            (((12+ dx ) % Dx, (13 + dy ) % Dy),(0,0)),
                                            (((13+ dx ) % Dx, (16 + dy ) % Dy),(0,0)),
                                            (((16+ dx ) % Dx, (12 + dy ) % Dy),(0,0)),
                                            (((17+ dx ) % Dx, (15 + dy ) % Dy),(0,0)),
                                            (((20+ dx ) % Dx, (11 + dy ) % Dy),(0,0)),
                                            (((24+ dx ) % Dx, (10 + dy ) % Dy),(0,0)),
                                                    ))
                    
                    detectors[x][y][a-17][t].add((
                                            (((16 + dx       ) % Dx, (25  + dy      ) % Dy),(0,0)),
                                            (((24 + dx       ) % Dx, (23  + dy      ) % Dy),(0,0)),
                                            (((28 + dx       ) % Dx, (22  + dy      ) % Dy),(0,0)),
                                            (((32 + dx       ) % Dx, (21  + dy      ) % Dy),(0,0)),
                                            (((31 + dx       ) % Dx, (18  + dy      ) % Dy),(0,0)),
                                            (((23 + dx       ) % Dx, (20  + dy      ) % Dy),(0,0)),
                                            (((22 + dx       ) % Dx, (17  + dy      ) % Dy),(0,0)),
                                            (((27 + dx       ) % Dx, (19  + dy      ) % Dy),(0,0)),
                                            (((18 + dx       ) % Dx, (18  + dy      ) % Dy),(0,0)),
                                            (((19 + dx       ) % Dx, (21  + dy      ) % Dy),(0,0)),
                                            ))
                    
                    detectors[x][y][a-16][t].add((
                                            (((24+7 + dx       ) % Dx, (23+8  + dy      ) % Dy),(0,0)),
                                            (((28+7 + dx       ) % Dx, (22+8  + dy      ) % Dy),(0,0)),
                                            (((27+7 + dx       ) % Dx, (19+8  + dy      ) % Dy),(0,0)),
                                            (((23+7 + dx       ) % Dx, (20+8  + dy      ) % Dy),(0,0)),
                                            (((22+7 + dx       ) % Dx, (17+8  + dy      ) % Dy),(0,0)),
                                            (((18+7 + dx       ) % Dx, (18+8  + dy      ) % Dy),(0,0)),
                                            (((19+7 + dx       ) % Dx, (21+8  + dy      ) % Dy),(0,0)),
                                            (((15+7 + dx       ) % Dx, (22+8  + dy      ) % Dy),(0,0)),
                                            (((14 +7+ dx       ) % Dx, (19+8  + dy      ) % Dy),(0,0)),
                                            (((32+7 + dx       ) % Dx, (21+8  + dy      ) % Dy),(0,0)),
                                            (((31 +7+ dx       ) % Dx, (18+8  + dy      ) % Dy),(0,0)),
                                            ))
                    
                    detectors[x][y][a-15][t].add((
                                            (((24+2*7 + dx       ) % Dx, (23+2*8  + dy      ) % Dy),(0,0)),
                                            (((28+2*7 + dx       ) % Dx, (22+2*8  + dy      ) % Dy),(0,0)),
                                            (((27+2*7 + dx       ) % Dx, (19+2*8  + dy      ) % Dy),(0,0)),
                                            (((23+2*7 + dx       ) % Dx, (20+2*8  + dy      ) % Dy),(0,0)),
                                            (((22+2*7 + dx       ) % Dx, (17+2*8  + dy      ) % Dy),(0,0)),
                                            (((18+2*7 + dx       ) % Dx, (18+2*8  + dy      ) % Dy),(0,0)),
                                            (((19+2*7 + dx       ) % Dx, (21+2*8  + dy      ) % Dy),(0,0)),
                                            (((15+2*7 + dx       ) % Dx, (22+2*8  + dy      ) % Dy),(0,0)),
                                            (((14+2*7+  dx       ) % Dx, (19+2*8  + dy      ) % Dy),(0,0)),
                                            (((32+2*7 + dx       ) % Dx, (21+2*8  + dy      ) % Dy),(0,0)),
                                            (((30+2*7+  dx       ) % Dx, (15+2*8  + dy      ) % Dy),(0,0)),
                                            ))
                    
                    detectors[x][y][a-14][t].add((
                                            (((24+3*7 + dx       ) % Dx, (23+3*8  + dy      ) % Dy),(0,0)),
                                            (((28+3*7 + dx       ) % Dx, (22+3*8  + dy      ) % Dy),(0,0)),
                                            (((27+3*7 + dx       ) % Dx, (19+3*8  + dy      ) % Dy),(0,0)),
                                            (((23+3*7 + dx       ) % Dx, (20+3*8  + dy      ) % Dy),(0,0)),
                                            (((22+3*7 + dx       ) % Dx, (17+3*8  + dy      ) % Dy),(0,0)),
                                            (((18+3*7 + dx       ) % Dx, (18+3*8  + dy      ) % Dy),(0,0)),
                                            (((19+3*7 + dx       ) % Dx, (21+3*8  + dy      ) % Dy),(0,0)),
                                            (((15+3*7 + dx       ) % Dx, (22+3*8  + dy      ) % Dy),(0,0)),
                                            (((14+3*7+  dx       ) % Dx, (19+3*8  + dy      ) % Dy),(0,0)),
                                            (((26+3*7 + dx       ) % Dx, (16+3*8  + dy      ) % Dy),(0,0)),
                                            ))
                
                    detectors[x][y][a-13][t].add((
                                            (((27+4*7 + dx       ) % Dx, (19+4*8  + dy      ) % Dy),(0,0)),
                                            (((23+4*7 + dx       ) % Dx, (20+4*8  + dy      ) % Dy),(0,0)),
                                            (((22+4*7 + dx       ) % Dx, (17+4*8  + dy      ) % Dy),(0,0)),
                                            (((19+4*7 + dx       ) % Dx, (21+4*8  + dy      ) % Dy),(0,0)),
                                            (((15+4*7 + dx       ) % Dx, (22+4*8  + dy      ) % Dy),(0,0)),
                                            (((26+4*7 + dx       ) % Dx, (16+4*8  + dy      ) % Dy),(0,0)),
                                            ))

                if t <= L_t - 1:
                    detectors[x][y][a-13][t].add(((((5 + dx + dtx) % Dx, (5 +  dy + dty) % Dy),((6 +  dx + dtx) % Dx, (8 + dy + dty) % Dy)),
                                    (((9 + dx + dtx) % Dx, (4 + dy + dty) % Dy), ((10  +  dx + dtx) % Dx, (7 + dy + dty) % Dy)),
                                        (((-2 + dx + dtx) % Dx, (10 + dy + dty) % Dy),((2 +  dx + dtx) % Dx, (9 + dy + dty) % Dy)))) 

                    if t >= L_t - 7:
                        detectors[x][y][a-20][t].add(((((5  + dx + dtx) % Dx, (5  + dy + dty) % Dy),((6  + dx + dtx) % Dx, (8 + dy + dty) % Dy)),
                                                        (((9  + dx + dtx) % Dx, (4   + dy + dty) % Dy),((10  + dx + dtx) % Dx, (7  + dy + dty) % Dy)),
                                                        (((13  + dx + dtx) % Dx, (3  + dy + dty) % Dy),((17  + dx + dtx) % Dx, (2  + dy + dty) % Dy))))
                        if t >= L_t - 6:
                            if a != 0: # edge case due to weird pauli web
                                detectors[x][y][a-19][t].add((
                                                            (((11+ dx + dtx) % Dx, (10 + dy + dty) % Dy),((15 + 0*7+ dx + dtx) % Dx, (9 + 0*8+ dy + dty) % Dy)), 
                                                            (((21+ dx + dtx) % Dx, (14 + 0*8+ dy + dty) % Dy),((25 + 0*7+ dx + dtx) % Dx, (13 + 0*8+ dy + dty) % Dy))
                                                            ))
                        if t >= L_t - 5:
                            if a == 0:
                                detectors[x][y][a-18][t].add(( 
                                                            (((16  + dx + dtx) % Dx, (25  + dy + dty) % Dy),((20  + dx + dtx) % Dx, (24 + dy + dty) % Dy)),
                                                            ))
                            else:
                                detectors[x][y][a-18][t].add(( 
                                                            (((16  + dx + dtx) % Dx, (25  + dy + dty) % Dy),((20  + dx + dtx) % Dx, (24 + dy + dty) % Dy)),
                                                            (((14  + dx + dtx) % Dx, (19 + dy + dty) % Dy),(0,0)), 
                                                            ))
                        if t >= L_t - 4:
                            detectors[x][y][a-17][t].add(( 
                                                        (((15 +7  + dx + dtx) % Dx, (22 + 8  + dy + dty) % Dy),((16+7  + dx + dtx) % Dx, (25+8 + dy + dty) % Dy)),
                                                        ))
                        if t >= L_t - 3:
                            detectors[x][y][a-16][t].add(( 
                                                        (((30 +2*7  + dx + dtx) % Dx, (15 + 2*8  + dy + dty) % Dy),((31+2*7  + dx + dtx) % Dx, (18+2*8 + dy + dty) % Dy)),
                                                        ))
                        if t >= L_t - 2:
                            detectors[x][y][a-15][t].add(( 
                                                        (((26  +3*7+ dx + dtx) % Dx, (16  +3*8+ dy + dty) % Dy),((30  +3*7+ dx + dtx) % Dx, (15 +3*8+ dy + dty) % Dy)),
                                                        (((32  +3*7+ dx + dtx) % Dx, (21 +3*8+ dy + dty) % Dy),(0,0)), 
                                                        ))
                        if t >= L_t - 1:
                            detectors[x][y][a-14][t].add(( 
                                                        (((14 +4*7  + dx + dtx) % Dx, (19 + 4*8  + dy + dty) % Dy),((18+4*7  + dx + dtx) % Dx, (18+4*8 + dy + dty) % Dy)),
                                                        (((24 +4*7  + dx + dtx) % Dx, (23 + 4*8  + dy + dty) % Dy),((28+4*7  + dx + dtx) % Dx, (22+4*8 + dy + dty) % Dy)),
                                                        ))

                if t <= L_t - 2:   
                    detectors[x][y][a-14][t].add(((((7 - 1*7  + dx + dtx) % Dx, (11 - 1*8 + dy + dty) % Dy),((11 - 1*7 + dx + dtx) % Dx, (10 -1*8 + dy + dty) % Dy)),
                                    (((-3 - 1*7 + dx + dtx) % Dx, (7 - 1*8 + dy + dty) % Dy), ((1 - 1*7 + dx + dtx) % Dx, (6 -1*8 + dy + dty) % Dy))))
                if t <= L_t - 3: 
                    detectors[x][y][a-15][t].add(((((13- 2*7+ dx + dtx) % Dx, (3 - 2*8+ dy + dty) % Dy),((9 -2*7+ dx + dtx) % Dx, (4-2*8+ dy + dty) % Dy)),
                                           ((15-2*7+ dx + dtx) % Dx, (9-2*8+ dy + dty) % Dy)))
                if t <= L_t - 4: 
                    detectors[x][y][a-16][t].add(((((13-3*7 + dx + dtx) % Dx, (3-3*8 + dy + dty) % Dy),((14-3*7 + dx + dtx) % Dx, (6-3*8 + dy + dty) % Dy))))
                if t <= L_t - 5: 
                    detectors[x][y][a-17][t].add(((((-1-4*7 + dx + dtx) % Dx, (13-4*8 + dy + dty) % Dy),((-2-4*7 + dx + dtx) % Dx, (10-4*8 + dy + dty) % Dy))))
                if t <= L_t - 6: 
                    detectors[x][y][a-18][t].add(((((-1 - 5*7 + dx + dtx) % Dx, (13 - 5*8 + dy + dty) % Dy),((3 - 5*7 + dx + dtx) % Dx, (12 - 5*8 + dy + dty) % Dy)),
                                            ((-3 - 5*7+ dx + dtx) % Dx, (7 - 5*8+ dy + dty) % Dy)))                    
                if t <= L_t - 7: 
                    detectors[x][y][a-19][t].add(((((11 - 6*7 + dx + dtx) % Dx, (10 - 6*8 + dy + dty) % Dy),((15- 6*7 + dx + dtx) % Dx, (9 -6*8 + dy + dty) % Dy)), 
                                                    (((1 - 6*7 + dx + dtx) % Dx, (6 - 6*8 + dy + dty) % Dy),((5 - 6*7 + dx + dtx) % Dx, (5 - 6*8 + dy + dty) % Dy))))
                if t <= L_t - 8: 
                    detectors[x][y][a-20][t].add(((((2 - 7*7 + dx + dtx) % Dx, (9 - 7*8 + dy + dty) % Dy),((3-7*7 + dx + dtx) % Dx, (12-7*8 + dy + dty) % Dy)),
                                                    (((6 - 7*7 + dx + dtx) % Dx, (8 - 7*8 + dy + dty) % Dy),((7 - 7*7 + dx + dtx) % Dx, (11- 7*8 + dy + dty) % Dy)),
                                                    (((10 - 7*7 + dx + dtx) % Dx, (7 - 7*8 + dy + dty) % Dy),((14 - 7*7 + dx + dtx) % Dx, (6 - 7*8 + dy + dty) % Dy))))

    i = 0
    m = []
    for a,xdet in enumerate(detectors): #find measurements for full det
        for b,ydet in enumerate(xdet):
            for w,det in enumerate(ydet):
                i += 1
                p = find_corresponding_meas(det, sliced_measurements)             
                p = {i - 26 * L_x * L_y for i in p} 
                for u in det[L_t]:
                    for uu in u:
                        uu = set(uu)
                        value = []
                        for meas in final_measurements.values():
                            for d in meas:
                                for key, val in d.items():
                                    if key in uu:
                                        value.append(val)
                                        break
                        if value is not None:
                            for v in value:
                                p.add(v)
                if len(p) > 0:
                    m.append(p)

############################ chromobius z-detector annotation
    colours = []
    cycle_length = t * 20

    for outer in range(L_x):
        start = 3 + outer % 3
        sub_cycle = [(start + i) % 3 for i in range(3)]

        for _ in range(L_y):
            for _ in range(2):  # Repeat the t*20 pattern twice
                for i in range(cycle_length):
                    colours.append(sub_cycle[i % 3])
############################
    

    for i, d in enumerate(m):
        d = list(d)     
        circuit += Circuit(f"DETECTOR(0,0,0,{colours[i]}) {' '.join([f'rec[{d[l]}]' for l in range(len(d))])}") 

    # OBSERVABLE:
    observable1 = {key: set() for key in range(L_t + 10)} 

    Dx = 13 * L_x
    Dy = 26 * L_y
    for t in range(0 ,L_t):
        for y in range(L_y):
            if t % 2 == 0: #even pattern
                observable1[t].add((((((9 + (-1) * (t//2)) % Dx),((4 + 26 * y + (-16) * (t//2)) % Dy)),(((10 + (-1) * (t//2)) % Dx), ((7 + 26 * y + (-16) * (t//2)) % Dy))),
                                (((14 + (-1) * (t//2)) % Dx),((19 + 26 * y + (-16) * (t//2)) % Dy))
                                ))
            if t % 2  == 1: #uneven pattern
                observable1[t].add((((((12  + (-1) * (t//2)) % Dx),((0 + 26 * y + (-16) * (t//2)) % Dy)),(((11 + (-1) * (t//2)) % Dx), ((-3 + 26 * y + (-16) * (t//2)) % Dy))),
                                   (((7 + (-1) * (t//2)) % Dx),((11 + 26 * y + (-16) * (t//2)) % Dy))
                                   ))   
            if t == L_t - 1 and t % 2 == 0: #final measurements if even
                observable1[L_t].add((
                                     (((12 + (-1) * (t//2)) % Dx), ((13 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((11 + (-1) * (t//2)) % Dx), ((10 + 26 * y + (-16) * (t//2)) % Dy)) ,
                                     (((9  + (-1) * (t//2)) % Dx), ((4  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((13 + (-1) * (t//2)) % Dx), ((3  + 26 * y + (-16) * (t//2)) % Dy)),

                                     (((10 + (-1) * (t//2)) % Dx), ((20 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((9  + (-1) * (t//2)) % Dx), ((17 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((12 + (-1) * (t//2)) % Dx), ((0  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((8  + (-1) * (t//2)) % Dx), ((1  + 26 * y + (-16) * (t//2)) % Dy)),
                                        )) 
                
            if t == L_t - 1 and t % 2 == 1: #final measurements if uneven
                observable1[L_t].add((
                                     (((10 + (-1) * (t//2)) % Dx), ((20 + 26 * y + (-16) * (t//2)) % Dy)) ,
                                     (((9  + (-1) * (t//2)) % Dx), ((17 + 26 * y + (-16) * (t//2)) % Dy)) ,
                                     (((8  + (-1) * (t//2)) % Dx), ((1  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((11 + (-1) * (t//2)) % Dx), ((-3 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((7  + (-1) * (t//2)) % Dx), ((11 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((12 + (-1) * (t//2)) % Dx), ((13 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((11 + (-1) * (t//2)) % Dx), ((10 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((9  + (-1) * (t//2)) % Dx), ((4  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((13 + (-1) * (t//2)) % Dx), ((3  + 26 * y + (-16) * (t//2)) % Dy))
                                        ))     

    observable2 = {key: set() for key in range(L_t + 10)} 

    Dx = 13 * L_x
    Dy = 26 * L_y
    for t in range(0 ,L_t):
        for y in range(L_y):
            if t % 2 == 0: #even pattern
                observable2[t].add((((((9 + (-1) * (t//2)) % Dx),((4 + 26 * y + (-16) * (t//2)) % Dy)),(((10 + (-1) * (t//2)) % Dx), ((7 + 26 * y + (-16) * (t//2)) % Dy))),
                                (((14 + (-1) * (t//2)) % Dx),((19 + 26 * y + (-16) * (t//2)) % Dy))
                                ))
            if t % 2  == 1: #uneven pattern
                observable2[t].add((((((12  + (-1) * (t//2)) % Dx),((0 + 26 * y + (-16) * (t//2)) % Dy)),(((11 + (-1) * (t//2)) % Dx), ((-3 + 26 * y + (-16) * (t//2)) % Dy))),
                                   (((7 + (-1) * (t//2)) % Dx),((11 + 26 * y + (-16) * (t//2)) % Dy))
                                   ))   
            if t == L_t - 1 and t % 2 == 0: #final measurements if even
                observable2[L_t].add((
                                     (((12 + (-1) * (t//2)) % Dx), ((13 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((11 + (-1) * (t//2)) % Dx), ((10 + 26 * y + (-16) * (t//2)) % Dy)) ,
                                     (((9  + (-1) * (t//2)) % Dx), ((4  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((13 + (-1) * (t//2)) % Dx), ((3  + 26 * y + (-16) * (t//2)) % Dy)),

                                     (((10 + (-1) * (t//2)) % Dx), ((20 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((9  + (-1) * (t//2)) % Dx), ((17 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((12 + (-1) * (t//2)) % Dx), ((0  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((8  + (-1) * (t//2)) % Dx), ((1  + 26 * y + (-16) * (t//2)) % Dy)),
                                        )) 
                
            if t == L_t - 1 and t % 2 == 1: #final measurements if uneven
                observable2[L_t].add((
                                     (((10 + (-1) * (t//2)) % Dx), ((20 + 26 * y + (-16) * (t//2)) % Dy)) ,
                                     (((9  + (-1) * (t//2)) % Dx), ((17 + 26 * y + (-16) * (t//2)) % Dy)) ,
                                     (((8  + (-1) * (t//2)) % Dx), ((1  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((11 + (-1) * (t//2)) % Dx), ((-3 + 26 * y + (-16) * (t//2)) % Dy)),

                                     (((7 + (-1) * (t//2)) % Dx), ((11 + 26 * y + (-16) * (t//2)) % Dy)),

                                     (((12 + (-1) * (t//2)) % Dx), ((13 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((11 + (-1) * (t//2)) % Dx),((10 + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((9  + (-1) * (t//2)) % Dx), ((4  + 26 * y + (-16) * (t//2)) % Dy)),
                                     (((13 + (-1) * (t//2)) % Dx),((3  + 26 * y + (-16) * (t//2)) % Dy))
                                        ))           
            
    m = find_corresponding_meas(observable1, sliced_measurements,False)
    cut_off_obs = []

    for tuple_group in observable1[L_t]:
        for target_tuple in tuple_group:
            for key, list_of_dicts in final_measurements.items():
                for sub_dict in list_of_dicts:
                    if target_tuple in sub_dict:
                        cut_off_obs.append(sub_dict[target_tuple])

    m = list(m)
    m = [i - 26 * L_x * L_y for i in m]
    m += cut_off_obs
    circuit_string = f"""OBSERVABLE_INCLUDE(0) {" ".join(f"rec[{_}]" for _ in m)}"""
    circuit += Circuit(circuit_string)
    return circuit

def addxdetectors(circuit, measurements,final_measurements, L_x, L_y, L_t):
    sliced_measurements = {}

    for t in range(L_t-1, -1, -1): # slice measurement schedule into timesteps
        if t == L_t-1:
            sliced_measurements[t] = (measurements[-14 * L_x * L_y:])
        else: 
            sliced_measurements[t] = (measurements[-14*((L_t - t ))*L_x*L_y:(-(L_t - t - 1)*14*L_x*L_y)] )

    def find_corresponding_meas(dict1, dict2, check = False): 
        result = set()
        for key in dict1:  
            if key in dict2:  
                for tuple_list in dict1[key]:
                    if isinstance(tuple_list[0][0], int):
                        anchor = tuple_list[0]
                        for measurements in dict2[key]:
                                number, coords, _ = measurements                      
                                if isinstance(coords[0], tuple):
                                    if anchor in coords:  # Check if the target_tuple exists inside
                                        result.add(number)       
                                else:
                                    if anchor == coords:  # Direct comparison
                                        result.add(number)
                    else:
                        for tup in tuple_list:
                            if isinstance(tup[0], tuple):
                                anchor = tup[0]
                            else:
                                anchor = tup
                            for measurements in dict2[key]:
                                number, coords, _ = measurements                      
                                if isinstance(coords[0], tuple):
                                    if anchor in coords:  # Check if the target_tuple exists inside
                                        result.add(number)       
                                else:
                                    if anchor == coords:  # Direct comparison
                                        result.add(number)
        return result
    

    # initialize detectors:
    detectors = [[[{key: set() for key in range(L_t -1, 0, -1)} for _ in range(L_t)] for _ in range(L_y)] for _ in range(L_x)]
    cutoffdetectors = [[[{key: set() for key in range(L_t-9 , L_t+1)} for _ in range(20)] for _ in range(L_y)] for _ in range(L_x)]
    
    # cutoff detectors:
    for t in range(L_t - 1, L_t): 
        for y in range(L_y):
            for x in range(L_x):
                dx = 13 * x
                dy = 26 * y + 13
                Dx = 13 * L_x
                Dy = 26 * L_y
                dtx = 7 * (L_t - 1 - t)
                dty = 8 * (L_t - 1 - t)

                a = t - 7
                if t >= L_t - 7:#L_t if t == L_t-1 else t
                    cutoffdetectors[x][y][a][L_t if t == L_t-1 else t].add(((((-47 + dx + dtx) % Dx, (-47 + dy + dty) % Dy),((-46 + dx + dtx) % Dx, (-44 + dy + dty) % Dy)),
                                                    (((-43 + dx + dtx) % Dx, (-48 + dy + dty) % Dy),((-42 + dx + dtx) % Dx, (-45 + dy + dty) % Dy)),
                                                    (((-39 + dx + dtx) % Dx, (-49 + dy + dty) % Dy),((-35 + dx + dtx) % Dx, (-50 + dy + dty) % Dy))))
                if t >= L_t - 6:
                    cutoffdetectors[x][y][a-1][L_t if t == L_t-1 else t].add(((((-31 + dx + dtx) % Dx, (-38 + dy + dty) % Dy),((-27 + dx + dtx) % Dx, (-39 + dy + dty) % Dy)),
                                                    (((-41 + dx + dtx) % Dx, (-42 + dy + dty) % Dy),((-37 + dx + dtx) % Dx, (-43 + dy + dty) % Dy))))
                if t >= L_t -5:
                    cutoffdetectors[x][y][a-2][L_t if t == L_t-1 else t].add(((((-36 + dx + dtx) % Dx, (-27 + dy + dty) % Dy),((-32 + dx + dtx) % Dx, (-28 + dy + dty) % Dy)),
                                            ((-38+ dx + dtx) % Dx, (-33+ dy + dty) % Dy)))
                if t>= L_t -4: 
                    cutoffdetectors[x][y][a-3][L_t if t == L_t-1 else t].add(((((-29 + dx + dtx) % Dx, (-19 + dy + dty) % Dy),((-30 + dx + dtx) % Dx, (-22 + dy + dty) % Dy))))   
                if t>= L_t -3:
                    cutoffdetectors[x][y][a-4][L_t if t == L_t-1 else t].add(((((-8 + dx + dtx) % Dx, (-21 + dy + dty) % Dy),((-7 + dx + dtx) % Dx, (-18 + dy + dty) % Dy))))
                if t>= L_t -2:
                    cutoffdetectors[x][y][a-5][L_t if t == L_t-1 else t].add(((((-1+ dx + dtx) % Dx, (-13+ dy + dty) % Dy),((-5+ dx + dtx) % Dx, (-12+ dy + dty) % Dy)),
                                           ((1+ dx + dtx) % Dx, (-7+ dy + dty) % Dy)))
                if t>= L_t -1:
                    cutoffdetectors[x][y][a-6][L_t if t == L_t-1 else t].add(((((0  + dx + dtx) % Dx, (3 + dy + dty) % Dy),((4 + dx + dtx) % Dx, (2 + dy + dty) % Dy)),
                                    (((-10 + dx + dtx) % Dx, (-1 + dy + dty) % Dy), ((-6 + dx + dtx) % Dx, (-2 + dy + dty) % Dy))))

    # fully contained & bottom cutoff detectors:
    for t in range(L_t, 0,-1): 
        for y in range(L_y):
            for x in range(L_x):
                dx = 13 * x
                dy = 26 * y + 13
                Dx = 13 * L_x
                Dy = 26 * L_y
                dtx = 7 * (L_t - 1 - t)
                dty = 8 * (L_t - 1 - t)

                a = L_t - 1 - t # a is detector label

                if t <= L_t - 1:
                    detectors[x][y][a-7][t].add(((((5 + dx + dtx) % Dx, (5 +  dy + dty) % Dy),((6 +  dx + dtx) % Dx, (8 + dy + dty) % Dy)),
                                    (((9 + dx + dtx) % Dx, (4 + dy + dty) % Dy), ((10  +  dx + dtx) % Dx, (7 + dy + dty) % Dy)),
                                        (((-2 + dx + dtx) % Dx, (10 + dy + dty) % Dy),((2 +  dx + dtx) % Dx, (9 + dy + dty) % Dy)))) 
                if t <= L_t - 2:   
                    detectors[x][y][a-8][t].add(((((7 - 1*7  + dx + dtx) % Dx, (11 - 1*8 + dy + dty) % Dy),((11 - 1*7 + dx + dtx) % Dx, (10 -1*8 + dy + dty) % Dy)),
                                    (((-3 - 1*7 + dx + dtx) % Dx, (7 - 1*8 + dy + dty) % Dy), ((1 - 1*7 + dx + dtx) % Dx, (6 -1*8 + dy + dty) % Dy))))
                if t <= L_t - 3: 
                    detectors[x][y][a-9][t].add(((((13- 2*7+ dx + dtx) % Dx, (3 - 2*8+ dy + dty) % Dy),((9 -2*7+ dx + dtx) % Dx, (4-2*8+ dy + dty) % Dy)),
                                           ((15-2*7+ dx + dtx) % Dx, (9-2*8+ dy + dty) % Dy)
                                           ))
                if t <= L_t - 4: 
                    detectors[x][y][a-10][t].add(((((13-3*7 + dx + dtx) % Dx, (3-3*8 + dy + dty) % Dy),((14-3*7 + dx + dtx) % Dx, (6-3*8 + dy + dty) % Dy))))
                if t <= L_t - 5: 
                    detectors[x][y][a-11][t].add(((((-1-4*7 + dx + dtx) % Dx, (13-4*8 + dy + dty) % Dy),((-2-4*7 + dx + dtx) % Dx, (10-4*8 + dy + dty) % Dy))))
                if t <= L_t - 6: 
                    detectors[x][y][a-12][t].add(((((-1 - 5*7 + dx + dtx) % Dx, (13 - 5*8 + dy + dty) % Dy),((3 - 5*7 + dx + dtx) % Dx, (12 - 5*8 + dy + dty) % Dy)),
                                            ((-3 - 5*7+ dx + dtx) % Dx, (7 - 5*8+ dy + dty) % Dy)))                    
                if t <= L_t - 7: 
                    detectors[x][y][a-13][t].add(((((11 - 6*7 + dx + dtx) % Dx, (10 - 6*8 + dy + dty) % Dy),((15- 6*7 + dx + dtx) % Dx, (9 -6*8 + dy + dty) % Dy)), 
                                                    (((1 - 6*7 + dx + dtx) % Dx, (6 - 6*8 + dy + dty) % Dy),((5 - 6*7 + dx + dtx) % Dx, (5 - 6*8 + dy + dty) % Dy))))
                if t <= L_t - 8: 
                    detectors[x][y][a-14][t].add(((((2 - 7*7 + dx + dtx) % Dx, (9 - 7*8 + dy + dty) % Dy),((3-7*7 + dx + dtx) % Dx, (12-7*8 + dy + dty) % Dy)),
                                                    (((6 - 7*7 + dx + dtx) % Dx, (8 - 7*8 + dy + dty) % Dy),((7 - 7*7 + dx + dtx) % Dx, (11- 7*8 + dy + dty) % Dy)),
                                                    (((10 - 7*7 + dx + dtx) % Dx, (7 - 7*8 + dy + dty) % Dy),((14 - 7*7 + dx + dtx) % Dx, (6 - 7*8 + dy + dty) % Dy))))

    i = 0
    m = []
    for a,xdet in enumerate(detectors): #find measurements for full det
        for b,ydet in enumerate(xdet):
            for w,det in enumerate(ydet):
                i += 1
                p = find_corresponding_meas(det, sliced_measurements) 
                p = {i - 26 * L_x * L_y for i in p}              
                if len(p) == 16:
                    m.append(p)

############################ chromobius x-detector annotation
    colours = []
    cycle_length = t * 20

    for outer in range(L_x):
        start = outer % 3
        sub_cycle = [(start + i) % 3 for i in range(3)]

        for _ in range(L_y):
            for _ in range(2):  # Repeat the t*20 pattern twice
                for i in range(cycle_length):
                    colours.append(sub_cycle[i % 3])
############################

    for i, d in enumerate(m):
        d = list(d)     
        circuit += Circuit(f"DETECTOR(0,0,0,{colours[i]}) {' '.join([f'rec[{d[l]}]' for l in range(len(d))])}") 

    return circuit

def extract_index_from_target(target: stim.DemTarget):
    s = str(target)
    if s.startswith("D"):
        return "D", int(s[1:])
    elif s.startswith("L"):
        return "L", int(s[1:])
    else:
        raise ValueError(f"Unknown target: {s}")

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

def dem_to_parity_check_matrix(dem: stim.DetectorErrorModel, include_observables=False):
    data = []
    for error_instruction in dem:
        detectors = []
        observables = []

        for target in error_instruction.targets_copy():
            kind, idx = extract_index_from_target(target)
            if kind == "D":
                detectors.append(idx)
            elif kind == "L" and include_observables:
                observables.append(idx)

        data.append((detectors, observables))

    max_det = max((max(dets) for dets, _ in data if dets), default=-1)
    max_obs = max((max(obs) for _, obs in data if obs), default=-1) if include_observables else -1

    num_cols = max_det + 1 + (max_obs + 1 if include_observables else 0)
    matrix = np.zeros((len(data), num_cols), dtype=np.uint8)

    for i, (dets, obs) in enumerate(data):
        for d in dets:
            matrix[i, d] = 1
        for o in obs:
            if include_observables:
                matrix[i, max_det + 1 + o] = 1

    return matrix

def generate_random_binary_array(L_x, L_y, L_t, p):
    length = 14 * L_x * L_y * L_t + 234
    num_ones = int(round(p * length))
  
    # Start with an array of all zeros
    # array = np.zeros(length, dtype=int)
    array = np.array([0]*length)
    
    # Randomly choose indices to set to 1
    array[:num_ones] = 1
    np.random.shuffle(array)
    
    return array

def main():
    L_x = 3      # number of horizontal "tiles"
    L_y = 3      # number of vertical "tiles"
    L_t = 14     # number of timesteps: (multiples of 13) + 1

    circuit, measurements, final_measurements = floquetcolour(L_x,L_y,L_t,0)
    circuit = addxdetectors(circuit,measurements,final_measurements, L_x,L_y,L_t)
    circuit = addzdetectors(circuit,measurements,final_measurements, L_x,L_y,L_t)

    circuit = addnoise(circuit,{"H", "R", "M","CZ"},0.25,0.25) #(circuit, faulty gates, single qubit noise, multi qubit noise)
    dem = circuit.detector_error_model(decompose_errors=True,ignore_decomposition_failures=True)

    # Decode with Chromobius.
    shots = 1
    dets, actual_obs_flips = circuit.compile_detector_sampler().sample(
        shots=shots,
        separate_observables=True,
        bit_packed=True,
    )
    decoder = chromobius.compile_decoder_for_dem(circuit.detector_error_model())
    predicted_obs_flips = decoder.predict_obs_flips_from_dets_bit_packed(dets)
    # count logical errors
    print(np.count_nonzero(np.any(predicted_obs_flips != actual_obs_flips, axis=1))/shots)   

if __name__ == "__main__":
    main()