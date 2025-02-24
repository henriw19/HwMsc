import stim
from stim import Circuit
import numpy as np
from matplotlib import pyplot as plt 
import networkx as nx
import re

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

    def measurex(target_qubits, ancilla,x,y,final:bool,p):
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
    
                    circuit += locals()[f"measure{"x" if i[-1] == 1 else "z"}"](target_qubits, a,x,y,0,p)
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

def adddetectors(circuit, measurements,final_measurements, L_x, L_y, L_t):
    # add detectors and observable to the toric floquet colour code

    sliced_measurements = {}

    for t in range(L_t-1, -1, -1): # slice measurement schedule into timesteps
        if t == L_t-1:
            sliced_measurements[t] = (measurements[-14 * L_x * L_y:])
        else: 
            sliced_measurements[t] = (measurements[-14*((L_t - t ))*L_x*L_y:(-(L_t - t - 1)*14*L_x*L_y)] )

    def find_corresponding_meas(dict1, dict2, check = False):   # find measurement number ("-x") corresponding to specific measurement at t
        result = set()
        #dict1: measurements to find, dict2: all measurements
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
    for t in range(L_t - 7, L_t): 
        for y in range(L_y):
            for x in range(L_x):
                dx = 13 * x
                dy = 26 * y
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
                    
    # fully contained detectors:
    for t in range(L_t-1 + 7,0,-1): 
        for y in range(L_y):
            for x in range(L_x):

                dx = 13 * x
                dy = 26 * y
                Dx = 13 * L_x
                Dy = 26 * L_y
                dtx = 7 * (L_t - 1 - t)
                dty = 8 * (L_t - 1 - t)

                a = L_t - 1  - t # a is detector label

                if t <= L_t - 1:
                    detectors[x][y][a-7][t].add(((((5 + dx + dtx) % Dx, (5 +  dy + dty) % Dy),((6 +  dx + dtx) % Dx, (8 + dy + dty) % Dy)),
                                    (((9 + dx + dtx) % Dx, (4 + dy + dty) % Dy), ((10  +  dx + dtx) % Dx, (7 + dy + dty) % Dy)),
                                        (((-2 + dx + dtx) % Dx, (10 + dy + dty) % Dy),((2 +  dx + dtx) % Dx, (9 + dy + dty) % Dy))))        
                if t <= L_t - 2:   
                    detectors[x][y][a-8][t].add(((((0  + dx + dtx) % Dx, (3 + dy + dty) % Dy),((4 + dx + dtx) % Dx, (2 + dy + dty) % Dy)),
                                    (((-10 + dx + dtx) % Dx, (-1 + dy + dty) % Dy), ((-6 + dx + dtx) % Dx, (-2 + dy + dty) % Dy))))
                if t <= L_t - 3: 
                    detectors[x][y][a-9][t].add(((((-1+ dx + dtx) % Dx, (-13+ dy + dty) % Dy),((-5+ dx + dtx) % Dx, (-12+ dy + dty) % Dy)),
                                           ((1+ dx + dtx) % Dx, (-7+ dy + dty) % Dy)))
                if t <= L_t - 4: 
                    detectors[x][y][a-10][t].add(((((-8 + dx + dtx) % Dx, (-21 + dy + dty) % Dy),((-7 + dx + dtx) % Dx, (-18 + dy + dty) % Dy))))
                if t <= L_t - 5: 
                    detectors[x][y][a-11][t].add(((((-29 + dx + dtx) % Dx, (-19 + dy + dty) % Dy),((-30 + dx + dtx) % Dx, (-22 + dy + dty) % Dy))))
                if t <= L_t - 6: 
                    detectors[x][y][a-12][t].add(((((-36 + dx + dtx) % Dx, (-27 + dy + dty) % Dy),((-32 + dx + dtx) % Dx, (-28 + dy + dty) % Dy)),
                                            ((-38+ dx + dtx) % Dx, (-33+ dy + dty) % Dy)))                    
                if t <= L_t - 7: 
                    detectors[x][y][a-13][t].add(((((-31 + dx + dtx) % Dx, (-38 + dy + dty) % Dy),((-27 + dx + dtx) % Dx, (-39 + dy + dty) % Dy)),
                                                    (((-41 + dx + dtx) % Dx, (-42 + dy + dty) % Dy),((-37 + dx + dtx) % Dx, (-43 + dy + dty) % Dy))))
                if t <= L_t - 8: 
                    detectors[x][y][a-14][t].add(((((-47 + dx + dtx) % Dx, (-47 + dy + dty) % Dy),((-46 + dx + dtx) % Dx, (-44 + dy + dty) % Dy)),
                                                    (((-43 + dx + dtx) % Dx, (-48 + dy + dty) % Dy),((-42 + dx + dtx) % Dx, (-45 + dy + dty) % Dy)),
                                                    (((-39 + dx + dtx) % Dx, (-49 + dy + dty) % Dy),((-35 + dx + dtx) % Dx, (-50 + dy + dty) % Dy))))
    

    i = 0
    m = []
    for xdet in detectors: #find measurements for full det
        for ydet in xdet:
            for det in ydet:
                i += 1
                p = find_corresponding_meas(det, sliced_measurements)  
                m.append(p)

    m2 = []
    for x, xdet in enumerate(cutoffdetectors): # find meas. for cutoff det
        for y, ydet in enumerate(xdet):
            for i, det in enumerate(ydet):
                p2 = find_corresponding_meas(det, sliced_measurements)
                p2 = {i - 26 * L_x * L_y for i in p2}
                for u in det[L_t]:
                    for uu in u:
                        value = None  
                        for d in final_measurements[(x, y)]:
                            if uu in d:
                                value = d[uu]
                                break 
                        if value is not None:
                            p2.add(value)  
                if len(p2) > 0:
                    m2.append(p2)
       
    for i,d in enumerate(m): 
        d = list(d)
        circuit.append("DETECTOR", [stim.target_rec(d[l] - 26 * L_x * L_y) for l in range(len(d))])





    # OBSERVABLE not working:
    observable = {key: set() for key in range(L_t+10)} 

    Dx = 13 * L_x
    Dy = 26 * L_y
    for t in range(0,L_t):
        for y in range(L_y):
            
            # if t % 2 == 0: #even pattern
            #     observable[t].add((((((9 + (-14) * (t//2)) % Dx),((4 + 26 * y + (-16) * (t//2)) % Dy)),(((10 + (-14) * (t//2)) % Dx), ((7 + 26 * y + (-16) * (t//2)) % Dy))),
            #                     (((14 + (-14) * (t//2)) % Dx),((19 + 26 * y + (-16) * (t//2)) % Dy))))
                
            # if t % 2 == 1: #uneven pattern
            #     observable[t].add((((((-2  + (-14) * ((t//2))) % Dx),((-3 + 26 * y + (-16) * (t//2)) % Dy)),(((-1 + (-14) * (t//2)) % Dx), ((0 + 26 * y + (-16) * (t//2)) % Dy))),
            #                        (((-6 + (-14) * ((t//2))) % Dx),((11 + 26 * y + (-16) * (t//2)) % Dy))))         
            
            # if t == L_t - 1 and t % 2 == 1: #final measurements uneven
            #     observable[L_t].add(((((-4+7 + (-7) * t) % Dx),((-9+8 + 26 * y + (-8) * t) % Dy)) ,
            #                          (((-3+7 + (-7) * t) % Dx),((-6+8 + 26 * y + (-8) * t) % Dy)) ,
            #                          (((-2+7 + (-7) * t) % Dx),((-3+8 + 26 * y + (-8) * t) % Dy)),
            #                          (((-5+7 + (-7) * t) % Dx),(( 1+8 + 26 * y + (-8) * t) % Dy))
            #                             ))
            
            # elif t == L_t - 1 and t % 2 == 0: #final measurements even
            #     observable[L_t].add((
            #         (((13 + (-7) * t) % Dx),((3  + 26 * y + (-8) * t) % Dy)) ,
            #                          (((9  + (-7) * t) % Dx),((4  + 26 * y + (-8) * t) % Dy)) ,
            #                          (((11 + (-7) * t) % Dx),((10 + 26 * y + (-8) * t) % Dy)),
            #                          (((12 + (-7) * t) % Dx),((13 + 26 * y + (-8) * t) % Dy)),
            #                         (((12 + (-7) * t) % Dx),((13 + 26 * y + (-8) * t) % Dy))
                                   
            #                             ))
            pass

    m = find_corresponding_meas(observable, sliced_measurements,True)
    cut_off_obs = []

    for tuple_group in observable[L_t]:
        for target_tuple in tuple_group:
            for key, list_of_dicts in final_measurements.items():
                for sub_dict in list_of_dicts:
                    if target_tuple in sub_dict:
                        print(target_tuple,sub_dict)
                        cut_off_obs.append(sub_dict[target_tuple])
  

    m = list(m)
    m = [i - 26 * L_x * L_y for i in m]
    m += cut_off_obs
    circuit_string = f"""OBSERVABLE_INCLUDE(0) {" ".join(f"rec[{_}]" for _ in m)}"""
    circuit += Circuit(circuit_string)

    return circuit

def main():
    L_x = 2      # number of horizontal "tiles"
    L_y = 1      # number of vertical "tiles"
    L_t = 14     # number of timesteps: (multiples of 13) + 1

    circuit, measurements, final_measurements = floquetcolour(L_x,L_y,L_t,0)
    circuit = adddetectors(circuit,measurements,final_measurements, L_x,L_y,L_t)

    dem = circuit.detector_error_model(decompose_errors=True)
    print(dem) 

if __name__ == "__main__":
    main()



