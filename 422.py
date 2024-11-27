import stim
from stim import Circuit
import numpy as np
from matplotlib import pyplot as plt 
import pymatching


def vanilla422():
    # Encoding 
    circuit = Circuit('''
    H 3
    CX 0 2 1 2 3 2 3 1 3 0
    ''')

    # Syndrome extraction
    # ZZZZ
    circuit += Circuit('''      
        H 4
        CZ 4 0 4 1 4 2 4 3
        H 4
        M 4
    ''')

    # XXXX
    circuit += Circuit('''      
        H 5
        CX 5 0 5 1 5 2 5 3
        H 5
        M 5
    ''')

    return circuit

def periodic422(t):
    # Normal encoding 
    circuit = Circuit('''
        H 3
        CX 0 2 1 2 3 2 3 1 3 0  
        H 4
        CZ 4 0 4 1 4 2 4 3
        H 4
        M 4         
    ''')

    for i in range(t):
        # Syndrome extraction

        # XXXX
        circuit += Circuit('''     
            H 5
            CX 5 0 5 1 5 2 5 3
            H 5
            M 5
            R 5
        ''')

        # Detectors
        if i > 0:
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])

        # ZZZZ
        circuit += Circuit('''      
            H 4
            CZ 4 0 4 1 4 2 4 3
            H 4
            M 4
            R 4
        ''')
        
        # Detectors
        circuit.append(f"DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])
        

    # measure logical Z
    circuit += Circuit('''   
            CX 0 6 3 6   
            M 6
            OBSERVABLE_INCLUDE(0) rec[-1] 
        ''')
    return circuit

def floquet422(L,p=0):
    def measurex(target_qubits, ancilla):
        return Circuit(f'''
                            H {ancilla}
                            CX {' '.join(f'{ancilla} {x}' for x in target_qubits)}
                            H {ancilla}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    def measurez(target_qubits, ancilla):
        # return Circuit(f'''
        #                     H {ancilla}
        #                     CZ {' '.join(f'{ancilla} {x}' for x in target_qubits)}
        #                     H {ancilla}
        #                     M {ancilla}
        #                     R {ancilla}
        #                     ''')
        return Circuit(f'''
                            CX {' '.join(f'{x} {ancilla}' for x in target_qubits)}
                            M {ancilla}
                            R {ancilla}
                            ''')

    circuit = Circuit()
    for l in range(L):
        for t in range(6):
            m = [measurez, measurex]

            # single qubit measurements
            circuit += m[t%2]([(t * 2) % 12], 12) 
            circuit += m[t%2]([(t * 2 + 1) % 12], 13)
            
            # inner hexagon measurement #1
            circuit += m[t%2]([(t * 2 + 2) % 12, (t * 2 + 4) % 12], 14)
            
            # outer hexagon measurement #1
            circuit += m[t%2]([(t * 2 + 3) % 12, (t * 2 + 5) % 12], 15)

            # if t == 3:
            #     circuit += Circuit(f'DEPOLARIZE1({p}) 8 3 2 5')
            
            # radial measurement
            circuit += m[t%2]([(t * 2 + 6) % 12, (t * 2 + 7) % 12], 16)

            # inner hexagon measurement #2
            circuit += m[t%2]([(t * 2 + 8) % 12, (t * 2 + 10) % 12], 17)

            # outer hexagon measurement #2
            circuit += m[t%2]([(t * 2 + 9) % 12, (t * 2 + 11) % 12], 18)

        circuit += Circuit(''' M 0 1 2 3 4 5 6 7 8 9 10 11''')

            
                
        # total measurements per period: 7 * 6 = 42
        circuit.append("DETECTOR", [stim.target_rec(-54), stim.target_rec(-53), stim.target_rec(-49), stim.target_rec(-48), stim.target_rec(-26), stim.target_rec(-25), stim.target_rec(-24), stim.target_rec(-23)])
        
    # measure logical Z
    circuit += Circuit('''   
                   OBSERVABLE_INCLUDE(0) rec[-54] rec[-52] rec[-49] rec[-40] rec[-38] rec[-35] rec[-26] rec[-24] rec[-21] rec[-4] rec[-6] rec[-10] rec[-12] 
        ''')
    
    # # measure logical X
    # circuit += Circuit('''   
    #                OBSERVABLE_INCLUDE(0) rec[-31] rec[-30] rec[-29] rec[-15] rec[-16] rec[-17] rec[-3] rec[-2] rec[-1] 
    #     ''')
    
    return circuit

def addnoise(circuit, faulty_gates, p_depol = 0):
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
                pass
                # for i,qubit in enumerate(target_qubits[::2]):
                #     print(i,qubit)
                #     noisy_circuit.append_operation("DEPOLARIZE2", [target_qubits[i * 2], target_qubits[(i * 2 + 1)]], p_depol)
            else:
                for target in target_qubits:
                    noisy_circuit.append_operation("DEPOLARIZE1", [target], p_depol)
            

        else:
            noisy_circuit += Circuit(f'''{gate} {" ".join(target_qubits)}''')
    return noisy_circuit

def log_err_counter(circuits, ps, num_shots):
    for t,circ in enumerate(circuits):
        xs = []
        ys = []
        
        for noise in ps:
            noisy_circuit = addnoise(circ, {"H","M","CZ","CX","R"}, noise)

            sampler = noisy_circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
            dem = noisy_circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            predictions = matcher.decode_batch(detection_events)
            num_errors = 0
            for shot in range(num_shots):
                actual_errors = observable_flips[shot]
                predicted_errors = predictions[shot]
                if not np.array_equal(actual_errors, predicted_errors):
                    num_errors += 1
            xs.append(noise)
            ys.append(num_errors / num_shots)
        plt.plot(xs, ys, label="L=" + str(t))
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    # print(repr(noisy_circuit))
    plt.show()
    # plt.savefig("logerr.png")

def main():
      
    circ = floquet422(1)
    
    sampler = circ.compile_detector_sampler()
    # data = sampler.sample(shots=100)
    dem = circ.detector_error_model(decompose_errors=True)
    # print(dem)
    # log_err_counter([circ], np.logspace(-4,-.3,7), 10000)
    # print(repr(circ))
    
    # Find constant and changing indices
    # constant_indices = np.where(np.all(data == data[0], axis=0))[0]  # Indices where all values match
    # changing_indices = np.where(~np.all(data == data[0], axis=0))[0]  # Indices where values differ

    # # Print results
    # print("Constant Indices:", constant_indices)
    # print("Changing Indices:", changing_indices)


if __name__ == "__main__":
    main()
