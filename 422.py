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

def periodic422(L):   
    circuit = Circuit()
 
    for i in range(L):
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
                            CX 0 4 1 4 2 4 3 4
                            M 4
                            R 4
                            ''')
        
        # Detectors
        if i > 0:
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3)])
        else:
            circuit.append("DETECTOR", [stim.target_rec(-1)])
    
    circuit += Circuit("M 0 1 2 3")

    
    circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3), stim.target_rec(-4),stim.target_rec(-5)])
    
    # measure logical Z
    circuit += Circuit('''   
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-4] 
    ''')
    return circuit

def nafloquet422(L):
    def measurex(target_qubits, ancilla):
        return Circuit(f'''
                            H {ancilla}
                            CX {' '.join(f'{ancilla} {x}' for x in target_qubits)}
                            H {ancilla}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    def measurez(target_qubits, ancilla):
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
            
            # radial measurement
            circuit += m[t%2]([(t * 2 + 6) % 12, (t * 2 + 7) % 12], 16)

            # inner hexagon measurement #2
            circuit += m[t%2]([(t * 2 + 8) % 12, (t * 2 + 10) % 12], 17)

            # outer hexagon measurement #2
            circuit += m[t%2]([(t * 2 + 9) % 12, (t * 2 + 11) % 12], 18)

        if l < L-1:
            circuit += Circuit('''   
                        OBSERVABLE_INCLUDE(0) rec[-40] rec[-37] rec[-26] rec[-23] rec[-12] rec[-9] 
                ''')
            
                
        # total measurements per period: 7 * 6 = 42
        
        # contained z detector
        circuit.append("DETECTOR", [stim.target_rec(-42), stim.target_rec(-41), stim.target_rec(-37), stim.target_rec(-36), stim.target_rec(-14), stim.target_rec(-13), stim.target_rec(-12), stim.target_rec(-11)])
        
        # contained x detector
        circuit.append("DETECTOR", [stim.target_rec(-35), stim.target_rec(-34), stim.target_rec(-30), stim.target_rec(-29), stim.target_rec(-7), stim.target_rec(-6), stim.target_rec(-5), stim.target_rec(-4)])

        if l > 1:
            # overlap z
            circuit.append("DETECTOR", [stim.target_rec(-28-42), stim.target_rec(-27-42), stim.target_rec(-23-42), stim.target_rec(-22-42), stim.target_rec(-42), stim.target_rec(-41), stim.target_rec(-40), stim.target_rec(-39)])
            
            # overlap x
            circuit.append("DETECTOR", [stim.target_rec(-21-42), stim.target_rec(-20-42), stim.target_rec(-16-42), stim.target_rec(-15-42), stim.target_rec(-35), stim.target_rec(-34), stim.target_rec(-33), stim.target_rec(-32)])

            # overlap z
            circuit.append("DETECTOR", [stim.target_rec(-14-42), stim.target_rec(-13-42), stim.target_rec(-9-42), stim.target_rec(-8-42), stim.target_rec(-28), stim.target_rec(-27), stim.target_rec(-26), stim.target_rec(-25)])

            # overlap x
            circuit.append("DETECTOR", [stim.target_rec(-7-42), stim.target_rec(-6-42), stim.target_rec(-2-42), stim.target_rec(-1-42), stim.target_rec(-21), stim.target_rec(-20), stim.target_rec(-19), stim.target_rec(-18)])

        if l == 0:
            # initial cut off z detectors
            circuit.append("DETECTOR", [stim.target_rec(-42), stim.target_rec(-41), stim.target_rec(-40), stim.target_rec(-39)])
            circuit.append("DETECTOR", [stim.target_rec(-28), stim.target_rec(-27), stim.target_rec(-26), stim.target_rec(-25)])

    
    circuit += Circuit(''' M 0 1 2 3 4 5 6 7 8 9 10 11''')
    
    # final cut off z detectors
    circuit.append("DETECTOR", [stim.target_rec(-28-12), stim.target_rec(-27-12), stim.target_rec(-23-12), stim.target_rec(-22-12), stim.target_rec(-7), stim.target_rec(-8), stim.target_rec(-9), stim.target_rec(-10), stim.target_rec(-11), stim.target_rec(-12)])
    circuit.append("DETECTOR", [stim.target_rec(-14-12), stim.target_rec(-13-12), stim.target_rec(-9-12), stim.target_rec(-8-12), stim.target_rec(-3), stim.target_rec(-4), stim.target_rec(-5), stim.target_rec(-6), stim.target_rec(-7), stim.target_rec(-8)])

    
    # measure logical Z
    circuit += Circuit('''   
            OBSERVABLE_INCLUDE(0) rec[-52] rec[-49] rec[-38] rec[-35] rec[-24] rec[-21] rec[-4] rec[-6] rec[-10] rec[-12] 
    ''')   
    return circuit

def dpfloquet422(L):
    circuit = Circuit()
    def measurex(target_qubits, ancilla):
        return Circuit(f'''
                            H {ancilla}
                            CX {' '.join(f'{ancilla} {x}' for x in target_qubits)}
                            H {ancilla}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    def measurez(target_qubits, ancilla):
        return Circuit(f'''
                            CX {' '.join(f'{x} {ancilla}' for x in target_qubits)}
                            M {ancilla}
                            R {ancilla}
                            ''')
    
    for l in range(L):
        for _ in range(2):
            circuit += measurez([0,1],6)
            circuit += measurez([3,4],6)

            circuit += measurex([1,4],6)
            circuit += measurex([1,4],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])

            circuit += measurez([1,2],6)
            circuit += measurez([4,5],6)

            circuit += measurex([2],6)
            circuit += measurex([2],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
            circuit += measurex([5],6)
            circuit += measurex([5],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])

            circuit += Circuit('''H 0 1 3 4''')

            circuit += measurez([0,2],6)
            circuit += measurez([3,5],6)

            circuit += measurex([0,3],6)
            circuit += measurex([0,3],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])

            circuit += measurez([0,1],6)
            circuit += measurez([3,4],6)

            circuit += measurex([1],6)
            circuit += measurex([1],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
            circuit += measurex([4],6)
            circuit += measurex([4],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
            
            circuit += Circuit('''H 0 2 3 5''')

            circuit += measurez([1,2],6)
            circuit += measurez([4,5],6)

            circuit += measurex([2,5],6)
            circuit += measurex([2,5],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])


            circuit += measurez([0,2],6)
            circuit += measurez([3,5],6)

            circuit += measurex([0],6)
            circuit += measurex([0],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
            circuit += measurex([3],6)
            circuit += measurex([3],6)
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
            
            circuit += Circuit('''H 1 2 4 5''')


        # 1st contained
        circuit.append("DETECTOR", [stim.target_rec(-29-30), stim.target_rec(-30-30), stim.target_rec(-21-30), stim.target_rec(-23-30), stim.target_rec(-12-30), stim.target_rec(-14-30), stim.target_rec(-5-30), stim.target_rec(-6-30)])

        #2nd contained
        circuit.append("DETECTOR", [stim.target_rec(-19-30), stim.target_rec(-20-30), stim.target_rec(-11-30), stim.target_rec(-13-30), stim.target_rec(-2-30), stim.target_rec(-4-30), stim.target_rec(-25), stim.target_rec(-26)])

        #3rd contained
        circuit.append("DETECTOR", [stim.target_rec(-9-30), stim.target_rec(-10-30), stim.target_rec(-1-30), stim.target_rec(-3-30), stim.target_rec(-22), stim.target_rec(-24), stim.target_rec(-15), stim.target_rec(-16)])

        #4th contained
        circuit.append("DETECTOR", [stim.target_rec(-29), stim.target_rec(-30), stim.target_rec(-21), stim.target_rec(-23), stim.target_rec(-12), stim.target_rec(-14), stim.target_rec(-5), stim.target_rec(-6)])
        
        if l > 0:
            # overlap detectors
            circuit.append("DETECTOR", [stim.target_rec(-19-30-30), stim.target_rec(-20-30-30), stim.target_rec(-11-30-30), stim.target_rec(-13-30-30), stim.target_rec(-2-30-30), stim.target_rec(-4-30-30),  stim.target_rec(-25-30), stim.target_rec(-26-30)])
            circuit.append("DETECTOR", [stim.target_rec(-9-30-30), stim.target_rec(-10-30-30), stim.target_rec(-1-30-30), stim.target_rec(-3-30-30), stim.target_rec(-22-30), stim.target_rec(-24-30), stim.target_rec(-15-30), stim.target_rec(-16-30)])

        if l == 0:
            # initial cut off
            circuit.append("DETECTOR", [stim.target_rec(-25-30), stim.target_rec(-26-30)])
       

        # Observable in intermediate steps 
        if l < L-1:
            circuit += Circuit(f'''   
                    OBSERVABLE_INCLUDE(0) rec[-60] rec[-56] rec[-53] rec[-44] rec[-40] rec[-36] rec[-33] rec[-24] rec[-20] rec[-16] rec[-13] rec[-4]
            ''')
    
    circuit += Circuit('''M 0 1 2 3 4 5''')
    # final cut off
    circuit.append("DETECTOR", [stim.target_rec(-26), stim.target_rec(-25), stim.target_rec(-19), stim.target_rec(-17), stim.target_rec(-8), stim.target_rec(-10), stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-4), stim.target_rec(-5)])

    # final observable
    circuit += Circuit(f'''   
            OBSERVABLE_INCLUDE(0) rec[-66] rec[-62] rec[-59] rec[-50] rec[-46] rec[-42] rec[-39] rec[-30] rec[-26] rec[-22] rec[-19] rec[-10] rec[-4] rec[-5]
    ''')

    return circuit

def addnoise(circuit, faulty_gates, single_qubit_noise = 0, multi_qubit_noise = 0):
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

def log_err_counter(circuits, single_qubit_noise, multi_qubit_noise, num_shots, colors, labels):
    for t,circ in enumerate(circuits):
        xs = []
        ys = []
        
        for i,noise in enumerate(single_qubit_noise):
            noisy_circuit = addnoise(circ, {"H","M","CZ","CX","R"}, single_qubit_noise[i], multi_qubit_noise[i])
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
        
        plt.plot(xs, ys, label=labels[t], color = colors[t])
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    plt.show()
    # plt.savefig('422comparison')

def detector_finder(circuit):
    sampler = circuit.compile_detector_sampler() 
    detection_events, observable_flips = sampler.sample(100, separate_observables=True)
    dem = circuit.detector_error_model()
    simulator = stim.TableauSimulator()
    simulator.do(circuit)
    stabilizers = simulator.canonical_stabilizers()
    tableau = simulator.current_inverse_tableau()
    num_qubits = 6
    for i in range(num_qubits):
        logical_x = tableau.x_output(i)
        logical_z = tableau.z_output(i)
        print(f"Qubit {i}: Logical X = {logical_x}, Logical Z = {logical_z}")
    return 0


def main():
    # log_err_counter([floq1,floq2,floq3,dpfl1,dpfl2,dpfl3], np.logspace(-4,-.7,10), np.logspace(-3,-.7,10), 10000, ['lime','aqua','olive','r','b','g'], ['nat 1','nat 2','nat 3', 'dp Floq 1','dp Floq 2','dp Floq 3'])

    # code = nafloquet422(3)
    # # noisy = addnoise(code,{"H","M","CZ","CX","R"},0.3,0.3)
    # dem = code.detector_error_model(decompose_errors=True)
    # print(dem)
    # log_err_counter([code], np.logspace(-4,-.7,10), np.logspace(-3,-.7,10), 10000, ['lime','aqua','olive','r','b','g'], ['nat 1'])
    
if __name__ == "__main__":
    main()
