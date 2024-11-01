import stim
from stim import Circuit
import numpy as np
from matplotlib import pyplot as plt 


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

def noisyvanilla422(circuit, p_depol = 0):
    noisy_circuit = stim.Circuit()
    for operation in circuit:
        operation = str(operation).split(' ')
        gate = operation[0]
        target_qubits = np.array(operation[1:]).astype(int)
        # Apply original gate
        noisy_circuit.append_operation(gate, target_qubits)

        # Apply depolarizing noise after each gate (33 in total)
        for target in target_qubits:
            noisy_circuit.append_operation("DEPOLARIZE1", [target], p_depol)
    return noisy_circuit


def main():
    ps_phy = array = np.arange(0, .3, .001)
    shots = 1000000
    rates = []

    # iterate over physical error rates
    circuit = vanilla422()
    for p_phy in ps_phy:
        noisy_circuit = noisyvanilla422(circuit, p_phy)
        sampler = noisy_circuit.compile_sampler()
        results = sampler.sample(shots=shots)  

        # for each p_phy divide number of detection events by number of circuit iterations which (statistically) had an error 
        rate = sum(1 for row in results if any(row))/(shots*(1-(1-p_phy)**33 ))
        rates.append(rate)

    plt.xlabel("probability of depolarizing error after each gate")
    plt.ylabel("error detection rate")
    plt.plot(ps_phy,rates)
    # plt.show()
    plt.savefig('singlequbitdepolarizing')

if __name__ == "__main__":
    main()
