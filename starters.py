from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator,noise
from qiskit.visualization import *
from matplotlib import pyplot as plt
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, thermal_relaxation_error
from qiskit_aer.noise.errors import ReadoutError
import numpy as np

# Define the [[4,2,2]] encoding circuit
def normal_422():
    qc = QuantumCircuit(4)
    # Encoding for [[4,2,2]] code
    qc.h(3)
    qc.cx([0,1,3,3,3], [2,2,2,1,0])
    return qc

def ancilla_422():

    qc = QuantumCircuit(6,2)
    qc.h(3)
    qc.cx([0,1,3,3,3], [2,2,2,1,0])
    qc.barrier()
    qc.draw('mpl')
    # plt.show()
    return qc


def floquet_422():
    qc = QuantumCircuit(12,12*5)
    # 1st time step
    qc.h([0,1])
    qc.measure(0,0)
    qc.measure(1,1)
    qc.measure([2,4],[2,3])
    qc.measure([3,5],[4,5])
    qc.measure([6,7],[6,7])
    qc.measure([8,10],[8,9])
    qc.measure([9,11],[10,11])

    # 2nd time step
    qc.h([0,1,10,11,4,5,6,7,8,9])
    qc.measure([0,10],[12,13])
    qc.measure([1,11],[14,15])
    qc.measure(2,16)
    qc.measure(3,17)
    qc.measure([4,6],[18,19])
    qc.measure([5,7],[20,21])
    qc.measure([8,9],[22,23])

    # t even
    qc.measure(t,i)
    qc.measure(t,j)



# Define the decoding circuit (for simplicity, inverse of encoding)
def decode_422():
    qc = QuantumCircuit(4)
    # Decoding for [[4,2,2]] code (inverse of encoding)
    qc.cx(1, 3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.h(0)
    return qc


def noise_model():
    # noise model
    noise_model = NoiseModel()
    # Define depolarizing noise for single-qubit and two-qubit gates
    single_qubit_error = depolarizing_error(0.01, 1)  # 1% error rate for single-qubit gates
    two_qubit_error = depolarizing_error(0.02, 2)     # 2% error rate for two-qubit gates

    # Define readout error (e.g., 2% chance of flipping 0 to 1 or 1 to 0)
    readout_error = ReadoutError([[0.98, 0.02], [0.02, 0.98]])

    # Add errors to the noise model
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model



qc = ancilla_422()           # Encoding

qc.x([0,1])                     # Error

qc.barrier()
qc.h([4,5])
qc.cx([4,4,4,4],[0,1,2,3])
qc.cz([5,5,5,5],[0,1,2,3])
qc.h([4,5])
qc.measure([4,5],[0,1])





noise_model = noise_model()
shots=1000
# Use the QASM simulator with the noise model
simulator = AerSimulator(noise_model = noise_model, method='statevector')
circ = transpile(qc, simulator)
result = simulator.run(circ, shots=shots).result()
counts = result.get_counts(circ)

err = 0
for key in counts:
    if sum(int(char) for char in key if char.isdigit())%2 != 0:
        err += counts[key]

print(err/shots)
        

plot_histogram(counts, title='state counts')
plt.savefig('counts')


# ideal: 0.1305