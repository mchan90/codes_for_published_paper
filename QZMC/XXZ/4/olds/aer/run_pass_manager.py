from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import qpy
from qiskit_aer import AerSimulator
backend_options = {'method': 'statevector', 'max_parallel_threads': 1}
sim = AerSimulator(**backend_options)
pass_manager = generate_preset_pass_manager(2, sim)
with open('circuits.qpy', 'rb') as file_:
    circuits = qpy.load(file_)
    n_circuit = len(circuits)
    circuits_opt = []
    for i_circuit in range(n_circuit):
        circuit_opt = pass_manager.run(circuits[i_circuit])
        circuits_opt.append(circuit_opt)
with open ('circuits_opt.qpy', 'wb') as file_:
    qpy.dump(circuits_opt,file_)
    