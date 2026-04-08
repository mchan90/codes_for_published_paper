from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit import qpy
import sys, pickle
from qiskit.quantum_info import SparsePauliOp


pickled_input_data = sys.stdin.buffer.read()
input_data = pickle.loads(pickled_input_data)
n_qubit =4
observable = SparsePauliOp.from_sparse_list([("Z", [2], 1)], num_qubits=5)
backend_options = {'method': 'statevector', 'max_parallel_threads': 1}
estimator = Estimator(options = {"default_precision":0.00,
                                     'backend_options':backend_options})
run_circuits_file = 'circuits_opt.qpy'
with open(run_circuits_file, 'rb') as file_:
    circuits = qpy.load(file_)
pubs = []
len_input = len(input_data)
for ind in range(len_input):
    i0     = input_data[ind][0]
    if (len(input_data[ind])>1):
        params = input_data[ind][1]
        pubs.append((circuits[i0],observable,params))
    else:
        pubs.append((circuits[i0],observable))
job = estimator.run(pubs)
result = job.result()
len_result = len(result)
list_out = []
for i in range(len_result):
    list_out.append(result[i].data.evs)
sys.stdout.buffer.write(pickle.dumps(list_out))
    