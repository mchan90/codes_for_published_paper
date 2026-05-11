import numpy as np
from mpi4py import MPI
import time
import random as rd
import copy
import gc
import pickle
from qiskit.quantum_info import SparsePauliOp
n_qubit = 1
dim     = 2**n_qubit

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()



n_grover = 14
N = 2**n_grover

t_f_max = 163840
dt = 0.5
num_tf = 32

nn = int(round(t_f_max / dt)) 
idxs = np.logspace(0, np.log10(nn), num=num_tf)
idxs_unique = np.unique(np.round(idxs).astype(int))
t_f_maxs = idxs_unique * dt

def Hamiltonian(s):
    vz = 1.0 - 2.0 * (1-s) *(1-1/N)
    vx = (1-s) * 2/np.sqrt(N) * np.sqrt(1-1/N)
    h = SparsePauliOp.from_list([('I',0.5),('X',-0.5*vx),('Z',-0.5*vz)])
    return h

def exact_optimal_schedule(tau):
    s = 0.5 + 0.5/np.sqrt(N-1) * np.tan((2 * tau-1)*np.arctan(np.sqrt(N-1)))
    return s

taus = []
s_list = []

with open('../../../../../on_the_fly/14/optimal_schedule','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        taus.append(float(ls[0]))
        s_list.append(float(ls[1]))

from scipy.interpolate import PchipInterpolator
optimal_schedule = PchipInterpolator(taus, s_list)

def linear_schedule(tau):
    return tau


# 1) Greedy “best‑fit decreasing” partitioning
#    (runs identically on every rank)
assignments = [[] for _ in range(cores)]
loads       = [0.0]*cores

# sort tasks descending so big jobs get placed first
for val in sorted(t_f_maxs, reverse=True):
    # pick the core with the smallest current total
    i = loads.index(min(loads))
    assignments[i].append(val)
    loads[i] += val

# 2) pick out *this* rank’s tasks
my_tasks = np.array(assignments[core])

# now my_tasks[core] is a list whose sum(loads[core]) is 
# roughly the same as loads[j] for any other j.
print(f"rank {core} load = {loads[core]:.3g}, tasks = {my_tasks}")

local_results = []


def ExactEvolution (eigen_e, eigen_v, eps, time):
    evol = np.zeros((dim,dim),dtype=complex)
    vec = np.zeros((dim),dtype=complex)
    for k in range(dim):
        vec[k] = np.exp(-1j*time*(eigen_e[k]-eps))
    exp_d = np.diag(vec)
    evol = eigen_v@exp_d@eigen_v.conj().T
    return evol

def adiabatic_evolve (schedule, t_f, dt, phi, phi_exact):
    n_steps = max(round(t_f/dt),1)
    dt_step = t_f/n_steps
    t = 0.0

    start_print = time.perf_counter()
    for step in range(n_steps+1):
        t_mid = t + 0.5 * dt_step if step<n_steps else t_f
        tau = t_mid/t_f
        s = schedule(tau)
        h_asp = Hamiltonian(s)
        eigen_e, eigen_v = np.linalg.eigh(h_asp)
        evol = ExactEvolution(eigen_e, eigen_v, 0, dt)
        phi = evol@phi
        fidelity = np.abs(phi.conj()@phi_exact)**2

        elapsed = time.perf_counter() - start_print
        if (elapsed>1):
            print(t, schedule(t/t_f), fidelity)
            start_print = time.perf_counter()
        t += dt_step
    return fidelity


for t_f in my_tasks:
    print('# start: {t_f}'.format(t_f=t_f))
    start = time.perf_counter()

    # prepare the initial state
    H0 = Hamiltonian(0).to_matrix()
    eigen_e, eigen_v = np.linalg.eigh(H0)
    phi = eigen_v[:,0]

    # prepare the exact state for compare
    H_target = Hamiltonian(1.0).to_matrix()
    eigen_e, eigen_v = np.linalg.eigh(H_target)
    phi_exact = eigen_v[:,0]

    ov2 = adiabatic_evolve(exact_optimal_schedule, t_f, dt, phi, phi_exact)

    local_results.append((t_f, ov2))

    print('#', t_f, ov2)

    elapsed = time.perf_counter() - start
    print('# done: {t_f} with {elapsed} s'.format(t_f=t_f, elapsed=elapsed))


comm.Barrier()
all_results = comm.gather(local_results, root=0)

if core == 0:
    # all_results is now a list of lists, one per rank
    flat = [item for sublist in all_results for item in sublist]
    # optionally sort by the interpol value
    flat.sort(key=lambda x: x[0])
    ordered_F = [f for (_, f) in flat]
    with open('fidelity','w') as file_:
        for i in range(len(t_f_maxs)):
            s = '{:.16e}    {:.16e}'.format(t_f_maxs[i], ordered_F[i])
            #print(s)
            s += '\n'
            file_.write(s)
