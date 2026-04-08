import numpy as np
import qiskit_nature
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LineLattice,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import ParityMapper

import sys
sys.path.append('../../../../')

# qiskit's parity mapper "assumes" (orb1,spin1), ... (orbn,spin1), (orb1,spin2), ..., (orbn,spin2)
# ordering of second quantized orbitals, which is non consistent qiskit's own FermiHubbardModel routine,
# which has (orb1,spin1), (orb1,spin2), ... (orbn,spin1), (orbn,spin2)
# We modifed FermiHubbardModel, which uses ordering of 
# (orb1,spin1), ... (orbn,spin1), (orb1,spin2), ..., (orbn,spin2)
# This is efficient for two-site case only. In other cases, using
# Jordan-Wigner with qiskit's implementation of FermiHubbardModel is better.
from fermi_hubbard_model import FermiHubbardModel

qiskit_nature.settings.use_pauli_sum_op = False

n_site  = 2
n_qubit = 2*n_site-2 # use two qubit reduction
dim = 2**n_qubit

U_max = 5.0
t_hop = 1.0
Us = np.linspace(0.0,5.0,11)
nU = len(Us)
print(Us)
hamiltonians = []

n_electrons = [1,1]
for U_coulomb in Us:
    t = t_hop
    mu = U_coulomb/2
    bc = BoundaryCondition.PERIODIC
    chain = LineLattice(num_nodes=n_site, boundary_condition=bc, edge_parameter=-t, onsite_parameter=-mu)
    HubbardChain = FermiHubbardModel(chain,onsite_interaction=U_coulomb)
    H = HubbardChain.second_q_op().simplify()
    mapper = ParityMapper(num_particles=n_electrons)
    hamiltonians.append(mapper.map(H))

Us_fine = np.linspace(0.0,5.0,101)
nU_fine = len(Us_fine)
eigen_energies_fine = np.zeros((nU_fine,dim),dtype=float)

iU_fine = 0
for U_coulomb in Us_fine:
    t = 1.0
    mu = U_coulomb/2
    bc = BoundaryCondition.PERIODIC
    chain = LineLattice(num_nodes=n_site, boundary_condition=bc, edge_parameter=-t, onsite_parameter=-mu)
    HubbardChain = FermiHubbardModel(chain,onsite_interaction=U_coulomb)
    h = HubbardChain.second_q_op().simplify()
    mapper = ParityMapper(num_particles=n_electrons)
    h_qubit = mapper.map(h)
    eigen_e, eigen_v = np.linalg.eigh(h_qubit.to_matrix())
    eigen_energies_fine[iU_fine,:] = eigen_e[:]
    iU_fine += 1

n_hamiltonians = nU

for alpha in range(n_hamiltonians):
    print(alpha, hamiltonians[alpha])

hamiltonian_diffs = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs.append((hamiltonians[alpha+1]-hamiltonians[alpha]).simplify())
    print(hamiltonian_diffs[alpha])

hamiltonian_diffs_list = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs_list.append(hamiltonian_diffs[alpha].to_list())
    print(hamiltonian_diffs_list[alpha])


# exact eigenvalues
eigen_energies_exact  = np.zeros((n_hamiltonians,dim),dtype=float)
eigen_vectors_exact   = np.zeros((n_hamiltonians,dim,dim),dtype=complex)
for alpha in range(n_hamiltonians):
    eigen_e, eigen_v = np.linalg.eigh(hamiltonians[alpha].to_matrix())
    indx = np.argsort(eigen_e.real)
    for i in range(dim):
        eigen_energies_exact[alpha,i]   = eigen_e[indx[i]].real
        eigen_vectors_exact[alpha,:,i] = eigen_v[:,indx[i]]


# degenerate perturbation theory
import copy
nsec = 0
sec_list = []
done = [False]*dim
for i in range(dim):
    if done[i]:
        continue
    else:
        l = [i]
        done[i] =True
        for j in range(i+1,dim):
            if done[j]:
                continue
            else:
                if (np.abs(eigen_energies_exact[0,i]-eigen_energies_exact[0,j])<1E-8):
                    l.append(j)
                    done[j] = True
        sec_list.append(l)
#print(sec_list)
nsec = len(sec_list)
nesec = np.zeros(nsec,dtype=int)
for isec in range(nsec):
    nesec[isec] = len(sec_list[isec])
    #print(nesec[isec])
# diagonalize hamiltonian_diffs[0] for each degenerate sector
hp_mat = (eigen_vectors_exact[0,:,:].conj().transpose())@(hamiltonian_diffs[0].to_matrix())@eigen_vectors_exact[0,:,:]
#deps_sec = []
#w_sec   = []
for isec in range(nsec):
    hp_proj = np.zeros((nesec[isec],nesec[isec]),dtype=complex)
    for i in range(nesec[isec]):
        ii = sec_list[isec][i]
        for j in range(nesec[isec]):
            jj = sec_list[isec][j]
            hp_proj[i,j] = hp_mat[ii,jj]
    deps, w = np.linalg.eigh(hp_proj)
    v_t = np.zeros((dim,nesec[isec]),dtype=complex)

    for i in range(nesec[isec]):
        ii = sec_list[isec][i]
        v_t[:,i] = copy.deepcopy(eigen_vectors_exact[0,:,ii])
    for i in range(nesec[isec]):
        ii = sec_list[isec][i]
        eigen_vectors_exact[0,:,ii] = v_t@w[:,i]
    #print(deps,w)
    #deps_sec.append(deps)
    #w_sec.append(w)
#print(deps_sec,w_sec)
#print(eigen_vectors_exact[0,:,:])


def ExactTimeEvolution (alpha, time):
    Vl = copy.deepcopy(eigen_vectors_exact[alpha,:,:])
    evol = np.zeros((dim,dim),dtype=complex)
    exp_d = np.diag(np.exp(-1j*eigen_energies_exact[alpha,:]*time))
    evol = Vl@exp_d@Vl.conj().T
    return evol

# Trotter evolution
from scipy.linalg import expm
X0 = (SparsePauliOp.from_list([('IX',1.0)])).to_matrix()
X1 = (SparsePauliOp.from_list([('XI',1.0)])).to_matrix()
ZZ = (SparsePauliOp.from_list([('ZZ',1.0)])).to_matrix()
def ExactTrotterEvolution (time, alpha, n_trotter, indx):
    theta_k = -2*t_hop*time/n_trotter
    theta_U = -time * Us[alpha]/n_trotter
    evol = np.identity(dim)
    if (indx==0):
        evol = expm(-1j*theta_U/4.0*ZZ)@evol
        evol = expm(-1j*theta_k/2.0*X0)@evol
        evol = expm(-1j*theta_k/2.0*X1)@evol
        for i_trotter in range(n_trotter-1):
            evol = expm(-1j*theta_U/2.0*ZZ)@evol
            evol = expm(-1j*theta_k/2.0*X0)@evol
            evol = expm(-1j*theta_k/2.0*X1)@evol
        evol = expm(-1j*theta_U/4.0*ZZ)@evol
    elif (indx==1):
        evol = expm(-1j*theta_k/4.0*X0)@evol
        evol = expm(-1j*theta_k/4.0*X1)@evol
        evol = expm(-1j*theta_U/2.0*ZZ)@evol
        for i_trotter in range(n_trotter-1):
            evol = expm(-1j*theta_k/2.0*X0)@evol
            evol = expm(-1j*theta_k/2.0*X1)@evol
            evol = expm(-1j*theta_U/2.0*ZZ)@evol
        evol = expm(-1j*theta_k/4.0*X0)@evol
        evol = expm(-1j*theta_k/4.0*X1)@evol
    return evol

def TrotterTimeEvolution_Phase(time, alpha):
    phase = Us[alpha]/2.0 * time
    return  phase

# exact results
norms_exact  = np.ones((n_hamiltonians,dim),dtype=float)
for i in range(dim):
    phi = eigen_vectors_exact[0,:,i]
    for alpha in range(1,n_hamiltonians):
        proj_matrix = np.outer(eigen_vectors_exact[alpha,:,i],eigen_vectors_exact[alpha,:,i].conj())
        phi = proj_matrix@phi
        norms_exact[alpha,i] = phi.conj()@phi
        #print(phi.conj()@hamiltonians[alpha].to_matrix()@phi/norms_exact[alpha,i])
    #print('##')


for alpha in range(n_hamiltonians):
    print(norms_exact[alpha], eigen_energies_exact[alpha,:])


print(eigen_vectors_exact[0,:,0])
print(eigen_vectors_exact[0,:,1])
print(eigen_vectors_exact[0,:,2])
print(eigen_vectors_exact[0,:,3])


from qiskit import QuantumRegister
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator


def Circuit_Initialize(circuit_: QuantumCircuit, i_: int, qr_: QuantumRegister):
    # initialize qr_[1:] to i_ th eigenvector
    match i_:
        case 0:
            circuit_.h(qr_[1])
            circuit_.h(qr_[2])
        case 1:
            circuit_.x(qr_[1])
            circuit_.h(qr_[1])
            circuit_.cx(qr_[1],qr_[2])
        case 2:
            circuit_.x(qr_[1])
            circuit_.x(qr_[2])
            circuit_.h(qr_[1])
            circuit_.cx(qr_[1],qr_[2])
        case 3:
            circuit_.x(qr_[1])
            circuit_.h(qr_[1])
            circuit_.x(qr_[2])
            circuit_.h(qr_[2])

# check initialize routine
for i in range(dim):
    qc = QuantumCircuit(3)
    Circuit_Initialize(qc,i,range(3))
    U_qc = Operator(qc)
    print(U_qc.data[0:2*dim:2,0])
    print(eigen_vectors_exact[0,:,i])
    print('##')


import random as rd
import pickle


nmc = int(100)
beta = 0.5


# observable amplitudes
n_obs = 3
# 0; norm, 1; dE1, 2; dE2
O_timelists         = [[[[[] for _ in range(nmc)] for _ in range(n_obs)] for _ in range(n_hamiltonians)] for _ in range(dim)]

#with open('dimer.time.binary','rb') as file_:
#    [O_timelists] = pickle.load(file_)

for i in range(dim):
    for alpha in range(1,n_hamiltonians):
        for i_obs in range(n_obs):
            for imc in range(nmc):
                times = []
                for alpha_ in range(1,alpha):
                    time = rd.gauss(0.0, beta)
                    times.append(time)

                time = rd.gauss(0.0, beta)
                times.append(time)

                time = rd.gauss(0.0, beta)
                times.append(time)

                for alpha_ in reversed(range(1,alpha)):
                    time = rd.gauss(0.0, beta)
                    times.append(time)
                O_timelists[i][alpha][i_obs][imc] = times
with open('dimer.time.binary','wb') as file_:
    pickle.dump([O_timelists],file_)
##

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
from qiskit.providers import JobStatus
import time as time_lib

service = QiskitRuntimeService(channel="ibm_quantum")

def run_estimator (max_circuits_, estimator_, circuits_, observables_):
    len_circuits_ = len(circuits_)
    num_jobs_     = len_circuits_//max_circuits_
    remainder_    = len_circuits_%max_circuits_
    if (remainder_>0):
        num_jobs_ += 1
    i_start_ = 0
    job_ids_ = []
    for _ in range(num_jobs_):
        i_end_ = min(i_start_ + max_circuits_,len_circuits_)
        job_ = estimator_.run(circuits_[i_start_:i_end_], observables_[i_start_:i_end_])
        job_ids_.append(job_.job_id())
        i_start_ += max_circuits_
    return job_ids_

def check_jobs (service_, max_circuits_, estimator_, circuits_, observables_, job_ids_):
    len_circuits_   = len(circuits_)
    num_jobs_       = len(job_ids_)
    done_           = True
    changed_        = False
    i_start_ = 0
    for ind_job_ in range(num_jobs_):
        job_id_ = job_ids_[ind_job_]
        job_ = service_.job(job_id_)
        if job_.status() is JobStatus.DONE:
            i_start_ += max_circuits_
            continue
        else:
            done_ = False
            # resubmit if there is a problem
            if job_.status() in [JobStatus.ERROR, JobStatus.CANCELLED]:
                #print(job.status())
                #if (job.status() is JobStatus.ERROR):
                #    job.cancel()
                # no need to cancel..
                i_end_ = min(i_start_ + max_circuits_,len_circuits_)
                job_new_ = estimator_.run(circuits_[i_start_:i_end_], observables_[i_start_:i_end_])
                job_ids_[ind_job_] = job_new_.job_id()
                changed_ = True
                s_ = ''
                s_ += '## There was a problem in submitting job_ids[{ind_job}], '.format(ind_job=ind_job_)
                s_ +='\n'
                s_ += '  {}'.format(job_id_)
                s_ +='\n'
                s_ += 'The job is resubmitted. New job_ids[{ind_job}] is, '.format(ind_job=ind_job_)
                s_ +='\n'
                s_ += '  {}'.format(job_ids_[ind_job_])
                print(s_)

            i_start_ += max_circuits_
    return [done_, changed_]


def moniter_jobs (service_, max_circuits_, estimator_, circuits_, observables_, job_ids_):
    len_circuits_   = len(circuits_)
    num_jobs_       = len(job_ids_)
    done_           = False
    changed_        = False
    while not done_:
        done_ = True
        i_start_ = 0
        for ind_job_ in range(num_jobs_):
            job_id_ = job_ids_[ind_job_]
            job_ = service_.job(job_id_)
            if job_.status() is JobStatus.DONE:
                i_start_ += max_circuits_
                continue
            else:
                done_ = False
                # resubmit if there is a problem
                if job_.status() in [JobStatus.ERROR, JobStatus.CANCELLED]:
                    #print(job_.status())
                    i_end_ = min(i_start_ + max_circuits_,len_circuits_)
                    job_new_ = estimator_.run(circuits_[i_start_:i_end_], observables_[i_start_:i_end_])
                    job_ids_[ind_job_] = job_new_.job_id()
                    changed_ = True
                    s_ = ''
                    s_ += '## There was a problem in submitting job_ids[{ind_job}], '.format(ind_job=ind_job_)
                    s_ +='\n'
                    s_ += '  {}'.format(job_id_)
                    s_ +='\n'
                    s_ += 'The job is resubmitted. New job_ids[{ind_job}] is, '.format(ind_job=ind_job_)
                    s_ +='\n'
                    s_ += '  {}'.format(job_ids_[ind_job_])
                    print(s_)

                i_start_ += max_circuits_
        time_lib.sleep(30)
    return [done_, changed_]

def moniter_jobs2 (service_, max_circuits_, estimator_, circuits_, observables_, job_ids_):
    len_circuits_   = len(circuits_)
    num_jobs_       = len(job_ids_)
    done_           = False
    problem_        = False
    while not done_:
        done_ = True
        i_start_ = 0
        for ind_job_ in range(num_jobs_):
            job_id_ = job_ids_[ind_job_]
            job_ = service_.job(job_id_)
            if job_.status() is JobStatus.DONE:
                i_start_ += max_circuits_
                continue
            else:
                done_ = False
                # exit if there is a problem
                if job_.status() in [JobStatus.ERROR, JobStatus.CANCELLED]:
                    s_ = ''
                    s_ += '## There was a problem in submitting job_ids[{ind_job}], '.format(ind_job=ind_job_)
                    s_ +='\n'
                    s_ += '  {}'.format(job_id_)
                    s_ +='\n'
                    print(s_)
                    problem_ = True
                i_start_ += max_circuits_
                if (problem_):
                    break
        if (problem_):
            break
        time_lib.sleep(30)
    return [done_, problem_]


def accumulate_job_results (service_, job_ids_):
    results_ = np.array([],dtype=float)
    for job_id_ in job_ids_:
        job_ = service_.job(job_id_)
        while job_.status() is not JobStatus.DONE:
            time_lib.sleep(30) 
        results_ = np.append(results_,job_.result().values)
    return results_

def cancel_jobs(service_, job_ids_):
    for job_id_ in job_ids_:
        job_ = service.job(job_id_)
        job_.cancel()


# ibm_perth exact time evolution, with nshot

#backend_name = "ibm_perth" # 
backend_name = "ibmq_qasm_simulator" # use the simulator
backend = service.get_backend(backend_name)
num_qubits_backend = backend.num_qubits

nshot    = 4000
max_circs = service.backend(backend_name).max_circuits
print(num_qubits_backend,max_circs)

options = Options()

options.execution.shots = nshot
#options.optimization_level = 3      # default option
#options.resilience_level = 1        # default option

z_hadamard_test = SparsePauliOp.from_sparse_list([('Z',[0],1)],num_qubits=n_qubit+1)
print(z_hadamard_test)


qr = QuantumRegister(n_qubit+1, 'q')
from qiskit.circuit import ParameterVector
params = ParameterVector('θ',17)


num_jobs_save = np.zeros((dim,n_hamiltonians), dtype=int)
job_ids_save = [[[] for _ in range(n_hamiltonians)] for _ in range(dim)]
params_list_save = [[[] for _ in range(n_hamiltonians)] for _ in range(dim)]
indx_status = np.zeros((dim,n_hamiltonians), dtype=int)


#with open('dimer.job_ids','r') as file_:
#    lines = file_.readlines()
#    ind_line = 0
#    for i in range(dim):
#        for alpha in range(1,n_hamiltonians):
#            line = lines[ind_line]
#            job_ids = (line.split())
#            job_ids_save[i][alpha]=job_ids
#            num_jobs_save[i][alpha] = len(job_ids)
#            #print(job_ids_save[i][alpha])
#            ind_line += 1
##
#with open('dimer.params.binary','rb') as file_:
#    [params_list_save] = pickle.load(file_)
#
#with open('dimer.indx.status','rb') as file_:
#    [indx_status] = pickle.load(file_)

# for now 
#print(indx_status)
#print(indx_status[1][6])
#indx_status[1][6]=0
#print(indx_status)


# define two-qubit arbitrary control operator
from typing import List, Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
class ControlledUnitary2Q(Gate):
    def __init__(self, theta: List[ParameterValueType], label: Optional[str] = None):
        """ Creat new CU2Q gate"""
        super().__init__('cu2q',3, theta, label = label)
    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from qiskit.circuit.quantumregister import QuantumRegister
        qr_ = QuantumRegister(3,"q")
        qc_ = QuantumCircuit(qr_, name=self.name)
        qc_.p(self.params[0],qr_[0])
        qc_.cu(self.params[1],self.params[2],self.params[3],0,qr_[0],qr_[1])
        qc_.cu(self.params[4],self.params[5],self.params[6],0,qr_[0],qr_[2])

        # qc_.crxx(self.params[7],qr_[0],[qr_[1],qr_[2]])
        qc_.h(qr_[1])
        qc_.h(qr_[2])
        qc_.cx(qr_[2],qr_[1])
        qc_.crz(self.params[7],qr_[0],qr_[1])
        qc_.cx(qr_[2],qr_[1])
        qc_.h(qr_[2])
        qc_.h(qr_[1])

        # qc_.cryy(self.params[8],qr_[0],[qr_[1],qr_[2]])
        qc_.rx(np.pi/2, qr_[1])
        qc_.rx(np.pi/2, qr_[2])
        qc_.cx(qr_[2],qr_[1])
        qc_.crz(self.params[8],qr_[0],qr_[1])
        qc_.cx(qr_[2],qr_[1])
        qc_.rx(-np.pi/2, qr_[2])
        qc_.rx(-np.pi/2, qr_[1])

        # qc_.rzz(self.params[9],qr_[0],[qr_[1],qr_[2]])
        qc_.cx(qr_[2],qr_[1])
        qc_.crz(self.params[9],qr_[0],qr_[1])
        qc_.cx(qr_[2],qr_[1])

        qc_.cu(self.params[10],self.params[11],self.params[12],0,qr_[0],qr_[1])
        qc_.cu(self.params[13],self.params[14],self.params[15],0,qr_[0],qr_[2])

        self.definition = qc_

from qiskit.quantum_info.synthesis.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
get_angles_and_phase = OneQubitEulerDecomposer(basis='U').angles_and_phase


def ComputeUnitary2QParams(U):
    param_value = np.zeros((16),dtype=float)

    Weyl = TwoQubitWeylDecomposition(U)

    K1r_angles = get_angles_and_phase(Weyl.K1r)
    K1l_angles = get_angles_and_phase(Weyl.K1l)
    K2r_angles = get_angles_and_phase(Weyl.K2r)
    K2l_angles = get_angles_and_phase(Weyl.K2l)

    param_value[0] = K1l_angles[3] + K1r_angles[3] + K2l_angles[3] + K2r_angles[3] + Weyl.global_phase

    param_value[1] = K2r_angles[0]
    param_value[2] = K2r_angles[1]
    param_value[3] = K2r_angles[2]

    param_value[4] = K2l_angles[0]
    param_value[5] = K2l_angles[1]
    param_value[6] = K2l_angles[2]

    param_value[7] = -2.0*Weyl.a
    param_value[8] = -2.0*Weyl.b
    param_value[9] = -2.0*Weyl.c

    param_value[10] = K1r_angles[0]
    param_value[11] = K1r_angles[1]
    param_value[12] = K1r_angles[2]

    param_value[13] = K1l_angles[0]
    param_value[14] = K1l_angles[1]
    param_value[15] = K1l_angles[2]

    return param_value

def ApplyControlledUnitary2Q(qc_: QuantumCircuit, qr_: QuantumRegister, params_: ParameterValueType):
        qc_.p(params_[0],qr_[0])
        qc_.cu(params_[1],params_[2],params_[3],0,qr_[0],qr_[1])
        qc_.cu(params_[4],params_[5],params_[6],0,qr_[0],qr_[2])

        # qc_.crxx(params[7],qr_[0],[qr_[1],qr_[2]])
        qc_.h(qr_[1])
        qc_.h(qr_[2])
        qc_.cx(qr_[2],qr_[1])
        qc_.crz(params_[7],qr_[0],qr_[1])
        qc_.cx(qr_[2],qr_[1])
        qc_.h(qr_[2])
        qc_.h(qr_[1])

        # qc_.cryy(params[8],qr_[0],[qr_[1],qr_[2]])
        qc_.rx(np.pi/2, qr_[1])
        qc_.rx(np.pi/2, qr_[2])
        qc_.cx(qr_[2],qr_[1])
        qc_.crz(params_[8],qr_[0],qr_[1])
        qc_.cx(qr_[2],qr_[1])
        qc_.rx(-np.pi/2, qr_[2])
        qc_.rx(-np.pi/2, qr_[1])

        # qc_.rzz(params[9],qr_[0],[qr_[1],qr_[2]])
        qc_.cx(qr_[2],qr_[1])
        qc_.crz(params_[9],qr_[0],qr_[1])
        qc_.cx(qr_[2],qr_[1])

        qc_.cu(params_[10],params_[11],params_[12],0,qr_[0],qr_[1])
        qc_.cu(params_[13],params_[14],params_[15],0,qr_[0],qr_[2])

# run qzmc;

norms_qzmc               = np.ones((n_hamiltonians,dim),dtype=float)
eigen_energies_qzmc      = np.zeros((n_hamiltonians,dim),dtype=float)
eigen_energies_qzmc[0,:] = eigen_energies_exact[0,:]


result_values_save = [[[] for _ in range(n_hamiltonians)] for _ in range(dim)]

n_trotter = 4
indx      = 0


for i in range(1):
    # initial eps

    eps = eigen_vectors_exact[0,:,i].conj().T@hamiltonians[1].to_matrix()@eigen_vectors_exact[0,:,i]
    eps = eps.real

    # real part measurement
    circuit_real = QuantumCircuit(qr,name='-')
    Circuit_Initialize(circuit_real,i,qr)
    circuit_real.h(qr[0])
    #circuit_real.append(ControlledUnitary2Q(params[:16]),qr)
    ApplyControlledUnitary2Q(circuit_real,qr,params[:16])
    circuit_real.p(params[16],qr[0])
    circuit_real.h(qr[0])

#    how about turning off transpilation?
#   transpilation should be on for qasm
    circuit_real = transpile(circuit_real,backend)
    
    for alpha in range(1,n_hamiltonians):
        # circuit construction and run
        match indx_status[i][alpha]:
            case 2: # job done
                with Session(service=service, backend=backend_name) as session:
                    result_values = accumulate_job_results(service, job_ids_save[i][alpha])
            case 1: # ongoing job
                params_list = params_list_save[i][alpha]
                circuits = [circuit_real.assign_parameters({params: params_}) for params_ in params_list]
                with Session(service=service, backend=backend_name) as session:
                    estimator = Estimator(session=session, options=options)
                    print('waiting...')
                    [done,changed] = moniter_jobs (service, max_circs, estimator, circuits, \
                                    [z_hadamard_test]*len(circuits), job_ids_save[i][alpha])
                    result_values = accumulate_job_results(service, job_ids_save[i][alpha])
            case 0: # new jobs
                # Generate new timelist for a clean start
                problem = True
                while (problem):
                    for i_obs in range(n_obs):
                        for imc in range(nmc):
                            times = []
                            for alpha_ in range(1,alpha):
                                time = rd.gauss(0.0, beta)
                                times.append(time)
        
                            time = rd.gauss(0.0, beta)
                            times.append(time)
        
                            time = rd.gauss(0.0, beta)
                            times.append(time)
        
                            for alpha_ in reversed(range(1,alpha)):
                                time = rd.gauss(0.0, beta)
                                times.append(time)
                            O_timelists[i][alpha][i_obs][imc] = times

                    with open('dimer.time.binary','wb') as file_:
                        pickle.dump([O_timelists],file_)

                    params_list = []
                    # 0; norm measurement
                    i_obs = 0
                    for imc in range(nmc):
                        U_evol = np.identity(dim,dtype=complex)
                        phase  = 0.0
                        nu     = 0
                        # construction of P
                        for alpha_ in range(1,alpha):
                            time = O_timelists[i][alpha][i_obs][imc][nu]
                            nu += 1
                            phase += eigen_energies_qzmc[alpha_,i] * time
                            phase += TrotterTimeEvolution_Phase(time,alpha_)
                            U_evol = ExactTrotterEvolution (time, alpha_, n_trotter, indx)@U_evol

                        time = O_timelists[i][alpha][i_obs][imc][nu]
                        nu += 1
                        phase += eps * time
                        phase += TrotterTimeEvolution_Phase(time,alpha)
                        U_evol = ExactTrotterEvolution (time, alpha, n_trotter, indx)@U_evol

                        # construction of P^\dagger

                        time = O_timelists[i][alpha][i_obs][imc][nu]
                        nu += 1
                        phase += eps * time
                        phase += TrotterTimeEvolution_Phase(time,alpha)
                        U_evol = ExactTrotterEvolution (time, alpha, n_trotter, indx)@U_evol

                        for alpha_ in reversed(range(1,alpha)):
                            time = O_timelists[i][alpha][i_obs][imc][nu]
                            nu += 1
                            phase += eigen_energies_qzmc[alpha_,i] * time
                            phase += TrotterTimeEvolution_Phase(time,alpha_)
                            U_evol = ExactTrotterEvolution (time, alpha_, n_trotter, indx)@U_evol

                        param_temp = ComputeUnitary2QParams(U_evol)

                        params_list.append(np.append(param_temp,phase))

                    # 1; dE1 measurement
                    i_obs = 1
                    nhd1 = len(hamiltonian_diffs[alpha-1])
                    for ihd in range(nhd1):
                        # pass for constant contribution
                        if (hamiltonian_diffs_list[alpha-1][ihd][0]=='I'*n_qubit):
                            continue
                        for imc in range(nmc):
                            U_evol = np.identity(dim,dtype=complex)
                            phase  = 0.0
                            nu     = 0
                            # construction of P
                            for alpha_ in range(1,alpha):
                                time = O_timelists[i][alpha][i_obs][imc][nu]
                                nu += 1
                                phase += eigen_energies_qzmc[alpha_,i] * time
                                phase += TrotterTimeEvolution_Phase(time,alpha_)
                                U_evol = ExactTrotterEvolution (time, alpha_, n_trotter, indx)@U_evol

                            # insertion of (H[alpha]-H[alpha-1])
                            Mat = hamiltonian_diffs[alpha-1].paulis[ihd].to_matrix()

                            U_evol = Mat@U_evol

                            # construction of P; continued

                            time = O_timelists[i][alpha][i_obs][imc][nu]
                            nu += 1
                            phase += eps * time
                            phase += TrotterTimeEvolution_Phase(time,alpha)
                            U_evol = ExactTrotterEvolution (time, alpha, n_trotter, indx)@U_evol

                            # construction of P^\dagger

                            time = O_timelists[i][alpha][i_obs][imc][nu]
                            nu += 1
                            phase += eps * time
                            phase += TrotterTimeEvolution_Phase(time,alpha)
                            U_evol = ExactTrotterEvolution (time, alpha, n_trotter, indx)@U_evol

                            for alpha_ in reversed(range(1,alpha)):
                                time = O_timelists[i][alpha][i_obs][imc][nu]
                                nu += 1
                                phase += eigen_energies_qzmc[alpha_,i] * time
                                phase += TrotterTimeEvolution_Phase(time,alpha_)
                                U_evol = ExactTrotterEvolution (time, alpha_, n_trotter, indx)@U_evol

                            param_temp = ComputeUnitary2QParams(U_evol)

                            params_list.append(np.append(param_temp,phase))
                    # 2; dE2 measurement
                    i_obs = 2
                    if (alpha<n_hamiltonians-1):
                        nhd2 = len(hamiltonian_diffs[alpha])
                    else:
                        nhd2 = 0
                    for ihd in range(nhd2):
                        # pass for constant contribution
                        if (hamiltonian_diffs_list[alpha][ihd][0]=='I'*n_qubit):
                            continue
                        for imc in range(nmc):
                            U_evol = np.identity(dim,dtype=complex)
                            phase  = 0.0
                            nu     = 0
                            # construction of P
                            for alpha_ in range(1,alpha):
                                time = O_timelists[i][alpha][i_obs][imc][nu]
                                nu += 1
                                phase += eigen_energies_qzmc[alpha_,i] * time
                                phase += TrotterTimeEvolution_Phase(time,alpha_)
                                U_evol = ExactTrotterEvolution (time, alpha_, n_trotter, indx)@U_evol

                            time = O_timelists[i][alpha][i_obs][imc][nu]
                            nu += 1
                            phase += eps * time
                            phase += TrotterTimeEvolution_Phase(time,alpha)
                            U_evol = ExactTrotterEvolution (time, alpha, n_trotter, indx)@U_evol

                            # insertion of (H[alpha+1]-H[alpha])
                            Mat = hamiltonian_diffs[alpha].paulis[ihd].to_matrix()

                            U_evol = Mat@U_evol

                            # construction of P^\dagger

                            time = O_timelists[i][alpha][i_obs][imc][nu]
                            nu += 1
                            phase += eps * time
                            phase += TrotterTimeEvolution_Phase(time,alpha)
                            U_evol = ExactTrotterEvolution (time, alpha, n_trotter, indx)@U_evol

                            for alpha_ in reversed(range(1,alpha)):
                                time = O_timelists[i][alpha][i_obs][imc][nu]
                                nu += 1
                                phase += eigen_energies_qzmc[alpha_,i] * time
                                phase += TrotterTimeEvolution_Phase(time,alpha_)
                                U_evol = ExactTrotterEvolution (time, alpha_, n_trotter, indx)@U_evol

                            param_temp = ComputeUnitary2QParams(U_evol)

                            params_list.append(np.append(param_temp,phase))
                    circuits = [circuit_real.assign_parameters({params: params_}) for params_ in params_list]
                    with Session(service=service, backend=backend_name) as session:
                        estimator = Estimator(session=session, options=options)
                        job_ids = run_estimator (max_circs, estimator, circuits, [z_hadamard_test]*len(circuits))
                        print('waiting...')
                        job_ids_save[i][alpha]=job_ids
                        params_list_save[i][alpha] = params_list
                        num_jobs_save[i][alpha] = len(job_ids)
                        indx_status[i][alpha] = 1
                        # save job ids, times, params for monitering and accumulating
                        with open('dimer.job_ids','w') as file_:
                            for i_ in range(dim):
                                for alpha_ in range(1, n_hamiltonians):
                                    s = ''
                                    for job_id in job_ids_save[i_][alpha_]:
                                        s += '  {}'.format(job_id)
                                    s += '\n'
                                    file_.write(s)
                        with open('dimer.params.binary','wb') as file_:
                            pickle.dump([params_list_save],file_)
                        with open('dimer.indx.status','wb') as file_:
                            pickle.dump([indx_status],file_)
                        [done, problem] = moniter_jobs2 (service, max_circs, estimator, circuits, \
                                        [z_hadamard_test]*len(circuits), job_ids_save[i][alpha])
                with Session(service=service, backend=backend_name) as session:
                    result_values = accumulate_job_results(service, job_ids_save[i][alpha])
        result_values_save[i][alpha] = result_values
        indx_status[i][alpha] = 2
        # compute values
        norm    = 0.0
        dE1     = 0.0
        dE2     = 0.0

        i_meas = 0
        # 0; norm
        for imc in range(nmc):
            norm += result_values_save[i][alpha][i_meas]
            i_meas += 1
        # 1; dE1
        nhd1 = len(hamiltonian_diffs[alpha-1])
        for ihd in range(nhd1):
            if (hamiltonian_diffs_list[alpha-1][ihd][0]=='I'*n_qubit):
                continue
            coeff = hamiltonian_diffs_list[alpha-1][ihd][1]
            for imc in range(nmc):
                dE1 +=  coeff * result_values_save[i][alpha][i_meas]
                i_meas += 1

        # 2; dE2
        if (alpha<n_hamiltonians-1):
            nhd2 = len(hamiltonian_diffs[alpha])
        else:
            nhd2 = 0
        for ihd in range(nhd2):
            if (hamiltonian_diffs_list[alpha][ihd][0]=='I'*n_qubit):
                continue
            coeff = hamiltonian_diffs_list[alpha][ihd][1]
            for imc in range(nmc):
                dE2 += coeff * result_values_save[i][alpha][i_meas]
                i_meas += 1
        
        dE1     /=norm
        dE2     /=norm

        norm    /=nmc

        # add constant contributions
        for ihd in range(nhd1):
            if (hamiltonian_diffs_list[alpha-1][ihd][0]=='I'*n_qubit):
                dE1 += hamiltonian_diffs_list[alpha-1][ihd][1]

        for ihd in range(nhd2):
            if (hamiltonian_diffs_list[alpha][ihd][0]=='I'*n_qubit):
                dE2 += hamiltonian_diffs_list[alpha][ihd][1]

        eigen_energies_qzmc[alpha,i] = eigen_energies_qzmc[alpha-1,i] + dE1
        norms_qzmc[alpha,i] = norm

        if (alpha<n_hamiltonians-1):
            eps = eigen_energies_qzmc[alpha,i] + dE2
            eps = eps.real
            
        print(alpha, norms_qzmc[alpha,i], eigen_energies_qzmc[alpha,i]-eigen_energies_exact[alpha,i])
        st = '# {i}/{dim}: {percent}%'.format(i=i+1,dim=dim,percent=((alpha)/(n_hamiltonians-1)*100))
        print(st)

with open('dimer.result.values.binary','wb') as file_:
    pickle.dump([result_values_save],file_)

#with open('dimer.result.values.binary','rb') as file_:
#    [result_values_save] = pickle.load(file_)
