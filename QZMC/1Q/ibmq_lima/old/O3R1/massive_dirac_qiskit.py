import numpy as np
import pickle
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.mappers import LogarithmicMapper
n_qubit = 1
dim     = 2**n_qubit
mapper = LogarithmicMapper()
t = 0.5
nld = 11
lds = np.linspace(0,1,num=nld)
Hld = []
i = 0
for ld in lds:
    h = SpinOp({
        "X_0": t,  
        "Z_0": -1.0 + 2.0* ld
    },
    spin=1/2
    )
    Hld.append(2*mapper.map(h.simplify())) # 2 is due to spin 1/2
    #print(Hld[i])
    i +=1
h = SpinOp({
    "Z_0": 2.0
},
spin=1/2
)
Hp = 2*mapper.map(h.simplify()) # 2 is due to spin 1/2import numpy as np
import pickle
from qiskit_nature.second_q.operators import SpinOp
from qiskit_nature.second_q.mappers import LogarithmicMapper
n_qubit = 1
dim     = 2**n_qubit
mapper = LogarithmicMapper()
t = 0.5
nld = 11
lds = np.linspace(0,1,num=nld)
Hld = []
i = 0
for ld in lds:
    h = SpinOp({
        "X_0": t,  
        "Z_0": -1.0 + 2.0* ld
    },
    spin=1/2
    )
    Hld.append(2*mapper.map(h.simplify())) # 2 is due to spin 1/2
    print(Hld[i])
    i +=1
h = SpinOp({
    "Z_0": 2.0
},
spin=1/2
)
Hp = 2*mapper.map(h.simplify()) # 2 is due to spin 1/2

from qiskit import QuantumRegister
from qiskit import QuantumCircuit

qr = QuantumRegister(n_qubit+1, 'q')

# exact eigenvalues
ell  = np.zeros((nld,dim),dtype=float)
vll  = np.zeros((nld,dim,dim),dtype=complex)
for ild in range(nld):
    El, Vl = np.linalg.eigh(Hld[ild].to_matrix())
    indx = np.argsort(El.real)
    for i in range(dim):
        ell[ild,i]   = El[indx[i]].real
        vll[ild,:,i] = Vl[:,indx[i]]

def ComputeUnitaryParams(U):
    theta = 2.0 * np.arccos(np.abs(U[0,0]))
    if (theta<1E-6):
        theta = 2.0*np.arcsin(np.abs(U[0,1]))
    gamma = np.angle(U[0,0])
    if (theta<1E-10):
        phi   = np.angle(U[1,1]/U[0,0])
        lam   = 0
    else:
        phi   = np.angle(U[1,0]/np.sin(theta/2)) - gamma
        lam   = np.angle(-U[0,1]/np.sin(theta/2)) - gamma
    return theta, phi, lam, gamma

import copy
X = np.zeros((2,2),dtype=complex)
Y = np.zeros((2,2),dtype=complex)
Z = np.zeros((2,2),dtype=complex)

X[0,1] = 1; X[1,0] = 1
Y[0,1] = -1j; Y[1,0] = 1j
Z[0,0] = 1; Z[1,1] = -1


def ExactTimeEvolution (ild, time):
    Vl = copy.deepcopy(vll[ild,:,:])
    evol = np.zeros((dim,dim),dtype=complex)
    exp_d = np.diag(np.exp(-1j*ell[ild,:]*time))
    evol = Vl@exp_d@Vl.conj().T
    return evol

def TrotterTimeEvolution (ild, Nt, time):
    dtime  = time/Nt
    dtx = dtime * t
    dtz = dtime * (2*lds[ild]-1)
    evol = np.identity(dim)
    for it in range(Nt):
        evol_X = np.cos(dtx)*np.identity(dim) - 1j*np.sin(dtx) * X
        evol_Z = np.cos(dtz)*np.identity(dim) - 1j*np.sin(dtz) * Z
        evol = evol_Z@evol
        evol = evol_X@evol
    return evol

# exact results
norm_exact   = np.ones((nld,dim),dtype=float)
E_exact      = np.zeros((nld,dim),dtype=float)
E_exact[0,:] = ell[0,:]
for k in range(dim):
    phi = vll[0,:,k]
    for ild in range(1,nld):
        Proj_matrix = np.outer(vll[ild,:,k],vll[ild,:,k].conj())
        phi = Proj_matrix@phi
        norm_exact[ild,k] = phi.conj()@phi
        E_exact[ild,k] = phi.conj()@Hld[ild].to_matrix()@phi/norm_exact[ild,k]
X_exact = np.zeros((nld,dim),dtype=float)
Y_exact = np.zeros((nld,dim),dtype=float)
Z_exact = np.zeros((nld,dim),dtype=float)
for ild in range(nld):
    for k in range(dim):
        X_exact[ild,k] = vll[ild,:,k].conj().transpose()@X@vll[ild,:,k]
        Y_exact[ild,k] = vll[ild,:,k].conj().transpose()@Y@vll[ild,:,k]
        Z_exact[ild,k] = vll[ild,:,k].conj().transpose()@Z@vll[ild,:,k]

# with machine
# load necessary Runtime libraries
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options
from qiskit.providers import JobStatus
import time as time_lib

service = QiskitRuntimeService(channel="ibm_quantum")

def run_estimator (max_circuits, estimator, circuits, observables):
    len_circuits = len(circuits)
    num_jobs = len_circuits//max_circuits
    remainder = len_circuits%max_circuits
    if (remainder>0):
        num_jobs += 1
    i_start = 0
    job_ids = []
    for _ in range(num_jobs):
        i_end = min(i_start + max_circuits,len_circuits)
        job = estimator.run(circuits[i_start:i_end], observables[i_start:i_end])
        job_ids.append(job.job_id())
        i_start += max_circuits
    return job_ids

def moniter_jobs (service, max_circuits, estimator, circuits, observables, job_ids):
    len_circuits = len(circuits)
    num_jobs     = len(job_ids)
    i_start = 0
    done = False
    while not done:
        done = True
        for ind_job in range(num_jobs):
            job_id = job_ids[ind_job]
            job = service.job(job_id)
            if job.status() is JobStatus.DONE:
                continue
            else:
                done = False
            # resubmit if there is a problem
            if job.status() in [JobStatus.ERROR, JobStatus.CANCELLED]:
                job.calcel()
                i_end = min(i_start + max_circuits,len_circuits)
                job_new = estimator.run(circuits[i_start:i_end], observables[i_start:i_end])
                job_id[ind_job] = job_new.job_id()
            i_start += max_circuits
        time_lib.sleep(10)
    return done

def accumulate_job_results (service, job_ids):
    results = np.array([],dtype=float)
    for job_id in job_ids:
        job = service.job(job_id)
        while job.status() is not JobStatus.DONE:
            time_lib.sleep(5) 
        results = np.append(results,job.result().values)
    return results


# ibmq_lima, exact time evolution, dE version, with nshot, w/o error mitigation
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector, Parameter
import random as rd
import pickle
import copy

backend_name = "ibmq_lima" # use ibmq_lima(5-qubit, minimum pending jobs)

nshot    = 4000
max_circs = service.backend(backend_name).max_circuits

options = Options()

options.execution.shots = nshot
options.optimization_level = 3 # default option
options.resilience_level = 1 # default option


observable = SparsePauliOp('IZ')

beta     = 5

nmc = int(4E2)
Emc      = np.zeros((nld,dim),dtype=float)
normmc   = np.zeros((nld,dim),dtype=float)
dE1norm_raw = np.zeros((nmc,nld,dim),dtype=float)
normmc_raw = np.zeros((nmc,nld,dim),dtype=float)

job_ids_save = []
with open('ibmq_lima.save','rb') as file_:
    [job_ids_save,params_list] = pickle.load(file_)

nhp = len(Hp)
ind_save = 0
max_save = len(job_ids_save)
for k in range(dim):
    normmc[0,k]   = 1
    Emc[0,k]      = ell[0,k]
    eps = vll[0,:,k].conj().T@Hld[1].to_matrix()@vll[0,:,k] # first order perturbated energy
    eps = eps.real
    ind_start = 1
    for ild in range(ind_start,nld):
        if (ind_save>=max_save):
            t_list = []
            for imc in range(nmc):
                t_ = []
                for jld in range(1,ild):
                    r = rd.gauss(0.0, 1.0)
                    time = beta * r
                    t_.append(time)
                r = rd.gauss(0.0, 1.0)
                time = beta * r
                t_.append(time)
                r = rd.gauss(0.0, 1.0)
                time = beta * r
                t_.append(time)
                for jld in reversed(range(1,ild)):
                    r = rd.gauss(0.0, 1.0)
                    time = beta * r
                    t_.append(time)
                t_list.append(t_)

            params_list = []
            for imc in range(nmc):
                # norm measurement
                phase = 0.0
                U_evol = np.identity(2,dtype=complex)
                ind = 0
                for jld in range(1,ild):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol

                time = t_list[imc][ind]
                ind += 1

                time = t_list[imc][ind]
                ind += 1
                phase = phase + eps*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                for jld in reversed(range(1,ild)):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol
                theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)
                params_list.append([theta,phi,lam,gamma,phase])

            for ihp in range(nhp):
                # for the present iteration
                for imc in range(nmc):
                    phase = 0.0
                    U_evol = np.identity(2,dtype=complex)
                    ind = 0
                    for jld in range(1,ild):
                        time = t_list[imc][ind]
                        ind += 1
                        phase = phase + Emc[jld,k]*time
                        U_evol = ExactTimeEvolution(jld,time)@U_evol

                    if (nhp==1):
                        Mat = Hp.to_pauli_op().to_matrix()/Hp.to_pauli_op().coeff
                    else:
                        Mat = Hp.to_pauli_op()[ihp].to_matrix()/Hp.to_pauli_op()[ihp].coeff

                    U_evol = Mat@U_evol

                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + eps*time
                    U_evol = ExactTimeEvolution(ild,time)@U_evol


                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + eps*time
                    U_evol = ExactTimeEvolution(ild,time)@U_evol

                    for jld in reversed(range(1,ild)):
                        time = t_list[imc][ind]
                        ind += 1
                        phase = phase + Emc[jld,k]*time
                        U_evol = ExactTimeEvolution(jld,time)@U_evol
                    theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)

                    params_list.append([theta,phi,lam,gamma,phase])
            for ihp in range(nhp):
                # for the next iteration
                for imc in range(nmc):
                    phase = 0.0
                    U_evol = np.identity(2,dtype=complex)
                    ind = 0
                    for jld in range(1,ild):
                        time = t_list[imc][ind]
                        ind += 1
                        phase = phase + Emc[jld,k]*time
                        U_evol = ExactTimeEvolution(jld,time)@U_evol

                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + eps*time
                    U_evol = ExactTimeEvolution(ild,time)@U_evol

                    if (nhp==1):
                        Mat = Hp.to_pauli_op().to_matrix()/Hp.to_pauli_op().coeff
                    else:
                        Mat = Hp.to_pauli_op()[ihp].to_matrix()/Hp.to_pauli_op()[ihp].coeff
                    U_evol = Mat@U_evol

                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + eps*time
                    U_evol = ExactTimeEvolution(ild,time)@U_evol

                    for jld in reversed(range(1,ild)):
                        time = t_list[imc][ind]
                        ind += 1
                        phase = phase + Emc[jld,k]*time
                        U_evol = ExactTimeEvolution(jld,time)@U_evol
                    theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)

                    params_list.append([theta,phi,lam,gamma,phase])
            # because there is a bug with paramter binding in the present qiskit version, I use this workaround
            circuits = []
            for params in params_list:
                circuit = QuantumCircuit(qr, name='total_circuit')
                circuit.initialize(vll[0,:,k],qr[1:])
                circuit.h(qr[0])
                circuit.cu(params[0],params[1],params[2],params[3],qr[0],qr[1])
                circuit.rz(params[4],qr[0])
                circuit.h(qr[0])
                circuits.append(circuit)
            with Session(service=service, backend=backend_name) as session:
                estimator = Estimator(session=session, options=options)
                job_ids = run_estimator (max_circs, estimator, circuits, [observable]*nmc*(1+nhp+nhp))
                job_ids_save.append(job_ids)
                print('waiting...')
                with open('ibmq_lima.save','wb') as file_:
                    pickle.dump([job_ids_save,params_list],file_)
                done    = moniter_jobs (session.service, max_circs, estimator, circuits, [observable]*nmc*(1+nhp+nhp), job_ids)
                result_values = accumulate_job_results(session.service, job_ids)
        else:
            if (ind_save==(max_save-1)):
                circuits = []
                for params in params_list:
                    circuit = QuantumCircuit(qr, name='total_circuit')
                    circuit.initialize(vll[0,:,k],qr[1:])
                    circuit.h(qr[0])
                    circuit.cu(params[0],params[1],params[2],params[3],qr[0],qr[1])
                    circuit.rz(params[4],qr[0])
                    circuit.h(qr[0])
                    circuits.append(circuit)
            with Session(service=service, backend=backend_name) as session:
                job_ids = job_ids_save[ind_save]
                if (ind_save==(max_save-1)):
                    estimator = Estimator(session=session, options=options)
                    print('waiting...')
                    done    = moniter_jobs (session.service, max_circs, estimator, circuits, [observable]*nmc*(1+nhp+nhp), job_ids)
                result_values = accumulate_job_results(session.service, job_ids)


        for imc in range(nmc):
            normmc_raw[imc,ild,k] = result_values[imc]

        dE1norm_raw[:,ild,k] = 0.0
        for ihp in range(nhp):
            if (nhp==1):
                coeff = float(Hp.to_pauli_op().coeff)
            else:
                coeff = float(Hp.to_pauli_op()[ihp].coeff)
            for imc in range(nmc):
                dE1norm_raw[imc,ild,k] += coeff * result_values[nmc*(ihp+1)+imc]

        dE2norm = 0.0
        for ihp in range(nhp):
            if (nhp==1):
                coeff = float(Hp.to_pauli_op().coeff)
            else:
                coeff = float(Hp.to_pauli_op()[ihp].coeff)
            for imc in range(nmc):
                dE2norm += coeff * result_values[nmc*(ihp+nhp+1)+imc]

        norm    = 0.0
        dE1norm = 0.0
        for imc in range(nmc):
            norm += normmc_raw[imc,ild,k]
            dE1norm += dE1norm_raw[imc,ild,k]

        dE1   = dE1norm/norm
        dE2   = dE2norm/norm
        norm  = norm/nmc
        
        Emc[ild,k] = Emc[ild-1,k] + dE1 * (lds[ild]-lds[ild-1])
        normmc[ild,k] = norm

        if (ild<nld-1):
            eps = Emc[ild,k] + dE2*(lds[ild+1]-lds[ild])
        print(ild, norm, (Emc[ild,k]-ell[ild,k])/(ell[ild,k])*100)
        if (ild<nld-1):
            print(ild, norm, eps ,ell[ild+1,k])
        st = '# {i}/{dim}: {percent}%'.format(i=k+1,dim=dim,percent=((ild)/(nld-1)*100))
        print(st)
        ind_save += 1
file_ = open('massive_dirac.save','wb')
pickle.dump([beta, nmc, lds, Emc, normmc],file_)
file_.close()
