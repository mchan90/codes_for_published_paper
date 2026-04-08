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
    done = False
    while not done:
        job_ids_bk = copy.deepcopy(job_ids)
        changed = False
        done = True
        i_start = 0
        for ind_job in range(num_jobs):
            job_id = job_ids[ind_job]
            job = service.job(job_id)
            if job.status() is JobStatus.DONE:
                i_start += max_circuits
                continue
            else:
                done = False
                # resubmit if there is a problem
                if job.status() in [JobStatus.ERROR, JobStatus.CANCELLED]:
                    #print(job.status())
                    #if (job.status() is JobStatus.ERROR):
                    #    job.cancel()
                    # no need to cancel..
                    i_end = min(i_start + max_circuits,len_circuits)
                    job_new = estimator.run(circuits[i_start:i_end], observables[i_start:i_end])
                    job_ids[ind_job] = job_new.job_id()
                    changed = True

                i_start += max_circuits
        if (changed):
            s = ''
            for ind_job in range(num_jobs):
                s += '  {}'.format(job_ids_bk[ind_job])
            s += '\n'
            print('job id changed from')
            print(s)
            s = ''
            for ind_job in range(num_jobs):
                s += '  {}'.format(job_ids[ind_job])
            s += '\n'
            print('to')
            print(s)
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
options.optimization_level = 3      # default option
options.resilience_level = 1        # default option


observable = SparsePauliOp('IZ')

beta     = 5

nmc = int(4E2)

Emc      = np.zeros((nld,dim),dtype=float)
normmc   = np.zeros((nld,dim),dtype=float)

with open('./norm.E.save','r') as file_:
    ild = 0
    for line in file_:
        if (line.startswith('#')):
            continue
        ls = line.split()
        for k in range(dim):
            normmc[ild,k] = float(ls[4*k+1])
            Emc[ild,k] = float(ls[4*k+3])
            #print(normmc[ild,k],Emc[ild,k])
        ild += 1

job_ids_save = []
#with open('xyz_ibmq_lima.save','r') as file_:
#    for line in file_:
#        job_ids = (line.split())[1:]
#        job_ids_save.append(job_ids)
#
#with open('params.binary','rb') as file_:
#    [params_list] = pickle.load(file_)

ind_save = 0
max_save = len(job_ids_save)
# computation of the fidelity, <X>, <Y>, <Z>

Xmc      = np.zeros((nld,dim),dtype=float)
Ymc      = np.zeros((nld,dim),dtype=float)
Zmc      = np.zeros((nld,dim),dtype=float)
Fmc      = np.zeros((nld,dim),dtype=float)

normmc_raw = np.zeros((nmc,nld,dim),dtype=float)
Xnormmc_raw = np.zeros((nmc,nld,dim),dtype=float)
Ynormmc_raw = np.zeros((nmc,nld,dim),dtype=float)
Znormmc_raw = np.zeros((nmc,nld,dim),dtype=float)
Fnormmc_raw = np.zeros((nmc,nld,dim),dtype=float)

Xmc[0,:] = X_exact[0,:]
Ymc[0,:] = Y_exact[0,:]
Zmc[0,:] = Z_exact[0,:]
Fmc[0,:] = 1


#for k in range(dim):
for k in range(1):      # do only the ground state
    ind_start = 1
    for ild in range(ind_start,nld):
        if (ind_save>=max_save):
            t_list = []
            circuits = []
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
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                time = t_list[imc][ind]
                ind += 1
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                for jld in reversed(range(1,ild)):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol
                theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)
                params_list.append([theta,phi,lam,gamma,phase])

                circuit_real = QuantumCircuit(qr, name='circuit_for_real_part')
                circuit_real.initialize(vll[0,:,k],qr[1:])
                circuit_real.h(qr[0])
                circuit_real.cu(theta,phi,lam,gamma,qr[0],qr[1])
                circuit_real.rz(phase,qr[0])
                circuit_real.h(qr[0])

                circuits.append(circuit_real)

            for imc in range(nmc):
                # X measurement
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
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                U_evol = X@U_evol

                time = t_list[imc][ind]
                ind += 1
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                for jld in reversed(range(1,ild)):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol
                theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)
                params_list.append([theta,phi,lam,gamma,phase])

                circuit_real = QuantumCircuit(qr, name='circuit_for_real_part')
                circuit_real.initialize(vll[0,:,k],qr[1:])
                circuit_real.h(qr[0])
                circuit_real.cu(theta,phi,lam,gamma,qr[0],qr[1])
                circuit_real.rz(phase,qr[0])
                circuit_real.h(qr[0])

                circuits.append(circuit_real)

            for imc in range(nmc):
                # Y measurement
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
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                U_evol = Y@U_evol

                time = t_list[imc][ind]
                ind += 1
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                for jld in reversed(range(1,ild)):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol
                theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)
                params_list.append([theta,phi,lam,gamma,phase])

                circuit_real = QuantumCircuit(qr, name='circuit_for_real_part')
                circuit_real.initialize(vll[0,:,k],qr[1:])
                circuit_real.h(qr[0])
                circuit_real.cu(theta,phi,lam,gamma,qr[0],qr[1])
                circuit_real.rz(phase,qr[0])
                circuit_real.h(qr[0])

                circuits.append(circuit_real)

            for imc in range(nmc):
                # Z measurement
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
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                U_evol = Z@U_evol

                time = t_list[imc][ind]
                ind += 1
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                for jld in reversed(range(1,ild)):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol
                theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)
                params_list.append([theta,phi,lam,gamma,phase])

                circuit_real = QuantumCircuit(qr, name='circuit_for_real_part')
                circuit_real.initialize(vll[0,:,k],qr[1:])
                circuit_real.h(qr[0])
                circuit_real.cu(theta,phi,lam,gamma,qr[0],qr[1])
                circuit_real.rz(phase,qr[0])
                circuit_real.h(qr[0])

                circuits.append(circuit_real)

            U_f = 2.0*np.outer(vll[ild,:,k],vll[ild,:,k].conj())-np.identity(dim)

            for imc in range(nmc):
                # fidelity measurement
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
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                U_evol = U_f@U_evol

                time = t_list[imc][ind]
                ind += 1
                phase = phase + Emc[ild,k]*time
                U_evol = ExactTimeEvolution(ild,time)@U_evol

                for jld in reversed(range(1,ild)):
                    time = t_list[imc][ind]
                    ind += 1
                    phase = phase + Emc[jld,k]*time
                    U_evol = ExactTimeEvolution(jld,time)@U_evol

                theta, phi, lam, gamma = ComputeUnitaryParams(U_evol)
                params_list.append([theta,phi,lam,gamma,phase])

                circuit_real = QuantumCircuit(qr, name='circuit_for_real_part')
                circuit_real.initialize(vll[0,:,k],qr[1:])
                circuit_real.h(qr[0])
                circuit_real.cu(theta,phi,lam,gamma,qr[0],qr[1])
                circuit_real.rz(phase,qr[0])
                circuit_real.h(qr[0])

                circuits.append(circuit_real)


            with Session(service=service, backend=backend_name) as session:
                estimator = Estimator(session=session, options=options)
                job_ids = run_estimator (max_circs, estimator, circuits, [observable]*nmc*5)
                job_ids_save.append(job_ids)
                print('waiting...')
                with open('xyz_ibmq_lima.save','w') as file_:
                    n_save = len(job_ids_save)
                    n_jobs = len(job_ids_save[0])
                    for i_save in range(n_save):
                        s = '{:d}'.format(i_save)
                        for i_jobs in range(n_jobs):
                            s += '  {}'.format(job_ids_save[i_save][i_jobs])
                        s += '\n'
                        file_.write(s)
                with open('params.binary','wb') as file_:
                    pickle.dump([params_list],file_)
                done    = moniter_jobs (session.service, max_circs, estimator, circuits, [observable]*nmc*5, job_ids)
                result_values = accumulate_job_results(session.service, job_ids)
        else:
            if (ind_save==(max_save-1)):
                circuits = []
                for params in params_list:
                    circuit_real = QuantumCircuit(qr, name='circuit_for_real_part')
                    circuit_real.initialize(vll[0,:,k],qr[1:])
                    circuit_real.h(qr[0])
                    circuit_real.cu(params[0],params[1],params[2],params[3],qr[0],qr[1])
                    circuit_real.rz(params[4],qr[0])
                    circuit_real.h(qr[0])
                    circuits.append(circuit_real)
            with Session(service=service, backend=backend_name) as session:
                job_ids = job_ids_save[ind_save]
                if (ind_save==(max_save-1)):
                    estimator = Estimator(session=session, options=options)
                    print('waiting...')
                    done    = moniter_jobs (session.service, max_circs, estimator, circuits, [observable]*nmc*5, job_ids)
                result_values = accumulate_job_results(session.service, job_ids)

        for imc in range(nmc):
            normmc_raw[imc,ild,k]  = result_values[imc]
            Xnormmc_raw[imc,ild,k] = result_values[nmc+imc]
            Ynormmc_raw[imc,ild,k] = result_values[2*nmc+imc]
            Znormmc_raw[imc,ild,k] = result_values[3*nmc+imc]
            Fnormmc_raw[imc,ild,k] = result_values[4*nmc+imc]

        norm   = 0.0
        Xnorm  = 0.0
        Ynorm  = 0.0
        Znorm  = 0.0
        Fnorm  = 0.0

        for imc in range(nmc):
            norm += normmc_raw[imc,ild,k]

        for imc in range(nmc):
            Xnorm += Xnormmc_raw[imc,ild,k]

        for imc in range(nmc):
            Ynorm += Ynormmc_raw[imc,ild,k]

        for imc in range(nmc):
            Znorm += Znormmc_raw[imc,ild,k]

        for imc in range(nmc):
            Fnorm += Fnormmc_raw[imc,ild,k]

        Xmc[ild,k] = Xnorm/norm
        Ymc[ild,k] = Ynorm/norm
        Zmc[ild,k] = Znorm/norm
        Fmc[ild,k] = 0.5 * (1.0 + Fnorm/norm)
        norm  = norm/nmc

        
        st = '# {i}/{dim}: {percent}%'.format(i=k+1,dim=dim,percent=((ild)/(nld-1)*100))
        st_fidelity = ', |<Φ|Ψ>|^2 =  {fidelity}'.format(fidelity=Fmc[ild,k])
        print(st+st_fidelity)
        ind_save += 1

# standard deviation calculation with the bootstrap method
import random as rd
std_X = np.zeros((nld,dim),dtype=float)
std_Y = np.zeros((nld,dim),dtype=float)
std_Z= np.zeros((nld,dim),dtype=float)
std_F= np.zeros((nld,dim),dtype=float)
n_boot = 1000
for k in range(1):
    for ild in range(1,nld):
        X_boot   = np.zeros((n_boot),dtype=float)
        Y_boot   = np.zeros((n_boot),dtype=float)
        Z_boot   = np.zeros((n_boot),dtype=float)
        F_boot   = np.zeros((n_boot),dtype=float)
        for i_boot in range(n_boot):
            norm_ = 0.0
            X_    = 0.0
            Y_    = 0.0
            Z_    = 0.0
            F_    = 0.0
            for imc in range(nmc):
                jmc = rd.randrange(nmc)
                norm_ += normmc_raw[jmc,ild,k]
                X_ += Xnormmc_raw[jmc,ild,k]
                Y_ += Ynormmc_raw[jmc,ild,k]
                Z_ += Znormmc_raw[jmc,ild,k]
                F_ += Fnormmc_raw[jmc,ild,k]
            X_ = X_/norm_
            Y_ = Y_/norm_
            Z_ = Z_/norm_
            F_ = 0.5 * (1.0 + F_/norm_) 
            X_boot[i_boot] = X_
            Y_boot[i_boot] = Y_
            Z_boot[i_boot] = Z_
            F_boot[i_boot] = F_
        X_boot_mean   = 0.0
        Y_boot_mean   = 0.0
        Z_boot_mean   = 0.0
        F_boot_mean   = 0.0
        for i_boot in range(n_boot):
            X_boot_mean   += X_boot[i_boot]
            Y_boot_mean   += Y_boot[i_boot]
            Z_boot_mean   += Z_boot[i_boot]
            F_boot_mean   += F_boot[i_boot]
        X_boot_mean /= n_boot
        Y_boot_mean /= n_boot
        Z_boot_mean /= n_boot
        F_boot_mean /= n_boot

        var_norm = 0.0
        var_X = 0.0
        var_Y = 0.0
        var_Z = 0.0
        var_F = 0.0
        for i_boot in range(n_boot):
            var_X += (X_boot[i_boot]-X_boot_mean)**2
            var_Y += (Y_boot[i_boot]-Y_boot_mean)**2
            var_Z += (Z_boot[i_boot]-Z_boot_mean)**2
            var_F += (F_boot[i_boot]-F_boot_mean)**2
        var_X    /= (n_boot-1)
        var_Y    /= (n_boot-1)
        var_Z    /= (n_boot-1)
        var_F    /= (n_boot-1)
        std_X[ild,k] = np.sqrt(var_X) 
        std_Y[ild,k] = np.sqrt(var_Y) 
        std_Z[ild,k] = np.sqrt(var_Z) 
        std_F[ild,k] = np.sqrt(var_F) 

with open('xyz.save','w') as file_:
    s = '# λ , X, std(X), Y, std(Y), Z, std(Z), F, std(F)'
    for ild in range(nld):
        s = '{:.16e}'.format(lds[ild])
        for k in range(1):
            s += '  {:.16e}  {:.16e}  {:.16e}  {:.16e}  {:.16e}  {:.16e}  {:.16e}  {:.16e}'.format(Xmc[ild,k],std_X[ild,k],Ymc[ild,k],std_Y[ild,k],Zmc[ild,k],std_Z[ild,k],Fmc[ild,k],std_F[ild,k])
        s += '\n'
        file_.write(s)
