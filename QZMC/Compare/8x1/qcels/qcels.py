# %%
from mpi4py import MPI
from datetime import datetime
import numpy as np
import copy
import psutil
from scipy.special import ive
import random as rd
import pickle
from scipy import sparse
import gc
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    Lattice,
    LineLattice,
    SquareLattice
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel, QuadraticHamiltonian
from qiskit_nature.second_q.circuit.library import BogoliubovTransform

def memory_usage(message: str = 'debug'):
    # this memory_usage function is imported from https://jybaek.tistory.com/895
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 30 # Bytes to GiB
    print(f"[{message}] memory usage: {rss: 10.5f} GiB")


def single_line (lines):
    return ''.join(lines.splitlines())

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()

n_x     = 8
n_y     = 1
n_site  = n_x * n_y
n_qubit = 2*n_site
dim = 2**n_qubit

n_dimer = n_site//2
# For a chain
n_inter = n_dimer 
# For a dimer 
#n_inter = 2*n_dimer


# %%
Uc           = 10.0
mu           = Uc/2.0 # half-filling fermi level
n_electrons  = n_site # half-filling
t_hop        = 1.0
t_intra      = t_hop
t_inters     = np.linspace(0.0,1.0,2)
nt_inter     = len(t_inters)
for it_inter in range(nt_inter):
    t_inters[it_inter] *= t_hop
kinetic_parts       = []
hamiltonians        = []
bc = BoundaryCondition.OPEN

# make interaction part first (fixed)

if (n_y==1):
    empty  = LineLattice(num_nodes=n_x,boundary_condition=bc, edge_parameter=0.0,\
                            onsite_parameter=0.0)
else:
    empty = SquareLattice(rows=n_x,cols=n_y, boundary_condition=bc, edge_parameter=(0,0),\
                        onsite_parameter=0.0)
interaction_part = FermiHubbardModel(lattice=empty, onsite_interaction=Uc).second_q_op().simplify()
it_inter            = 0
for t_inter in t_inters:
    if (n_y==1):
        square  = LineLattice(num_nodes=n_x,boundary_condition=bc, edge_parameter=-t_inter,\
                                onsite_parameter=-mu)
    else:
        square = SquareLattice(rows=n_x,cols=n_y, boundary_condition=bc, edge_parameter=(-t_inter,-t_inter),\
                                onsite_parameter=-mu)
    square_modified_graph = square.graph
    # replace intra dimer hoppings
    # assumes dimer direction in x-axis
    for i_dimer in range(n_dimer):
        i_site = 2*i_dimer
        square_modified_graph.update_edge(i_site,i_site+1,-t_intra)
    
    square_modified = Lattice(square_modified_graph)
    
    hopping_matrix = np.zeros((n_qubit,n_qubit),dtype=complex)
    for i_site, j_site, weight in square_modified_graph.weighted_edge_list():
        i_up = 2*i_site
        i_dn = 2*i_site + 1
        j_up = 2*j_site
        j_dn = 2*j_site + 1
        hopping_matrix[i_up,j_up] = weight
        hopping_matrix[i_dn,j_dn] = weight
        hopping_matrix[j_up,i_up] = weight # conjugate actually
        hopping_matrix[j_dn,i_dn] = weight

    kinetic_part = QuadraticHamiltonian(hermitian_part=hopping_matrix)
    kinetic_parts.append(kinetic_part)
    hamiltonian = kinetic_part.second_q_op().simplify() + interaction_part

    # constant_term, to match with references
    # Create the constant term as a FermionicOp
    constant_term = FermionicOp({'': 
        0.25 * Uc * n_site
    })

    hamiltonian += constant_term

    mapper = JordanWignerMapper()
    hamiltonians.append(mapper.map(hamiltonian))
    if (core==0):
        print(it_inter,hamiltonians[it_inter])
    it_inter += 1


# %%
n_hamiltonians = len(hamiltonians)
# sectorize
nsec = 1
dim_sub = [0 for _ in range(nsec)]
indx_sub = [[] for _ in range(nsec)]
iindx_sub = [[] for _ in range(nsec)]

n_up_target = [0 for _ in range(nsec)]
n_dn_target = [0 for _ in range(nsec)]

# consider only half-filled sector
n_up_target[0] = n_site//2
n_dn_target[0] = n_site//2

for isec in range(nsec):
    dim_sub[isec] = 0
    indx_sub[isec] = []

    for i in range(dim):
        s = f'{i:0{n_qubit}b}'
        n_up = 0
        for j in range(n_site):
            n_up += int(s[2*j])
        n_dn = 0
        for j in range(n_site):
            n_dn += int(s[2*j+1])
        if (n_up==n_up_target[isec] and n_dn==n_dn_target[isec]):
            dim_sub[isec] += 1
            indx_sub[isec].append(i)

    if (core==0):
        st = '# dimension of subspace with (n_up,n_dn) = ({n_up},{n_dn}) = {dim_sub}'.format(n_up=n_up_target[isec],n_dn=n_dn_target[isec],dim_sub=dim_sub[isec])
        print(st)
    indx_sub[isec] = np.array(indx_sub[isec])
    # inverse of indx_sub
    iindx_sub[isec] = -np.ones((dim),dtype=int)
    for i in range(dim_sub[isec]):
        iindx_sub[isec][indx_sub[isec][i]] = i


# %%
# using symmetry
def ApplyParticleHole(i):
    bit_expr = bin(i)[2:].zfill(n_qubit)
    operated = ''
    for k in range(n_qubit):
        if (bit_expr[k] =='0'):
            operated += '1'
        else :
            operated += '0'
    return int(operated,2)
def ApplySpinReverse(i):
    bit_expr = bin(i)[2:].zfill(n_qubit)
    operated = ''
    n_orbital = n_qubit//2
    for k in range(n_orbital):
        i_up = 2*k
        i_dn = 2*k+1
        operated += bit_expr[i_dn]
        operated += bit_expr[i_up]
    return int(operated,2)
def ApplyOrderReverse(i):
    bit_expr = bin(i)[2:].zfill(n_qubit)
    operated = ''
    for k in reversed(range(n_qubit)):
        operated += bit_expr[k]
    return int(operated,2)

def OrderReverseSign(i):
    return 1
# compose particle hole symmetry operator Ph
isec = 0
Or = np.zeros((dim_sub[0],dim_sub[0]),dtype=complex)
for jj in range(dim_sub[0]):
    j = indx_sub[isec][jj]
    i = ApplyOrderReverse(j)
    ii = iindx_sub[isec][i]
    Or[ii,jj] = OrderReverseSign(i)


def ParticleHoleSign(i):
    bit_expr = bin(i)[2:].zfill(n_qubit)
    operated = ''
    n_orbital = n_qubit//2
    sign_exp = 0
    for k in range(n_orbital):
        i_up = 2*k
        i_dn = 2*k+1
        sign_exp += (int(bit_expr[i_up]) + int(bit_expr[i_dn])) * k
    sign_exp = sign_exp % 2
    return (-1) ** sign_exp
# compose particle hole symmetry operator Ph
isec = 0
Ph = np.zeros((dim_sub[0],dim_sub[0]),dtype=complex)
for jj in range(dim_sub[0]):
    j = indx_sub[isec][jj]
    i = ApplyParticleHole(j)
    ii = iindx_sub[isec][i]
    Ph[ii,jj] = ParticleHoleSign(i)

def SpinReverseSign(i):
    bit_expr = bin(i)[2:].zfill(n_qubit)
    operated = ''
    n_orbital = n_qubit//2
    sign_exp = 0
    for k in range(n_orbital):
        i_up = 2*k
        i_dn = 2*k+1
        sign_exp += int(bit_expr[i_up]) * int(bit_expr[i_dn])
    sign_exp = sign_exp % 2
    return (-1) ** sign_exp
# compose spin reverse symmetry operator Sr
isec = 0
Sr = np.zeros((dim_sub[0],dim_sub[0]),dtype=complex)
for jj in range(dim_sub[0]):
    j = indx_sub[isec][jj]
    i = ApplySpinReverse(j)
    ii = iindx_sub[isec][i]
    Sr[ii,jj] = SpinReverseSign(i)

# compose the projection operator
isec = 0
identity = np.identity((dim_sub[isec]),dtype=complex)
projection = (identity + Or) 
projection = (identity + Ph) @ projection
if (n_dimer%2==0):
    projection = (identity + Sr) @ projection
else:
    projection = (identity - Sr) @ projection
projection /= 8

# symmetry sectorization
isec = 0
done = [False] * dim_sub[isec]
S = []
indx_S = 0
for i in range(dim_sub[isec]):
    if (done[i]):
        continue
    S.append([i])
    # apply symmetry operation
    ss = []
    for l in S[indx_S]:
        j = ApplyOrderReverse(indx_sub[isec][l])
        jj = iindx_sub[isec][j]
        if (jj not in S[indx_S]):
            ss += [jj]
            done[jj] = True
    S[indx_S] += ss

    ss = []
    for l in S[indx_S]:
        j = ApplySpinReverse(indx_sub[isec][l])
        jj = iindx_sub[isec][j]
        if (jj not in S[indx_S]):
            ss += [jj]
            done[jj] = True
    S[indx_S] += ss

    ss = []
    for l in S[indx_S]:
        j = ApplyParticleHole(indx_sub[isec][l])
        jj = iindx_sub[isec][j]
        if (jj not in S[indx_S]):
            ss += [jj]
            done[jj] = True
    S[indx_S] += ss

    indx_S += 1
num_S = len(S)


# find the reduced basis by finding eigenstate of projection operator
# it can be diffent. But, in thsi case, dim_reduced = num_S because all S has projection=1 eigenstate
dim_reduced = num_S
basis_transform = np.zeros((num_S,dim_sub[isec]),dtype=complex)
for indx_S in range(num_S):
    dim = len(S[indx_S])
    project_small = np.zeros((dim,dim),dtype=complex)
    for jjj in range(dim):
        jj = S[indx_S][jjj]
        for iii in range(dim):
            ii = S[indx_S][iii]
            project_small[iii,jjj] = projection[ii,jj]
    eigen_e, eigen_v = np.linalg.eigh(project_small)
    # only eigenvalue=1 (maximum) is valid
    #print(np.sum(np.abs(eigen_e)>0.5))
    for jjj in range(dim):
            jj = S[indx_S][jjj]
            basis_transform[indx_S,jj] = np.conj(eigen_v[jjj,-1])
#print(basis_transform[0,:])
if (core==0):
    print('reduced dimension: ', dim_reduced)

# %%
eigen_energies_exact   = []
eigen_vectors_exact   = []
H_subs = []
for isec in range(nsec):
    eigen_energies_exact.append(np.zeros((n_hamiltonians,dim_reduced),dtype=float))
    eigen_vectors_exact.append(np.zeros((n_hamiltonians,dim_reduced,dim_reduced),dtype=complex))
    H_subs.append([])

start_time = datetime.now()
for isec in range(nsec):
    eigen_e               = np.zeros((dim_reduced),dtype=float)
    eigen_v               = np.zeros((dim_reduced,dim_reduced),dtype=complex)
    for alpha in range(n_hamiltonians):
        # project hamiltonian on to specified sector
        H_sparse = hamiltonians[alpha].to_matrix(sparse=True)
        H_sparse.eliminate_zeros()
        jsec = isec
        row      = []
        col      = []
        data     = []
        for ii in range(dim_sub[isec]):
            i = indx_sub[isec][ii]
            for ind in range(H_sparse.indptr[i],H_sparse.indptr[i+1]):
                # j is always in indx_sub[isec], because the Hamiltonian does not mix it
                #print(i,j)
                j = H_sparse.indices[ind]
                jj = iindx_sub[jsec][j]
                row.append(jj)
                col.append(ii)
                #print(ii,jj)
                data.append(H_sparse.data[ind])
        H_sub = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[jsec], dim_sub[isec]))

        H_sub_reduced = basis_transform@(H_sub.toarray())@basis_transform.conj().T

        H_subs[isec].append(H_sub_reduced)

        # diagonalize sectorized hamiltonian
        i_core = alpha%cores
        if (core==i_core):
            eigen_e, eigen_v = np.linalg.eigh(H_sub_reduced)
            eigen_energies_exact[isec][alpha,:]   = eigen_e
            eigen_vectors_exact[isec][alpha,:,:] = eigen_v
    del eigen_e, eigen_v
    gc.collect()
comm.Barrier()
bcast_start_time = datetime.now()
for isec in range(nsec):
    for alpha in range(n_hamiltonians):
        i_core = alpha%cores
        comm.Bcast([eigen_energies_exact[isec][alpha,:] ,MPI.DOUBLE],root=i_core)
        comm.Bcast([eigen_vectors_exact[isec][alpha,:,:],MPI.DOUBLE_COMPLEX],root=i_core)
end_time = datetime.now()
elapsed = end_time-start_time
elapsed = elapsed.total_seconds()
if (core==0):
    st = '# {percent}%, elapsed time = {elapsed} secs'.format(percent=(100),elapsed=elapsed)
    memory_usage(st)
    st = 'Bcast time = {elapsed} secs'.format(elapsed=(end_time-bcast_start_time).total_seconds())
    print(st)


# %%
def Apply_ExactEvolution(isec, alpha, eps, time, v):
    w = v.conj()@eigen_vectors_exact[isec][alpha,:,:]
    vec = np.exp(-1j*time*(eigen_energies_exact[isec][alpha,:]-eps))
    w = vec*w.conj()
    w = eigen_vectors_exact[isec][alpha,:,:]@w
    return w

def Apply_ExactGaussian(isec, alpha, eps, beta, v):
    w = v.conj()@eigen_vectors_exact[isec][alpha,:,:]
    vec = np.exp(-0.5 * beta ** 2*(eigen_energies_exact[isec][alpha,:]-eps)**2)
    w = vec*w.conj()
    w = eigen_vectors_exact[isec][alpha,:,:]@w
    return w


# %%
# kinetic parts
hamiltonians_kinetic = []
for alpha in range(n_hamiltonians):
    h = kinetic_parts[alpha].second_q_op().simplify()
    hk = mapper.map(h)
    hamiltonians_kinetic.append(hk)
    if (core==0):
        print(hamiltonians_kinetic[alpha])
# interacting-parts
constant_term = FermionicOp({'': 
    0.25 * Uc * n_site
})
hamiltonian_coulomb = mapper.map(interaction_part+constant_term)
if (core==0):
    print(hamiltonian_coulomb)

# %%
tau_scale = np.pi/(4.0*np.abs(eigen_energies_exact[0][-1][0]))


# %%
if (core==0):
    print(tau_scale)


# %%
# isec = 0
# phi_start = eigen_vectors_kinetic[isec][-1,:,0] # kinetic part solution (Hartree Fock)
# eta = np.abs(phi_start.conj().T@eigen_vectors_exact[isec][-1,:,0])**2
# construction of starting vector that can be easily constructed within a quantum cirucit
if (n_x==4):
    if (np.abs(Uc-4.0)<1e-8):
        eta_0 = 0.4570251 # gives eta = 0.4 for U=4, 4x1
    elif (np.abs(Uc-10.0)<1e-8):
        eta_0 = 0.434594 # gives eta = 0.4 for U=10, 4x1
elif (n_x==8):
    if (np.abs(Uc-4.0)<1e-8):
        eta_0 = 0.639429 # gives eta = 0.4 for U=4, 4x1
    elif (np.abs(Uc-10.0)<1e-8):
        eta_0 = 0.544645           # gives eta = 0.4 for U=10, 4x1
else:
    eta_0 = 1.0

isec = 0
phi_start_large = np.zeros((dim_sub[isec]),dtype=complex)
theta  = 2.0 * np.arctan(-0.5/t_hop*(Uc/2 + np.sqrt(Uc**2/4+4*t_hop**2)))
# make it different
theta -=    2.0 * np.arccos(eta_0**(1.0/n_site))
basis_index_dimer = [3, 6, 9, 12]
values_dimer      = [np.cos(theta/2.0), np.sin(theta/2.0), -np.sin(theta/2.0), np.cos(theta/2.0)]
basis_indices = []
values        = []
dim_basis = 4**n_dimer
#normsum = 0.0
for i_basis in range(dim_basis):
    indx = i_basis
    ind = 0
    ss  = ''
    value = (1.0/2.0)**(n_dimer/2)
    for i_dimer in range(n_dimer):
        jj = indx%4
        indx = indx// 4
        ind += basis_index_dimer[jj] * 16**i_dimer
        ss = str(jj) + ss
        value = value * values_dimer[jj]
    basis_indices.append(ind)
    values.append(value)
    phi_start_large[iindx_sub[isec][ind]] = value
    #if (core==0):
    #    print(jj, ind)
    #normsum += value**2
#if (core==0):
#    print('normsum=',normsum)

phi_start = basis_transform@phi_start_large
eta = np.abs(phi_start@eigen_vectors_exact[isec][-1,:,0])**2
if (core==0):
    print(eta)


# %%
def compute_amplitude(time):
    isec = 0
    alpha = n_hamiltonians-1
    phi = copy.copy(phi_start)
    phi = Apply_ExactEvolution(isec,alpha,0.0,time,phi)
    amplitude = phi_start.conj()@phi
    return amplitude


def compute_trotter_amplitude(time, n_trotter, indx):
    isec = 0
    alpha = n_hamiltonians-1
    phi = copy.copy(phi_start)
    phi = Apply_TrotterEvolution(isec, alpha, time, 0.0, n_trotter, indx,phi)
    amplitude = phi_start.conj()@phi
    return amplitude

def compute_amplitude_samples(amplitude, n_shot):
    # real part
    computed_value = amplitude.real

    # # shot errors
    p_up = (computed_value + 1.0)/2.0
    if (p_up>1.0 or p_up<0.0):
        shot_error = 0.0
    else:
        sample_shot = np.random.binomial(n_shot,p_up)
        shot_error = 2*(sample_shot/(n_shot) - p_up)
    computed_value += shot_error

    amp_sample_real = computed_value
    

    # imag part
    computed_value = amplitude.imag

    # # shot errors
    p_up = (computed_value + 1.0)/2.0
    if (p_up>1.0 or p_up<0.0):
        shot_error = 0.0
    else:
        sample_shot = np.random.binomial(n_shot,p_up)
        shot_error = 2*(sample_shot/(n_shot) - p_up)
    computed_value += shot_error

    amp_sample_imag = computed_value

    amplitude_sample = amp_sample_real + 1j * amp_sample_imag
    return amplitude_sample




# %%
from scipy.optimize import minimize
def qcels_opt_fun(x, ts, Z_est):
    NT = ts.shape[0]
    Z_fit=np.zeros(NT,dtype = complex)
    Z_fit=(x[0]+1j*x[1])*np.exp(-1j*x[2]*ts)
    return (np.linalg.norm(Z_fit-Z_est)**2/NT)

def qcels_opt(ts, Z_est, x0, bounds = None, method = 'SLSQP'):

    fun = lambda x: qcels_opt_fun(x, ts, Z_est)
    if( bounds ):
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)
    else:
        res=minimize(fun,x0,method = 'SLSQP',bounds=bounds)

    return res


# %%
isec = 0
eta = np.abs(phi_start.conj().T@eigen_vectors_exact[isec][-1,:,0])**2
populations = np.zeros(dim_reduced)
for k in range(dim_reduced):
    populations[k] = np.abs(phi_start.conj().T@eigen_vectors_exact[isec][-1,:,k])**2
#print(eta,np.sum(populations))

# heuristic estimate of relative gap
indx=0
while sum(populations[0:indx+1]) <4*eta/3: # comes infinite loop for large enough eta
    indx +=1
relative_gap = (eigen_energies_exact[isec][-1,indx]-eigen_energies_exact[isec][-1,0])*tau_scale
if (core==0):
    s = 'relative gap= ' + str(relative_gap)
    print(s)

# %%
# to filter
from numpy.polynomial.chebyshev import chebval

def M_unnormalized(x,d,delta):
    inside = 1.0 + 2.0 * ((np.cos(x)-np.cos(delta))/(1+np.cos(delta)))
    c = np.zeros(d+1)
    c[-1] = 1.0
    return chebval(inside,c)
    
    
def M_fourier_coeffs(d,delta):
    M = 2*d + 1 # number of nodes to perform FFT
    x = (2*np.pi/M)*np.arange(M)
    y = M_unnormalized(x,d,delta)
    coeffs_raw = np.fft.fft(y,M)
    return (1.0/M)*coeffs_raw


def reconstruct_from_fourier(x,fourier_coeffs):
    d = (fourier_coeffs.shape[0]-1)//2
    y = np.zeros(fourier_coeffs.shape)
    k = np.zeros(fourier_coeffs.shape)
    k[:d+1] = np.arange(d+1)
    k[d+1:] = np.arange(-d,0) # k = 0,1,...,d,-d,-d+1,...,-1
    exp_array = np.exp(1.0j*np.tensordot(k,x,axes=0))
    return np.matmul(fourier_coeffs,exp_array)
        
    
def M_fourier_coeffs_normalized(d,delta):
    coeffs = M_fourier_coeffs(d,delta)
    coeffs = coeffs/(coeffs[0]*2*np.pi)
    return coeffs
    

def H_fourier_coeffs(d):
    H_coeffs = np.zeros(2*d+1,dtype=np.complex128)
    H_coeffs[0] = 0.5
    H_coeffs[1:d+1] = -1.0j/np.arange(1,d+1)/np.pi*(np.arange(1,d+1)%2)
    H_coeffs[d+1:] = -1.0j/np.arange(-d,0)/np.pi*(np.arange(-d,0)%2)
    return H_coeffs
    

def F_fourier_coeffs(d,delta):
    H_coeffs = H_fourier_coeffs(d)
    M_coeffs = M_fourier_coeffs_normalized(d,delta)
    coeffs = 2*np.pi * H_coeffs * M_coeffs
    return coeffs
    
def find_max_error(F_coeffs,delta,Nsample=0):
    if Nsample == 0:
        Nsample = F_coeffs.shape[0]*10
    x = np.pi*(2*np.random.rand(Nsample)-1)
    y = reconstruct_from_fourier(x,F_coeffs)
    y_target = x>0
    x_valid = (np.abs(x)>delta) * (np.abs(x+np.pi)>delta) * (np.abs(x-np.pi)>delta)
    return np.max(np.abs(y-y_target)*x_valid)
    
    
def compute_total_evolution_time(F_coeffs):
    d = (F_coeffs.shape[0]-1)//2
    T = np.zeros(2*d+1)
    T[:d+1] = np.arange(d+1)
    T[d+1:] = -np.arange(-d,0)
    return np.dot(T,np.abs(F_coeffs))

# fix random number seed
# np.random.seed(42)


# %%
n_shot = 2048
n_iter = 30
n_time = 5
#small_t_list_qcels = [25, 50, 100, 200, 600, 800]
#time_max_list = 1150+T0/4*(np.arange(n_test_qcels))
#time_max_list = [1.5 * n_time*2**j for j in range(1,12)]
#time_max_list = [10.0 * n_time*2**j for j in range(1,12)]
#time_max_list = [20.0 * j for j in range(1,300)]
time_max_list = [10, 50, 160, 500, 1800, 3600, 7200, 14400, 28800]
#time_max_list = [50, 100, 400, ]
#time_max_list = [26, 52, 104, 156, 280, 520, 1040, 2080, 4160, 8320]
#time_max_list = np.arange(100,520)
#T0 = 8000
#small_T_list_QCELS=[25,50,100,200,600,800]
#time_max_list = 1150+T0/4*(np.arange(10)) #circuit depth for QCELS
#time_max_list = np.hstack((small_T_list_QCELS,time_max_list))

max_evolution_time_list = np.zeros(len(time_max_list),dtype=float)
error_for_time_list     = np.zeros(len(time_max_list),dtype=float)


# %%
# preprocess
d = int(15/relative_gap) 
n_sample = 4096 # int(15/eta**2*np.log(d))
delta = relative_gap/4
# filter preparation
indxs = np.arange(2*d+1)
ks = np.zeros((2*d+1),dtype=int)
for j in range(1,d+1):
    ks[j] = j
    ks[j+d] = -j

C = F_fourier_coeffs(d,delta)
Coeffs = np.zeros((2*d+1),dtype=complex)
Coeffs[0] = 0.5
for j in range(1,d+1):
    Coeffs[j] = C[j]
    Coeffs[j+d] = -C[j] 

# prepare importance sampling
Coeffs_sum = np.abs(Coeffs[0])
for j in range(1,2*d+1):
    Coeffs_sum += np.abs(Coeffs[j])
Coeffs_prob = np.abs(Coeffs)/Coeffs_sum

Coeffs_ratio = np.zeros((2*d+1),dtype=complex)
for j in range(2*d+1):
    if (Coeffs_prob[j]>1e-10):
        Coeffs_ratio[j] = Coeffs_sum * Coeffs[j]/np.abs(Coeffs[j])
# find C99
# sum_tmp = Coeffs_prob[0] + np.sum(Coeffs_prob[d+1:])
# for i in range(1,d+1):
#     sum_tmp += Coeffs_prob[i]
#     if (sum_tmp>0.99):
#         C99 = i/d
#         break


# %%
# exact filter
def ExactFilteredEvolution (isec, alpha, ks, Coeffs, eps, time):
    Vl = copy.deepcopy(eigen_vectors_exact[isec][alpha,:,:])
    evol = np.zeros((dim_reduced,dim_reduced),dtype=complex)
    vec = np.zeros(dim_reduced,dtype=complex)
    for j in range(2*d+1):
        for k in range(dim_reduced):
            vec[k] += Coeffs[j] * np.exp(-1j*time*eigen_energies_exact[isec][alpha,k]-1j*ks[j]*tau_scale*(eigen_energies_exact[isec][alpha,k]-eps))
    exp_d = np.diag(vec)
    evol = Vl@exp_d@Vl.conj().T
    return evol
#def Apply_ExactFilter (isec, alpha, ks, Coeffs, eps, phi):
#    phi_ = phi
#    ef = ExactFilter(isec, alpha, ks, Coeffs, eps)
#    phi_ = ef@phi_
#    #w = np.zeros((dim_sub[isec]),dtype=complex)
#    #for j in range(2*d+1):
#    #    w += Coeffs[j] * Apply_ExactEvolution(isec,alpha,eps,ks[j]*tau_scale,phi)
#    return phi_
#
def compute_amplitude_with_filter(ks, Coeffs, eps, time):
    isec = 0
    alpha = n_hamiltonians-1
    phi = copy.copy(phi_start)
    phi = ExactFilteredEvolution(isec, alpha, ks, Coeffs, eps, time)@phi
    amplitude = phi_start.conj()@phi
    return amplitude

# %%
# QCELS without sampling error
indx_time = 0
n_iter = 1
# estimation of the upperbound
lambda_init = eigen_energies_exact[0][0,0]*tau_scale

# center of the eigenvalue filter
x_center = lambda_init + relative_gap/2.0

for time_max in time_max_list:


    # compute time list first
    n_level = round(np.log2(time_max/n_time))

    amplitudes_qcels        = np.zeros((n_iter,n_level+1,n_time),dtype=complex)
    tau_scale_qcels         = np.zeros((n_level+1),dtype=float)
    times_qcels             = np.zeros((n_level+1,n_time),dtype=float)
    for i_level in range(n_level+1):
        tau_scale_qcels[i_level] = time_max/n_time/(2**(n_level-i_level))
        times_qcels[i_level,:] = tau_scale_qcels[i_level] * np.arange(n_time)
    # find max_time
    max_time   = 0.0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            for i_time in range(n_time):
                time_ = times_qcels[i_level,i_time] + d # C99*d
                max_time = max(max_time,np.abs(time_))
#                    if (core==1):
#                        print(i_iter,i_level,i_time,i_sample,ks[samples_qcels[i_iter,i_level,i_time,i_sample]])
    max_evolution_time_list[indx_time] = max_time

    n_jobs = n_iter * (n_level+1) * n_time
    i_jobs = 0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            for i_time in range(n_time):
                if (i_jobs%cores==core):
                    time_ = times_qcels[i_level,i_time]
                    time_unscaled_filter = time_ * tau_scale
                    amplitude = compute_amplitude_with_filter(ks,Coeffs,x_center/tau_scale,time_unscaled_filter)
                    amplitudes_qcels[i_iter,i_level,i_time] = amplitude 
                i_jobs += 1
    # bcast amplitudes_qcels
    comm.Barrier()
    i_jobs = 0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            for i_time in range(n_time):
                i_core = i_jobs%cores
                amplitude = comm.bcast(amplitudes_qcels[i_iter,i_level,i_time],root=i_core)
                amplitudes_qcels[i_iter,i_level,i_time] = amplitude
                #if (core==0):
                #    print(i_iter,i_level,i_time,i_sample,amplitudes_sample_qcels[i_iter,i_level,i_time,i_sample])
                comm.Barrier()
                i_jobs  += 1
    # bcast end
#    comm.Barrier()
#    comm.Abort(1)

    error = 0.0


    for i_iter in range(n_iter):

        for i_level in range(n_level+1):
            filtered_amplitudes_sample = np.zeros((n_time),dtype=complex)
            for i_time in range(n_time):
                # compute filtered CDF from amplitudes
                filtered = amplitudes_qcels[i_iter,i_level,i_time]
                filtered_amplitudes_sample[i_time] = filtered
                #
            if (i_level==0):
                x0 = np.array((0.5,0.0,lambda_init))
                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample,x0)
            else:
                x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
                bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample, x0,bounds=bnds)

            #Update initial guess for next iteration
            ground_coefficient_QCELS=res.x[0]
            ground_coefficient_QCELS2=res.x[1]
            ground_energy_estimate_QCELS=res.x[2]
            #Update the estimation interval
            lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau_scale_qcels[i_level])
            lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau_scale_qcels[i_level])
        # sum error
        error += np.abs(ground_energy_estimate_QCELS-eigen_energies_exact[0][-1,0]*tau_scale)
    error /= n_iter
    error_for_time_list[indx_time] = error
    if (core==0):
        #print(time_max, max_time, error)
        #print(d, max_time*tau_scale, error/tau_scale)
        print(time_max, max_time*tau_scale, error/tau_scale)
    indx_time += 1

#indx_time = 0
#for time_max in time_max_list:
#    # estimation of the upperbound
#    lambda_init = eigen_energies_exact[0][0,0]*tau_scale
#
#
#    # center of the eigenvalue filter
#    x_center = lambda_init + relative_gap/2.0
#
#    # compute time list first
#    n_level = round(np.log2(time_max/n_time))
#
#    n_jobs = n_iter * (n_level+1) * n_time * n_sample
#    remainder            = n_jobs%cores
#    n_jobs_for_core = n_jobs//cores
#    if (core<remainder):
#        n_jobs_for_core += 1
#
#
#    amplitudes_sample_qcels = np.zeros((n_jobs_for_core),dtype=complex)
#    tau_scale_qcels         = np.zeros((n_level+1),dtype=float)
#    times_qcels             = np.zeros((n_level+1,n_time),dtype=float)
#    samples_qcels           = np.zeros((n_jobs_for_core),dtype=int)
#    for i_level in range(n_level+1):
#        tau_scale_qcels[i_level] = time_max/n_time/(2**(n_level-i_level))
#        times_qcels[i_level,:] = tau_scale_qcels[i_level] * np.arange(n_time)
#    samples_qcels = np.random.choice(indxs, size=n_jobs_for_core, p=Coeffs_prob)
#
#
#
#    i_jobs = 0
#    i_jobs_core = 0
#    max_time = 0.0
#    for i_iter in range(n_iter):
#        for i_level in range(n_level+1):
#            for i_time in range(n_time):
#                for i_sample in range(n_sample):
#                    if (i_jobs%cores==core):
#                        time_ = times_qcels[i_level,i_time]+ ks[samples_qcels[i_jobs_core]]
#                        time_unscaled_filter = time_ * tau_scale
#                        amplitude = compute_amplitude(time_unscaled_filter)
#                        amplitudes_sample_qcels[i_jobs_core] = compute_amplitude_samples(amplitude, n_shot)
#                        max_time = max(max_time,np.abs(time_))
#                        i_jobs_core += 1
#                    i_jobs += 1
#    max_evolution_time_list[indx_time] = max_time
#
#    error = 0.0
#    i_jobs = 0
#    i_jobs_core = 0
#    for i_iter in range(n_iter):
#        for i_level in range(n_level+1):
#            filtered_amplitudes_sample = np.zeros((n_time),dtype=complex)
#            for i_time in range(n_time):
#                # compute filtered CDF from amplitudes
#                filtered = 0.0
#                for i_sample in range(n_sample):
#                    if (i_jobs%cores==core):
#                        indx = samples_qcels[i_jobs_core]
#                        k = ks[indx]
#                        phase = x_center * k
#                        filtered += Coeffs_ratio[indx] * amplitudes_sample_qcels[i_jobs_core] * np.exp(1j*phase)
#                        i_jobs_core += 1
#                    i_jobs += 1
#                filtered /= n_sample
#                filtered_amplitudes_sample[i_time] += filtered
#                #
#            comm.Allreduce(MPI.IN_PLACE, filtered_amplitudes_sample, op=MPI.SUM)
#            if (i_level==0):
#                x0 = np.array((0.5,0.0,lambda_init))
#                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample,x0)
#            else:
#                x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
#                bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
#                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample, x0,bounds=bnds)
#
#            #Update initial guess for next iteration
#            ground_coefficient_QCELS=res.x[0]
#            ground_coefficient_QCELS2=res.x[1]
#            ground_energy_estimate_QCELS=res.x[2]
#            #Update the estimation interval
#            lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau_scale_qcels[i_level])
#            lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau_scale_qcels[i_level])
#        # sum error
#        error += np.abs(ground_energy_estimate_QCELS-eigen_energies_exact[0][-1,0]*tau_scale)
#    error /= n_iter
#    error_for_time_list[indx_time] = error
#    if (core==0):
#        print(time_max*tau_scale, max_time*tau_scale, error/tau_scale)
#    indx_time += 1
#
#comm.Barrier()
#comm.Allreduce(MPI.IN_PLACE, max_evolution_time_list, op=MPI.MAX)

# %%
if (core==0):
    with open('dE_vs_evolution','w') as file_:
        s = '# n_sample_total= '+str(n_sample*(n_level+1)*(n_time))+', n_shot= '+str(n_shot)+', n_iter= '+str(n_iter)
        s += '\n'
        file_.write(s)
        indx_time = 0
        for time_max in time_max_list:
            s = '{:}'.format(max_evolution_time_list[indx_time]*tau_scale)
            s += '  {:.16e}'.format(error_for_time_list[indx_time]/tau_scale)
            print(s)
            s += '\n'
            file_.write(s)
            indx_time += 1


# %%
# repetition test
time_max = time_max_list[4]
n_level = round(np.log2(time_max/n_time))
n_samples = [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
error_for_n_samples = np.zeros((len(n_samples)),dtype=float)
n_shot = 2048
n_iter = 30


# %%
# QCELS for small overlap
# preprocess
indx_n_sample = 0
for n_sample in n_samples:
    # estimation of the upperbound
    lambda_init = eigen_energies_exact[0][0,0]*tau_scale


    # center of the eigenvalue filter
    x_center = lambda_init + relative_gap/2.0

    # compute time list first

    n_jobs = n_iter * (n_level+1) * n_time * n_sample
    remainder            = n_jobs%cores
    n_jobs_for_core = n_jobs//cores
    if (core<remainder):
        n_jobs_for_core += 1


    amplitudes_sample_qcels = np.zeros((n_jobs_for_core),dtype=complex)
    tau_scale_qcels         = np.zeros((n_level+1),dtype=float)
    times_qcels             = np.zeros((n_level+1,n_time),dtype=float)
    samples_qcels           = np.zeros((n_jobs_for_core),dtype=int)
    for i_level in range(n_level+1):
        tau_scale_qcels[i_level] = time_max/n_time/(2**(n_level-i_level))
        times_qcels[i_level,:] = tau_scale_qcels[i_level] * np.arange(n_time)
    samples_qcels = np.random.choice(indxs, size=n_jobs_for_core, p=Coeffs_prob)

    i_jobs = 0
    i_jobs_core = 0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            for i_time in range(n_time):
                for i_sample in range(n_sample):
                    if (i_jobs%cores==core):
                        time_ = times_qcels[i_level,i_time]+ ks[samples_qcels[i_jobs_core]]
                        time_unscaled_filter = time_ * tau_scale
                        amplitude = compute_amplitude(time_unscaled_filter)
                        amplitudes_sample_qcels[i_jobs_core] = compute_amplitude_samples(amplitude, n_shot)
                        i_jobs_core += 1
                    i_jobs += 1
    error = 0.0
    i_jobs = 0
    i_jobs_core = 0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            filtered_amplitudes_sample = np.zeros((n_time),dtype=complex)
            for i_time in range(n_time):
                # compute filtered CDF from amplitudes
                filtered = 0.0
                for i_sample in range(n_sample):
                    if (i_jobs%cores==core):
                        indx = samples_qcels[i_jobs_core]
                        k = ks[indx]
                        phase = x_center * k
                        filtered += Coeffs_ratio[indx] * amplitudes_sample_qcels[i_jobs_core] * np.exp(1j*phase)
                        i_jobs_core += 1
                    i_jobs += 1
                filtered /= n_sample
                filtered_amplitudes_sample[i_time] += filtered
                #
            comm.Allreduce(MPI.IN_PLACE, filtered_amplitudes_sample, op=MPI.SUM)
            if (i_level==0):
                x0 = np.array((0.5,0.0,lambda_init))
                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample,x0)
            else:
                x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
                bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample, x0,bounds=bnds)

            #Update initial guess for next iteration
            ground_coefficient_QCELS=res.x[0]
            ground_coefficient_QCELS2=res.x[1]
            ground_energy_estimate_QCELS=res.x[2]
            #Update the estimation interval
            lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau_scale_qcels[i_level])
            lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau_scale_qcels[i_level])
        # sum error
        error += np.abs(ground_energy_estimate_QCELS-eigen_energies_exact[0][-1,0]*tau_scale)
    error /= n_iter
    error_for_n_samples[indx_n_sample] = error
    if (core==0):
        print(n_sample*(n_level+1)*(n_time), error/tau_scale)
    indx_n_sample += 1



# %%
if (core==0):
    with open('dE_vs_repetitions_with_exact','w') as file_:
        s = '# time_max = '+str(time_max)+', n_shot= '+str(n_shot)+', n_iter='+str(n_iter)
        s += '\n'
        file_.write(s)
        indx_n_sample = 0
        for n_sample in n_samples:
            s = '{:}'.format(n_sample*(n_level+1)*(n_time))
            s += '  {:.16e}'.format(error_for_n_samples[indx_n_sample]/tau_scale)
            print(s)
            s += '\n'
            file_.write(s)
            indx_n_sample += 1

comm.Barrier()


# %%
# prepare trotterization
del eigen_vectors_exact
gc.collect()
# exact eigenvalues for interaction parts
eigen_energies_coulomb  = []
H_subs_coulomb = []
for isec in range(nsec):
    eigen_energies_coulomb.append(np.zeros((dim_reduced),dtype=float))
    H_subs_coulomb.append([])

for isec in range(nsec):
    eigen_e               = np.zeros((dim_reduced),dtype=float)
    start_time = datetime.now()
    # project hamiltonian on to specified sector
    H_sparse = hamiltonian_coulomb.to_matrix(sparse=True)
    H_sparse.eliminate_zeros()
    jsec = isec
    row      = []
    col      = []
    data     = []
    for ii in range(dim_sub[isec]):
        i = indx_sub[isec][ii]
        for ind in range(H_sparse.indptr[i],H_sparse.indptr[i+1]):
            # j is always in indx_sub[isec], because the Hamiltonian does not mix it
            #print(i,j)
            j = H_sparse.indices[ind]
            jj = iindx_sub[jsec][j]
            row.append(jj)
            col.append(ii)
            #print(ii,jj)
            data.append(H_sparse.data[ind])
    H_sub = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[jsec], dim_sub[isec]))

    H_sub_reduced = basis_transform@(H_sub.toarray())@basis_transform.conj().T

    H_subs_coulomb[isec].append(H_sub_reduced)

    eigen_e = H_sub_reduced.diagonal() # coulomb interaction matrix is diagonal in occupation number basis

    eigen_energies_coulomb[isec][:]   = eigen_e.real
    end_time = datetime.now()
    elapsed = end_time-start_time
    elapsed = elapsed.total_seconds()
    gc.collect()
    if (core==0):
        st = '# Coulomb, elapsed time = {elapsed} secs'.format(elapsed=elapsed)
        memory_usage(st)


# %%
# exact eigenvalues for kinetic parts
eigen_energies_kinetic   = []
eigen_vectors_kinetic = []
H_subs_kinetic = []
for isec in range(nsec):
    eigen_energies_kinetic.append(np.zeros((n_hamiltonians,dim_reduced),dtype=float))
    eigen_vectors_kinetic.append(np.zeros((n_hamiltonians,dim_reduced,dim_reduced),dtype=complex))
    H_subs_kinetic.append([])

for isec in range(nsec):
    eigen_e               = np.zeros((dim_reduced),dtype=float)
    eigen_v               = np.zeros((dim_reduced,dim_reduced),dtype=complex)
    for alpha in range(n_hamiltonians):
        start_time = datetime.now()
        # project hamiltonian on to specified sector
        H_sparse = hamiltonians_kinetic[alpha].to_matrix(sparse=True)
        H_sparse.eliminate_zeros()
        jsec = isec
        row      = []
        col      = []
        data     = []
        for ii in range(dim_sub[isec]):
            i = indx_sub[isec][ii]
            for ind in range(H_sparse.indptr[i],H_sparse.indptr[i+1]):
                # j is always in indx_sub[isec], because the Hamiltonian does not mix it
                #print(i,j)
                j = H_sparse.indices[ind]
                jj = iindx_sub[jsec][j]
                row.append(jj)
                col.append(ii)
                #print(ii,jj)
                data.append(H_sparse.data[ind])
        H_sub = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[jsec], dim_sub[isec]))

        H_sub_reduced = basis_transform@(H_sub.toarray())@basis_transform.conj().T

        H_subs_kinetic[isec].append(H_sub_reduced)

        # diagonalize sectorized hamiltonian
        i_core = alpha%cores
        if (core==i_core):
            eigen_e, eigen_v = np.linalg.eigh(H_sub_reduced)
            eigen_energies_kinetic[isec][alpha,:]   = eigen_e
            eigen_vectors_kinetic[isec][alpha,:,:] = eigen_v
    del eigen_e, eigen_v
    gc.collect()
comm.Barrier()
bcast_start_time = datetime.now()
for isec in range(nsec):
    for alpha in range(n_hamiltonians):
        i_core = alpha%cores
        comm.Bcast([eigen_energies_kinetic[isec][alpha,:] ,MPI.DOUBLE],root=i_core)
        comm.Bcast([eigen_vectors_kinetic[isec][alpha,:,:],MPI.DOUBLE_COMPLEX],root=i_core)
end_time = datetime.now()
elapsed = end_time-start_time
elapsed = elapsed.total_seconds()
if (core==0):
    st = '# {percent}%, elapsed time = {elapsed} secs'.format(percent=(100),elapsed=elapsed)
    memory_usage(st)
    st = 'Bcast time = {elapsed} secs'.format(elapsed=(end_time-bcast_start_time).total_seconds())
    print(st)


def Apply_ExactEvolution_coulomb(isec, eps, time, v):
    vec = np.exp(-1j*time*(eigen_energies_coulomb[isec][:]-eps))
    w = vec * v
    return w


# %%

def Apply_ExactEvolution_kinetic(isec, alpha, eps, time, v):
    w = v.conj()@eigen_vectors_kinetic[isec][alpha,:,:]
    vec = np.exp(-1j*time*(eigen_energies_kinetic[isec][alpha,:]-eps))
    w = vec*w.conj()
    w = eigen_vectors_kinetic[isec][alpha,:,:]@w
    return w


def Apply_TrotterEvolution(isec, alpha, time, eps, n_trotter, indx, v):
    dt = time/n_trotter
    if (indx==0): 
        # first order trotter
        w = v
        for i_trotter in range(n_trotter):
            w = Apply_ExactEvolution_coulomb (isec, 0.0, dt, w)
            w = Apply_ExactEvolution_kinetic (isec, alpha, 0.0, dt, w)
    elif (indx==1):
        w = v
        for i_trotter in range(n_trotter):
            w = Apply_ExactEvolution_kinetic (isec, alpha, 0.0, dt, w)
            w = Apply_ExactEvolution_coulomb (isec, 0.0, dt, w)
    return w*np.exp(1j*eps*time)

def Apply_TrotterEvolution2(isec, alpha, time, eps, n_trotter, indx, v):
    dt = time/n_trotter
    vec = np.exp(-1j*dt*(eigen_energies_kinetic[isec][alpha,:]-eps))
    M = eigen_vectors_kinetic[isec][alpha,:,:]@np.diag(vec)@eigen_vectors_kinetic[isec][alpha,:,:].conj().T
    if (indx==0): 
        # first order trotter
        w = v
        for i_trotter in range(n_trotter):
            w = Apply_ExactEvolution_coulomb (isec, 0.0, dt, w)
            w = M@w
            #w = Apply_ExactEvolution_kinetic (isec, alpha, 0.0, dt, w)
    elif (indx==1):
        w = v
        for i_trotter in range(n_trotter):
            w = M@w
            #w = Apply_ExactEvolution_kinetic (isec, alpha, 0.0, dt, w)
            w = Apply_ExactEvolution_coulomb (isec, 0.0, dt, w)
    return w*np.exp(1j*eps*time)

# for trotter
# fix time_max
time_max = time_max_list[4]
n_level = round(np.log2(time_max/n_time))
n_sample = 256
max_n_trotter_list = [2, 4, 6, 8]
max_n_trotter_list+= [10, 20, 30, 40, 50, 75, 100]
max_n_trotter_list+= [200, 300, 400, 500, 1000, 2000, 3000, 4000]
#max_n_trotter_list+= [2000,3000,4000,5000]
error_for_trotter_list = np.zeros((len(max_n_trotter_list)),dtype=float)
max_n_trotter_run_list = np.zeros((len(max_n_trotter_list)),dtype=float)
n_shot = 2048
n_iter = 30
trotter_permutation_indx = 1

gc.collect()


# %%
# trotterized case
indx_n_trotter = 0
for n_trotter in max_n_trotter_list:
    start_time = datetime.now()
    # estimation of the upperbound
    lambda_init = eigen_energies_exact[0][0,0]*tau_scale


    # center of the eigenvalue filter
    x_center = lambda_init + relative_gap/2.0

    # compute time list first

    n_jobs = n_iter * (n_level+1) * n_time * n_sample
    remainder            = n_jobs%cores
    n_jobs_for_core = n_jobs//cores
    if (core<remainder):
        n_jobs_for_core += 1


    amplitudes_sample_qcels = np.zeros((n_jobs_for_core),dtype=complex)
    tau_scale_qcels         = np.zeros((n_level+1),dtype=float)
    times_qcels             = np.zeros((n_level+1,n_time),dtype=float)
    samples_qcels           = np.zeros((n_jobs_for_core),dtype=int)
    for i_level in range(n_level+1):
        tau_scale_qcels[i_level] = time_max/n_time/(2**(n_level-i_level))
        times_qcels[i_level,:] = tau_scale_qcels[i_level] * np.arange(n_time)
    samples_qcels = np.random.choice(indxs, size=n_jobs_for_core, p=Coeffs_prob)

    i_jobs = 0
    i_jobs_core = 0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            for i_time in range(n_time):
                for i_sample in range(n_sample):
                    if (i_jobs%cores==core):
                        time_ = times_qcels[i_level,i_time]+ ks[samples_qcels[i_jobs_core]]
                        time_unscaled_filter = time_ * tau_scale
                        amplitude = compute_trotter_amplitude(time_unscaled_filter,n_trotter,trotter_permutation_indx)
                        amplitudes_sample_qcels[i_jobs_core] = compute_amplitude_samples(amplitude, n_shot)
                        i_jobs_core += 1
                    i_jobs += 1

    error = 0.0
    i_jobs = 0
    i_jobs_core = 0
    for i_iter in range(n_iter):
        for i_level in range(n_level+1):
            filtered_amplitudes_sample = np.zeros((n_time),dtype=complex)
            for i_time in range(n_time):
                # compute filtered CDF from amplitudes
                filtered = 0.0
                for i_sample in range(n_sample):
                    if (i_jobs%cores==core):
                        indx = samples_qcels[i_jobs_core]
                        k = ks[indx]
                        phase = x_center * k
                        filtered += Coeffs_ratio[indx] * amplitudes_sample_qcels[i_jobs_core] * np.exp(1j*phase)
                        i_jobs_core += 1
                    i_jobs += 1
                filtered /= n_sample
                filtered_amplitudes_sample[i_time] += filtered
                #
            comm.Allreduce(MPI.IN_PLACE, filtered_amplitudes_sample, op=MPI.SUM)
            if (i_level==0):
                x0 = np.array((0.5,0.0,lambda_init))
                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample,x0)
            else:
                x0=np.array((ground_coefficient_QCELS,ground_coefficient_QCELS2,ground_energy_estimate_QCELS))
                bnds=((-np.inf,np.inf),(-np.inf,np.inf),(lambda_min,lambda_max)) 
                res = qcels_opt(times_qcels[i_level,:],filtered_amplitudes_sample, x0,bounds=bnds)

            #Update initial guess for next iteration
            ground_coefficient_QCELS=res.x[0]
            ground_coefficient_QCELS2=res.x[1]
            ground_energy_estimate_QCELS=res.x[2]
            #Update the estimation interval
            lambda_min=ground_energy_estimate_QCELS-np.pi/(2*tau_scale_qcels[i_level])
            lambda_max=ground_energy_estimate_QCELS+np.pi/(2*tau_scale_qcels[i_level])
        # sum error
        error += np.abs(ground_energy_estimate_QCELS-eigen_energies_exact[0][-1,0]*tau_scale)
    error /= n_iter
    error_for_trotter_list[indx_n_trotter] = error
    max_n_trotter_run_list[indx_n_trotter] = n_trotter
    if (core==0):
        print(n_trotter, error/tau_scale)
    indx_n_trotter += 1
    end_time = datetime.now()
    elapsed = end_time-start_time
    elapsed = elapsed.total_seconds()
    gc.collect()
    comm.Barrier()
    if (core==0):
        st = '# elapsed time = {elapsed} secs'.format(elapsed=elapsed)
        memory_usage(st)



# %%
if (core==0):
    with open('dE_vs_n_trotter','w') as file_:
        s = '# time_max= '+str(time_max)+', n_shot= '+str(n_shot)+', n_iter='+str(n_iter)+', n_sample=' + str(n_sample*(n_level+1)*n_time)
        s += '\n'
        file_.write(s)
        indx_n_trotter = 0
        for max_n_trotter in max_n_trotter_list:
            s = '{:}'.format(max_n_trotter_run_list[indx_n_trotter])
            s += '  {:.16e}'.format(error_for_trotter_list[indx_n_trotter]/tau_scale)
            print(s)
            s += '\n'
            file_.write(s)
            indx_n_trotter += 1
#
#
#
#
#
#
#
#
