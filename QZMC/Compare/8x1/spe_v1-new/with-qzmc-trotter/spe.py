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
# solution finding method
# # 1: v1; find interval
sol_method = 1
# # 2: v2; find optimal point
# sol_method = 2

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
it_inter            = 0
bc = BoundaryCondition.OPEN

# make interaction part first (fixed)

if (n_y==1):
    empty  = LineLattice(num_nodes=n_x,boundary_condition=bc, edge_parameter=0.0,\
                            onsite_parameter=0.0)
else:
    empty = SquareLattice(rows=n_x,cols=n_y, boundary_condition=bc, edge_parameter=(0,0),\
                        onsite_parameter=0.0)
interaction_part = FermiHubbardModel(lattice=empty, onsite_interaction=Uc).second_q_op().simplify()
for t_inter in t_inters:
    t_inter = t_inters[it_inter]
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
def ApproximateStepFunction(Coeffs, d, x):
    step = 0.5
    for j in range(1,d+1):
        step += Coeffs[j] * np.exp(1j*(j)*x)
        step += -Coeffs[j] * np.exp(-1j*(j)*x) # negative freq contribution
    return step


# %%
def ApproximateStepFunction_importance(Coeffs_sum, sample, x):
    step = 0.0
    Ns = len(sample)
    for i in range(Ns):
        step += np.sin(sample[i]*x)
    step *= 2.0 * Coeffs_sum/Ns
    step += 0.5
    return step


# %%
def CDF_from_evolution(amplitudes, Coeffs, x):
    cdf = 0.0
    d   = len(Coeffs) - 1
    for j in range(1,d+1):
        cdf += np.abs(Coeffs[j]) * (amplitudes[j].real * np.sin(j*x)+ amplitudes[j].imag * np.cos(j*x))
    cdf *= 2.0 
    cdf += 0.5
    return cdf

def CDF_deriv_from_evolution(amplitudes, Coeffs, x):
    cdf = 0.0
    d   = len(Coeffs) - 1
    for j in range(1,d+1):
        cdf += np.abs(Coeffs[j]) * (j*amplitudes[j].real * np.cos(j*x)- j*amplitudes[j].imag * np.sin(j*x))
    cdf *= 2.0 
    return cdf

def CDF_from_evolution_importance(amplitudes, Coeffs_sum, sample, x):
    cdf = 0.0
    n_sample = len(sample)
    for i_sample in range(n_sample):
        j = sample[i_sample]
        cdf += (amplitudes[j].real * np.sin(j*x)+ amplitudes[j].imag * np.cos(j*x))
    cdf *= 2.0  * Coeffs_sum/n_sample
    cdf += 0.5
    return cdf

def CDF_from_evolution_importance2(amplitudes, Coeffs_sum, sample, x):
    cdf = 0.0
    n_sample = len(sample)
    for i_sample in range(n_sample):
        j = sample[i_sample]
        cdf += (amplitudes[i_sample].real * np.sin(j*x)+ amplitudes[i_sample].imag * np.cos(j*x))
    cdf *= 2.0  * Coeffs_sum/n_sample
    cdf += 0.5
    return cdf

def CDF_deriv_from_evolution_importance2(amplitudes, Coeffs_sum, sample, x):
    cdf = 0.0
    n_sample = len(sample)
    for i_sample in range(n_sample):
        j = sample[i_sample]
        cdf += (j*amplitudes[i_sample].real * np.cos(j*x) - j* amplitudes[i_sample].imag * np.sin(j*x))
    cdf *= 2.0  * Coeffs_sum/n_sample
    return cdf




# %%
# LL implementation imported from QCLES code
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

#isec = 0
##phi_start = eigen_vectors_exact[isec][0,:,0] + 
##phi_start = eigen_vectors_exact[isec][-1,:,0]*np.sqrt(0.4) + eigen_vectors_exact[isec][-1,:,1]*np.sqrt(0.6)
#phi_start = eigen_vectors_kinetic[isec][-1,:,0]
#eta = np.abs(phi_start@eigen_vectors_exact[isec][-1,:,0])**2
#print(eta)

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
#print(M_fourier_coeffs(3,0.1),M_fourier_coeffs(4,0.1))

# %%
#delta_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
#eps_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
#delta_list = [x * tau_scale for x in eps_list ]
#d_list = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]
#d_list = [i for i in range(1,801)]
#d_list = [24*i for i in range(1,201)]
d_list = [21, 41, 81, 201, 401, 801, 1601, 4001, 6001, 12001, 24001, 48001]
#d_list = [501, 1001, 2001, 5001, 10001, 20001, 40001, 80001, 120001]
#d_list += [4800*i for i in range(2,6)]
#d_list.append([2400*i+1 for i in range(2,)])
delta_list = [4/d for d in d_list]
eps_list = [ x/tau_scale for x in delta_list]
error_for_delta_list = np.zeros((len(delta_list)),dtype=float)
max_evolution_time_list = np.zeros((len(delta_list)),dtype=float)
#if (core==0):
#    print(delta_list)
#delta_list = [0.001]
#n_sample = 1800
#n_shot   = 1
n_sample = 3000
n_shot   = 2048
n_iter   = 30



# %%
# compute C99
#d = 150
#delta = 4/d
#C = F_fourier_coeffs(d,delta)
#Coeffs = C[:d+1] # only positive coefficients are needed, because negative ones are just - of that, and 0th value is 0.5
#
## prepare importance sampling
#Coeffs_sum = 0.0
#for j in range(1,d+1):
#    Coeffs_sum += np.abs(Coeffs[j])
##print(Coeffs_sum)
#Coeffs_prob = np.abs(Coeffs)/Coeffs_sum
#Coeffs_prob[0] = 0.0
## find C99
#sum_tmp = 0.0
#for i in range(1,d+1):
#    sum_tmp += Coeffs_prob[i]
#    if (sum_tmp>0.99):
#        C99 = i/d
#        break

# %%
from scipy.optimize import root_scalar, minimize_scalar
def find_all_roots(func, x_start, x_end, num_points=100, method='bisect'):
    x_values = np.linspace(x_start, x_end, num_points)
    f_values = func(x_values)

    roots = []
    for i in range(len(f_values) - 1):
        if f_values[i] * f_values[i + 1] < 0:
            bracket = (x_values[i], x_values[i + 1])
            result = root_scalar(func, method=method, bracket=bracket)
            if result.converged:
                roots.append(result.root)
    
    return sorted(roots)

# %%
#d = 397
#delta = 4/d
#t_max = tau_scale*d
#
## compute step function expansion coeffs
#C = F_fourier_coeffs(d,delta)
#Coeffs = C[:d+1] # only positive coefficients are needed, because negative ones are just - of that, and 0th value is 0.5
#print(np.abs(Coeffs[1::2])/np.abs(Coeffs[1]))

# %%
# without sampling error
indx_delta = 0 
#for delta in delta_list:
#    d = round(4/delta)
for d in d_list:
    delta = 4/d
    t_max = tau_scale*d

    # compute step function expansion coeffs
    C = F_fourier_coeffs(d,delta)
    Coeffs = C[:d+1] # only positive coefficients are needed, because negative ones are just - of that, and 0th value is 0.5

    # prepare importance sampling
    Coeffs_sum = 0.0
    for j in range(1,d+1):
        Coeffs_sum += np.abs(Coeffs[j])
    #print(Coeffs_sum)
    Coeffs_prob = np.abs(Coeffs)/Coeffs_sum
    Coeffs_prob[0] = 0.0

    # find C99
    sum_tmp = 0.0
    for i in range(1,d+1):
        sum_tmp += Coeffs_prob[i]
        if (sum_tmp>0.99):
            C99 = i/d
            break


    #print(Coeffs_prob[1::2])
    #print(np.sum(Coeffs_prob))
    ks = [j for j in range(d+1)]
    #print(ks)

    # compute amplitudes first
    amplitudes = np.zeros((d+1),dtype=complex)
    # only odd parts contributes
    n_alive = 0
    k_alive = []
    for k in range(1,d+1):
        if (Coeffs_prob[k]<1e-10):
            continue
        k_alive.append(k)
        n_alive += 1
    #if (core==0):
    #    print(n_alive)
    for ii in range(core,n_alive,cores):
        k = k_alive[ii]
        isec = 0
        alpha = n_hamiltonians-1
        time = tau_scale * k
        phi = copy.copy(phi_start)
        phi = Apply_ExactEvolution(isec,alpha,0.0,time,phi)
        amplitudes[k] = phi_start.conj()@phi
    # collect and bcast
    for ii in range(n_alive):
        i_core = ii%cores
        k = k_alive[ii]
        amplitude = comm.bcast(amplitudes[k],root=i_core)
        amplitudes[k] = amplitude
        comm.Barrier()


    error_avg = 0.0
    max_evolution_time = 0.0
    # find ground state energy estimation through binary search

    # cost estimation
    #total_evolution_time = np.sum(samples) * tau_scale
    max_evolution_time = max(d * tau_scale,max_evolution_time)
    #print(max_evolution_time,t_max)

    # energy estimation
    if (sol_method==1):
        x_min = -np.pi/3.0
        x_max = np.pi/3.0

        cdf_1 = lambda x: CDF_from_evolution(amplitudes,Coeffs,x) - (eta/4.0)

        result = find_all_roots(cdf_1, x_min, x_max) 
        if (len(result)<1):
            x_min_opt = x_min # no root
        else:
            x_min_opt = result[0]

        eigen_energy_estimate = x_min_opt
    elif (sol_method==2):
        # search step and find maximum in the region
        x_min = -np.pi/3.0
        x_max = np.pi/3.0

        cdf_1 = lambda x: CDF_from_evolution(amplitudes,Coeffs,x) - (eta/4.0)
        cdf_2 = lambda x: CDF_from_evolution(amplitudes,Coeffs,x) - (3*eta/4.0)
        negative_cdf_deriv = lambda x : -CDF_deriv_from_evolution(amplitudes,Coeffs,x)

        #result = root_scalar(cdf,method='bisect',bracket=[x_min,x_max])
        result = find_all_roots(cdf_1, x_min, x_max) 
        if (len(result)<1):
            x_min_opt = x_min # no root
        else:
            x_min_opt = result[0]
        result = find_all_roots(cdf_2, x_min_opt, x_max) 
        if (len(result)<1):
            x_max_opt = x_max # no root
        else:
            x_max_opt = result[0]

        result = minimize_scalar(negative_cdf_deriv, bounds=(x_min_opt,x_max_opt), method='bounded')

        eigen_energy_estimate = result.x
    #count = 0
    #while width > delta:
    #    xs = np.linspace(-width/2.0,width/2.0,num=Nx) + eigen_energy_estimate
    #    #print(xs)
    #    cdf = CDF_from_evolution(amplitudes,Coeffs,xs)
    #    #print(cdf)
    #    indicator_list = cdf>=(eta/2.00)
    #    #print(indicator_list)
    #    ix = np.nonzero(indicator_list)[0][0]
    #    eigen_energy_estimate_old = copy.copy(eigen_energy_estimate)
    #    eigen_energy_estimate = xs[ix]
    #    width /= 2
    #    count += 1
    #    #print(width,count,eigen_energy_estimate/tau_scale-eigen_energies_exact[0][-1,0])

    #    #if (np.abs(eigen_energy_estimate-eigen_energy_estimate_old)<delta):
    #    #    break
    error = np.abs(eigen_energy_estimate/tau_scale-eigen_energies_exact[0][-1,0])
    error_avg = error
    if (core==0):
        #print(delta/tau_scale,max_evolution_time,error_avg)
        print(d_list[indx_delta],C99*d,error_avg)
    error_for_delta_list[indx_delta] = error_avg
    max_evolution_time_list[indx_delta] = C99*d*tau_scale
    indx_delta += 1



# %%
# save to file
if (core==0):
    with open('dE_vs_evolution','w') as file_:
        s = '# n_sample= '+str(n_sample)+', n_shot= '+str(n_shot)+', n_iter= '+str(n_iter)+', n_sample= '+str(n_sample)
        s += '\n'
        file_.write(s)
        indx_delta = 0
        for delta in delta_list:
            d = round(4/delta)
            t_max = tau_scale*d
            s = '{:}'.format(max_evolution_time_list[indx_delta])
            s += '  {:.16e}'.format(error_for_delta_list[indx_delta])
            print(s)
            s += '\n'
            file_.write(s)
            indx_delta += 1


# %%
# repetitions, with exact time evolution
d = 12001
delta = 4/d
n_samples = [10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000]
error_for_n_samples = np.zeros((len(n_samples)),dtype=float)
n_shot = 2048
n_iter = 30
# n_samples~d is enough to get a converged values?


# %%
# parameters are from LinLin2022

indx_n_sample = 0

t_max = tau_scale*d

# compute step function expansion coeffs
C = F_fourier_coeffs(d,delta)
Coeffs = C[:d+1] # only positive coefficients are needed, because negative ones are just - of that, and 0th value is 0.5

# prepare importance sampling
Coeffs_sum = 0.0
for j in range(1,d+1):
    Coeffs_sum += np.abs(Coeffs[j])
#print(Coeffs_sum)
Coeffs_prob = np.abs(Coeffs)/Coeffs_sum
Coeffs_prob[0] = 0.0
#print(Coeffs_prob[1::2])
#print(np.sum(Coeffs_prob))
ks = [j for j in range(d+1)]

# compute amplitudes first
amplitudes = np.zeros((d+1),dtype=complex)
# only odd parts contributes
n_alive = 0
k_alive = []
for k in range(1,d+1):
    if (Coeffs_prob[k]<1e-10):
        continue
    k_alive.append(k)
    n_alive += 1
#if (core==0):
#    print(n_alive)
for ii in range(core,n_alive,cores):
    k = k_alive[ii]
    isec = 0
    alpha = n_hamiltonians-1
    time = tau_scale * k
    phi = copy.copy(phi_start)
    phi = Apply_ExactEvolution(isec,alpha,0.0,time,phi)
    amplitudes[k] = phi_start.conj()@phi
# collect and bcast
for ii in range(n_alive):
    i_core = ii%cores
    k = k_alive[ii]
    amplitude = comm.bcast(amplitudes[k],root=i_core)
    amplitudes[k] = amplitude
    comm.Barrier()


for n_sample in n_samples:

    error_avg = 0.0
    for i_iter in range(n_iter):
        # find ground state energy estimation through binary search

        samples = np.random.choice(ks, size=n_sample, p=Coeffs_prob)
    
        amplitudes_sample = np.zeros((n_sample),dtype=complex)
        for i_sample in range(n_sample):
            k = samples[i_sample]
            # real part
            computed_value = amplitudes[k].real

            # # shot errors
            p_up = (computed_value + 1.0)/2.0
            sample_shot = np.random.binomial(n_shot,p_up)
            shot_error = 2*(sample_shot/(n_shot) - p_up)
            computed_value += shot_error

            amp_sample_real = computed_value
            

            # imag part
            computed_value = amplitudes[k].imag

            # # shot errors
            p_up = (computed_value + 1.0)/2.0
            sample_shot = np.random.binomial(n_shot,p_up)
            shot_error = 2*(sample_shot/(n_shot) - p_up)
            computed_value += shot_error

            amp_sample_imag = computed_value

            amplitudes_sample[i_sample] = amp_sample_real + 1j * amp_sample_imag
    
        # energy estimation
        if (sol_method==1):
            x_min = -np.pi/3.0
            x_max = np.pi/3.0

            cdf_1 = lambda x: CDF_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x) - (eta/4.0)

            result = find_all_roots(cdf_1, x_min, x_max) 
            if (len(result)<1):
                x_min_opt = x_min # no root
            else:
                x_min_opt = result[0]

            eigen_energy_estimate = x_min_opt
        elif (sol_method==2):
            # search step and find maximum in the region
            x_min = -np.pi/3.0
            x_max = np.pi/3.0

            cdf_1 = lambda x: CDF_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x) - (eta/4.0)
            cdf_2 = lambda x: CDF_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x) - (3*eta/4.0)
            negative_cdf_deriv = lambda x : -CDF_deriv_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x)

            #result = root_scalar(cdf,method='bisect',bracket=[x_min,x_max])
            result = find_all_roots(cdf_1, x_min, x_max) 
            if (len(result)<1):
                x_min_opt = x_min # no root
            else:
                x_min_opt = result[0]
            result = find_all_roots(cdf_2, x_min_opt, x_max) 
            if (len(result)<1):
                x_max_opt = x_max # no root
            else:
                x_max_opt = result[0]

            result = minimize_scalar(negative_cdf_deriv, bounds=(x_min_opt,x_max_opt), method='bounded')

            eigen_energy_estimate = result.x
        error = np.abs(eigen_energy_estimate/tau_scale-eigen_energies_exact[0][-1,0])

        error_avg += error
    error_avg /= n_iter
    if (core==0):
        print(n_sample,error_avg)
    error_for_n_samples[indx_n_sample] = error_avg
    indx_n_sample += 1



# %%
# save to file
if (core==0):
    with open('dE_vs_repetitions_with_exact','w') as file_:
        s = '# eps= '+str(delta/tau_scale)+', n_shot= '+str(n_shot)+', n_iter='+str(n_iter)
        s += '\n'
        file_.write(s)
        indx_n_sample = 0
        for n_sample in n_samples:
            s = '{:}'.format(n_sample)
            s += '  {:.16e}'.format(error_for_n_samples[indx_n_sample])
            print(s)
            s += '\n'
            file_.write(s)
            indx_n_sample += 1
# #
# #
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

## fix eps
d = 12001
delta = 4/d
eps = delta/tau_scale
n_sample = 3000 # after the convergence
max_n_trotter_list = [388, 624, 782, 1172, 1568, 1960, 2352]
#max_n_trotter_list = [2, 4, 6, 8]
#max_n_trotter_list+= [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#max_n_trotter_list+= [200,300,400,500,600,700,800,900,1000]
#max_n_trotter_list+= [2000,3000,4000,5000]
#max_n_trotter_list= np.arange(2000,5000,100)
error_for_trotter_list = np.zeros((len(max_n_trotter_list)),dtype=float)
max_n_trotter_run_list = np.zeros((len(max_n_trotter_list)),dtype=float)
n_shot = 2048
n_iter = 30

#d = round(4/delta)
#t_max = tau_scale*d
#
## compute step function expansion coeffs
#C = F_fourier_coeffs(d,delta)
#Coeffs = C[:d+1] # only positive coefficients are needed, because negative ones are just - of that, and 0th value is 0.5

# prepare importance sampling
#Coeffs_sum = 0.0
#for j in range(1,d+1):
#    Coeffs_sum += np.abs(Coeffs[j])
##print(Coeffs_sum)
#Coeffs_prob = np.abs(Coeffs)/Coeffs_sum
#Coeffs_prob[0] = 0.0
#ks = [j for j in range(d+1)]
#if (core==0):
#    for j in range(d+1):
#        print(j, Coeffs_prob[j])


# without sampling error
# indx_trotter = 0
# for max_n_trotter in max_n_trotter_list:
#     max_n_trotter_run = max_n_trotter
#     # compute amplitudes first
#     amplitudes = np.zeros((d+1),dtype=complex)
#     n_trotters = np.zeros((d+1),dtype=int)
#     # only odd parts contributes
#     n_alive = 0
#     k_alive = []
#     for k in range(1,d+1):
#         if (Coeffs_prob[k]<1e-10):
#             continue
#         k_alive.append(k)
#         n_alive += 1
#     #if (core==0):
#     #    print(n_alive)
#     for ii in range(core,n_alive,cores):
#         k = k_alive[ii]
#         isec = 0
#         alpha = n_hamiltonians-1
#         time = tau_scale * k
#         n_trotter = max_n_trotter
#         n_trotters[k] = n_trotter
#         phi = copy.copy(phi_start)
#         phi = Apply_TrotterEvolution(isec, alpha, time, 0.0, n_trotter, indx, phi)
#         amplitudes[k] = phi_start.conj()@phi
#     # collect and bcast
#     for ii in range(n_alive):
#         i_core = ii%cores
#         k = k_alive[ii]
#         amplitude = comm.bcast(amplitudes[k],root=i_core)
#         amplitudes[k] = amplitude
#         comm.Barrier()
# 
# 
#     error_avg = 0.0
#     max_evolution_time = 0.0
#     # find ground state energy estimation through binary search
# 
#     # cost estimation
#     #total_evolution_time = np.sum(samples) * tau_scale
#     max_evolution_time = max(d * tau_scale,max_evolution_time)
#     #print(max_evolution_time,t_max)
# 
#     # energy estimation
#     if (sol_method==1):
#         x_min = -np.pi/3.0
#         x_max = np.pi/3.0
# 
#         cdf_1 = lambda x: CDF_from_evolution(amplitudes,Coeffs,x) - (eta/4.0)
# 
#         result = find_all_roots(cdf_1, x_min, x_max) 
#         if (len(result)<1):
#             x_min_opt = x_min # no root
#         else:
#             x_min_opt = result[0]
# 
#         eigen_energy_estimate = x_min_opt
#     elif (sol_method==2):
#         # search step and find maximum in the region
#         x_min = -np.pi/3.0
#         x_max = np.pi/3.0
# 
#         cdf_1 = lambda x: CDF_from_evolution(amplitudes,Coeffs,x) - (eta/4.0)
#         cdf_2 = lambda x: CDF_from_evolution(amplitudes,Coeffs,x) - (3*eta/4.0)
#         negative_cdf_deriv = lambda x : -CDF_deriv_from_evolution(amplitudes,Coeffs,x)
# 
#         #result = root_scalar(cdf,method='bisect',bracket=[x_min,x_max])
#         result = find_all_roots(cdf_1, x_min, x_max) 
#         if (len(result)<1):
#             x_min_opt = x_min # no root
#         else:
#             x_min_opt = result[0]
#         result = find_all_roots(cdf_2, x_min_opt, x_max) 
#         if (len(result)<1):
#             x_max_opt = x_max # no root
#         else:
#             x_max_opt = result[0]
# 
#         result = minimize_scalar(negative_cdf_deriv, bounds=(x_min_opt,x_max_opt), method='bounded')
# 
#         eigen_energy_estimate = result.x
#     #count = 0
#     #while width > delta:
#     #    xs = np.linspace(-width/2.0,width/2.0,num=Nx) + eigen_energy_estimate
#     #    #print(xs)
#     #    cdf = CDF_from_evolution(amplitudes,Coeffs,xs)
#     #    #print(cdf)
#     #    indicator_list = cdf>=(eta/2.00)
#     #    #print(indicator_list)
#     #    ix = np.nonzero(indicator_list)[0][0]
#     #    eigen_energy_estimate_old = copy.copy(eigen_energy_estimate)
#     #    eigen_energy_estimate = xs[ix]
#     #    width /= 2
#     #    count += 1
#     #    #print(width,count,eigen_energy_estimate/tau_scale-eigen_energies_exact[0][-1,0])
# 
#     #    #if (np.abs(eigen_energy_estimate-eigen_energy_estimate_old)<delta):
#     #    #    break
#     error = np.abs(eigen_energy_estimate/tau_scale-eigen_energies_exact[0][-1,0])
#     error_avg = error
#     if (core==0):
#         print(max_n_trotter,max_n_trotter_run,error_avg)
#     max_n_trotter_run_list[indx_trotter] = max_n_trotter_run
#     error_for_trotter_list[indx_trotter] = error_avg
#     indx_trotter +=1
#     end_time = datetime.now()
#     elapsed = end_time-start_time
#     elapsed = elapsed.total_seconds()
#     if (core==0):
#         st = '# elapsed time = {elapsed} secs'.format(elapsed=elapsed)
#         memory_usage(st)
# 
# 
# 
# # %%
# # save to file
# if (core==0):
#     with open('dE_vs_n_trotter','w') as file_:
#         s = '# eps= '+str(eps)+', n_shot= '+str(n_shot)+', n_iter='+str(n_iter)+', n_sample=' + str(n_sample)
#         s += '\n'
#         file_.write(s)
#         indx_n_trotter = 0
#         for max_n_trotter in max_n_trotter_list:
#             s = '{:}'.format(max_n_trotter_run_list[indx_n_trotter])
#             s += '  {:.16e}'.format(error_for_trotter_list[indx_n_trotter])
#             print(s)
#             s += '\n'
#             file_.write(s)
#             indx_n_trotter += 1

# parameters are from LinLin2022
# trotter version
# parallelized
d = round(4/delta)
t_max = tau_scale*d

# compute step function expansion coeffs
C = F_fourier_coeffs(d,delta)
Coeffs = C[:d+1] # only positive coefficients are needed, because negative ones are just - of that, and 0th value is 0.5

# prepare importance sampling
Coeffs_sum = 0.0
for j in range(1,d+1):
    Coeffs_sum += np.abs(Coeffs[j])
#print(Coeffs_sum)
Coeffs_prob = np.abs(Coeffs)/Coeffs_sum
Coeffs_prob[0] = 0.0
ks = [j for j in range(d+1)]
#if (core==0):
#    for j in range(d+1):
#        print(j, Coeffs_prob[j])

# compute trotter amplitudes


samples_list = []
indx_trotter = 0
for max_n_trotter in max_n_trotter_list:
    start_time = datetime.now()
    # compute amplitudes first
    amplitudes = np.zeros((d+1),dtype=complex)
    n_trotters = np.zeros((d+1),dtype=int)
    # only odd parts contributes
    n_alive = 0
    k_alive = []
    for k in range(1,d+1):
        if (Coeffs_prob[k]<1e-10):
            continue
        k_alive.append(k)
        n_alive += 1
    for ii in range(core,n_alive,cores):
        k = k_alive[ii]
        isec = 0
        alpha = n_hamiltonians-1
        time = tau_scale * k
        n_trotter = max_n_trotter
        n_trotters[k] = n_trotter
        indx = 1
        phi = copy.copy(phi_start)
        phi = Apply_TrotterEvolution(isec, alpha, time, 0.0, n_trotter, indx, phi)
        amplitudes[k] = phi_start.conj()@phi
    # collect and bcast
    for ii in range(n_alive):
        i_core = ii%cores
        k = k_alive[ii]
        n_trotter = comm.bcast(n_trotters[k],root=i_core)
        n_trotters[k] = n_trotter
        amplitude = comm.bcast(amplitudes[k],root=i_core)
        amplitudes[k] = amplitude
        comm.Barrier()
        
    error_avg = 0.0
    max_n_trotter_run = 0
    for i_iter in range(n_iter):
        # find ground state energy estimation through binary search

        samples = np.random.choice(ks, size=n_sample, p=Coeffs_prob)
    
        amplitudes_sample = np.zeros((n_sample),dtype=complex)
        for i_sample in range(n_sample):
            k = samples[i_sample]
            max_n_trotter_run = max(max_n_trotter_run,n_trotters[k])
            # real part
            computed_value = amplitudes[k].real

            # # shot errors
            p_up = (computed_value + 1.0)/2.0
            sample_shot = np.random.binomial(n_shot,p_up)
            shot_error = 2*(sample_shot/(n_shot) - p_up)
            computed_value += shot_error

            amp_sample_real = computed_value
            

            # imag part
            computed_value = amplitudes[k].imag

            # # shot errors
            p_up = (computed_value + 1.0)/2.0
            sample_shot = np.random.binomial(n_shot,p_up)
            shot_error = 2*(sample_shot/(n_shot) - p_up)
            computed_value += shot_error

            amp_sample_imag = computed_value

            amplitudes_sample[i_sample] = amp_sample_real + 1j * amp_sample_imag
    
        #print(max_evolution_time,t_max)

        # energy estimation
        if (sol_method==1):
            x_min = -np.pi/3.0
            x_max = np.pi/3.0

            cdf_1 = lambda x: CDF_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x) - (eta/4.0)

            result = find_all_roots(cdf_1, x_min, x_max) 
            if (len(result)<1):
                x_min_opt = x_min # no root
            else:
                x_min_opt = result[0]

            eigen_energy_estimate = x_min_opt
        elif (sol_method==2):
            # search step and find maximum in the region
            x_min = -np.pi/3.0
            x_max = np.pi/3.0

            cdf_1 = lambda x: CDF_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x) - (eta/4.0)
            cdf_2 = lambda x: CDF_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x) - (3*eta/4.0)
            negative_cdf_deriv = lambda x : -CDF_deriv_from_evolution_importance2(amplitudes_sample,Coeffs_sum,samples,x)

            #result = root_scalar(cdf,method='bisect',bracket=[x_min,x_max])
            result = find_all_roots(cdf_1, x_min, x_max) 
            if (len(result)<1):
                x_min_opt = x_min # no root
            else:
                x_min_opt = result[0]
            result = find_all_roots(cdf_2, x_min_opt, x_max) 
            if (len(result)<1):
                x_max_opt = x_max # no root
            else:
                x_max_opt = result[0]

            result = minimize_scalar(negative_cdf_deriv, bounds=(x_min_opt,x_max_opt), method='bounded')

            eigen_energy_estimate = result.x

        error = np.abs(eigen_energy_estimate/tau_scale-eigen_energies_exact[0][-1,0])
        #print(i_iter,error)

        error_avg += error
    error_avg /= n_iter
    if (core==0):
        print(max_n_trotter,max_n_trotter_run,error_avg)
    max_n_trotter_run_list[indx_trotter] = max_n_trotter_run
    error_for_trotter_list[indx_trotter] = error_avg
    indx_trotter +=1
    end_time = datetime.now()
    elapsed = end_time-start_time
    elapsed = elapsed.total_seconds()
    if (core==0):
        st = '# elapsed time = {elapsed} secs'.format(elapsed=elapsed)
        memory_usage(st)
# save to file
if (core==0):
    with open('dE_vs_n_trotter','w') as file_:
        s = '# eps= '+str(eps)+', n_shot= '+str(n_shot)+', n_iter='+str(n_iter)+', n_sample=' + str(n_sample)
        s += '\n'
        file_.write(s)
        indx_n_trotter = 0
        for max_n_trotter in max_n_trotter_list:
            s = '{:}'.format(max_n_trotter_run_list[indx_n_trotter])
            s += '  {:.16e}'.format(error_for_trotter_list[indx_n_trotter])
            print(s)
            s += '\n'
            file_.write(s)
            indx_n_trotter += 1





