# %%
#!/home/mchan/Doc/venv_new_qiskit/bin/python3 -u
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

n_x     = 4
n_y     = 1
n_site  = n_x * n_y
n_qubit = 2*n_site
dim = 2**n_qubit

n_dimer = n_site//2
# For a chain
n_inter = n_dimer - 1 # open boundary condition
# For a dimer 
#n_inter = 2*n_dimer


# %%
Uc           = 4.0
mu           = Uc/2.0 # half-filling fermi level
n_electrons  = n_site # half-filling
t_hop        = 1.0
t_intra      = t_hop
#if (n_x==8):
t_inters     = np.linspace(0.0,1.0,5)
#elif (n_x==4):
#    t_inters     = np.linspace(0.0,1.0,3)
#t_inters = [0.0, 
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

# %%
if (core==0):
    print('# Hamiltonian differences')
hamiltonian_diffs = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs.append((hamiltonians[alpha+1]-hamiltonians[alpha]).simplify())
    if (core==0):
        print(alpha, single_line(str(hamiltonian_diffs[alpha])))


# %%
if (core==0):
    print('# Hamiltonian differences_list')
hamiltonian_diffs_list = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs_list.append(hamiltonian_diffs[alpha].to_list())
    if (core==0):
        print(hamiltonian_diffs_list[alpha])

# %%
hamiltonian_diffs_reduced = []
for alpha in range(n_hamiltonians-1):
    if (n_x==4):
        factor = 2*2
    if (n_x==8):
        factor = 2*2*2 # 2 for spin, 2 for XX and YY, 2 for symmetry
    pauli_list = ['IIIXZXII']
    coeffs     = np.asarray([factor * -0.5 * (t_inters[alpha+1]-t_inters[alpha])])
    for i_pauli in range(len(pauli_list)):
        for i_qubit in range(n_qubit-len(pauli_list[i_pauli])):
            pauli_list[i_pauli] = 'I' + pauli_list[i_pauli]
    dH = SparsePauliOp(pauli_list, coeffs=coeffs)

    if (n_x==8):
        factor = 2*2 # 2 for spin, 2 for XX and YY
        pauli_list = ['XZXIIIIII']
        coeffs     = np.asarray([factor * -0.5 * (t_inters[alpha+1]-t_inters[alpha])])
        for i_pauli in range(len(pauli_list)):
            for i_qubit in range(n_qubit-len(pauli_list[i_pauli])):
                pauli_list[i_pauli] = 'I' + pauli_list[i_pauli]
        dH += SparsePauliOp(pauli_list, coeffs=coeffs)
    hamiltonian_diffs_reduced.append(dH)
    if (core==0):
        print(alpha, single_line(str(hamiltonian_diffs_reduced[alpha])))
if (core==0):
    print('# Hamiltonian differences_reduced_list')
hamiltonian_diffs_reduced_list = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs_reduced_list.append(hamiltonian_diffs_reduced[alpha].to_list())
    if (core==0):
        print(hamiltonian_diffs_reduced_list[alpha])

# %%
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

norms_exact  = np.ones((nsec,n_hamiltonians),dtype=float)
for isec in range(nsec):
    phi = eigen_vectors_exact[isec][0,:,0]
    for alpha in range(1,n_hamiltonians):
        coeff = eigen_vectors_exact[isec][alpha,:,0].conj()@phi
        phi = coeff * eigen_vectors_exact[isec][alpha,:,0]
        norms_exact[isec,alpha] = np.real(phi.conj()@phi)
        if (core==0):
            print(alpha, norms_exact[isec,alpha])

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

#def Apply_ExactEvolution(isec, alpha, eps, time, v):
#    M = H_subs[isec][alpha]-eps*sparse.eye(dim_sub[isec])
#    M = -1j * M * time
#    w = sparse.linalg.expm_multiply(M,v)
#    return w
#
#def Apply_ExactGaussian(isec, alpha, eps, beta, v):
#    M = H_subs[isec][alpha]-eps*sparse.eye(dim_sub[isec])
#    M = -0.5 * beta**2 * M@M
#    w = sparse.linalg.expm_multiply(M,v)
#    return w

#def ExactEvolution (isec, alpha, eps, time):
#    Vl = copy.deepcopy(eigen_vectors_exact[isec][alpha,:,:])
#    evol = np.zeros((dim_sub[isec],dim_sub[isec]),dtype=complex)
#    vec = np.zeros((dim_sub[isec]),dtype=complex)
#    for k in range(dim_sub[isec]):
#        vec[k] = np.exp(-1j*time*(eigen_energies_exact[isec][alpha,k]-eps))
#    exp_d = np.diag(vec)
#    evol = Vl@exp_d@Vl.conj().T
#    return evol
#
#def ExactGaussian (isec, alpha, eps, beta):
#    Vl = copy.deepcopy(eigen_vectors_exact[isec][alpha,:,:])
#    evol = np.zeros((dim_sub[isec],dim_sub[isec]),dtype=complex)
#    vec = np.zeros((dim_sub[isec]),dtype=float)
#    for k in range(dim_sub[isec]):
#        vec[k] = np.exp(-0.5 * beta ** 2 * (eigen_energies_exact[isec][alpha,k]-eps)**2)
#    exp_d = np.diag(vec)
#    evol = Vl@exp_d@Vl.conj().T
#    return evol

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
#v = eigen_vectors_exact[0][0,:,0]
#v1 = ExactGaussian(0,1,0.1,2)@v
#v2 = Apply_ExactGaussian(0,1,0.1,2,v)
#print(np.max(np.abs(v1-v2)))
#
#v1 = ExactEvolution(0,1,0.1,2)@v
#v2 = Apply_ExactEvolution(0,1,0.1,2,v)
#print(np.max(np.abs(v1-v2)))

# %%
# prepare sectored pauli (project it)
isec = 0
sectored_pauli = [[None for _ in range(n_hamiltonians)] for _ in range(nsec)]
for alpha in range(n_hamiltonians-1):
    nhd = len(hamiltonian_diffs_reduced_list[alpha])
    sectored_pauli[isec][alpha] = [None for _ in range(nhd)]
    for ihd in range(nhd):
        pauli = hamiltonian_diffs_reduced_list[alpha][ihd][0]
        pauli_op = SparsePauliOp(pauli)
        pauli_sparse = pauli_op.to_matrix(sparse=True)
        pauli_sparse.eliminate_zeros()
        jsec = isec
        row      = []
        col      = []
        data     = []
        for ii in range(dim_sub[isec]):
            i = indx_sub[isec][ii]
            for ind in range(pauli_sparse.indptr[i],pauli_sparse.indptr[i+1]):
                # j is always in indx_sub[isec], because the Hamiltonian does not mix it
                #print(i,j)
                j = pauli_sparse.indices[ind]
                jj = iindx_sub[jsec][j]
                # project it to isec subspace
                if (jj<0):
                    continue
                row.append(jj)
                col.append(ii)
                #print(ii,jj)
                data.append(pauli_sparse.data[ind])
        pauli_sub = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[jsec], dim_sub[isec]))
        pauli_sub_reduced = basis_transform@(pauli_sub.toarray())@basis_transform.conj().T
        sectored_pauli[isec][alpha][ihd]=pauli_sub_reduced
        #print(sectored_pauli[isec][alpha][ihd])

#v = eigen_vectors_exact[0][0,:,0]
#v1 = TrotterEvolution(0,0,1.23,0.1,2,0)@v
#v2 = Apply_TrotterEvolution(0,0,1.23,0.1,2,0,v)
#print(np.max(np.abs(v1-v2)))

# %%
#isec = 0
#phi_start = eigen_vectors_kinetic[isec][-1,:,0] # kinetic part solution (Hartree Fock)
#eta = np.abs(phi_start.conj().T@eigen_vectors_exact[isec][-1,:,0])**2
#print(eta)

# %%
# use pre-computed C99
if (n_hamiltonians==2):
    C99 = 6.3959839
if (n_hamiltonians==3):
    C99 = 8.6154461
if (n_hamiltonians==4):
    C99 = 10.75940
if (n_hamiltonians==5):
    C99 = 12.829328
if (n_hamiltonians==9):
    C99 = 20.73744

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
#eta_0 = 1.0

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
n_shot = 2048
nmc    = 16384
n_iter = 30
# beta parameter list
# N_alpha = 4 is fixed
#beta_list = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#beta_list = [2.4, 2.6, 2.8, 3.0, 4.0]
#beta_list = np.linspace(0.1,6.0,num=41)
beta_list = np.arange(0.1, 5.1, 0.1)
#beta_list = [2.0]
norms_for_beta_list     = np.zeros(len(beta_list),dtype=float)
error_for_beta_list     = np.zeros(len(beta_list),dtype=float)


# %%
# fix random seed for now
#np.random.seed(42)

# %%
# from exact gaussian. no iteration is needed
isec = 0
indx_beta =0
for beta in beta_list:
    i_core = indx_beta%cores
    if (core!=i_core):
        indx_beta += 1
        continue
    eps = eigen_energies_exact[0][0,0]
    norms_qzmc              = np.ones((n_hamiltonians),dtype=float)
    eigen_energies_qzmc     = np.zeros((n_hamiltonians),dtype=float)
    eigen_energies_qzmc[0] = eigen_energies_exact[0][0,0]
    phi = copy.copy(phi_start)
    phi = Apply_ExactGaussian(isec,0,eigen_energies_qzmc[0],beta,phi)
    norms_qzmc[0] = np.real(phi.conj().T@phi)
    for alpha in range(1,n_hamiltonians):
        phi_0 = copy.copy(phi)
        phi_1 = (H_subs[isec][alpha]-H_subs[isec][alpha-1])@phi
        phi_1 = Apply_ExactGaussian(isec,alpha,eps,beta,phi_1)
        phi = Apply_ExactGaussian(isec,alpha,eps,beta,phi)

        norms_qzmc[alpha] = np.real(phi.conj().T@phi)
        dE1 = np.real(phi.conj().T@phi_1)/norms_qzmc[alpha]
        eigen_energies_qzmc[alpha] = eigen_energies_qzmc[alpha-1] + dE1

        # rerun with computed energy
        eps = eigen_energies_qzmc[alpha] 
        phi = phi_0
        phi_1 = (H_subs[isec][alpha]-H_subs[isec][alpha-1])@phi
        phi_1 = Apply_ExactGaussian(isec,alpha,eps,beta,phi_1)
        phi = Apply_ExactGaussian(isec,alpha,eps,beta,phi)
        norms_qzmc[alpha] = np.real(phi.conj().T@phi)
        dE1 = np.real(phi.conj().T@phi_1)/norms_qzmc[alpha]
        eigen_energies_qzmc[alpha] = eigen_energies_qzmc[alpha-1] + dE1

        if (alpha<n_hamiltonians-1):
            phi_2 = (H_subs[isec][alpha+1]-H_subs[isec][alpha])@phi
            dE2 = np.real(phi.conj().T@phi_2)/norms_qzmc[alpha]
            eps = eigen_energies_qzmc[alpha] + dE2
        #if (core==0):
        #    st = '# {percent:.1f}%, elapsed time = {elapsed} secs'.format(percent=((alpha)/(n_hamiltonians-1)*100),elapsed=elapsed)
        #    memory_usage(st)
        #    print(alpha, norms_qzmc[alpha], eigen_energies_qzmc[alpha]-eigen_energies_exact[isec][alpha,0])
        #    if (alpha<n_hamiltonians-1):
        #        print('precision of the predictor for next', eps-eigen_energies_exact[isec][alpha+1,0])
        #    st = '# {percent:.1f}%'.format(percent=((alpha)/(n_hamiltonians-1)*100))
        #    print(st)
        gc.collect()
    error = np.abs(eigen_energies_qzmc[-1] - eigen_energies_exact[0][-1,0])
    norms_for_beta_list[indx_beta] = norms_qzmc[-1]
    error_for_beta_list[indx_beta] = error
    indx_beta += 1

indx_beta =0
for beta in beta_list:
    i_core = indx_beta%cores
    norm = comm.bcast(norms_for_beta_list[indx_beta],root=i_core)
    norms_for_beta_list[indx_beta] = norm
    error = comm.bcast(error_for_beta_list[indx_beta],root=i_core)
    error_for_beta_list[indx_beta] = error
    if (core==0):
        print(beta,norms_for_beta_list[indx_beta],error_for_beta_list[indx_beta])
    indx_beta += 1
        

# %%
if (core==0):
    with open('dE_vs_evolution','w') as file_:
        s = '# from exact gaussian'
        s += '\n'
        file_.write(s)
        indx_beta = 0
        for beta in beta_list:
            s = '{:}'.format(beta*C99)
            s += '  {:.16e}'.format(error_for_beta_list[indx_beta])
        #    print(s)
            s += '\n'
            file_.write(s)
            indx_beta += 1


# %%
#indx_beta =0
#for beta in beta_list:
#    error = 0.0
#    max_time = 0.0
#    for i_iter in range(n_iter):
#        # pick timelists
#        n_obs = 3
#        # 0; norm, 1; dE1, 2; dE2
#        O_timelists         = [[[None for _ in range(nmc)] for _ in range(n_obs)] for _ in range(n_hamiltonians)]
#        # %%
#        if (core==0):
#            for alpha in range(1,n_hamiltonians):
#                for i_obs in range(n_obs):
#                    for imc in range(nmc):
#                        times = np.random.normal(0.0, beta, size=2*(alpha+1))
#                        O_timelists[alpha][i_obs][imc] = times
#        O_timelists = comm.bcast(O_timelists,root=0)
#        # max time
#        for alpha in range(1,n_hamiltonians):
#            for i_obs in range(n_obs):
#                for imc in range(nmc):
#                    time_sum = np.sum(np.abs(O_timelists[alpha][i_obs][imc]))
#                    max_time = max(max_time,time_sum)
#
#        eps = eigen_energies_exact[0][0,0]
#        norms_qzmc              = np.ones((n_hamiltonians),dtype=float)
#        eigen_energies_qzmc     = np.zeros((n_hamiltonians),dtype=float)
#        eigen_energies_qzmc[0] = eigen_energies_exact[0][0,0]
#
#        for alpha in range(1,n_hamiltonians):
#            start_time = datetime.now()
#                        
#
#            nhd1 = len(hamiltonian_diffs_reduced_list[alpha-1])
#            nhd1_ = nhd1 # no constant contribution
#
#            if (alpha<n_hamiltonians-1):
#                nhd2 = len(hamiltonian_diffs_reduced_list[alpha])
#            else:
#                nhd2 = 0
#            nhd2_ = nhd2 # no constant contribution
#
#            n_pubs = nmc * (1+nhd1_+nhd2_)
#
#            n_pubs_for_ = [0 for _ in range(cores)]
#            remainder         = n_pubs%cores
#            for i_core in range(cores):
#                n_pubs_for_[i_core] = n_pubs//cores
#                if (i_core<remainder):
#                    n_pubs_for_[i_core] += 1
#            #if (core==0 and alpha==1):
#            #    print('# of different quantum circuits to run = ', n_pubs)
#
#            i_start         = sum(n_pubs_for_[:core])
#            i_end           = i_start + n_pubs_for_[core]
#
#            ind_pub = 0
#            ind_pub_core = 0
#
#            result_values_core = [0.0 for _ in range(n_pubs_for_[core])]
#
#            # norm (i_obs==0)
#            i_obs = 0
#            for imc in range(nmc):
#                # check my turn
#                my_turn = ind_pub>=i_start and ind_pub<i_end
#                ind_pub += 1
#                if (not my_turn):
#                    continue
#                # initialize
#                times = O_timelists[alpha][i_obs][imc]
#                i_time = 0
#                phase  = 0.0
#                phi = copy.copy(phi_start)
#                # 
#                for alpha_ in range(alpha):
#                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
#                    phi = ExactEvolution(isec,alpha_,0.0,times[i_time])@phi
#                    i_time += 1
#                # P_{\alpha}
#                phase += eps * times[i_time]
#                phi = ExactEvolution(isec,alpha,0.0,times[i_time])@phi
#                i_time += 1
#
#                # P_{\alpha}
#                phase += eps * times[i_time]
#                phi = ExactEvolution(isec,alpha,0.0,times[i_time])@phi
#                i_time += 1
#
#                for alpha_ in reversed(range(alpha)):
#                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
#                    phi = ExactEvolution(isec,alpha_,0.0,times[i_time])@phi
#                    i_time += 1
#
#                amplitude = phi_start.conj().T@phi
#                amplitude *= np.exp(1j*phase)
#                result_values_core[ind_pub_core] = amplitude.real
#                ind_pub_core += 1
#
#            # dE1 (i_obs==1)
#            i_obs = 1
#            for ihd in range(nhd1):
#                for imc in range(nmc):
#                    # check my turn
#                    my_turn = ind_pub>=i_start and ind_pub<i_end
#                    ind_pub += 1
#                    if (not my_turn):
#                        continue
#                    # initialize
#                    times = O_timelists[alpha][i_obs][imc]
#                    i_time = 0
#                    phase  = 0.0
#                    phi = copy.copy(phi_start)
#                    # 
#                    for alpha_ in range(alpha):
#                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
#                        phi = ExactEvolution(isec,alpha_,0.0,times[i_time])@phi
#                        i_time += 1
#                    # apply pauli
#                    phi = sectored_pauli[0][alpha-1][ihd]@phi
#
#                    # P_{\alpha}
#                    phase += eps * times[i_time]
#                    phi = ExactEvolution(isec,alpha,0.0,times[i_time])@phi
#                    i_time += 1
#
#                    # P_{\alpha}
#                    phase += eps * times[i_time]
#                    phi = ExactEvolution(isec,alpha,0.0,times[i_time])@phi
#                    i_time += 1
#
#                    for alpha_ in reversed(range(alpha)):
#                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
#                        phi = ExactEvolution(isec,alpha_,0.0,times[i_time])@phi
#                        i_time += 1
#
#                    amplitude = phi_start.conj().T@phi
#                    amplitude *= np.exp(1j*phase)
#                    result_values_core[ind_pub_core] = amplitude.real
#                    ind_pub_core += 1
#
#            # dE2 (i_obs==2)
#            i_obs = 2
#            for ihd in range(nhd1):
#                for imc in range(nmc):
#                    # check my turn
#                    my_turn = ind_pub>=i_start and ind_pub<i_end
#                    ind_pub += 1
#                    if (not my_turn):
#                        continue
#                    # initialize
#                    times = O_timelists[alpha][i_obs][imc]
#                    i_time = 0
#                    phase  = 0.0
#                    phi = copy.copy(phi_start)
#                    # 
#                    for alpha_ in range(alpha):
#                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
#                        phi = ExactEvolution(isec,alpha_,0.0,times[i_time])@phi
#                        i_time += 1
#                    # P_{\alpha}
#
#                    phase += eps * times[i_time]
#                    phi = ExactEvolution(isec,alpha,0.0,times[i_time])@phi
#                    i_time += 1
#
#                    # apply pauli
#                    phi = sectored_pauli[0][alpha][ihd]@phi
#
#                    # P_{\alpha}
#                    phase += eps * times[i_time]
#                    phi = ExactEvolution(isec,alpha,0.0,times[i_time])@phi
#                    i_time += 1
#
#                    for alpha_ in reversed(range(alpha)):
#                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
#                        phi = ExactEvolution(isec,alpha_,0.0,times[i_time])@phi
#                        i_time += 1
#
#                    amplitude = phi_start.conj().T@phi
#                    amplitude *= np.exp(1j*phase)
#                    result_values_core[ind_pub_core] = amplitude.real
#                    ind_pub_core += 1
#            for i in range(n_pubs_for_[core]):
#                computed_value = result_values_core[i]
#
#                # shot errors
#                p_up = (computed_value + 1.0)/2.0
#                sample = np.random.binomial(n_shot,p_up)
#                shot_error = 2*(sample/(n_shot) - p_up)
#                computed_value += shot_error
#
#                result_values_core[i] = computed_value
#            # bcast
#            #print(result_values_core)
#            comm.Barrier()
#            result_values = []
#            for i_core in range(cores):
#                if (n_pubs_for_[i_core]==0):
#                    continue
#                result_values_temp = comm.bcast(result_values_core,root=i_core)
#                result_values += result_values_temp
#                comm.Barrier()
#
#            # compute energy eigenvalues
#            i_meas = 0
#
#            # 0; norm
#            norm    = 0.0
#            i_obs   = 0
#            for imc in range(nmc):
#                norm   += result_values[i_meas]
#                i_meas += 1
#            # 1; dE1
#            dE1norm = 0.0
#            i_obs   = 1
#            for ihd in range(nhd1):
#                coeff = hamiltonian_diffs_reduced_list[alpha-1][ihd][1]
#                for imc in range(nmc):
#                    dE1norm += coeff *result_values[i_meas]
#                    i_meas += 1
#            # 2; dE2
#            dE2norm = 0.0
#            i_obs   = 2
#            for ihd in range(nhd2):
#                coeff = hamiltonian_diffs_reduced_list[alpha][ihd][1]
#                for imc in range(nmc):
#                    dE2norm += coeff *result_values[i_meas]
#                    i_meas += 1
#
#            norm = norm.real
#            dE1norm = dE1norm.real
#            dE2norm = dE2norm.real
#    
#            dE1norm /= norm
#            dE2norm /= norm
#            norm    /= nmc
#
#            eigen_energies_qzmc[alpha] = eigen_energies_qzmc[alpha-1] + dE1norm
#            norms_qzmc[alpha] = norm
#
#            if (alpha<n_hamiltonians-1):
#                eps = eigen_energies_qzmc[alpha] + dE2norm
#                eps = eps.real
#
#            #if (core==0):
#            #    st = '# {percent:.1f}%, elapsed time = {elapsed} secs'.format(percent=((alpha)/(n_hamiltonians-1)*100),elapsed=elapsed)
#            #    memory_usage(st)
#            #    print(alpha, norms_qzmc[alpha], eigen_energies_qzmc[alpha]-eigen_energies_exact[isec][alpha,0])
#            #    if (alpha<n_hamiltonians-1):
#            #        print('precision of the predictor for next', eps-eigen_energies_exact[isec][alpha+1,0])
#            #    st = '# {percent:.1f}%'.format(percent=((alpha)/(n_hamiltonians-1)*100))
#            #    print(st)
#        error += np.abs(eigen_energies_qzmc[-1] - eigen_energies_exact[0][-1,0])
#    error /= n_iter
#    error_for_beta_list[indx_beta] = error
#    if (core==0):
#        print(beta*C99, max_time, error)
#    indx_beta += 1
#

# %%
#if (core==0):
#    with open('dE_vs_evolution','w') as file_:
#        s = '# nmc= '+str(nmc)+', n_shot= '+str(n_shot)+', n_iter= '+str(n_iter)
#        s += '\n'
#        file_.write(s)
#        indx_beta = 0
#        for beta in beta_list:
#            s = '{:}'.format(beta*(n_hamiltonians))
#            s += '  {:.16e}'.format(error_for_beta_list[indx_beta])
#            print(s)
#            s += '\n'
#            file_.write(s)
#            indx_beta += 1
#

# %%
# repetition test
beta = 1.6
nmcs = [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
error_for_nmcs = np.zeros((len(nmcs)),dtype=float)
n_shot = 2048
n_iter = 30


# %%
# compute n_pubs_sum
n_pubs_sum = 0
for alpha in range(1,n_hamiltonians):
    nhd1 = len(hamiltonian_diffs_reduced_list[alpha-1])
    nhd1_ = nhd1 # no constant contribution
    
    if (alpha<n_hamiltonians-1):
        nhd2 = len(hamiltonian_diffs_reduced_list[alpha])
    else:
        nhd2 = 0
    nhd2_ = nhd2 # no constant contribution
    
    # rerun correction
    n_pubs = (2+2*nhd1_+nhd2_)

    n_pubs_sum += n_pubs

    #print(n_pubs_sum)

# preprocess
indx_nmc = 0
for nmc in nmcs:
    error = 0.0
    #max_time = 0.0
    for i_iter in range(n_iter):
        # pick timelists
#       no need to save timelists
#        n_obs = 3 
#        # 0; norm, 1; dE1, 2; dE2
#        O_timelists         = [[[None for _ in range(nmc)] for _ in range(n_obs)] for _ in range(n_hamiltonians)]
#        # %%
#        if (core==0):
#            for alpha in range(1,n_hamiltonians):
#                for i_obs in range(n_obs):
#                    for imc in range(nmc):
#                        times = np.random.normal(0.0, beta, size=2*(alpha+1))
#                        O_timelists[alpha][i_obs][imc] = times
#        comm.Barrier()
#        O_timelists = comm.bcast(O_timelists,root=0)
#        # max time
#        for alpha in range(1,n_hamiltonians):
#            for i_obs in range(n_obs):
#                for imc in range(nmc):
#                    time_sum = np.sum(np.abs(O_timelists[alpha][i_obs][imc]))
#                    max_time = max(max_time,time_sum)

        eps = eigen_energies_exact[0][0,0]
        norms_qzmc              = np.ones((n_hamiltonians),dtype=float)
        eigen_energies_qzmc     = np.zeros((n_hamiltonians),dtype=float)
        eigen_energies_qzmc[0] = eigen_energies_exact[0][0,0]

        for alpha in range(1,n_hamiltonians):
            start_time = datetime.now()
                        

            nhd1 = len(hamiltonian_diffs_reduced_list[alpha-1])
            nhd1_ = nhd1 # no constant contribution

            n_pubs = nmc * (1+nhd1_)

            n_pubs_for_ = [0 for _ in range(cores)]
            remainder         = n_pubs%cores
            for i_core in range(cores):
                n_pubs_for_[i_core] = n_pubs//cores
                if (i_core<remainder):
                    n_pubs_for_[i_core] += 1
            #if (core==0 and alpha==1):
            #    print('# of different quantum circuits to run = ', n_pubs)

            i_start         = sum(n_pubs_for_[:core])
            i_end           = i_start + n_pubs_for_[core]

            ind_pub = 0
            ind_pub_core = 0

            result_values_core = [0.0 for _ in range(n_pubs_for_[core])]

            # norm (i_obs==0)
            i_obs = 0
            for imc in range(nmc):
                # check my turn
                my_turn = ind_pub>=i_start and ind_pub<i_end
                ind_pub += 1
                if (not my_turn):
                    continue
                # initialize
                times = np.random.normal(0.0, beta, size=2*(alpha+1))
                i_time = 0
                phase  = 0.0
                phi = copy.copy(phi_start)
                # 
                for alpha_ in range(alpha):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                    i_time += 1
                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                i_time += 1

                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                i_time += 1

                for alpha_ in reversed(range(alpha)):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                    i_time += 1

                amplitude = phi_start.conj().T@phi
                amplitude *= np.exp(1j*phase)
                result_values_core[ind_pub_core] = amplitude.real
                ind_pub_core += 1

            # dE1 (i_obs==1)
            i_obs = 1
            for ihd in range(nhd1):
                for imc in range(nmc):
                    # check my turn
                    my_turn = ind_pub>=i_start and ind_pub<i_end
                    ind_pub += 1
                    if (not my_turn):
                        continue
                    # initialize
                    times = np.random.normal(0.0, beta, size=2*(alpha+1))
                    i_time = 0
                    phase  = 0.0
                    phi = copy.copy(phi_start)
                    # 
                    for alpha_ in range(alpha):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                        i_time += 1
                    # apply pauli
                    phi = sectored_pauli[0][alpha-1][ihd]@phi

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                    i_time += 1

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                    i_time += 1

                    for alpha_ in reversed(range(alpha)):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                        i_time += 1

                    amplitude = phi_start.conj().T@phi
                    amplitude *= np.exp(1j*phase)
                    result_values_core[ind_pub_core] = amplitude.real
                    ind_pub_core += 1

            for i in range(n_pubs_for_[core]):
                computed_value = result_values_core[i]

                # shot errors
                p_up = (computed_value + 1.0)/2.0
                sample = np.random.binomial(n_shot,p_up)
                shot_error = 2*(sample/(n_shot) - p_up)
                computed_value += shot_error

                result_values_core[i] = computed_value
            # bcast
            #print(result_values_core)
            comm.Barrier()
            result_values = []
            for i_core in range(cores):
                if (n_pubs_for_[i_core]==0):
                    continue
                result_values_temp = comm.bcast(result_values_core,root=i_core)
                result_values += result_values_temp
                comm.Barrier()

            # compute energy eigenvalues
            i_meas = 0

            # 0; norm
            norm    = 0.0
            i_obs   = 0
            for imc in range(nmc):
                norm   += result_values[i_meas]
                i_meas += 1
            # 1; dE1
            dE1norm = 0.0
            i_obs   = 1
            for ihd in range(nhd1):
                coeff = hamiltonian_diffs_reduced_list[alpha-1][ihd][1]
                for imc in range(nmc):
                    dE1norm += coeff *result_values[i_meas]
                    i_meas += 1

            norm = norm.real
            dE1norm = dE1norm.real
    
            dE1norm /= norm
            norm    /= nmc

            eps = eigen_energies_qzmc[alpha-1] + dE1norm

            # rerun

            if (alpha<n_hamiltonians-1):
                nhd2 = len(hamiltonian_diffs_reduced_list[alpha])
            else:
                nhd2 = 0
            nhd2_ = nhd2 # no constant contribution

            n_pubs = nmc * (1+nhd1_+nhd2_)

            n_pubs_for_ = [0 for _ in range(cores)]
            remainder         = n_pubs%cores
            for i_core in range(cores):
                n_pubs_for_[i_core] = n_pubs//cores
                if (i_core<remainder):
                    n_pubs_for_[i_core] += 1
            #if (core==0 and alpha==1):
            #    print('# of different quantum circuits to run = ', n_pubs)

            i_start         = sum(n_pubs_for_[:core])
            i_end           = i_start + n_pubs_for_[core]

            ind_pub = 0
            ind_pub_core = 0

            result_values_core = [0.0 for _ in range(n_pubs_for_[core])]

            # norm (i_obs==0)
            i_obs = 0
            for imc in range(nmc):
                # check my turn
                my_turn = ind_pub>=i_start and ind_pub<i_end
                ind_pub += 1
                if (not my_turn):
                    continue
                # initialize
                times = np.random.normal(0.0, beta, size=2*(alpha+1))
                i_time = 0
                phase  = 0.0
                phi = copy.copy(phi_start)
                # 
                for alpha_ in range(alpha):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                    i_time += 1
                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                i_time += 1

                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                i_time += 1

                for alpha_ in reversed(range(alpha)):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                    i_time += 1

                amplitude = phi_start.conj().T@phi
                amplitude *= np.exp(1j*phase)
                result_values_core[ind_pub_core] = amplitude.real
                ind_pub_core += 1

            # dE1 (i_obs==1)
            i_obs = 1
            for ihd in range(nhd1):
                for imc in range(nmc):
                    # check my turn
                    my_turn = ind_pub>=i_start and ind_pub<i_end
                    ind_pub += 1
                    if (not my_turn):
                        continue
                    # initialize
                    times = np.random.normal(0.0, beta, size=2*(alpha+1))
                    i_time = 0
                    phase  = 0.0
                    phi = copy.copy(phi_start)
                    # 
                    for alpha_ in range(alpha):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                        i_time += 1
                    # apply pauli
                    phi = sectored_pauli[0][alpha-1][ihd]@phi

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                    i_time += 1

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                    i_time += 1

                    for alpha_ in reversed(range(alpha)):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                        i_time += 1

                    amplitude = phi_start.conj().T@phi
                    amplitude *= np.exp(1j*phase)
                    result_values_core[ind_pub_core] = amplitude.real
                    ind_pub_core += 1

            # dE2 (i_obs==2)
            i_obs = 2
            for ihd in range(nhd2):
                for imc in range(nmc):
                    # check my turn
                    my_turn = ind_pub>=i_start and ind_pub<i_end
                    ind_pub += 1
                    if (not my_turn):
                        continue
                    # initialize
                    times = np.random.normal(0.0, beta, size=2*(alpha+1))
                    i_time = 0
                    phase  = 0.0
                    phi = copy.copy(phi_start)
                    # 
                    for alpha_ in range(alpha):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                        i_time += 1
                    # P_{\alpha}

                    phase += eps * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                    i_time += 1

                    # apply pauli
                    phi = sectored_pauli[0][alpha][ihd]@phi

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_ExactEvolution(isec,alpha,0.0,times[i_time],phi)
                    i_time += 1

                    for alpha_ in reversed(range(alpha)):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                        i_time += 1

                    amplitude = phi_start.conj().T@phi
                    amplitude *= np.exp(1j*phase)
                    result_values_core[ind_pub_core] = amplitude.real
                    ind_pub_core += 1
            for i in range(n_pubs_for_[core]):
                computed_value = result_values_core[i]

                # shot errors
                p_up = (computed_value + 1.0)/2.0
                sample = np.random.binomial(n_shot,p_up)
                shot_error = 2*(sample/(n_shot) - p_up)
                computed_value += shot_error

                result_values_core[i] = computed_value
            # bcast
            #print(result_values_core)
            comm.Barrier()
            result_values = []
            for i_core in range(cores):
                if (n_pubs_for_[i_core]==0):
                    continue
                result_values_temp = comm.bcast(result_values_core,root=i_core)
                result_values += result_values_temp
                comm.Barrier()

            # compute energy eigenvalues
            i_meas = 0

            # 0; norm
            norm    = 0.0
            i_obs   = 0
            for imc in range(nmc):
                norm   += result_values[i_meas]
                i_meas += 1
            # 1; dE1
            dE1norm = 0.0
            i_obs   = 1
            for ihd in range(nhd1):
                coeff = hamiltonian_diffs_reduced_list[alpha-1][ihd][1]
                for imc in range(nmc):
                    dE1norm += coeff *result_values[i_meas]
                    i_meas += 1
            # 2; dE2
            dE2norm = 0.0
            i_obs   = 2
            for ihd in range(nhd2):
                coeff = hamiltonian_diffs_reduced_list[alpha][ihd][1]
                for imc in range(nmc):
                    dE2norm += coeff *result_values[i_meas]
                    i_meas += 1

            norm = norm.real
            dE1norm = dE1norm.real
            dE2norm = dE2norm.real
    
            dE1norm /= norm
            dE2norm /= norm
            norm    /= nmc

            eigen_energies_qzmc[alpha] = eigen_energies_qzmc[alpha-1] + dE1norm
            norms_qzmc[alpha] = norm

            if (alpha<n_hamiltonians-1):
                eps = eigen_energies_qzmc[alpha] + dE2norm
                eps = eps.real


            #if (core==0):
            #    st = '# {percent:.1f}%, elapsed time = {elapsed} secs'.format(percent=((alpha)/(n_hamiltonians-1)*100),elapsed=elapsed)
            #    memory_usage(st)
            #    print(alpha, norms_qzmc[alpha], eigen_energies_qzmc[alpha]-eigen_energies_exact[isec][alpha,0])
            #    if (alpha<n_hamiltonians-1):
            #        print('precision of the predictor for next', eps-eigen_energies_exact[isec][alpha+1,0])
            #    st = '# {percent:.1f}%'.format(percent=((alpha)/(n_hamiltonians-1)*100))
            #    print(st)
        error += np.abs(eigen_energies_qzmc[-1] - eigen_energies_exact[0][-1,0])
    error /= n_iter
    error_for_nmcs[indx_nmc] = error
    if (core==0):
        print(n_pubs_sum*nmc, error)
    indx_nmc += 1

# %%
if (core==0):
    with open('dE_vs_repetitions_with_exact','w') as file_:
        s = '# beta = '+str(beta)+', n_shot= '+str(n_shot)+', n_iter='+str(n_iter)
        s += '\n'
        file_.write(s)
        indx_nmc = 0
        for nmc in nmcs:
            s = '{:}'.format(n_pubs_sum*nmc)
            s += '  {:.16e}'.format(error_for_nmcs[indx_nmc])
        #    print(s)
            s += '\n'
            file_.write(s)
            indx_nmc += 1
#
#
## %%
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
#
## check sparsity of phi_start
#if (core==0): 
#    # check sparsity
#    nnz = 0
#    for i in range(dim_sub[isec]):
#        if (np.abs(phi_start[i])>1e-10):
#            nnz += 1
#    print('nnz: ',nnz, nnz/(dim_sub[isec]))
#
## test Trotter evolution
#v = phi_start
#time = 1
#n_trotter = 1000
## 
## # Trotter test #1
#start_time = datetime.now()
#w = Apply_TrotterEvolution(0,-1,time,0.0,n_trotter,1,v)
#end_time = datetime.now()
#elapsed = end_time-start_time
#elapsed = elapsed.total_seconds()
#if (core==0):
#    print('Ver # 1: ',n_trotter)
#    print('Trotter step: ',n_trotter)
#    print('Elapsed time: ',elapsed)
#
## Trotter test #2
#start_time = datetime.now()
#w = Apply_TrotterEvolution2(0,-1,time,0.0,n_trotter,1,v)
#end_time = datetime.now()
#elapsed = end_time-start_time
#elapsed = elapsed.total_seconds()
#if (core==0):
#    print('Ver # 2: ',n_trotter)
#    print('Trotter step: ',n_trotter)
#    print('Elapsed time: ',elapsed)


# for trotter
beta = 1.6
nmc  = 16384
max_n_trotter_list = [2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#max_n_trotter_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
error_for_trotter_list = np.zeros((len(max_n_trotter_list)),dtype=float)
max_n_trotter_run_list = np.zeros((len(max_n_trotter_list)),dtype=float)
n_shot = 2048
n_iter = 30
trotter_permutation_indx     = 1
trotter_permutation_indx_dag = 1 # fixed to 1 for a fair comparision


# %%
def NumberOfTrotterSteps(alpha, max_n_trotter):
    trotter_factor = (n_dimer + t_inters[alpha]/t_intra * n_inter)/(n_dimer + n_inter)
    return max(1,int(trotter_factor*max_n_trotter))

# %%
# trotterized case
indx_max_n_trotter = 0
for max_n_trotter in max_n_trotter_list:
    start_time = datetime.now()

    n_trotters = [NumberOfTrotterSteps(alpha, max_n_trotter) for alpha in range(n_hamiltonians)]

    error = 0.0
    #max_time = 0.0
    for i_iter in range(n_iter):
        # pick timelists
        # n_obs = 3
        # 0; norm, 1; dE1, 2; dE2
        # O_timelists         = [[[None for _ in range(nmc)] for _ in range(n_obs)] for _ in range(n_hamiltonians)]
        # # %%
        # if (core==0):
        #     for alpha in range(1,n_hamiltonians):
        #         for i_obs in range(n_obs):
        #             for imc in range(nmc):
        #                 times = np.random.normal(0.0, beta, size=2*(alpha+1))
        #                 O_timelists[alpha][i_obs][imc] = times
        # comm.Barrier()
        # O_timelists = comm.bcast(O_timelists,root=0)
        # max time
        #for alpha in range(1,n_hamiltonians):
        #    for i_obs in range(n_obs):
        #        for imc in range(nmc):
        #            time_sum = np.sum(np.abs(O_timelists[alpha][i_obs][imc]))
        #            max_time = max(max_time,time_sum)

        eps = eigen_energies_exact[0][0,0]
        norms_qzmc              = np.ones((n_hamiltonians),dtype=float)
        eigen_energies_qzmc     = np.zeros((n_hamiltonians),dtype=float)
        eigen_energies_qzmc[0] = eigen_energies_exact[0][0,0]

        for alpha in range(1,n_hamiltonians):
                        

            nhd1 = len(hamiltonian_diffs_reduced_list[alpha-1])
            nhd1_ = nhd1 # no constant contribution

            n_pubs = nmc * (1+nhd1_)

            n_pubs_for_ = [0 for _ in range(cores)]
            remainder         = n_pubs%cores
            for i_core in range(cores):
                n_pubs_for_[i_core] = n_pubs//cores
                if (i_core<remainder):
                    n_pubs_for_[i_core] += 1
            #if (core==0 and alpha==1):
            #    print('# of different quantum circuits to run = ', n_pubs)

            i_start         = sum(n_pubs_for_[:core])
            i_end           = i_start + n_pubs_for_[core]

            ind_pub = 0
            ind_pub_core = 0

            result_values_core = [0.0 for _ in range(n_pubs_for_[core])]

            # norm (i_obs==0)
            i_obs = 0
            for imc in range(nmc):
                # check my turn
                my_turn = ind_pub>=i_start and ind_pub<i_end
                ind_pub += 1
                if (not my_turn):
                    continue
                # initialize
                times = np.random.normal(0.0, beta, size=2*(alpha+1))
                i_time = 0
                phase  = 0.0
                phi = copy.copy(phi_start)
                # 
                for alpha_ in range(alpha):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx,phi)
                    i_time += 1
                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx,phi)
                i_time += 1

                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx_dag,phi)
                i_time += 1

                for alpha_ in reversed(range(alpha)):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx_dag,phi)
                    i_time += 1

                amplitude = phi_start.conj().T@phi
                amplitude *= np.exp(1j*phase)
                result_values_core[ind_pub_core] = amplitude.real
                ind_pub_core += 1

            # dE1 (i_obs==1)
            i_obs = 1
            for ihd in range(nhd1):
                for imc in range(nmc):
                    # check my turn
                    my_turn = ind_pub>=i_start and ind_pub<i_end
                    ind_pub += 1
                    if (not my_turn):
                        continue
                    # initialize
                    times = np.random.normal(0.0, beta, size=2*(alpha+1))
                    i_time = 0
                    phase  = 0.0
                    phi = copy.copy(phi_start)
                    # 
                    for alpha_ in range(alpha):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx,phi)
                        i_time += 1
                    # apply pauli
                    phi = sectored_pauli[0][alpha-1][ihd]@phi

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx,phi)
                    i_time += 1

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx_dag,phi)
                    i_time += 1

                    for alpha_ in reversed(range(alpha)):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx_dag,phi)
                        i_time += 1

                    amplitude = phi_start.conj().T@phi
                    amplitude *= np.exp(1j*phase)
                    result_values_core[ind_pub_core] = amplitude.real
                    ind_pub_core += 1

            for i in range(n_pubs_for_[core]):
                computed_value = result_values_core[i]

                # shot errors
                p_up = (computed_value + 1.0)/2.0
                sample = np.random.binomial(n_shot,p_up)
                shot_error = 2*(sample/(n_shot) - p_up)
                computed_value += shot_error

                result_values_core[i] = computed_value
            # bcast
            #print(result_values_core)
            comm.Barrier()
            result_values = []
            for i_core in range(cores):
                if (n_pubs_for_[i_core]==0):
                    continue
                result_values_temp = comm.bcast(result_values_core,root=i_core)
                result_values += result_values_temp
                comm.Barrier()

            # compute energy eigenvalues
            i_meas = 0

            # 0; norm
            norm    = 0.0
            i_obs   = 0
            for imc in range(nmc):
                norm   += result_values[i_meas]
                i_meas += 1
            # 1; dE1
            dE1norm = 0.0
            i_obs   = 1
            for ihd in range(nhd1):
                coeff = hamiltonian_diffs_reduced_list[alpha-1][ihd][1]
                for imc in range(nmc):
                    dE1norm += coeff *result_values[i_meas]
                    i_meas += 1

            norm = norm.real
            dE1norm = dE1norm.real
    
            dE1norm /= norm
            norm    /= nmc

            eps = eigen_energies_qzmc[alpha-1] + dE1norm

            # rerun 
            if (alpha<n_hamiltonians-1):
                nhd2 = len(hamiltonian_diffs_reduced_list[alpha])
            else:
                nhd2 = 0
            nhd2_ = nhd2 # no constant contribution

            n_pubs = nmc * (1+nhd1_+nhd2_)

            n_pubs_for_ = [0 for _ in range(cores)]
            remainder         = n_pubs%cores
            for i_core in range(cores):
                n_pubs_for_[i_core] = n_pubs//cores
                if (i_core<remainder):
                    n_pubs_for_[i_core] += 1
            #if (core==0 and alpha==1):
            #    print('# of different quantum circuits to run = ', n_pubs)

            i_start         = sum(n_pubs_for_[:core])
            i_end           = i_start + n_pubs_for_[core]

            ind_pub = 0
            ind_pub_core = 0

            result_values_core = [0.0 for _ in range(n_pubs_for_[core])]

            # norm (i_obs==0)
            i_obs = 0
            for imc in range(nmc):
                # check my turn
                my_turn = ind_pub>=i_start and ind_pub<i_end
                ind_pub += 1
                if (not my_turn):
                    continue
                # initialize
                times = np.random.normal(0.0, beta, size=2*(alpha+1))
                i_time = 0
                phase  = 0.0
                phi = copy.copy(phi_start)
                # 
                for alpha_ in range(alpha):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx,phi)
                    i_time += 1
                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx,phi)
                i_time += 1

                # P_{\alpha}
                phase += eps * times[i_time]
                phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx_dag,phi)
                i_time += 1

                for alpha_ in reversed(range(alpha)):
                    phase += eigen_energies_qzmc[alpha_] * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx_dag,phi)
                    i_time += 1

                amplitude = phi_start.conj().T@phi
                amplitude *= np.exp(1j*phase)
                result_values_core[ind_pub_core] = amplitude.real
                ind_pub_core += 1

            # dE1 (i_obs==1)
            i_obs = 1
            for ihd in range(nhd1):
                for imc in range(nmc):
                    # check my turn
                    my_turn = ind_pub>=i_start and ind_pub<i_end
                    ind_pub += 1
                    if (not my_turn):
                        continue
                    # initialize
                    times = np.random.normal(0.0, beta, size=2*(alpha+1))
                    i_time = 0
                    phase  = 0.0
                    phi = copy.copy(phi_start)
                    # 
                    for alpha_ in range(alpha):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx,phi)
                        i_time += 1
                    # apply pauli
                    phi = sectored_pauli[0][alpha-1][ihd]@phi

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx,phi)
                    i_time += 1

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx_dag,phi)
                    i_time += 1

                    for alpha_ in reversed(range(alpha)):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx_dag,phi)
                        i_time += 1

                    amplitude = phi_start.conj().T@phi
                    amplitude *= np.exp(1j*phase)
                    result_values_core[ind_pub_core] = amplitude.real
                    ind_pub_core += 1

            # dE2 (i_obs==2)
            i_obs = 2
            for ihd in range(nhd2):
                for imc in range(nmc):
                    # check my turn
                    my_turn = ind_pub>=i_start and ind_pub<i_end
                    ind_pub += 1
                    if (not my_turn):
                        continue
                    # initialize
                    times = np.random.normal(0.0, beta, size=2*(alpha+1))
                    i_time = 0
                    phase  = 0.0
                    phi = copy.copy(phi_start)
                    # 
                    for alpha_ in range(alpha):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx,phi)
                        i_time += 1
                    # P_{\alpha}

                    phase += eps * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx,phi)
                    i_time += 1

                    # apply pauli
                    phi = sectored_pauli[0][alpha][ihd]@phi

                    # P_{\alpha}
                    phase += eps * times[i_time]
                    phi = Apply_TrotterEvolution(isec, alpha, times[i_time], 0.0, n_trotters[alpha], trotter_permutation_indx_dag,phi)
                    i_time += 1

                    for alpha_ in reversed(range(alpha)):
                        phase += eigen_energies_qzmc[alpha_] * times[i_time]
                        phi = Apply_TrotterEvolution(isec, alpha_, times[i_time], 0.0, n_trotters[alpha_], trotter_permutation_indx_dag,phi)
                        i_time += 1

                    amplitude = phi_start.conj().T@phi
                    amplitude *= np.exp(1j*phase)
                    result_values_core[ind_pub_core] = amplitude.real
                    ind_pub_core += 1
            for i in range(n_pubs_for_[core]):
                computed_value = result_values_core[i]

                # shot errors
                p_up = (computed_value + 1.0)/2.0
                sample = np.random.binomial(n_shot,p_up)
                shot_error = 2*(sample/(n_shot) - p_up)
                computed_value += shot_error

                result_values_core[i] = computed_value
            # bcast
            #print(result_values_core)
            comm.Barrier()
            result_values = []
            for i_core in range(cores):
                if (n_pubs_for_[i_core]==0):
                    continue
                result_values_temp = comm.bcast(result_values_core,root=i_core)
                result_values += result_values_temp
                comm.Barrier()

            # compute energy eigenvalues
            i_meas = 0

            # 0; norm
            norm    = 0.0
            i_obs   = 0
            for imc in range(nmc):
                norm   += result_values[i_meas]
                i_meas += 1
            # 1; dE1
            dE1norm = 0.0
            i_obs   = 1
            for ihd in range(nhd1):
                coeff = hamiltonian_diffs_reduced_list[alpha-1][ihd][1]
                for imc in range(nmc):
                    dE1norm += coeff *result_values[i_meas]
                    i_meas += 1
            # 2; dE2
            dE2norm = 0.0
            i_obs   = 2
            for ihd in range(nhd2):
                coeff = hamiltonian_diffs_reduced_list[alpha][ihd][1]
                for imc in range(nmc):
                    dE2norm += coeff *result_values[i_meas]
                    i_meas += 1

            norm = norm.real
            dE1norm = dE1norm.real
            dE2norm = dE2norm.real
    
            dE1norm /= norm
            dE2norm /= norm
            norm    /= nmc

            eigen_energies_qzmc[alpha] = eigen_energies_qzmc[alpha-1] + dE1norm
            norms_qzmc[alpha] = norm

            if (alpha<n_hamiltonians-1):
                eps = eigen_energies_qzmc[alpha] + dE2norm
                eps = eps.real


            #if (core==0):
            #    st = '# {percent:.1f}%, elapsed time = {elapsed} secs'.format(percent=((alpha)/(n_hamiltonians-1)*100),elapsed=elapsed)
            #    memory_usage(st)
            #    print(alpha, norms_qzmc[alpha], eigen_energies_qzmc[alpha]-eigen_energies_exact[isec][alpha,0])
            #    if (alpha<n_hamiltonians-1):
            #        print('precision of the predictor for next', eps-eigen_energies_exact[isec][alpha+1,0])
            #    st = '# {percent:.1f}%'.format(percent=((alpha)/(n_hamiltonians-1)*100))
            #    print(st)
        error += np.abs(eigen_energies_qzmc[-1] - eigen_energies_exact[0][-1,0])
    error /= n_iter
    error_for_trotter_list[indx_max_n_trotter] = error
    max_n_trotter_run_list[indx_max_n_trotter] = 2*np.sum(n_trotters)
    if (core==0):
        print(2*np.sum(n_trotters), error)
    indx_max_n_trotter += 1
    end_time = datetime.now()
    elapsed = end_time-start_time
    elapsed = elapsed.total_seconds()
    if (core==0):
        st = '# elapsed time = {elapsed} secs'.format(elapsed=elapsed)
        memory_usage(st)



# %%
if (core==0):
    with open('dE_vs_n_trotter','w') as file_:
        s = '# beta= '+str(beta)+', n_shot= '+str(n_shot)+', n_iter= '+str(n_iter)+', nmc= '+str(nmc)
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

