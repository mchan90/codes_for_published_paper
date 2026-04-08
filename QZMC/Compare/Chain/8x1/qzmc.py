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

n_x     = 8
n_y     = 1
n_site  = n_x * n_y
n_qubit = 2*n_site
dim = 2**n_qubit

n_dimer = n_site//2
# For a chain
n_inter = n_dimer 
# For a ladder
#n_inter = 2*n_dimer


# %%
Uc           = 5.0
mu           = Uc/2.0 # half-filling fermi level
n_electrons  = n_site # half-filling
t_hop        = 1.0
t_intra      = t_hop
#if (n_x==8):
#t_inters            = np.linspace(0.0,1.0,11)
#t_inters = np.linspace(0.0,1.0,2)
t_inters            = np.linspace(0.0,1.0,n_site+1)
#elif (n_x==4):
#    t_inters     = np.linspace(0.0,1.0,3)
nt_inter     = len(t_inters)
for it_inter in range(nt_inter):
    t_inters[it_inter] *= t_hop
kinetic_parts       = []
hamiltonians        = []
it_inter            = 0
bc = BoundaryCondition.PERIODIC

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
    factor = 4*n_dimer # 2 for spin, n_dimer for geometry
    pauli_list = ['XZXII']
    coeffs     = np.asarray([factor * -0.5 * (t_inters[alpha+1]-t_inters[alpha])])
    for i_pauli in range(len(pauli_list)):
        for i_qubit in range(n_qubit-len(pauli_list[i_pauli])):
            pauli_list[i_pauli] = 'I' + pauli_list[i_pauli]
    dH = SparsePauliOp(pauli_list, coeffs=coeffs)

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
row      = []
col      = []
data     = []
for jj in range(dim_sub[0]):
    j = indx_sub[isec][jj]
    i = ApplyOrderReverse(j)
    ii = iindx_sub[isec][i]
    row.append(ii)
    col.append(jj)
    data.append(OrderReverseSign(i))
Or = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[isec], dim_sub[isec]))


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
row      = []
col      = []
data     = []
for jj in range(dim_sub[0]):
    j = indx_sub[isec][jj]
    i = ApplyParticleHole(j)
    ii = iindx_sub[isec][i]
    row.append(ii)
    col.append(jj)
    data.append(ParticleHoleSign(i))
Ph = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[isec], dim_sub[isec]))

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
row      = []
col      = []
data     = []
for jj in range(dim_sub[0]):
    j = indx_sub[isec][jj]
    i = ApplySpinReverse(j)
    ii = iindx_sub[isec][i]
    row.append(ii)
    col.append(jj)
    data.append(SpinReverseSign(i))
Sr = sparse.csc_matrix((data, (row, col)), shape=(dim_sub[isec], dim_sub[isec]))

# compose the projection operator
isec = 0
identity = sparse.eye(dim_sub[isec])
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
# find the reduced basis by finding eigenstate of projection operator
isec = 0
row = []
col = []
data = []
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
            row.append(indx_S)
            col.append(jj)
            data.append(np.conj(eigen_v[jjj,-1]))
basis_transform = sparse.csc_matrix((data, (row,col)),shape=(num_S,dim_sub[isec]))
#print(basis_transform[0,:])
if (core==0):
    print('reduced dimension: ', dim_reduced)

# %%
eigen_energies_exact = []
eigen_vectors_exact = []
H_subs = []
n_eig = 1
for isec in range(nsec):
    eigen_energies_exact.append(np.zeros((n_hamiltonians,n_eig),dtype=float))
    eigen_vectors_exact.append(np.zeros((n_hamiltonians,dim_reduced,n_eig),dtype=complex))
    H_subs.append([])

start_time = datetime.now()
for isec in range(nsec):
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

        H_sub_reduced = basis_transform@(H_sub)@basis_transform.conj().T

        H_subs[isec].append(H_sub_reduced)

        # diagonalize sectorized hamiltonian
        eigen_e, eigen_v = sparse.linalg.eigsh(H_sub_reduced, k=n_eig, which='SA')
        eigen_energies_exact[isec][alpha,:] = eigen_e
        if (core==0):
            print(eigen_e[0])
        eigen_e, eigen_v = sparse.linalg.eigsh(H_sub, k=n_eig, which='SA')
        if (core==0):
            print(eigen_e[0])
    gc.collect()
end_time = datetime.now()
elapsed = end_time-start_time
elapsed = elapsed.total_seconds()
if (core==0):
    st = '# {percent}%, elapsed time = {elapsed} secs'.format(percent=(100),elapsed=elapsed)
    memory_usage(st)


# %%
def Apply_ExactEvolution(isec, alpha, eps, time, v):
    M = H_subs[isec][alpha]-eps*sparse.eye(dim_reduced)
    M = -1j * M * time
    w = sparse.linalg.expm_multiply(M,v)
    return w

def Apply_ExactGaussian(isec, alpha, eps, beta, v):
    M = H_subs[isec][alpha]-eps*sparse.eye(dim_reduced)
    M = -0.5 * beta**2 * M@M
    w = sparse.linalg.expm_multiply(M,v)
    return w


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
        pauli_sub_reduced = basis_transform@pauli_sub@basis_transform.conj().T
        sectored_pauli[isec][alpha][ihd]=pauli_sub_reduced
        #print(sectored_pauli[isec][alpha][ihd])

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



beta   = 3.0
n_shot = 10000
nmc       = int(16*n_site**2)

error = 0.0
# pick timelists
eps = eigen_energies_exact[0][0,0]
norms_qzmc              = np.ones((n_hamiltonians),dtype=float)
eigen_energies_qzmc     = np.zeros((n_hamiltonians),dtype=float)
eigen_energies_qzmc[0] = eigen_energies_exact[0][0,0]

for alpha in range(1,n_hamiltonians):
    start_time = datetime.now()
                

    nhd1 = len(hamiltonian_diffs_reduced_list[alpha-1])
    nhd1_ = nhd1 # no constant contribution

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
        times = np.random.normal(0.0, beta, size=2*(alpha))
        i_time = 0
        phase  = 0.0
        phi = copy.copy(phi_start)
        # 
        for alpha_ in range(1,alpha):
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

        for alpha_ in reversed(range(1,alpha)):
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
            times = np.random.normal(0.0, beta, size=2*(alpha))
            i_time = 0
            phase  = 0.0
            phi = copy.copy(phi_start)
            # 
            for alpha_ in range(1,alpha):
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

            for alpha_ in reversed(range(1,alpha)):
                phase += eigen_energies_qzmc[alpha_] * times[i_time]
                phi = Apply_ExactEvolution(isec,alpha_,0.0,times[i_time],phi)
                i_time += 1

            amplitude = phi_start.conj().T@phi
            amplitude *= np.exp(1j*phase)
            result_values_core[ind_pub_core] = amplitude.real
            ind_pub_core += 1

    # dE2 (i_obs==2)
    i_obs = 2
    for ihd in range(nhd1):
        for imc in range(nmc):
            # check my turn
            my_turn = ind_pub>=i_start and ind_pub<i_end
            ind_pub += 1
            if (not my_turn):
                continue
            # initialize
            times = np.random.normal(0.0, beta, size=2*(alpha))
            i_time = 0
            phase  = 0.0
            phi = copy.copy(phi_start)
            # 
            for alpha_ in range(1,alpha):
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

            for alpha_ in reversed(range(1,alpha)):
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

    if (core==0):
        st = '# {percent:.1f}%, elapsed time = {elapsed} secs'.format(percent=((alpha)/(n_hamiltonians-1)*100),elapsed=elapsed)
        memory_usage(st)
        print(alpha, norms_qzmc[alpha], eigen_energies_qzmc[alpha]-eigen_energies_exact[isec][alpha,0])
        if (alpha<n_hamiltonians-1):
            print('precision of the predictor for next', eps-eigen_energies_exact[isec][alpha+1,0])
        st = '# {percent:.1f}%'.format(percent=((alpha)/(n_hamiltonians-1)*100))
        print(st)
# %%
if (core==0):
    with open('qzmc_with_exact_evolution','w') as file_:
        s = '# beta = '+str(beta)+', n_shot= '+str(n_shot)+', nmc= '+str(nmc)
        s += '\n'
        file_.write(s)
        for alpha in range(n_hamiltonians):
            s = '{:}'.format(t_inters[alpha])
            s += '  {:.16e}'.format(norms_qzmc[alpha])
            s += '  {:.16e}'.format(eigen_energies_qzmc[alpha])
            s += '\n'
            file_.write(s)
##
