from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()


def brackets_from_grid_all_nomerge(x, f, fallback=True):
    """
    x: strictly increasing 1D array (len >= 3)
    f: same length
    return: [(iL, iB, iR), ...]  # one bracket per local minimum (center), left-to-right
    - plateaus (consecutive centers) are returned as separate brackets for each index
    - if fallback=True, return a single bracket when no interior local minimum exists
    """
    x = np.asarray(x); f = np.asarray(f)
    n = len(x)
    assert n >= 3

    centers = np.where((f[1:-1] <= f[:-2]) & (f[1:-1] <= f[2:]))[0] + 1  # 1..n-2

    triples = [(i-1, i, i+1) for i in centers]  # naturally in ascending order

    if not triples and fallback:
        j = int(np.argmin(f))
        if j == 0:
            triples = [(0, 1, 2)]
        elif j == n - 1:
            triples = [(n-3, n-2, n-1)]
        else:
            triples = [(j-1, j, j+1)]

    return triples

def bracket_from_grid_simple(x, f):
    """
    x: strictly increasing 1D array (len >= 3)
    f: same length
    return: (iL, iB, iR)  # index triple, Brent-style bracket
    """
    x = np.asarray(x); f = np.asarray(f)
    n = len(x)
    assert n >= 3

    # 1) Find interior local minima: f[i-1] >= f[i] <= f[i+1]
    #   (using <= is safe for floating-point ties)
    local_min_mask = (f[1:-1] <= f[:-2]) & (f[1:-1] <= f[2:])
    idxs = np.where(local_min_mask)[0] + 1

    if idxs.size > 0:
        # Pick the local minimum with the smallest f value as the center
        i = idxs[np.argmin(f[idxs])]
        return (i-1, i, i+1)

    # 2) No interior local minimum (monotonic / boundary-minimum case)
    j = int(np.argmin(f))
    if j == 0:
        return (0, 1, 2)
    elif j == n - 1:
        return (n-3, n-2, n-1)
    else:
        return (j-1, j, j+1)

def dumpERIs(fcidump, header, int1e=None, int2e=None, ecore=0., tol=1e-9):
    with open(fcidump,'w') as f:
        f.writelines(header)
        n = int1e.shape[0]
        if int2e is not None:
            for i in range(n):
                for j in range(i + 1):
                    for k in range(i + 1):
                        if k == i: 	
                            lmax = j + 1
                        else:
                            lmax = k + 1
                        for l in range(lmax):
                            line = str(int2e[i, j, k, l])\
                            + ' ' + str(i + 1) \
                            + ' ' + str(j + 1) \
                            + ' ' + str(k + 1) \
                            + ' ' + str(l + 1) + '\n'
                            if np.abs(int2e[i, j, k, l]) > tol:
                                f.writelines(line)
        if int1e is not None:
            for i in range(n):
                for j in range(i + 1):
                    line = str(int1e[i, j])\
                    + ' ' + str(i + 1) \
                    + ' ' + str(j + 1) \
                    + ' 0 0\n'
                    if np.abs(int1e[i, j]) > tol:
                        f.writelines(line)
        line = str(ecore) + ' 0 0 0 0\n'
        f.writelines(line)
    return 0

def loadERIs(fcidump):
    with open(fcidump,'r') as f:
        line = f.readline().split(',')
        norb = int(line[0].split(' ')[-1])
        nelec= int(line[1].split('=')[-1])
        ms2  = int(line[2].split('=')[-1])
        f.readline()
        f.readline()
        f.readline()
        n = norb 
        e = 0.0
        int1e = np.zeros((n,n))
        int2e = np.zeros((n,n,n,n))
        for line in f.readlines():
            data = line.split()
            ind = [int(x)-1 for x in data[1:]]
            if ind[2] == -1 and ind[3]== -1:
                if ind[0] == -1 and ind[1] ==-1:
                    e = float(data[0])
                else :
                    int1e[ind[0],ind[1]] = float(data[0])
                    int1e[ind[1],ind[0]] = float(data[0])
            else:
                int2e[ind[0], ind[1], ind[2], ind[3]] = float(data[0])
                int2e[ind[1], ind[0], ind[2], ind[3]] = float(data[0])
                int2e[ind[0], ind[1], ind[3], ind[2]] = float(data[0])
                int2e[ind[1], ind[0], ind[3], ind[2]] = float(data[0])
                int2e[ind[2], ind[3], ind[0], ind[1]] = float(data[0])
                int2e[ind[3], ind[2], ind[0], ind[1]] = float(data[0])
                int2e[ind[2], ind[3], ind[1], ind[0]] = float(data[0])
                int2e[ind[3], ind[2], ind[1], ind[0]] = float(data[0])
    return e,int1e,int2e,norb,nelec,ms2

def fcidump_interpolate(info_i, info_f, coeff):
    ecore_i, int1e_i, int2e_i = info_i
    ecore_f, int1e_f, int2e_f = info_f
    ecore = ecore_i * (1.0 - coeff) + ecore_f * coeff
    int1e = int1e_i * (1.0 - coeff) + int1e_f * coeff
    int2e = int2e_i * (1.0 - coeff) + int2e_f * coeff
    return ecore, int1e, int2e

from pyscf import gto, scf, fci
 
#==================================================================
# Molecule
#==================================================================
mol = gto.Mole()
mol.verbose = 0
mol.atom = '''
 Fe                 5.22000000    1.05000000   -7.95000000
 S                  3.86000000   -0.28000000   -9.06000000
 S                  5.00000000    0.95000000   -5.66000000
 S                  4.77000000    3.18000000   -8.74000000
 S                  7.23000000    0.28000000   -8.38000000
 Fe                 5.88000000   -1.05000000   -9.49000000
 S                  6.10000000   -0.95000000  -11.79000000
 S                  6.33000000   -3.18000000   -8.71000000
 C                  6.00000000    4.34000000   -8.17000000
 H                  6.46000000    4.81000000   -9.01000000
 H                  5.53000000    5.08000000   -7.55000000
 H                  6.74000000    3.82000000   -7.60000000
 C                  3.33000000    1.31000000   -5.18000000
 H                  2.71000000    0.46000000   -5.37000000
 H                  3.30000000    1.54000000   -4.13000000
 H                  2.97000000    2.15000000   -5.73000000
 C                  5.10000000   -4.34000000   -9.28000000
 H                  5.56000000   -5.05000000   -9.93000000
 H                  4.67000000   -4.84000000   -8.44000000
 H                  4.34000000   -3.81000000   -9.81000000
 C                  7.77000000   -1.31000000  -12.27000000
 H                  7.84000000   -1.35000000  -13.34000000
 H                  8.42000000   -0.54000000  -11.90000000
 H                  8.06000000   -2.25000000  -11.86000000
'''
mol.basis = 'tzp-dkh'
mol.charge = -2
mol.spin = 0
mol.build()
mol.symmetry = False
mol.build()

material_home = '../../../../'

ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs('../FCIDUMP_0')
ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs(material_home + 'DFT/output/FCIDUMP_FULL')
info_i = (ecore_i, h1e_i, g2e_i)
info_f = (ecore_f, h1e_f, g2e_f)

h1e_diff = h1e_f - h1e_i
h2e_diff = g2e_f - g2e_i

import os

#output_dir = "output"
#if (core==0):
#    if not os.path.isdir(output_dir):
#        os.makedirs(output_dir)
#        print(f"'./'{output_dir}' was created.")
#    else:
#        print(f"./'{output_dir}' already exists.")

def compute_T_est_and_gap (fcivec, e):

    mci = fci.direct_spin1.FCI(mol)

    s0 = None 
    max_T_ASPest = 0.0
    gap_list = []
    nroots = 10
    assert len(e) == len(fcivec) and len(fcivec) == nroots
    for i in range(nroots):
        mult = fci.spin_op.spin_square0(fcivec[i], norb, nelec)[1]
        if np.abs(mult - 1.) < 0.01:
            if s0 is None:
                s0 = i 
                e0 = e[i]
                continue
            rdm1, rdm2 = mci.trans_rdm12(fcivec[i], fcivec[s0], norb, nelec) 
            epsil = np.abs(np.einsum('pq,qp', h1e_diff, rdm1) + 0.5 * np.einsum('pqrs,pqrs', h2e_diff, rdm2))
            de = e[i] - e0 
            T_ASPest = epsil / (de * de)
            gap_list.append(de)
            if T_ASPest > max_T_ASPest:
                max_T_ASPest = T_ASPest
    print(gap_list)
    gap_min = min(gap_list)
    print(gap_min)

    return max_T_ASPest, gap_min

comm.Barrier()

# Use previously found values
s_list = []
T_list = []
gap_list = []
with open('../est/T_asp_est','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        s_list.append(float(ls[0]))
        T_list.append(float(ls[1]))
        gap_list.append(float(ls[2]))


i_min, i_mid, i_max = bracket_from_grid_simple(s_list, gap_list)

dip_list = brackets_from_grid_all_nomerge(s_list, gap_list)

n_dips = len(dip_list)

print('# n_dips: ', n_dips)
print('# dip_lists: ', dip_list)

if (cores<n_dips):
    print('# increase the number of cores at least to: ', n_dips)

base = cores // n_dips
rem  = cores %  n_dips
cut = rem * (base + 1)
if core < cut:
    i_dip    = core // (base + 1)
    local    = core %  (base + 1)
    cores_per_dips = base + 1
else:
    r2     = core - cut
    i_dip    = rem + r2 // base 
    local    = r2 %  base  
    cores_per_dips = base

print(core, i_dip, cores_per_dips)


i_min, i_mid, i_max = dip_list[i_dip]


ss = s_list[i_min:i_max+1]
Ts = T_list[i_min:i_max+1]
gaps = gap_list[i_min:i_max+1]

print(ss[0],ss[2])

T_asp_max = max(T_list)

#==================================================================
# Linearly interpolated Hamiltonian
#==================================================================
# load final Hamiltonian
eps_x=1e-06

DeltaS = ss[2]-ss[0]

n_add  = max(cores_per_dips,2)

iter_max = int(np.log(DeltaS/eps_x)/np.log((max(base,2)+1)/2)) + 1

print("cores_per_dips: ",cores_per_dips, "eps_x: ", eps_x)
print("iter_max: ", iter_max)

for iter_ in range(iter_max):
    if (cores_per_dips==1):
        s_cores = [ss[0] + 1/3*DeltaS, ss[0] + 2/3*DeltaS]
    else:
        s_core = ss[0] + (local+1)/(cores_per_dips+1) * DeltaS
        s_cores = [s_core]

    for s_core in s_cores:
        print("iter_: ", iter_, "s_core", s_core)
        
        local_results = []
        
        start = time.perf_counter()
        #==================================================================
        # FCI calculation
        #==================================================================
        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_core)
        mf = scf.RHF(mol)
        mci = fci.direct_spin1.FCI(mol)
        mci.spin = 0 
        mci.conv_tol = 1e-10
        mci.max_cycle = 2000
        mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
        e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
        e = np.array(e)
        
        T_asp_est, gap = compute_T_est_and_gap(ci, e+ecore)
        print('T_asp_est: ', T_asp_est)
        print('gap: ', gap)
        local_results.append((s_core, T_asp_est, gap))
        elapsed = time.perf_counter() - start
        print('# done: {s_core} with {elapsed} s'.format(s_core=s_core, elapsed=elapsed))
        ##
    if (local==0):
        local_results.append((ss[0], Ts[0], gaps[0]))
        local_results.append((ss[2], Ts[2], gaps[2]))
    comm.Barrier()

    all_results = comm.allgather(local_results)

    # all_results is now a list of lists, one per rank
    flat = [item for sublist in all_results for item in sublist]
    # optionally sort by the s_core value
    flat.sort(key=lambda x: x[0])
    # if you just want the T_asp_est values in order of s_core list:
    ordered_s = [s for (s, _, _) in flat]
    ordered_T = [t for (_, t, _) in flat]
    ordered_g = [g for (_, _, g) in flat]

    # check convergence
    # filter 
    filtered = [(s, t, g) for (s, t, g) in flat if ss[0] <= s <= ss[2]]

    ordered_s_filtered = [s for (s, _, _) in filtered]
    ordered_T_filtered = [t for (_, t, _) in filtered]
    ordered_g_filtered = [g for (_, _, g) in filtered]

    i_min, i_mid, i_max = bracket_from_grid_simple(ordered_s_filtered, ordered_g_filtered)

    ss = ordered_s_filtered[i_min:i_max+1]
    print(ss)
    Ts = ordered_T_filtered[i_min:i_max+1]
    print(Ts)
    gaps = ordered_g_filtered[i_min:i_max+1]
    print(gaps)

    if (core==0):
        slen = len(ordered_s)
        with open('T_asp_est.'+str(iter_),'w') as file_:
            for i in range(slen):
                s = '{:.16e}    {:.16e}    {:.16e}'.format(ordered_s[i], ordered_T[i], ordered_g[i])
                s += '\n'
                file_.write(s)

    T_asp_max = max(T_asp_max,max(ordered_T))

    DeltaS = ss[2]-ss[0]
    is_last = (iter_== iter_max-1)

    #DeltaS<eps_x or 
    if (is_last):
        j = int(np.argmin(ordered_g))  # 0=left, 1=middle, 2=right minimum
        print ('# gap minimum is found at ', ordered_s[j])
        print ('# with the gap minimum value of ', ordered_g[j])
        print ('# and the estimated T_asp is ', T_asp_max)
        break

comm.Barrier()
if (core==0):
    with open('DONE','w') as file_:
        s = str(n_dips)
        file_.write(s)

# now explicitly finalize and exit
comm.Barrier()
MPI.Finalize()
