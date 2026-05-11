
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()

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

output_dir = "output"
if (core==0):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"'./'{output_dir}' was created.")
    else:
        print(f"./'{output_dir}' already exists.")

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

#==================================================================
# Linearly interpolated Hamiltonian
#==================================================================
# load final Hamiltonian
interpol_list = np.linspace(0,10000,num=1001,dtype=int)
my_tasks = interpol_list[core :: cores]
local_results = []

for interpol in my_tasks:
    print('# start: {interpol}'.format(interpol=interpol))
    start = time.perf_counter()
    #==================================================================
    # FCI calculation
    #==================================================================
    ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, interpol/10000.)
    mf = scf.RHF(mol)
    mci = fci.direct_spin1.FCI(mol)
    mci.spin = 0 
    mci.conv_tol = 1e-10
    mci.max_cycle = 2000
    mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)    
    e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
    e = np.array(e)
##    do not save eigenvalues and eigenstates for the final Hamiltonian
#    np.save("./output/E%d.npy" % interpol, e+ecore)
#    np.save("./output/fcivec%d.npy" % interpol, ci)

    T_asp_est, gap = compute_T_est_and_gap(ci, e+ecore)
    print('T_asp_est: ', T_asp_est)
    local_results.append((int(interpol), T_asp_est, gap))
    elapsed = time.perf_counter() - start
    print('# done: {interpol} with {elapsed} s'.format(interpol=interpol, elapsed=elapsed))
##
comm.Barrier()
all_results = comm.gather(local_results, root=0)

if core == 0:
    # all_results is now a list of lists, one per rank
    flat = [item for sublist in all_results for item in sublist]
    # optionally sort by the interpol value
    flat.sort(key=lambda x: x[0])
    # if you just want the T_asp_est values in order of interpol_list:
    ordered_T = [t for (_, t, g) in flat]
    ordered_g = [g for (_, t, g) in flat]
    #print("Gathered results:", flat)
    # now `flat` or `ordered_T` is your combined list on rank 0
    with open('T_asp_est','w') as file_:
        for i in range(len(interpol_list)):
            s = '{:.16e}    {:.16e}    {:.16e}'.format(interpol_list[i]/10000.0, ordered_T[i], ordered_g[i])
            #print(s)
            s += '\n'
            file_.write(s)

# now explicitly finalize and exit
comm.Barrier()
MPI.Finalize()
