"""
Generating FCIDUMP for the initial mean-field Hamiltonian of ASP using pyscf.
Numbering is based on the order of the MF states with the largest FCI amplitude.

Original Author: Seunghoon Lee, Jan 17, 2022
(companion code to Lee et al., Nature Communications 14, 1952 (2023)).
Adapted by Mancheon Han (2025) for the application of the constant
geometric speed schedule to adiabatic state preparation.
"""

import numpy as np
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

from pyscf import gto, scf, fci
 
#==================================================================
# Molecule
#==================================================================
bond_length = 3.5
mol = gto.M(
    atom='N 0 0 0; N 0 0 {bond_length}'.format(bond_length=bond_length),  #
    basis='STO-3G',
    spin=0,
    charge=0,
    unit='A'
)
mol.verbose = 5
mol.build()
mol.symmetry = False
mol.build()

material_home = './'

#==================================================================
# FCI calculation for the final Hamiltonian
#==================================================================
mf = scf.RHF(mol)
# load final Hamiltonian
ecore, h1e, g2e, norb, nelec, ms = loadERIs(material_home+'/DFT/output/FCIDUMP_FULL')
mci = fci.direct_spin1.FCI(mol)
mci.conv_tol = 1e-10
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the final Hamiltonian
np.save("./E_FULL.npy", e+ecore)
np.save("./fcivec_FULL.npy", ci)

# read exact eigenvector
fcivec_exact = ci[0]
#

#==================================================================
# FCIDUMP for initial mean-field Hamiltonian
#==================================================================
from pyscf.fci import cistring
import h5py
f = h5py.File(material_home+'/DFT/output/hs_bp86.chk', 'r')
fock = np.array(f['scf']['mo_energy'])
header =""" &FCI NORB=  10,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,
  ISYM=1,
 &END
"""
norb = 10
neleca = 7
nelecb = 7
tol    = 0.
ncore = (mol.nelectron - neleca - nelecb) // 2
act_idx = np.array([1,2,3,4,5,6,7,8,9,10]) - 1
core_idx = np.array(list(set(range(10)) - set(act_idx)))
assert len(act_idx) == norb
fock_cas = fock[act_idx]
fock_cas = np.diag(fock_cas)
if (len(core_idx)==0):
    fock_core = 0.0 
else:
    fock_core = np.sum(fock[core_idx])
print(fock_core)

na = cistring.num_strings(norb, neleca)
nb = cistring.num_strings(norb, nelecb)
assert(fcivec_exact.shape == (na, nb))
addra, addrb = np.where(abs(fcivec_exact) > tol)
strsa = cistring.addrs2str(norb, neleca, addra)
strsb = cistring.addrs2str(norb, nelecb, addrb)
occa = cistring._strs2occslst(strsa, norb)
occb = cistring._strs2occslst(strsb, norb)
ci_tol = fcivec_exact[addra,addrb]

# sort occupation strings in the order of the largest FCI amplitude for ground state
weight = np.square(ci_tol)
idx   = np.argsort(-weight)
iindx = np.argsort(idx)
weight = weight[idx]
ci_tol = ci_tol[idx]
occa = occa[idx]
occb = occb[idx]

samp = [0]

print(samp[0], weight[samp[0]], iindx[samp[0]])

sequence = [i for i in range(norb)]
epsilon = 0.0  # energy shift in Hartree
i = samp[0]
print(i, occa[i], occb[i], ci_tol[i])
fock_shift = fock_cas.copy()
for j in list(set(sequence) - set(occa[i])):
    fock_shift[j,j] += epsilon 
dumpERIs('./FCIDUMP_0', header, int1e=fock_shift, ecore=fock_core)

#==================================================================
# FCI calculation for the initial Hamiltonian
#==================================================================
ecore, h1e, g2e, norb, nelec, ms = loadERIs('./FCIDUMP_0')
mci = fci.direct_spin1.FCI(mol)
mci.conv_tol = 1e-10
mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=10, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the initial Hamiltonian
np.save("./E_0.npy", e+ecore)
np.save("./fcivec_0.npy", ci)

