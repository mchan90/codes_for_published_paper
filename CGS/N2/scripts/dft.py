"""
Generating FCIDUMP for the final Hamiltonian of ASP using pyscf.

Original Author: Seunghoon Lee, Jan 17, 2022

Modified by Mancheon Han, Apr 11, 2025
"""

## check
#import os
#print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
##check

import numpy as np
from pyscf import gto, scf, dft, ao2mo

# dumpERI function
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

# make output folder
import os

folder = "output"
if not os.path.isdir(folder):
    os.makedirs(folder)
    print(f"'./'{folder}' was created.")
else:
    print(f"./'{folder}' already exists.")
#


#==================================================================
# SCF
#==================================================================
mf = scf.sfx2c(scf.RKS(mol))
mf.chkfile = './output/hs_bp86.chk'
mf.max_cycle = 500
mf.conv_tol = 1.e-4
mf.xc = 'b88,p86' 
mf.scf()

mf2 = scf.newton(mf)
mf2.chkfile = './output/hs_bp86.chk'
mf2.conv_tol = 1.e-12
e_dft = mf2.kernel()

sum_eps = np.dot(mf2.mo_occ, mf2.mo_energy)

np.save('./output/dE_DFT', e_dft-sum_eps)


#==================================================================
# Dump integrals
#==================================================================
mo = mf2.mo_coeff
norb = 10
nelec = [7,7]
from pyscf import mcscf
mc = mcscf.CASCI(mf2, norb, nelec)
mc.mo_coeff = mo
act_idx = [1,2,3,4,5,6,7,8,9,10]
assert len(act_idx) == norb
mo = mc.sort_mo(act_idx)
mc.mo_coeff = mo

h1e, ecore = mc.get_h1eff()
g2e = mc.get_h2eff()
g2e = ao2mo.restore(1, g2e, norb)
header =""" &FCI NORB=  10,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
dumpERIs('./output/FCIDUMP_FULL', header, int1e=h1e, int2e=g2e, ecore=ecore)  


# save mo_coeff
def save_mo_coeff(filename: str, mo_coeff: np.ndarray, overwrite: bool = False) -> None:
    """
    Save molecular orbital coefficients to a .npy or .npz file.

    Parameters
    ----------
    filename : str
        File path to save MO coefficients. Must end with .npy or .npz.
    mo_coeff : np.ndarray
        Molecular orbital coefficient array (shape: (norb, norb) or similar).
    overwrite : bool, optional
        If True, overwrite existing file. Default is False.

    Raises
    ------
    FileExistsError
        If the file exists and overwrite is False.
    ValueError
        If filename does not end with .npy or .npz.
    """
    # Check extension
    base, ext = os.path.splitext(filename)
    if ext not in ('.npy', '.npz'):
        raise ValueError("Filename must end with .npy or .npz")

    # Check overwrite
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"File '{filename}' already exists. Use overwrite=True to replace.")

    # Save using numpy
    if ext == '.npy':
        np.save(filename, mo_coeff)
    else:
        # In .npz, store under key 'mo_coeff'
        np.savez_compressed(filename, mo_coeff=mo_coeff)

save_mo_coeff('./output/mo_coeff.npy', mo, overwrite=True)


print('done')
