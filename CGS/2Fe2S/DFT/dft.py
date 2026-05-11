"""
Generating FCIDUMP for the final Hamiltonian of ASP using pyscf.

Original Author: Seunghoon Lee, Jan 17, 2022
(companion code to Lee et al., Nature Communications 14, 1952 (2023)).
Modified by Mancheon Han, Apr 11, 2025, for the application of the
constant geometric speed schedule to adiabatic state preparation.
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
mol = gto.Mole()
mol.verbose = 5
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
mf2.kernel()


#==================================================================
# Dump integrals
#==================================================================
mo = mf2.mo_coeff
norb = 12 
nelec = [7, 7]
from pyscf import mcscf
mc = mcscf.CASCI(mf, norb, nelec)
mc.mo_coeff = mo
act_cp_op = [78, 84]
act_d = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
act_idx = act_cp_op + act_d
assert len(act_idx) == norb
mo = mc.sort_mo(act_idx)
mc.mo_coeff = mo
#from pyscf import tools
#tools.molden.from_mo(mol, 'fe2s2_actonly.molden', mo[:,86:98])

h1e, ecore = mc.get_h1eff()
g2e = mc.get_h2eff()
g2e = ao2mo.restore(1, g2e, norb)
header =""" &FCI NORB=  12,NELEC=14,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,
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
