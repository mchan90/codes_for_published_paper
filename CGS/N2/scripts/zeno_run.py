"""
Find optimal path by using the quantum zeno approach
"""

import numpy as np
import copy
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
from pyscf.fci.addons import overlap
 
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
mol.verbose = 0
mol.build()
mol.symmetry = False
mol.build()

material_home = '../'

#==================================================================
# Linearly interpolated Hamiltonian
#==================================================================
# load final Hamiltonian
ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs('../FCIDUMP_0')
ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs(material_home + '/DFT/output/FCIDUMP_FULL')
info_i = (ecore_i, h1e_i, g2e_i)
info_f = (ecore_f, h1e_f, g2e_f)


fcivec_exact_0 = np.load(material_home+ "/fcivec_FULL.npy")[0] 

nroots = 1


# find fcivec_exact internally to stability
ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, 1.0)
mf = scf.RHF(mol)
mci = fci.direct_spin1.FCI(mol)
mci.spin = 0 
mci.conv_tol = 1e-10
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000, ci0=fcivec_exact_0)

fcivec_exact = copy.copy(ci)



def norm2(s_new, phi):
    ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_new)
    mf = scf.RHF(mol)
    mci = fci.direct_spin1.FCI(mol)
    mci.spin = 0 
    mci.conv_tol = 1e-10
    mci.max_cycle = 2000
    mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
    e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000, ci0=phi)
    e = np.array(e)
    ov = overlap(ci, phi, norb, nelec) 
    ov = np.abs(ov)**2
    print('norm2: ', s_new, ov)
    return ov

def norm2_1(phi):
    ov = overlap(fcivec_exact, phi, norb, nelec) 
    ov = np.abs(ov)**2
    print('norm2: ', 1.0, ov)
    return ov
import numpy as np

import bisect
from scipy.interpolate import PchipInterpolator

def RootFinding (f, a, b, c, fa=None, fb=None, fc=None, tol=1e-3, itermax=100):
    if (fa==None):
        fa = f(a)
    if (fb==None):
        fb = f(b)
    if (fc==None):
        fc = f(c)
    if (np.abs(fa)<tol):
        return a
    if (np.abs(fb)<tol):
        return b
    if (np.abs(fc)<tol):
        return c
    if (fb>fc and fc>0):
        print(b,fb, fc, c)
        return c
    print('start', fb, fc)
    # find next candidate using second order interpolation
    # f(x) = -c (x-a)^2 + f(x)
    ratio = np.sqrt(fa/(fa-fb))
    x = a + ratio * (b-a)

    if (x>c):
        x = c
    fx = f(x)
    if (np.abs(fx)<tol):
        return x

    xs = [a, b, x]
    fs = [fa, fb, fx]
    
    # xs is sorted from f abs value
    pairs = sorted(zip(fs, xs), key=lambda pair: np.abs(pair[0]))
    fs, xs = map(list, zip(*pairs))

    # generator of (x,f) for which f>0
    pos_pairs = ((x, f) for x, f in zip(xs, fs) if f > 0)
    
    # pick the one with the largest x
    x_min, f_min = max(pos_pairs, key=lambda xf: xf[0])

    try:
        x_max = min(x for x, f in zip(xs, fs) if f < 0)
    except ValueError:
        x_max = None

    if (x_max==None):
        x_max = c



    # now in the loop
    for ind in range(itermax):


        # 1) Inverse Quadratic Interpolation
        # Interpolate using three points: (x_i, f_i) for i=0,1,2
        x_i0, x_i1, x_i2 = xs
        f_i0, f_i1, f_i2 = fs
        denom0 = (f_i0 - f_i1) * (f_i0 - f_i2)
        denom1 = (f_i1 - f_i0) * (f_i1 - f_i2)
        denom2 = (f_i2 - f_i0) * (f_i2 - f_i1)
        x_iqi = (f_i1 * f_i2 / denom0) * x_i0 \
                + (f_i0 * f_i2 / denom1) * x_i1 \
                + (f_i0 * f_i1 / denom2) * x_i2

        # 2) If outside the bracket, fall back IQI -> Secant -> Bisection
        if x_iqi < x_min or x_iqi > x_max:
            # Secant method between the two points closest to zero: x_i0, x_i1
            x_sec = x_i1 - f_i1 * (x_i1 - x_i0) / (f_i1 - f_i0)
            if x_sec < x_min or x_sec > x_max:
                x_new = 0.5 * (x_min + x_max)
            else:
                x_new = x_sec
        else:
            x_new = x_iqi

        # Evaluate the function and check convergence
        f_new = f(x_new)
        if abs(f_new) < tol:
            return x_new
        if (abs(f_new) >= max(abs(fi) for fi in fs)): # worse solution; use bisection
            # update bracket before using bisection
            print('worse-solution:', x_new, f_new)
            print('boundary(before): ',x_min, x_max)
            if f_new * f_min < 0:
                x_max, f_max = x_new, f_new
            else:
                x_min, f_min = x_new, f_new
            print('boundary(after): ',x_min, x_max)

            x_new = 0.5 * (x_min + x_max)
            f_new = f(x_new)

            print('bisection:', x_new, f_new)
            if abs(f_new) < tol:
                return x_new

        print(xs, fs)
        print('Root-finding:', ind, x_new, f_new)
        print('boundary: ',x_min, x_max)

        # 4) Update the bracket
        if f_new * f_min < 0:
            x_max, f_max = x_new, f_new
        else:
            x_min, f_min = x_new, f_new

        # 3) Update the interpolation candidate
        xs.append(x_new)
        fs.append(f_new)
        # Keep the three points with the smallest absolute value
        pairs = sorted(zip(fs, xs), key=lambda p: abs(p[0]))[:3]
        fs, xs = map(list, zip(*pairs))


    # Return the midpoint when the iteration cap is reached
    return 0.5 * (x_min + x_max)


#def insert_pair(xs, ys, x_new, y_new):
#    # find the position where x_new should go to keep xs sorted
#    idx = bisect.bisect_left(xs, x_new)
#    # insert into both lists at the same index
#    xs.insert(idx, x_new)
#    ys.insert(idx, y_new)
#
#
#def RootFinding (f, a, b, c, fa=None, fb=None, fc=None, tol=1e-3, itermax=100):
#    if (fa==None):
#        fa = f(a)
#    if (fb==None):
#        print('preprocess_b:', b)
#        fb = f(b)
#    if (fc==None):
#        print('preprocess_c:', c)
#        fc = f(c)
#    if (np.abs(fa)<tol):
#        return a
#    if (np.abs(fb)<tol):
#        return b
#    if (np.abs(fc)<tol):
#        return c
#    if (fa*fc>0): # no root in the range, return c
#        return c
#
#    x_list = [c, b, a]
#    fx_list = [fc, fb, fa]
#
#    inv_f_interpol = PchipInterpolator(fx_list, x_list)
#
#    for i in range(itermax):
#        x = inv_f_interpol(0)
#        fx = f(x)
#
#        print('Root-finding:', i, x, fx)
#
#        if (np.abs(fx)<tol):
#            return x
#
#        insert_pair(fx_list, x_list, fx, x)
#
#        inv_f_interpol = PchipInterpolator(fx_list, x_list)
#
#    # solution not found; return c
#    return c
    


#def Chandrupatla (f, a, b, fa=None, fb=None,  tol=1e-3, itermax=100):
#    if (fa==None):
#        fa = f(a)
#    if (fb==None):
#        fb = f(b)
#    if (np.abs(fa)<tol):
#        return a
#    if (np.abs(fb)<tol):
#        return b
#    if (fa*fb>0): # no root in the range, return fb
#        return b
#    # Regula Falsi
#    for i in range(itermax):
#        c_rf = (a*fb - b*fa)/(fb-fa)
#
#        alpha = 0.25
#
#        c_min = a * (1-alpha) + b * alpha
#        c_max = a * (alpha) + b * (1-alpha)
#
#        if (c_rf<c_min or c_rf>c_max):
#            x = (a+b)/2 # use bisection
#        else:
#            x = c_rf
#
#        fx = f(x)
#
#        print('Chandrupatla:', x, fx)
#
#        if (np.abs(fx)<tol):
#            return x
#
#        if (fa*fx<0):
#            b, fb = x, fx
#        else:
#            a, fa = x, fa
#    return x 




# with exact overlap
s_list = [0]
hamiltonians = []
eigen_energies_exact = []
overlaps_exact = []
s = s_list[0]
alpha = 0

# computation of initial eigenstate
ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s)
mf = scf.RHF(mol)
mci = fci.direct_spin1.FCI(mol)
mci.spin = 0 
mci.conv_tol = 1e-10
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
ci = [None, None]
e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000)
e = np.array(e)
# save eigenvalues and eigenstates for the final Hamiltonian
#np.save("./output/E%d.npy" % alpha, e+ecore)
#np.save("./output/fcivec%d.npy" % alpha, ci)


eigen_energies_exact.append(e+ecore)
print(eigen_energies_exact[0])
# 

phi = ci



gamma = 0.05
cnorm2 = 1.0
r_norm2 = 1.0-gamma


# read saved ones
# s_list = [0, 0.20965698732204546, 0.3423676964541109, 0.43999257673464237, 0.5146650471425253,
#         0.5695519973601271, 0.6078369502734623, 0.6316800023636886, 0.6323448311597658]
# 
# for i_save in range(8):
#     alpha += 1
#     e = np.load("./output/E%d.npy" % alpha)
#     ci = np.load("./output/fcivec%d.npy" % alpha)
#     eigen_energies_exact.append(e)
# 
#     ov = overlap(ci, phi, norb, nelec) 
#     ov = np.abs(ov)**2
#     overlaps_exact.append(ov)
#     cnorm2 = cnorm2 * ov
#     s = s_list[alpha]
#     print('zeno', s, ov, cnorm2)
#     phi = ci

tol_norm = 1e-3

while s<1.0:

    func = lambda s: norm2(s,phi)-r_norm2
    f1   = norm2_1(phi) - r_norm2
    # find interval first
    ds_try = min(0.1, (1-s)/2)
    s_next = RootFinding(func, s, s+ds_try, 1.0, fa=1-r_norm2, fc=f1, tol=tol_norm)
    s_list.append(s_next)
    s = s_next


    alpha += 1
    ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s)
    mf = scf.RHF(mol)
    mci = fci.direct_spin1.FCI(mol)
    mci.spin = 0 
    mci.conv_tol = 1e-10
    mci.max_cycle = 2000
    mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
    ci = [None, None]
    e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000, ci0=phi)
    e = np.array(e)
    # save eigenvalues and eigenstates for the final Hamiltonian
    #np.save("./output/E%d.npy" % alpha, e+ecore)
    #np.save("./output/fcivec%d.npy" % alpha, ci)
    
    eigen_energies_exact.append(e+ecore)
    

    ov = overlap(ci, phi, norb, nelec) 
    ov = np.abs(ov)**2
    overlaps_exact.append(ov)
    cnorm2 = cnorm2 * ov
    print('zeno', s,ov, cnorm2)

    phi = ci

# get a tau-list
dtau = np.zeros(len(s_list)-1,dtype=float)
for i in range(len(s_list)-1):
    dtau[i] = np.sqrt(1-overlaps_exact[i])
dtau = dtau/np.sum(dtau)
print(dtau)
taus = np.zeros(len(s_list),dtype=float)
for i in range(len(s_list)-1):
    taus[i+1] = taus[i] + dtau[i]
print(taus)

from scipy.interpolate import PchipInterpolator

pchip = PchipInterpolator(taus, s_list)

# write data
with open('optimal_schedule','w') as file_:
    for i in range(len(s_list)):
        s = '{:.16e}    {:.16e}'.format(taus[i], s_list[i])
        print(s)
        s += '\n'
        file_.write(s)



#
##
