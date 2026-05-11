import numpy as np
from math import comb
from mpi4py import MPI
import time
import copy
from pyscf import gto,scf,fci,lib
from scipy.sparse.linalg import LinearOperator, onenormest, expm_multiply
from pyscf.fci.addons import overlap
from pyscf.fci import direct_spin1, cistring
from warnings import warn
from scipy.interpolate import PchipInterpolator

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()

einsum = lib.einsum

nroots = 1 # we only find a ground state

# local expm routine



class ERIs():
    def __init__(self, h1e, g2e, mo_coeff=None):
        ''' 
            h1e: 1-elec Hamiltonian in site basis 
            g2e: 2-elec Hamiltonian in site basis
                 chemists notation (pr|qs)=<pq|rs>
            mo_coeff: moa, mob 
        '''
        if mo_coeff is not None:
            mo = mo_coeff
            
            h1e = einsum('uv,up,vq->pq',h1e,mo,mo)
            g2e = einsum('uvxy,up,vr->prxy',g2e,mo,mo)
            g2e = einsum('prxy,xq,ys->prqs',g2e,mo,mo)

        self.mo_coeff = mo_coeff
        self.h1e = h1e
        self.g2e = g2e

class CIObject():
    def __init__(self, fcivec, norb, nelec):
        '''
           fcivec: ground state spin1 fcivec
           norb: size of site basis
           nelec: nea, neb
        '''
        self.r = fcivec.copy()
        self.i = np.zeros_like(fcivec) # no imaginary part? why?
        self.norb = norb
        self.nelec = nelec

def compute_weight(ci, fci):
    norb = ci.norb
    nelec= ci.nelec

    rr1 = overlap(ci.r, fci, norb, nelec)
    ir1 = overlap(ci.i, fci, norb, nelec)
    ov = rr1 + 1j*ir1
    return np.vdot(ov,ov).real

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

def compute_hamiltonian_diff(info_i, info_f):
    ecore_i, int1e_i, int2e_i = info_i
    ecore_f, int1e_f, int2e_f = info_f
    ecore_diff = ecore_f -ecore_i 
    int1e_diff = int1e_f -int1e_i 
    int2e_diff = int2e_f -int2e_i 
    return ecore_diff, int1e_diff, int2e_diff

from scipy.optimize import brentq
def global_optimize(f, f_deriv, x_min, x_max, omega_max, opt_flag):
    delta_x = np.pi/(2.1*omega_max) # delta_x<np.pi/(2.0*omega_max)
    nx = int((x_max-x_min)/delta_x) + 1
    roots = []
    xs = np.linspace(x_min,x_max,nx)
    derivs = np.zeros((nx), dtype=float)
    for ix in range(nx):
        x = xs[ix]
        derivs[ix] = f_deriv(x)
    
    if (opt_flag==0): # minimize
        for ix in range(nx-1):
            # check signs
            if (derivs[ix]>0.0): # not a minimum
                continue
            if (derivs[ix+1]<0.0): # do not change a sign
                continue
            else:
                root =brentq(f_deriv,xs[ix],xs[ix+1])
                roots.append(root)
    else: # maximize
        for ix in range(nx-1):
            # check signs
            if (derivs[ix]<0.0): # not a maximum
                continue
            if (derivs[ix+1]>0.0): # do not change a sign
                continue
            else:
                root =brentq(f_deriv,xs[ix],xs[ix+1])
                roots.append(root)
    return roots

def global_optimize_mpi(f, f_deriv, x_min, x_max, omega_max, opt_flag):
    delta_x = np.pi/(2.1*omega_max) # delta_x<np.pi/(2.0*omega_max)
    
    x_range_core = (x_max-x_min)/cores
    x_min_core = x_min + x_range_core * core
    x_max_core = x_min + x_range_core * (core+1)

    #print(x_min_core,x_max_core)
#    print('min, max:', x_min, x_max)
#    print('delta_x:', delta_x)
#
#    # test
#    nx = round((x_max-x_min)/delta_x) + 1
#    xs = np.linspace(x_min,x_max,nx)
#    for ix in range(nx):
#        x = xs[ix]
#        print(x, f(x), f_deriv(x))
#    # test

    nx = max(round((x_max_core-x_min_core)/delta_x) + 1,2)
    roots = []
    xs = np.linspace(x_min_core,x_max_core,nx)
    derivs = np.zeros((nx), dtype=float)
    for ix in range(nx):
        x = xs[ix]
        derivs[ix] = f_deriv(x)
        #print('# ', x, derivs[ix])
    
    if (opt_flag==0): # minimize
        for ix in range(nx-1):
            # check signs
            if (derivs[ix]>0.0): # not a minimum
                continue
            if (derivs[ix+1]<0.0): # do not change a sign
                continue
            else:
                root =brentq(f_deriv,xs[ix],xs[ix+1])
                roots.append(root)
    else: # maximize
        for ix in range(nx-1):
            # check signs
            if (derivs[ix]<0.0): # not a maximum
                continue
            if (derivs[ix+1]>0.0): # do not change a sign
                continue
            else:
                root =brentq(f_deriv,xs[ix],xs[ix+1])
                roots.append(root)
    #print(xs, derivs)
    candidates, f_at_candidates = [], []
    #print(roots)
    for root in roots:
        f_val = f(root)
        candidates.append(root)
        f_at_candidates.append(f_val)
    if (len(candidates)>0):
        if (opt_flag==0): # minimize
            idx = int(np.argmin(f_at_candidates))
        else: # maximize
            idx = int(np.argmax(f_at_candidates))
        root_core = candidates[idx]
        f_val_core = f_at_candidates[idx]
    else:
        root_core = None
        f_val_core = None

    comm.Barrier()
    roots = comm.gather(root_core, root=0)
    f_vals = comm.gather(f_val_core, root=0)
    #print(roots, f_vals)

    if core == 0:
        valid = [(r, v) for r, v in zip(roots, f_vals) if r is not None]
        if valid:
            if opt_flag == 0:
                global_root, global_val = min(valid, key=lambda rv: rv[1])
            else:
                global_root, global_val = max(valid, key=lambda rv: rv[1])
        else:
            global_root, global_val = None, None
    else:
        global_root, global_val = None, None
    
    # Broadcast the result to all processes
    global_root = comm.bcast(global_root, root=0)
    global_val  = comm.bcast(global_val,  root=0)
    
    # Return value
    return global_root, global_val

def isolate_interval(f, a, b, fa, h=1.0/1024, r=2, tol=5e-3):
    """
    isloate interval using geometric expansion
    """
    xs = [a]
    fs = [fa]

    i = 0
    last = False
    while not last:
        i += 1
        x = a + h * 2**(i-1)
        if (x>=b):
            last = True
            x = b

        fx = f(x)
        print(x, fx)

        xs.append(x)
        fs.append(fx)

        if (fa*fs[i]<=0):
            return xs[i-1], xs[i], fs[i-1], fs[i]
        if (abs(fs[i])<tol): # root is found
            return xs[i-1], xs[i], fs[i-1], fs[i]
    
    m = i+1
    i_list = range(m)

    # not found interval yet. Do a adaptive refinement using a cubic spline

    spline = PchipInterpolator(i_list, fs)

    approx_roots = spline.roots()
    approx_roots = approx_roots[(approx_roots >= 0) & (approx_roots <= m-1)]

    # 4. Bracket around first approximate root
    if approx_roots.size > 0:
        i0 = approx_roots[0]
        idx = np.searchsorted(i_list, i0)
        i = min(max(idx,0),m-1)
        print('found from interpolation')
        return xs[i-1], xs[i], fs[i-1], fs[i]
    else:
        return None

def RootFinding (f, a, b, fa=None, fb=None, tol=5e-3, itermax=100):
    if (fa==None):
        fa = f(a)
    if (fb==None):
        fb = f(b)
    if (np.abs(fa)<tol):
        return a
    if (np.abs(fb)<tol):
        return b
    x = 0.5 * (a + b)
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
        x_max = b



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

ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs(material_home + '/FCIDUMP_0')
ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs(material_home + '/DFT/output/FCIDUMP_FULL')
info_i = (ecore_i, h1e_i, g2e_i)
info_f = (ecore_f, h1e_f, g2e_f)

fcivec_00      = np.load(material_home + "/fcivec_0.npy")[0]
fcivec_exact_0 = np.load(material_home + "/fcivec_FULL.npy")[0] 

# find fcivec_exact internally to stability
ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, 0.0)
mf = scf.RHF(mol)
mci = fci.direct_spin1.FCI(mol)
mci.spin = 0 
mci.conv_tol = 1e-10
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=0.5, ss=0.0)    
E0, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000, ci0=fcivec_00)

fcivec_0 = copy.copy(ci)


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


phi_exact = CIObject(fcivec_exact, norb, nelec)

import ctypes
from pyscf import lib
from pyscf.fci.direct_spin1 import _unpack, absorb_h1e, FCIvector
from pyscf.fci.addons import overlap

# Load the optimized C library for FCI
libfci = lib.load_library('libfci')

def arnoldi(matvec, v, m, reorth=True, tol=1e-12):
    n = v.shape[0]
    Q = np.zeros((n, m+1), dtype=v.dtype)
    Hh = np.zeros((m+1, m), dtype=v.dtype)
    krylov_beta = np.linalg.norm(v)
    Q[:,0] = v / krylov_beta

    for j in range(m):
        w = matvec(Q[:,j])
        # Modified Gram–Schmidt
        for i in range(j+1):
            Hh[i,j] = np.vdot(Q[:,i], w)
            w -= Hh[i,j] * Q[:,i]
        # Re-orthogonalization (optional)
        if reorth:
            for i in range(j+1):
                corr = np.vdot(Q[:,i], w)
                Hh[i,j] += corr
                w -= corr * Q[:,i]
        Hh[j+1,j] = np.linalg.norm(w)
        if Hh[j+1,j] < tol:
            return Q, Hh, krylov_beta
        Q[:,j+1] = w / Hh[j+1,j]

    return Q, Hh, krylov_beta

def adiabatic_evolve_krylov_with_zeno (t1, dt, phi, info_i, info_f, beta, gamma=0.04, tol_norm=0.005, dl_small_ratio = 1e-6):

    r_norm2 = 1.0 - gamma

    t_list = [0]
    s_list = [0]
    ej     = E0
    e1     = 0.0


    norb, nelec = phi.norb, phi.nelec
    # compute hamiltonian difference
    ecore_diff, h1e_diff, g2e_diff = compute_hamiltonian_diff (info_i, info_f)
    h2e_diff = absorb_h1e(h1e_diff, g2e_diff, norb, nelec, 0.5)
    h2e_diff_ptr = h2e_diff.ctypes.data_as(ctypes.c_void_p)
    # 
    # Compute the link indices once
    link_a, link_b = _unpack(norb, nelec, None)
    na, nlinka = link_a.shape[:2]
    nb, nlinkb = link_b.shape[:2]
    li_ptr = link_a.ctypes.data_as(ctypes.c_void_p)
    lb_ptr = link_b.ctypes.data_as(ctypes.c_void_p)

    # Pre-allocate buffers
    ci_norb    = ctypes.c_int(norb)
    ci_na      = ctypes.c_int(na)
    ci_nb      = ctypes.c_int(nb)
    ci_nlinka  = ctypes.c_int(nlinka)
    ci_nlinkb  = ctypes.c_int(nlinkb)
    li_ptr     = link_a.ctypes.data_as(ctypes.c_void_p)
    lb_ptr     = link_b.ctypes.data_as(ctypes.c_void_p)

    # 3) Pre-compute the out_buf pointer once
    out_buf = np.empty((na*nb), dtype=np.float64).view(FCIvector)
    # Pre-compute the pointers as well
    out_ptr = out_buf.ctypes.data_as(ctypes.c_void_p)

    def hop(vec, h2e_ptr, out_buf):
        # Call the C routine (two-electron contraction)
        #start = time.perf_counter()
        libfci.FCIcontract_2e_spin1(
            h2e_ptr,
            vec.ctypes.data_as(ctypes.c_void_p),
            out_buf.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(na), ctypes.c_int(nb),
            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
            li_ptr, lb_ptr
        )
        #elapsed = time.perf_counter() - start
        #print('hop_:', elapsed)

    def matvec_general(x_flat, h2e_ptr, out_buf):
        # x_flat: 1-D complex128 of length N
        # 2) copy data into the 2-D buffers (no new allocation!)
        xr_buf[:, :] = x_flat.real.reshape((na, nb))
        xi_buf[:, :] = x_flat.imag.reshape((na, nb))
        # 3) call your C-kernel
        hop(xr_buf, h2e_ptr, out_buf)
        real_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
        hop(xi_buf, h2e_ptr, out_buf)
        # 4) combine and flatten
        imag_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
        y_buf.real = real_buf
        y_buf.imag = imag_buf

        return y_buf

    N = na*nb

    xr_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
    xi_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
    real_buf = np.empty(N, dtype=np.float64)
    imag_buf = np.empty(N, dtype=np.float64)
    y_buf  = np.empty((N), dtype=np.complex128)

    # before run the evolution, pre-compute 1 norm estimate
    # norm0 = NormEstimation (norb, nelec, 0.0)
    # norm1 = NormEstimation (norb, nelec, 1.0)
    # print('norms:', norm0, norm1)

    # make room to save krylov variables
    H_diff_krylov  = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
    Hh_krylov      = np.zeros((m_krylov+1,m_krylov), dtype=np.complex128)
    H_krylov       = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
    Q_krylov       = np.zeros((N, m_krylov+1), dtype=np.complex128)
    phi_krylov     = np.zeros((m_krylov), dtype=np.complex128)

    Hj_krylov      = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
    H1_krylov      = np.zeros((m_krylov,m_krylov), dtype=np.complex128)

    phi_cplx = np.asarray(phi.r) + 1j * np.asarray(phi.i)

    phi_flat = np.asarray(phi_cplx).reshape(-1)
    fcivec_flat = np.asarray(fcivec_exact).reshape(-1)

    def compute_fj_krylov (alpha, s, phi_asp):
        # construct krylov hamiltonian
        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s)
        h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
        h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
        matvec = lambda x: matvec_general(x, h2e_ptr, out_buf)
        Q_krylov[:], Hh_krylov[:], _ = arnoldi(matvec, phi_asp, m_krylov)

        phi_krylov[:] = Q_krylov[:,:m_krylov].conj().T@phi_asp

        return compute_fj(alpha, Hh_krylov[:m_krylov,:m_krylov], phi_krylov)

        

    def compute_pj_krylov (alpha, s, s_new, ej, phi_asp):
        # construct krylov hamiltonian
        s_mid = (s+s_new)/2.0

        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_mid)
        h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
        h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
        matvec = lambda x: matvec_general(x, h2e_ptr, out_buf)
        Q_krylov[:], Hh_krylov[:], _ = arnoldi(matvec, phi_asp, m_krylov)

        phi_krylov[:] = Q_krylov[:,:m_krylov].conj().T@phi_asp

        # H diff construction
        matvec = lambda x: matvec_general(x, h2e_diff_ptr, out_buf)
        for j in range(m_krylov):
            y = matvec(Q_krylov[:,j])
            H_diff_krylov[:,j] = Q_krylov[:,:m_krylov].conj().T@ y 
        #H_diff_krylov[:] = np.conj(H_diff_krylov)


        # Hj construction
        s_relative = s - s_mid
        Hj_krylov[:] = Hh_krylov[:m_krylov,:m_krylov] +  s_relative * H_diff_krylov

        # H1 construction
        s_relative = s_new - s_mid
        H1_krylov[:] = Hh_krylov[:m_krylov,:m_krylov] +  s_relative * H_diff_krylov

        optimal_point, optimal_value = compute_pj(alpha, Hj_krylov, H1_krylov, ej, phi_krylov)

        return optimal_point, optimal_value, s_mid

    def compute_pj_fj_ratio_krylov (alpha, s, s_new, phi_asp):
        # construct krylov hamiltonian
        s_mid = (s+s_new)/2.0

        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_mid)
        h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
        h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
        matvec = lambda x: matvec_general(x, h2e_ptr, out_buf)
        Q_krylov[:], Hh_krylov[:], _ = arnoldi(matvec, phi_asp, m_krylov)

        phi_krylov[:] = Q_krylov[:,:m_krylov].conj().T@phi_asp

        # H diff construction
        matvec = lambda x: matvec_general(x, h2e_diff_ptr, out_buf)
        for j in range(m_krylov):
            y = matvec(Q_krylov[:,j])
            H_diff_krylov[:,j] = Q_krylov[:,:m_krylov].conj().T@ y 
        #H_diff_krylov[:] = np.conj(H_diff_krylov)


        # Hj construction
        s_relative = s - s_mid
        Hj_krylov[:] = Hh_krylov[:m_krylov,:m_krylov] +  s_relative * H_diff_krylov


        # compute fj
        ej, fj = compute_fj(alpha, Hj_krylov, phi_krylov)

        # H1 construction
        s_relative = s_new - s_mid
        H1_krylov[:] = Hh_krylov[:m_krylov,:m_krylov] +  s_relative * H_diff_krylov

        e1, pj = compute_pj(alpha, Hj_krylov, H1_krylov, ej, phi_krylov)

        return pj/fj, pj, s_mid



    def Apply_AdiabaticEvolution (schedule, t_1, t_2):
        time = t_2-t_1
        n_steps = max(round(time/dt),1)
        dt_step = time/n_steps
        t = t_1

        #print('s: ',s_list[-2])
        #print('s_mid: ',s_mid)
        #print('s_next: ',s_list[-1])

        phi_save = copy.copy(phi_krylov)

        for step in range(n_steps):
            # Hamiltonian interpolate
            t_mid = t + 0.5 * dt_step # if step<n_steps else t_2
            s_value = schedule(t_mid)
            s_relative = s_value - s_mid
            H_krylov[:] = -1j*dt_step*(Hh_krylov[:m_krylov,:m_krylov] + s_relative * H_diff_krylov)
            phi_krylov_next = expm_multiply(H_krylov, phi_krylov)
            phi_krylov[:] = phi_krylov_next
            norm = np.linalg.norm(phi_krylov)
            phi_krylov[:] /= norm

            t += dt_step

        # return to the original basis
        phi_flat[:] = Q_krylov[:,:m_krylov]@phi_krylov

        # fcivec to check
        fcivec_krylov = Q_krylov[:,:m_krylov].conj().T@fcivec_flat

        print('# ASP done: ',np.abs(np.vdot(phi_save,phi_krylov))**2)
        print('# Fidelity: ',np.abs(np.vdot(fcivec_krylov,phi_krylov))**2)

    alpha = 0
    s = s_list[0]
    dl_list = []
    fj = 1.0

    # recomputation of E0
    # not a problem
    # ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, 0)
    # h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
    # h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
    # y = matvec_general(phi_flat, h2e_ptr, out_buf)
    # E0_new = phi_flat.conj().T@y
    # print('# recompute ', eigen_energies_qzmc[0], E0_new)
    # eigen_energies_qzmc[0] = E0_new.real

    # test

    # s_next = 0.375

    # pj_fj, pj, s_mid = compute_pj_fj_ratio_krylov (alpha, s, s_next, phi_flat)

    # print('pj/fj : {rat}'.format(rat=pj_fj))
    # print('pj: {pj}'.format(pj=pj))

    # if (pj_fj>1.0): # statistical error
    #     dl = dl_small_ratio * dl_list[0]
    # else:
    #     dl = np.sqrt(1.0-pj_fj)


    # if (alpha==0):
    #     ratio = 1.0
    # else:
    #     ratio =  dl/dl_list[0]

    # s_list.append(s_next)
    # dl_list.append(dl)
    # t_list.append(t_list[-1] + ratio * t1)
    # optimal_schedule = PchipInterpolator(t_list, s_list)

    # print('# zeno', s_list[-2], s_list[-1], pj)
    # print('# tf: ', t_list[-2], t_list[-1])

    # # prepare Krylov 

    # # adiabatic evolution
    # Apply_AdiabaticEvolution(optimal_schedule, t_list[-2], t_list[-1])
    # alpha += 1
    # s = s_list[-1]

    # ej, fj = compute_fj_krylov(alpha, s, phi_flat)

    # s_next = 0.75

    # pj_fj, pj, s_mid = compute_pj_fj_ratio_krylov (alpha, s, s_next, phi_flat)

    # print('pj/fj : {rat}'.format(rat=pj_fj))
    # print('pj: {pj}'.format(pj=pj))

    # print('test end')

    # test

    while s<1.0:
        print('fj:', fj)
        func = lambda x: compute_pj_fj_ratio_krylov (alpha, s, x, phi_flat)[0] - r_norm2

        fs = 1-r_norm2

        # isolate interval first
        result = isolate_interval(func, s, 1.0, fs, h=1.0/1024, r=2.0, tol=tol_norm)
        if (result==None):
            s_next = 1.0
        else:
            print(result)
            s1, s2, fs1, fs2 = result
            s_next = RootFinding(func, s1, s2, fa=fs1, fb=fs2, tol=tol_norm)
            s_next = comm.bcast(s_next, root=0)

        #pj, ci0 = compute_pj_with_exact_overlap(s_next, phi, cis)
        pj_fj, pj, s_mid = compute_pj_fj_ratio_krylov (alpha, s, s_next, phi_flat)

        print('pj/fj : {rat}'.format(rat=pj_fj))
        print('pj: {pj}'.format(pj=pj))

        if (pj_fj>1.0): # statistical error
            dl = dl_small_ratio * dl_list[0]
        else:
            dl = np.sqrt(1.0-pj_fj)


        if (alpha==0):
            ratio = 1.0
        else:
            ratio =  dl/dl_list[0]

        s_list.append(s_next)
        dl_list.append(dl)
        t_list.append(t_list[-1] + ratio * t1)
        optimal_schedule = PchipInterpolator(t_list, s_list)

        print('# zeno', s_list[-2], s_list[-1], pj, pj_fj)
        print('# tf: ', t_list[-2], t_list[-1])

        # adiabatic evolution
        Apply_AdiabaticEvolution(optimal_schedule, t_list[-2], t_list[-1])
        alpha += 1
        s = s_list[-1]

        ej, fj = compute_fj_krylov(alpha, s, phi_flat)

    return (t_list, s_list)


def compute_fj_from_amplitudes(sampled_times, amplitudes, eps):
    fj = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        phase = eps * times[0]
        fj += amplitudes[imc] * np.exp(1j*phase)
    fj /= nmc
    fj = fj.real
    return fj

def compute_fj_deriv_from_amplitudes(sampled_times, amplitudes, eps):
    fj_deriv = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        phase = eps * times[0]
        factor = 1j * times[0]
        fj_deriv += amplitudes[imc] * np.exp(1j*phase) * factor
    fj_deriv /= nmc
    fj_deriv = fj_deriv.real
    return fj_deriv

def compute_pj_from_amplitudes(alpha, sampled_times, amplitudes, ej, eps):
    pj = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        phase = 0.0

        if (alpha>0):
            phase += ej * times[0]

        phase += eps * times[1]

        if (alpha>0):
            phase += ej * times[2]

        pj += amplitudes[imc] * np.exp(1j*phase)
    pj /= nmc
    pj = pj.real
    return pj

def compute_pj_deriv_from_amplitudes(alpha, sampled_times, amplitudes, ej, eps):
    pj_deriv = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        phase = 0.0
        factor= 0.0

        if (alpha>0):
            phase += ej * times[0]

        phase += eps * times[1]
        factor += times[1]
        factor *= 1j

        if (alpha>0):
            phase += ej * times[2]

        pj_deriv += amplitudes[imc] * np.exp(1j*phase) * factor
    pj_deriv /= nmc
    pj_deriv = pj_deriv.real
    return pj_deriv

def compute_fj (alpha, hj, phi_asp):
    if (alpha==0):
        return E0, 1.0
    eigen_ej, eigen_vj = np.linalg.eigh(hj)
    sqrt2 = np.sqrt(2)
    sigma_gauss = sqrt2 * beta

    if (core==0):
        sampled_times = []
        omega_max = 0.0

        for imc in range(nmc):
            times = np.random.normal(0.0, sigma_gauss, size=1)
            sampled_times.append(times)
            omega_max = max(omega_max,np.abs(times[0]))
    else:
        sampled_times = None
        omega_max = None

    sampled_times = comm.bcast(sampled_times, root=0)
    omega_max     = comm.bcast(omega_max, root=0)

    # precompute
    vec = eigen_vj.conj().T@phi_asp

    # computation of fj
    amplitudes = []
    for imc in range(nmc):
        times = sampled_times[imc]

        expi_1 = np.exp(-1j*times[0]*eigen_ej)

        w = expi_1*vec

        z = np.vdot(vec, w)

        amplitudes.append(z)


    f = lambda x: compute_fj_from_amplitudes(sampled_times, amplitudes, x)
    df = lambda x: compute_fj_deriv_from_amplitudes(sampled_times, amplitudes, x)

    x_min = eigen_ej[0]-1.0/(sqrt2*beta)-np.pi/(2.1*omega_max)
    x_max = eigen_ej[1]+1.0/(sqrt2*beta)+np.pi/(2.1*omega_max)

    ej, fj = global_optimize_mpi(f,df,x_min,x_max,omega_max,1)

    print('fj opt_result: ', ej, fj)

    return ej, fj

def compute_pj (alpha, hj, h1, ej, phi_asp):
    eigen_ej, eigen_vj = np.linalg.eigh(hj)
    eigen_e1, eigen_v1 = np.linalg.eigh(h1)

    #print('amplitude calc start')
    sqrt2 = np.sqrt(2)

    if (core==0):
        sampled_times = []
        omega_max = 0.0
        for imc in range(nmc):
            times = np.random.normal(0.0, beta, size=3)
            times[1] *= sqrt2
            sampled_times.append(times)
            omega_max = max(omega_max,np.abs(times[1]))
    else:
        sampled_times = None
        omega_max = None

    sampled_times = comm.bcast(sampled_times, root=0)
    omega_max     = comm.bcast(omega_max, root=0)

    # precompute

    if (alpha>0):
        vec = eigen_vj.conj().T@phi_asp
        Trans = eigen_v1.conj().T@eigen_vj

        # computation of pj
        amplitudes = []
        for imc in range(nmc):
            times = sampled_times[imc]

            # compute expi_1, expi_2, expi_3
            expi_1 = np.exp(-1j*times[0]*eigen_ej)
            expi_2 = np.exp(-1j*times[1]*eigen_e1)
            expi_3 = np.exp(-1j*times[2]*eigen_ej)

            w = expi_1*vec
            w = Trans@w
            w = expi_2*w
            w = Trans.conj().T@w
            w = expi_3*w

            z = np.vdot(vec, w)

            amplitudes.append(z)
    else:
        vec = eigen_v1.conj().T@phi_asp

        # computation of pj
        amplitudes = []
        for imc in range(nmc):
            times = sampled_times[imc]

            # compute expi_1, expi_2, expi_3
            expi_2 = np.exp(-1j*times[1]*eigen_e1)
            w = expi_2*vec
            z = np.vdot(vec, w)

            amplitudes.append(z)

    #print('omega_max', omega_max)

    f = lambda x: compute_pj_from_amplitudes(alpha, sampled_times, amplitudes, ej, x)
    df = lambda x: compute_pj_deriv_from_amplitudes(alpha, sampled_times, amplitudes, ej, x)

    # test
    print('# test')
    print('# ovsj:', np.abs(eigen_vj[:,0].conj().T@phi_asp)**2)
    print('# ovs1:', np.abs(eigen_v1[:,0].conj().T@phi_asp)**2)
    # compute expi_1, expi_2, expi_3
    if (alpha>0):
        expi_1 = np.exp(-0.5 * beta**2 * (eigen_ej-eigen_ej[0])**2)
        expi_2 = np.exp(-beta**2 * (eigen_e1-eigen_e1[0])**2)
        expi_3 = np.exp(-0.5 * beta**2 * (eigen_ej-eigen_ej[0])**2)

        w = expi_1*vec
        w = Trans@w
        w = expi_2*w
        w = Trans.conj().T@w
        w = expi_3*w

        z = np.vdot(vec, w)
    else:
        vec = eigen_v1.conj().T@phi_asp
        expi_2 = np.exp(-beta**2 * (eigen_e1-eigen_e1[0])**2)
        w = expi_2*vec

        z = np.vdot(vec, w)
    #w = Trans.conj().T@w
    #z = np.abs(vec.conj().T@w)
    print('# ovs_est:', np.abs(z))
    #x_lin = np.linspace(-60,-20,101)
    #for x in x_lin:
    #    print(x, f(x))
    #print('# test end')
    # test

    #x_min = eigen_e1[0]-0.1
    #x_max = eigen_e1[1]+0.1
    x_min = eigen_e1[0]-1.0/(sqrt2*beta)-np.pi/(2.1*omega_max)
    x_max = eigen_e1[1]+1.0/(sqrt2*beta)+np.pi/(2.1*omega_max)

    # test

    #x_lin = np.linspace(x_min, x_max + 3*(x_max-x_min),101)
    #for x in x_lin:
    #    print(x, f(x))
    #print('# plot test end')

    # test

    #print('s', s_next)
    #print('E', eigen_e)
    #print('ov', np.abs(eigen_v[:,0].conj().T@phi_asp)**2)

    print(eigen_e1[0], f(eigen_e1[0]))


    optimal_point, optimal_value = global_optimize_mpi(f,df,x_min,x_max,omega_max,1)

    print('opt_result: ', optimal_point, optimal_value)
    return optimal_point, optimal_value




#def adiabatic_evolve_krylov(taus, ss, schedule, t_f, dt, phi, fcivec_exact, info_i, info_f, m_krylov):
#
#    norb, nelec = phi.norb, phi.nelec
#    # compute hamiltonian difference
#    ecore_diff, h1e_diff, g2e_diff = compute_hamiltonian_diff (info_i, info_f)
#    h2e_diff = absorb_h1e(h1e_diff, g2e_diff, norb, nelec, 0.5)
#    h2e_diff_ptr = h2e_diff.ctypes.data_as(ctypes.c_void_p)
#    # 
#    # Compute the link indices once
#    link_a, link_b = _unpack(norb, nelec, None)
#    na, nlinka = link_a.shape[:2]
#    nb, nlinkb = link_b.shape[:2]
#    li_ptr = link_a.ctypes.data_as(ctypes.c_void_p)
#    lb_ptr = link_b.ctypes.data_as(ctypes.c_void_p)
#
#
#    # Pre-allocate buffers
#    ci_norb    = ctypes.c_int(norb)
#    ci_na      = ctypes.c_int(na)
#    ci_nb      = ctypes.c_int(nb)
#    ci_nlinka  = ctypes.c_int(nlinka)
#    ci_nlinkb  = ctypes.c_int(nlinkb)
#    li_ptr     = link_a.ctypes.data_as(ctypes.c_void_p)
#    lb_ptr     = link_b.ctypes.data_as(ctypes.c_void_p)
#
#    # 3) Pre-compute the out_buf pointer once
#    out_buf = np.empty((na*nb), dtype=np.float64).view(FCIvector)
#    # Pre-compute the pointers as well
#    out_ptr = out_buf.ctypes.data_as(ctypes.c_void_p)
#
#    def hop(vec, h2e_ptr, out_buf):
#        # Call the C routine (two-electron contraction)
#        #start = time.perf_counter()
#        libfci.FCIcontract_2e_spin1(
#            h2e_ptr,
#            vec.ctypes.data_as(ctypes.c_void_p),
#            out_buf.ctypes.data_as(ctypes.c_void_p),
#            ctypes.c_int(norb),
#            ctypes.c_int(na), ctypes.c_int(nb),
#            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
#            li_ptr, lb_ptr
#        )
#        #elapsed = time.perf_counter() - start
#        #print('hop_:', elapsed)
#
#    # calculation of trace
#    tr0 = TraceCalculation(norb, nelec, 0, 0.0)
#    tr1 = TraceCalculation(norb, nelec, 0, 1.0)
#    print('Traces: ', tr0, tr1)
#
#    # Begin the time loop
#    n_steps = round(t_f/dt) 
#    t = 0.0
#    overlap_val = None
#
#    N = na*nb
#
#    xr_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
#    xi_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
#    real_buf = np.empty(N, dtype=np.float64)
#    imag_buf = np.empty(N, dtype=np.float64)
#    y_buf  = np.empty((N), dtype=np.complex128)
#
#    # before run the evolution, pre-compute 1 norm estimate
#    norm0 = NormEstimation (norb, nelec, 0.0)
#    norm1 = NormEstimation (norb, nelec, 1.0)
#    print('norms:', norm0, norm1)
#
#    # make room to save krylov variables
#    H_diff_krylov  = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
#    Hh_krylov      = np.zeros((m_krylov+1,m_krylov), dtype=np.complex128)
#    H_krylov      = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
#    Q_krylov       = np.zeros((N, m_krylov+1), dtype=np.complex128)
#
#    start_print = time.perf_counter()
#    
#    k_coarse = 0   
#    max_k_coarse = len(ss) -2
#    phi_cplx = phi.r + 1j * phi.i
#    phi_flat = phi_cplx.ravel()
#    fcivec_flat = fcivec_exact.ravel()
#
#    dt_step = dt
#    for step in range(n_steps+1):
#
#        # Hamiltonian interpolate
#        t_mid = t + 0.5 * dt_step if step<n_steps else t_f
#        s_value = schedule(t_mid/t_f)
#        if (s_value>ss[k_coarse]):
#            # 
#            if (k_coarse>0): # not the first time
#                phi_flat = Q_krylov[:,:m_krylov]@phi_krylov
#            # skip until ss[k_coarse]<s_value<ss[k_coarse+1]
#            while k_coarse < max_k_coarse and s_value > ss[k_coarse+1]:
#                k_coarse +=1
#            # preprocess for each interval
#            s_mid = (ss[k_coarse] + ss[k_coarse+1])/2.0
#            print('# interval', ss[k_coarse], s_value, ss[k_coarse+1], k_coarse)
#            ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_mid)
#            h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
#            h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
#
#
#            
#            def matvec(x_flat):
#                # x_flat: 1-D complex128 of length N
#                # 2) copy data into the 2-D buffers (no new allocation!)
#                xr_buf[:, :] = x_flat.real.reshape((na, nb))
#                xi_buf[:, :] = x_flat.imag.reshape((na, nb))
#                # 3) call your C-kernel
#                hop(xr_buf, h2e_ptr, out_buf)
#                real_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
#                hop(xi_buf, h2e_ptr, out_buf)
#                # 4) combine and flatten
#                imag_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
#                y_buf.real = real_buf
#                y_buf.imag = imag_buf
#
#                return y_buf
#
#            Q_krylov, Hh_krylov, beta_krylov = arnoldi(matvec, phi_flat, m_krylov)
#
#            # Krylov projection of the Hamiltonian difference
#            
#            def matvec2(x_flat):
#                # x_flat: 1-D complex128 of length N
#                # 2) copy data into the 2-D buffers (no new allocation!)
#                xr_buf[:, :] = x_flat.real.reshape((na, nb))
#                xi_buf[:, :] = x_flat.imag.reshape((na, nb))
#                # 3) call your C-kernel
#                hop(xr_buf, h2e_diff_ptr, out_buf)
#                real_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
#                hop(xi_buf, h2e_diff_ptr, out_buf)
#                # 4) combine and flatten
#                imag_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
#                y_buf.real = real_buf
#                y_buf.imag = imag_buf
#
#                return y_buf
#
#            # print('# krylov s',s_mid)
#            # print("# Q shape:", Q_krylov.shape)     
#            # print("# Hessenberg shape:", Hh_krylov.shape)  
#            # print("# Initial norm beta:", beta_krylov)
#
#            # debug (no problem)
#            # H_tmp  = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
#            # for j in range(m_krylov):
#            #     y = matvec(Q_krylov[:,j])
#            #     H_tmp[:,j] = y.conj() @ Q_krylov[:,:m_krylov]
#            # H_tmp = np.conj(H_tmp)
#            # # debug
#            # for jj in range(m_krylov):
#            #     for ii in range(m_krylov):
#            #         v_debug = np.abs(H_tmp[ii,jj]-Hh_krylov[ii,jj])
#            #         if (v_debug>1e-10):
#            #             print (ii,jj, v_debug)
#            # print('debug end')
#
#            for j in range(m_krylov):
#                y = matvec2(Q_krylov[:,j])
#                H_diff_krylov[:,j] = y.conj() @ Q_krylov[:,:m_krylov]
#            H_diff_krylov = np.conj(H_diff_krylov)
#
#            # debug
#            # for jj in range(m_krylov):
#            #     for ii in range(m_krylov):
#            #         v_debug = np.abs(H_diff_krylov[ii,jj])
#            #         if (v_debug>1e-10):
#            #             print (ii,jj, H_diff_krylov[ii,jj])
#            # print('debug end')
#
#            phi_krylov = phi_flat.conj()@Q_krylov[:,:m_krylov]
#            phi_krylov = phi_krylov.conj()
#
#            # debug
#            #print('debug start')
#            #for ii in range(m_krylov):
#            #    v_debug = np.abs(phi_krylov[ii])
#            #    if (v_debug>1e-10):
#            #        print(ii, v_debug)
#            #print('debug end')
#            # 
#
#            # fcivec to check
#            fcivec_krylov = fcivec_flat.conj()@Q_krylov[:,:m_krylov]
#            fcivec_krylov = fcivec_krylov.conj()
#
#            k_coarse += 1
#
#        s_relative = s_value - s_mid
#        #print(s_relative)
#
#        H_krylov = -1j*dt_step*(Hh_krylov[:m_krylov,:m_krylov] + s_relative * H_diff_krylov)
#        phi_krylov_next = expm_multiply(H_krylov, phi_krylov)
#        # dont' do phi_krylov = .. -> it does not work
#        # debug
#        # overlap2 = np.abs(np.vdot(phi_krylov_next,phi_krylov))**2
#        # print ('# debug, ov = ', overlap2)
#
#        phi_krylov = phi_krylov_next
#
#        
#        norm = np.linalg.norm(phi_krylov)
#        #print(norm)
#        #for ii in range(na):
#        #    for jj in range(nb):
#        #        if (np.abs(phi.r[ii,jj])>1e-10):
#        #            print(ii, jj, phi.r[ii,jj])
#        phi_krylov /= norm
#
#        # Overlap calculation
#
#        overlap2 = np.abs(np.vdot(fcivec_krylov,phi_krylov))**2
#
#        elapsed = time.perf_counter() - start_print
#        if (elapsed>10):
#            print(t, schedule(t/t_f), overlap2)
#            start_print = time.perf_counter()
#
#        t += dt_step
#
#    return overlap2



#def adiabatic_evolve(schedule, t_f, dt, phi, fcivec_exact, info_i, info_f):
#
#    norb, nelec = phi.norb, phi.nelec
#    # Compute the link indices once
#    link_a, link_b = _unpack(norb, nelec, None)
#    na, nlinka = link_a.shape[:2]
#    nb, nlinkb = link_b.shape[:2]
#    li_ptr = link_a.ctypes.data_as(ctypes.c_void_p)
#    lb_ptr = link_b.ctypes.data_as(ctypes.c_void_p)
#
#
#    # Pre-allocate buffers
#    ci_norb    = ctypes.c_int(norb)
#    ci_na      = ctypes.c_int(na)
#    ci_nb      = ctypes.c_int(nb)
#    ci_nlinka  = ctypes.c_int(nlinka)
#    ci_nlinkb  = ctypes.c_int(nlinkb)
#    li_ptr     = link_a.ctypes.data_as(ctypes.c_void_p)
#    lb_ptr     = link_b.ctypes.data_as(ctypes.c_void_p)
#
#    # 3) Pre-compute the out_buf pointer once
#    out_buf = np.empty((na*nb), dtype=np.float64).view(FCIvector)
#    # Pre-compute the pointers as well
#    out_ptr = out_buf.ctypes.data_as(ctypes.c_void_p)
#
#    def hop(vec, h2e_ptr, out_buf):
#        # Call the C routine (two-electron contraction)
#        #start = time.perf_counter()
#        libfci.FCIcontract_2e_spin1(
#            h2e_ptr,
#            vec.ctypes.data_as(ctypes.c_void_p),
#            out_buf.ctypes.data_as(ctypes.c_void_p),
#            ctypes.c_int(norb),
#            ctypes.c_int(na), ctypes.c_int(nb),
#            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
#            li_ptr, lb_ptr
#        )
#        #elapsed = time.perf_counter() - start
#        #print('hop_:', elapsed)
#
#    # calculation of trace
#    tr0 = TraceCalculation(norb, nelec, 0, 0.0)
#    tr1 = TraceCalculation(norb, nelec, 0, 1.0)
#    print('Traces: ', tr0, tr1)
#
#    # Begin the time loop
#    n_steps = round(t_f/dt) 
#    t = 0.0
#    overlap_val = None
#
#    N = na*nb
#
#    xr_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
#    xi_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
#    real_buf = np.empty(N, dtype=np.float64)
#    imag_buf = np.empty(N, dtype=np.float64)
#    y_buf  = np.empty((N), dtype=np.complex128)
#
#    # before run the evolution, pre-compute 1 norm estimate
#    norm0 = NormEstimation (norb, nelec, 0.0)
#    norm1 = NormEstimation (norb, nelec, 1.0)
#    print('norms:', norm0, norm1)
#
#    for step in range(n_steps+1):
#        dt_step = dt
#        # Hamiltonian interpolate
#        t_mid = t + 0.5 * dt_step if step<n_steps else t_f
#        frac = schedule(t_mid/t_f)
#        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, frac)
#        h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
#        h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
#
#        phi_cplx = phi.r + 1j*phi.i           # shape (na, nb), dtype=complex128
#        phi_flat = phi_cplx.ravel()
#
#        
#        def matvec(x_flat):
#            # x_flat: 1-D complex128 of length N
#            # 2) copy data into the 2-D buffers (no new allocation!)
#            xr_buf[:, :] = x_flat.real.reshape((na, nb))
#            xi_buf[:, :] = x_flat.imag.reshape((na, nb))
#            # 3) call your C-kernel
#            hop(xr_buf, h2e_ptr, out_buf)
#            real_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
#            hop(xi_buf, h2e_ptr, out_buf)
#            # 4) combine and flatten
#            imag_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
#            y_buf.real = imag_buf
#            y_buf.imag = -real_buf
#
#            return y_buf * dt_step
#
#
#        def rmatvec(x):
#            # H is Hermitian ⇒ Hᴴ x = conj( H ( conj(x) ) )
#            #print('rmatvec work')
#            return (-1j* np.conj(matvec(np.conj(x))))
#
#        H_op = LinearOperator((N,N), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)
#        #
#        trtr = -1j * dt_step * (tr0 * (1-frac) + tr1 * frac)
#        normA = dt_step * (norm0 * (1-frac) + norm1 * frac)
#        #print('normA(in)', normA)
#        phi_next_flat = expm_multiply(H_op, phi_flat)
#        
#        # 4) reshape back and view as FCIvector
#        phi_next_mat = phi_next_flat.reshape((na, nb))#.view(FCIvector)
#
#        phi.r = phi_next_mat.real
#        phi.i = phi_next_mat.imag
#
#        norm = np.linalg.norm(phi.r + 1j*phi.i)
#        #print(norm)
#        #for ii in range(na):
#        #    for jj in range(nb):
#        #        if (np.abs(phi.r[ii,jj])>1e-10):
#        #            print(ii, jj, phi.r[ii,jj])
#        phi.r /= norm;  phi.i /= norm
#
#        
#
#        # Overlap calculation
#        overlap2 = compute_weight(phi, fcivec_exact)
#
#        print(t, schedule(t/t_f), overlap2)
#
#        t += dt_step
#
#    return overlap2


gap_ref = 3.3412792970821670e-03 # at R=3.0A
t1_ref  = 105 # at R=3.0A
gap = 0.000826475819970085

gamma = 0.04
tol_norm = 0.005
t1 = t1_ref * gap_ref/gap
dt = 0.5
if (t1<dt):
    t1 = dt
beta = 2.0/gap
nmc = 10000

print('t1: ',t1)
print('beta: ',beta)

m_krylov = 100

print('# start: finding optimal schedule')
start = time.perf_counter()
phi = CIObject(fcivec_0, norb, nelec)
t_list, s_list = adiabatic_evolve_krylov_with_zeno(t1, dt, phi, info_i, info_f, beta, gamma, tol_norm)
elapsed = time.perf_counter() - start
print('# done: {t_f} with {elapsed} s'.format(t_f=t_list[-1], elapsed=elapsed))


optimal_schedule = PchipInterpolator(t_list, s_list)

# write data
if (core==0):
    with open('optimal_schedule','w') as file_:
        for i in range(len(s_list)):
            s = '{:.16e}    {:.16e}'.format(t_list[i]/t_list[-1], s_list[i])
            print(s)
            s += '\n'
            file_.write(s)

if (core==0):
    t_list_ = np.linspace(t_list[0],t_list[-1],num=101)
    s_list_ = optimal_schedule(t_list_)
    with open('optimal_schedule_interpol','w') as file_:
        for i in range(len(s_list_)):
            s = '{:.16e}    {:.16e}'.format(t_list_[i]/t_list_[-1], s_list_[i])
            #print(s)
            s += '\n'
            file_.write(s)
