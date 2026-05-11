import numpy as np
from math import comb
from mpi4py import MPI
import time
import copy
from pyscf import gto,scf,fci,lib
from scipy.sparse.linalg import LinearOperator, onenormest, expm_multiply
from scipy.interpolate import PchipInterpolator
from pyscf.fci.addons import overlap
from pyscf.fci import direct_spin1, cistring
from warnings import warn

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()

einsum = lib.einsum

nroots = 1 # we only find a ground state



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


material_home = '../../../'

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
mci = fci.addons.fix_spin_(mci, shift=3.0, ss=0.0)    
E0, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=nroots, max_space=30, max_cycle=2000, ci0=fcivec_00)

fcivec_0 = copy.copy(ci)


# find fcivec_exact internally to stability
ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, 1.0)
mf = scf.RHF(mol)
mci = fci.direct_spin1.FCI(mol)
mci.spin = 0 
mci.conv_tol = 1e-10
mci.max_cycle = 2000
mci = fci.addons.fix_spin_(mci, shift=3.0, ss=0.0)    
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


def linear_schedule (tau):
    return tau

def adiabatic_evolve_krylov(taus, ss, t_f, dt, phi, fcivec_exact, info_i, info_f, m_krylov):


    schedule = PchipInterpolator(taus, ss)
    #schedule = linear_schedule

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

    # Begin the time loop
    n_steps = max(round(t_f/dt),1)
    dt_step = t_f/n_steps
    t = 0.0
    overlap_val = None

    N = na*nb

    xr_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
    xi_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
    real_buf = np.empty(N, dtype=np.float64)
    imag_buf = np.empty(N, dtype=np.float64)
    y_buf  = np.empty((N), dtype=np.complex128)


    # make room to save krylov variables
    H_diff_krylov  = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
    Hh_krylov      = np.zeros((m_krylov+1,m_krylov), dtype=np.complex128)
    H_krylov      = np.zeros((m_krylov,m_krylov), dtype=np.complex128)
    Q_krylov       = np.zeros((N, m_krylov+1), dtype=np.complex128)
    phi_krylov     = np.zeros((m_krylov), dtype=np.complex128)
    fcivec_krylov     = np.zeros((m_krylov), dtype=np.complex128)

    start_print = time.perf_counter()
    
    k_coarse = 0   
    max_k_coarse = len(ss) -2

    phi_cplx = np.asarray(phi.r) + 1j * np.asarray(phi.i)
    phi_flat = np.asarray(phi_cplx).reshape(-1)
    fcivec_flat = np.asarray(fcivec_exact).reshape(-1)
    for step in range(n_steps+1):
        # Hamiltonian interpolate
        if (step<n_steps):
            t_mid = t + 0.5 * dt_step
            s_value = schedule(t_mid/t_f)
            last = False
            #print(t_mid,s_value, t_f)
        else:
            t_mid = t_f
            s_mid = 1.0
            #print(schedule(t_mid/t_f))
            last = True
        if (s_value>ss[k_coarse] and (not last)):
            print(t_mid,s_value, t_f)
            # 
            if (k_coarse>0): # not the first time
                phi_flat[:] = Q_krylov[:,:m_krylov]@phi_krylov
            # skip until ss[k_coarse]<s_value<ss[k_coarse+1]
            while k_coarse < max_k_coarse and s_value > ss[k_coarse+1]:
                k_coarse +=1
            # preprocess for each interval
            s_mid = (ss[k_coarse] + ss[k_coarse+1])/2.0
            print('# interval', ss[k_coarse], s_value, ss[k_coarse+1], k_coarse)
            ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_mid)
            h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
            h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)


            
            def matvec(x_flat):
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

            Q_krylov[:], Hh_krylov[:], _ = arnoldi(matvec, phi_flat, m_krylov)

            # Krylov projection of the Hamiltonian difference
            
            def matvec2(x_flat):
                # x_flat: 1-D complex128 of length N
                # 2) copy data into the 2-D buffers (no new allocation!)
                xr_buf[:, :] = x_flat.real.reshape((na, nb))
                xi_buf[:, :] = x_flat.imag.reshape((na, nb))
                # 3) call your C-kernel
                hop(xr_buf, h2e_diff_ptr, out_buf)
                real_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
                hop(xi_buf, h2e_diff_ptr, out_buf)
                # 4) combine and flatten
                imag_buf[:] = np.asarray(out_buf)   # now a plain ndarray, dtype=float64
                y_buf.real = real_buf
                y_buf.imag = imag_buf

                return y_buf


            for j in range(m_krylov):
                y = matvec2(Q_krylov[:,j])
                H_diff_krylov[:,j] = Q_krylov[:,:m_krylov].conj().T@ y 

            phi_krylov[:] = Q_krylov[:,:m_krylov].conj().T@phi_flat


            # fcivec to check
            fcivec_krylov[:] = Q_krylov[:,:m_krylov].conj().T@fcivec_flat

            k_coarse += 1

        s_relative = s_value - s_mid
        #print(s_relative)

        H_krylov[:] = -1j*dt_step*(Hh_krylov[:m_krylov,:m_krylov] + s_relative * H_diff_krylov)
        phi_krylov_next = expm_multiply(H_krylov, phi_krylov)
        phi_krylov[:] = phi_krylov_next
        norm = np.linalg.norm(phi_krylov)

        phi_krylov[:]/=norm

        # Overlap calculation

        overlap2 = np.abs(np.vdot(fcivec_krylov,phi_krylov))**2

        elapsed = time.perf_counter() - start_print
        if (elapsed>10):
            print(t, schedule(t/t_f), overlap2)
            start_print = time.perf_counter()

        t += dt_step

    return overlap2



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



taus_css = []
ss_css = []

with open('../schedule','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        taus_css.append(float(ls[0]))
        ss_css.append(float(ls[1]))




t_f_max = 73750.14590053925
dt = 0.5
num_tf = 32

nn = int(round(t_f_max / dt)) 
idxs = np.logspace(0, np.log10(nn), num=num_tf)
idxs_unique = np.unique(np.round(idxs).astype(int))
t_f_maxs = idxs_unique * dt



# 1) Greedy “best‑fit decreasing” partitioning
#    (runs identically on every rank)
assignments = [[] for _ in range(cores)]
loads       = [0.0]*cores

# sort tasks descending so big jobs get placed first
for val in sorted(t_f_maxs, reverse=True):
    # pick the core with the smallest current total
    i = loads.index(min(loads))
    assignments[i].append(val)
    loads[i] += val

# 2) pick out *this* rank’s tasks
my_tasks = np.array(assignments[core])

# now my_tasks[core] is a list whose sum(loads[core]) is 
# roughly the same as loads[j] for any other j.
print(f"rank {core} load = {loads[core]:.3g}, tasks = {my_tasks}")

local_results = []

m_krylov = 100

for t_f in my_tasks:
    print('# start: {t_f}'.format(t_f=t_f))
    start = time.perf_counter()

    phi = CIObject(fcivec_0, norb, nelec)
    ov2 = adiabatic_evolve_krylov(taus_css, ss_css, t_f, dt, phi, fcivec_exact, info_i, info_f, m_krylov)

    local_results.append((t_f, ov2))

    print('#', t_f, ov2)

    elapsed = time.perf_counter() - start
    print('# done: {t_f} with {elapsed} s'.format(t_f=t_f, elapsed=elapsed))


comm.Barrier()
all_results = comm.gather(local_results, root=0)

if core == 0:
    # all_results is now a list of lists, one per rank
    flat = [item for sublist in all_results for item in sublist]
    # optionally sort by the interpol value
    flat.sort(key=lambda x: x[0])
    ordered_F = [f for (_, f) in flat]
    with open('fidelity','w') as file_:
        for i in range(len(t_f_maxs)):
            s = '{:.16e}    {:.16e}'.format(t_f_maxs[i], ordered_F[i])
            #print(s)
            s += '\n'
            file_.write(s)
