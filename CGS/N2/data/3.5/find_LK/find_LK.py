import numpy as np
from pyscf.fci import spin_op
from mpi4py import MPI
from scipy.integrate import cumulative_trapezoid, trapezoid, simpson
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline
from scipy.sparse.linalg import LinearOperator, gmres
import ctypes
from pyscf import lib
import copy
import time

from pyscf.fci.direct_spin1 import _unpack, absorb_h1e, FCIvector
# Load the optimized C library for FCI
libfci = lib.load_library('libfci')

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()

n_roots_initial = 10


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

def orthogonalize(b, phi):
    # 1. Inner product (overlap) <phi|b>
    # vdot flattens multidimensional inputs to 1D internally, so this is safe
    ovlp = np.vdot(phi, b)
    
    # 2. phi may not be perfectly normalized (norm=1), so divide by squared norm
    phi_norm_sq = np.vdot(phi, phi)
    
    # 3. Remove the parallel component (Gram-Schmidt)
    b_perp = b - (ovlp / phi_norm_sq) * phi
    
    return b_perp

# integration routines

def get_weights (y, x):
    # find a weights
    n = len(x)
    d = np.zeros(n, dtype=float)
    for i in range(1, n-1):
        sl = (y[i]-y[i-1])/(x[i]-x[i-1])
        sr = (y[i+1]-y[i])/(x[i+1]-x[i])
        d[i] = 2*(sr-sl)/(x[i+1]-x[i-1])
    weights = np.zeros(n-1, dtype=float)
    weights[0] = np.abs(d[1]) * (x[1]-x[0]) ** 2 # /12
    weights[-1] = np.abs(d[-2]) * (x[-1]-x[-2]) ** 2 # /12
    for i in range(1, n-2):
        weights[i] = (np.abs(d[i]) + np.abs(d[i+1]))/2.0 * (x[i+1]-x[i])**2 #/12
    return weights

def distribute_new_points(weights, num_new_points):
    """
    Allocation function that keeps each existing point (1 slot) and matches the total (M + num_new) exactly.
    """
    M = len(weights)
    target_total = M + num_new_points
    
    # Working copies of the variables
    active_mask = np.ones(M, dtype=bool) # Intervals still being allocated
    final_slots = np.zeros(M, dtype=float) # Final slot counts (float)
    
    # 0. Weight safeguard
    safe_weights = np.maximum(weights, 1e-12)
    
    # --- [Iterative Locking Loop] ---
    while True:
        # 1. Sum of weights over the currently surviving intervals
        current_w_sum = np.sum(safe_weights[active_mask])
        
        # 2. Remaining budget the surviving intervals can share
        # (total target) - (sum of intervals already locked at 1)
        remaining_budget = target_total - np.sum(final_slots[~active_mask])
        
        # Edge case: stop if the remaining budget is zero or negative (rare)
        if remaining_budget <= 0:
            break
            
        # 3. Tentative quota calculation
        if current_w_sum == 0:
            # Distribute equally if all remaining weights are zero
            n_active = np.sum(active_mask)
            quotas = np.ones(n_active) * (remaining_budget / n_active)
        else:
            quotas = (safe_weights[active_mask] / current_w_sum) * remaining_budget
            
        # 4. [Check] Are there any intervals below 1.0?
        # (slightly below 1.0 to absorb floating-point error)
        under_min_indices_local = np.where(quotas < 0.999999)[0]
        
        if len(under_min_indices_local) == 0:
            # All >= 1.0: store the current quota and exit
            final_slots[active_mask] = quotas
            break
        else:
            # 5. [Lock] Pin intervals below 1.0 to '1' and exclude them from the allocation
            # Map active_mask indices back to the full index space
            full_indices = np.where(active_mask)[0]
            bad_indices = full_indices[under_min_indices_local]
            
            final_slots[bad_indices] = 1.0      # Locked to 1
            active_mask[bad_indices] = False    # Removed from the active set
            
            # Loop again: surviving intervals recompute with the reduced budget
            # (the larger shares shrink as a result)

    # --- [Integer Rounding (Largest Remainder Method)] ---
    # All entries of final_slots are now >= 1.0 (float); their sum equals target_total.
    # Convert to integers while handling the fractional-part residual.
    
    slots_int = np.floor(final_slots).astype(int)
    fractional_parts = final_slots - slots_int
    
    # Difference between the current integer sum and the target
    diff = target_total - np.sum(slots_int)
    
    # Distribute the remainder in decreasing order of fractional part
    priority = np.argsort(fractional_parts)[::-1]
    for i in range(int(diff)):
        slots_int[priority[i]] += 1
    # Subtract 1 since we return the *number of points to add*
    return slots_int - 1

def generate_new_points(x, counts):
    """
    x: current grid points (length N)
    counts: number of points to add per interval (N-1 ints)
    return: 1D array of the newly inserted points (x_new)
    """
    new_points_list = []
    
    # Iterate only over intervals with counts > 0
    # (nonzero shortens the loop; faster)
    indices = np.nonzero(counts)[0]
    
    for i in indices:
        n = counts[i]
        start, end = x[i], x[i+1]
        
        # Build n+2 points including both endpoints via linspace,
        # then slice [1:-1] to keep only the n interior points
        pts = np.linspace(start, end, n + 2)[1:-1]
        new_points_list.append(pts)
    
    if len(new_points_list) > 0:
        return np.concatenate(new_points_list)
    else:
        return np.array([])

def compute_monotonic_cumulative_integral(y, x):
    """
    High-precision integration that keeps F(x) monotonically increasing whenever y >= 0
    """
    # 1. Build the cubic spline
    cs = CubicSpline(x, y)
    
    # 2. Per-interval integrated values (Delta F)
    # antiderivative() integrates analytically using the spline coefficients
    # F(x_{i+1}) - F(x_i)
    
    # Method A: call integrate() for each interval (straightforward)
    delta_F = np.array([cs.integrate(x[i], x[i+1]) for i in range(len(x)-1)])
    
    # 3. Enforce monotonicity (hybrid strategy)
    # If a spline-integrated value is negative (overshoot), replace that interval with the trapezoid value
    
    # Trapezoid computation (per-interval)
    h = np.diff(x)
    trapz_areas = (y[:-1] + y[1:]) * h / 2
    
    # Identify intervals where the spline misbehaves (negative or too small)
    # Physically, when y>=0 the integral must be >= 0
    bad_indices = np.where(delta_F < 0)[0]
    
    if len(bad_indices) > 0:
        # Replace only those intervals with the trapezoidal area
        delta_F[bad_indices] = trapz_areas[bad_indices]
        
    # 4. Cumulative sum
    F = np.concatenate(([0], np.cumsum(delta_F)))
    
    return F


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
mol.verbose = 0
mol.build()
mol.symmetry = False
mol.build()

material_home = '../'

ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs(material_home + '/FCIDUMP_0')
ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs(material_home + '/DFT/output/FCIDUMP_FULL')
info_i = (ecore_i, h1e_i, g2e_i)
info_f = (ecore_f, h1e_f, g2e_f)
import os

taus_css = []
ss_css = []

with open(material_home + '/on_the_fly/optimal_schedule','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        taus_css.append(float(ls[0]))
        ss_css.append(float(ls[1]))



optimal_schedule = PchipInterpolator(taus_css, ss_css)

comm.Barrier()


n_iter = 3
ntau_base = 256
n_iter_fci = 20


for i_iter in range(n_iter):
    if (i_iter==0):
        tau_grid_list = np.linspace(0,1,num=ntau_base)
        s_grid_list = optimal_schedule(tau_grid_list)
        ss   = np.array([])
        vs   = np.array([])
        kappas = np.array([])
        gaps  = np.array([])
    elif (i_iter%2==1):
        weights = get_weights(vs, ss) ** (1/3)
        n_new_points = distribute_new_points(weights, ntau_base)
        s_grid_list = generate_new_points(ss, n_new_points)
    elif (i_iter%2==0):
        weights = get_weights(kvs, ss) ** (1/3)
        n_new_points = distribute_new_points(weights, ntau_base)
        s_grid_list = generate_new_points(ss, n_new_points)

    ind_grid_list = range(len(s_grid_list))

    my_tasks = ind_grid_list[core :: cores]
    local_results = []
    
    start_0 = time.perf_counter()
    
    for ind in my_tasks:
        s_val = s_grid_list[ind]
        start = time.perf_counter()
    
        print('# start: s = {s}'.format(s=s_val))
        n_roots = n_roots_initial
        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_val)
        mf = scf.RHF(mol)
        mci = fci.direct_spin1.FCI(mol)
        mci.spin = 0 
        mci.conv_tol = 1e-12
        mci.max_cycle = 2000
        mci = fci.addons.fix_spin_(mci, shift=0.05, ss=0.0)
        for iter_fci in range(n_iter_fci):
            start_fci = time.perf_counter()
            max_space = max(30, n_roots * 4)
            if (iter_fci==0):
                e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=n_roots, max_space=max_space)
            else:
                e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=n_roots, max_space=max_space, ci0=copy.copy(ci))

            elapsed = time.perf_counter() - start_fci
            print('# : FCI done with n_roots= {n_roots}, with {elapsed} s: '.format(n_roots=n_roots, elapsed=elapsed))

            n_roots_found = 0
            energies_found = []
            for i_roots in range(n_roots):
                mult = fci.spin_op.spin_square0(ci[i_roots], norb, nelec)[1]
                if (np.abs(mult -1.0) < 0.01):
                    if (n_roots_found==0):
                        ci0 = ci[i_roots]
                    energies_found.append(e[i_roots])
                    n_roots_found += 1
            if (n_roots_found>=2):
                e0 = energies_found[0]
                gap = energies_found[1]-e0
                break
            n_roots += 2

                    
        print(energies_found)

        phi = np.asarray(ci0,dtype=np.float64).reshape(-1) # flatten
    
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
    
        out_buf = np.empty((na*nb), dtype=np.float64).view(FCIvector)
    
        N = na*nb
    
        xr_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
        xi_buf = np.empty((na, nb), dtype=np.float64).view(FCIvector)
        real_buf = np.empty(N, dtype=np.float64)
        imag_buf = np.empty(N, dtype=np.float64)
    
        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_val)
        h2e = absorb_h1e(h1e, g2e, norb, nelec, 0.5)
        h2e_ptr = h2e.ctypes.data_as(ctypes.c_void_p)
    
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
    
    
        # implement Q(H-E_0) for the stability
        def matvec(x_flat):
            xr_buf[:, :] = x_flat.real.reshape((na, nb))
            # 3) call your C-kernel
            hop(xr_buf, h2e_ptr, out_buf)
            real_buf[:] = np.asarray(out_buf).reshape(-1)   # now a plain ndarray, dtype=float64
            
            # extract e0 * x_flat
            real_buf[:] = real_buf[:] - e0 * x_flat
            real_buf[:] = real_buf[:]-np.vdot(phi,real_buf) * phi # extract phi parallel part
            return real_buf
    
        def matvec2(x_flat):
            # 2) copy data into the 2-D buffers (no new allocation!)
            xr_buf[:, :] = x_flat.real.reshape((na, nb))
            # 3) call your C-kernel
            hop(xr_buf, h2e_diff_ptr, out_buf)
            real_buf[:] = np.asarray(out_buf).reshape(-1)   # now a plain ndarray, dtype=float64
            return real_buf
    
        # 1: compute b
        vec_temp = matvec2(phi)
        E_dot = np.vdot(phi,vec_temp)
        b = (E_dot * phi - vec_temp)
    
        # 2: define linear operator Q (H-E_0)@ x
        A_op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        scale = np.sqrt(np.vdot(b,b))
        print('# scale: ', scale)
    
    
        # 3: solve linear equation
        phi_dot, exit_code = gmres(A_op, b, restart=2000, rtol=1e-8)
        
        err = np.abs(np.vdot(phi_dot,phi))
        
        # orthogonalize phi_dot to phi
        phi_dot = orthogonalize(phi_dot,phi)
    
        speed = np.sqrt(np.vdot(phi_dot,phi_dot))
    
    
        # check eq error
        residual = b - matvec(phi_dot)
        residual_abs = np.sqrt(np.vdot(residual,residual))
    
        print('# exit code: ', exit_code)
        print('# s: ', s_val, '# speed: ', speed)
        print('# non-orthogonality: ', err/speed)
        print('# residual: ', residual_abs/scale)
    
        # find second derivative
    
        # 1: compute b
        vec_temp = matvec2(phi_dot)
        b = (E_dot * phi_dot - vec_temp)
        b = orthogonalize(b,phi)
        b *=2
    
        scale = np.sqrt(np.vdot(b,b))
        print('# scale: ', scale)
    
        # 3: solve linear equation
        phi_ddot, exit_code = gmres(A_op, b, restart=2000, rtol=1e-8)
    
        err = np.abs(np.vdot(phi_ddot,phi))
        phi_ddot = orthogonalize(phi_ddot,phi)
    
    
        # check eq error
        residual = b - matvec(phi_ddot)
        residual_abs = np.sqrt(np.vdot(residual,residual))
    
        # 4: extract only curvature component
    
        norm_phi_ddot = np.sqrt(np.vdot(phi_ddot,phi_ddot))
    
        phi_ddot = orthogonalize(phi_ddot,phi_dot)
    
        kappa = np.sqrt(np.vdot(phi_ddot,phi_ddot))/speed**2
    
    
        print('# kappa: ', kappa)
        print('# exit code: ', exit_code)
        print('# non-orthogonality: ', err/norm_phi_ddot)
        print('# residual: ', residual_abs/scale)
    
        local_results.append((s_val, speed, kappa, gap))
        elapsed = time.perf_counter() - start
        print('# done: {s_val} with {elapsed} s'.format(s_val=s_val, elapsed=elapsed))
    
    comm.Barrier()
    all_results = comm.allgather(local_results)
    
    # all_results is now a list of lists, one per rank
    flat = [item for sublist in all_results for item in sublist]
    # optionally sort by the s_core value
    flat.sort(key=lambda x: x[0])
    s_news, v_news, kappa_news, gap_news = map(np.array, zip(*flat))


    s_combined = np.concatenate([ss, s_news])
    v_combined = np.concatenate([vs, v_news])
    kappa_combined = np.concatenate([kappas, kappa_news])
    gap_combined = np.concatenate([gaps, gap_news])


    sort_indices = np.argsort(s_combined)

    ss = s_combined[sort_indices]
    vs = v_combined[sort_indices]
    kappas = kappa_combined[sort_indices]
    gaps = gap_combined[sort_indices]

    l_values = compute_monotonic_cumulative_integral (vs, ss)

    L = l_values[-1]
    taus = l_values/L
    inverse_schedule = PchipInterpolator(ss, taus)

    kvs = kappas*vs

    Ks = compute_monotonic_cumulative_integral (kvs, ss)
    
    if core == 0 and i_iter<(n_iter-1):
        with open('est.'+str(i_iter),'w') as file_:
            for i in range(len(taus)):
                s = '{:.16e}    {:.16e}    {:.16e}    {:.16e}'.format(taus[i], ss[i], vs[i], kappas[i])
                s += '\n'
                file_.write(s)
    
        l_simpson = simpson(vs, ss)
        with open('l_vs_s.'+str(i_iter),'w') as file_:
            s = '# Total adiabatic path length is {:.16e}'.format(l_values[-1])
            s += '\n'
            s += '# Error estimate is {:.16e}'.format(np.abs(l_values[-1]-l_simpson))
            s += '\n'
            file_.write(s)
            s = '# s, l(s)'
            s += '\n'
            file_.write(s)
            for i in range(len(l_values)):
                s = '{:.16e}    {:.16e}'.format(ss[i], l_values[i])
                s += '\n'
                file_.write(s)
        K_simpson = simpson(kvs, ss)
        with open('K_vs_s.'+str(i_iter),'w') as file_:
            s = '# K is  {:.16e}'.format(Ks[-1])
            s += '\n'
            s += '# Error estimate is {:.16e}'.format(np.abs(Ks[-1]-K_simpson))
            s += '\n'
            file_.write(s)
            s = '# s, K(s)'
            s += '\n'
            file_.write(s)
            for i in range(len(ss)):
                s = '{:.16e}    {:.16e}'.format(ss[i], Ks[i])
                s += '\n'
                file_.write(s)

        with open('gaps.'+str(i_iter),'w') as file_:
            s = '# s, \Delta(s)'
            s += '\n'
            file_.write(s)
            for i in range(len(ss)):
                s = '{:.16e}    {:.16e}'.format(ss[i], gaps[i])
                s += '\n'
                file_.write(s)

    comm.Barrier()
    elapsed = time.perf_counter() - start_0
    print('# done: iteration # {i_iter} with {elapsed} s'.format(i_iter=i_iter, elapsed=elapsed))
# now explicitly finalize and exit
if (core==0):
    with open('est','w') as file_:
        for i in range(len(taus)):
            s = '{:.16e}    {:.16e}    {:.16e}    {:.16e}'.format(taus[i], ss[i], vs[i], kappas[i])
            s += '\n'
            file_.write(s)

    l_simpson = simpson(vs, ss)
    with open('l_vs_s','w') as file_:
        s = '# Total adiabatic path length is {:.16e}'.format(l_values[-1])
        s += '\n'
        s += '# Error estimate is {:.16e}'.format(np.abs(l_values[-1]-l_simpson))
        s += '\n'
        file_.write(s)
        s = '# s, l(s)'
        s += '\n'
        file_.write(s)
        for i in range(len(l_values)):
            s = '{:.16e}    {:.16e}'.format(ss[i], l_values[i])
            s += '\n'
            file_.write(s)

    K_simpson = simpson(kvs, ss)
    with open('K_vs_s','w') as file_:
        s = '# K is  {:.16e}'.format(Ks[-1])
        s += '\n'
        s += '# Error estimate is {:.16e}'.format(np.abs(Ks[-1]-K_simpson))
        s += '\n'
        file_.write(s)
        s = '# s, K(s)'
        s += '\n'
        file_.write(s)
        for i in range(len(ss)):
            s = '{:.16e}    {:.16e}'.format(ss[i], Ks[i])
            s += '\n'
            file_.write(s)

    with open('gaps','w') as file_:
        s = '# s, \Delta(s)'
        s += '\n'
        file_.write(s)
        for i in range(len(ss)):
            s = '{:.16e}    {:.16e}'.format(ss[i], gaps[i])
            s += '\n'
            file_.write(s)


comm.Barrier()
MPI.Finalize()
