# %%
import numpy as np
import time as time_lib
import random as rd
import copy
import psutil
import gc
import pickle
from mpi4py import MPI
from qiskit.quantum_info import SparsePauliOp
from scipy.interpolate import PchipInterpolator, AAA
n_qubit = 1
dim     = 2**n_qubit

comm = MPI.COMM_WORLD

core = comm.Get_rank()
cores = comm.Get_size()

# %%
def memory_usage(message: str = 'debug'):
    # this memory_usage function is imported from https://jybaek.tistory.com/895
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 30 # Bytes to GiB
    print(f"[{message}] memory usage: {rss: 10.5f} GiB")

# %%
# global optimizer for the band-limited function by mchan, 250403
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



# %%
# N for grover
n_grover = 10
N = 2**n_grover

# %%
# parameter to find schedule
beta = 2.0*np.sqrt(N)
nmc  = 10000
gamma = 0.04

# %%
def Hamiltonian(s):
    vz = 1.0 - 2.0 * (1-s) *(1-1/N)
    vx = (1-s) * 2/np.sqrt(N) * np.sqrt(1-1/N)
    h = SparsePauliOp.from_list([('I',0.5),('X',-0.5*vx),('Z',-0.5*vz)])
    return h


h = Hamiltonian(0)
eigen_e, eigen_v = np.linalg.eigh(h.to_matrix())

eigen_energies_qzmc = []

def Apply_ExactGaussian(eigen_e, eigen_v, eps, beta, v):
    w = v.conj()@eigen_v
    vec = np.exp(-0.5 * beta ** 2*(eigen_e-eps)**2)
    w = vec*w.conj()
    w = eigen_v@w
    return w

def Apply_ExactEvolution(eigen_e, eigen_v, eps, time, v):
    w = v.conj()@eigen_v
    vec = np.exp(-1j*time*(eigen_e-eps))
    w = vec*w.conj()
    w = eigen_v@w
    return w

def Apply_AdiabaticEvolution(schedule, t1, t2, dt, v):
    time = t2-t1
    n_steps = max(round(time/dt),1)
    dt_step = time/n_steps
    y = copy.copy(v)
    t = t1
    for step in range(n_steps+1):
        # Hamiltonian interpolate
        #print('##;', t1, t, t2)
        t_mid = t + 0.5 * dt_step if step<n_steps else t2
        s = schedule(t_mid)
        h = Hamiltonian(s)
        eigen_e, eigen_v = np.linalg.eigh(h.to_matrix())
        w = y.conj()@eigen_v
        vec = np.exp(-1j*dt_step*(eigen_e))
        w = vec*w.conj()
        w = eigen_v@w
        y = w
        t += dt_step
    return y

def compute_aj_from_amplitudes(sampled_times, amplitudes, eps):
    aj = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        i_time = 0
        phase = 0.0

        phase += eps * times[i_time]
        i_time += 1

        phase += eps * times[i_time]
        i_time += 1

        aj += amplitudes[imc] * np.exp(1j*phase)
    aj /= nmc
    aj = aj.real
    return aj

def compute_bj_from_amplitudes(alpha, sampled_times, amplitudes, eps):
    bj = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        i_time = 0
        phase = 0.0

        if (alpha>0):
            phase += eigen_energies_qzmc[alpha] * times[i_time]
            i_time += 1

        phase += eps * times[i_time]
        i_time += 1

        phase += eps * times[i_time]
        i_time += 1

        if (alpha>0):
            phase += eigen_energies_qzmc[alpha] * times[i_time]
            i_time += 1

        bj += amplitudes[imc] * np.exp(1j*phase)
    bj /= nmc
    bj = bj.real
    return bj

def compute_bj_deriv_from_amplitudes(alpha, sampled_times, amplitudes, eps):
    bj_deriv = 0.0
    for imc in range(nmc):
        times = sampled_times[imc]
        i_time = 0
        phase = 0.0
        factor= 0.0

        if (alpha>0):
            phase += eigen_energies_qzmc[alpha] * times[i_time]
            i_time += 1

        phase += eps * times[i_time]
        factor += times[i_time]
        i_time += 1

        phase += eps * times[i_time]
        factor += times[i_time]
        i_time += 1

        factor *= 1j

        if (alpha>0):
            phase += eigen_energies_qzmc[alpha] * times[i_time]
            i_time += 1

        bj_deriv += amplitudes[imc] * np.exp(1j*phase) * factor
    bj_deriv /= nmc
    bj_deriv = bj_deriv.real
    return bj_deriv

def compute_aj (alpha, s, phi_asp):
    if (alpha==0):
        return 1.0
    h = Hamiltonian(s)
    eigen_e, eigen_v = np.linalg.eigh(h.to_matrix())

    if (core==0):
        sampled_times = []
        omega_max = 0.0

        sampled_times = []
        omega_max = 0.0
        for imc in range(nmc):
            times = np.random.normal(0.0, beta, size=2)
            sampled_times.append(times)
            omega_max = max(omega_max,np.abs(times[0]+times[1]))
    else:
        sampled_times = None
        omega_max = None

    sampled_times = comm.bcast(sampled_times, root=0)
    omega_max     = comm.bcast(omega_max, root=0)

    # computation of aj
    amplitudes = []
    for imc in range(nmc):
        times = sampled_times[imc]
        i_time = 0

        phi = copy.copy(phi_asp)

        phi = Apply_ExactEvolution(eigen_e, eigen_v, 0.0,times[i_time],phi)
        i_time += 1

        phi = Apply_ExactEvolution(eigen_e, eigen_v, 0.0,times[i_time],phi)
        i_time += 1

        amplitudes.append(phi_asp.T.conj()@phi)

    aj = compute_aj_from_amplitudes(sampled_times, amplitudes, eigen_energies_qzmc[alpha])

    return aj

def compute_bj (alpha, s, s_next, phi_asp, last=False):
    hj = Hamiltonian(s)
    eigen_ej, eigen_vj = np.linalg.eigh(hj.to_matrix())

    h = Hamiltonian(s_next)
    eigen_e, eigen_v = np.linalg.eigh(h.to_matrix())

    #print('amplitude calc start')

    if (core==0):
        sampled_times = []
        omega_max = 0.0
        for imc in range(nmc):
            times = np.random.normal(0.0, beta, size=4)
            sampled_times.append(times)
            omega_max = max(omega_max,np.abs(times[1]+times[2]))
    else:
        sampled_times = None
        omega_max = None

    sampled_times = comm.bcast(sampled_times, root=0)
    omega_max     = comm.bcast(omega_max, root=0)

    # computation of bj
    amplitudes = []
    for imc in range(nmc):
        times = sampled_times[imc]
        i_time = 0

        phi = copy.copy(phi_asp)

        if (alpha>0):
            phi = Apply_ExactEvolution(eigen_ej, eigen_vj, 0.0,times[i_time],phi)
            i_time += 1

        phi = Apply_ExactEvolution(eigen_e, eigen_v, 0.0,times[i_time],phi)
        i_time += 1

        phi = Apply_ExactEvolution(eigen_e, eigen_v, 0.0,times[i_time],phi)
        i_time += 1

        if (alpha>0):
            phi = Apply_ExactEvolution(eigen_ej, eigen_vj, 0.0,times[i_time],phi)
            i_time += 1

        amplitudes.append(phi_asp.T.conj()@phi)

    #print('omega_max', omega_max)

    f = lambda x: compute_bj_from_amplitudes(alpha, sampled_times, amplitudes, x)
    df = lambda x: compute_bj_deriv_from_amplitudes(alpha, sampled_times, amplitudes, x)

    x_min = eigen_e[0]-0.1
    x_max = eigen_e[1]+0.1
    #print('s', s_next)
    #print('E', eigen_e)
    #print('ov', np.abs(eigen_v[:,0].conj().T@phi_asp)**2)

    print(f(eigen_e[0]))



    #print('optimal point finding')

    optimal_point, optimal_value = global_optimize_mpi(f,df,x_min,x_max,omega_max,1)
    #print(optimal_point, optimal_value)
    if (last==True):
        return optimal_point, optimal_value
    else:
        return optimal_value


# %%

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


#def isolate_interval(f, a, c, fa, fc, N0=10, tol=5e-3):
#    """
#    Isolate a bracketing interval [xL, xR] within [a, c] where f(x)=0 is expected.
#    - N0: initial number of subdivisions
#    - tol: threshold for near-zero detection
#    """
#
#    # Initial uniform sampling
#    xs = np.linspace(a, c, N0 + 1)
#    fs = np.zeros((N0+1),dtype=float)
#    fs[0] = fa
#    fs[-1] = fc
#
#    for i in range(1, N0):
#        fs[i] = f(xs[i])
#        if fa * fs[i] <=0:
#            return xs[i-1], xs[i], fs[i-1], fs[i]
#        if abs(fs[i]) < tol: # do not need to find a 
#            return a, xs[i], fa, fs[i]
#
#    # not found interval yet. Do a adaptive refinement using a cubic spline
#
#    spline = PchipInterpolator(xs, fs)
#
#    approx_roots = spline.roots()
#    approx_roots = approx_roots[(approx_roots >= a) & (approx_roots <= c)]    
#
#    # 4. Bracket around first approximate root
#    if approx_roots.size > 0:
#        x0 = approx_roots[0]
#        idx = np.searchsorted(xs, x0)
#        i = min(max(idx-1,0),N0-1) 
#        print('found from interpolation')
#        return xs[i-1], xs[i], fs[i-1], fs[i]
#    else:
#        return None

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






# fix seed to test
#base_seed = 12345
#np.random.seed(base_seed)


# %%
r_norm2 = 1.0-gamma

tol_norm = 0.005

s_list = [0]
alpha = 0
s = s_list[0]
t_list = [0]
dl_list  = []

h = Hamiltonian(1.0)
eigen_e_target, eigen_v_target = np.linalg.eigh(h.to_matrix())

# preparation of ASP parameters

h = Hamiltonian(0)
eigen_e, eigen_v = np.linalg.eigh(h.to_matrix())

eigen_energies_qzmc.append(eigen_e)
phi_asp = eigen_v[:,0]
t1_base = 8
t1    = t1_base * np.sqrt(N/1024)
dt    = 0.5

dl_small_ratio = 1e-6 # set this to the statistical error will be plausible

while s<1.0:
    aj = compute_aj(alpha, s, phi_asp)
    print('aj:', aj)
    func = lambda x: compute_bj(alpha, s, x, phi_asp)/aj- r_norm2

    # find a root
    fs = 1-r_norm2

    # isolate interval first
    result = isolate_interval(func, s, 1.0, fs, h=0.0009765625, r=2.0, tol=tol_norm)
    if (result==None):
        s_next = 1.0
    else:
        print(result)
        s1, s2, fs1, fs2 = result
        s_next = RootFinding(func, s1, s2, fa=fs1, fb=fs2, tol=tol_norm)
        s_next = comm.bcast(s_next, root=0)

    e, bj = compute_bj(alpha, s, s_next, phi_asp, last=True)

    print('bj: {bj}'.format(bj=bj))

    if (bj>aj): # statistical error
        dl = dl_small_ratio * dl_list[0]
    else:
        dl = np.sqrt(1.0-bj/aj)


    if (alpha==0):
        ratio = 1.0
    else:
        ratio =  dl/dl_list[0]

    s_list.append(s_next)
    dl_list.append(dl)
    t_list.append(t_list[-1] + ratio * t1)
    eigen_energies_qzmc.append(e)
    optimal_schedule = PchipInterpolator(t_list, s_list)
    #optimal_schedule = AAA(t_list, s_list)

    # adiabatic evolution
    phi_asp = Apply_AdiabaticEvolution(optimal_schedule, t_list[-2], t_list[-1], dt, phi_asp)

    print('# zeno', s_list[-2], s_list[-1], bj)
    print('# tf: ', t_list[-2], t_list[-1])
    print(np.abs(np.vdot(phi_asp,eigen_v_target[:,0]))**2)


    alpha += 1
    s = s_list[-1]

optimal_schedule = PchipInterpolator(t_list, s_list)
#optimal_schedule = AAA(t_list, s_list)

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
