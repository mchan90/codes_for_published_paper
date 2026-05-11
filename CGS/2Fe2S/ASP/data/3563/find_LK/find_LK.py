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

n_roots = 5


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
    # 1. 스칼라 곱 (Overlap) 계산: <phi|b>
    # vdot은 입력이 다차원 배열이어도 1D로 펴서 계산해주므로 안전함
    ovlp = np.vdot(phi, b)
    
    # 2. phi가 완벽히 규격화(norm=1) 되어있지 않을 수 있으니 norm 제곱으로 나눠줌
    phi_norm_sq = np.vdot(phi, phi)
    
    # 3. 평행 성분 제거 (Gram-Schmidt)
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
    기존 점(1개)을 유지하면서, 총합(M + num_new)을 정확히 맞추는 할당 함수
    """
    M = len(weights)
    target_total = M + num_new_points
    
    # 작업용 변수 복사
    active_mask = np.ones(M, dtype=bool) # 아직 배분 중인 구간들
    final_slots = np.zeros(M, dtype=float) # 최종 슬롯 개수 (실수)
    
    # 0. Weight 안전장치
    safe_weights = np.maximum(weights, 1e-12)
    
    # --- [Iterative Locking Loop] ---
    while True:
        # 1. 현재 살아남은 애들의 Weight 합
        current_w_sum = np.sum(safe_weights[active_mask])
        
        # 2. 살아남은 애들이 나눠가질 수 있는 남은 예산
        # (총 목표) - (이미 확정된 1개짜리들의 합)
        remaining_budget = target_total - np.sum(final_slots[~active_mask])
        
        # 예외: 남은 예산이 없거나 음수면 중단 (보통 안 생김)
        if remaining_budget <= 0:
            break
            
        # 3. 임시 할당량 계산 (Quota)
        if current_w_sum == 0:
            # 남은 애들 Weight가 다 0이면 균등 배분
            n_active = np.sum(active_mask)
            quotas = np.ones(n_active) * (remaining_budget / n_active)
        else:
            quotas = (safe_weights[active_mask] / current_w_sum) * remaining_budget
            
        # 4. [검사] 1.0 미만인 애들이 있는가?
        # (부동소수점 오차 고려하여 1.0보다 살짝 작게 잡음)
        under_min_indices_local = np.where(quotas < 0.999999)[0]
        
        if len(under_min_indices_local) == 0:
            # 모두가 1.0 이상이면 OK! 현재 계산된 Quota 저장하고 탈출
            final_slots[active_mask] = quotas
            break
        else:
            # 5. [Lock] 1.0 미만인 애들을 찾아서 '1'로 확정짓고 배분에서 제외
            # active_mask 상에서의 인덱스를 전체 인덱스로 변환해야 함
            full_indices = np.where(active_mask)[0]
            bad_indices = full_indices[under_min_indices_local]
            
            final_slots[bad_indices] = 1.0      # 1개로 확정
            active_mask[bad_indices] = False    # 목록에서 퇴출
            
            # 루프 다시 돔 -> 줄어든 예산을 가지고 남은 애들이 다시 계산
            # (부자들의 파이가 줄어들게 됨)

    # --- [Integer Rounding (Largest Remainder Method)] ---
    # 이제 final_slots는 모두 >= 1.0 인 실수임. 합은 target_total과 같음.
    # 정수로 변환하면서 오차(소수점)를 처리해야 함.
    
    slots_int = np.floor(final_slots).astype(int)
    fractional_parts = final_slots - slots_int
    
    # 현재 정수 합과 목표의 차이
    diff = target_total - np.sum(slots_int)
    
    # 소수점이 큰 순서대로 남은 갯수 배분
    priority = np.argsort(fractional_parts)[::-1]
    for i in range(int(diff)):
        slots_int[priority[i]] += 1
    # 최종적으로 '추가할 점의 개수'를 반환해야 하므로 1을 뺌
    return slots_int - 1

def generate_new_points(x, counts):
    """
    x: 현재 격자 점들 (길이 N)
    counts: 각 구간(N-1개)에 추가할 점의 개수 리스트 (정수 배열)
    return: 새로 추가된 점들의 1차원 배열 (x_new)
    """
    new_points_list = []
    
    # counts가 0보다 큰 구간만 순회
    # (nonzero를 쓰면 루프 횟수를 줄여서 더 빠름)
    indices = np.nonzero(counts)[0]
    
    for i in indices:
        n = counts[i]
        start, end = x[i], x[i+1]
        
        # linspace로 양 끝점 포함 n+2개를 만들고,
        # 슬라이싱[1:-1]으로 내부 점 n개만 취함
        pts = np.linspace(start, end, n + 2)[1:-1]
        new_points_list.append(pts)
    
    if len(new_points_list) > 0:
        return np.concatenate(new_points_list)
    else:
        return np.array([])

def compute_monotonic_cumulative_integral(y, x):
    """
    y >= 0 일 때, F(x)가 항상 증가하도록 보장하는 고정밀 적분
    """
    # 1. Cubic Spline 생성
    cs = CubicSpline(x, y)
    
    # 2. 각 구간별 적분값 계산 (Delta F)
    # antiderivative()를 쓰면 계수를 이용해 해석적으로 적분함
    # F(x_{i+1}) - F(x_i)
    
    # 방법 A: 일일이 integrate 호출 (직관적)
    delta_F = np.array([cs.integrate(x[i], x[i+1]) for i in range(len(x)-1)])
    
    # 3. Monotonicity 강제 (Hybrid Strategy)
    # Spline 적분값이 음수가 나오면(오버슈팅), 그 구간만 Trapezoid로 대체
    
    # Trapezoid 계산 (해당 구간만)
    h = np.diff(x)
    trapz_areas = (y[:-1] + y[1:]) * h / 2
    
    # Spline이 사고 친 곳(0보다 작거나, 혹은 너무 작은 값) 찾기
    # 물리적으로 y>=0이면 적분은 무조건 >= 0이어야 함
    bad_indices = np.where(delta_F < 0)[0]
    
    if len(bad_indices) > 0:
        # 그 구간만 사다리꼴 넓이로 교체
        delta_F[bad_indices] = trapz_areas[bad_indices]
        
    # 4. 누적 합 계산
    F = np.concatenate(([0], np.cumsum(delta_F)))
    
    return F


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
indx_home = '../'

ecore_i, h1e_i, g2e_i, norb, nelec, ms = loadERIs(indx_home + '/FCIDUMP_0')
ecore_f, h1e_f, g2e_f, norb, nelec, ms = loadERIs(material_home + '/DFT/output/FCIDUMP_FULL')
info_i = (ecore_i, h1e_i, g2e_i)
info_f = (ecore_f, h1e_f, g2e_f)

import os

taus_css = []
ss_css = []

with open(indx_home + '/qz/optimal_schedule','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        taus_css.append(float(ls[0]))
        ss_css.append(float(ls[1]))


optimal_schedule = PchipInterpolator(taus_css, ss_css)

comm.Barrier()


n_iter = 1
ntau_base = 256


for i_iter in range(n_iter):
    if (i_iter==0):
        tau_grid_list = np.linspace(0,1,num=ntau_base)
        s_grid_list = optimal_schedule(tau_grid_list)
        ss   = np.array([])
        vs   = np.array([])
        kappas = np.array([])
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
        ecore, h1e, g2e = fcidump_interpolate(info_i, info_f, s_val)
        mf = scf.RHF(mol)
        mci = fci.direct_spin1.FCI(mol)
        mci.spin = 0 
        mci.conv_tol = 1e-12
        mci.max_cycle = 2000
        mci = fci.addons.fix_spin_(mci, shift=3.0, ss=0.0)
        e, ci = mci.kernel(h1e, g2e, norb, nelec, nroots=n_roots, max_space=30)
    
    
        e0 = copy.copy(e[0])
        ci0 = copy.copy(ci[0])
        print(e)
        phi = np.asarray(ci0,dtype=np.float64).reshape(-1) # flatten
    
        ecore_diff, h1e_diff, g2e_diff = compute_hamiltonian_diff (info_i, info_f)
        h2e_diff = absorb_h1e(h1e_diff, g2e_diff, norb, nelec, 0.5)
        h2e_diff_ptr = h2e_diff.ctypes.data_as(ctypes.c_void_p)
        # 
        # link index 한 번 계산
        link_a, link_b = _unpack(norb, nelec, None)
        na, nlinka = link_a.shape[:2]
        nb, nlinkb = link_b.shape[:2]
        li_ptr = link_a.ctypes.data_as(ctypes.c_void_p)
        lb_ptr = link_b.ctypes.data_as(ctypes.c_void_p)
    
    
        # 버퍼 미리 할당
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
            # C 함수 호출 (두 전자 contraction)
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
    
        local_results.append((s_val, speed, kappa))
        elapsed = time.perf_counter() - start
        print('# done: {s_val} with {elapsed} s'.format(s_val=s_val, elapsed=elapsed))
    
    comm.Barrier()
    all_results = comm.allgather(local_results)
    
    # all_results is now a list of lists, one per rank
    flat = [item for sublist in all_results for item in sublist]
    # optionally sort by the s_core value
    flat.sort(key=lambda x: x[0])
    s_news, v_news, kappa_news = map(np.array, zip(*flat))


    s_combined = np.concatenate([ss, s_news])
    v_combined = np.concatenate([vs, v_news])
    kappa_combined = np.concatenate([kappas, kappa_news])

    sort_indices = np.argsort(s_combined)

    ss = s_combined[sort_indices]
    vs = v_combined[sort_indices]
    kappas = kappa_combined[sort_indices]

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

comm.Barrier()
MPI.Finalize()
