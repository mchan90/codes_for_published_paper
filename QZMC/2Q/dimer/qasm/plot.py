import numpy as np
import qiskit_nature
from qiskit_nature.second_q.hamiltonians.lattices import (
    BoundaryCondition,
    LineLattice,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import ParityMapper

import sys
sys.path.append('../')

# qiskit's parity mapper "assumes" (orb1,spin1), ... (orbn,spin1), (orb1,spin2), ..., (orbn,spin2)
# ordering of second quantized orbitals, which is non consistent qiskit's own FermiHubbardModel routine,
# which has (orb1,spin1), (orb1,spin2), ... (orbn,spin1), (orbn,spin2)
# We modifed FermiHubbardModel, which uses ordering of 
# (orb1,spin1), ... (orbn,spin1), (orb1,spin2), ..., (orbn,spin2)
# This is efficient for two-site case only. In other cases, using
# Jordan-Wigner with qiskit's implementation of FermiHubbardModel is better.
from fermi_hubbard_model import FermiHubbardModel

qiskit_nature.settings.use_pauli_sum_op = False

n_site  = 2
n_qubit = 2*n_site-2 # use two qubit reduction
dim = 2**n_qubit

U_max = 5.0
Us = np.linspace(0.0,5.0,11)
nU = len(Us)
print(Us)
hamiltonians = []

n_electrons = [1,1]
for U_coulomb in Us:
    t = 1.0
    mu = U_coulomb/2
    bc = BoundaryCondition.PERIODIC
    chain = LineLattice(num_nodes=n_site, boundary_condition=bc, edge_parameter=-t, onsite_parameter=-mu)
    HubbardChain = FermiHubbardModel(chain,onsite_interaction=U_coulomb)
    H = HubbardChain.second_q_op().simplify()
    mapper = ParityMapper(num_particles=n_electrons)
    hamiltonians.append(mapper.map(H))

Us_fine = np.linspace(0.0,5.0,101)
nU_fine = len(Us_fine)
eigen_energies_fine = np.zeros((nU_fine,dim),dtype=float)

iU_fine = 0
for U_coulomb in Us_fine:
    t = 1.0
    mu = U_coulomb/2
    bc = BoundaryCondition.PERIODIC
    chain = LineLattice(num_nodes=n_site, boundary_condition=bc, edge_parameter=-t, onsite_parameter=-mu)
    HubbardChain = FermiHubbardModel(chain,onsite_interaction=U_coulomb)
    h = HubbardChain.second_q_op().simplify()
    mapper = ParityMapper(num_particles=n_electrons)
    h_qubit = mapper.map(h)
    eigen_e, eigen_v = np.linalg.eigh(h_qubit.to_matrix())
    eigen_energies_fine[iU_fine,:] = eigen_e[:]
    iU_fine += 1

n_hamiltonians = nU

for alpha in range(n_hamiltonians):
    print(alpha, hamiltonians[alpha])

hamiltonian_diffs = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs.append((hamiltonians[alpha+1]-hamiltonians[alpha]).simplify())
    print(hamiltonian_diffs[alpha])

hamiltonian_diffs_list = []
for alpha in range(n_hamiltonians-1):
    hamiltonian_diffs_list.append(hamiltonian_diffs[alpha].to_list())
    print(hamiltonian_diffs_list[alpha])


# exact eigenvalues
eigen_energies_exact  = np.zeros((n_hamiltonians,dim),dtype=float)
eigen_vectors_exact   = np.zeros((n_hamiltonians,dim,dim),dtype=complex)
for alpha in range(n_hamiltonians):
    eigen_e, eigen_v = np.linalg.eigh(hamiltonians[alpha].to_matrix())
    indx = np.argsort(eigen_e.real)
    for i in range(dim):
        eigen_energies_exact[alpha,i]   = eigen_e[indx[i]].real
        eigen_vectors_exact[alpha,:,i] = eigen_v[:,indx[i]]


# degenerate perturbation theory
import copy
nsec = 0
sec_list = []
done = [False]*dim
for i in range(dim):
    if done[i]:
        continue
    else:
        l = [i]
        done[i] =True
        for j in range(i+1,dim):
            if done[j]:
                continue
            else:
                if (np.abs(eigen_energies_exact[0,i]-eigen_energies_exact[0,j])<1E-8):
                    l.append(j)
                    done[j] = True
        sec_list.append(l)
#print(sec_list)
nsec = len(sec_list)
nesec = np.zeros(nsec,dtype=int)
for isec in range(nsec):
    nesec[isec] = len(sec_list[isec])
    #print(nesec[isec])
# diagonalize hamiltonian_diffs[0] for each degenerate sector
hp_mat = (eigen_vectors_exact[0,:,:].conj().transpose())@(hamiltonian_diffs[0].to_matrix())@eigen_vectors_exact[0,:,:]
#deps_sec = []
#w_sec   = []
for isec in range(nsec):
    hp_proj = np.zeros((nesec[isec],nesec[isec]),dtype=complex)
    for i in range(nesec[isec]):
        ii = sec_list[isec][i]
        for j in range(nesec[isec]):
            jj = sec_list[isec][j]
            hp_proj[i,j] = hp_mat[ii,jj]
    deps, w = np.linalg.eigh(hp_proj)
    v_t = np.zeros((dim,nesec[isec]),dtype=complex)

    for i in range(nesec[isec]):
        ii = sec_list[isec][i]
        v_t[:,i] = copy.deepcopy(eigen_vectors_exact[0,:,ii])
    for i in range(nesec[isec]):
        ii = sec_list[isec][i]
        eigen_vectors_exact[0,:,ii] = v_t@w[:,i]
    #print(deps,w)
    #deps_sec.append(deps)
    #w_sec.append(w)
#print(deps_sec,w_sec)
#print(eigen_vectors_exact[0,:,:])


def ExactTimeEvolution (alpha, time):
    Vl = copy.deepcopy(eigen_vectors_exact[alpha,:,:])
    evol = np.zeros((dim,dim),dtype=complex)
    exp_d = np.diag(np.exp(-1j*eigen_energies_exact[alpha,:]*time))
    evol = Vl@exp_d@Vl.conj().T
    return evol

# exact results
norms_exact  = np.ones((n_hamiltonians,dim),dtype=float)
for i in range(dim):
    phi = eigen_vectors_exact[0,:,i]
    for alpha in range(1,n_hamiltonians):
        proj_matrix = np.outer(eigen_vectors_exact[alpha,:,i],eigen_vectors_exact[alpha,:,i].conj())
        phi = proj_matrix@phi
        norms_exact[alpha,i] = phi.conj()@phi
        #print(phi.conj()@hamiltonians[alpha].to_matrix()@phi/norms_exact[alpha,i])
    #print('##')


for alpha in range(n_hamiltonians):
    print(norms_exact[alpha], eigen_energies_exact[alpha,:])


print(eigen_vectors_exact[0,:,0])
print(eigen_vectors_exact[0,:,1])
print(eigen_vectors_exact[0,:,2])
print(eigen_vectors_exact[0,:,3])



import random as rd
import pickle


nmc = int(100)
beta = 0.5


# observable amplitudes
n_obs = 3
# 0; norm, 1; dE1, 2; dE2
O_timelists         = [[[[] for _ in range(n_obs)] for _ in range(n_hamiltonians)] for _ in range(dim)]

with open('dimer.time.binary','rb') as file_:
    [O_timelists] = pickle.load(file_)

# run qzmc;

norms_qzmc               = np.ones((n_hamiltonians,dim),dtype=float)
eigen_energies_qzmc      = np.zeros((n_hamiltonians,dim),dtype=float)
eigen_energies_qzmc[0,:] = eigen_energies_exact[0,:]
# draw figure
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('pdf', 'svg')

with open('dimer.result.values.binary','rb') as file_:
    [result_values_save] = pickle.load(file_)

for i in range(dim):
    # initial eps

    eps = eigen_vectors_exact[0,:,i].conj().T@hamiltonians[1].to_matrix()@eigen_vectors_exact[0,:,i]
    eps = eps.real
    
    for alpha in range(1,n_hamiltonians):
        # compute values
        norm    = 0.0
        dE1     = 0.0
        dE2     = 0.0

        i_meas = 0
        # 0; norm
        for imc in range(nmc):
            norm += result_values_save[i][alpha][i_meas]
            i_meas += 1
        # 1; dE1
        nhd1 = len(hamiltonian_diffs[alpha-1])
        for ihd in range(nhd1):
            if (hamiltonian_diffs_list[alpha-1][ihd][0]=='I'*n_qubit):
                continue
            coeff = hamiltonian_diffs_list[alpha-1][ihd][1]
            for imc in range(nmc):
                dE1 +=  coeff * result_values_save[i][alpha][i_meas]
                i_meas += 1

        # 2; dE2
        if (alpha<n_hamiltonians-1):
            nhd2 = len(hamiltonian_diffs[alpha])
        else:
            nhd2 = 0
        for ihd in range(nhd2):
            if (hamiltonian_diffs_list[alpha][ihd][0]=='I'*n_qubit):
                continue
            coeff = hamiltonian_diffs_list[alpha][ihd][1]
            for imc in range(nmc):
                dE2 += coeff * result_values_save[i][alpha][i_meas]
                i_meas += 1
        
        dE1     /=norm
        dE2     /=norm

        norm    /=nmc

        # add constant contributions
        for ihd in range(nhd1):
            if (hamiltonian_diffs_list[alpha-1][ihd][0]=='I'*n_qubit):
                dE1 += hamiltonian_diffs_list[alpha-1][ihd][1]

        for ihd in range(nhd2):
            if (hamiltonian_diffs_list[alpha][ihd][0]=='I'*n_qubit):
                dE2 += hamiltonian_diffs_list[alpha][ihd][1]

        eigen_energies_qzmc[alpha,i] = eigen_energies_qzmc[alpha-1,i] + dE1
        norms_qzmc[alpha,i] = norm

        if (alpha<n_hamiltonians-1):
            eps = eigen_energies_qzmc[alpha,i] + dE2
            eps = eps.real
            
        print(alpha, norms_qzmc[alpha,i], eigen_energies_qzmc[alpha,i]-eigen_energies_exact[alpha,i])
        st = '# {i}/{dim}: {percent}%'.format(i=i+1,dim=dim,percent=((alpha)/(n_hamiltonians-1)*100))
        print(st)


import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('pdf','svg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


cm    = 1.0/2.54
rat   = 1.5
width = 8.6*cm * 1.5
height = 8.6*cm  * 1.5
plt.figure(figsize=(width,height),dpi=600)

marker_style_ref = dict(marker='',linestyle='dashed',color='black')

marker_style_1  = dict(marker='s', linestyle='none', color='tab:red', alpha=0.4,  fillstyle='full')

marker_style_4  = dict(marker='s', linestyle='none', color='black', alpha=1.0,  fillstyle='none')

marker_style_5  = dict(marker='D', linestyle='none', color='black', alpha=1.0,  fillstyle='none')

marker_style_6  = dict(marker='x', linestyle='none', color='black', alpha=1.0,  fillstyle='none')


axes = []
ax = plt.subplot2grid((1,1),(0,0))
axes.append(ax)

# plot (a)

y_min    =  -6
y_max    =  2
#
x_min    =  0
x_max    =  5
#
axes[0].set_xlim(x_min,x_max)
axes[0].set_ylim(y_min,y_max)
# setup labels
axes[0].text(-0.30,0.98,'(b)',transform=axes[0].transAxes)
#axes[0].text(0.5,-0.2,r'$R(\AA)$',transform=axes[0].transAxes)
axes[0].set_xlabel(r'$U/t$')
#axes[0].text(-0.25,0.5,r'$E(Ha)$',transform=axes[0].transAxes)
axes[0].set_ylabel(r'$E/t$')
# setup tics
axes[0].tick_params(axis='x', which='both', direction='in')
axes[0].tick_params(axis='y', which='both', direction='in')
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0].xaxis.set_major_formatter('{x:0.1f}')
axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

axes[0].yaxis.set_major_locator(ticker.MultipleLocator(1))
axes[0].yaxis.set_major_formatter('{x:0.1f}')
axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

# plot
axes[0].plot(Us_fine,eigen_energies_fine[:,0], label='', **marker_style_ref)
axes[0].plot(Us_fine,eigen_energies_fine[:,1], label='', **marker_style_ref)
axes[0].plot(Us_fine,eigen_energies_fine[:,2], label='', **marker_style_ref)
axes[0].plot(Us_fine,eigen_energies_fine[:,3], label='', **marker_style_ref)

axes[0].plot(Us,eigen_energies_qzmc[:,0], label ='', markersize=6, **marker_style_1)
axes[0].plot(Us,eigen_energies_qzmc[:,1], label ='', markersize=6, **marker_style_1)
axes[0].plot(Us,eigen_energies_qzmc[:,2], label ='', markersize=6, **marker_style_1)
axes[0].plot(Us,eigen_energies_qzmc[:,3], label ='', markersize=6, **marker_style_1)
#axes[0,0].plot(lds,E_sim[:,1], label ='', markersize=6, **marker_style_1_2)
#
#axes[0,0].plot(lds,E_ibmq[:,0], label='', markersize=4, **marker_style_2_1)
#axes[0,0].plot(lds,E_ibmq[:,1], label='', markersize=4, **marker_style_2_2)
#
#axes[0,0].plot(lds,E_ibmq_em[:,0], label='', markersize=6, **marker_style_3_1)
#axes[0,0].plot(lds,E_ibmq_em[:,1], label='', markersize=6, **marker_style_3_2)
#
#
### plot (b)
##y_min    = -1.05
##y_max    =  1.05
##
##x_min    =  -0.05
##x_max    =  1.05
##
##axes[0,1].set_xlim(x_min,x_max)
##axes[0,1].set_ylim(y_min,y_max)
### setup labels
##axes[0,1].text(-0.25,0.95,'(b)',transform=axes[0,1].transAxes)
##axes[0,1].text(0.5,-0.2,s=r'$\lambda$',transform=axes[0,1].transAxes)
##axes[0,1].text(-0.25,0.5,s=r'$\langle O\rangle$',transform=axes[0,1].transAxes)
##
##
##axes[0,1].text(0.4,0.15,s=r'$X$',transform=axes[0,1].transAxes)
##axes[0,1].text(0.8,0.6,s=r'$Y$',transform=axes[0,1].transAxes)
##axes[0,1].text(0.38,0.8,s=r'$Z$',transform=axes[0,1].transAxes)
### setup tics
##axes[0,1].tick_params(axis='x', which='both', direction='in')
##axes[0,1].tick_params(axis='y', which='both', direction='in')
##axes[0,1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
##axes[0,1].xaxis.set_major_formatter('{x:0.1f}')
##axes[0,1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
##
##axes[0,1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
##axes[0,1].yaxis.set_major_formatter('{x:0.1f}')
##axes[0,1].yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
##
### plot
##axes[0,1].plot(lds,Z_exact[:,0], label='', **marker_style_ref)
##axes[0,1].plot(lds,X_exact[:,0], label='', **marker_style_ref)
##axes[0,1].plot(lds,Y_exact[:,0], label='', **marker_style_ref)
##
##axes[0,1].plot(lds,Z_sim[:], label ='', markersize=4, **marker_style_1_1)
##axes[0,1].plot(lds,X_sim[:], label ='', markersize=4, **marker_style_1_2)
##axes[0,1].plot(lds,Y_sim[:], label ='', markersize=4, **marker_style_1_3)
##
##axes[0,1].plot(lds,Z_ibmq[:], label ='', markersize=4, **marker_style_2_1)
##axes[0,1].plot(lds,X_ibmq[:], label ='', markersize=4, **marker_style_2_2)
##axes[0,1].plot(lds,Y_ibmq[:], label ='', markersize=4, **marker_style_2_3)
##
##axes[0,1].plot(lds,Z_ibmq_em[:], label ='', markersize=4, **marker_style_3_1)
##axes[0,1].plot(lds,X_ibmq_em[:], label ='', markersize=4, **marker_style_3_2)
##axes[0,1].plot(lds,Y_ibmq_em[:], label ='', markersize=4, **marker_style_3_3)
##
### plot (c)
##y_min    = 0.0
##y_max    = 1.05
##
##x_min    =  -0.05
##x_max    =  1.05
##
##axes[1,0].set_xlim(x_min,x_max)
##axes[1,0].set_ylim(y_min,y_max)
### setup labels
##axes[1,0].text(-0.25,0.95,'(c)',transform=axes[1,0].transAxes)
##axes[1,0].text(0.5,-0.2,r'$\lambda$',transform=axes[1,0].transAxes)
##axes[1,0].text(-0.28,0.5,r'$||\Psi||^2$',transform=axes[1,0].transAxes)
### setup tics
##axes[1,0].tick_params(axis='x', which='both', direction='in')
##axes[1,0].tick_params(axis='y', which='both', direction='in')
##axes[1,0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
##axes[1,0].xaxis.set_major_formatter('{x:0.1f}')
##axes[1,0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
##
###axes[1,0].set_yscale('log')
##axes[1,1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
##axes[1,1].yaxis.set_major_formatter('{x:0.1f}')
##axes[1,1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
##
### plot
##axes[1,0].plot(lds,norm_exact[:,0], label='', **marker_style_ref)
##
##axes[1,0].plot(lds[1:],norm_sim[1:,0], label ='noiseless', markersize=4, **marker_style_4)
##
##axes[1,0].plot(lds[1:],norm_ibmq[1:,0], label='ibmq_lima', markersize=4, **marker_style_5)
##
##axes[1,0].plot(lds[1:],norm_ibmq_em[1:,0], label='ibmq_lima (em)', markersize=4, **marker_style_6)
##
##axes[1,0].legend(loc='lower right')
##
### plot (d)
##y_min    = 0.0
##y_max    = 1.05
##
##x_min    =  -0.05
##x_max    =  1.05
##
##axes[1,1].set_xlim(x_min,x_max)
##axes[1,1].set_ylim(y_min,y_max)
### setup labels
##axes[1,1].text(-0.25,0.95,'(d)',transform=axes[1,1].transAxes)
##axes[1,1].text(0.5,-0.2,r'$\lambda$',transform=axes[1,1].transAxes)
##axes[1,1].text(-0.25,0.5,r'$\mathcal{F}$',transform=axes[1,1].transAxes)
### setup tics
##axes[1,1].tick_params(axis='x', which='both', direction='in')
##axes[1,1].tick_params(axis='y', which='both', direction='in')
##axes[1,1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
##axes[1,1].xaxis.set_major_formatter('{x:0.1f}')
##axes[1,1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
##
###axes[1,1].set_yscale('log')
##axes[1,1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
##axes[1,1].yaxis.set_major_formatter('{x:0.1f}')
##axes[1,1].yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
##
### plot
##axes[1,1].axhline(y=1.0, label='', **marker_style_ref)
##
##axes[1,1].plot(lds[1:],F_sim[1:], label ='noiseless', markersize=4, **marker_style_4)
##
##axes[1,1].plot(lds[1:],F_ibmq[1:], label='ibmq_lima', markersize=4, **marker_style_5)
##
##axes[1,1].plot(lds[1:],F_ibmq_em[1:], label='ibmq_lima(em)', markersize=4, **marker_style_6)
##axes[1,1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('dimer.pdf')
