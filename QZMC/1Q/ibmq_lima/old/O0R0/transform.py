import numpy as np 
import pickle
# job id file

file_ = open('ibmq_lima.save','rb') 
[job_ids_save,params_list] = pickle.load(file_)
file_.close()

file = open('ibmq_lima.save.formatted','w')
n_save = len(job_ids_save)
n_jobs = len(job_ids_save[0])
for i_save in range(n_save):
    s = '{:d}'.format(i_save)
    for k in range(n_jobs):
        s += '  {}'.format(job_ids_save[i_save][k])
    s += '\n'
    file.write(s)
file_.close()

# data file

file_ = open('massive_dirac.save','rb')
[beta, nmc, lds, Emc, normmc] = pickle.load(file_)
file_.close()
s ='{:.16g}'.format(beta)
dim = len(Emc[0,:])

file = open('massive_dirac.save.formatted','w')
s ='# {beta:.16g}\n'.format(beta=beta)
file.write(s)
s ='# {nmc:d}\n'.format(nmc=nmc)
file.write(s)
s ='# {nld:d}\n'.format(nld=len(lds))
file.write(s)

ild = 0
for ld in lds:
    #s =('{:.16g} '*(2*dim+1) +'\n').format(ld, normmc[ild,:], Emc[ild,:])
    s = '{:.16e}'.format(ld)
    for k in range(dim):
        s += '  {:.16e}  {:.16e}'.format(normmc[ild,k],Emc[ild,k])
    s += '\n'
    file.write(s)
    ild += 1 
file.close()
