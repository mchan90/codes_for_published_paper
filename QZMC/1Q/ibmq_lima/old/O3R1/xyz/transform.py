import numpy as np 
import pickle

# job id file

file_ = open('xyz_ibmq_lima.save','rb') 
[job_ids_save,params_list] = pickle.load(file_)
file_.close()

file = open('xyz_ibmq_lima.save.formatted','w')
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
file_ = open('xyz.save','rb')
[Xmc,Ymc,Zmc,Fmc] = pickle.load(file_)
file_.close()
file = open('xyz.save.formatted','w')
ild = 0
for ld in lds:
    s = '{:.16e}'.format(ld)
    s += '  {:.16e}'.format(Xmc[ild,0])
    s += '  {:.16e}'.format(Ymc[ild,0])
    s += '  {:.16e}'.format(Zmc[ild,0])
    s += '  {:.16e}'.format(Fmc[ild,0])
    s += '\n'
    file.write(s)
    ild += 1 
file.close()
