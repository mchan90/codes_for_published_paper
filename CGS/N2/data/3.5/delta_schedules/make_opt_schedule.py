import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

folder = '../'

file_path = folder + '/find_LK/l_vs_s'

ss            = []
lls            = []
with open(file_path,'r') as file_:
    lines = file_.readlines()
    for line in lines:
        if (line.startswith('#')):
            continue
        ls = line.split()
        ss.append(float(ls[0]))
        lls.append(float(ls[1]))

folder = './'
folder_sub = folder + 'opt'
if not os.path.isdir(folder_sub):
    os.makedirs(folder_sub)
    print(f"'./{folder_sub} was created.")
else:
    print(f"./{folder_sub} already exists.")

taus = [x/lls[-1] for x in lls]

schedule = PchipInterpolator(taus, ss)

# only a few important points are used for a practical reason

ntau = 21

taus_grids = np.linspace(0,1,num=ntau)

ss_grids = schedule(taus_grids)

file_path = folder_sub + '/schedule'
with open(file_path,'w') as file_:
    for i in range(len(taus_grids)):
        s = '{:.16e}    {:.16e}'.format(taus_grids[i], ss_grids[i])
        s += '\n'
        file_.write(s)

folder_sub = folder_sub + '/asp'

if not os.path.isdir(folder_sub):
    os.makedirs(folder_sub)
    print(f"'./{folder_sub} was created.")
else:
    print(f"./{folder_sub} already exists.")
