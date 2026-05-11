import os
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

folder = '../'

file_path = folder + '/find_LK/gaps'

ss            = []
gaps          = []
with open(file_path,'r') as file_:
    lines = file_.readlines()
    for line in lines:
        if (line.startswith('#')):
            continue
        ls = line.split()
        ss.append(float(ls[0]))
        gaps.append(float(ls[1]))

p_list = np.linspace(1.0,2.0,num=11)


folder = './'

for p in p_list:
    folder_sub = folder + '{p:.1f}'.format(p=p)
    if not os.path.isdir(folder_sub):
        os.makedirs(folder_sub)
        print(f"'./{folder_sub} was created.")
    else:
        print(f"./{folder_sub} already exists.")

    gaps_mp = [1/x**p for x in gaps]

    taus = cumulative_trapezoid(gaps_mp, ss, initial=0)

    taus_max = taus[-1]

    taus /= taus_max

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
