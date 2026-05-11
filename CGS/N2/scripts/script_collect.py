import os
import numpy as np
import shutil
import subprocess


r_min = 1.5
r_max = 4.0
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]

data_dir = '../data'

f_crit = 0.75

collected = []
skipped  = [False] * len(bond_lengths)

for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    print(j, bond_length)
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)

    file_path = folder + '/est/T_asp_est'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue
    gaps          = []
    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            ls = line.split()
            gaps.append(float(ls[2]))
    gap = min(gaps)

    file_path = folder + '/asp/linear/fidelity'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    tfs = [] 
    fidelities = []
    
    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            ls = line.split()
            tfs.append(float(ls[0]))
            fidelities.append(float(ls[1]))
    print(tfs[0], tfs[-1])
    
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    pchip = PchipInterpolator(tfs, fidelities)

    func = lambda x: pchip(x) - f_crit
    t_asp_linear = brentq(func, tfs[0], tfs[-1])

    file_path = folder + '/asp/optimal/fidelity'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    tfs = [] 
    fidelities = []
    
    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            ls = line.split()
            tfs.append(float(ls[0]))
            fidelities.append(float(ls[1]))
    
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    pchip = PchipInterpolator(tfs, fidelities)

    func = lambda x: pchip(x) - f_crit
    t_asp_optimal = brentq(func, tfs[0], tfs[-1])

    
    print(gap, t_asp_linear, t_asp_optimal)

    collected.append([bond_length, t_asp_linear, t_asp_optimal, gap])

with open('data_collected','w') as file_:
    ind = 0
    for i in range(len(bond_lengths)):
        if (skipped[i]):
            continue
        s = '{:.16e}    {:.16e}    {:.16e}    {:.16e}'.format(collected[ind][0],collected[ind][1],collected[ind][2],collected[ind][3])
        s += '\n'
        file_.write(s)
        ind += 1
    
