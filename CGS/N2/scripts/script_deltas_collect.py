import os
import numpy as np
import re

number_pattern = re.compile(r"""
    [+-]?                           # optional sign
    (?:\d+\.\d*|\.\d+|\d+)          # decimal or integer
    (?:[eE][+-]?\d+)?               # optional exponent
""", re.VERBOSE)

r_min = 1.5
r_max = 4.0
step  = 0.1
n = int(round((r_max - r_min) / step)) + 1
bond_lengths = [r_min + step * i for i in range(n)]


p_list = np.linspace(1.0,2.0,num=11)
f_crit = 0.75

n_p = len(p_list)

t_asp_p = np.zeros((n,n_p),dtype=float)
gapmins = np.zeros((n), dtype=float)

skipped  = [False] * len(bond_lengths)

data_dir = '../data'
for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)

    # find the gap

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
    gapmin = min(gaps)

    # find adiabatic_path_length

    for i_p in range(n_p):
        p = p_list[i_p]
        folder_sub = folder + '/delta_schedules'+'/{p:.1f}'.format(p=p)
        file_path = folder_sub + '/asp/fidelity'

        tfs = [] 
        fidelities = []
        
        with open(file_path,'r') as file_:
            lines = file_.readlines()
            for line in lines:
                ls = line.split()
                tfs.append(float(ls[0]))
                fidelities.append(float(ls[1]))
        #print(tfs[0], tfs[-1])
        
        from scipy.interpolate import PchipInterpolator
        from scipy.optimize import brentq
        pchip = PchipInterpolator(tfs, fidelities)

        func = lambda x: pchip(x) - f_crit
        t_asp_p[j,i_p] = brentq(func, tfs[0], tfs[-1])
    gapmins[j] = gapmin
    print(gapmin,0,t_asp_p[j,0])

with open('T_asp_p_collected','w') as file_:
    ind = 0
    for j in range(len(bond_lengths)):
        if (skipped[j]):
            continue
        s = '{:.16e}    {:.16e}'.format(bond_lengths[j],gapmins[j])
        for i_p in range(n_p):
            s += '    {:.16e}'.format(t_asp_p[j,i_p])
        s += '\n'
        file_.write(s)
        ind += 1
