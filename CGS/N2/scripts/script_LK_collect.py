import os
import numpy as np
import re

number_pattern = re.compile(r"""
    [+-]?                           # optional sign
    (?:\d+\.\d*|\.\d+|\d+)          # decimal or integer
    (?:[eE][+-]?\d+)?               # optional exponent
""", re.VERBOSE)

r_min = 0.5
r_max = 4.0
step  = 0.1
n = int(round((r_max - r_min) / step)) + 1
bond_lengths = [r_min + step * i for i in range(n)]


data_dir = '../data'

collected = []
for j in range(len(bond_lengths)):
    bond_length = bond_lengths[j]
    folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)

    # gap_min

    file_path = folder + '/est/T_asp_est'

    gaps          = []
    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            ls = line.split()
            gaps.append(float(ls[2]))
    gap = min(gaps)

    file_path = folder + '/find_LK/l_vs_s'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        continue

    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# Total adiabatic path length is')):
                raw_nums = number_pattern.findall(line)
                L = float(raw_nums[0])

    file_path = folder + '/find_LK/K_vs_s'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        continue

    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# K is')):
                raw_nums = number_pattern.findall(line)
                K = float(raw_nums[0])
    
    collected.append([bond_length, gap, L, K])

with open('LK_collected','w') as file_:
    ind = 0
    for i in range(len(bond_lengths)):
        s = '{:.16e}    {:.16e}    {:.16e}    {:.16e}'.format(collected[ind][0],collected[ind][1],collected[ind][2],collected[ind][3])
        s += '\n'
        file_.write(s)
        ind += 1
