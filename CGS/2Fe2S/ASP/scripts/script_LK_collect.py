import os
import numpy as np
import shutil
import subprocess
import re

number_pattern = re.compile(r"""
    [+-]?                           # optional sign
    (?:\d+\.\d*|\.\d+|\d+)          # decimal or integer
    (?:[eE][+-]?\d+)?               # optional exponent
""", re.VERBOSE)


indx_list = []
weight_list = []
data_dir = '../data'
with open('../find_data_points/out','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        indx_list.append(int(ls[1]))
        weight_list.append(float(ls[3]))

collected = []
skipped  = [False] * len(indx_list)

for j in range(len(indx_list)):
    indx = indx_list[j]
    print(j, indx)
    folder = data_dir + '/' + str(indx_list[j])

    # gap_min

    file_path = folder + '/gap_min_finding/log_0'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# with the gap minimum value')):
                raw_nums = number_pattern.findall(line)
                gap = float(raw_nums[0])

    file_path = folder + '/find_LK/l_vs_s'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
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
        skipped[j] = True
        continue

    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# K is')):
                raw_nums = number_pattern.findall(line)
                K = float(raw_nums[0])
    
    collected.append([gap, L, K])

with open('LK_collected','w') as file_:
    ind = 0
    for i in range(len(indx_list)):
        if (skipped[i]):
            continue
        s = '{:.16e}    {:.16e}    {:.16e}'.format(collected[ind][0],collected[ind][1],collected[ind][2])
        s += '\n'
        file_.write(s)
        ind += 1
