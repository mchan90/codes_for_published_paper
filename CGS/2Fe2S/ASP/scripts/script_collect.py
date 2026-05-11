import os
import numpy as np
import shutil
import subprocess
import re

number_pattern = re.compile(r"""
    [+-]?                           # 선택적 부호
    (?:\d+\.\d*|\.\d+|\d+)          # 소수 혹은 정수
    (?:[eE][+-]?\d+)?               # 선택적 지수부
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

    # T_asp_est

    file_path = folder + '/T_max_finding/log_0'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# and the estimated T_asp is')):
                raw_nums = number_pattern.findall(line)
                T_asp = float(raw_nums[0])

    # T_asp_est_opt

    file_path = folder + '/opt_T_max_finding/log_0'

    if not os.path.isfile(file_path):
        print(f"'{file_path}' does not exists.")
        skipped[j] = True
        continue

    with open(file_path,'r') as file_:
        lines = file_.readlines()
        for line in lines:
            if (line.startswith('# and the estimated T_asp is')):
                raw_nums = number_pattern.findall(line)
                T_asp_opt = float(raw_nums[0])

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
    
    with open(folder+'/est_values','w') as file_:
        s = '# T_asp_est_max: {:.16e}'.format(T_asp)
        s += '\n'
        file_.write(s)

        s = '# T_asp_est_opt_max: {:.16e}'.format(T_asp_opt)
        s += '\n'
        file_.write(s)

        s = '# gap_est_min: {:.16e}'.format(gap)
        s += '\n'
        file_.write(s)

    collected.append([weight_list[j], T_asp, T_asp_opt, gap])

with open('data_collected','w') as file_:
    ind = 0
    for i in range(len(indx_list)):
        if (skipped[i]):
            continue
        s = '{:.16e}    {:.16e}    {:.16e}    {:.16e}'.format(collected[ind][0],collected[ind][1],collected[ind][2],collected[ind][3])
        s += '\n'
        file_.write(s)
        ind += 1
    
