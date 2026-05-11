import numpy as np
taus = []
s_list = []

with open('optimal_schedule','r') as file_:
    lines = file_.readlines()
    for line in lines:
        ls = line.split()
        taus.append(float(ls[0]))
        s_list.append(float(ls[1]))

from scipy.interpolate import PchipInterpolator
pchip = PchipInterpolator(taus, s_list)
taus = np.linspace(taus[0],taus[-1],num=100001)
s_list = pchip(taus)
with open('optimal_schedule_interpol_2','w') as file_:
    for i in range(len(s_list)):
        s = '{:.16e}    {:.16e}'.format(taus[i],s_list[i])
        #print(s)
        s += '\n'
        file_.write(s)
