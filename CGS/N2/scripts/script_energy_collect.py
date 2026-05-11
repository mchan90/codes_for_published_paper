import numpy as np
r_min = 0.4
r_max = 4.0
step  = 0.1
n = int((r_max-r_min) /step) + 1  # 19
bond_lengths = [r_min + step * i for i in range(n)]


data_dir = '../data'
with open('Energies_DFT','w') as file_:
    for j in range(len(bond_lengths)):
        bond_length = bond_lengths[j]
        folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)
        Es = np.load(folder + "/E_0.npy")
        dE = np.load(folder + "/DFT/output/dE_DFT.npy")
        s = '{:.16e}'.format(bond_length)
        for j in range(len(Es)):
            s += '    {:.16e}'.format(Es[j]+dE)
        s += '\n'


        file_.write(s)

with open('Energies_FCI','w') as file_:
    for j in range(len(bond_lengths)):
        bond_length = bond_lengths[j]
        folder = data_dir + '/{bond_length:.1f}'.format(bond_length=bond_length)
        Es = np.load(folder + "/E_FULL.npy")
        s = '{:.16e}'.format(bond_length)
        for j in range(len(Es)):
            s += '    {:.16e}'.format(Es[j])
        s += '\n'
        print(bond_length, Es)

        file_.write(s)
