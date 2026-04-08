import pickle

#result_values_save = [[[] for _ in range(n_hamiltonians)] for _ in range(dim)]
with open('H2.result.values.binary','rb') as file_:
    [result_values_save] = pickle.load(file_)

dim =  len(result_values_save)
n_hamiltonians = len(result_values_save[0])
#print(dim,n_hamiltonians)
with open('H2.result.values','w') as file_:
    for i in range(dim):
        for alpha in range(1,n_hamiltonians):
            s = ''
            for value in result_values_save[i][alpha]:
                s += '  {:.16e}'.format(value)
            s += '\n'
            file_.write(s)
    
with open('H2.time.binary','rb') as file_:
    [O_timelists] = pickle.load(file_)

n_obs = len(O_timelists[0][1])
nmc   = len(O_timelists[0][1][0])
#print(n_obs,nmc)

with open('H2.time','w') as file_:
    for i in range(dim):
        for alpha in range(1,n_hamiltonians):
            for i_obs in range(n_obs):
                for imc in range(nmc):
                    s = ''
                    for time in O_timelists[i][alpha][i_obs][imc]:
                        s += '  {:.16e}'.format(time)
                    s += '\n'
                    file_.write(s)
