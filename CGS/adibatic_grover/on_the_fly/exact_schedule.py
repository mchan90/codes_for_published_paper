import numpy as np

n_grover_list = np.arange(1,21,1)

for j in range(len(n_grover_list)):
    n_grover = n_grover_list[j]

    N = 2**n_grover

    folder = './' + str(n_grover)
    
    def exact_optimal_schedule(tau):
        s = 0.5 + 0.5/np.sqrt(N-1) * np.tan((2 * tau-1)*np.arctan(np.sqrt(N-1)))
        return s
    
    t_list = np.linspace(0,1,num=101)
    s_list = exact_optimal_schedule(t_list)
    
    with open(folder+'/exact_optimal_schedule','w') as file_:
        for i in range(len(s_list)):
            s = '{:.16e}    {:.16e}'.format(t_list[i], s_list[i])
            s += '\n'
            file_.write(s)
