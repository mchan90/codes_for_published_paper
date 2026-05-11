# README for scripts us
## data_points
file generated from figure 2 C of Lee2023. (found from matching initial overlaps, x-axis of the figure)
## for initialization
### run scripts in ../DFT, ../FCI
### script_prepare_repetition.py
prepare initial points

## after initialization
### script_run_est.py 
to get gap min and adiabatic evolution time estimates
### script_run_find_LK.py 
to get adiabatic path length L and total curvature K
### script_run_zeno.py
to get segmented cgs based on exact eigenstates
### script_run_on_the_fly.py
to get segmented cgs on the fly asp
### script_run_asp.py
to get fidelity vs asp time, must be conducted after on_the_fly
### script_run_max_T.py
to get more precise evolution time estimate using linear schedule (should after run_est)
### script_run_max_T_opt.py
to get more precise maximum evolution time estimate using cgs schedule (should after run_est)
### script_run_min_gap.py
to get more precise gap min estimate (should after run_est)

## collection scripts
### script_collect.py
collect adiabatic evolution time estimates
### script_LK_collect.py
collect L and K


