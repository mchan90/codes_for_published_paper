# README for scripts us
## for initialization
### script_run_dft.py
dft run
### script_prepare_repetition.py
prepare initial points from dft calculation

## after initialization
### script_run_est.py 
to get gap min
### script_run_find_LK.py 
to get adiabatic path length L and total curvature K
### script_run_zeno.py
to get segmented cgs based on exact eigenstates
### script_run_on_the_fly.py
to get segmented cgs on the fly asp
### script_run_asp.py
to get fidelity vs asp time, must be conducted after on_the_fly
### script_run_delta_ps.py
asp simulation with ds/dt~\Delta^{p} schedule

## collection scripts
### script_enery_collect.py
collect DFT and FCI energies
### script_collect.py
collect adiabatic evolution times
### script_LK_collect.py
collect L and K
### script_deltas_collect.py
collect \Delta^{p} schedules data
