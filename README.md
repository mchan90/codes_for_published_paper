# Codes for published articles

This repository contains codes developed for the following publications.
If you use any of the codes, please cite the corresponding paper.

## Citations

### `QZMC/` — Quantum Zeno Monte Carlo

Mancheon Han, Hyowon Park, and Sangkook Choi,  
"Quantum Zeno Monte Carlo for computing observables,"  
*npj Quantum Information* **11**, 46 (2025).  
https://doi.org/10.1038/s41534-025-01002-3

```bibtex
@article{han2025qzmc,
  title     = {Quantum Zeno Monte Carlo for computing observables},
  author    = {Han, Mancheon and Park, Hyowon and Choi, Sangkook},
  journal   = {npj Quantum Information},
  volume    = {11},
  pages     = {46},
  year      = {2025},
  publisher = {Springer Nature},
  doi       = {10.1038/s41534-025-01002-3}
}
```

### `CGS/` — Constant Geometric Speed schedule for adiabatic state preparation

Mancheon Han, Hyowon Park, and Sangkook Choi,  
"Constant Geometric Speed schedule for adiabatic state preparation: Towards quadratic speedup without prior spectral knowledge,"  
*Phys. Rev. Research* (accepted, 2026).  
https://doi.org/10.1103/ygs3-xgb1

```bibtex
@article{han2026cgs,
  title     = {Constant Geometric Speed schedule for adiabatic state preparation: Towards quadratic speedup without prior spectral knowledge},
  author    = {Han, Mancheon and Park, Hyowon and Choi, Sangkook},
  journal   = {Physical Review Research},
  year      = {2026},
  doi       = {10.1103/ygs3-xgb1}
}
```

## Repository Structure

- `QZMC/` — Each directory corresponds to systems we considered. `fig.ipynb` plots figures in the main text, and `figS.ipynb` plots figures in the supplementary.
- `CGS/` — Subdirectories correspond to the systems considered (`2Fe2S/`, `N2/`, `adiabatic_grover/`). The top-level `plot_S1.ipynb` plots Fig. S1; each system folder contains its own `plot.ipynb` for the main-text figures and per-task subfolders (e.g. `scripts/`, `DFT/`, `FCI/`, `asp/`, `on_the_fly/`) for the simulation pipeline.
