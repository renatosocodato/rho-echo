# Rho-Echo

Rho-Echo is an installable mechanochemical model of how RhoA signaling history and recovery shape subsequent microglial responses to amyloid-β. The seven-state model links RhoA and Src signaling to distinct myosin-II outputs and represents recoverable reserve as an inferred state.

## Install and run

Use Python 3.10 or later in a virtual environment. Install the versioned wheel from GitHub Releases.

```sh
python -m pip install https://github.com/renatosocodato/rho-echo/releases/download/v1.0.0/rho_echo_kernel-1.0.0-py3-none-any.whl
rho-echo --version
rho-echo check
rho-echo example > request.json
rho-echo simulate request.json > trajectory.json
```

The reference check compares 216 values from a fixed trajectory with the packaged fixture. The example emits a complete request with inputs, effective parameters and sampling times. The simulation returns time points, states and derived observables, together with solver status and model metadata. For use from Python, see the [kernel API guide](code/python/windowed_rhoa/rho_echo_kernel/README.md).

## How the software is organized

| Component | Role |
| --- | --- |
| [Kernel API](code/python/windowed_rhoa/rho_echo_kernel/api.py) and [schemas](code/python/windowed_rhoa/rho_echo_kernel/schemas.py) | Public simulation entry point and serializable input and output types |
| [Canonical model](code/python/windowed_rhoa_model.py) | Equations, observables and solver implementation |
| [Origin manifest](code/python/windowed_rhoa/rho_echo_kernel/origin_manifest.json) | Model identity and source hashes checked during numerical replay |
| [Equilibrium and first-passage runner](code/pipelines/run_rho_echo_formal_closure.py) | Model calculations for Figure 6 |

## Manuscript and reproduction

This software accompanies *A mechanochemical model links RhoA signaling history and recovery to microglial responses to amyloid-β*. The manuscript cites [the original reproducibility snapshot](https://github.com/renatosocodato/rho-echo/tree/ea6b6fe84efca6ac3262d9b28cf4f0656c53a389). Release v1.0.0 keeps the same equations, calibration inputs and numerical replay.

To reproduce the equilibrium grid, verify that jump-process drift recovers the ODE and recompute the seeded first-passage ensembles, run from the release source checkout:

```sh
python -m pip install '.[replay]'
python code/pipelines/run_rho_echo_formal_closure.py --n 1000 --out results/figure6-replay
```

The included calcium calibration records the three fitted values used by this runner. Optional sensitivity sweeps require additional inputs from the study archive. The broader analysis and figure code and the data replication package have reserved Zenodo DOIs **10.5281/zenodo.19796404** and **10.5281/zenodo.19796406**, respectively.

## Cite and verify

Cite the specific software version used. See [CITATION.cff](CITATION.cff), the [MIT license](LICENSE) and the [release files and SHA-256 checksums](https://github.com/renatosocodato/rho-echo/releases/tag/v1.0.0).

For development, install the package from the source checkout and run `python -m pytest tests` with pytest installed.

GitHub Actions checks the installed wheel and CLI and replays the Figure 6 calculations. The hosted jobs use a cross-platform envelope (relative tolerance 1e-6, absolute tolerance 1e-8) for differences in adaptive integration. The original strict comparison is available with `rho-echo check`. The packaged reference values and strict tolerances are unchanged.
