# Rho-Echo

Rho-Echo is a mechanochemical model of how RhoA signaling history and recovery shape subsequent microglial responses. The software exposes the seven-state model, including the proposed recoverable reserve, Src-linked signaling and distinct myosin-II outputs. Model coordinates represent effective cellular states.

## Install and verify

Use Python 3.10 or later.

```sh
python -m pip install .
python -m windowed_rhoa.rho_echo_kernel.benchmark --check
```

The benchmark checks a fixed simulated trajectory against the reference fixture. See the [kernel guide](code/python/windowed_rhoa/rho_echo_kernel/README.md) for a runnable example and the [source manifest](code/python/windowed_rhoa/rho_echo_kernel/origin_manifest.json) for the equation version and source hashes.

## Study replication

This repository provides the installable model and numerical benchmark. The manuscript-specific analysis and figure replay materials are described separately in the study's software and data deposits. Their reserved Zenodo identifiers are `10.5281/zenodo.19796404` and `10.5281/zenodo.19796406`, respectively. Figure replay from processed inputs and the benchmark have different scopes.

See [CITATION.cff](CITATION.cff) for citation metadata and [LICENSE](LICENSE) for the software license.

## Reproduce the equilibrium and first-passage calculations

From the source checkout, install NumPy, SciPy and Numba, then run:

```sh
python code/pipelines/run_rho_echo_formal_closure.py --n 1000 --out results/figure6-replay
```

This runs the equilibrium grid, checks that the jump process recovers the ODE drift and recomputes the seeded first-passage ensembles. The included calcium calibration contains only the three fitted parameter values used by the model. The optional sensitivity sweeps use additional frozen inputs from the study archive.
