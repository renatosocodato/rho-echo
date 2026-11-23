# Rho-Echo kernel interface

`windowed_rhoa.rho_echo_kernel` is the stable, versioned boundary around the
canonical seven-state Rho-Echo controller in `code/python/windowed_rhoa_model.py`.
It delegates every derivative, input function, observable, and integration call
to that source; it does not carry a second copy of the equations.

The public contract provides immutable request, parameter, input, and trajectory
schemas; explicit state ordering; a static origin manifest with source hashes;
and a multi-input golden trajectory. Environment-dependent parameter discovery
is intentionally outside the boundary. A request therefore serializes every
parameter required to reproduce its trajectory.

```python
from windowed_rhoa.rho_echo_kernel import (
    InputProgram,
    KernelParameters,
    RectangularPulse,
    SimulationRequest,
    simulate,
)

request = SimulationRequest.regular_grid(
    start=0.0,
    stop=60.0,
    samples=121,
    inputs=InputProgram(
        amyloid_beta=RectangularPulse(onset=0.0, amplitude=1.0, duration=40.0)
    ),
    parameters=KernelParameters.defaults().with_updates(src_mode="abeta_modulated"),
)
trajectory = simulate(request)
```

Run the portable numerical replay with:

```bash
PYTHONPATH=code/python python -m windowed_rhoa.rho_echo_kernel.benchmark --check
```

Generate a candidate fixture for a deliberately versioned kernel change with
`--stdout` or `--output`. Never replace the fixture after an incidental numerical
change: an equation, solver, state-order, observable, or default-parameter change
requires a new model version and a separately named fixture.
