# Python API

The **Dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Quantum solvers

::: dynamiqs.integrators
    options:
        table: true
        members:
        - sesolve
        - mesolve
        - smesolve
        - sepropagator
        - mepropagator
        - floquet

## Core

### Time-dependent arrays

::: dynamiqs.time_array
    options:
        table: true
        members:
        - TimeArray
        - constant
        - pwc
        - modulated
        - timecallable

### Solvers (dq.solver)

::: dynamiqs.solver
    options:
        table: true
        members:
        - Tsit5
        - Dopri5
        - Dopri8
        - Kvaerno3
        - Kvaerno5
        - Euler
        - Rouchon1
        - Rouchon2
        - Expm

### Gradients (dq.gradient)

::: dynamiqs.gradient
    options:
        table: true
        members:
        - Autograd
        - CheckpointAutograd

### Options

::: dynamiqs.options
    options:
        table: true
        members:
        - Options

### Results

::: dynamiqs.result
    options:
        table: true
        members:
        - SESolveResult
        - MESolveResult
        - SEPropagatorResult
        - MEPropagatorResult
        - FloquetResult

## Utilities

### Operators

::: dynamiqs.utils.operators
    options:
        table: true
        members:
        - eye
        - zero
        - destroy
        - create
        - number
        - parity
        - displace
        - squeeze
        - quadrature
        - position
        - momentum
        - sigmax
        - sigmay
        - sigmaz
        - sigmap
        - sigmam
        - hadamard
        - rx
        - ry
        - rz
        - sgate
        - tgate
        - cnot
        - toffoli


### States

::: dynamiqs.utils.states
    options:
        table: true
        members:
        - fock
        - fock_dm
        - basis
        - basis_dm
        - coherent
        - coherent_dm
        - ground
        - excited
        - thermal_dm


### Quantum utilities

::: dynamiqs.utils.quantum_utils
    options:
        table: true
        members:
        - dag
        - powm
        - expm
        - cosm
        - sinm
        - trace
        - tracemm
        - ptrace
        - tensor
        - expect
        - norm
        - unit
        - dissipator
        - lindbladian
        - isket
        - isbra
        - isdm
        - isop
        - isherm
        - toket
        - tobra
        - todm
        - proj
        - braket
        - overlap
        - fidelity
        - entropy_vn
        - bloch_coordinates
        - wigner


### JAX-related utilities

::: dynamiqs.utils.jax_utils
    options:
        table: true
        members:
        - to_qutip
        - set_device
        - set_precision
        - set_matmul_precision


### Vectorization

::: dynamiqs.utils.vectorization
    options:
        table: true
        members:
        - operator_to_vector
        - vector_to_operator
        - spre
        - spost
        - sprepost
        - sdissipator
        - slindbladian


### Quantum optimal control

::: dynamiqs.utils.optimal_control
    options:
        table: true
        members:
        - snap_gate
        - cd_gate


### Random arrays (dq.random)

::: dynamiqs.random
    options:
        table: true
        members:
        - real
        - complex
        - herm
        - psd
        - dm
        - ket


### Plotting (dq.plot)

::: dynamiqs.plot
    options:
        table: true
        members:
        - wigner
        - wigner_mosaic
        - wigner_gif
        - pwc_pulse
        - fock
        - fock_evolution
        - hinton
        - gifit
        - grid
        - mplstyle
