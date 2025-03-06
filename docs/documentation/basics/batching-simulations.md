# Batching simulations

Batching allows **running many independent simulations concurrently**. It can dramatically speedup simulations, especially on GPUs. In this tutorial, we explain how to batch quantum simulations in Dynamiqs.

```python
import dynamiqs as dq
import jax.numpy as jnp
import timeit
```

## Batching in short

Batching in Dynamiqs is achieved by **passing a list of Hamiltonians, initial states, or jump operators** to the simulation functions. The result of a batched simulation is a single qarray that contains all the individual simulations results. For example, let's simulate the Schrödinger equation for all combinations of the three Hamiltonians $\{\sigma_x, \sigma_y, \sigma_z\}$ and the four initial states $\{\ket{g}, \ket{e}, \ket{+}, \ket{-}\}$:

```python
# define three Hamiltonians
H = [dq.sigmax(), dq.sigmay(), dq.sigmaz()]  # (3, 2, 2)

# define four initial states
g = dq.basis(2, 0)
e = dq.basis(2, 1)
plus = (g + e).unit()
minus = (g - e).unit()
psi = [g, e, plus, minus]  # (4, 2, 1)

# run the simulation
tsave = jnp.linspace(0.0, 1.0, 11)  # (11,)
result = dq.sesolve(H, psi, tsave)
print(f'Shape of result.states: {result.states.shape}')
```

```text title="Output"
Shape of result.states: (3, 4, 11, 2, 1)
```

The returned states is a qarray with shape _(3, 4, 11, 2, 1)_, where _3_ is the number of Hamiltonians, _4_ is the number of initial states, _11_ is the number of saved states, and _(2, 1)_ is the shape of a single state.

!!! Note
    All relevant `result` attributes are batched. For example if you specified `exp_ops`, the resulting expectation values `result.expects` will be an array with shape _(3, 4, len(exp_ops), 11)_.

Importantly, **batched simulations are not run sequentially in a `for` loop**. What is meant by *batching* is that instead of evolving from initial to final time a single state with shape _(2, 1)_ for each combination of argument, the whole batched state _(3, 4, 2, 1)_ is evolved **once** from initial to final time, which is much more efficient.

## Batching modes

There are two ways to batch simulations in Dynamiqs: **cartesian batching** and **flat batching**.

### Cartesian batching

The simulation runs for all possible combinations of Hamiltonians, jump operators and initial states. This is the default mode.

=== "`dq.sesolve`"
    For `dq.sesolve`, the returned qarray has shape:
    ```
    result.states.shape = (...H, ...psi0, ntsave, n, 1)
    ```
    where `...x` indicates the batching shape of `x`, i.e. its shape without the last two dimensions.

    !!! Example "Example: Cartesian batching with `dq.sesolve`"
        - `H` has shape _(2, 3, n, n)_,
        - `psi0` has shape _(4, n, 1)_,

        then `result.states` has shape _(2, 3, 4, ntsave, n, 1)_.

=== "`dq.mesolve`"
    For `dq.mesolve`, the returned qarray has shape:
    ```
    result.states.shape = (...H, ...L0, ...L1,  (...), ...rho0, ntsave, n, n)
    ```
    where `...x` indicates the batching shape of `x`, i.e. its shape without the last two dimensions.

    !!! Example "Example: Cartesian batching with `dq.mesolve`"
        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
        - `rho0` has shape _(7, n, n)_,

        then `result.states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.

### Flat batching

The simulation runs for each set of Hamiltonians, jump operators and initial states using broadcasting. This mode can be activated by passing `options=dq.Options(cartesian_batching=False)`. In particular for [`dq.mesolve()`][dynamiqs.mesolve], each jump operator can be batched independently from the others.

??? Note "What is broadcasting?"
    JAX and NumPy broadcasting semantics are very powerful and allow you to write concise and efficient code. For more information, see the [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

=== "`dq.sesolve`"
    For `dq.sesolve`, the returned qarray has shape:
    ```
    result.states.shape = (..., ntsave, n, 1)
    ```
    where `... = jnp.broadcast_shapes(H, psi0)` is the broadcasted shape of all arguments.

    !!! Example "Example: Flat batching with `dq.sesolve`"
        - `H` has shape _(2, 3, n, n)_,
        - `psi0` has shape _(3, n, 1)_,

        then `result.states` has shape _(2, 3, ntsave, n, 1)_.

=== "`dq.mesolve`"
    For `dq.mesolve`, the returned qarray has shape:
    ```
    result.states.shape = (..., ntsave, n, n)
    ```
    where `... = jnp.broadcast_shapes(H, L0, L1, ..., rho0)` is the broadcasted shape of all arguments.

    !!! Example "Example: Flat batching with `dq.mesolve`"
        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
        - `rho0` has shape _(3, n, n)_,

        then `result.states` has shape _(2, 3, ntsave, n, n)_.

!!! Note
    Any batch shape is valid as input as long as it is broadcastable with other arguments.

    For example for `dq.sesolve()` with `H` of shape _(2, 3, n, n)_, `psi0` can be of shape: _(n, 1)_, _(3, n, 1)_, _(2, 1, n, 1)_, _(2, 3, n, 1)_, _(..., 2, 3, n, 1)_, etc. By playing with the arguments shape, you have complete freedom over the simulation you want to run.


## Creating batched arguments

### Single-dimensional batching

There are multiple ways to create a batched argument.

=== "Using lists"
    The most straightforward way is to pass a list of values:
    ```python
    # define several Hamiltonians
    Hx, Hy, Hz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
    H = [Hx, Hy, Hz]  # (3, 2, 2)
    ```
=== "Using JAX broadcasting"
    It is often useful to sweep a parameter:
    ```python
    # define several Hamiltonians
    omega = jnp.linspace(0.0, 1.0, 21)
    H = omega[:, None, None] * dq.sigmaz()  # (21, 2, 2)
    ```
=== "Using Dynamiqs functions"
    Or you can use Dynamiqs utility functions directly:
    ```python
    # define several initial states
    alpha = [1.0, 2.0, 3.0]
    psis = dq.coherent(16, alpha)  # (3, 16, 1)
    ```

### Multi-dimensional batching

The previous examples illustrate batching over one dimension, but you can batch over as many dimensions as you want:
=== "Using lists"
    ```python
    # define several Hamiltonians
    H = [
        [Hx, 2 * Hx, 3 * Hx, 4 * Hx],
        [Hy, 2 * Hy, 3 * Hy, 4 * Hy],
        [Hz, 2 * Hz, 3 * Hz, 4 * Hz]
    ]  # (3, 4, 2, 2)
    ```
=== "Using JAX broadcasting"
    ```python
    # define several Hamiltonians
    omega = jnp.linspace(0.0, 1.0, 21)[:, None, None, None]
    eps = jnp.linspace(0.0, 10.0, 11)[:, None, None]
    H = omega * dq.sigmaz() + eps * dq.sigmaz()  # (21, 11, 2, 2)
    ```
=== "Using Dynamiqs functions"
    ```python
    # define several initial states
    alpha_real = jnp.linspace(0, 1.0, 5)
    alpha_imag = jnp.linspace(0, 1.0, 6)
    alpha = alpha_real[:, None] + 1j * alpha_imag  # (5, 6)
    psis = dq.coherent(16, alpha)  # (5, 6, 16, 1)
    ```

### Batching over a time-qarray

We have seen how to batch over time-independent objects, but how about time-dependent ones? It's essentially the same, you have to pass a batched [`TimeQArray`][dynamiqs.TimeQArray], in short:

=== "For a `PWCTimeQArray`"
    The batching of the returned time-qarray is specified by `values`. For example, to define a PWC operator batched over a parameter $\theta$:
    ```pycon
    >>> thetas = jnp.linspace(0.0, 1.0, 11)  # (11,)
    >>> times = [0.0, 1.0, 2.0]
    >>> values = thetas[:, None] * jnp.array([3.0, -2.0])  # (11, 2)
    >>> qarray = dq.sigmaz()
    >>> H = dq.pwc(times, values, qarray)
    >>> H.shape
    (11, 2, 2)
    ```
=== "For a `ModulatedTimeQArray`"
    The batching of the returned time-qarray is specified by the qarray returned by `f`. For example, to define a modulated Hamiltonian $H(t)=\cos(\omega t)\sigma_x$ batched over the parameter $\omega$:
    ```pycon
    >>> omegas = jnp.linspace(0.0, 1.0, 11)  # (11,)
    >>> f = lambda t: jnp.cos(omegas * t)
    >>> H = dq.modulated(f, dq.sigmax())
    >>> H.shape
    (11, 2, 2)
    ```
=== "For a `CallableTimeQArray`"
    The batching of the returned time-qarray is specified by the qarray returned by `f`. For example, to define an arbitrary time-dependent operator batched over a parameter $\theta$:
    ```pycon
    >>> thetas = jnp.linspace(0.0, 1.0, 11)  # (11,)
    >>> f = lambda t: thetas[:, None, None] * dq.asqarray([[t, 0], [0, 1 - t]])
    >>> H = dq.timecallable(f)
    >>> H.shape
    (11, 2, 2)
    ```

## Why batching?

When batching multiple simulations, the state is not a 2-D qarray that evolves in time but a N-D qarray which holds all independent simulations. This allows running **multiple simulations simultaneously** with great efficiency, especially on GPUs. Moreover, it usually simplifies the simulation code and also the subsequent analysis of the results, because they are all gathered in a single large qarray.

Common use cases for batching include:

- simulating a system with different values of a parameter (e.g. a drive amplitude),
- simulating a system with different initial states (e.g. for gate tomography),
- performing an optimisation using multiple starting points with random initial guesses (for parameters fitting or quantum optimal control).

### Quick benchmark

To illustrate the performance gain of batching, let us compare the total run time between using a for loop vs using a batched simulation. We will simulate a set of 3,000 Hamiltonians on a two-level system:

```ipython
# define 3000 Hamiltonians
omega = jnp.linspace(0.0, 10.0, 100)[:, None, None]
epsilon = jnp.linspace(0.0, 1.0, 30)[:, None, None, None]
H = omega * dq.sigmaz() + epsilon * dq.sigmax()  # (100, 30, 2, 2)

# other simulation parameters
psi0 = dq.basis(2, 0)
tsave = jnp.linspace(0.0, 1.0, 50)
options = dq.Options(progress_meter=False)

# running the simulations successively
def run_unbatched():
    results = []
    for i in range(len(omega)):
        for j in range(len(epsilon)):
            result = dq.sesolve(H[i, j], psi0, tsave, options=options)
            results.append(result)
    return results

# running the simulations simultaneously
def run_batched():
    return dq.sesolve(H, psi0, tsave, options=options)

# exclude JIT time from benchmarking by running each function once first
%timeit -n1 -r1 -q run_unbatched()
%timeit -n1 -r1 -q run_batched()

# time functions
%timeit run_unbatched()
%timeit run_batched()
```

```text title="Output"
2.59 s ± 52.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
44.1 ms ± 2.66 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

As we can see, the batched simulation is **much faster** than the unbatched one. In this simple example, it is about 60 times faster. The gain in performance will be even more significant for larger simulations, or when using a GPU.
