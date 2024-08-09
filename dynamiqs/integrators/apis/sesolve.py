from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...options import Options
from ...qarrays import QArray, QArrayLike, asqarray
from ...result import SEResult
from ...solver import Dopri5, Dopri8, Euler, Expm, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_array import Shape, TimeArray
from .._utils import (
    _astimearray,
    _cartesian_vectorize,
    _flat_vectorize,
    catch_xla_runtime_error,
    get_integrator_class,
)
from ..sesolve.diffrax_integrator import (
    SESolveDopri5Integrator,
    SESolveDopri8Integrator,
    SESolveEulerIntegrator,
    SESolveKvaerno3Integrator,
    SESolveKvaerno5Integrator,
    SESolveTsit5Integrator,
)
from ..sesolve.expm_integrator import SESolveExpmIntegrator


def sesolve(
    H: QArrayLike | TimeArray,
    psi0: QArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[QArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SEResult:
    r"""Solve the Schrödinger equation.

    This function computes the evolution of the state vector $\ket{\psi(t)}$ at time
    $t$, starting from an initial state $\ket{\psi_0}$, according to the Schrödinger
    equation (with $\hbar=1$ and where time is implicit(1))
    $$
        \frac{\dd\ket{\psi}}{\dt} = -i H \ket{\psi},
    $$
    where $H$ is the system's Hamiltonian.
    { .annotate }

    1. With explicit time dependence:
        - $\ket\psi\to\ket{\psi(t)}$
        - $H\to H(t)$

    Note-: Defining a time-dependent Hamiltonian
        If the Hamiltonian depends on time, it can be converted to a time-array using
        [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        Both the Hamiltonian `H` and the initial state `psi0` can be batched to
        solve multiple Schrödinger equations concurrently. All other arguments are
        common to every batch. See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.

    Args:
        H _(qarray-like or time-array of shape (...H, n, n))_: Hamiltonian.
        psi0 _(qarray-like of shape (...psi0, n, 1))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        exp_ops _(list of qarray-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. Defaults to
            [`dq.solver.Tsit5`][dynamiqs.solver.Tsit5] (supported:
            [`Tsit5`][dynamiqs.solver.Tsit5], [`Dopri5`][dynamiqs.solver.Dopri5],
            [`Dopri8`][dynamiqs.solver.Dopri8],
            [`Kvaerno3`][dynamiqs.solver.Kvaerno3],
            [`Kvaerno5`][dynamiqs.solver.Kvaerno5],
            [`Euler`][dynamiqs.solver.Euler],
            [`Expm`][dynamiqs.solver.Expm]).
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.SEResult`][dynamiqs.SEResult] object holding the result of the
            Schrödinger equation integration. Use the attributes `states` and `expects`
            to access saved quantities, more details in
            [`dq.SEResult`][dynamiqs.SEResult].
    """  # noqa: E501
    # === convert arguments
    H = _astimearray(H)
    psi0 = asqarray(psi0)
    tsave = jnp.asarray(tsave)
    exp_ops = [asqarray(exp_op) for exp_op in exp_ops] if exp_ops is not None else None

    # === check arguments
    _check_sesolve_args(H, psi0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_sesolve(H, psi0, tsave, exp_ops, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_sesolve(
    H: TimeArray,
    psi0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SEResult:
    # === vectorize function
    # we vectorize over H and psi0, all other arguments are not vectorized

    if not options.cartesian_batching:
        broadcast_shape = jnp.broadcast_shapes(H.shape[:-2], psi0.shape[:-2])
        H = H.broadcast_to(*(broadcast_shape + H.shape[-2:]))
        psi0 = psi0.broadcast_to(*(broadcast_shape + psi0.shape[-2:]))

    # `n_batch` is a pytree. Each leaf of this pytree gives the number of times
    # this leaf should be vmapped on.
    n_batch = (
        H.in_axes,
        Shape(psi0.shape[:-2]),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
        Shape(),
    )

    # the result is vectorized over `_saved` and `infos`
    out_axes = SEResult(False, False, False, False, 0, 0)

    # compute vectorized function with given batching strategy
    if options.cartesian_batching:
        f = _cartesian_vectorize(_sesolve, n_batch, out_axes)
    else:
        f = _flat_vectorize(_sesolve, n_batch, out_axes)

    # === apply vectorized function
    return f(H, psi0, tsave, exp_ops, solver, gradient, options)


def _sesolve(
    H: TimeArray,
    psi0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SEResult:
    # === select integrator class
    integrators = {
        Euler: SESolveEulerIntegrator,
        Dopri5: SESolveDopri5Integrator,
        Dopri8: SESolveDopri8Integrator,
        Tsit5: SESolveTsit5Integrator,
        Kvaerno3: SESolveKvaerno3Integrator,
        Kvaerno5: SESolveKvaerno5Integrator,
        Expm: SESolveExpmIntegrator,
    }
    integrator_class = get_integrator_class(integrators, solver)

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_class(tsave, psi0, H, exp_ops, solver, gradient, options)

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_sesolve_args(H: TimeArray, psi0: QArray, exp_ops: list[QArray] | None):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check exp_ops shape
    if exp_ops is not None:
        if not isinstance(exp_ops, list):
            raise TypeError(f'Argument `exp_ops` must be a list, got {type(exp_ops)}.')

        for exp_op in exp_ops:
            check_shape(exp_op, 'exp_ops', '(n, n)')
