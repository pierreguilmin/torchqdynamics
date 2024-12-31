from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import SESolveResult
from ...solver import Dopri5, Dopri8, Euler, Expm, Kvaerno3, Kvaerno5, Solver, Tsit5
from ...time_qarray import TimeQArray
from .._utils import (
    _astimeqarray,
    assert_solver_supported,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.diffrax_integrator import (
    sesolve_dopri5_integrator_constructor,
    sesolve_dopri8_integrator_constructor,
    sesolve_euler_integrator_constructor,
    sesolve_kvaerno3_integrator_constructor,
    sesolve_kvaerno5_integrator_constructor,
    sesolve_tsit5_integrator_constructor,
)
from ..core.expm_integrator import sesolve_expm_integrator_constructor


def sesolve(
    H: QArrayLike | TimeQArray,
    psi0: QArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[QArrayLike] | None = None,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SESolveResult:
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
        If the Hamiltonian depends on time, it can be converted to a time-qarray using
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
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
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
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options, see [`dq.Options`][dynamiqs.Options] (supported:
            `save_states`, `cartesian_batching`, `progress_meter`, `t0`,
            `save_extra`).

    Returns:
        [`dq.SESolveResult`][dynamiqs.SESolveResult] object holding the result of the
            Schrödinger equation integration. Use the attributes `states` and `expects`
            to access saved quantities, more details in
            [`dq.SESolveResult`][dynamiqs.SESolveResult].
    """  # noqa: E501
    # === convert arguments
    H = _astimeqarray(H)
    psi0 = asqarray(psi0)
    tsave = jnp.asarray(tsave)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_sesolve_args(H, psi0, exp_ops)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'sesolve')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    return _vectorized_sesolve(H, psi0, tsave, exp_ops, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_sesolve(
    H: TimeQArray,
    psi0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SESolveResult:
    # vectorize input over H and psi0
    in_axes = (H.in_axes, 0, None, None, None, None, None)
    out_axes = SESolveResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, psi0.ndim - 2, 0, 0, 0, 0, 0)
        f = cartesian_vmap(_sesolve, in_axes, out_axes, nvmap)
    else:
        n = H.shape[-1]
        bshape = jnp.broadcast_shapes(H.shape[:-2], psi0.shape[:-2])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        H = H.broadcast_to(*bshape, n, n)
        psi0 = psi0.broadcast_to(*bshape, n, 1)
        # vectorize the function
        f = multi_vmap(_sesolve, in_axes, out_axes, nvmap)

    return f(H, psi0, tsave, exp_ops, solver, gradient, options)


def _sesolve(
    H: TimeQArray,
    psi0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    solver: Solver,
    gradient: Gradient | None,
    options: Options,
) -> SESolveResult:
    # === select integrator constructor
    integrator_constructors = {
        Euler: sesolve_euler_integrator_constructor,
        Dopri5: sesolve_dopri5_integrator_constructor,
        Dopri8: sesolve_dopri8_integrator_constructor,
        Tsit5: sesolve_tsit5_integrator_constructor,
        Kvaerno3: sesolve_kvaerno3_integrator_constructor,
        Kvaerno5: sesolve_kvaerno5_integrator_constructor,
        Expm: sesolve_expm_integrator_constructor,
    }
    assert_solver_supported(solver, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(solver)]

    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=tsave,
        y0=psi0,
        solver=solver,
        gradient=gradient,
        result_class=SESolveResult,
        options=options,
        H=H,
        Es=exp_ops,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_sesolve_args(H: TimeQArray, psi0: QArray, exp_ops: list[QArray] | None):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')
