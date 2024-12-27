from __future__ import annotations

from jaxtyping import ArrayLike, PRNGKeyArray

from ...gradient import Gradient
from ...options import Options
from ...qarrays.qarray import QArrayLike
from ...result import JSSESolveResult
from ...solver import Solver
from ...time_qarray import TimeQArray


def jssesolve(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    psi0: QArrayLike,
    tsave: ArrayLike,
    keys: PRNGKeyArray,
    *,
    exp_ops: list[QArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> JSSESolveResult:
    r"""Solve the jump stochastic Schrödinger equation (SSE).

    Warning:
        This function has not been implemented yet. The following API is indicative
        of the planned implementation.

    This function computes the evolution of the state vector $\ket{\psi(t)}$ at time $t$,
    starting from an initial state $\ket{\psi_0}$, according to the jump SSE (with
    $\hbar=1$ and where time is implicit(1))
    $$
        \dd\ket\psi = \dots
    $$
    where $H$ is the system's Hamiltonian, $\{L_k\}$ is a collection of jump operators,
    each continuously measured with perfect efficiency, and $\dd N_k$ are independent
    point processes with law
    $$
        \begin{split}
            \mathbb{P}[\dd N_k = 0] &= 1 - \mathbb{P}[\dd N_k = 1], \\\\
            \mathbb{P}[\dd N_k = 1] &= \braket{\psi| L_k^\dag L_k|\psi} \dt.
        \end{split}
    $$
    { .annotate }

    1. With explicit time dependence:
        - $\ket\psi\to\ket{\psi(t)}$
        - $H\to H(t)$
        - $L_k\to L_k(t)$
        - $\dd N_k\to \dd N_k(t)$

    Note-: Jump vs. diffusive SSE
        In quantum optics the _jump_ SSE corresponds to photon counting schemes, as
        opposed to the _diffusive_ SSE which corresponds to homodyne or heterodyne
        detection schemes.

    The continuous-time measurements are defined by the point processes $\dd N_k$. The
    solver returns the times at which the detector clicked:
    $I_k = \{t \text{ s.t. } \dd N_k(t)=1\}$.

    Note-: Defining a time-dependent Hamiltonian or jump operator
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
        [`dq.modulated()`][dynamiqs.modulated], or
        [`dq.timecallable()`][dynamiqs.timecallable]. See the
        [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
        tutorial for more details.

    Note-: Running multiple simulations concurrently
        The Hamiltonian `H` and the initial state `psi0` can be batched to
        solve multiple SSEs concurrently. All other arguments (including the PRNG key)
        are common to every batch. See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.

        Batching on `jump_ops` is not yet supported, if this is needed don't
        hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).

    Warning:
        For now, `jssesolve()` only supports linearly spaced `tsave` with values that
        are exact multiples of the solver fixed step size `dt`.

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-qarray, each of shape (n, n))_: List of
            jump operators.
        psi0 _(qarray-like of shape (...psi0, n, 1))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        keys _(list of PRNG keys)_: PRNG keys used to sample the point processes.
            The number of elements defines the number of sampled stochastic
            trajectories.
        exp_ops _(list of array-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        solver: Solver for the integration. No defaults for now, you have to specify a
            solver (supported: [`EulerMaruyama`][dynamiqs.solver.EulerMaruyama]).
        gradient: Algorithm used to compute the gradient. The default is
            solver-dependent, refer to the documentation of the chosen solver for more
            details.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.JSSESolveResult`][dynamiqs.JSSESolveResult] object holding the result of
            the jump SSE integration. Use the attributes `states`, `clicktimes`
            and `expects` to access saved quantities, more details in
            [`dq.JSSESolveResult`][dynamiqs.JSSESolveResult].
    """  # noqa: E501
    return NotImplemented
