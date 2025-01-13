from __future__ import annotations

from typing import ClassVar

import equinox as eqx

from ._utils import tree_str_inline
from .gradient import Autograd, CheckpointAutograd, Gradient

__all__ = [
    'Dopri5',
    'Dopri8',
    'Euler',
    'EulerMaruyama',
    'Expm',
    'Kvaerno3',
    'Kvaerno5',
    'Rouchon1',
    'Rouchon2',
    'Tsit5',
]


_TupleGradient = tuple[type[Gradient], ...]


# === generic solvers options
class Solver(eqx.Module):
    # should be eqx.AbstractClassVar, but this conflicts with the __future__ imports
    SUPPORTED_GRADIENT: ClassVar[_TupleGradient]

    @classmethod
    def supports_gradient(cls, gradient: Gradient | None) -> bool:
        return isinstance(gradient, cls.SUPPORTED_GRADIENT)

    @classmethod
    def assert_supports_gradient(cls, gradient: Gradient | None) -> None:
        if gradient is not None and not cls.supports_gradient(gradient):
            support_str = ', '.join(f'`{x.__name__}`' for x in cls.SUPPORTED_GRADIENT)
            raise ValueError(
                f'Solver `{cls.__name__}` does not support gradient'
                f' `{type(gradient).__name__}` (supported gradient types:'
                f' {support_str}).'
            )

    def __str__(self) -> str:
        return tree_str_inline(self)


# === expm solver options
class Expm(Solver):
    r"""Explicit matrix exponentiation to compute propagators.

    Explicitly batch-compute the propagators for all time intervals in `tsave`. These
    propagators are then iteratively applied:

    - starting from the initial state for [`dq.sesolve()`][dynamiqs.sesolve] and
      [`dq.mesolve()`][dynamiqs.mesolve], to compute states for all times in `tsave`,
    - starting from the identity matrix for [`dq.sepropagator()`][dynamiqs.sepropagator]
      and [`dq.mepropagator()`][dynamiqs.mepropagator], to compute propagators for all
      times in `tsave`.

    For the Schrödinger equation with constant Hamiltonian $H$, the propagator from
    time $t_0$ to time $t_1$ is an $n\times n$ matrix given by
    $$
        U(t_0, t_1) = \exp(-i (t_1 - t_0) H).
    $$

    For the Lindblad master equation with constant Liouvillian $\mathcal{L}$, the
    problem is vectorized and the propagator from time $t_0$ to time $t_1$ is an
    $n^2\times n^2$ matrix given by
    $$
        \mathcal{U}(t_0, t_1) = \exp((t_1 - t_0)\mathcal{L}).
    $$

    Warning:
        If the Hamiltonian or jump operators are sparse qarrays, they will be silently
        converted to dense qarrays before computing their matrix exponentials.

    Warning:
        This solver is not recommended for open systems of large dimension, due to
        the $\mathcal{O}(n^6)$ scaling of computing the Liouvillian exponential.

    Warning:
        This solver only supports constant or piecewise constant Hamiltonian and jump
        operators.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd,)

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass


# === generic ODE/SDE solvers options
class _DESolver(Solver):
    pass


class _DEFixedStep(_DESolver):
    dt: float


class _DEAdaptiveStep(_DESolver):
    rtol: float = 1e-6
    atol: float = 1e-6
    safety_factor: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    max_steps: int = 100_000


# === public solvers options
class Euler(_DEFixedStep):
    """Euler method (fixed step size ODE solver).

    This solver is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Euler`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Euler).

    Warning:
        This solver is not recommended for general use.

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)


class EulerMaruyama(_DEFixedStep):
    r"""Euler-Maruyama method (fixed step size SDE solver).

    For a fixed step size $\dt$, it has weak order of convergence $\dt$ and strong order
    of convergence $\sqrt{\dt}$.

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd,)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)


class Rouchon1(_DEFixedStep):
    """First-order Rouchon method (fixed step size ODE solver).

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)

    # normalize: The default scheme is trace-preserving at first order only. This
    # parameter sets the normalisation behaviour:
    # - `None`: The scheme is not normalized.
    # - `'sqrt'`: The Kraus map is renormalized with a matrix square root. Ideal
    #   for stiff problems, recommended for time-independent problems.
    # - `cholesky`: The Kraus map is renormalized at each time step using a Cholesky
    #   decomposition. Ideal for stiff problems, recommended for time-dependent
    #   problems.

    # TODO: fix, strings are not valid JAX types
    # normalize: Literal['sqrt', 'cholesky'] | None = None


class Rouchon2(_DEFixedStep):
    """Second-order Rouchon method (fixed step size ODE solver).

    Warning:
        This solver has not been ported to JAX yet.

    Args:
        dt: Fixed time step.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(self, dt: float):
        super().__init__(dt)


class Dopri5(_DEAdaptiveStep):
    """Dormand-Prince method of order 5 (adaptive step size ODE solver).

    This solver is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Dopri5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri5).

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Dopri8(_DEAdaptiveStep):
    """Dormand-Prince method of order 8 (adaptive step size ODE solver).

    This solver is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Dopri8`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Dopri8).

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Tsit5(_DEAdaptiveStep):
    """Tsitouras method of order 5 (adaptive step size ODE solver).

    This solver is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Tsit5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Tsit5).

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Kvaerno3(_DEAdaptiveStep):
    """Kvaerno's method of order 3 (adaptive step size and implicit ODE solver).

    This method is suitable for stiff problems, typically those with Hamiltonians or
    Liouvillians that have eigenvalues spanning different orders of magnitudes. This is
    for instance the case with problems involving high-order polynomials of the bosonic
    annihilation and creation operators, in large dimensions.

    This solver is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Kvaerno3`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno3).

    Warning:
        If you find that your simulation is slow or that the progress bar gets stuck,
        consider switching to double-precision with
        [`dq.set_precision('double')`][dynamiqs.set_precision]. See more details in
        [The sharp bits 🔪](../../documentation/getting_started/sharp-bits.md) tutorial.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)


class Kvaerno5(_DEAdaptiveStep):
    """Kvaerno's method of order 5 (adaptive step size and implicit ODE solver).

    This method is suitable for stiff problems, typically those with Hamiltonians or
    Liouvillians that have eigenvalues spanning different orders of magnitudes. This is
    for instance the case with problems involving high-order polynomials of the bosonic
    annihilation and creation operators, in large dimensions.

    This solver is implemented by the [Diffrax](https://docs.kidger.site/diffrax/)
    library, see [`diffrax.Kvaerno5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/#diffrax.Kvaerno5).

    Warning:
        If you find that your simulation is slow or that the progress bar gets stuck,
        consider switching to double-precision with
        [`dq.set_precision('double')`][dynamiqs.set_precision]. See more details in
        [The sharp bits 🔪](../../documentation/getting_started/sharp-bits.md) tutorial.

    Args:
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        safety_factor: Safety factor for adaptive step sizing.
        min_factor: Minimum factor for adaptive step sizing.
        max_factor: Maximum factor for adaptive step sizing.
        max_steps: Maximum number of steps.

    Note-: Supported gradients
        This solver supports differentiation with
        [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] and
        [`dq.gradient.CheckpointAutograd`][dynamiqs.gradient.CheckpointAutograd]
        (default).
    """

    SUPPORTED_GRADIENT: ClassVar[_TupleGradient] = (Autograd, CheckpointAutograd)

    # dummy init to have the signature in the documentation
    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        max_steps: int = 100_000,
    ):
        super().__init__(rtol, atol, safety_factor, min_factor, max_factor, max_steps)
