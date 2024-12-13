from __future__ import annotations

import warnings
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
from jax import Array
from jaxtyping import PyTree
from ...gradient import Autograd, CheckpointAutograd
from ...qarrays.utils import tree_sum
from .abstract_integrator import BaseIntegrator
from ...result import Result
from ...utils.general import dag
from .abstract_integrator import BaseIntegrator
from .save_mixin import SaveMixin
from .interfaces import SEInterface, MEInterface
from ...utils.vectorization import operator_to_vector, slindbladian

class DiffraxIntegrator(BaseIntegrator, SaveMixin):
    """Integrator using the Diffrax library."""

    # subclasses should implement: stepsize_controller, dt0, max_steps, diffrax_solver,
    # terms, discontinuity_ts, infos()

    @property
    @abstractmethod
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        pass

    @property
    @abstractmethod
    def dt0(self) -> float | None:
        pass

    @property
    @abstractmethod
    def max_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def diffrax_solver(self) -> dx.AbstractSolver:
        pass

    @property
    @abstractmethod
    def terms(self) -> dx.AbstractTerm:
        pass

    def run(self) -> Result:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)

            # === prepare diffrax arguments
            fn = lambda t, y, args: self.save(y)  # noqa: ARG005
            subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
            subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
            saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, CheckpointAutograd):
                adjoint = dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.DirectAdjoint()

            # === solve differential equation with diffrax
            y0 = operator_to_vector(self.y0)
            solution = dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )

        # === collect and return results
        saved = self.postprocess_saved(*solution.ys)
        return self.result(saved, infos=self.infos(solution.stats))

    @abstractmethod
    def infos(self, stats: dict[str, Array]) -> PyTree:
        pass


class FixedStepDiffraxIntegrator(DiffraxIntegrator):
    """Integrator using a fixed step Diffrax solver."""

    # subclasses should implement: diffrax_solver, terms, discontinuity_ts

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(stats['num_steps'])

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.ConstantStepSize()

    @property
    def dt0(self) -> float | None:
        return self.solver.dt

    @property
    def max_steps(self) -> int:
        return 100_000  # TODO: fix hard-coded max_steps


class AdaptiveStepDiffraxIntegrator(DiffraxIntegrator):
    """Integrator using an adaptive step Diffrax solver."""

    # subclasses should implement: diffrax_solver, terms, discontinuity_ts

    class Infos(eqx.Module):
        nsteps: Array
        naccepted: Array
        nrejected: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                return (
                    f'avg. {self.nsteps.mean():.1f} steps ({self.naccepted.mean():.1f}'
                    f' accepted, {self.nrejected.mean():.1f} rejected) | infos shape'
                    f' {self.nsteps.shape}'
                )
            return (
                f'{self.nsteps} steps ({self.naccepted} accepted,'
                f' {self.nrejected} rejected)'
            )

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(
            stats['num_steps'], stats['num_accepted_steps'], stats['num_rejected_steps']
        )

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
            jump_ts=self.discontinuity_ts,
        )

    @property
    def dt0(self) -> float | None:
        return None

    @property
    def max_steps(self) -> int:
        return self.solver.max_steps


# fmt: off
# ruff: noqa
class EulerIntegrator(FixedStepDiffraxIntegrator): diffrax_solver = dx.Euler()
class Dopri5Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Dopri5()
class Dopri8Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Dopri8()
class Tsit5Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Tsit5()
class Kvaerno3Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Kvaerno3()
class Kvaerno5Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Kvaerno5()
# fmt: on


class SEDiffraxIntegrator(DiffraxIntegrator, SEInterface):
    """Integrator solving the Schrödinger equation with Diffrax."""

    # subclasses should implement: diffrax_solver, discontinuity_ts

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class MEDiffraxIntegrator(DiffraxIntegrator, MEInterface):
    """Integrator solving the Lindblad master equation with Diffrax."""

    # subclasses should implement: diffrax_solver, discontinuity_ts

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation for a single loss channel is:
        # (1) drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        # An alternative but similar equation is:
        # (2) drho/dt = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        # While (1) and (2) are equivalent assuming that rho is hermitian, they differ
        # once you take into account numerical errors.
        # Decomposing rho = rho_s + rho_a with Hermitian rho_s and anti-Hermitian rho_a,
        # we get that:
        #  - if rho evolves according to (1), both rho_s and rho_a also evolve
        #    according to (1);
        #  - if rho evolves according to (2), rho_s evolves closely to (1) up
        #    to a constant error that depends on rho_a (which is small up to numerical
        #    precision), while rho_a is strictly constant.
        # In practice, we still use (2) because it involves less matrix multiplications,
        # and is thus more efficient numerically with only a negligible numerical error
        # induced on the dynamics.

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            return self.Liouv @ y

        return dx.ODETerm(vector_field)
