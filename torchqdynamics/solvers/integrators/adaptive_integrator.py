from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from ...utils.solver_utils import hairer_norm
from ...utils.tensor_types import dtype_complex_to_real


class AdaptiveIntegrator(ABC):
    """A parent class for all adaptive time step ODE integrators.

    This performs all the necessary steps for the integration of an ODE of the form
    `dy/dt = f(t, y)` with initial condition `y(t0) = y0`. For details about the
    integration method, see Chapter II.4 of `Hairer et al., Solving Ordinary
    Differential Equations I (1993), Springer Series in Computational Mathematics`.
    """

    def __init__(
        self,
        f: Callable,
        factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        atol: float = 1e-8,
        rtol: float = 1e-6,
    ):
        self.f = f
        self.factor = factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.atol = atol
        self.rtol = rtol
        self.order = None
        self.tableau = None

    @abstractmethod
    def build_tableau(self, target: Tensor):
        """Build the Butcher tableau of the integrator."""
        pass

    @abstractmethod
    def step(
        self, f0: Tensor, y0: Tensor, t0: float, dt: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a single step of the ODE integration."""
        pass

    def get_error(self, y_err: Tensor, y0: Tensor, y1: Tensor) -> float:
        """Compute the error of a given solution.

        See Equation (4.11) of `Hairer et al., Solving Ordinary Differential Equations I
        (1993), Springer Series in Computational Mathematics`.
        """
        scale = self.atol + self.rtol * torch.max(y0.abs(), y1.abs())
        return hairer_norm(y_err / scale).max()

    def init_tstep(self, f0: Tensor, y0: Tensor, t0: float) -> float:
        """Initialize the time step of an adaptive step size integrator.

        See Equation (4.14) of `Hairer et al., Solving Ordinary Differential Equations I
        (1993), Springer Series in Computational Mathematics` for the detailed steps.
        For this function, we keep the same notations as in the book.
        """
        sc = self.atol + torch.abs(y0) * self.rtol
        d0, d1 = hairer_norm(y0 / sc).max(), hairer_norm(f0 / sc).max()

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * f0
        f1 = self.f(t0 + h0, y1)
        d2 = hairer_norm((f1 - f0) / sc).max() / h0
        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / float(self.order + 1))

        return min(100 * h0, h1)

    @torch.no_grad()
    def update_tstep(self, dt, error):
        """Update the time step of an adaptive step size integrator.

        See Equation (4.12) and (4.13) of `Hairer et al., Solving Ordinary Differential
        Equations I (1993), Springer Series in Computational Mathematics` for the
        detailed steps.
        """
        if error == 0:  # no error -> maximally increase the time step
            return dt * self.max_factor

        # optimal time step
        dt_opt = dt * error ** (-1.0 / self.order)

        if error <= 1:  # time step accepted -> take next time step at least as large
            return dt * min(self.max_factor, max(1.0, self.factor * dt_opt))

        if error > 1:  # time step rejected -> reduce next time step
            return dt * min(0.9, max(self.min_factor, self.factor * dt_opt))


class DormandPrince5(AdaptiveIntegrator):
    """Dormand-Prince method for adaptive time step ODE integration.

    This is a fifth order solver that uses a fourth order solution to estimate the
    integration error. It does so using only six function evaluations. See `Dormand and
    Prince, A family of embedded Runge-Kutta formulae (1980), Journal of Computational
    and Applied Mathematics`. See also `Shampine, Some Practical Runge-Kutta Formulas
    (1986), Mathematics of Computation`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = 5

    def build_tableau(self, target):
        """Build the Butcher tableau of the integrator."""
        alpha = [1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0, 0.0]
        beta = [
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
        csol5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
        csol4 = [
            1951 / 21600,
            0,
            22642 / 50085,
            451 / 720,
            -12231 / 42400,
            649 / 6300,
            1 / 60,
        ]

        # extract target information
        dtype = target.dtype
        float_dtype = dtype_complex_to_real(dtype)
        device = target.device

        # initialize tensors
        alpha = torch.tensor(alpha, dtype=float_dtype, device=device)
        beta = torch.tensor(beta, dtype=dtype, device=device)
        csol5 = torch.tensor(csol5, dtype=dtype, device=device)
        csol4 = torch.tensor(csol4, dtype=dtype, device=device)

        return alpha, beta, csol5, csol5 - csol4

    def step(
        self, f0: Tensor, y0: Tensor, t0: float, dt: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute a single step of the ODE integration."""
        # create butcher tableau if not already done
        if self.tableau is None:
            self.tableau = self.build_tableau(f0)

        # import butcher tableau
        alpha, beta, csol, cerr = self.tableau

        # compute iterated Runge-Kutta values
        k = torch.zeros(7, *f0.shape, dtype=f0.dtype, device=f0.device)
        k[0] = f0
        for i in range(1, 7):
            ti = t0 + dt * alpha[i - 1]
            yi = y0 + dt * torch.einsum('b,b...', beta[i - 1, :i], k[:i])
            k[i] = self.f(ti, yi)

        # compute results
        y1 = y0 + dt * torch.einsum('b,b...', csol[:6], k[:6])
        y1_err = dt * torch.einsum('b,b...', cerr, k)
        f1 = k[-1]
        return f1, y1, y1_err
