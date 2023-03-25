from math import sqrt
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .odeint import AdjointQSolver, odeint
from .solver import Rouchon, SolverOption
from .solver_utils import inv_sqrtm, kraus_map
from .types import TDOperator, TensorLike
from .utils import trace


def mesolve(
    H: TDOperator,
    jump_ops: List[Tensor],
    rho0: Tensor,
    t_save: TensorLike,
    *,
    save_states: bool = True,
    exp_ops: Optional[List[Tensor]] = None,
    solver: Optional[SolverOption] = None,
    gradient_alg: Optional[Literal['autograd', 'adjoint']] = None,
    parameters: Optional[Tuple[nn.Parameter, ...]] = None,
) -> Tuple[Tensor, Tensor]:
    """Solve the Lindblad master equation for a Hamiltonian and set of jump operators.

    The Hamiltonian `H` and the initial density matrix `rho0` can be batched over to
    solve multiple master equations in a single run. The jump operators `jump_ops` and
    time list `t_save` are common to all batches.

    `mesolve` can be differentiated through using either the default PyTorch autograd
    library (`gradient_alg="autograd"`), or a custom adjoint state differentiation
    (`gradient_alg="adjoint"`). For the latter, a solver that is stable in the backward
    pass should be used (e.g. Rouchon solver). By default (if no gradient is required),
    the graph of operations is not stored for improved performance of the solver.

    For time-dependent problems, the Hamiltonian `H` can be passed as a function with
    signature `H(t: float) -> Tensor`. Piecewise constant Hamiltonians can also be
    passed as... TODO: complete with Hamiltonian format

    Available solvers:
      - `Rouchon1` (alias of `Rouchon`)
      - `Rouchon1_5`
      - `Rouchon2`

    Args:
        H (Tensor or Callable): Hamiltonian.
            Can be a tensor of shape (n, n) or (b_H, n, n) if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=t_save[-1]`.
        jump_ops (list of Tensor): List of jump operators.
            Each jump operator should be a tensor of shape (n, n).
        rho0 (Tensor): Initial density matrix.
            Tensor of shape (n, n) or (b_rho, n, n) if batched.
        t_save (Tensor, np.ndarray or list): Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=t_save[-1]`.
        save_states (bool, optional): If `True`, the density matrix is saved at every
            time value in `t_save`. If `False`, only the final density matrix is
            stored and returned. Defaults to `True`.
        exp_ops (list of Tensor, optional): List of operators for which the expectation
            value is computed at every time value in `t_save`.
        solver (SolverOption, optional): Solver used to compute the master equation
            solutions. See the list of available solvers.
        gradient_alg (str, optional): Algorithm used for computing gradients in the
            backward pass. Defaults to `None`.
        parameters (tuple of nn.Parameter): Parameters with respect to which gradients
            are computed during the adjoint state backward pass.

    Returns:
        A tuple `(rho_save, exp_save)` where
            `rho_save` is a tensor with the computed density matrices at `t_save`
                times, and of shape (len(t_save), n, n) or (b_H, b_rho, len(t_save), n,
                n) if batched. If `save_states` is `False`, only the final density
                matrix is returned with the same shape as the initial input.
            `exp_save` is a tensor with the computed expectation values at `t_save`
                times, and of shape (len(exp_ops), len(t_save)) or (b_H, b_rho,
                len(exp_ops), len(t_save)) if batched.
    """
    # batch H by default
    H_batched = H[None, ...] if H.dim() == 2 else H

    if len(jump_ops) == 0:
        raise ValueError('Argument `jump_ops` must be a non-empty list of tensors.')
    jump_ops = torch.stack(jump_ops)

    # batch rho0 by default
    b_H = H_batched.size(0)
    rho0_batched = rho0[None, ...] if rho0.dim() == 2 else rho0
    rho0_batched = rho0_batched[None, ...].repeat(b_H, 1, 1, 1)  # (b_H, b_rho, n, n)

    t_save = torch.as_tensor(t_save)
    if exp_ops is None:
        exp_ops = torch.tensor([])
    else:
        exp_ops = torch.stack(exp_ops)
    if solver is None:
        # TODO: Replace by adaptive time step solver when implemented.
        solver = Rouchon(dt=1e-2)

    # define the QSolver
    if isinstance(solver, Rouchon):
        if solver.order == 1:
            qsolver = MERouchon1(H_batched, jump_ops, solver)
        elif solver.order == 1.5:
            qsolver = MERouchon1_5(H_batched, jump_ops, solver)
        elif solver.order == 2:
            qsolver = MERouchon2(H_batched, jump_ops, solver)
    else:
        raise NotImplementedError

    # compute the result
    rho_save, exp_save = odeint(
        qsolver, rho0_batched, t_save, exp_ops, save_states, gradient_alg
    )

    # restore correct batching
    if rho0.dim() == 2:
        rho_save = rho_save.squeeze(1)
        exp_save = exp_save.squeeze(1)
    if H.dim() == 2:
        rho_save = rho_save.squeeze(0)
        exp_save = exp_save.squeeze(0)

    return rho_save, exp_save


class MERouchon(AdjointQSolver):
    def __init__(self, H: TDOperator, jump_ops: Tensor, solver_options: SolverOption):
        # Args:
        #     H: (b_H, n, n)

        # convert H and jump_ops to sizes compatible with (b_H, len(jump_ops), n, n)
        self.H = H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops[None, ...]  # (1, len(jump_ops), n, n)
        self.sum_nojump = (jump_ops.adjoint() @ jump_ops).sum(dim=0)  # (n, n)
        n = H.shape[-1]
        self.I = torch.eye(n).to(H)  # (n, n)
        self.options = solver_options


class MERouchon1(MERouchon):
    def forward(self, t: float, dt: float, rho: Tensor):
        """Compute rho(t+dt) using a Rouchon method of order 1."""
        # Args:
        #     rho: (b_H, b_rho, n, n)
        #
        # Returns:
        #     (b_H, b_rho, n, n)

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh  # (b_H, 1, n, n)
        M1s = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, M0) + kraus_map(rho, M1s)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def forward_adjoint(self, t: float, dt: float, phi: Tensor):
        raise NotImplementedError


class MERouchon1_5(MERouchon):
    def forward(self, t: float, dt: float, rho: Tensor):
        """Compute rho(t+dt) using a Rouchon method of order 1.5."""
        # Args:
        #     rho: (b_H, b_rho, n, n)
        #
        # Returns:
        #     (b_H, b_rho, n, n)

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh  # (b_H, 1, n, n)
        Ms = sqrt(dt) * self.jump_ops  # (1, len(jump_ops), n, n)

        # build normalization matrix
        S = M0.adjoint() @ M0 + dt * self.sum_nojump  # (b_H, 1, n, n)
        # TODO: fix `inv_sqrtm` (size not compatible and linalg.solve RuntimeError)
        S_inv_sqrtm = inv_sqrtm(S)  # (b_H, 1, n, n)

        # compute rho(t+dt)
        rho = kraus_map(rho, S_inv_sqrtm)
        rho = kraus_map(rho, M0) + kraus_map(rho, Ms)

        # no need to normalize by the trace because this scheme is trace
        # preserving by construction

        return rho

    def forward_adjoint(self, t: float, dt: float, phi: Tensor):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def forward(self, t: float, dt: float, rho: Tensor):
        r"""Compute rho(t+dt) using a Rouchon method of order 2.

        Note:
            For fast time-varying Hamiltonians, this method is not order 2 because the
            second-order time derivative term is neglected. This term could be added in
            the zero-th order Kraus operator if needed, as `M0 += -0.5j * dt**2 *
            \dot{H}`.
        """
        # Args:
        #     rho: (b_H, b_rho, n, n)
        #
        # Returns:
        #     (b_H, b_rho, n, n)

        # non-hermitian Hamiltonian at time t
        H_nh = self.H - 0.5j * self.sum_nojump  # (b_H, 1, n, n)

        # build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh  # (b_H, 1, n, n)
        M1s = 0.5 * sqrt(dt) * (
            self.jump_ops @ M0 + M0 @ self.jump_ops
        )  # (b_H, len(jump_ops), n, n)

        # compute rho(t+dt)
        tmp = kraus_map(rho, M1s)
        rho = kraus_map(rho, M0) + tmp + 0.5 * kraus_map(tmp, M1s)

        # normalize by the trace
        rho = rho / trace(rho)[..., None, None].real

        return rho

    def forward_adjoint(self, t: float, dt: float, phi: Tensor):
        raise NotImplementedError
