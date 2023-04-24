from torch import Tensor

<<<<<<< HEAD
from ..ode.forward_solver import ForwardSolver
=======
from ..ode.ode_forward_solver import ODEForwardSolver
>>>>>>> 78bc0c8 (Reorganize main folders)
from ..utils.solver_utils import lindbladian


class MEEuler(ForwardSolver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops  # (len(jump_ops), n, n)

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # Args:
        #     rho: (b_H, b_rho, n, n)
        #
        # Returns:
        #     (b_H, b_rho, n, n)

        return rho + self.options.dt * lindbladian(rho, self.H, self.jump_ops)
