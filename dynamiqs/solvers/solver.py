from __future__ import annotations

from abc import ABC, abstractmethod
from time import time

import torch
from torch import Tensor

from .options import Options
from .result import Result
from .utils.td_tensor import TDTensor
from .utils.utils import bexpect


class Solver(ABC):
    def __init__(
        self,
        H: TDTensor,
        y0: Tensor,
        t_save: Tensor,
        exp_ops: Tensor,
        options: Options,
    ):
        """

        Args:
            H:
            y0: Initial quantum state, of shape `(..., m, n)`.
            t_save: Times for which results are saved.
            exp_ops:
            options:
        """
        self.H = H
        self.y0 = y0
        self.t_save = t_save
        self.exp_ops = exp_ops
        self.options = options

        # aliases
        self.cdtype = self.options.cdtype
        self.rdtype = self.options.rdtype
        self.device = self.options.device

        # initialize saving logic
        self._init_time_logic()

        # initialize save tensors
        batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]

        if self.options.save_states:
            # y_save: (..., len(t_save), m, n)
            y_save = torch.zeros(
                *batch_sizes, len(t_save), m, n, dtype=self.cdtype, device=self.device
            )
        else:
            y_save = None

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(t_save))
            exp_save = torch.zeros(
                *batch_sizes,
                len(exp_ops),
                len(t_save),
                dtype=self.cdtype,
                device=self.device,
            )
        else:
            exp_save = None

        self.result = Result(options, y_save, t_save, exp_save)

    def _init_time_logic(self):
        self.t_stop = self.t_save
        self.t_stop_counter = 0

        self.t_save_mask = torch.isin(self.t_stop, self.t_save)
        self.t_save_counter = 0

    def run(self):
        self.result.start_time = time()
        self._run()
        self.result.end_time = time()

    def _run(self):
        if self.options.gradient_alg is None:
            self.run_nograd()

    @abstractmethod
    def run_nograd(self):
        pass

    def next_t_stop(self) -> float:
        return self.t_stop[self.t_stop_counter].item()

    def save(self, y: Tensor):
        self._save(y)
        self.t_stop_counter += 1

    def _save(self, y: Tensor):
        if self.t_save_mask[self.t_stop_counter]:
            self._save_y(y)
            self._save_exp_ops(y)
            self.t_save_counter += 1

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            self.result.y_save[..., self.t_save_counter, :, :] = y
        # otherwise only save the state if it is the final state
        elif self.t_save_counter == len(self.t_save) - 1:
            self.result.y_save = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            self.result.exp_save[..., self.t_save_counter] = bexpect(self.exp_ops, y)


class AutogradSolver(Solver):
    def _run(self):
        super()._run()
        if self.options.gradient_alg == 'autograd':
            self.run_autograd()

    def run_nograd(self):
        with torch.inference_mode():
            self.run_autograd()

    @abstractmethod
    def run_autograd(self):
        pass


class AdjointSolver(AutogradSolver):
    def _run(self):
        super()._run()
        if self.options.gradient_alg == 'adjoint':
            self.run_adjoint()

    @abstractmethod
    def run_adjoint(self):
        """Integrate an ODE using the adjoint method in the backward pass."""
        pass
