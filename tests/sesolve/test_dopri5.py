import pytest

from dynamiqs.gradient import Autograd
from dynamiqs.solver import Dopri5

from ..solver_tester import SolverTester
from .closed_system import cavity, tdqubit


class TestSEDopri5(SolverTester):
    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_correctness(self, system):
        self._test_correctness(system, Dopri5())

    @pytest.mark.parametrize('system', [cavity, tdqubit])
    def test_autograd(self, system):
        self._test_gradient(system, Dopri5(), Autograd())
