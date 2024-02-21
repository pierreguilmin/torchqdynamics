from __future__ import annotations

import equinox as eqx
from jaxtyping import Scalar

__all__ = ['Options']


class Options(eqx.Module):
    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: Scalar | None = None

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        t0: Scalar | None = None,
    ):
        """Generic options for the quantum solvers.

        Args:
            save_states: If `True`, the state is saved at every time in `tsave`,
                otherwise only the final state is returned.
            verbose: If `True`, print information about the integration, otherwise
                nothing is printed.
            cartesian_batching: If `True`, batched arguments are treated as separated
                batch dimensions, otherwise the batching is performed over a single
                shared batched dimension.
            t0: Initial time. If `None`, defaults to the first time in `tsave`.
            save_extra _(function, optional)_: A function with signature
                `f(Array) -> PyTree` that takes a state as input and returns a PyTree.
                This can be used to save additional arbitrary data during the
                integration.
        """
        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.t0 = t0
