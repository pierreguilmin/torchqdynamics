from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter
from .qarrays.qarray import QArray
from .utils.global_settings import get_progress_meter

__all__ = ['Options']


class Options(eqx.Module):
    save_states: bool = True
    save_propagators: bool = True
    cartesian_batching: bool = True
    progress_meter: AbstractProgressMeter | bool | None = None
    t0: ScalarLike | None = None
    save_extra: callable[[QArray], PyTree] | None = None
    nmaxclick: int = 10_000

    def __init__(
        self,
        save_states: bool = True,
        save_propagators: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | bool | None = None,
        t0: ScalarLike | None = None,
        save_extra: callable[[QArray], PyTree] | None = None,
        nmaxclick: int = 10_000,
    ):
        self.save_states = save_states
        self.save_propagators = save_propagators
        self.cartesian_batching = cartesian_batching
        self.progress_meter = progress_meter
        self.t0 = t0
        self.nmaxclick = nmaxclick

        # make `save_extra` a valid Pytree with `Partial`
        self.save_extra = jtu.Partial(save_extra) if save_extra is not None else None

    def __str__(self) -> str:
        return tree_str_inline(self)

    def initialise(self) -> Options:
        # We need to call this before entering JIT-compiled functions. Why? Because
        # `progress_meter` is defined at runtime by the default value set in the global
        # settings. Now things become a bit tricky:
        # - We can't get the default value to set it in the `__init__`, because the
        #   default argument to many functions is `Options()`, so it would be set
        #   forever to the default `progress_meter` value at the time of the function
        #   import.
        # - A simple workaround would be to use a property to get the `progress_meter`
        #   dynamically, but then changing the default value would not change the
        #   `options` object attributes, and we would cache hit the JIT-compiled
        #   function for any previous existing `options` object.
        return Options(
            save_states=self.save_states,
            save_propagators=self.save_propagators,
            cartesian_batching=self.cartesian_batching,
            progress_meter=get_progress_meter(self.progress_meter),
            t0=self.t0,
            save_extra=self.save_extra,
            nmaxclick=self.nmaxclick,
        )


def check_options(options: Options, solver_name: str):
    supported_options = {
        'sesolve': (
            'save_states',
            'cartesian_batching',
            'progress_meter',
            't0',
            'save_extra',
        ),
        'mesolve': (
            'save_states',
            'cartesian_batching',
            'progress_meter',
            't0',
            'save_extra',
        ),
        'sepropagator': ('save_propagators', 'progress_meter', 't0', 'save_extra'),
        'mepropagator': ('save_propagators', 'cartesian_batching', 't0', 'save_extra'),
        'floquet': ('progress_meter', 't0'),
        'jssesolve': (
            'save_states',
            'cartesian_batching',
            't0',
            'save_extra',
            'nmaxclick',
        ),
        'dssesolve': ('save_states', 'cartesian_batching', 'save_extra'),
        'jsmesolve': ('save_states', 'cartesian_batching', 'save_extra', 'nmaxclick'),
        'dsmesolve': ('save_states', 'cartesian_batching', 'save_extra'),
    }
    valid_options = supported_options[solver_name]

    # check that all attributes are set to their default values except for the ones
    # specified in `valid_options`
    for key, value in options.__dict__.items():
        if key not in valid_options and value != getattr(Options(), key):
            valid_options_str = ', '.join(f'`{x}`' for x in valid_options)
            raise ValueError(
                f'Option `{key}` was set to `{value}` but is not used by '
                f'the solver `dq.{solver_name}()` (valid options: '
                f'{valid_options_str}).'
            )
