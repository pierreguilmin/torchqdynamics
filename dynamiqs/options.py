from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter
from .qarrays.qarray import QArray

__all__ = ['Options']


class Options(eqx.Module):
    """Generic options for quantum solvers.

    The `Options` class provides a unified interface for specifying options specific to
    quantum solvers. Each quantum solver may only use a subset of these options,
    and an error will be raised if you modify an option that the solver does not
    support.

    === "`dq.sesolve()`"
        ```python
        Options(
            save_states: bool = True,
            cartesian_batching: bool = True,
            progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
            t0: ScalarLike | None = None,
            save_extra: callable[[Array], PyTree] | None = None,
        )
        ```

        **Parameters**

        - **save_states** – If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **progress_meter** – Progress meter indicating how far the solve has
            progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress
            meter. Pass `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        - **t0** – Initial time. If `None`, defaults to the first time in `tsave`.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.

    === "`dq.mesolve()`"
        ```python
        Options(
            save_states: bool = True,
            cartesian_batching: bool = True,
            progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
            t0: ScalarLike | None = None,
            save_extra: callable[[Array], PyTree] | None = None,
        )
        ```

        **Parameters**

        - **save_states** – If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **progress_meter** – Progress meter indicating how far the solve has
            progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress
            meter. Pass `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        - **t0** – Initial time. If `None`, defaults to the first time in `tsave`.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.

    === "`dq.sepropagator()`"
        ```python
        Options(
            save_propagators: bool = True,
            progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
            t0: ScalarLike | None = None,
            save_extra: callable[[Array], PyTree] | None = None,
        )
        ```

        **Parameters**

        - **save_propagators** – If `True`, the propagator is saved at every time in
            `tsave`, otherwise only the final propagator is returned.
        - **progress_meter** – Progress meter indicating how far the solve has
            progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress
            meter. Pass `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        - **t0** – Initial time. If `None`, defaults to the first time in `tsave`.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.

    === "`dq.mepropagator()`"
        ```python
        Options(
            save_propagators: bool = True,
            cartesian_batching: bool = True,
            t0: ScalarLike | None = None,
            save_extra: callable[[Array], PyTree] | None = None,
        )
        ```

        **Parameters**

        - **save_propagators** – If `True`, the propagator is saved at every time in
            `tsave`, otherwise only the final propagator is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **t0** – Initial time. If `None`, defaults to the first time in `tsave`.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.

    === "`dq.floquet()`"
        ```python
        Options(
            progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
            t0: ScalarLike | None = None,
        )
        ```

        **Parameters**

        - **progress_meter** – Progress meter indicating how far the solve has
            progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress
            meter. Pass `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        - **t0** – Initial time. If `None`, defaults to the first time in `tsave`.

    === "`dq.jssesolve()`"
        ```python
        Options(
            save_states: bool = True,
            cartesian_batching: bool = True,
            save_extra: callable[[Array], PyTree] | None = None,
            nmaxclick: int = 10_000,
            smart_sampling: bool = False,
        )
        ```

        **Parameters**

        - **save_states** – If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.
        - **nmaxclick** - Maximum buffer size for `result.clicktimes`, should be set
            higher than the expected maximum number of jump event.
        - **smart_sampling** - If `True`, the no jump trajectory is simulated only once,
            and only trajectories with one or more jumps are sampled in `result.states`.
            The no jump state is accessible in `result.no_jump_state` with its
            associated probability `result.no_jump_proba`.

    === "`dq.dssesolve()`"
        ```python
        Options(
            save_states: bool = True,
            cartesian_batching: bool = True,
            save_extra: callable[[Array], PyTree] | None = None,
        )
        ```

        **Parameters**

        - **save_states** – If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.


    === "`dq.jsmesolve()`"
        ```python
        Options(
            save_states: bool = True,
            cartesian_batching: bool = True,
            save_extra: callable[[Array], PyTree] | None = None,
            nmaxclick: int = 10_000,
        )
        ```

        **Parameters**

        - **save_states** – If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.
        - **nmaxclick** - Maximum buffer size for `result.clicktimes`, should be set
            higher than the expected maximum number of jump event.

    === "`dq.dsmesolve()`"
        ```python
        Options(
            save_states: bool = True,
            cartesian_batching: bool = True,
            save_extra: callable[[Array], PyTree] | None = None,
        )
        ```

        **Parameters**

        - **save_states** – If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        - **cartesian_batching** – If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        - **save_extra** _(function, optional)_ – A function with signature
            `f(QArray) -> PyTree` that takes a state or propagator as input and returns
            a PyTree. This can be used to save additional arbitrary data during the
            integration. The additional data is accessible in the `extra` attribute of
            the result object returned by the solvers.
    """  # noqa: RUF002

    save_states: bool = True
    save_propagators: bool = True
    cartesian_batching: bool = True
    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter()
    t0: ScalarLike | None = None
    save_extra: callable[[QArray], PyTree] | None = None
    nmaxclick: int = 10_000
    smart_sampling: bool = False

    def __init__(
        self,
        save_states: bool = True,
        save_propagators: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),  # noqa: B008
        t0: ScalarLike | None = None,
        save_extra: callable[[QArray], PyTree] | None = None,
        nmaxclick: int = 10_000,
        smart_sampling: bool = False,
    ):
        if progress_meter is None:
            progress_meter = NoProgressMeter()

        self.save_states = save_states
        self.save_propagators = save_propagators
        self.cartesian_batching = cartesian_batching
        self.progress_meter = progress_meter
        self.t0 = t0
        self.nmaxclick = nmaxclick
        self.smart_sampling = smart_sampling

        # make `save_extra` a valid Pytree with `Partial`
        self.save_extra = jtu.Partial(save_extra) if save_extra is not None else None

    def __str__(self) -> str:
        return tree_str_inline(self)


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
            'save_extra',
            'smart_sampling',
        ),
        'dssesolve': ('save_states', 'cartesian_batching', 'save_extra'),
        'jsmesolve': ('save_states', 'cartesian_batching', 'save_extra'),
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
                f'the quantum solver `dq.{solver_name}()` (valid options: '
                f'{valid_options_str}).'
            )
