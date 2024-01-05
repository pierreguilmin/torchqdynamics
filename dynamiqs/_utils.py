from __future__ import annotations

from typing import Any

from jax import numpy as jnp
from jaxtyping import Array

from .utils import dag, isket


def type_str(type: Any) -> str:
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def split_complex(x: Array) -> Array:
    return jnp.stack((x.real, x.imag), axis=-1)


def merge_complex(x: Array) -> Array:
    return x[..., 0] + 1j * x[..., 1]


def check_time_array(x: Array, arg_name: str, allow_empty=False):
    # check that a time array is valid (it must be a 1D array sorted in strictly
    # ascending order and containing only positive values)
    if x.ndim != 1:
        raise ValueError(
            f'Argument `{arg_name}` must be a 1D array, but is a {x.ndim}D array.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument `{arg_name}` must contain at least one element.')
    if not jnp.all(jnp.diff(x) > 0):
        raise ValueError(
            f'Argument `{arg_name}` must be sorted in strictly ascending order.'
        )
    if not jnp.all(x >= 0):
        raise ValueError(f'Argument `{arg_name}` must contain positive values only.')


def bexpect(O: Array, x: Array) -> Array:
    # batched over O
    if isket(x):
        return jnp.einsum('ij,...jk,kl->...', dag(x), O, x)  # <x|O|x>
    return jnp.einsum('...ij,ji->...', O, x)  # tr(Ox)


def save_fn(_t, y, args):
    options, exp_ops = args
    res = {}
    if options.save_states:
        res['states'] = y
    if options.save_expects:
        res['expects'] = bexpect(exp_ops, y)
    return res
