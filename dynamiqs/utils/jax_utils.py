from __future__ import annotations

from typing import Literal

import jax
import numpy as np
from jax.typing import ArrayLike
from qutip import Qobj

from .._checks import check_shape
from .utils import isbra, isket, isop
from .utils.general import _hdim

__all__ = ['to_qutip', 'set_device', 'set_precision', 'set_matmul_precision']


def to_qutip(x: ArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert an array-like object into a QuTiP quantum object (or a list of QuTiP
    quantum objects if it has more than two dimensions).

    Args:
        x _(array_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra,
            density matrix or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the large
            Hilbert space of the composite system, defaults to `None` (a single system
            with the same dimension as `x`).

    Returns:
        QuTiP quantum object or list of QuTiP quantum objects.

    Examples:
        >>> psi = dq.fock(3, 1)
        >>> psi
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_qutip(psi)
        Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        For a batched array:
        >>> rhos = jnp.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        (5, 16, 16)
        >>> len(dq.to_qutip(rhos))
        5

        Note that the tensor product structure is not inferred automatically, it must be
        specified with the `dims` argument:
        >>> I = dq.eye(3, 2)
        >>> dq.to_qutip(I)
        Quantum object: dims = [[6], [6]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
        >>> dq.to_qutip(I, (3, 2))
        Quantum object: dims = [[3, 2], [3, 2]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    x = np.asarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    if x.ndim > 2:
        return [to_qutip(sub_x) for sub_x in x]
    else:
        dims = [_hdim(x)] if dims is None else list(dims)
        if isket(x):  # [[3], [1]] or for composite systems [[3, 4], [1, 1]]
            dims = [dims, [1] * len(dims)]
        elif isbra(x):  # [[1], [3]] or for composite systems [[1, 1], [3, 4]]
            dims = [[1] * len(dims), dims]
        elif isop(x):  # [[3], [3]] or for composite systems [[3, 4], [3, 4]]
            dims = [dims, dims]
        return Qobj(x, dims=dims)


def set_device(device: Literal['cpu', 'gpu', 'tpu'], index: int = 0):
    """Configure the default device.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        jax.config.update('jax_default_device', jax.devices(device)[index])
        ```

    See [JAX documentation on devices](https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement).

    Args:
        device _(string 'cpu', 'gpu', or 'tpu')_: Default device.
        index: Index of the device, useful for clusters with multiple GPUs.
    """
    jax.config.update('jax_default_device', jax.devices(device)[index])


def set_precision(precision: Literal['simple', 'double']):
    """Configure the default floating point precision.

    Two options are available:

    - `'simple'` sets default precision to `float32` and `complex64` (default setting),
    - `'double'` sets default precision to `float64` and `complex128`.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        if precision == 'simple':
            jax.config.update('jax_enable_x64', False)
        elif precision == 'double':
            jax.config.update('jax_enable_x64', True)
        ```
         See [JAX documentation on double precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

    Args:
        precision _(string 'simple' or 'double')_: Default precision.
    """
    if precision == 'simple':
        jax.config.update('jax_enable_x64', False)
    elif precision == 'double':
        jax.config.update('jax_enable_x64', True)
    else:
        raise ValueError(
            f"Argument `precision` should be a string 'simple' or 'double', but is"
            f" '{precision}'."
        )


def set_matmul_precision(matmul_precision: Literal['low', 'high', 'highest']):
    """Configure the default precision for matrix multiplications on GPUs and TPUs.

    Some devices allow trading off accuracy for speed when performing matrix
    multiplications (matmul). Three options are available:

    - `'low'` reduces matmul precision to `bfloat16` (fastest but least accurate),
    - `'high'` reduces matmul precision to `bfloat16_3x` or `tensorfloat32` if available
        (faster but less accurate),
    - `'highest'` keeps matmul precision to `float32` or `float64` as applicable
        (slowest but most accurate, default setting).

    Note-: Equivalent JAX syntax
        This function is equivalent to setting `jax_default_matmul_precision` in
        `jax.config`. See [JAX documentation on matmul precision](https://jax.readthedocs.io/en/latest/_autosummary/jax.default_matmul_precision.html)
        and [JAX documentation on the different available options](https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.Precision).

    Args:
        matmul_precision _(string 'low', 'high', or 'highest')_: Default precision
            for matrix multiplications on GPUs and TPUs.
    """
    if matmul_precision == 'low':
        jax.config.update('jax_default_matmul_precision', 'fastest')
    elif matmul_precision == 'high':
        jax.config.update('jax_default_matmul_precision', 'high')
    elif matmul_precision == 'highest':
        jax.config.update('jax_default_matmul_precision', 'highest')
    else:
        raise ValueError(
            f"Argument `matmul_precision` should be a string 'low', 'high', or"
            f" 'highest', but is '{matmul_precision}'."
        )
