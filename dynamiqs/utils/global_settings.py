from __future__ import annotations

from typing import Literal

import jax

from ..qarrays.layout import Layout, dense, dia

__all__ = ['set_device', 'set_precision', 'set_matmul_precision', 'set_layout']


def set_device(device: Literal['cpu', 'gpu', 'tpu']):
    """Configure the default device.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        jax.config.update('jax_default_device', jax.devices(device)[0])
        ```

    See [JAX documentation on devices](https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement).

    Args:
        device _(string 'cpu', 'gpu', or 'tpu')_: Default device.
    """
    jax.config.update('jax_default_device', jax.devices(device)[0])


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


_DEFAULT_LAYOUT = dia


def set_layout(layout: Literal['dense', 'dia']):
    """Configure the default matrix layout for operators supporting this option.

    Two layouts are supported by most operators (see the list of available operators in
    the [Python API](/python_api/index.html#operators))):

    - `'dense'`: JAX native dense layout,
    - `'dia'`: dynamiqs sparse diagonal layout, only non-zero diagonals are stored.

    Note:
        The default layout upon importing dynamiqs is `'dia'`.

    Args:
        layout _(string 'dense' or 'dia')_: Default matrix layout for operators.

    Examples:
        >>> dq.eye(4)
        SparseDIAQArray: shape=(4, 4), dims=(4,), dtype=complex64, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]
        >>> dq.set_layout('dense')
        >>> dq.eye(4)
        DenseQArray: shape=(4, 4), dims=(4,), dtype=complex64
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
        >>> dq.set_layout('dia')  # back to default layout
    """
    layouts = {'dense': dense, 'dia': dia}
    if layout not in layouts:
        raise ValueError(
            f"Argument `layout` should be a string 'dense' or 'dia', but is {layout}."
        )

    global _DEFAULT_LAYOUT  # noqa: PLW0603
    _DEFAULT_LAYOUT = layouts[layout]


def get_layout(layout: Layout | None = None) -> Layout:
    if layout is None:
        return _DEFAULT_LAYOUT
    elif isinstance(layout, Layout):
        return layout
    else:
        raise TypeError(
            'Argument `layout` must be `dq.dense`, `dq.dia` or `None`, but is'
            f' `{layout}`.'
        )