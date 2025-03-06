from __future__ import annotations

from typing import Literal

import jax

from ..progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter
from ..qarrays.layout import dense, dia, set_global_layout

__all__ = [
    'set_device',
    'set_layout',
    'set_matmul_precision',
    'set_precision',
    'set_progress_meter',
]

_DEFAULT_PROGRESS_METER: AbstractProgressMeter = TqdmProgressMeter()


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
        index: Index of the device to use, defaults to 0.
    """
    jax.config.update('jax_default_device', jax.devices(device)[index])


def set_precision(precision: Literal['single', 'double']):
    """Configure the default floating point precision.

    Two options are available:

    - `'single'` sets default precision to `float32` and `complex64` (default setting),
    - `'double'` sets default precision to `float64` and `complex128`.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        if precision == 'single':
            jax.config.update('jax_enable_x64', False)
        elif precision == 'double':
            jax.config.update('jax_enable_x64', True)
        ```
         See [JAX documentation on double precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

    Args:
        precision _(string 'single' or 'double')_: Default precision.
    """
    if precision == 'single':
        jax.config.update('jax_enable_x64', False)
    elif precision == 'double':
        jax.config.update('jax_enable_x64', True)
    else:
        raise ValueError(
            f"Argument `precision` should be a string 'single' or 'double', but is"
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

    Note:
        This setting applies only to single precision matrices (`float32` or
        `complex64`).

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
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]
        >>> dq.set_layout('dense')
        >>> dq.eye(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
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

    set_global_layout(layouts[layout])


def set_progress_meter(progress_meter: AbstractProgressMeter | bool):
    """Configure the default progress meter.

    Args:
        progress_meter: Default progress meter. Set to `True` for a [tqdm](https://github.com/tqdm/tqdm)
            progress meter, and `False` for no output. See other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
    """
    global _DEFAULT_PROGRESS_METER  # noqa: PLW0603

    if progress_meter is True:
        progress_meter = TqdmProgressMeter()
    elif progress_meter is False:
        progress_meter = NoProgressMeter()

    _DEFAULT_PROGRESS_METER = progress_meter


def get_progress_meter(
    progress_meter: AbstractProgressMeter | bool | None,
) -> AbstractProgressMeter:
    if progress_meter is None:
        return _DEFAULT_PROGRESS_METER
    elif progress_meter is True:
        return TqdmProgressMeter()
    elif progress_meter is False:
        return NoProgressMeter()
    return progress_meter
