from __future__ import annotations

from collections.abc import Iterable

import jax.numpy as jnp
from jax import Array

__all__ = [
    'format_pulse_param',
    'format_pulse_params',
    'flat_envelope',
    'raised_cosine_envelope',
    'raised_cosine_drag_envelope',
]

FloatOrComplex = float | complex
PulseParamType = float | complex | Array


def format_pulse_param(parameter: PulseParamType) -> Array:
    """Format the pulse parameter into a one-dimensional JAX array.

    Args:
        parameter (PulseParamType): Parameter of the pulse, either a float, a complex,
            or a 1D array.

    Raises:
        ValueError: If the parameter has more than one dimension, i.e. we need the pulse
            parameters to be either a constant or a one-dimensional array on which we
            will batch over.

    Returns:
        _(array of shape (n))_ Formatted pulse parameter, where `n = len(parameter)`.
    """
    if not isinstance(parameter, FloatOrComplex) and len(parameter.shape) > 1:
        raise ValueError(
            'Parameter needs to be a 1D array or a float, but got shape '
            f'{parameter.shape}.'
        )
    return jnp.atleast_1d(parameter)


def format_pulse_params(parameters: Iterable[PulseParamType]) -> list[Array]:
    """Transforms N parameters into unsqueezed shape (len(param), 1, ..., 1) for
    explicit vectorization. Batch the dimensions according to the sequential order of
    the provided parameters.

    Args:
        parameters (Iterable[PulseParamType]): Parameters of the pulse.
            Each of them is either a float, a complex, or a 1D array.

    Returns:
        list[Array]: Parameters with expanded dimensions for vectorization.

    Examples:
        >>> parameters = [jnp.arange(3), 1.2, jnp.arange(7)]
        >>> [param.shape for param in format_pulse_params(parameters)]
        [(3, 1, 1), (1, 1, 1), (1, 1, 7)]

    """
    parameters = [format_pulse_param(param) for param in parameters]
    N_params = len(parameters)
    return [
        jnp.expand_dims(param, list(range(idx)) + list(range(idx + 1, N_params)))
        for idx, param in enumerate(parameters)
    ]


def _flat_envelope(t: float | Array, /, pad_times: Array, hold_times: Array) -> Array:
    nonzero_times = (t > pad_times) & (t <= pad_times + hold_times)
    return jnp.where(nonzero_times, 1, 0)


def flat_envelope(
    t: float | Array, /, pad_times: PulseParamType, hold_times: PulseParamType
) -> Array:
    """Returns the flat pulse envelopes at time(s) `t`, potentially batching over the
    pulse parameters (`pad_times`, `hold_times`). Since this is only an envelope, the
    amplitude is exactly `1.0` during the hold times and `0.0` otherwise.

    Args:
        t (float | Array): Time at which to evaluate the envelope function.
        pad_times (PulseParamType): Zero paddings before and after the hold times.
        hold_times (PulseParamType): Durations of the flat pulse.

    Returns:
        _(array of shape (t?, pad_times?, hold_times?))_ Pulse envelopes.
    """
    t, pad_times, hold_times = format_pulse_params([t, pad_times, hold_times])
    return jnp.squeeze(_flat_envelope(t, pad_times=pad_times, hold_times=hold_times))


def _raised_cosine_envelope(
    t: float | Array, /, gate_times: Array, carrier_freqs: Array, carrier_phases: Array
) -> Array:
    return (
        (1 - jnp.cos(2 * jnp.pi / gate_times * t))
        / 2
        * jnp.cos(carrier_freqs * t + carrier_phases)
    )


def raised_cosine_envelope(
    t: float | Array,
    /,
    gate_times: PulseParamType,
    carrier_freqs: PulseParamType = 0.0,
    carrier_phases: PulseParamType = 0.0,
) -> Array:
    """Returns the cosine pulse envelopes at time(s) `t`, potentially batching over the
    pulse parameters (`gate_times`, `carrier_freqs`, `carrier_phases`). The envelope
    starts and end at 0 amplitude at t=0.0 and t=gate_time, and reaches a maximal
    amplitude of `1.0`.

    Args:
        t (float | Array): Time at which to evaluate the envelope function.
        gate_times (PulseParamType): Total durations of the gate.
        carrier_freqs (PulseParamType): Carrier frequencies of the pulse, i.e. the
            frequencies of the fast oscillations inside the smooth cosine envelope.
            Defaults to 0.0 which corresponds to getting only the smooth envelope.
        carrier_phases (PulseParamType): Phases of the carrier at t=0.0.

    Returns:
        _(array of shape (t?, gate_times?, carrier_freqs?, carrier_phases?))_ Pulse
            envelopes.
    """
    t, gate_times, carrier_freqs, carrier_phases = format_pulse_params(
        [t, gate_times, carrier_freqs, carrier_phases]
    )
    return jnp.squeeze(
        _raised_cosine_envelope(
            t,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
        )
    )


def _raised_cosine_drag_envelope(
    t: float | Array,
    /,
    gate_times: Array,
    carrier_freqs: Array,
    carrier_phases: Array,
    drag_params: Array,
) -> Array:
    in_phase_envelope = (
        (1 - jnp.cos(2 * jnp.pi / gate_times * t))
        / 2
        * jnp.cos(carrier_freqs * t + carrier_phases)
    )
    quadrature_envelope = (
        drag_params
        * jnp.sin(2 * jnp.pi / gate_times * t)
        * (jnp.pi / gate_times)
        * jnp.sin(carrier_freqs * t + carrier_phases)
    )
    return in_phase_envelope + quadrature_envelope


def raised_cosine_drag_envelope(
    t: float | Array,
    /,
    gate_times: PulseParamType,
    carrier_freqs: PulseParamType = 0.0,
    carrier_phases: PulseParamType = 0.0,
    drag_params: PulseParamType = 0.0,
) -> Array:
    """Same as `raised_cosine_envelope`, but with a "quadrature" DRAG pulse, commonly
    used to minimize leakage in single-qubit gates. See https://arxiv.org/abs/0901.0534
    for more details.

    Args:
        t (float | Array): Time at which to evaluate the envelope function.
        gate_times (PulseParamType): Total durations of the gate.
        carrier_freqs (PulseParamType): Carrier frequencies of the pulse, i.e. the
            frequencies of the fast oscillations inside the smooth cosine envelope.
            Defaults to 0.0 which corresponds to getting only the smooth envelope.
        carrier_phases (PulseParamType): Phases of the carrier at t=0.0.
        drag_params (PulseParamType): DRAG parameters, i.e. the amplitude of the
            quadrature part of the signal, which can be fine tuned to minimize leakage
            of phase errors.

    Returns:
        _(array of shape (t?, gate_times?, carrier_freqs?, carrier_phases?))_ Pulse
            envelopes.
    """
    t, gate_times, carrier_freqs, carrier_phases, drag_params = format_pulse_params(
        [t, gate_times, carrier_freqs, carrier_phases, drag_params]
    )
    return jnp.squeeze(
        _raised_cosine_drag_envelope(
            t,
            gate_times=gate_times,
            carrier_freqs=carrier_freqs,
            carrier_phases=carrier_phases,
            drag_params=drag_params,
        )
    )
