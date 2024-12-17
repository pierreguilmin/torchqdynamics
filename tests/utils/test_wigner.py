import jax.numpy as jnp

import dynamiqs as dq


def test_wigner_cat():
    # parameters
    n = 16
    alpha = 2.0
    npixels = 101
    xmax, ymax = 5, 3

    # state to test
    cat = dq.unit(dq.coherent(n, alpha) + dq.coherent(n, -alpha))

    # reference expectation value
    nbar = dq.expect(dq.number(n), cat).real

    # compute wigner and expectation value
    x_vec, y_vec, wig = dq.wigner(cat, npixels=npixels, xmax=xmax, ymax=ymax)
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    x_vec, y_vec = jnp.meshgrid(x_vec, y_vec, indexing='ij')
    nbar_vec = x_vec**2 + y_vec**2 - 0.5
    nbar_wig = jnp.sum(wig * nbar_vec) * dx * dy

    assert jnp.allclose(nbar, nbar_wig)


def test_wigner_coherent():
    # parameters
    n = 16
    alpha = 2.0
    npixels = 101
    xmax, ymax = 5, 5

    # state to test
    coh = dq.coherent(n, alpha)

    # reference expectation value
    a = dq.expect(dq.destroy(n), coh)

    # compute wigner and expectation value
    x_vec, y_vec, wig = dq.wigner(coh, npixels=npixels, xmax=xmax, ymax=ymax)
    dx, dy = x_vec[1] - x_vec[0], y_vec[1] - y_vec[0]
    x_vec, y_vec = jnp.meshgrid(x_vec, y_vec, indexing='ij')
    a_vec = x_vec + 1j * y_vec
    a_wig = jnp.sum(wig * a_vec) * dx * dy

    assert jnp.allclose(a, a_wig)
