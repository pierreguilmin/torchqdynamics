import random

import jax.numpy as jnp
import pytest

import dynamiqs as dq


class TestSparseDIA:
    @pytest.fixture(autouse=True)
    def _setup(self):
        N = 10
        num_diags = 3
        diags = jnp.arange(num_diags * N).reshape(num_diags, N)
        offsets = tuple(range(-(N // 2), (N // 2) + 1))

        self.matrixA = jnp.zeros((N, N))
        self.matrixB = jnp.zeros((N, N))

        for offset, diag in zip(offsets, diags):
            self.matrixA = self.matrixA.at[:].add(
                jnp.diag(diag[: N - abs(offset)], k=offset)
            )
            self.matrixB = self.matrixB.at[:].add(
                jnp.diag(diag[: N - abs(offset)], k=offset)
            )

        self.sparseA = dq.to_sparse(self.matrixA)
        self.sparseB = dq.to_sparse(self.matrixB)

    def test_convert(self, rtol=1e-05, atol=1e-08):
        assert jnp.allclose(
            self.matrixA, dq.to_sparse(self.matrixA).to_dense(), rtol=rtol, atol=atol
        )

    def test_add(self, rtol=1e-05, atol=1e-08):
        out_dia_dia = (self.sparseA + self.sparseB).to_dense()
        out_dia_dense = self.sparseA + self.matrixB
        out_dense_dia = self.matrixA + self.sparseB
        out_dense_dense = self.matrixA + self.matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_sub(self, rtol=1e-05, atol=1e-08):
        out_dia_dia = (self.sparseA - self.sparseB).to_dense()
        out_dia_dense = self.sparseA - self.matrixB
        out_dense_dia = self.matrixA - self.sparseB
        out_dense_dense = self.matrixA - self.matrixB

        assert jnp.allclose(out_dense_dense, out_dia_dia, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dia_dense, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_dense, out_dense_dia, rtol=rtol, atol=atol)

    def test_mul(self, rtol=1e-05, atol=1e-08):
        random_float = random.uniform(1.0, 10.0)

        out_dense_left = random_float * self.matrixA
        out_dense_right = self.matrixA * random_float
        out_dia_left = (random_float * self.sparseA).to_dense()
        out_dia_right = (self.sparseA * random_float).to_dense()

        assert jnp.allclose(out_dense_left, out_dense_right, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_left, out_dia_left, rtol=rtol, atol=atol)
        assert jnp.allclose(out_dense_left, out_dia_right, rtol=rtol, atol=atol)
