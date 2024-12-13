from __future__ import annotations

import numpy as np

from .._checks import check_shape
from ..qarrays.qarray import QArray, QArrayLike
from ..qarrays.utils import asqarray, tree_sum
from .general import dag
from .operators import eye

__all__ = [
    'operator_to_vector',
    'sdissipator',
    'slindbladian',
    'spost',
    'spre',
    'sprepost',
    'vector_to_operator',
]


def operator_to_vector(x: QArrayLike) -> QArray:
    r"""Returns the vectorized version of an operator.

    The vectorized column vector $\kett{A}$ (shape $n^2\times 1$) is obtained by
    stacking the columns of the matrix $A$ (shape $n\times n$) on top of one another:
    $$
        A = \begin{pmatrix} a & b \\\\ c & d \end{pmatrix}
        \to
        \kett{A} = \begin{pmatrix} a \\\\ c \\\\ b \\\\ d \end{pmatrix}.
    $$

    Args:
        x _(qarray_like of shape (..., n, n))_: Operator.

    Returns:
        _(qarray of shape (..., n^2, 1))_ Vectorized operator.

    Examples:
        >>> A = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        >>> A
        Array([[1.+1.j, 2.+2.j],
               [3.+3.j, 4.+4.j]], dtype=complex64)

        # todo: temporary fix
        # >>> dq.operator_to_vector(A)
        # Array([[1.+1.j],
        #        [3.+3.j],
        #        [2.+2.j],
        #        [4.+4.j]], dtype=complex64)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    bshape = x.shape[:-2]
    return x.mT.reshape(*bshape, -1, 1)


def vector_to_operator(x: QArrayLike) -> QArray:
    r"""Returns the operator version of a vectorized operator.

    The matrix $A$ (shape $n\times n$) is obtained by stacking horizontally next to
    each other each group of $n$ elements of the vectorized column vector $\kett{A}$
    (shape $n^2\times 1$):
    $$
        \kett{A} = \begin{pmatrix} a \\\\ b \\\\ c \\\\ d \end{pmatrix}
        \to
        A = \begin{pmatrix} a & c \\\\ b & d \end{pmatrix}.
    $$

    Args:
        x _(qarray_like of shape (..., n^2, 1))_: Vectorized operator.

    Returns:
        _(qarray of shape (..., n, n))_ Operator.

    Examples:
        >>> Avec = jnp.array([[1 + 1j], [2 + 2j], [3 + 3j], [4 + 4j]])
        >>> Avec
        Array([[1.+1.j],
               [2.+2.j],
               [3.+3.j],
               [4.+4.j]], dtype=complex64)

        # todo: temporary fix
        # >>> dq.vector_to_operator(Avec)
        # Array([[1.+1.j, 3.+3.j],
        #        [2.+2.j, 4.+4.j]], dtype=complex64)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n^2, 1)')
    bshape = x.shape[:-2]
    n = int(np.sqrt(x.shape[-2]))
    return x.reshape(*bshape, n, n).mT


def spre(x: QArrayLike) -> QArray:
    r"""Returns the superoperator formed from pre-multiplication by an operator.

    Pre-multiplication by matrix $A$ is defined by the superoperator
    $I_n \otimes A$ in vectorized form:
    $$
        AX \to (I_n \otimes A) \kett{X}.
    $$

    Args:
        x _(qarray_like of shape (..., n, n))_: Operator.

    Returns:
        _(qarray of shape (..., n^2, n^2))_ Pre-multiplication superoperator.

    Examples:
        >>> dq.spre(dq.destroy(3)).shape
        (9, 9)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return eye(*x.dims).superand(x)


def spost(x: QArrayLike) -> QArray:
    r"""Returns the superoperator formed from post-multiplication by an operator.

    Post-multiplication by matrix $A$ is defined by the superoperator
    $A^\mathrm{T} \otimes I_n$ in vectorized form:
    $$
        XA \to (A^\mathrm{T} \otimes I_n) \kett{X}.
    $$

    Args:
        x _(qarray_like of shape (..., n, n))_: Operator.

    Returns:
        _(qarray of shape (..., n^2, n^2))_ Post-multiplication superoperator.

    Examples:
        >>> dq.spost(dq.destroy(3)).shape
        (9, 9)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return x.mT.superand(eye(*x.dims))


def sprepost(x: QArrayLike, y: QArrayLike) -> QArray:
    r"""Returns the superoperator formed from pre- and post-multiplication by operators.

    Pre-multiplication by matrix $A$ and post-multiplication by matrix $B$ is defined
    by the superoperator $B^\mathrm{T} \otimes A$ in vectorized form:
    $$
        AXB \to (B^\mathrm{T} \otimes A) \kett{X}.
    $$

    Args:
        x _(qarray_like of shape (..., n, n))_: Operator for pre-multiplication.
        y _(qarray_like of shape (..., n, n))_: Operator for post-multiplication.

    Returns:
        _(Qarray of shape (..., n^2, n^2))_ Pre- and post-multiplication superoperator.

    Examples:
        >>> dq.sprepost(dq.destroy(3), dq.create(3)).shape
        (9, 9)
    """
    x = asqarray(x)
    y = asqarray(y)
    check_shape(x, 'x', '(..., n, n)')
    check_shape(y, 'y', '(..., n, n)')
    return y.mT.superand(x)


def sdissipator(L: QArrayLike) -> QArray:
    r"""Returns the Lindblad dissipation superoperator (in matrix form).

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:
    $$
        \mathcal{D}[L] (\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    The vectorized form of this superoperator is:
    $$
        L^\* \otimes L
        - \frac{1}{2} (I_n \otimes L^\dag L)
        - \frac{1}{2} (L^\mathrm{T} L^\* \otimes I_n).
    $$

    Args:
        L _(qarray_like of shape (..., n, n))_: Jump operator.

    Returns:
        _(qarray of shape (..., n^2, n^2))_ Dissipation superoperator.

    See also:
        - [`dq.dissipator()`][dynamiqs.dissipator]: applies the dissipation
            superoperator to a state using only $n\times n$ matrix multiplications.
    """
    L = asqarray(L)
    check_shape(L, 'L', '(..., n, n)')
    Ldag = dag(L)
    LdagL = Ldag @ L
    return sprepost(L, Ldag) - 0.5 * spre(LdagL) - 0.5 * spost(LdagL)


def slindbladian(H: QArrayLike, jump_ops: list[QArrayLike]) -> QArray:
    r"""Returns the Lindbladian superoperator (in matrix form).

    The Lindbladian superoperator $\mathcal{L}$ is defined by:
    $$
        \mathcal{L} (\rho) = -i[H,\rho] + \sum_{k=1}^N \mathcal{D}[L_k] (\rho),
    $$

    where $H$ is the system Hamiltonian, $\{L_k\}$ is a set of $N$ jump operators
    (arbitrary operators) and $\mathcal{D}[L]$ is the Lindblad dissipation superoperator
    (see [`dq.sdissipator()`][dynamiqs.sdissipator]).

    The vectorized form of this superoperator is:
    $$
        -i (I_n \otimes H) + i (H^\mathrm{T} \otimes I_n) + \sum_{k=1}^N \left(
            L_k^\* \otimes L_k
            - \frac{1}{2} (I_n \otimes L_k^\dag L_k)
            - \frac{1}{2} (L_k^\mathrm{T} L_k^\* \otimes I_n)
        \right).
    $$

    Note:
        This superoperator is also sometimes called *Liouvillian*.

    Args:
        H _(qarray_like of shape (..., n, n))_: Hamiltonian.
        jump_ops _(list of qarray_like, each of shape (..., n, n))_: List of jump
            operators.

    Returns:
        _(qarray of shape (..., n^2, n^2))_ Lindbladian superoperator.

    See also:
        - [`dq.lindbladian()`][dynamiqs.lindbladian]: applies the Lindbladian
            superoperator to a state using only $n\times n$ matrix multiplications.
    """
    H = asqarray(H)
    jump_ops = [asqarray(L) for L in jump_ops]

    # === check H shape
    check_shape(H, 'H', '(..., n, n)')

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)')

    terms = [-1j * spre(H), 1j * spost(H)] + [sdissipator(L) for L in jump_ops]
    return tree_sum(terms)
