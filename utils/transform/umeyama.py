from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np


def umeyama(src, dst, estimate_scale=True):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """
    if src.shape[1] == 2:
        return _asm_solve(src, dst, estimate_scale)
    else:
        return _umeyama(src, dst, estimate_scale)


def _umeyama(src, dst, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


def _asm_solve(src, dst, estimate_scale=True):
    """Estimate 2D-only similarity transformation, x2 faster than Umeyema.
    Parameters
    ----------
    src : (M, 2) array
        Source coordinates.
    dst : (M, 2) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (3, 3)
        The homogeneous similarity transformation matrix.
    """
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    dot_result = src_demean.reshape(-1) @ dst_demean.reshape(-1)
    norm_pow_2 = np.linalg.norm(src_demean) ** 2

    a = dot_result / norm_pow_2
    b = np.sum(np.cross(src_demean, dst_demean)) / norm_pow_2

    sr = np.array([[a, -b],
                   [b, a]])
    if not estimate_scale:
        sr = sr / np.linalg.norm(sr, axis=0, keepdims=True)
    t = dst_mean - src_mean @ sr.T

    ret = np.identity(3)
    ret[:2, :2] = sr
    ret[:2, 2] = t
    return ret


if __name__ == "__main__":
    import time
    p_a = np.random.randn(118, 2)

    rad = 0.1
    r = np.array([
        [np.cos(rad), np.sin(rad)],
        [-np.sin(rad), np.cos(rad)]])
    t = np.random.randn(2)
    s = np.random.rand(1)

    p_b = (p_a * s) @ r.T + t
    print("Random  src   : \n", r * s, '\n', t)

    start_time = time.time()
    for _ in range(1000):
        m = _umeyama(p_a, p_b)
    use_time = (time.time() - start_time) / 1000
    print("Umeyama solve : \n", m, "Use time:{}".format(use_time))

    start_time = time.time()
    for _ in range(1000):
        m = umeyama(p_a, p_b)
    use_time = (time.time() - start_time) / 1000
    print("Asm_solver solve : \n", m, "Use time:{}".format(use_time))
