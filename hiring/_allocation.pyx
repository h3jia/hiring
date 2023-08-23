import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport atan2, pi, sqrt, isfinite
from libc.stdint cimport int8_t

__all__ = ['_set_mask', '_merge_layers', '_zoom_layers']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double _interp(double x, double x0, double dx, double y0, double y1) nogil:
    return y0 + (y1 - y0) * (x - x0) / dx


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _set_mask(int8_t[:, ::1] mask, const double[:, ::1] xs, const double[:, ::1] ys,
              const double[::1] r_in, const double[::1] r_out, int n_x, int n_y, int n_phi,
              int fill_value):
    cdef size_t i, j
    cdef double dphi = 2 * pi / (n_phi - 1.)
    cdef double *tmp_r = <double *> malloc(n_x * n_y * sizeof(double))
    cdef double *tmp_phi = <double *> malloc(n_x * n_y * sizeof(double))
    cdef int *tmp_k = <int *> malloc(n_x * n_y * sizeof(int))
    if not tmp_r or not tmp_phi or not tmp_k:
        raise MemoryError('cannot malloc required array in _set_mask.')
    try:
        for i in prange(n_x, nogil=True, schedule='static'):
            for j in range(n_y):
                tmp_r[i * n_y + j] = sqrt(xs[i, j] * xs[i, j] + ys[i, j] * ys[i, j])
                tmp_phi[i * n_y + j] = (atan2(ys[i, j], xs[i, j]) + 2 * pi) % (2 * pi)
                tmp_k[i * n_y + j] = int(tmp_phi[i * n_y + j] / dphi)
                if tmp_r[i * n_y + j] >= _interp(tmp_phi[i * n_y + j],
                                                 tmp_k[i * n_y + j] * dphi,
                                                 dphi,
                                                 r_in[tmp_k[i * n_y + j]],
                                                 r_in[tmp_k[i * n_y + j] + 1]):
                    if tmp_r[i * n_y + j] <= _interp(tmp_phi[i * n_y + j],
                                                     tmp_k[i * n_y + j] * dphi,
                                                     dphi,
                                                     r_out[tmp_k[i * n_y + j]],
                                                     r_out[tmp_k[i * n_y + j] + 1]):
                        mask[i, j] = fill_value
    finally:
        free(tmp_r)
        free(tmp_phi)
        free(tmp_k)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _merge_layers(float[:, ::1] low, const float[:, ::1] high, int n_x, int n_y, int pad_x,
                  int pad_y):
    cdef size_t i, j
    for i in prange(n_x, nogil=True, schedule='static'):
        for j in range(n_y):
            if isfinite(high[i, j]):
                low[i + pad_x, j + pad_y] = high[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _zoom_layers(const float[:, ::1] input, float[:, ::1] output, int n_x, int n_y, int n_zoom):
    cdef size_t i, j, k, l
    cdef int m = (n_zoom - 1) // 2

    for i in range(m):
        for j in range(m):
            output[i, j] = input[0, 0]
    for i in range(n_x * n_zoom - m, n_x * n_zoom):
        for j in range(m):
            output[i, j] = input[n_x - 1, 0]
    for i in range(m):
        for j in range(n_y * n_zoom - m, n_y * n_zoom):
            output[i, j] = input[0, n_y - 1]
    for i in range(n_x * n_zoom - m, n_x * n_zoom):
        for j in range(n_y * n_zoom - m, n_y * n_zoom):
            output[i, j] = input[n_x - 1, n_y - 1]

    output[m, m] = input[0, 0]

    for i in range(1, n_x):
        output[m + i * n_zoom, m] = input[i, 0]
        for j in range(1, n_zoom):
            output[m + (i - 1) * n_zoom + j, m] = _interp(j, 0., n_zoom,
                                                          output[m + (i - 1) * n_zoom, m],
                                                          output[m + i * n_zoom, m])
    for i in range(1, n_y):
        output[m, m + i * n_zoom] = input[0, i]
        for j in range(1, n_zoom):
            output[m, m + (i - 1) * n_zoom + j] = _interp(j, 0., n_zoom,
                                                          output[m, m + (i - 1) * n_zoom],
                                                          output[m, m + i * n_zoom])

    for i in prange(1, n_x, nogil=True, schedule='static'):
        for j in range(1, n_y):
            output[m + i * n_zoom, m + j * n_zoom] = input[i, j]

    for i in prange(1, n_x, nogil=True, schedule='static'):
        for j in range(1, n_y):
            for k in range(1, n_zoom):
                output[m + i * n_zoom, m + (j - 1) * n_zoom + k] = _interp(
                    k, 0., n_zoom, output[m + i * n_zoom, m + (j - 1) * n_zoom],
                    output[m + i * n_zoom, m + j * n_zoom]
                )
            for k in range(1, n_zoom):
                output[m + (i - 1) * n_zoom + k, m + j * n_zoom] = _interp(
                    k, 0., n_zoom, output[m + (i - 1) * n_zoom, m + j * n_zoom],
                    output[m + i * n_zoom, m + j * n_zoom]
                )

    for i in prange(1, n_x, nogil=True, schedule='static'):
        for j in range(1, n_y):
            for k in range(1, n_zoom):
                for l in range(1, n_zoom):
                    output[m + (i - 1) * n_zoom + k, m + (j - 1) * n_zoom + l] = _interp(
                        k, 0., n_zoom, output[m + (i - 1) * n_zoom, m + (j - 1) * n_zoom + l],
                        output[m + i * n_zoom, m + (j - 1) * n_zoom + l]
                    )

    for i in prange(m, n_x * n_zoom - m, nogil=True, schedule='static'):
        for j in range(m):
            output[i, j] = output[i, m]
        for j in range(n_y * n_zoom - m, n_y * n_zoom):
            output[i, j] = output[i, n_y * n_zoom - m - 1]

    for i in prange(m, n_y * n_zoom - m, nogil=True, schedule='static'):
        for j in range(m):
            output[j, i] = output[m, i]
        for j in range(n_x * n_zoom - m, n_x * n_zoom):
            output[j, i] = output[n_x * n_zoom - m - 1, i]
