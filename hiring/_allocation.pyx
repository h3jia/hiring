import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport atan2, pi, sqrt, isfinite, sinf, cosf, floorf
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
              const double[::1] r_in, const double[::1] r_out, int n_xy, int n_phi,
              int fill_value):
    cdef size_t i, j
    cdef double dphi = 2 * pi / (n_phi - 1.)
    cdef double *tmp_r = <double *> malloc(n_xy * n_xy * sizeof(double))
    cdef double *tmp_phi = <double *> malloc(n_xy * n_xy * sizeof(double))
    cdef int *tmp_k = <int *> malloc(n_xy * n_xy * sizeof(int))
    if not tmp_r or not tmp_phi or not tmp_k:
        raise MemoryError('cannot malloc required array in _set_mask.')
    try:
        for i in prange(n_xy, nogil=True, schedule='static'):
            for j in range(n_xy):
                tmp_r[i * n_xy + j] = sqrt(xs[i, j] * xs[i, j] + ys[i, j] * ys[i, j])
                tmp_phi[i * n_xy + j] = (atan2(ys[i, j], xs[i, j]) + 2 * pi) % (2 * pi)
                tmp_k[i * n_xy + j] = int(tmp_phi[i * n_xy + j] / dphi)
                if tmp_r[i * n_xy + j] >= _interp(tmp_phi[i * n_xy + j],
                                                  tmp_k[i * n_xy + j] * dphi,
                                                  dphi,
                                                  r_in[tmp_k[i * n_xy + j]],
                                                  r_in[tmp_k[i * n_xy + j] + 1]):
                    if tmp_r[i * n_xy + j] <= _interp(tmp_phi[i * n_xy + j],
                                                      tmp_k[i * n_xy + j] * dphi,
                                                      dphi,
                                                      r_out[tmp_k[i * n_xy + j]],
                                                      r_out[tmp_k[i * n_xy + j] + 1]):
                        mask[i, j] = fill_value
    finally:
        free(tmp_r)
        free(tmp_phi)
        free(tmp_k)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _merge_layers(float[:, ::1] low, const float[:, ::1] high, int n_xy, int pad_xy):
    cdef size_t i, j
    for i in prange(n_xy, nogil=True, schedule='static'):
        for j in range(n_xy):
            if isfinite(high[i, j]):
                low[i + pad_xy, j + pad_xy] = high[i, j]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _zoom_layers(const float[:, ::1] input, float[:, ::1] output, int n_xy, int n_zoom):
    cdef size_t i, j, k, l
    cdef int m = (n_zoom - 1) // 2

    for i in range(m):
        for j in range(m):
            output[i, j] = input[0, 0]
    for i in range(n_xy * n_zoom - m, n_xy * n_zoom):
        for j in range(m):
            output[i, j] = input[n_xy - 1, 0]
    for i in range(m):
        for j in range(n_xy * n_zoom - m, n_xy * n_zoom):
            output[i, j] = input[0, n_xy - 1]
    for i in range(n_xy * n_zoom - m, n_xy * n_zoom):
        for j in range(n_xy * n_zoom - m, n_xy * n_zoom):
            output[i, j] = input[n_xy - 1, n_xy - 1]

    output[m, m] = input[0, 0]

    for i in range(1, n_xy):
        output[m + i * n_zoom, m] = input[i, 0]
        for j in range(1, n_zoom):
            output[m + (i - 1) * n_zoom + j, m] = _interp(j, 0., n_zoom,
                                                          output[m + (i - 1) * n_zoom, m],
                                                          output[m + i * n_zoom, m])
    for i in range(1, n_xy):
        output[m, m + i * n_zoom] = input[0, i]
        for j in range(1, n_zoom):
            output[m, m + (i - 1) * n_zoom + j] = _interp(j, 0., n_zoom,
                                                          output[m, m + (i - 1) * n_zoom],
                                                          output[m, m + i * n_zoom])

    for i in prange(1, n_xy, nogil=True, schedule='static'):
        for j in range(1, n_xy):
            output[m + i * n_zoom, m + j * n_zoom] = input[i, j]

    for i in prange(1, n_xy, nogil=True, schedule='static'):
        for j in range(1, n_xy):
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

    for i in prange(1, n_xy, nogil=True, schedule='static'):
        for j in range(1, n_xy):
            for k in range(1, n_zoom):
                for l in range(1, n_zoom):
                    output[m + (i - 1) * n_zoom + k, m + (j - 1) * n_zoom + l] = _interp(
                        k, 0., n_zoom, output[m + (i - 1) * n_zoom, m + (j - 1) * n_zoom + l],
                        output[m + i * n_zoom, m + (j - 1) * n_zoom + l]
                    )

    for i in prange(m, n_xy * n_zoom - m, nogil=True, schedule='static'):
        for j in range(m):
            output[i, j] = output[i, m]
        for j in range(n_xy * n_zoom - m, n_xy * n_zoom):
            output[i, j] = output[i, n_xy * n_zoom - m - 1]

    for i in prange(m, n_xy * n_zoom - m, nogil=True, schedule='static'):
        for j in range(m):
            output[j, i] = output[m, i]
        for j in range(n_xy * n_zoom - m, n_xy * n_zoom):
            output[j, i] = output[n_xy * n_zoom - m - 1, i]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _rotate_image(const float[:, ::1] image, float[:, ::1] output, int n_xy, float angle,
                  int circle_mask):
    cdef size_t i, j
    cdef float c_xy = (n_xy - 1.) / 2.0  # center of the image
    cdef float rad_angle = angle * pi / 180.0  # convert to radians
    cdef float cos_a = cosf(rad_angle)
    cdef float sin_a = sinf(rad_angle)
    cdef float r_square = c_xy * c_xy  # radius squared for circle mask

    cdef float *x_rot = <float *> malloc(n_xy * n_xy * sizeof(float))
    cdef float *y_rot = <float *> malloc(n_xy * n_xy * sizeof(float))
    cdef int *x_orig = <int *> malloc(n_xy * n_xy * sizeof(int))
    cdef int *y_orig = <int *> malloc(n_xy * n_xy * sizeof(int))

    if not x_rot or not y_rot or not x_orig or not y_orig:
        raise MemoryError('cannot malloc required array in _rotate_image.')

    try:
        if angle and circle_mask:
            for i in prange(n_xy, nogil=True, schedule='static'):
                for j in range(n_xy):
                    if ((j - c_xy)**2 + (i - c_xy)**2) <= r_square:  # inside the circle mask
                        x_rot[i * n_xy + j] = cos_a * (j - c_xy) + sin_a * (i - c_xy) + c_xy
                        y_rot[i * n_xy + j] = -sin_a * (j - c_xy) + cos_a * (i - c_xy) + c_xy
                        x_orig[i * n_xy + j] = int(floorf(x_rot[i * n_xy + j]))
                        y_orig[i * n_xy + j] = int(floorf(y_rot[i * n_xy + j]))
                        output[i, j] = (
                            ((1 - (x_rot[i * n_xy + j] - x_orig[i * n_xy + j])) *
                             (1 - (y_rot[i * n_xy + j] - y_orig[i * n_xy + j])) *
                             image[y_orig[i * n_xy + j], x_orig[i * n_xy + j]]) +
                            ((x_rot[i * n_xy + j] - x_orig[i * n_xy + j]) *
                             (1 - (y_rot[i * n_xy + j] - y_orig[i * n_xy + j])) *
                             image[y_orig[i * n_xy + j],
                                   min(x_orig[i * n_xy + j] + 1, n_xy - 1)]) +
                            ((1 - (x_rot[i * n_xy + j] - x_orig[i * n_xy + j])) *
                             (y_rot[i * n_xy + j] - y_orig[i * n_xy + j]) *
                             image[min(y_orig[i * n_xy + j] + 1, n_xy - 1),
                                   x_orig[i * n_xy + j]]) +
                            ((x_rot[i * n_xy + j] - x_orig[i * n_xy + j]) *
                             (y_rot[i * n_xy + j] - y_orig[i * n_xy + j]) *
                             image[min(y_orig[i * n_xy + j] + 1, n_xy - 1),
                                   min(x_orig[i * n_xy + j] + 1, n_xy - 1)])
                        )
                    else:
                        output[i, j] = 0.

        elif angle and (not circle_mask):
            for i in prange(n_xy, nogil=True, schedule='static'):
                for j in range(n_xy):
                    x_rot[i * n_xy + j] = cos_a * (j - c_xy) + sin_a * (i - c_xy) + c_xy
                    y_rot[i * n_xy + j] = -sin_a * (j - c_xy) + cos_a * (i - c_xy) + c_xy
                    if ((0. <= x_rot[i * n_xy + j] <= (n_xy - 1.)) and
                        (0. <= y_rot[i * n_xy + j] <= (n_xy - 1.))):
                        x_orig[i * n_xy + j] = int(floorf(x_rot[i * n_xy + j]))
                        y_orig[i * n_xy + j] = int(floorf(y_rot[i * n_xy + j]))
                        output[i, j] = (
                            ((1 - (x_rot[i * n_xy + j] - x_orig[i * n_xy + j])) *
                             (1 - (y_rot[i * n_xy + j] - y_orig[i * n_xy + j])) *
                             image[y_orig[i * n_xy + j], x_orig[i * n_xy + j]]) +
                            ((x_rot[i * n_xy + j] - x_orig[i * n_xy + j]) *
                             (1 - (y_rot[i * n_xy + j] - y_orig[i * n_xy + j])) *
                             image[y_orig[i * n_xy + j],
                                   min(x_orig[i * n_xy + j] + 1, n_xy - 1)]) +
                            ((1 - (x_rot[i * n_xy + j] - x_orig[i * n_xy + j])) *
                             (y_rot[i * n_xy + j] - y_orig[i * n_xy + j]) *
                             image[min(y_orig[i * n_xy + j] + 1, n_xy - 1),
                                   x_orig[i * n_xy + j]]) +
                            ((x_rot[i * n_xy + j] - x_orig[i * n_xy + j]) *
                             (y_rot[i * n_xy + j] - y_orig[i * n_xy + j]) *
                             image[min(y_orig[i * n_xy + j] + 1, n_xy - 1),
                                   min(x_orig[i * n_xy + j] + 1, n_xy - 1)])
                        )
                    else:
                        output[i, j] = 0.

        elif (not angle) and circle_mask:
            for i in prange(n_xy, nogil=True, schedule='static'):
                for j in range(n_xy):
                    if ((j - c_xy)**2 + (i - c_xy)**2) <= r_square:
                        output[i, j] = image[i, j]
                    else:
                        output[i, j] = 0.

        elif (not angle) and (not circle_mask):
            for i in prange(n_xy, nogil=True, schedule='static'):
                for j in range(n_xy):
                    output[i, j] = image[i, j]

    finally:
        free(x_rot)
        free(y_rot)
        free(x_orig)
        free(y_orig)
