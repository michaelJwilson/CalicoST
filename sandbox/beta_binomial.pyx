from libc.math cimport fabs
cimport cython
from cython.parallel cimport prange
from libc.math cimport log

import numpy as np
import scipy.special as sc
cimport scipy.special.cython_special as csc

"""
python setup.py build_ext --inplace
"""

cdef extern from "omp.h":
    int omp_get_max_threads()

def serial_G(k, x, y):
    return 0.25j * sc.hankel1(0, k*np.abs(x - y))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_G(
    double k,
    double[:,:] x,
    double[:,:] y,
    double complex[:,:] out
) nogil:    
    cdef int i, j

    for i in prange(x.shape[0]):
        for j in range(y.shape[0]):
            out[i,j] = 0.25j * csc.hankel1(0, k*fabs(x[i,j] - y[i,j]))

def parallel_G(k, x, y):
    out = np.empty_like(x, dtype='complex128')
    
    _parallel_G(k, x, y, out)
    
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_beta_binomial(
    double[:] k,
    double[:] n,
    double[:] a,
    double[:] b,
    double[:] out
) nogil:
  cdef int chunk, j, chunk_size, length, idx
  
  length = len(k)
  chunk_size = 256

  for chunk in prange(0, length + chunk_size, chunk_size):
      for j in range(chunk_size):
          idx = j + chunk

          if idx >= length:
              break

	  # NB -(n + 1.).ln() - get_lnbeta(n - k + 1., k + 1.) - get_lnbeta(a, b) + get_lnbeta(k + a, n - k + b)
          out[idx] = -log(n[idx] + 1.) - csc.betaln(n[idx] - k[idx] + 1., k[idx] + 1.) - csc.betaln(a[idx], b[idx]) + csc.betaln(k[idx] + a[idx], n[idx] - k[idx] + b[idx])

def print_available_threads():
    cdef int available_threads = omp_get_max_threads()    
    print("Available threads:", available_threads)

def parallel_beta_binomial(k, n, a, b):
    out = np.zeros_like(a, dtype='double')

    print_available_threads()

    _parallel_beta_binomial(k, n, a, b, out)

    return out
