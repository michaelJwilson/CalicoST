from libc.math cimport fabs
cimport cython
from cython.parallel cimport prange
from libc.math cimport log, NAN

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

def get_available_threads():
    cdef int available_threads = omp_get_max_threads()

    return available_threads

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_beta_binomial(
    double[:] k,
    double[:] n,
    double[:] a,
    double[:] b,
    double[:] w,
    double[:] out,
    int nthreads
) nogil:
  cdef int length, idx
  
  length = len(k)

  for idx in prange(0, length, schedule="guided"):      
      if a[idx] <= 0.0:
          out[idx] = NAN
	      
      elif b[idx] <= 0.0:
          out[idx] = NAN
	      
      else:
          out[idx] = w[idx] * (-log(n[idx] + 1.) - csc.betaln(n[idx] - k[idx] + 1., k[idx] + 1.) - csc.betaln(a[idx], b[idx]) + csc.betaln(k[idx] + a[idx], n[idx] - k[idx] + b[idx]))

def parallel_beta_binomial(k, n, a, b, w):
    out = np.zeros_like(a, dtype="float")
    nthreads = get_available_threads()

    _parallel_beta_binomial(k, n, a, b, w, out, nthreads)

    return out.sum()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _parallel_beta_binomial_zeropoint(
    double[:] k,
    double[:] n,
    double[:] a,
    double[:] b,
    double[:] w,
    double[:] out,
    int nthreads
) nogil:
  cdef int length, idx

  length = len(k)

  for idx in prange(0, length, schedule="guided"):
      if a[idx] <= 0.0:
          out[idx] = NAN

      elif b[idx] <= 0.0:
          out[idx] = NAN

      else:
          out[idx] = w[idx] * (-csc.betaln(a[idx], b[idx]) + csc.betaln(k[idx] + a[idx], n[idx] - k[idx] + b[idx]))

def parallel_beta_binomial_zeropoint(k, n, a, b, w):
    out = np.zeros_like(a, dtype="float")
    nthreads = get_available_threads()

    _parallel_beta_binomial_zeropoint(k, n, a, b, w, out, nthreads)

    return out.sum()
