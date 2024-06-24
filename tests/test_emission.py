import numpy as np
import scipy
from numpy.testing import assert_allclose
from scipy.special import loggamma

from calicost.compute_emission import get_log_gamma, get_log_negbinomial


def test_gamma():
    ks = np.arange(1_000)

    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html
    exp = loggamma(ks)
    result = get_log_gamma(ks)

    assert_allclose(exp, result)


def test_negbinomial():
    kk, nn, pp = np.array([20, 8]), np.array([30, 10]), np.array([0.25, 0.41118372])
    
    exp = scipy.stats.nbinom.logpmf(kk, nn, pp)
    result = get_log_negbinomial(
        nn, pp, kk,
    )

    assert_allclose(exp, result)


def test_betabinomial():
    
