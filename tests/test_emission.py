import numpy as np
import scipy
from numpy.testing import assert_allclose
from scipy.special import loggamma

from calicost.compute_emission import get_log_gamma, get_log_negbinomial, get_log_betabinomial


def test_gamma():
    ks = np.arange(1_000)

    # NB see https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.loggamma.html
    exp = loggamma(ks)
    result = get_log_gamma(ks)

    assert_allclose(exp, result)


def test_negbinomial():
    kk, nn, pp = np.array([20, 8]), np.array([30, 10]), np.array([0.25, 0.41118372])

    # NB supports non-integer n, but not k.
    exp = scipy.stats.nbinom.logpmf(kk, nn, pp)
    result = get_log_negbinomial(
        nn,
        pp,
        kk,
    )

    assert_allclose(exp, result)    

def test_betabinomial():
    kk, nn, aa, bb = (
        np.array([10, 20, 1., 1.]),
        np.array([30, 40, 1., 1.]),
        np.array([10, 20, 0.3475, 3_475.]),
        np.array([50, 60, 0.3277, 3_277.]),
    )

    # NB returns -inf for non-integer k and nan for non-integer n.
    #    supports non-integer a,b. 
    exp = scipy.stats.betabinom.logpmf(kk, nn, aa, bb)
    result = get_log_betabinomial(
        nn, kk, aa, bb, log_gamma_nn=None, log_gamma_kk=None, log_gamma_nn_kk=None
    )

    # assert_allclose(exp, result)

    print()
    print(exp)
    print(result)
    
