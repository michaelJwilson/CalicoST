import calicostem
import line_profiler
import numpy as np
import scipy
import pytest
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import (
    hmrfmix_reassignment_posterior_concatenate_emission,
    hmrfmix_reassignment_posterior_concatenate_emission_v1,
)
from calicost.utils_distribution_fitting import Weighted_NegativeBinomial, Weighted_BetaBinom
from scipy.sparse import csr_matrix
from scipy.stats import betabinom
from sklearn.preprocessing import OneHotEncoder


def test_Weighted_BetaBinom(benchmark):
    """
    x2 speedup of Rust vs scipy.
    """
    np.random.seed(314)
    
    nclass, len_exog = 5, 100_000
    
    tau = 20.
    ps = 0.5 + 0.2 * np.random.uniform(size=nclass)

    aa = tau * ps
    bb = tau * (1. - ps)

    params = np.concatenate([ps, np.array([tau])])
        
    state = np.random.randint(low=0, high=nclass, size=len_exog)
    exog = OneHotEncoder().fit_transform(state.reshape(-1, 1)).toarray()

    exposure = np.random.randint(low=10, high=25, size=len_exog)   
    endog = np.array([scipy.stats.betabinom.rvs(xp, aa[ss], bb[ss]) for ss, xp in zip(state, exposure)])
    
    weights = 0.8 + 0.1 * np.random.uniform(size=len_exog)
    
    beta_binom = Weighted_BetaBinom(endog, exog, weights, exposure)
    
    def call():
        return beta_binom.nloglikeobs(params)

    aa = np.array([aa[ss] for ss in state])
    bb = np.array([bb[ss] for ss in state])
    
    exp = -scipy.stats.betabinom.logpmf(endog, exposure, aa, bb).dot(weights)
    result = benchmark(call)

    assert np.allclose(exp, result)
    assert np.allclose(result, 199240.50086169314)
