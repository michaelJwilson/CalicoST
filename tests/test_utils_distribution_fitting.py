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
from calicost.utils_distribution_fitting import Weighted_NegativeBinomial, Weighted_BetaBinom, Weighted_BetaBinom_mix
from scipy.sparse import csr_matrix
from scipy.stats import betabinom
from sklearn.preprocessing import OneHotEncoder

# TODO HACK COPY
def get_raw_spatial_data():
    # TODO HACK
    root = "/Users/mw9568/runs/CalicoSTdata/HT225C1_joint"

    inkwargs = np.load(f"{root}/kwargs.npz")
    res = np.load(f"{root}/res.npz")
    single_base_nb_mean = np.load(f"{root}/single_base_nb_mean.npy")
    single_tumor_prop = np.load(f"{root}/single_tumor_prop.npy")
    single_X = np.load(f"{root}/single_X.npy")
    single_total_bb_RD = np.load(f"{root}/single_total_bb_RD.npy")
    smooth_mat = np.load(f"{root}/smooth_mat.npz")
    smooth_mat = csr_matrix(
        (smooth_mat["data"], smooth_mat["indices"], smooth_mat["indptr"]),
        shape=smooth_mat["shape"],
    )

    kwargs = {}
    kwargs["logmu_shift"] = inkwargs["logmu_shift"]
    kwargs["sample_length"] = inkwargs["sample_length"]

    return (
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        kwargs,
    )

# TODO HACK COPY
@pytest.mark.skip
def test_get_raw_spatial_data():
    """
    Explicit demo of expected shapes, etc.
    """
    (
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        kwargs,
    ) = get_raw_spatial_data()

    logmu_shift = kwargs["logmu_shift"]
    sample_length = kwargs["sample_length"]

    n_obs, n_comp, n_spots = single_X.shape
    n_clones = len(kwargs["sample_length"])

    assert single_base_nb_mean.shape == (n_obs, n_spots)
    assert single_tumor_prop.shape == (n_spots,)
    assert single_X.shape == (n_obs, n_comp, n_spots)
    assert single_total_bb_RD.shape == (n_obs, n_spots)
    assert smooth_mat.shape == (n_spots, n_spots)
    assert sample_length.shape == (n_clones,)
    assert np.all(sample_length == n_obs)

    # TODO HACK last 1?
    assert logmu_shift.shape == (n_clones, 1)

    # NB expect (will fail):
    assert logmu_shift.shape == (n_clones, n_spots)

# TODO HACK COPY
def get_spatial_data():
    """
    Raw data + generated model parameters.
    """
    np.random.seed(314)

    # TODO HACK
    # see https://github.com/raphael-group/CalicoST/blob/4696325d5ca103d0d72ea2d471c60d1d753b097b/src/calicost/hmrf.py#L765
    n_states = 7

    (
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        kwargs,
    ) = get_raw_spatial_data()

    # NB usually n_spots, or one spot / clone.
    N = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_clones = len(kwargs["sample_length"])

    # TODO
    new_log_mu = np.log(2.0 + 2.0 * np.random.uniform(size=N))
    new_log_mu = np.tile(new_log_mu, (n_states, 1))

    new_alphas = 0.01 * np.ones_like(new_log_mu, dtype=float)

    new_p_binom = np.random.uniform(size=N)
    new_p_binom = np.tile(new_p_binom, (n_states, 1))

    new_taus = np.ones_like(new_p_binom)

    hmm = hmm_nophasing_v2()

    return (
        kwargs,
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        hmm,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
    )


@pytest.fixture
def spatial_data():
    return get_spatial_data()


def test_Weighted_BetaBinom(benchmark):
    """
    2x speedup of Rust vs scipy.
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


def test_Weighted_BetaBinom_mix(benchmark, spatial_data):
    """
    3.3x speedup of Rust vs scipy.
    """
    np.random.seed(314)
    
    (
        kwargs,
        res,
        single_base_nb_mean,
        single_tumor_prop,
        single_X,
        single_total_bb_RD,
        smooth_mat,
        hmm,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
    ) = spatial_data

    n_obs, n_comp, n_spots = single_X.shape

    # TODO HACK match number of spots in run.
    # n_spots = 5852
    # single_tumor_prop = single_tumor_prop[:n_spots]

    nclass, len_exog = 7, n_spots

    # NB construct pseduo counts (a,b) for each class.
    tau = 20.
    ps = 0.5 + 0.2 * np.random.uniform(size=nclass)

    aa = tau * ps
    bb = tau * (1. - ps)

    params = np.concatenate([ps, np.array([tau])])
        
    assert len(params) == (nclass + 1)

    # NB assign each spot to a class.
    state = np.random.randint(low=0, high=nclass, size=len_exog)
    exog = OneHotEncoder().fit_transform(state.reshape(-1, 1)).toarray()

    assert exog.shape == (n_spots, nclass)

    exposure = np.random.randint(low=10, high=25, size=len_exog)   
    endog = np.array([scipy.stats.betabinom.rvs(xp, aa[ss], bb[ss]) for ss, xp in zip(state, exposure)])
    
    weights = 0.8 + 0.1 * np.random.uniform(size=len_exog)
    
    beta_binom = Weighted_BetaBinom_mix(endog, exog, weights, exposure, single_tumor_prop)
    
    def call():
        return beta_binom.nloglikeobs(params)
    
    result = benchmark(call)

    print(n_spots, result)
    
    # NB regression test.
    # assert np.allclose(result, 12_143.90731147436)
