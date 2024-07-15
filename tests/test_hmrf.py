import calicostem
import line_profiler
import numpy as np
import pytest
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.hmrf import (
    hmrfmix_reassignment_posterior_concatenate_emission_v1,
    hmrfmix_reassignment_posterior_concatenate_emission)
from calicost.utils_tumor import get_tumor_weight
from scipy.sparse import csr_matrix

ITERATIONS = 1
ROUNDS = 2

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


def get_spatial_data():
    """
    Raw data + generated model parameters.
    """
    np.random.seed(314)

    # TODO HACK
    # see https://github.com/raphael-group/CalicoST/blob/4696325d5ca103d0d72ea2d471c60d1d753b097b/src/calicost/hmrf.py#L765
    n_states = 4

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


@pytest.mark.skip
def test_get_spatial_data(spatial_data):
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

    # NB usually n_spots, or one spot / clone.
    n_spots = single_X.shape[2]
    n_obs = single_X.shape[0]
    n_states = new_log_mu.shape[0]
    n_clones = len(kwargs["sample_length"])

    assert new_log_mu.shape == (n_states, n_spots)
    assert new_log_mu.shape == new_alphas.shape
    assert new_p_binom.shape == new_p_binom.shape
    assert new_taus == new_taus.shape

def test_hmrfmix_reassignment_posterior_concatenate_emission_v1(
    benchmark, spatial_data
):
    """
    pytest -s test_hmrf.py::test_hmrfmix_reassignment_posterior_concatenate_emission_v1
    """
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

    def call():
        # See emacs +764 ../src/calicost/hmrf.py
        #     emacs +201 ../src/calicost/hmm_NB_BB_nophasing_v2.py
        return hmrfmix_reassignment_posterior_concatenate_emission_v1(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            smooth_mat,
            hmm,
            kwargs["logmu_shift"],
            kwargs["sample_length"],
            dry_run=True,
        )

    benchmark.group = "hmrfmix_reassignment_posterior_concatenate_emission"
    benchmark(call)

def test_hmrfmix_reassignment_posterior_concatenate_emission(
    benchmark, spatial_data
):
    """
    pytest -s test_hmrf.py::test_hmrfmix_reassignment_posterior_concatenate_emission_v2

    Tests the new vectorized version of the HMRF emission calc.
    """
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

    def call():
        # See emacs +812 ../src/calicost/hmrf.py
        #     emacs +201 ../src/calicost/hmm_NB_BB_nophasing_v2.py
        return hmrfmix_reassignment_posterior_concatenate_emission(
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            single_tumor_prop,
            new_log_mu,
            new_alphas,
            new_p_binom,
            new_taus,
            smooth_mat,
            hmm,
            kwargs["logmu_shift"],
            kwargs["sample_length"],
        )

    benchmark.group = "hmrfmix_reassignment_posterior_concatenate_emission"
    tmp_log_emission_rdr, tmp_log_emission_baf = benchmark(call)

    # See emacs +764 ../src/calicost/hmrf.py
    #     emacs +201 ../src/calicost/hmm_NB_BB_nophasing_v2.py
    exp_rdr, exp_baf = hmrfmix_reassignment_posterior_concatenate_emission_v1(
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        single_tumor_prop,
        new_log_mu,
        new_alphas,
        new_p_binom,
        new_taus,
        smooth_mat,
        hmm,
        kwargs["logmu_shift"],
        kwargs["sample_length"],
    )


    for result, exp in zip((tmp_log_emission_rdr, tmp_log_emission_baf), (exp_rdr, exp_baf)):
        good = np.isclose(result, exp, atol=1.0e-6, equal_nan=True)
        mean = np.mean(good)

        print()
        print(mean)
        print(np.nanmin(result), result[0, 0, :])
        print(np.nanmin(exp), exp[0, 0, :])
        
        # NB TODO Rust NaNs matched to 0.0s
        assert mean >= 0.9998

    