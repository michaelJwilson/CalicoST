import numpy as np
import scipy
from scipy.special import loggamma
from numba import njit, prange
from numpy.testing import assert_allclose

__max_gamma = 10_000
__log_gamma_table = loggamma(np.arange(__max_gamma))


@njit(cache=True, parallel=False)
def convert_params(mean, std):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See
        https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """
    p = mean / std**2
    n = mean * p / (1.0 - p)

    return np.round(n), p


@njit(cache=True)
def get_log_factorial(ks):
    """
    gamma(n + 1) = n! for integer n.
    """
    return get_log_gamma(ks + 1)


@njit(cache=True)
def get_log_gamma(ks, stirling=True):
    """
    gamma(n) = (n-1)! for integer n.
    """
    result = np.zeros(len(ks), dtype=float)

    for ii in range(len(ks)):
        # NB expected to be integers
        kk = int(round(ks[ii]))

        # TODO why are these occurring?
        if kk < 0:
            result[ii] = np.inf
        elif kk >= __max_gamma:
            if stirling:
                result[ii] = (kk - 1) * np.log(kk) - kk
            else:
                result[ii] = np.inf
        else:
            result[ii] = __log_gamma_table[kk]

    return result


@njit(cache=True)
def get_log_negbinomial(nn, pp, kk, log_factorial_kk=None):
    nn, kk = np.round(nn), np.round(kk)
    
    if log_factorial_kk is None:
        log_factorial_kk = get_log_factorial(kk)
        
    result = nn * np.log(pp) + kk * np.log(1.0 - pp)
    result += get_log_gamma(kk + nn)
    result -= get_log_gamma(nn)
    result -= log_factorial_kk

    return result


@njit(cache=True)
def get_log_betabinomial(
    nn, kk, aa, bb, log_gamma_nn=None, log_gamma_kk=None, log_gamma_nn_kk=None
):
    nn, kk = np.round(nn), np.round(kk)
    aa, bb = np.ceil(aa), np.ceil(bb)
    
    if log_gamma_nn is None:
        log_gamma_nn = get_log_gamma(nn + 1)
    if log_gamma_kk is None:
        log_gamma_kk = get_log_gamma(kk + 1)
    if log_gamma_nn_kk is None:
        log_gamma_nn_kk = get_log_gamma(nn - kk + 1)

    result = (
        log_gamma_nn
        + get_log_gamma(kk + aa)
        + get_log_gamma(nn - kk + bb)
        + get_log_gamma(aa + bb)
    )
    
    result -= (
        log_gamma_kk
        + log_gamma_nn_kk
        + get_log_gamma(nn + aa + bb)
        + get_log_gamma(aa)
        + get_log_gamma(bb)
    )

    return result

@njit(cache=True, parallel=True)
def compute_emission_probability_nb_betabinom(X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus):
    n_states = log_mu.shape[0]
    (n_obs, n_comp, n_spots) = X.shape

    log_emission_rdr = np.zeros((n_states, n_obs, n_spots), dtype=float)
    log_emission_baf = np.zeros((n_states, n_obs, n_spots), dtype=float)
    
    for s in prange(n_spots):
        idx_nonzero_rdr = np.where(base_nb_mean[:,s] > 0)[0]
        idx_nonzero_baf = np.where(total_bb_RD[:,s] > 0)[0]

        if len(idx_nonzero_rdr) > 0:
            kk = X[idx_nonzero_rdr, 0, s]
            log_factorial_kk = get_log_factorial(kk)
            
            for i in np.arange(n_states):
                nb_mean = base_nb_mean[idx_nonzero_rdr,s] * np.exp(log_mu[i, s])
                nb_std = np.sqrt(nb_mean + alphas[i, s] * nb_mean**2)

                nn, pp = convert_params(nb_mean, nb_std)

                # log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(kk, nn, pp)
                log_emission_rdr[i, idx_nonzero_rdr, s] = get_log_negbinomial(
                    nn, pp, kk, log_factorial_kk=log_factorial_kk
                )
                
        if len(idx_nonzero_baf) > 0:
            kk = X[idx_nonzero_baf, 1, s]
            nn = total_bb_RD[idx_nonzero_baf, s]

            log_gamma_nn = get_log_gamma(nn + 1)
            log_gamma_kk = get_log_gamma(kk + 1)
            log_gamma_nn_kk = get_log_gamma(nn - kk + 1)

            for i in np.arange(n_states):
                aa = p_binom[i, s] * taus[i, s] * np.ones(len(kk))
                bb = (1. - p_binom[i, s]) * taus[i, s] * np.ones(len(kk))

                # DEPRECATE
                # log_emission_baf[i, idx_nonzero_baf, s] = scipy.stats.betabinom.logpmf(kk, nn, aa, bb)

                # TODO BUG? very large pseudo-counts.
                # exp = scipy.stats.betabinom.logpmf(kk, nn, aa, bb)
                result = get_log_betabinomial(                                                                                                                
                    nn,                                                                                                                                                                       
                    kk,                                                                                                                                                                       
                    aa,                                                                                                                                                                       
                    bb,                                                                                                                                                                       
                    log_gamma_nn=log_gamma_nn,                                                                                                                                                
                    log_gamma_kk=log_gamma_kk,
                    log_gamma_nn_kk=log_gamma_nn_kk,                                                                                                                                          
                ) 
                
                log_emission_baf[i, idx_nonzero_baf, s] = result
                """
                try:
                    assert_allclose(exp, result, atol=5.e-2, rtol=1.e-2)
                except Exception as E:
                    print(f"\n\n{E}\n{nn}\n{kk}\n{p_binom[i, s]}\n{taus[i, s]}\n{aa}\n{bb}")
                    exit(0)
                """ 
    return log_emission_rdr, log_emission_baf
    
# TODO DEPRECATE
@njit(cache=True, parallel=True)
def compute_emission_probability_nb_betabinom_mix_v1(
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop
):
    n_states = log_mu.shape[0]
    (n_obs, n_comp, n_spots) = X.shape

    log_emission_rdr = np.zeros((n_states, n_obs, n_spots), dtype=float)
    log_emission_baf = np.zeros((n_states, n_obs, n_spots), dtype=float)

    # TODO HACK
    # spots_to_process = range(n_states)
    spots_to_process = np.where(np.sum(base_nb_mean + total_bb_RD, axis=0) > 0.0)[0]

    for ii in prange(n_states):
        for s in spots_to_process:
            # NB expression from NB distribution,
            #    see https://en.wikipedia.org/wiki/Negative_binomial_distribution
            idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0.0)[0]
            idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0.0)[0]

            if len(idx_nonzero_rdr) > 0:
                # nb_mean = base_nb_mean[idx_nonzero_rdr,s] * (tumor_prop[s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[s])
                nb_mean = base_nb_mean[idx_nonzero_rdr, s]
                nb_mean *= (
                    tumor_prop[idx_nonzero_rdr, s] * np.exp(log_mu[ii, s])
                    + 1.0
                    - tumor_prop[idx_nonzero_rdr, s]
                )

                nb_std = np.sqrt(nb_mean + alphas[ii, s] * nb_mean**2)

                nn, pp = convert_params(nb_mean, nb_std)
                kk = X[idx_nonzero_rdr, 0, s]

                # DEPRECATE
                # log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(X[idx_nonzero_rdr, 0, s], n, p)

                log_factor_kk = get_log_factorial(kk)
                log_emission_rdr[ii, idx_nonzero_rdr, s] = get_log_negbinomial(
                    nn, pp, kk, log_factorial_kk=log_factor_kk
                )

            # AF from BetaBinom distribution
            if len(idx_nonzero_baf) > 0:
                this_weighted_tp = tumor_prop[idx_nonzero_baf, s]

                """
                # TODO construct this_weighted_tp[idx_nonzero_baf] only                                                                                                                                          
                if ("logmu_shift" in kwargs) and ("sample_length" in kwargs):
                    sample_length, logmu_shift = kwargs["sample_length"], kwargs["logmu_shift"]
                    this_weighted_tp = []

                    for c in range(len(sample_length)):
                        range_s = np.sum(sample_length[:c])
                        
                        # DEPRECATE                                                                                                                                                                                
                        # range_t = np.sum(sample_length[:(c+1)])                                                                                                                                                  

                        range_t = range_s + sample_length[c]

                        this_tumor_prop = tumor_prop[range_s:range_t,s]
                        
                        interim = this_tumor_prop * np.exp(log_mu[ii, s] - logmu_shift[c,s])
                        interim /= (interim + 1. - this_tumor_prop)

                        this_weighted_tp.append(interim)
                        
                    this_weighted_tp = np.concatenate(this_weighted_tp)
                """
                interim = this_weighted_tp + 0.5 * (1.0 - this_weighted_tp)

                mix_p_A = p_binom[ii, s] * interim
                mix_p_B = (1.0 - p_binom[ii, s]) * interim

                # NB see https://en.wikipedia.org/wiki/Beta-binomial_distribution,
                #        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
                kk, nn, aa, bb = (
                    X[idx_nonzero_baf, 1, s],
                    total_bb_RD[idx_nonzero_baf, s],
                    mix_p_A * taus[ii, s],
                    mix_p_B * taus[ii, s],
                )

                # kk, nn, aa, bb = np.round(kk), np.round(nn), np.round(aa), np.round(bb)

                # DEPRECATE
                # log_emission_baf[ii, idx_nonzero_baf, s] += scipy.stats.betabinom.logpmf(kk, nn, aa, bb)
                
                log_emission_baf[ii, idx_nonzero_baf, s] += get_log_betabinomial(
                    nn,
                    kk,
                    aa,
                    bb,
                )

    return log_emission_rdr, log_emission_baf


@njit(cache=True, parallel=True)
def compute_emission_probability_nb_betabinom_mix(
    X, base_nb_mean, log_mu, alphas, total_bb_RD, p_binom, taus, tumor_prop
):
    n_states = log_mu.shape[0]
    (n_obs, n_comp, n_spots) = X.shape

    log_emission_rdr = np.zeros((n_states, n_obs, n_spots), dtype=float)
    log_emission_baf = np.zeros((n_states, n_obs, n_spots), dtype=float)

    for s in prange(n_spots):
        idx_nonzero_rdr = np.where(base_nb_mean[:, s] > 0.0)[0]
        idx_nonzero_baf = np.where(total_bb_RD[:, s] > 0.0)[0]

        if len(idx_nonzero_rdr) > 0:
            # DEPRECATE
            # nb_mean = base_nb_mean[idx_nonzero_rdr,s] * (tumor_prop[s] * np.exp(log_mu[i, s]) + 1 - tumor_prop[s])

            # NB expression from NB distribution: kk failures after nn successes with P(success) == pp.
            #    see https://en.wikipedia.org/wiki/Negative_binomial_distribution
            kk = X[idx_nonzero_rdr, 0, s]
            log_factorial_kk = get_log_factorial(kk)

            for ii in range(n_states):
                nb_mean = base_nb_mean[idx_nonzero_rdr, s]
                nb_mean *= (
                    tumor_prop[idx_nonzero_rdr, s] * np.exp(log_mu[ii, s])
                    + 1.0
                    - tumor_prop[idx_nonzero_rdr, s]
                )

                nb_std = np.sqrt(nb_mean + alphas[ii, s] * nb_mean**2)
                nn, pp = convert_params(nb_mean, nb_std)

                # TODO
                # nn, kk = np.round(nn), np.round(kk)

                # DEPRECATE
                # log_emission_rdr[i, idx_nonzero_rdr, s] = scipy.stats.nbinom.logpmf(X[idx_nonzero_rdr, 0, s], n, p)
                
                log_emission_rdr[ii, idx_nonzero_rdr, s] = get_log_negbinomial(
                    nn, pp, kk, log_factorial_kk=log_factorial_kk
                )

        # AF from BetaBinom distribution
        if len(idx_nonzero_baf) > 0:
            this_weighted_tp = tumor_prop[idx_nonzero_baf, s]

            # NB kk successes in nn trials with pseudo-counts aa, bb.  <mean> = nn * aa / (aa + bb).
            #
            #   See https://en.wikipedia.org/wiki/Beta-binomial_distribution,
            #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
            kk = X[idx_nonzero_baf, 1, s]
            nn = total_bb_RD[idx_nonzero_baf, s]

            log_gamma_nn = get_log_gamma(nn + 1)
            log_gamma_kk = get_log_gamma(kk + 1)
            log_gamma_nn_kk = get_log_gamma(nn - kk + 1)

            for ii in range(n_states):
                # TODO construct this_weighted_tp[idx_nonzero_baf] only
                # if ("logmu_shift" in kwargs) and ("sample_length" in kwargs):
                #     sample_length, logmu_shift = kwargs["sample_length"], kwargs["logmu_shift"]
                #     this_weighted_tp = -99. * np.ones(len(sample_length))
                #
                #     for cc in range(len(sample_length)):
                #         # DEPRECATE
                #         # range_t = np.sum(sample_length[:(cc+1)])
                #
                #         range_s = np.sum(sample_length[:cc])
                #         range_t = range_s + sample_length[cc]
                #
                #         this_tumor_prop = tumor_prop[range_s:range_t,s]
                #
                #         interim = this_tumor_prop * np.exp(log_mu[ii, s] - logmu_shift[cc,s])
                #         interim /= (interim + 1. - this_tumor_prop)
                #
                #         this_weighted_tp[cc] = interim

                interim = this_weighted_tp + 0.5 * (1.0 - this_weighted_tp)

                mix_p_A = p_binom[ii, s]
                mix_p_B = (1.0 - p_binom[ii, s])

                # TODO BUG? taus >> 1 for well-defined (integer pseudo counts)
                # NB norm = (aa + bb) = (interim * taus[ii, s]).
                norm = interim * taus[ii, s]
                aa, bb = mix_p_A * norm, mix_p_B * norm
                
                # DEPRECATE
                # log_emission_baf[ii, idx_nonzero_baf, s] = scipy.stats.betabinom.logpmf(kk, nn, aa, bb)
                
                log_emission_baf[ii, idx_nonzero_baf, s] = get_log_betabinomial(
                    nn,
                    kk,
                    aa,
                    bb,
                    log_gamma_nn=log_gamma_nn,
                    log_gamma_kk=log_gamma_kk,
                    log_gamma_nn_kk=log_gamma_nn_kk,
                )
                
    return log_emission_rdr, log_emission_baf
