import functools
import inspect
import logging
import time
import os

import calicostem
import numpy as np
import scipy
import scipy.integrate
import scipy.stats
import statsmodels
import statsmodels.api as sm
from numba import jit, njit
from scipy import linalg, special
from scipy.special import betaln, loggamma, logsumexp
from sklearn import cluster
from sklearn.utils import check_random_state
from statsmodels.base.model import GenericLikelihoodModel

from calicost.utils_profiling import profile

# DEPRECATE
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger(__name__)

def convert_params(mean, std):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """
    p = mean/std**2
    n = mean*p/(1.0 - p)
    return n, p

def convert_params_var(mean, var):
     """
     Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports
     See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
     """
     p = mean / var
     n = mean * p / (1. - p)

     return n, p


class Weighted_NegativeBinomial(GenericLikelihoodModel):
    """
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]), where exog is the design matrix, and params[-1] is 1 / overdispersion.
    This function fits the NB params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Multiplication constant outside the exponential term. In scRNA-seq or SRT data, this term is the total UMI count per cell/spot.
    """
    def __init__(self, endog, exog, weights, exposure, seed=0, **kwds):
        super(Weighted_NegativeBinomial, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.seed = seed

    @profile
    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        nb_std = np.sqrt(nb_mean + params[-1] * nb_mean**2)
        
        n, p = convert_params(nb_mean, nb_std)
        
        llf = scipy.stats.nbinom.logpmf(self.endog, n, p)
        
        return -llf.dot(self.weights)

    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwds):
        self.exog_names.append('alpha')
        
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)

        return super(Weighted_NegativeBinomial, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            **kwds
        )


class Weighted_NegativeBinomial_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, seed=0, **kwds):
        super(Weighted_NegativeBinomial_mix, self).__init__(endog, exog, **kwds)
        
        self.weights = weights
        self.exposure = exposure
        self.seed = seed
        self.tumor_prop = tumor_prop

        self.n_spots = len(tumor_prop)
        self.n_states = exog.shape[1]

        # self.nloglikeobs_zeropoint = self.exposure * (1. - self.tumor_prop)   
        
        logger.info(f"Fitting Weighted_NegativeBinomial_mix for {self.n_spots} spots and {self.n_states} states.")
        
    @profile
    def nloglikeobs(self, params):
        nb_mean = self.exposure * (self.tumor_prop * (self.exog @ np.exp(params[:-1])) + 1. - self.tumor_prop)
        nb_var = nb_mean + params[-1] * nb_mean**2
        
        n, p = convert_params_var(nb_mean, nb_var)

        # return -scipy.stats.nbinom.logpmf(self.endog, n, p).dot(self.weights)
        return -calicostem.nb(self.endog.astype(float), n.astype(float), p).dot(self.weights)
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append('alpha')
        
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.1 * np.ones(self.nparams), 0.01)
                
        return super(Weighted_NegativeBinomial_mix, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            **kwds
        )


class Weighted_BetaBinom(GenericLikelihoodModel):
    """
    Beta-binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)), where p = exog @ params[:-1] and tau = params[-1].
    This function fits the BetaBin params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Total number of trials. In BAF case, this is the total number of SNP-covering UMIs.
    """
    def __init__(self, endog, exog, weights, exposure, **kwds):
        super(Weighted_BetaBinom, self).__init__(endog, exog, **kwds)
        
        self.weights = weights
        self.exposure = exposure

    @profile
    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = self.exog @ (1. - params[:-1]) * params[-1]

        # NB negative sum log likelihood.
        # return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(self.weights)
        return -calicostem.bb(self.endog.astype(float), self.exposure.astype(float), a, b).dot(self.weights)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        self.exog_names.append("tau")
        
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1)
                
        return super(Weighted_BetaBinom, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            **kwds)


class Weighted_BetaBinom_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, weights, exposure, tumor_prop, **kwds):
        super(Weighted_BetaBinom_mix, self).__init__(endog, exog, **kwds)
        
        self.weights = weights
        self.tumor_prop = tumor_prop
        self.exposure = exposure
        
        self.n_spots = len(tumor_prop)
        self.n_states = exog.shape[1]

        self.tumor_shift = 0.5 * (1. - self.tumor_prop)
        self.nloglikeobs_zeropoint = -np.log(self.exposure + 1.) - betaln(self.exposure - self.endog + 1., self.endog  + 1.)
                
    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1] * self.tumor_prop + self.tumor_shift) * params[-1]
        b = ((self.exog @ (1. - params[:-1])) * self.tumor_prop + self.tumor_shift) * params[-1]

        # return -(scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)).dot(self.weights)
        return -(self.nloglikeobs_zeropoint + betaln(self.endog + a, self.exposure - self.endog + b) - betaln(a, b)).dot(self.weights)
        
    def fit(self, start_params=None, maxiter=10_000, maxfun=5_000, **kwds):
        self.exog_names.append("tau")

        logger.info(f"Fitting Weighted_BetaBinom_mix for {self.n_spots} spots and {self.n_states} states.")
        
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = np.append(0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams), 1)

        start = time.time()
                
        result = super(Weighted_BetaBinom_mix, self).fit(
            start_params=start_params,
            maxiter=maxiter,
            maxfun=maxfun,
            **kwds
        )

        run_time = time.time() - start

        logger.info(f"Fitted Weighted_BetaBinom_mix in {run_time:.3f} seconds.")
        
        return result

class Weighted_BetaBinom_fixdispersion(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, **kwds):
        super(Weighted_BetaBinom_fixdispersion, self).__init__(endog, exog, **kwds)
        self.tau = tau
        self.weights = weights
        self.exposure = exposure

    @profile
    def nloglikeobs(self, params):
        a = (self.exog @ params) * self.tau
        b = (1 - self.exog @ params) * self.tau
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.1 * np.ones(self.nparams)
        
        return super(Weighted_BetaBinom_fixdispersion, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class Weighted_BetaBinom_fixdispersion_mix(GenericLikelihoodModel):
    def __init__(self, endog, exog, tau, weights, exposure, tumor_prop, **kwds):
        super(Weighted_BetaBinom_fixdispersion_mix, self).__init__(endog, exog, **kwds)
        self.tau = tau
        self.weights = weights
        self.exposure = exposure
        self.tumor_prop = tumor_prop

    @profile
    def nloglikeobs(self, params):
        a = (self.exog @ params * self.tumor_prop + 0.5 * (1 - self.tumor_prop)) * self.tau
        b = ((1 - self.exog @ params) * self.tumor_prop + 0.5 * (1 - self.tumor_prop)) * self.tau
        llf = scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.1 * np.ones(self.nparams)
        
        return super(Weighted_BetaBinom_fixdispersion_mix, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)


class BAF_Binom(GenericLikelihoodModel):
    """
    Binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)), where p = exog @ params[:-1] and tau = params[-1].
    This function fits the BetaBin params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Attributes
    ----------
    endog : array, (n_samples,)
        Y values.

    exog : array, (n_samples, n_features)
        Design matrix.

    weights : array, (n_samples,)
        Sample weights.

    exposure : array, (n_samples,)
        Total number of trials. In BAF case, this is the total number of SNP-covering UMIs.
    """
    def __init__(self, endog, exog, weights, exposure, offset, scaling, **kwds):
        super(BAF_Binom, self).__init__(endog, exog, **kwds)
        self.weights = weights
        self.exposure = exposure
        self.offset = offset
        self.scaling = scaling
    
    def nloglikeobs(self, params):
        linear_term = self.exog @ params
        p = self.scaling / (1 + np.exp(-linear_term + self.offset))
        llf = scipy.stats.binom.logpmf(self.endog, self.exposure, p)
        neg_sum_llf = -llf.dot(self.weights)
        return neg_sum_llf
    
    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            else:
                start_params = 0.5 / np.sum(self.exog.shape[1]) *  np.ones(self.nparams)
        return super(BAF_Binom, self).fit(start_params=start_params,
                                               maxiter=maxiter, maxfun=maxfun,
                                               **kwds)
