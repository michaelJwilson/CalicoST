import contextlib
import functools
import gzip
import dill
import inspect
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import scipy
import scipy.integrate
import scipy.stats
import statsmodels
import statsmodels.api as sm
from scipy.optimize import minimize
from numba import jit, njit
from scipy import linalg, special
from scipy.special import loggamma, logsumexp
from sklearn import cluster
from sklearn.utils import check_random_state
from statsmodels.base.model import GenericLikelihoodModel

logger = logging.getLogger(__name__)

num_threads = "2"

logger.info(f"Setting number of threads for MKL/BLAS/LAPACK/OMP to {num_threads}.")

os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["OMP_NUM_THREADS"] = num_threads

def convert_params(mean, alpha):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://mathworld.wolfram.com/NegativeBinomialDistribution.html
    """
    p = 1.0 / (1.0 + mean * alpha)
    n = 1.0 / alpha
    
    return n, p


@contextlib.contextmanager
def save_stdout(fpath):
    """
    Context manager to write stdout to fpath.
    """
    original = sys.stdout

    with open(fpath, "w") as ff:
        sys.stdout = ff

        try:
            yield

        finally:
            sys.stdout = original


class WeightedModel(GenericLikelihoodModel, ABC):
    """
    An ABC for defined emission models.

    Attributes
    ----------
    endog : array, (n_samples,)                                                                                                                                                                                                                                                         Y values.
    exog : array, (n_samples, n_features)
        Design matrix.
    weights : array, (n_samples,)
        Sample weights.
    exposure : array, (n_samples,)
        Multiplication constant outside the exponential term. In scRNA-seq or SRT data, this term is the total UMI count per cell/spot.
    """
    def __init__(self, endog, exog, weights, exposure, *args, tumor_prop=None, method="nm", snapshot=True, **kwargs):
        # super().__init__(endog, exog, **kwargs)

        self.endog = endog
        self.exog = exog
        
        self.tumor_prop = tumor_prop
        self.weights = weights
        self.exposure = exposure
        self.method = method
        self.bounds = None
        self.class_balance = np.sum(exog, axis=0)
        self.class_balance_weights = exog.T @ weights
        
        # NB __post_init__ validates the expected tumor proportion and handles incrementing instance count.
        self.__post_init__()

        logger.info(
            f"Initializing {self.get_ninstance()}th instance of {self.__class__.__name__} model for endog.shape={endog.shape}, exog.shape={exog.shape} & weighted class balance={self.class_balance_weights}."
        )

        if snapshot:
            ninst = self.get_ninstance()
            class_name = self.__class__.__name__.lower()
            
            self.snapshot(f"snapshots/{class_name}/{class_name}_snapshot_{ninst}.dill")
        
    @abstractmethod
    def nloglikeobs(self, params):
        """
        Negative log-likelihood for the emission model.
        """
        pass

    @abstractmethod
    def get_default_start_params(self):
        pass

    @abstractmethod
    def get_ext_param_name(self):
        """
        Named parameter in the model.
        """
        pass

    @abstractmethod
    def __post_init__(self):
        """
        Validation and customisation for the derived class.
        E.g. validate the tumor_proportion and increment the instance
        count of the derived class.
        """
        pass

    def get_ninstance(self):
        """
        Return the instance count for the given model
        """
        return self.ninstance

    def get_nparams(self):
        return len(self.get_default_start_params())
        
    def __callback__(self, params):
        """
        Define callback for writing parameter chain to file.
        """
        print(f"{params} {self.nloglikeobs(params):.6f};")

    def fit(
        self,
        start_params=None,
        maxiter=1_500,
        maxfun=5_000,
        xtol=1.e-4,
        ftol=1.e-4,   
        write_chain=True,
        **kwargs,
    ):
        ext_param_name = self.get_ext_param_name()

        # self.exog_names.append(ext_param_name)

        # TODO
        method = self.method # if start_params is None else "nm"
        
        if start_params is None:
            if hasattr(self, "start_params"):
                start_params = self.start_params
                start_params_str = "existing"
            else:
                start_params = self.get_default_start_params()
                start_params_str = "default"
        else:
            start_params_str = "input"
            
        logger.info(
            f"Starting {self.__class__.__name__} {method} optimization @ ({start_params_str}) {start_params}."
        )

        if self.bounds is not None:
            logger.info(f"Assuming bounds of {self.bounds}; use an appropriate solver.")

        start = time.time()
        
        kwargs.pop("disp", None)

        tmp_path = f"{self.__class__.__name__.lower()}_chain.tmp"

        ninst = self.get_ninstance()

        # TODO mkdir chains 
        class_name = self.__class__.__name__.lower()
        final_path = f"chains/{class_name}/{class_name}_chain_{ninst}_{start_params_str}.txt.gzip"

        Path(final_path).parent.mkdir(parents=True, exist_ok=True)

        with save_stdout(tmp_path):
            """
            result = super().fit(
                start_params=start_params,
                maxiter=maxiter,
                maxfun=maxfun,
                skip_hessian=True,
                callback=self.__callback__,
                full_output=True,
                retall=True,
                disp=False,
                xtol=xtol,
                ftol=ftol,
                method=method,
                bounds=self.bounds,
                **kwargs,
            )
            """
            result = minimize(
                self.nloglikeobs,
                start_params,
                method=method,
                bounds=self.bounds,
                callback=self.__callback__,
                options={"maxiter": maxiter, "disp": False}
            )
            
        # NB specific to nm (Nelder-Mead) optimization.
        # niter, params = result.mle_retvals["iterations"], result.params
        
        niter, params = result.nit, result.x
        
        runtime = time.time() - start

        np.set_printoptions(precision=6)
        
        logger.info(
            f"{self.__class__.__name__} optimization in {runtime:.2f}s, with {niter} iterations.  Best-fit: {params}"
        )

        np.set_printoptions(precision=3)
        """
        if write_chain:
            logger.info(f"Writing chain to {final_path}")
            
            with open(tmp_path) as fin:
                with gzip.open(final_path, "wt") as fout:
                    fout.write(
                        f"#  {self.__class__.__name__} {ninst} @ {time.asctime()}\n"
                    )
                    
                    fout.write(
                        f"#  method:{method},start_type:{start_params_str},class_balance:{self.class_balance},class_balance_weighted:{self.class_balance_weights},runtime:{runtime:.6f},shape:{self.endog.shape[0]},"
                        + ",".join(
                            f"{key}:{value}"
                            for key, value in result.mle_retvals.items()
                        )
                        + "\n"
                    )

                    for line in fin:
                        fout.write(line)
        """
        os.remove(tmp_path)
        
        return result.x

    def snapshot(self, fpath):
        logger.info(f"Creating snapshot @ {fpath}")
        
        with open(fpath, "wb") as f:
            dill.dump(self, f)


class Weighted_NegativeBinomial(WeightedModel):
    """
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]),
    where exog is the design matrix, and params[-1] is 1 / overdispersion.  This function
    fits the NB params when samples are weighted by weights:

    max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)
    """
    ninstance = 0

    def nloglikeobs(self, params):
        nb_mean = np.exp(self.exog @ params[:-1]) * self.exposure
        n, p = convert_params(nb_mean, params[-1])

        return -scipy.stats.nbinom.logpmf(self.endog, n, p).dot(self.weights)

    def get_default_start_params(self):
        return np.append(0.1 * np.ones(self.exog.shape[1]), 1.e-2)

    def get_ext_param_name(self):
        return "alpha"

    def __post_init__(self):
        self.method = "lbfgs"
        self.bounds = (self.get_nparams() - 1) * [(None, None)] + [(1.e-4, None)]
        
        assert self.tumor_prop is None
        
        Weighted_NegativeBinomial.ninstance += 1


class Weighted_NegativeBinomial_mix(WeightedModel):
    """                                                                                                                                                                 
    Negative Binomial model endog ~ NB(exposure * exp(exog @ params[:-1]), params[-1]),                                                                                
    where exog is the design matrix, and params[-1] is 1 / overdispersion.  This function                                                                               
    fits the NB params when samples are weighted by weights:                                                                                                                                                                                                                                                                                   

    max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)                                                                                                    

    Adapated for varying tumor proportion.
    """
    ninstance = 0

    def nloglikeobs(self, params):
        nb_mean = self.exposure * (
            self.tumor_prop * np.exp(self.exog @ params[:-1]) + 1. - self.tumor_prop
        )

        # NB n == number of success, p == probability of a single success.
        n, p = convert_params(nb_mean, params[-1])

        # NB endog == , exposure                                                                                                                                                                                                                                                                                                                                       
        #                                                                                                                                                                                                                                                                                                                                                              
	#    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html  
        
        return -scipy.stats.nbinom.logpmf(self.endog, n, p).dot(self.weights)

    def get_default_start_params(self):
        return np.append(0.1 * np.ones(self.exog.shape[1]), 1.e-2)

    def get_ext_param_name(self):
        return "alpha"

    def __post_init__(self):
        self.method = "lbfgs"
        self.bounds = (self.get_nparams() - 1) * [(None, None)] + [(1.e-4, None)]
        
        assert self.tumor_prop is not None, "Tumor proportion must be defined."

        Weighted_NegativeBinomial_mix.ninstance += 1


class Weighted_BetaBinom(WeightedModel):
    """
    Beta-binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)),
    where p = exog @ params[:-1] and tau = params[-1].  This function fits the
    BetaBin params when samples are weighted by weights:

    max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)
    """
    ninstance = 0

    def nloglikeobs(self, params):
        a = (self.exog @ params[:-1]) * params[-1]
        b = (1.0 - self.exog @ params[:-1]) * params[-1]
        
        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(
            self.weights
        )

    def get_default_start_params(self):
        # TODO remove number of states.
        # DEPRECATE np.sum(self.exog.shape[1])
        return np.append(0.5 * np.ones(self.exog.shape[1]), 1.)

    def get_ext_param_name(self):
        return "tau"

    def __post_init__(self):
        self.method = "lbfgs"
        self.bounds = self.exog.shape[1] * [(0., 1.)] + [(1.e-2, self.exposure.max())]
        
        assert self.tumor_prop is None
        
        Weighted_BetaBinom.ninstance += 1
      

class Weighted_BetaBinom_mix(WeightedModel):
    """
    Beta-binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)),                                                                                             
    where p = exog @ params[:-1] and tau = params[-1].  This function fits the                                                                                          
    BetaBin params when samples are weighted by weights:                                                                                                                                                                                                                                                                                        
    max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params) 

    Adapated for varying tumor proportion.
    """
    ninstance = 0

    def nloglikeobs(self, params):
        a = (
            self.exog @ params[:-1] * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * params[-1]

        b = (
            (1 - self.exog @ params[:-1]) * self.tumor_prop
            + 0.5 * (1 - self.tumor_prop)
        ) * params[-1]

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(
            self.weights
        )

    def get_default_start_params(self):
        # DEPRECATE np.sum(self.exog.shape[1])
        return np.append(0.5 * np.ones(self.exog.shape[1]), 1)

    def get_ext_param_name(self):
        return "tau"

    def __post_init__(self):
        self.method = "lbfgs"
        self.bounds = self.exog.shape[1] * [(0., 1.)] + [(1.e-2, self.exposure.max())]
        
        assert self.tumor_prop is not None, "Tumor proportion must be defined."

        Weighted_BetaBinom_mix.ninstance += 1


class Weighted_BetaBinom_fixdispersion(WeightedModel):
    ninstance = 0
    
    # NB custom __init__ required to handle tau.
    def __init__(self, endog, exog, tau, weights, exposure, *args, tumor_prop=None, method="nm", snapshot=True, **kwargs):
        super().__init__(endog, exog, weights, exposure, *args, tumor_prop=tumor_prop, method=method, snapshot=snapshot, **kwargs)

        self.tumor_prop = tumor_prop

        self.tau = tau
        self.weights = weights
        self.exposure = exposure

        self.__post_init__()

        logger.info(
            f"Initializing {self.__class__.__name__} model for endog.shape = {endog.shape} and fixed dispersion = {tau}."
        )

    def nloglikeobs(self, params):
        a = (self.exog @ params) * self.tau
        b = (1 - self.exog @ params) * self.tau

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(
            self.weights
        )

    def get_default_start_params(self):
        return 0.1 * np.ones(self.exog.shape[1])

    def get_ext_param_name(self):
        return None
    
    def __post_init__(self):
        assert self.tumor_prop is None
        
        Weighted_BetaBinom_fixdispersion.ninstance += 1


class Weighted_BetaBinom_fixdispersion_mix(WeightedModel):
    # NB custom __init__ required to handle tau.
    def __init__(self, endog, exog, tau, weights, exposure, *args, tumor_prop=None, **kwargs):
        super().__init__(endog, exog, **kwargs)

        self.tumor_prop = tumor_prop

        self.tau = tau
        self.weights = weights
        self.exposure = exposure

        self.__post_init__()

        logger.info(
            f"Initializing {self.__class__.__name__} model for endog.shape = {endog.shape}."
        )

    def nloglikeobs(self, params):
        a = (
            self.exog @ params * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * self.tau

        b = (
            (1 - self.exog @ params) * self.tumor_prop + 0.5 * (1 - self.tumor_prop)
        ) * self.tau

        return -scipy.stats.betabinom.logpmf(self.endog, self.exposure, a, b).dot(
            self.weights
        )

    def get_default_start_params(self):
        return 0.1 * np.ones(self.exog.shape[1])

    def __post_init__(self):
        assert self.tumor_prop is not None, "Tumor proportion must be defined."

        Weighted_BetaBinom_fixdispersion_mix.ninstance += 1


class BAF_Binom(GenericLikelihoodModel):
    """
    Binomial model endog ~ BetaBin(exposure, tau * p, tau * (1 - p)), where p = exog @ params[:-1] and tau = params[-1].
    This function fits the BetaBin params when samples are weighted by weights: max_{params} \sum_{s} weights_s * log P(endog_s | exog_s; params)

    Used to estimate tumor proportion in utils_hmrf.

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
            if hasattr(self, "start_params"):
                start_params = self.start_params
            else:
                start_params = 0.5 / np.sum(self.exog.shape[1]) * np.ones(self.nparams)
        return super(BAF_Binom, self).fit(
            start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds
        )
