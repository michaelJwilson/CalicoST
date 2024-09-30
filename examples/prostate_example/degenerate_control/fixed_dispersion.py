import sys
import dill
import logging
import numpy as np

from scipy.stats import betabinom
from scipy.optimize import minimize_scalar
from calicost.utils_distribution_fitting import (
    Weighted_BetaBinom,
    Weighted_BetaBinom_fixdispersion,
)

logger = logging.getLogger("calicost")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s - %(process)d - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
)

handler.setFormatter(formatter)
logger.addHandler(handler)

def solve(endog, exog, exposure, weights):
    logger.info("Solving for sample state means...")

    means = []
    
    for selection in exog.T:
        selection = selection.astype(bool)

        mean = np.sum(endog[selection] * weights[selection]) / np.sum(
            exposure[selection] * weights[selection]
        )
        
        means.append(mean)

        ps = endog[selection] / exposure[selection]
                
    means = np.array(means)
    
    logger.info(f"Found sample state means={means}.")

    def nloglikeobs(tau):
        a = (exog @ means) * tau
        b = (1.0 - exog @ means) * tau

        return -betabinom.logpmf(endog, exposure, a, b).dot(
            weights
        )

    result = minimize_scalar(nloglikeobs, bracket=[1., exposure.max()], method="Golden", tol=1.e-2)

    print(result)

    return np.concatenate([means, np.array([result.x])])


if __name__ == "__main__":
    fpath = "snapshots/weighted_betabinom/weighted_betabinom_snapshot_1.dill"

    with open(fpath, "rb") as f:
        ds = dill.load(f)

    logger.info(f"Loaded {ds.get_ninstance()}th instance of {ds.__class__.__name__}")

    # NB {"Nelder-Mead"}
    ds.bounds = ds.exog.shape[1] * [(0., 1.)] + [(1.e-2, ds.exposure.max())]

    ds.method = "Powell"
    best_fit = ds.fit()
    
    ds.method = "Nelder-Mead"
    best_fit = ds.fit()
    
    start_params = ds.get_default_start_params()
    start_params[-1] = best_fit[-1]
    
    new_fit = ds.fit(start_params=start_params)
    
    ds.method = "Powell"
    new_fit = ds.fit(start_params=start_params)
    
    # NB solve for sample means and tau.
    est_params = solve(ds.endog, ds.exog, ds.exposure, ds.weights)

    replica = Weighted_BetaBinom(
        ds.endog,
        ds.exog,
        ds.weights,
        ds.exposure,
        snapshot=False,
    )
    replica.method = "Powell"    
    conditioned_fit = replica.fit(start_params=est_params, xtol=1.e-2)
    
    """
    best_fit_tau = best_fit.params[-1]

    fixed_model = Weighted_BetaBinom_fixdispersion(
        ds.endog, ds.exog, best_fit_tau, ds.weights, ds.exposure
    )
    fixed_model.method = "nm"

    logger.info(
        f"Loaded {fixed_model.get_ninstance()}th instance of {fixed_model.__class__.__name__}"
    )

    start_params = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    fixed_best_fit = fixed_model.fit(start_params=start_params)
    """
