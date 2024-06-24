import copy
import functools
import logging
import subprocess
import sys
from pathlib import Path

# import anndata
import numpy as np
import pandas as pd
import pylab as pl
import pyro

# import scanpy as sc
import scipy
import torch
from calicost.arg_parse import *
from calicost.find_integer_copynumber import *
from calicost.hmm_NB_BB_phaseswitch import *
from calicost.hmrf import *
from calicost.parse_input import *
from calicost.phasing import *
from calicost.utils_distribution_fitting import *
from calicost.utils_hmrf import *
from calicost.utils_IO import *
from calicost.utils_plotting import *
from calicost.hmm_NB_BB_nophasing_v2 import hmm_nophasing_v2
from calicost.sandbox.profile_context import ProfileContext
from calicost.compute_emission import compute_emission_probability_nb_betabinom_mix, convert_params
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def get_true_clone_cnas():
    clones = pd.read_csv("calicost/data/truth_clone_labels.tsv", header=0, index_col=0, sep="\t")
    cnas = pd.read_csv("calicost/data/true_cnas_per_bin.tsv", header=0, index_col=0, sep="\t")

    for label in ["clone_0", "clone_1", "clone_2", "normal"]:
        ucns, counts = np.unique(cnas[f"{label}_cn"], return_counts=True)

        print(f"{label}:")

        for ucn, count in zip(ucns, counts):
            print("\t", ucn, count)

    return clones, cnas


def get_data():
    bin_data = dict(np.load("calicost/data/binned_data.npz", allow_pickle=True))

    # lengths: sum of lengths = n_observations.
    lengths = bin_data["lengths"]
    single_X = bin_data["single_X"]

    single_base_nb_mean = bin_data["single_base_nb_mean"]
    single_total_bb_RD = bin_data["single_total_bb_RD"]

    # log_sitewise_transmat: n_observations, the log transition probability of phase switch, i.e. for known Z.
    log_sitewise_transmat = bin_data["log_sitewise_transmat"]
    single_tumor_prop = bin_data["single_tumor_prop"]

    return (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        single_tumor_prop,
    )


def aggregate_data(
    lengths,
    true_clones,
    single_X,
    single_base_nb_mean,
    single_total_bb_RD,
    log_sitewise_transmat,
):
    # aggregating counts across spots for each clone
    clone_index = [
        np.where(true_clones.labels == c)[0]
        for c in np.sort(true_clones.labels.unique())
    ]

    X, base_nb_mean, total_bb_RD = merge_pseudobulk_by_index(
        single_X, single_base_nb_mean, single_total_bb_RD, clone_index
    )

    n_clones = X.shape[2]

    X = np.vstack([X[:, 0, :].flatten("F"), X[:, 1, :].flatten("F")]).T.reshape(
        -1, 2, 1
    )
    
    base_nb_mean = base_nb_mean.flatten("F").reshape(-1, 1)
    total_bb_RD = total_bb_RD.flatten("F").reshape(-1, 1)
    log_sitewise_transmat = np.tile(log_sitewise_transmat, n_clones)
    lengths = np.tile(lengths, n_clones)

    return lengths, X, base_nb_mean, total_bb_RD, log_sitewise_transmat


# @profile
def profile_pipeline_baum_welch(*args, **kwargs):
    return pipeline_baum_welch(*args, **kwargs)


# kernprof -l -v run_hmm.py
if __name__ == "__main__":
    n_states = 7

    # what parameters should Baum-Welch optimize:
    # s for start probability,
    # m for negative binomial distribution params (transcripts),
    # p for Beta-binomial distribution params (baf)
    params = "smp"

    # HMM self-transition probability in the transition probability matrix
    t = 1.0 - 1.0e-4

    # random seed for initialization of negative binomial and Beta-binomial distribution mean parameters
    gmm_random_state = 0

    # whether to fix Negative Binomial dispersion parameter, otherwise infer
    fix_NB_dispersion = False

    # whether to fix Beta-Binomial dispersion parameter, otherwise infer
    fix_BB_dispersion = False

    # whether the Negative Binomial dispersion parameter is shared across all clones
    shared_NB_dispersion = True

    # whether the Beta-Binomial dispersion parameter is shared across all clones
    shared_BB_dispersion = True

    # This should be deprecated. We once tried to infer the transition matrix,
    is_diag = True

    # and this is the indicator for whether the transition matrix takes the "diagonal" form,
    # with the same self-transition across all states and the equal probability to other state.
    max_iter = 30
    tol = 1e-4

    # file to write the HMM results, None means return the results without writing to file.
    outputfilename = None
    true_clones, true_cnas = get_true_clone_cnas()

    (
        lengths,
        single_X,
        single_base_nb_mean,
        single_total_bb_RD,
        log_sitewise_transmat,
        single_tumor_prop,
    ) = get_data()
    
    if False:
        lengths, X, base_nb_mean, total_bb_RD, log_sitewise_transmat = aggregate_data(
            lengths,
            true_clones,
            single_X,
            single_base_nb_mean,
            single_total_bb_RD,
            log_sitewise_transmat,
        )
    else:
        X, base_nb_mean, total_bb_RD, tumor_prop = single_X, single_base_nb_mean, single_total_bb_RD.astype(float), single_tumor_prop

    if Path("init_log_mu.npy").exists() and Path("init_p_binom.npy").exists():
        init_log_mu = np.load("init_log_mu.npy")                                                                                                                                                                   
        init_p_binom = np.load("init_p_binom.npy")  
    else:
        # initialize negative binomial and Beta-binomial distribution mean parameters per clone
        init_log_mu, init_p_binom = initialization_by_gmm(
            n_states,
            X,
            base_nb_mean,
            total_bb_RD,
            params,
            random_state=gmm_random_state,
            in_log_space=False,
            only_minor=False,
        )
        
        np.save("init_log_mu.npy", init_log_mu)
        np.save("init_p_binom.npy", init_p_binom)

    """
    res = profile_pipeline_baum_welch(
        outputfilename,
        X,
        lengths,
        n_states,
        base_nb_mean,
        total_bb_RD,
        log_sitewise_transmat,
        hmmclass=hmm_nophasing_v2,
        params=params,
        t=t,
        random_state=gmm_random_state,
        fix_NB_dispersion=fix_NB_dispersion,
        shared_NB_dispersion=shared_NB_dispersion,
        fix_BB_dispersion=fix_BB_dispersion,
        shared_BB_dispersion=shared_BB_dispersion,
        is_diag=is_diag,
        init_log_mu=init_log_mu,
        init_p_binom=init_p_binom,
        init_alphas=None,
        init_taus=None,
        max_iter=max_iter,
        tol=tol,
    )
    """
    nrepeat = 1

    n_states = init_log_mu.shape[0]
    (n_obs, n_comp, n_spots) = X.shape
    
    tumor_prop = np.ones((n_obs, n_spots))

    alphas = np.random.uniform(low=0.0, high=1.0, size=(n_obs, n_spots))
    taus = np.random.uniform(low=0.0, high=1.0, size=(n_obs, n_spots))
    
    # NB jit compilation.
    compute_emission_probability_nb_betabinom_mix(
        X,
        base_nb_mean,
        init_log_mu,
        alphas,
        total_bb_RD,
	init_p_binom,
        taus,
        tumor_prop,
    )
    
    with ProfileContext() as context:
        for _ in range(nrepeat):
            """
            # NB exp. runtime 3.599 seconds.
            log_emission_rdr, log_emission_baf = hmm_nophasing_v2.compute_emission_probability_nb_betabinom_mix(
                X,
                base_nb_mean,
                init_log_mu,
                alphas,
                total_bb_RD,
                init_p_binom,
                taus,
                tumor_prop,
            )
            
            np.save("log_emission_rdr.npy", log_emission_rdr)
            np.save("log_emission_baf.npy", log_emission_baf)
            """
            log_emission_rdr, log_emission_baf = compute_emission_probability_nb_betabinom_mix(
                X,
                base_nb_mean,
                init_log_mu,
                alphas,
                total_bb_RD,
		init_p_binom,
                taus,
                tumor_prop,
            )

    
    truth_runtime = 3.599 # [seconds]        
    truth_log_emission_rdr = np.load("log_emission_rdr.npy")
    truth_log_emission_baf = np.load("log_emission_baf.npy")

    pl.loglog(-truth_log_emission_rdr.ravel(), -log_emission_rdr.ravel(), marker=',', lw=0.0, c='k')
    pl.xlabel("Original log(RDR)")
    pl.ylabel("New log(RDR)")
    pl.show()
    
    """
    pl.loglog(-truth_log_emission_baf.ravel(), -log_emission_baf.ravel(), marker=',', lw=0.0, c='k')
    pl.xlabel("Original log(BAF)")
    pl.ylabel("New log(BAF)")
    pl.show()
    """
    """
    # plot the data colored by the MAP estimate of the hidden states
    RDR = X[:, 0, 0] / base_nb_mean[:, 0]
    BAF = X[:, 1, 0] / total_bb_RD[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(20, 4), dpi=150, facecolor="white")

    seaborn.scatterplot(
        x=np.arange(len(RDR)),
        y=RDR,
        hue=res["pred_cnv"],
        palette="tab10",
        linewidth=0,
        s=10,
        ax=axes[0],
    )

    seaborn.scatterplot(
        x=np.arange(len(BAF)),
        y=BAF,
        hue=res["pred_cnv"],
        palette="tab10",
        linewidth=0,
        s=10,
        ax=axes[1],
    )

    # Multiple HMM sequences are concatenated, put a vertical bar to separate them
    for i in range(len(lengths)):
        axes[0].axvline(np.sum(lengths[:i]), color="black", linestyle="--")
        axes[1].axvline(np.sum(lengths[:i]), color="black", linestyle="--")

    fig.tight_layout()
    fig.show()
    """
    print("Done.")
