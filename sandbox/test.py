import timeit
from beta_binomial import parallel_G, serial_G, parallel_beta_binomial, parallel_beta_binomial_zeropoint
from scipy.stats import betabinom
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pylab as pl


def main():
    """
    N, k = 100, 1
    x, y = np.linspace(-100, 100, N), np.linspace(-100, 100, N)
    x, y = np.meshgrid(x, y)

    def serial():
        serial_G(k, x, y)

    def parallel():
        parallel_G(k, x, y)

    time_serial = timeit.timeit(serial, number=3)
    time_parallel = timeit.timeit(parallel, number=3)

    print(x.size)
    print("Serial method took {:.3} seconds".format(time_serial))
    print("Parallel method took {:.3} seconds".format(time_parallel))
    """


def get_basic_dataset(N):
    ks = np.random.randint(0, 25, N).astype(float)
    ns = ks + np.random.randint(0, 25, N).astype(float)

    aa = 100.0 * np.random.uniform(size=N)
    bb = 100.0 * np.random.uniform(size=N)

    return ks, ns, aa, bb


def get_mock_dataset():
    n_spots, n_states = (36210, 6)

    tumor_prop = np.random.uniform(0.91, 0.95, n_spots)

    encoder = OneHotEncoder()
    
    state = np.random.randint(0, 6, n_spots)
    exog = encoder.fit_transform(state.reshape(-1, 1))
    
    exposure = np.random.randint(1, 155704, n_spots).astype(float)

    params = np.array(6 * [0.08333333] + [1.0])

    endog = np.random.randint(0, 78365, n_spots).astype(float)

    a = (exog @ params[:-1] * tumor_prop + 0.5 * (1. - tumor_prop)) * params[-1]
    b = ((1. - exog @ params[:-1]) * tumor_prop + 0.5 * (1. - tumor_prop)) * params[-1]
    
    return endog, exposure, a, b


def get_betabinomial_speedup(dataset, ntrials=100, verbose=True):
    np.random.seed(42)

    ks, ns, aa, bb = dataset
    # state, params, tumor_prop, ns, ks, exog, aa, bb = get_mock_dataset()

    ws = np.ones_like(ks)
    
    exp = betabinom.logpmf(ks, ns, aa, bb).dot(ws)
    result = parallel_beta_binomial(ks, ns, aa, bb, ws)

    def serial():
        betabinom.logpmf(ks, ns, aa, bb).dot(ws)

    def parallel():
        parallel_beta_binomial(ks, ns, aa, bb, ws)

    time_serial = timeit.timeit(serial, number=ntrials)
    time_parallel = timeit.timeit(parallel, number=ntrials)

    speedup = time_serial / time_parallel

    if verbose:
        print(len(ks))
        print(f"Serial method took {time_serial:.3} seconds")
        print(f"Parallel method took {time_parallel:.3} seconds")
        print(f"Speedup factor: {speedup:.3}x")

    # print(exp)
    # print(result)
        
    assert np.allclose(exp, result, atol=1.0e-6, equal_nan=True)

    return speedup


def main():
    ns = np.logspace(1, 6, 6).astype(int)
    xs = [get_betabinomial_speedup(get_basic_dataset(n)) for n in ns]

    print()
    print(ns)
    print(xs)
    """
    pl.axhline(16.0, c="k", lw=0.5)
    pl.semilogx(ns, xs)

    pl.xlabel("# elements")
    pl.ylabel("Speed up [x]")

    pl.savefig("scaling.pdf")
    """
    """
    dataset = get_mock_dataset()
    
    get_betabinomial_speedup(dataset)
    """
    
if __name__ == "__main__":    
    main()
