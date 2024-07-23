import timeit
from beta_binomial import parallel_G, serial_G, parallel_beta_binomial
from scipy.stats import betabinom

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

def get_betabinomial_speedup(N, ntrials=50, verbose=True):
    np.random.seed(42)
    
    ks = np.random.randint(0, 25, N).astype(float)
    ns = ks + np.random.randint(0, 25, N).astype(float)

    aa = 100. * np.random.uniform(size=N)
    bb = 100. * np.random.uniform(size=N)

    exp = betabinom.logpmf(ks, ns, aa, bb)
    result = parallel_beta_binomial(ks, ns, aa, bb)

    def serial():                                                                                                                                                                               
        betabinom.logpmf(ks, ns, aa, bb)
        
    def parallel():                                                                                                                                                                             
        parallel_beta_binomial(ks, ns, aa, bb)
    
    time_serial = timeit.timeit(serial, number=ntrials)
    time_parallel = timeit.timeit(parallel, number=ntrials)

    speedup = time_serial / time_parallel

    if verbose:
        print(N)                                                                                                                                                                                
        print(f"Serial method took {time_serial:.3} seconds")
        print(f"Parallel method took {time_parallel:.3} seconds")
        print(f"Speedup factor: {speedup:.3}x")
    
    assert np.allclose(exp, result, atol=1.e-6, equal_nan=True)

    return speedup

def main():
    ns = np.logspace(1, 6, 6).astype(int)
    xs = [get_betabinomial_speedup(N, ntrials=10) for N in ns]

    print(ns)
    print(xs)

    pl.semilogx(ns, xs)
    pl.savefig("scaling.pdf")
    
if __name__ == "__main__":
    main()
