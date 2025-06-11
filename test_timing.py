import time
from tqdm import tqdm

import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise, log_weights
import time
import matplotlib.pyplot as plt

print(jax.devices())

from data import get_data
from prior import prior_dists, logprior, sample_from_priors
from loglikelihood import loglikelihood

q_true = 0.8
seed = 42
sigma = 1
n_live = 500
PATH_SAVE = f"./"

dict_data = get_data(q_true, seed, sigma)


logl = partial(loglikelihood, dict_data=jax.tree.map(jnp.array, dict_data))


def time_likelihood(logl, n, rng_key=jax.random.PRNGKey(0)):
    """Function to time the loglikelihood function."""
    start = time.time()
    jax.vmap(logl)(sample_from_priors(rng_key, n)).block_until_ready()
    end = time.time()
    return end - start


times = []
jit_times = []
xs = [1, 10, 100, 500, 1000]
for i in xs:
    # warmup to jit it
    t_warm = time_likelihood(logl, i)
    print(f"Warmup time for {i} evaluations: {t_warm:.4f} seconds")
    t = time_likelihood(logl, i)
    jit_times.append(t_warm)
    times.append(t)
    print(f"Time taken for {i} evaluations: {t:.4f} seconds")


plt.plot(xs, times, marker="o", label="jitted")
plt.plot(xs, jit_times, marker="o", label="Compiled (warmup)")

plt.xlabel("Number of evaluations")
plt.ylabel("Time (seconds)")
plt.title("Time taken for loglikelihood evaluations")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.savefig("timing_loglikelihood.png")
