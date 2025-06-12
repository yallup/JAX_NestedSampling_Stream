import time
import jax
import jax.numpy as jnp
from functools import partial

from data import get_data
from prior import sample_from_priors
from loglikelihood import loglikelihood

print("JAX devices:", jax.devices())

# Load test data
dict_data = jnp.load("dict_data.npz", allow_pickle=True)
dict_data = {k: v for k, v in dict_data.items()}
dict_data_jax = jax.tree.map(jnp.array, dict_data)

# Create likelihood function
logl = partial(loglikelihood, dict_data=dict_data_jax)

def test_vectorization_performance(logl, batch_sizes=[1, 2, 5, 10], n_trials=2):
    """Test vectorization performance across different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        times = []
        
        for trial in range(n_trials):
            # Generate random parameters
            key = jax.random.PRNGKey(trial)
            params = sample_from_priors(key, batch_size)
            
            # Warmup
            _ = jax.vmap(logl)(params).block_until_ready()
            
            # Time the execution
            start = time.time()
            result = jax.vmap(logl)(params).block_until_ready()
            end = time.time()
            
            exec_time = end - start
            times.append(exec_time)
            print(f"  Trial {trial + 1}: {exec_time:.4f}s")
        
        avg_time = sum(times) / len(times)
        time_per_eval = avg_time / batch_size
        results[batch_size] = {
            'avg_time': avg_time,
            'time_per_eval': time_per_eval,
            'speedup_vs_batch1': None
        }
        print(f"  Average: {avg_time:.4f}s ({time_per_eval:.4f}s per evaluation)")
    
    # Calculate speedup relative to batch size 1 (or smallest batch)
    baseline_time_per_eval = results[min(batch_sizes)]['time_per_eval']
    for batch_size in batch_sizes:
        speedup = baseline_time_per_eval / results[batch_size]['time_per_eval']
        results[batch_size]['speedup_vs_batch1'] = speedup
        print(f"Batch {batch_size}: {speedup:.2f}x speedup per evaluation")
    
    return results

if __name__ == "__main__":
    print("Testing vectorization performance...")
    results = test_vectorization_performance(logl)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for batch_size, metrics in results.items():
        print(f"Batch {batch_size:3d}: {metrics['avg_time']:.4f}s total, "
              f"{metrics['time_per_eval']:.4f}s per eval, "
              f"{metrics['speedup_vs_batch1']:.2f}x speedup")