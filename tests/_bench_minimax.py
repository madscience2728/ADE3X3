"""Quick benchmark: measure per-eval wall time with minimax local search."""
import json, numpy as np, sys, os, time, importlib

os.environ['STEP84_ALGEBRAIC_MODE'] = '1'
os.environ['STEP84_SUPPORT_MAX'] = '9'
os.environ['STEP84_MINIMAX_SWEEPS'] = '1'
os.environ['STEP84_MINIMAX_FINE_RANGE'] = '0.003'
os.environ['STEP84_ISLAND_POPULATIONS'] = '4,4'
os.environ['STEP84_WARM_SEEDS'] = '2'
sys.path.insert(0, '.')

import src.ade3x3.steps.ade3x3_step84_metaheuristic_rank19_search as mod
importlib.reload(mod)
from src.ade3x3.steps.ade3x3_step84_metaheuristic_rank19_search import (
    SparseTermGene, Individual, evaluate_individual_worker
)

d = json.load(open('optimized_als_r10_at_0.09.json'))
genes = []
for td in d['terms']:
    genes.append(SparseTermGene(
        tuple(int(i) for i in td['alpha_support']),
        tuple(int(i) for i in td['beta_support']),
        tuple(int(i) for i in td['gamma_support']),
        np.array(td['alpha_values'], dtype=np.float64),
        np.array(td['beta_values'], dtype=np.float64),
        np.array(td['gamma_values'], dtype=np.float64),
    ))
ind = Individual(genes, 'bench')
payload = {'eval_seed': 42, 'individual': ind.to_dict()}

# Warm up
evaluate_individual_worker(payload)

# Benchmark
times = []
for i in range(3):
    t0 = time.perf_counter()
    r = evaluate_individual_worker(payload)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    times.append(elapsed)
    timing = r["timing"]
    print(f"Eval {i+1}: {elapsed:.3f}s  fitness={r['fitness']:.10f}  "
          f"minimax_ms={timing.get('minimax_ms', 0):.0f}  "
          f"als_ms={timing.get('als_ms', 0):.0f}  "
          f"polish_ms={timing.get('polish_ms', 0):.0f}")

print(f"\nMean eval time: {np.mean(times):.3f}s")
print(f"Minimax share: {r['timing'].get('minimax_ms', 0)/1000:.3f}s")
print(f"ALS share: {r['timing'].get('als_ms', 0)/1000:.3f}s")
print(f"Polish share: {r['timing'].get('polish_ms', 0)/1000:.3f}s")
