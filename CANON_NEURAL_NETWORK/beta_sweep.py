"""
Beta sweep: Force the KL below the 25-nat wall.
Tests whether 25 nats is the true minimum or just where beta=0.001 landed.

Runs N=9, strategy_dim=9 at beta = {0.005, 0.01, 0.05}
Each for 30k steps (enough to see if error degrades gracefully or collapses).
"""
import subprocess, sys, os

betas = [0.005, 0.01, 0.05]
base_cmd = [
    sys.executable,
    os.path.join(os.path.dirname(__file__), "train_kolmogorov.py"),
    "--N", "9",
    "--strategy_dim", "9",
    "--steps", "30000",
    "--seed", "42",
]

for beta in betas:
    cmd = base_cmd + ["--beta", str(beta)]
    print(f"\n{'='*60}")
    print(f"  BETA SWEEP: beta={beta}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)
