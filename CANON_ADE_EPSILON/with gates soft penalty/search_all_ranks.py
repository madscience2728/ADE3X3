import argparse
import json
import math
import os
import tempfile
import threading
import time
from multiprocessing import Manager, Pool, freeze_support

import numpy as np
from scipy.optimize import minimize


DIM = 9
GATE_REG = 1e-6
GATE3_EPS = 1e-4
SOFTMIN_TEMP = 0.1
STATUS_SECONDS = 20.0
ALL_RANKS = [13, 19, 20, 21, 22, 23, 27]
RANK_WEIGHTS = {13: 2, 19: 3, 20: 4, 21: 4, 22: 4, 23: 2, 27: 2}
DEAD_PAIRS = [(s, t) for s in range(3) for t in range(3) if s != t]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CANON_ROOT = os.path.join(REPO_ROOT, "CANON_ADE_EPSILON")
CLIFF_RESULTS_DIR = os.path.join(CANON_ROOT, "results", "cliff_search")

WORKER_CONFIGS = [
    {"worker_id": 0, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_0"},
    {"worker_id": 1, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_1"},
    {"worker_id": 2, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_2"},
    {"worker_id": 3, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_3"},
    {"worker_id": 4, "lam2": 2.0, "lam3": 0.0, "rep": 0.0, "label": "gate2_strong"},
    {"worker_id": 5, "lam2": 0.5, "lam3": 0.0, "rep": 0.0, "label": "gate2_medium"},
    {"worker_id": 6, "lam2": 0.1, "lam3": 0.0, "rep": 0.0, "label": "gate2_weak"},
    {"worker_id": 7, "lam2": 0.0, "lam3": 2.0, "rep": 0.0, "label": "gate3_strong"},
    {"worker_id": 8, "lam2": 0.0, "lam3": 0.5, "rep": 0.0, "label": "gate3_medium"},
    {"worker_id": 9, "lam2": 0.0, "lam3": 0.1, "rep": 0.0, "label": "gate3_weak"},
    {"worker_id": 10, "lam2": 0.5, "lam3": 0.5, "rep": 0.0, "label": "balanced_high"},
    {"worker_id": 11, "lam2": 0.1, "lam3": 0.1, "rep": 0.0, "label": "balanced_mid"},
    {"worker_id": 12, "lam2": 0.01, "lam3": 0.01, "rep": 0.0, "label": "balanced_low"},
    {"worker_id": 13, "lam2": 0.0, "lam3": 0.0, "rep": 1.0, "label": "repulse_strong"},
    {"worker_id": 14, "lam2": 0.0, "lam3": 0.0, "rep": 0.5, "label": "repulse_med"},
    {"worker_id": 15, "lam2": 0.0, "lam3": 0.0, "rep": 0.1, "label": "repulse_weak"},
    {"worker_id": 16, "lam2": 2.0, "lam3": 0.0, "rep": 0.5, "label": "gate2_plus_repulse"},
    {"worker_id": 17, "lam2": 0.0, "lam3": 2.0, "rep": 0.5, "label": "gate3_plus_repulse"},
    {"worker_id": 18, "lam2": 1.0, "lam3": 0.0, "rep": -1.0, "label": "gate2_attracted"},
    {"worker_id": 19, "lam2": 0.0, "lam3": 1.0, "rep": -1.0, "label": "gate3_attracted"},
    {"worker_id": 20, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_lottery"},
    {"worker_id": 21, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_lottery"},
    {"worker_id": 22, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_lottery"},
    {"worker_id": 23, "lam2": 0.0, "lam3": 0.0, "rep": 0.0, "label": "pure_frob_lottery"},
]


def build_target_tensor():
    tensor = np.zeros((9, 9, 9), dtype=np.float64)
    for r in range(3):
        for s in range(3):
            for u in range(3):
                tensor[r * 3 + s, s * 3 + u, r * 3 + u] = 1.0
    return tensor


TENSOR = build_target_tensor()


def resolve_checkpoint_dir(path):
    if os.path.isabs(path):
        return path
    return os.path.join(CANON_ROOT, path)


def unpack_factors(x, rank):
    alpha = x[: rank * DIM].reshape(rank, DIM)
    beta = x[rank * DIM : 2 * rank * DIM].reshape(rank, DIM)
    gamma = x[2 * rank * DIM :].reshape(rank, DIM)
    return alpha, beta, gamma


def step51(alpha3, beta3):
    rank = alpha3.shape[0]
    sigma = np.einsum("krs,ksu->kru", alpha3, beta3).reshape(rank, 9)
    e0 = (alpha3[:, :, 0:1] * beta3[:, 0:1, :]).reshape(rank, 9)
    e1 = (alpha3[:, :, 1:2] * beta3[:, 1:2, :]).reshape(rank, 9)
    e2 = (alpha3[:, :, 2:3] * beta3[:, 2:3, :]).reshape(rank, 9)
    h_mat = np.hstack([e0 - e1, e1 - e2])
    delta = np.hstack(
        [
            (alpha3[:, :, s : s + 1] * beta3[:, t : t + 1, :]).reshape(rank, 9)
            for s, t in DEAD_PAIRS
        ]
    )
    return sigma, h_mat, delta


def gate2_penalty_with_grads(h_mat, delta, reg=GATE_REG):
    gram = h_mat.T @ h_mat + reg * np.eye(h_mat.shape[1])
    coeff = np.linalg.solve(gram, h_mat.T @ delta)
    delta_perp = delta - h_mat @ coeff
    penalty = float(np.sum(delta_perp * delta_perp))
    grad_delta = delta_perp.copy()
    grad_h = -delta_perp @ coeff.T
    return penalty, grad_h, grad_delta, delta_perp


def gate3_penalty_with_grads(sigma, h_mat, delta, reg=GATE_REG, eps_sv=GATE3_EPS):
    nuisance = np.hstack([h_mat, delta])
    gram_n = nuisance.T @ nuisance + reg * np.eye(nuisance.shape[1])
    coeff_sigma = np.linalg.solve(gram_n, nuisance.T @ sigma)
    sigma_perp = sigma - nuisance @ coeff_sigma
    gram_sigma = sigma_perp.T @ sigma_perp + eps_sv * np.eye(9)
    sign, logdet = np.linalg.slogdet(gram_sigma)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e6, np.zeros_like(sigma), np.zeros_like(h_mat), np.zeros_like(delta)

    penalty = -float(logdet)
    inv_gram_sigma = np.linalg.solve(gram_sigma, np.eye(9))
    grad_sigma_perp = 2.0 * sigma_perp @ inv_gram_sigma
    coeff_grad = np.linalg.solve(gram_n, nuisance.T @ grad_sigma_perp)
    grad_sigma = grad_sigma_perp - nuisance @ coeff_grad
    grad_nuisance = -grad_sigma_perp @ coeff_sigma.T
    grad_h = grad_nuisance[:, :18]
    grad_delta = grad_nuisance[:, 18:]
    return penalty, grad_sigma, grad_h, grad_delta


def softmin_with_weights(a_val, b_val, temp=SOFTMIN_TEMP):
    pivot = min(a_val, b_val)
    exp_a = math.exp(-(a_val - pivot) / temp)
    exp_b = math.exp(-(b_val - pivot) / temp)
    denom = exp_a + exp_b
    soft = pivot - temp * math.log(denom)
    return soft, exp_a / denom, exp_b / denom


def backprop_step51(alpha3, beta3, grad_sigma=None, grad_h=None, grad_delta=None):
    rank = alpha3.shape[0]
    grad_alpha3 = np.zeros_like(alpha3)
    grad_beta3 = np.zeros_like(beta3)

    if grad_sigma is not None:
        grad_sigma3 = grad_sigma.reshape(rank, 3, 3)
        grad_alpha3 += np.einsum("kru,ksu->krs", grad_sigma3, beta3)
        grad_beta3 += np.einsum("kru,krs->ksu", grad_sigma3, alpha3)

    if grad_h is not None:
        grad_h1 = grad_h[:, :9].reshape(rank, 3, 3)
        grad_h2 = grad_h[:, 9:].reshape(rank, 3, 3)
        grad_e0 = grad_h1
        grad_e1 = -grad_h1 + grad_h2
        grad_e2 = -grad_h2
        for idx, grad_e in enumerate([grad_e0, grad_e1, grad_e2]):
            grad_alpha3[:, :, idx] += np.sum(grad_e * beta3[:, idx : idx + 1, :], axis=2)
            grad_beta3[:, idx, :] += np.sum(grad_e * alpha3[:, :, idx : idx + 1], axis=1)

    if grad_delta is not None:
        for pair_idx, (s_idx, t_idx) in enumerate(DEAD_PAIRS):
            grad_block = grad_delta[:, pair_idx * 9 : (pair_idx + 1) * 9].reshape(rank, 3, 3)
            grad_alpha3[:, :, s_idx] += np.sum(grad_block * beta3[:, t_idx : t_idx + 1, :], axis=2)
            grad_beta3[:, t_idx, :] += np.sum(grad_block * alpha3[:, :, s_idx : s_idx + 1], axis=1)

    return grad_alpha3, grad_beta3


def objective_and_grad(x, rank, config, no_gate):
    alpha, beta, gamma = unpack_factors(x, rank)
    approx = np.einsum("ki,kj,kl->ijl", alpha, beta, gamma)
    error = approx - TENSOR

    frob_loss = 0.5 * float(np.sum(error * error))
    grad_alpha = np.einsum("ijl,kj,kl->ki", error, beta, gamma)
    grad_beta = np.einsum("ijl,ki,kl->kj", error, alpha, gamma)
    grad_gamma = np.einsum("ijl,ki,kj->kl", error, alpha, beta)

    lam2 = 0.0 if no_gate else config["lam2"]
    lam3 = 0.0 if no_gate else config["lam3"]
    rep = 0.0 if no_gate else config["rep"]

    total_loss = frob_loss
    if lam2 == 0.0 and lam3 == 0.0 and rep == 0.0:
        grad = np.concatenate([grad_alpha.ravel(), grad_beta.ravel(), grad_gamma.ravel()])
        return total_loss, grad

    alpha3 = alpha.reshape(rank, 3, 3)
    beta3 = beta.reshape(rank, 3, 3)
    sigma, h_mat, delta = step51(alpha3, beta3)

    gate2_pen, gate2_grad_h, gate2_grad_delta, _ = gate2_penalty_with_grads(h_mat, delta)
    gate3_pen, gate3_grad_sigma, gate3_grad_h, gate3_grad_delta = gate3_penalty_with_grads(sigma, h_mat, delta)

    gate2_grad_alpha3, gate2_grad_beta3 = backprop_step51(alpha3, beta3, grad_h=gate2_grad_h, grad_delta=gate2_grad_delta)
    gate3_grad_alpha3, gate3_grad_beta3 = backprop_step51(
        alpha3,
        beta3,
        grad_sigma=gate3_grad_sigma,
        grad_h=gate3_grad_h,
        grad_delta=gate3_grad_delta,
    )

    total_loss += lam2 * gate2_pen + lam3 * gate3_pen
    grad_alpha += lam2 * gate2_grad_alpha3.reshape(rank, DIM) + lam3 * gate3_grad_alpha3.reshape(rank, DIM)
    grad_beta += lam2 * gate2_grad_beta3.reshape(rank, DIM) + lam3 * gate3_grad_beta3.reshape(rank, DIM)

    if rep != 0.0:
        soft, w2, w3 = softmin_with_weights(gate2_pen, gate3_pen)
        total_loss -= rep * soft
        grad_alpha -= rep * (w2 * gate2_grad_alpha3.reshape(rank, DIM) + w3 * gate3_grad_alpha3.reshape(rank, DIM))
        grad_beta -= rep * (w2 * gate2_grad_beta3.reshape(rank, DIM) + w3 * gate3_grad_beta3.reshape(rank, DIM))

    grad = np.concatenate([grad_alpha.ravel(), grad_beta.ravel(), grad_gamma.ravel()])
    return total_loss, grad


def frobenius_residual(x, rank):
    alpha, beta, gamma = unpack_factors(x, rank)
    approx = np.einsum("ki,kj,kl->ijl", alpha, beta, gamma)
    return float(np.linalg.norm(approx - TENSOR))


def rank_with_tol(matrix, tol=1e-8):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    threshold = max(tol, tol * singular_values[0])
    return int(np.sum(singular_values > threshold))


def gate_metrics(x, rank):
    alpha, beta, _ = unpack_factors(x, rank)
    alpha3 = alpha.reshape(rank, 3, 3)
    beta3 = beta.reshape(rank, 3, 3)
    sigma, h_mat, delta = step51(alpha3, beta3)
    _, _, _, delta_perp = gate2_penalty_with_grads(h_mat, delta)
    nuisance = np.hstack([h_mat, delta])
    augmented = np.hstack([sigma, nuisance])
    rank_h = rank_with_tol(h_mat)
    rank_n = rank_with_tol(nuisance)
    rank_aug = rank_with_tol(augmented)
    return {
        "delta_leak": int(rank_n - rank_h),
        "aug_gap": int(rank_aug - rank_n),
        "D_perp": float(np.linalg.norm(delta_perp)),
    }


def precision_closure(rank, frob):
    return {
        "float16": 3.0 * rank * (2.0 ** -10) >= frob,
        "bfloat16": 3.0 * rank * (2.0 ** -7) >= frob,
        "int8": 3.0 * rank * (2.0 ** -8) >= frob,
    }


def min_bits(rank, frob):
    if frob <= 0.0:
        return float("inf")
    return float(np.log2(3.0 * rank / frob))


def checkpoint_path(checkpoint_dir, rank):
    return os.path.join(checkpoint_dir, f"rank{rank}_best.json")


def load_solution(path, rank):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    alpha = np.asarray(data["alpha"], dtype=np.float64)
    beta = np.asarray(data["beta"], dtype=np.float64)
    gamma = np.asarray(data["gamma"], dtype=np.float64)
    if alpha.shape != (rank, DIM) or beta.shape != (rank, DIM) or gamma.shape != (rank, DIM):
        raise ValueError(f"Unexpected factor shape in {path}")
    x = np.concatenate([alpha.ravel(), beta.ravel(), gamma.ravel()])
    return data, x, frobenius_residual(x, rank)


def maybe_load_checkpoint(rank, checkpoint_dir):
    candidates = [
        checkpoint_path(checkpoint_dir, rank),
        os.path.join(CLIFF_RESULTS_DIR, f"rank{rank}_best.json"),
    ]
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            data, x, frob = load_solution(candidate, rank)
        except Exception:
            continue
        return candidate, data, x, frob
    return None, None, None, None


def maybe_load_higher_rank_best(rank, checkpoint_dir):
    higher_rank = rank + 1
    if higher_rank not in ALL_RANKS:
        return None
    candidate, _, x, _ = maybe_load_checkpoint(higher_rank, checkpoint_dir)
    return None if candidate is None else x


def atomic_write_checkpoint(checkpoint_dir, rank, x, frob, worker_id, worker_label):
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics = gate_metrics(x, rank)
    alpha, beta, gamma = unpack_factors(x, rank)
    payload = {
        "rank": rank,
        "frobenius": float(frob),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "gamma": gamma.tolist(),
        "worker_id": int(worker_id),
        "worker_label": worker_label,
        "delta_perp": metrics["D_perp"],
        "delta_leak": metrics["delta_leak"],
        "aug_gap": metrics["aug_gap"],
    }

    fd, temp_path = tempfile.mkstemp(prefix=f"rank{rank}_", suffix=".json.tmp", dir=checkpoint_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temp_path, checkpoint_path(checkpoint_dir, rank))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def random_init(rng, rank, config):
    if config["label"] == "pure_frob_lottery":
        scale = rng.uniform(0.1, 3.0) / math.sqrt(rank)
    else:
        scale = rng.uniform(0.3, 1.5) / math.sqrt(rank)
    return rng.standard_normal(3 * rank * DIM) * scale


def perturb_best_init(rng, shared_state, rank):
    best_x = shared_state.get(get_best_x_key(rank))
    if best_x is None:
        return None
    sigma = rng.uniform(0.002, 0.1)
    return np.asarray(best_x, dtype=np.float64) + rng.standard_normal(3 * rank * DIM) * sigma


def delete_row_init(rng, higher_rank_x, rank):
    if higher_rank_x is None:
        return None
    higher_alpha, higher_beta, higher_gamma = unpack_factors(np.asarray(higher_rank_x, dtype=np.float64), rank + 1)
    drop_idx = int(rng.integers(0, rank + 1))
    keep_mask = np.ones(rank + 1, dtype=bool)
    keep_mask[drop_idx] = False
    alpha = higher_alpha[keep_mask]
    beta = higher_beta[keep_mask]
    gamma = higher_gamma[keep_mask]
    return np.concatenate([alpha.ravel(), beta.ravel(), gamma.ravel()])


def choose_init(rng, rank, config, shared_state, higher_rank_x):
    pick = rng.random()
    if pick < 0.60:
        return random_init(rng, rank, config)
    if pick < 0.90:
        candidate = perturb_best_init(rng, shared_state, rank)
        return candidate if candidate is not None else random_init(rng, rank, config)
    candidate = delete_row_init(rng, higher_rank_x, rank)
    if candidate is not None:
        return candidate
    candidate = perturb_best_init(rng, shared_state, rank)
    return candidate if candidate is not None else random_init(rng, rank, config)


def update_count(shared_counts, worker_id):
    shared_counts[worker_id] = int(shared_counts.get(worker_id, 0)) + 1


def update_hit(shared_hits, worker_id):
    shared_hits[worker_id] = int(shared_hits.get(worker_id, 0)) + 1


def get_best_key(rank):
    return f"best_f_{rank}"


def get_best_x_key(rank):
    return f"best_x_{rank}"


def get_best_worker_id_key(rank):
    return f"best_worker_id_{rank}"


def get_best_worker_label_key(rank):
    return f"best_worker_label_{rank}"


def allocate_rank_workers(active_ranks, total_workers):
    if len(active_ranks) == 1:
        return {active_ranks[0]: total_workers}

    total_weight = sum(RANK_WEIGHTS[rank] for rank in active_ranks)
    raw = {rank: total_workers * RANK_WEIGHTS[rank] / total_weight for rank in active_ranks}
    alloc = {rank: int(math.floor(raw[rank])) for rank in active_ranks}
    assigned = sum(alloc.values())

    if total_workers >= len(active_ranks):
        for rank in active_ranks:
            if alloc[rank] == 0:
                alloc[rank] = 1
                assigned += 1

    while assigned > total_workers:
        rank = max(active_ranks, key=lambda value: (alloc[value], raw[value] - math.floor(raw[value])))
        if alloc[rank] > 1:
            alloc[rank] -= 1
            assigned -= 1
        else:
            break

    remaining = total_workers - assigned
    if remaining > 0:
        remainders = sorted(active_ranks, key=lambda value: (raw[value] - math.floor(raw[value]), RANK_WEIGHTS[value]), reverse=True)
        for idx in range(remaining):
            alloc[remainders[idx % len(remainders)]] += 1

    return alloc


def build_worker_assignments(active_ranks, total_workers):
    allocations = allocate_rank_workers(active_ranks, total_workers)
    rank_slots = []
    while len(rank_slots) < total_workers:
        for rank in active_ranks:
            if allocations[rank] > rank_slots.count(rank):
                rank_slots.append(rank)
            if len(rank_slots) == total_workers:
                break

    assignments = []
    for worker_id in range(total_workers):
        config = dict(WORKER_CONFIGS[worker_id])
        config["assigned_rank"] = rank_slots[worker_id]
        assignments.append(config)
    return allocations, assignments


def initialize_rank_state(shared_state, active_ranks, checkpoint_dir):
    loaded = []
    for rank in active_ranks:
        loaded_path, loaded_data, loaded_x, loaded_f = maybe_load_checkpoint(rank, checkpoint_dir)
        if loaded_path is not None and loaded_x is not None and loaded_f is not None:
            shared_state[get_best_key(rank)] = float(loaded_f)
            shared_state[get_best_x_key(rank)] = loaded_x
            shared_state[get_best_worker_id_key(rank)] = int(loaded_data.get("worker_id", -1)) if loaded_data is not None else -1
            shared_state[get_best_worker_label_key(rank)] = loaded_data.get("worker_label", "loaded_checkpoint") if loaded_data is not None else "none"
            print(f"  R={rank}: loaded frob={float(loaded_f):.6f} from {loaded_path}", flush=True)
        else:
            shared_state[get_best_key(rank)] = float("inf")
            shared_state[get_best_x_key(rank)] = None
            shared_state[get_best_worker_id_key(rank)] = -1
            shared_state[get_best_worker_label_key(rank)] = "none"
            print(f"  R={rank}: cold start", flush=True)
        if loaded_path is not None and loaded_x is not None and loaded_f is not None:
            loaded.append((rank, loaded_path, loaded_x, float(loaded_f), shared_state[get_best_worker_label_key(rank)]))
    return loaded


def worker_loop(args):
    rank, config, shared_state, state_lock, start_time, wall_seconds, no_gate, checkpoint_dir, higher_rank_x = args
    worker_id = config["worker_id"]
    worker_label = config["label"]
    rng = np.random.default_rng(rank * 100_003 + worker_id * 9_973 + os.getpid())

    while True:
        if wall_seconds > 0 and time.time() - start_time >= wall_seconds:
            break

        x0 = choose_init(rng, rank, config, shared_state, higher_rank_x)
        candidate_x = None
        candidate_f = float("inf")
        improved = False
        is_owner = False

        try:
            result = minimize(
                lambda vec: objective_and_grad(vec, rank, config, no_gate),
                x0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-11},
            )
            candidate_x = np.asarray(result.x, dtype=np.float64)
            candidate_f = frobenius_residual(candidate_x, rank)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            candidate_x = None

        with state_lock:
            update_count(shared_state["counts"], worker_id)
            shared_state["last_frob"][worker_id] = float(candidate_f) if np.isfinite(candidate_f) else float("inf")
            current_best = float(shared_state.get(get_best_key(rank), np.inf))
            improved = candidate_x is not None and candidate_f < current_best
            if improved:
                shared_state[get_best_key(rank)] = float(candidate_f)
                shared_state[get_best_x_key(rank)] = candidate_x
                shared_state[get_best_worker_id_key(rank)] = worker_id
                shared_state[get_best_worker_label_key(rank)] = worker_label
                update_hit(shared_state["hits"], worker_id)
                is_owner = True

        if improved and is_owner and candidate_x is not None:
            atomic_write_checkpoint(checkpoint_dir, rank, candidate_x, candidate_f, worker_id, worker_label)

    return worker_id


def print_worker_config_table(assignments, workers):
    print("Worker config table:", flush=True)
    print(" ID | rank | lam2  | lam3  | rep   | label", flush=True)
    print("-" * 52, flush=True)
    for config in assignments:
        active = "*" if config["worker_id"] < workers else " "
        print(
            f"{active}{config['worker_id']:>2} | {config['assigned_rank']:>4} | {config['lam2']:>5.2f} | {config['lam3']:>5.2f} | {config['rep']:>5.2f} | {config['label']}",
            flush=True,
        )
    print("* active workers", flush=True)


def print_startup_summary(active_ranks, workers, checkpoint_dir, assignments, loaded_rows, shared_state):
    print(f"Ranks: {active_ranks}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print_worker_config_table(assignments, workers)


def print_status(active_ranks, assignments, workers, shared_state):
    global_best = (None, float("inf"), None, None)

    print("\n" + "=" * 108, flush=True)
    print("Per-rank best:", flush=True)
    for rank in active_ranks:
        best_f = float(shared_state.get(get_best_key(rank), np.inf))
        best_x = shared_state.get(get_best_x_key(rank))
        best_label = shared_state.get(get_best_worker_label_key(rank), "n/a")
        best_worker_id = shared_state.get(get_best_worker_id_key(rank), None)
        if np.isfinite(best_f) and best_x is not None:
            closures = precision_closure(rank, best_f)
            metrics = gate_metrics(np.asarray(best_x, dtype=np.float64), rank)
            print(
                f"  R={rank}: best={best_f:.8f} by {best_label} (worker {best_worker_id}) | f16={closures['float16']} bf16={closures['bfloat16']} int8={closures['int8']} min_b={min_bits(rank, best_f):.2f} | D_perp={metrics['D_perp']:.6f} leak={metrics['delta_leak']} gap={metrics['aug_gap']}",
                flush=True,
            )
            if best_f < global_best[1]:
                global_best = (rank, best_f, best_label, best_worker_id)
        else:
            print(f"  R={rank}: no result yet", flush=True)

    if global_best[0] is not None:
        print(
            f"Global best across active ranks: R={global_best[0]} frob={global_best[1]:.8f} by {global_best[2]} (worker {global_best[3]})",
            flush=True,
        )
    else:
        print("Global best across active ranks: none yet", flush=True)

    print("-" * 108, flush=True)
    print(" ID | rank | label                | trials | hits | last_frob", flush=True)
    print("-" * 108, flush=True)
    counts = shared_state["counts"]
    hits = shared_state["hits"]
    last_frob = shared_state["last_frob"]
    for config in assignments:
        worker_id = config["worker_id"]
        trials_text = str(int(counts.get(worker_id, 0)))
        last_val = float(last_frob.get(worker_id, np.inf))
        last_text = f"{last_val:.6f}" if np.isfinite(last_val) else "inf"
        hits_val = int(hits.get(worker_id, 0))
        print(
            f"{worker_id:>3} | {config['assigned_rank']:>4} | {config['label']:<20} | {trials_text:>6} | {hits_val:>4} | {last_text}",
            flush=True,
        )
    print("=" * 108, flush=True)


def status_thread_main(active_ranks, assignments, workers, shared_state, stop_event):
    while not stop_event.wait(STATUS_SECONDS):
        print_status(active_ranks, assignments, workers, shared_state)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=None, choices=ALL_RANKS)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join("results", "gated_search"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.workers > 24:
        raise ValueError("--workers must be between 1 and 24")

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    active_ranks = [args.rank] if args.rank is not None else list(ALL_RANKS)
    allocations, assignments = build_worker_assignments(active_ranks, args.workers)
    higher_rank_by_rank = {rank: maybe_load_higher_rank_best(rank, checkpoint_dir) for rank in active_ranks}
    wall_seconds = 0.0 if args.hours <= 0 else args.hours * 3600.0
    start_time = time.time()

    with Manager() as manager:
        shared_state = manager.dict()
        shared_state["counts"] = manager.dict({worker_id: 0 for worker_id in range(24)})
        shared_state["hits"] = manager.dict({worker_id: 0 for worker_id in range(24)})
        shared_state["last_frob"] = manager.dict({worker_id: float("inf") for worker_id in range(24)})
        state_lock = manager.RLock()
        loaded_rows = initialize_rank_state(shared_state, active_ranks, checkpoint_dir)

        print_startup_summary(
            active_ranks,
            args.workers,
            checkpoint_dir,
            assignments,
            loaded_rows,
            shared_state,
        )
        print(f"Worker allocation by rank: {allocations}", flush=True)
        print_status(active_ranks, assignments, args.workers, shared_state)

        stop_event = threading.Event()
        thread = threading.Thread(
            target=status_thread_main,
            args=(active_ranks, assignments, args.workers, shared_state, stop_event),
            daemon=True,
        )
        thread.start()

        worker_args = []
        for worker_id in range(args.workers):
            config = assignments[worker_id]
            worker_args.append(
                (
                    config["assigned_rank"],
                    config,
                    shared_state,
                    state_lock,
                    start_time,
                    wall_seconds,
                    args.no_gate,
                    checkpoint_dir,
                    higher_rank_by_rank[config["assigned_rank"]],
                )
            )

        pool = Pool(processes=args.workers)
        async_result = pool.map_async(worker_loop, worker_args)

        try:
            while True:
                if async_result.ready():
                    async_result.get()
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received; terminating pool.", flush=True)
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()
            stop_event.set()
            thread.join(timeout=1.0)
            print_status(active_ranks, assignments, args.workers, shared_state)


if __name__ == "__main__":
    freeze_support()
    main()
