from __future__ import annotations

import numpy as np

from ml_seed.stage2.model import ConvRanker, require_torch
from ml_seed.stage4.invert import image_to_C

torch, _nn = require_torch()


def _model_device(model: ConvRanker) -> "torch.device":
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_seed(
    model: ConvRanker,
    n_steps: int = 400,
    lr: float = 0.01,
    noise_scale: float = 0.1,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
    ucb_beta: float = 2.0,
    ucb_samples: int = 8,
    grad_noise_scale: float = 0.005,
) -> dict:
    """Optimize a random image toward lower UCB acquisition score.

    Instead of minimising the plain model prediction, each gradient step
    minimises  ``mean(score) - ucb_beta * std(score)``  estimated over
    ``ucb_samples`` MC-Dropout forward passes.  The uncertainty bonus
    (``-beta * std``) biases the search toward regions the model is *less
    sure about* — exactly the regions that could be better than anything
    in the training set.  Annealed Langevin gradient noise helps escape
    local optima in the early stages.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    ucb_samples = max(1, int(ucb_samples))

    device = _model_device(model)
    img_np = rng.uniform(-noise_scale, noise_scale, (27, 27)).astype(np.float32)
    img = torch.tensor(img_np, device=device, requires_grad=True)
    opt = torch.optim.Adam([img], lr=lr)

    trajectory: list[float] = []
    best_score = float("inf")
    best_img = img_np.copy()
    best_step = 0

    # Keep dropout active throughout so MC samples give real uncertainty.
    model.train()
    with torch.no_grad():
        score_init = float(model(img.unsqueeze(0).unsqueeze(0)).squeeze().item())

    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        x = img.unsqueeze(0).unsqueeze(0)  # [1, 1, 27, 27]

        # MC-Dropout forward passes batched into a single GPU call.
        x_batch = x.expand(ucb_samples, -1, -1, -1)  # [ucb_samples, 1, 27, 27]
        mc_scores = model(x_batch).squeeze(-1)  # [ucb_samples]
        mean_score = mc_scores.mean()
        std_score = mc_scores.std(unbiased=True) if ucb_samples > 1 else mean_score.detach() * 0.0

        # UCB acquisition: low mean (good basin) + high uncertainty (unexplored).
        acquisition = mean_score - ucb_beta * std_score
        acquisition.backward()

        # Annealed Langevin noise: explore early, exploit late.
        if grad_noise_scale > 0.0 and img.grad is not None:
            noise_std = grad_noise_scale / (1.0 + step * 0.01)
            img.grad.add_(torch.randn_like(img.grad) * noise_std)

        opt.step()

        with torch.no_grad():
            img.clamp_(-1.0, 1.0)
            # Score after the update (single pass for efficiency).
            score_now = float(model(img.unsqueeze(0).unsqueeze(0)).squeeze().item())
            trajectory.append(score_now)
            if score_now < best_score:
                best_score = score_now
                best_img = img.detach().cpu().numpy().copy()
                best_step = step + 1

    C = image_to_C(best_img, alpha=alpha)
    return {
        "image": best_img.astype(np.float32, copy=False),
        "C": C,
        "score_init": score_init,
        "score_final": best_score,
        "score_traj": trajectory,
        "best_step": best_step,
    }


def generate_batch(
    model: ConvRanker,
    n_seeds: int = 10,
    n_steps: int = 400,
    lr: float = 0.01,
    noise_scale: float = 0.1,
    alpha_values: list[float] | tuple[float, ...] = (0.5, 1.0, 2.0),
    ucb_beta: float = 2.0,
    ucb_samples: int = 8,
    grad_noise_scale: float = 0.005,
) -> list[dict]:
    """Generate a batch of candidate tensors across multiple amplitude scales."""
    if n_seeds < 1:
        return []
    alpha_list = [float(alpha) for alpha in alpha_values]
    if not alpha_list:
        raise ValueError("alpha_values must contain at least one value")

    rng = np.random.default_rng()
    results: list[dict] = []
    for index in range(n_seeds):
        alpha = alpha_list[index % len(alpha_list)]
        result = generate_seed(
            model,
            n_steps=n_steps,
            lr=lr,
            noise_scale=noise_scale,
            alpha=alpha,
            rng=rng,
            ucb_beta=ucb_beta,
            ucb_samples=ucb_samples,
            grad_noise_scale=grad_noise_scale,
        )
        result["alpha"] = alpha
        result["seed_idx"] = index
        results.append(result)
    results.sort(key=lambda record: record["score_final"])
    return results