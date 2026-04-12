"""One-shot resume script for N=20 FULL model with small LR."""
import math, os, time, torch
from data import sample_batch, frobenius_relative_error
from model import KethVaraiMachine

N, steps, batch_size = 20, 50_000, 16_384
lr, eta_min = 1e-4, 1e-6
device = torch.device("cuda")
ckpt = "results/benchmark_ckpts/full_N20_s0.pt"
full_ckpt = "results/benchmark_ckpts/full_N20_s0_full.pt"

model = KethVaraiMachine(N=N, latent_dim=64, encoder_depth=3, encoder_width=128).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

# Verify loaded checkpoint
model.eval()
with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
    Av, Bv, Cv = sample_batch(16_384, device)
    err0 = frobenius_relative_error(model(Av, Bv).float(), Cv.float()).item()
bits0 = -math.log2(err0)
print(f"Loaded checkpoint: rel={err0:.3e}  bits={bits0:.1f}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=eta_min)
scaler = torch.amp.GradScaler(enabled=True)

best = err0
t0 = time.time()

for step in range(1, steps + 1):
    model.train()
    A, B, C = sample_batch(batch_size, device)
    with torch.amp.autocast(device_type="cuda"):
        C_hat = model(A, B)
        loss = frobenius_relative_error(C_hat.float(), C.float())
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if step % 500 == 0 or step == 1:
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            Av, Bv, Cv = sample_batch(16_384, device)
            err = frobenius_relative_error(model(Av, Bv).float(), Cv.float()).item()
        improved = err < best
        if improved:
            best = err
            torch.save(model.state_dict(), ckpt)
        # Save full state for future resume
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best": best,
        }, full_ckpt)
        bits = -math.log2(best) if best > 0 else float("inf")
        elapsed = time.time() - t0
        marker = " *" if improved else ""
        print(f"  RESUME N=20 step {step:>7d}/{steps}  rel={err:.3e}  bits={bits:.1f}  lr={scheduler.get_last_lr()[0]:.1e}  t={elapsed:.0f}s{marker}")

        if best < 1e-6:
            break

print(f"\nFinal best: rel={best:.3e}  bits={-math.log2(best):.2f}")
print(f"Saved model -> {ckpt}")
print(f"Saved full state -> {full_ckpt}")
