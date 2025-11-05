
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import math

# Setup
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

# LR warmup
def warmup_lr(step):
    return min((step + 1) / 5000, 1.0)

scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr)

# Optional: stability settings
MAX_LOSS = 10.0     # clip loss magnitude (adjustable)
MAX_GRAD_NORM = 1.0 # clip gradient magnitude

for step, x0 in enumerate(tqdm(dataloader)):
    x0 = x0.to(device)
    t = torch.randint(0, diffusion.num_timesteps, (x0.size(0),), device=device)
    noise = torch.randn_like(x0)
    xt = diffusion.q_sample(x0, t, noise)

    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, noise)

    # --- Exploding loss protection ---
    if not torch.isfinite(loss):
        print(f"[Warning] step {step}: NaN/Inf loss detected. Skipping step.")
        optimizer.zero_grad(set_to_none=True)
        continue

    if loss.item() > MAX_LOSS:
        print(f"[Warning] step {step}: Clipping loss {loss.item():.2f} → {MAX_LOSS}")
        loss = torch.tensor(MAX_LOSS, device=loss.device, dtype=loss.dtype)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # --- NaN gradient protection ---
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
    if not math.isfinite(grad_norm):
        print(f"[Warning] step {step}: NaN/Inf gradient detected. Skipping step.")
        optimizer.zero_grad(set_to_none=True)
        continue

    optimizer.step()
    scheduler.step()
    ema.update()

    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item():.4f}")
        # save checkpoint, log tensorboard, etc.
