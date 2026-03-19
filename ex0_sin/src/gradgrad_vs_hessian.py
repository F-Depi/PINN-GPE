import torch
import torch.nn as nn
import time

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, xy):
        return self.net(xy)

model = PINN()

# ---- Method 1: Manual autograd.grad (your current approach) ----
def laplacian_manual(model, xy):
    xy.requires_grad_(True)
    u = model(xy)
    grads = torch.autograd.grad(u, xy, torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(grads[:, 0:1], xy, torch.ones_like(grads[:, 0:1]),
                                   create_graph=True, retain_graph=True)[0][:, 0:1]
    d2u_dy2 = torch.autograd.grad(grads[:, 1:2], xy, torch.ones_like(grads[:, 1:2]),
                                   create_graph=True, retain_graph=True)[0][:, 1:2]
    return d2u_dx2 + d2u_dy2

# ---- Method 2: torch.autograd.functional.hessian (per-sample loop) ----
def laplacian_hessian(model, xy):
    laps = []
    for i in range(xy.shape[0]):
        def f(x):
            return model(x.unsqueeze(0)).squeeze()
        H = torch.autograd.functional.hessian(f, xy[i], create_graph=True)
        laps.append(H[0, 0] + H[1, 1])  # trace
    return torch.stack(laps).unsqueeze(1)

# ---- Method 3: torch.func (functorch) ----
from torch.func import vmap, jacrev

def laplacian_functorch(model, xy):
    def f(x):
        return model(x.unsqueeze(0)).squeeze()
    hessian_fn = jacrev(jacrev(f))
    # vmap over batch
    H = vmap(hessian_fn)(xy)  # (N, 2, 2)
    return (H[:, 0, 0] + H[:, 1, 1]).unsqueeze(1)  # trace

# ---- Timing ----
xy = torch.rand(1000, 2)

for name, fn in [("Manual autograd", laplacian_manual),
                  ("Hessian (loop)",  laplacian_hessian),
                  ("Functorch",       laplacian_functorch)]:
    # Warmup
    for _ in range(3):
        _ = fn(model, xy.clone())

    start = time.perf_counter()
    for _ in range(10):
        _ = fn(model, xy.clone())
    elapsed = (time.perf_counter() - start) / 10
    print(f"{name:20s}: {elapsed*1000:.2f} ms")
