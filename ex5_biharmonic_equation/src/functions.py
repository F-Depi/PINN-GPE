from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc
import json
import os


@dataclass
class DomainConfig:
    input_dim: int = 2
    x_l: list[float] = field(default_factory=list)
    x_r: list[float] = field(default_factory=list)
    N_points: int = 10_000
    V_ext: Callable = field(default=lambda x: 0.5 * x**2)
    Na: float = 1.0
    BC_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    BC_h: np.ndarray = field(default_factory=lambda: np.array([]))
    BC_w: float = 1.0
    norm_weight: float = 1.0


@dataclass
class ModelConfig:
    neurons_per_layer: int = 8
    layers: int = 2
    positive: bool = True


@dataclass
class TrainConfig:
    optimizer: str = "adam"
    max_epochs: int = 100_000
    min_loss: float = 1e-8
    print_every: int = 10
    save_every: int = 100
    comment: str = ""

    # Adam
    learning_rate: float = 1e-3
    adam_epochs: int = 50_000  # only for "adam+lbfgs"

    # L-BFGS
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_max_eval: int | None = None
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: str | None = "strong_wolfe"


def build_model(cfg: ModelConfig, input_dim: int, device, load_path=None) -> nn.Sequential:

    layers = [
        nn.Linear(input_dim, cfg.neurons_per_layer),
        nn.Tanh(),
    ]

    for _ in range(cfg.layers - 1):
        layers += [nn.Linear(cfg.neurons_per_layer, cfg.neurons_per_layer), nn.Tanh()]

    layers.append(nn.Linear(cfg.neurons_per_layer, 1))

    if cfg.positive:
        layers.append(nn.Softplus())

    model = nn.Sequential(*layers)
    

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    return model.to(device)


def make_collocation_points(cfg: DomainConfig, device):
    sample = qmc.LatinHypercube(d=cfg.input_dim).random(n=cfg.N_points)
    sample = qmc.scale(sample, cfg.x_l, cfg.x_r)
    X_f = torch.tensor(sample, dtype=torch.float32, device=device)
    V = torch.tensor(cfg.V_ext(sample), dtype=torch.float32, device=device)
    return X_f, V


def compute_laplacian(output, X):
    """Compute u_xx, u_yy, and Laplacian = u_xx + u_yy."""
    grad_u = torch.autograd.grad(
        output, X, grad_outputs=torch.ones_like(output), create_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True
    )[0][:, 1:2]

    laplacian = u_xx + u_yy
    return u_xx, u_yy, laplacian


def compute_bilaplacian(laplacian, X):
    """Compute Δ²u = Δ(Δu) by taking the Laplacian of the Laplacian."""
    grad_lap = torch.autograd.grad(
        laplacian, X, grad_outputs=torch.ones_like(laplacian), create_graph=True
    )[0]
    lap_x = grad_lap[:, 0:1]
    lap_y = grad_lap[:, 1:2]

    lap_xx = torch.autograd.grad(
        lap_x, X, grad_outputs=torch.ones_like(lap_x), create_graph=True
    )[0][:, 0:1]

    lap_yy = torch.autograd.grad(
        lap_y, X, grad_outputs=torch.ones_like(lap_y), create_graph=True
    )[0][:, 1:2]

    bilaplacian = lap_xx + lap_yy
    return bilaplacian


def f_exact(X):
    """f(x,y) = [160000(x²+y²)² - 64000(x²+y²) + 3200] exp(-10(x²+y²))"""
    x = X[:, 0:1]
    y = X[:, 1:2]
    r2 = x ** 2 + y ** 2
    return (160000 * r2 ** 2 - 64000 * r2 + 3200) * torch.exp(-10 * r2)


def h_exact(X):
    """h(x,y) = [400(x²+y²) - 40] exp(-10(x²+y²))"""
    x = X[:, 0:1]
    y = X[:, 1:2]
    r2 = x ** 2 + y ** 2
    return (400 * r2 - 40) * torch.exp(-10 * r2)


def compute_loss(model, mu, u_pred, X_f, potential, domain, epoch):
    """
    Solves the biharmonic BVP:
        Δ²u = f       in Ω = [-1,1]²
        u   = g       on ∂Ω
        Δu  = h       on ∂Ω
    """

    # --- PDE residual at interior collocation points ---
    _, _, laplacian = compute_laplacian(u_pred, X_f)
    bilaplacian = compute_bilaplacian(laplacian, X_f)

    L_pde = (bilaplacian - f_exact(X_f)).pow(2).mean()

    # --- Boundary conditions ---
    device = u_pred.device
    L_bc1 = torch.tensor(0.0, device=device)
    L_bc2 = torch.tensor(0.0, device=device)

    if domain.BC_grid.size > 0:
        # Boundary points (require grad so we can differentiate through them)
        BC_grid_t = (
            torch.tensor(domain.BC_grid, dtype=torch.float32, device=device)
            .requires_grad_(True)
        )
        BC_g_t = torch.tensor(domain.BC_h, dtype=torch.float32, device=device)

        # Predict u at boundary
        u_bc_pred = model(BC_grid_t)

        # BC1: u = g on ∂Ω
        L_bc1 = (u_bc_pred - BC_g_t).pow(2).mean()

        # BC2: Δu = h on ∂Ω  (Laplacian evaluated at boundary points)
        _, _, laplacian_bc = compute_laplacian(u_bc_pred, BC_grid_t)
        L_bc2 = (laplacian_bc - h_exact(BC_grid_t)).pow(2).mean()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:6d} | PDE Loss {L_pde.item():.6e} | BC Loss {L_bc1.item():.6e} + {L_bc2.item():.6e}")
    return L_pde + domain.BC_w * (L_bc1 + L_bc2)


def make_save_path(model_cfg: ModelConfig, domain: DomainConfig, comment: str) -> str:

    model_dir = f"{domain.input_dim:.0f}D_N{model_cfg.neurons_per_layer}_L{model_cfg.layers}"
    if model_cfg.positive:
        model_dir += "_pos"
    dir_ = os.path.join("models", model_dir)
    fname = (
        f"PINN_Na{domain.Na:.1f}"
        f"_x{'_'.join(f'{v:.2f}' for v in domain.x_l)}"
        f"-{'_'.join(f'{v:.2f}' for v in domain.x_r)}"
        f"{comment}"
    )
    os.makedirs(dir_, exist_ok=True)
    return os.path.join(dir_, fname)


def make_bc_grid_2d(x_l, x_r, n=100):
    xl, yl = x_l
    xr, yr = x_r
    t = np.linspace(0, 1, n)
    bottom = np.stack([xl + t * (xr - xl), np.full(n, yl)], axis=1)
    top = np.stack([xl + t * (xr - xl), np.full(n, yr)], axis=1)
    left = np.stack([np.full(n, xl), yl + t * (yr - yl)], axis=1)
    right = np.stack([np.full(n, xr), yl + t * (yr - yl)], axis=1)
    BC_grid = np.concatenate([bottom, top, left, right], axis=0)
    BC_h = np.zeros((len(BC_grid), 1))
    return BC_grid, BC_h


class GPESolver:
    def __init__(self, domain: DomainConfig, model_cfg: ModelConfig,
                 train_cfg: TrainConfig, load_model=None, mu=1.0):
        self.domain = domain
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State
        self.model = build_model(model_cfg, domain.input_dim, self.device, load_model)
        self.mu = nn.Parameter(torch.tensor(mu, device=self.device))
        self.X_f_base, self.potential = make_collocation_points(domain, self.device)
        self.X_f_base.requires_grad_(True)

        self.best_loss = float("inf")
        self.best_mu = None
        self.best_state_dict = None
        self.loss_history: list[float] = []
        self.save_base = make_save_path(model_cfg, domain, train_cfg.comment)

    @property
    def _params(self):
        return list(self.model.parameters()) + [self.mu]

    def _forward(self):
        """ Get the grid of collocation points X_f and evaluate the model """
        X_f = self.X_f_base.detach().requires_grad_(True)
        h = self.model(X_f)
        return h, X_f

    def _loss(self, h, X_f, epoch):
        return compute_loss(self.model, self.mu, h, X_f, self.potential,
                            self.domain, epoch)

    def _update_best_loss(self):
        if self.loss_history[-1] < self.best_loss:
            self.best_loss = self.loss_history[-1]
            self.best_mu = self.mu.item()
            self.best_state_dict = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }

    # ---- Training loops ----

    def _train_adam(self, max_epochs, offset=0):
        cfg = self.train_cfg
        opt = torch.optim.Adam(self._params, lr=cfg.learning_rate)

        for epoch in range(offset, max_epochs + offset + 1):

            h, X_f = self._forward()
            loss = self._loss(h, X_f, epoch)
            self.loss_history.append(loss.item())
            self._update_best_loss()

            if loss.item() < cfg.min_loss:
                print(f"[Adam] Converged at epoch {epoch}")
                return epoch

            if epoch % cfg.save_every == 0:
                self.save()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % cfg.print_every == 0:
                print(f"[Adam] {epoch:6d} | loss {loss.item():.6e} | μ {self.mu.item():.4f}")

        return offset + max_epochs

    def _train_lbfgs(self, max_epochs, offset=0):
        cfg = self.train_cfg
        opt = torch.optim.LBFGS(
            self._params,
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            max_eval=cfg.lbfgs_max_eval,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=cfg.lbfgs_line_search_fn,
        )

        for epoch in range(offset, max_epochs + offset + 1):
            current_loss = [None]

            def closure():
                opt.zero_grad()
                h, X_f = self._forward()
                loss = self._loss(h, X_f, epoch)
                loss.backward()
                current_loss[0] = loss.item()
                return loss

            opt.step(closure)
            self.loss_history.append(current_loss[0]) # pyright: ignore
            self._update_best_loss()

            if current_loss[0] < cfg.min_loss: # pyright: ignore
                print(f"[L-BFGS] Converged at epoch {epoch}")
                return epoch

            if epoch % cfg.save_every == 0:
                self.save()
            if epoch % cfg.print_every == 0:
                print(f"[L-BFGS] {epoch:6d} | loss {current_loss[0]:.6e} | μ {self.mu.item():.4f}")

            # Early stop, usually in a local minimum
            if epoch >= cfg.save_every + 1:
                recent = self.loss_history[-cfg.save_every - 1 :]
                m = np.mean(recent)
                if m != 0 and np.std(recent) / abs(m) < 1e-6:
                    print(f"[L-BFGS] Plateau at epoch {epoch}")
                    return epoch

        return offset + max_epochs

    def train(self):
        cfg = self.train_cfg
        if cfg.optimizer == "adam":
            self._train_adam(cfg.max_epochs)
        elif cfg.optimizer == "lbfgs":
            self._train_lbfgs(cfg.max_epochs)
        elif cfg.optimizer == "adam+lbfgs":
            last = self._train_adam(min(cfg.adam_epochs, cfg.max_epochs))
            remaining = cfg.max_epochs - last
            if remaining > 0 and self.best_loss >= cfg.min_loss:
                self._train_lbfgs(remaining, offset=last)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")
        print(f"{self.save_base}.pth")
        self.save()

    def save(self):
        base = self.save_base
        d = self.domain
        m = self.model_cfg
        t = self.train_cfg

        torch.save(self.best_state_dict, f"{base}.pth")

        with open(f"{base}_history.csv", "w") as f:
            f.write("epoch,loss\n")
            for i, v in enumerate(self.loss_history):
                f.write(f"{i},{v}\n")

        with open(f"{base}_param.json", "w") as f:
            json.dump({
                # Results
                "loss": self.best_loss,
                "mu": self.best_mu,
                "epoch": len(self.loss_history),

                # ModelConfig
                "neurons_per_layer": m.neurons_per_layer,
                "layers": m.layers,

                # TrainConfig
                "optimizer": t.optimizer,
                "learning_rate": t.learning_rate,
                "lbfgs_lr": t.lbfgs_lr,
                "lbfgs_max_iter": t.lbfgs_max_iter,
                "lbfgs_history_size": t.lbfgs_history_size,
                "adam_epochs": t.adam_epochs,
                "max_epochs": t.max_epochs,
                "min_loss": t.min_loss,
                "comment": t.comment,
                "save_every": t.save_every,
                "print_every": t.print_every,

                # DomainConfig
                "Na": d.Na,
                "x_l": d.x_l,
                "x_r": d.x_r,
                "N_points": d.N_points,
                "BC_grid": d.BC_grid.tolist(),
                "BC_h": d.BC_h.tolist(),
                "BC_w": d.BC_w,
                "norm_weight": d.norm_weight,

                # Collocation data
                "x": self.X_f_base.cpu().detach().numpy().tolist(),
                "V_ext": self.potential.cpu().detach().numpy().tolist(),
            }, f, indent=4)

