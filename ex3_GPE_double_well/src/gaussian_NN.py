from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc
import json
import os

@dataclass
class GaussianModelConfig:
    neurons_per_layer: int = 8
    layers: int = 2
    positive: bool = True
    gaussian_init: bool = True


@dataclass
class GaussianTrainConfig:
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

@dataclass
class GaussianDomain:
    input_dim: int = 2
    r: float = 10.0
    N_points: int = 100_000
    sigma: float = 1.0


def build_model(cfg: GaussianModelConfig, input_dim: int, device, load_path=None) -> nn.Sequential:
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


def make_collocation_points(cfg: GaussianDomain, device):
    D = cfg.input_dim
    sample = qmc.LatinHypercube(d=D).random(n=cfg.N_points)
    sample = qmc.scale(sample, [-cfg.r] * D, [cfg.r] * D)
    X_f = torch.tensor(sample, dtype=torch.float32, device=device)
    return X_f


def make_save_path(model_cfg: GaussianModelConfig, domain: GaussianDomain) -> str:
    import os

    dir_ = os.path.join(
        "models", f"N{model_cfg.neurons_per_layer}_L{model_cfg.layers}"
    )
    fname = f"gaussian_{domain.input_dim:.0f}D_sigma{domain.sigma:.2f}_r{GaussianDomain.r:.1f}"
    os.makedirs(dir_, exist_ok=True)
    return os.path.join(dir_, fname)


class gaussian_initializer:
    def __init__(self, domain: GaussianDomain, model_cfg: GaussianModelConfig,
                 train_cfg: GaussianTrainConfig, load_model=None):
        self.domain = domain
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # State
        self.model = build_model(model_cfg, domain.input_dim, self.device, load_model)
        self.X_f_base = make_collocation_points(domain, self.device)
        self.X_f_base.requires_grad_(True)

        self.best_loss = float("inf")
        self.best_state_dict = None
        self.loss_history: list[float] = []
        self.save_base = make_save_path(model_cfg, domain)

    @property
    def _params(self):
        return self.model.parameters()

    def _forward(self):
        """ Get the grid of collocation points X_f and evaluate the model """
        X_f = self.X_f_base.detach().requires_grad_(True)
        h = self.model(X_f)
        return h, X_f

    def _compute_loss(self, h, X_f):
        d = self.domain.input_dim
        sigma2 = self.domain.sigma ** 2
        h_gaussian = torch.exp(- torch.sum(X_f**2, dim=1, keepdim=True) / (2 * sigma2)) * (2 * np.pi * sigma2) ** (-d / 2)
        L_f = torch.mean((h - h_gaussian) ** 2)
        return L_f

    def _update_best_loss(self):
        if self.loss_history[-1] < self.best_loss:
            self.best_loss = self.loss_history[-1]
            self.best_state_dict = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }

    # ---- Training loops ----

    def _train_adam(self, max_epochs, offset=0):
        cfg = self.train_cfg
        opt = torch.optim.Adam(self._params, lr=cfg.learning_rate)

        for epoch in range(offset, max_epochs + offset + 1):

            h, X_f = self._forward()
            loss = self._compute_loss(h, X_f)
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
                print(f"[Adam] {epoch:6d} | loss {loss.item():.6e}")

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
                loss = self._compute_loss(h, X_f)
                loss.backward()
                current_loss[0] = loss.item() # pyright: ignore
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
                print(f"[L-BFGS] {epoch:6d} | loss {current_loss[0]:.6e}")

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
                "epoch": len(self.loss_history),

                # GaussianModelConfig
                "neurons_per_layer": m.neurons_per_layer,
                "layers": m.layers,

                # GaussianTrainConfig
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
                "r": d.r,
                "N_points": d.N_points,
            }, f, indent=4)


def get_gaussian(input_dim, NpL, layers, r=10.0, N_points=100_000, sigma=1.0):
    d = GaussianDomain(
        input_dim=input_dim,
        r=r,
        N_points=N_points,
        sigma=sigma,
    )

    m = GaussianModelConfig(
        neurons_per_layer=NpL,
        layers=layers,
        positive=True,
    )

    t = GaussianTrainConfig(
        optimizer="lbfgs",
        min_loss=1e-10,
        save_every=100,
        print_every=1000,
    )

    gaussian_initializer(d, m, t).train()

    print(f"Computed {input_dim}D Gaussian with sigma={sigma} on a {NpL} neurons {layers} layers NN")
