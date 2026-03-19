import os
import json
import numpy as np
from scipy.stats import qmc
from dataclasses import dataclass, field
import torch
import torch.nn as nn


@dataclass
class PINNConfig:
    neurons_per_layer: int = 8
    layers: int = 2

    param: list[float] = field(default_factory=list)

    t_l: float = 0.0
    t_r: float = 0.0
    x_l: float = 0.0
    x_r: float = 0.0
    N_points: int = 20000
    N_periodic_BC: int = 50

    optimizer: str = "adam"

    # Adam hyperparameters
    learning_rate: float = 1e-3

    # L-BFGS hyperparameters
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20        # max iterations per optimizer.step()
    lbfgs_max_eval: int | None = None  # max function evaluations per step (None = 1.25 * max_iter)
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_history_size: int = 100
    lbfgs_line_search_fn: str | None = "strong_wolfe"  # None or "strong_wolfe"

    # For "adam+lbfgs": how many epochs to run Adam before switching
    adam_epochs: int = 50_000

    max_epochs: int = 100_000
    min_loss: float = 1e-8

    BC_grid: np.ndarray = field(default_factory=lambda: np.array([]))  # array of shape (t, x, N_BC)
    BC_h: np.ndarray = field(default_factory=lambda: np.array([]))  # array of shape (Re(u), Im(u), N_BC)
    BC_w: float = 1.0  # weight for the BC loss term

    comment: str = ""
    save_every: int = 1000
    print_every: int = 100


class GPESolver:
    def __init__(self, config: PINNConfig, load_model=None):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = self._build_model(load_model)

        # Grid for simple BC points
        self.BC_grid = torch.tensor(config.BC_grid, dtype=torch.float32, device=self.device)
        self.BC_h = torch.tensor(config.BC_h, dtype=torch.float32, device=self.device)

        # Grid for periodic BC
        t = qmc.LatinHypercube(d=1).random(n=self.config.N_periodic_BC)
        t = qmc.scale(t, self.config.t_l, self.config.t_r)
        side_grid = torch.tensor(
            [[t_i, x_i] for t_i in t.flatten() for x_i in [-5, 5]],
            dtype=torch.float32, device=self.device,
        ).requires_grad_(True)
        self.side_grid = side_grid

        self.best_loss = float("inf")
        self.best_state_dict = None
        self.loss_history: list[float] = []

    def _make_adam(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def _make_lbfgs(self) -> torch.optim.LBFGS:
        cfg = self.config
        return torch.optim.LBFGS(
            self.model.parameters(),
            lr=cfg.lbfgs_lr,
            max_iter=cfg.lbfgs_max_iter,
            max_eval=cfg.lbfgs_max_eval,
            tolerance_grad=cfg.lbfgs_tolerance_grad,
            tolerance_change=cfg.lbfgs_tolerance_change,
            history_size=cfg.lbfgs_history_size,
            line_search_fn=cfg.lbfgs_line_search_fn,
        )

    def _build_model(self, load_model) -> nn.Sequential:
        cfg = self.config
        hidden = []
        for _ in range(cfg.layers - 1):
            hidden += [nn.Linear(cfg.neurons_per_layer, cfg.neurons_per_layer), nn.Tanh()]

        model = nn.Sequential(
            nn.Linear(2, cfg.neurons_per_layer),
            nn.Tanh(),
            *hidden,
            nn.Linear(cfg.neurons_per_layer, 2),
        )

        if load_model is not None:
            model.load_state_dict(torch.load(load_model))
            print(f"Loaded model from {load_model}")
        return model.to(self.device)

    def _compute_derivatives(self, u: torch.Tensor, v: torch.Tensor,
                             X_f: torch.Tensor):
        ones = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, X_f, grad_outputs=ones, create_graph=True)[0]
        grad_v = torch.autograd.grad(v, X_f, grad_outputs=ones, create_graph=True)[0]

        u_t = grad_u[:, 0:1]
        u_x = grad_u[:, 1:2]
        v_t = grad_v[:, 0:1]
        v_x = grad_v[:, 1:2]

        u_xx = torch.autograd.grad(u_x, X_f, grad_outputs=ones, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x, X_f, grad_outputs=ones, create_graph=True)[0][:, 1:2]

        return u_t, v_t, u_xx, v_xx

    def _boundaries(self):
        side_grid = self.side_grid.clone().detach().requires_grad_(True)
        h_side = self.model(side_grid)

        u = h_side[:, 0:1] # Re(u)
        v = h_side[:, 1:2] # Im(u)
        ones = torch.ones_like(u)
        u_x = torch.autograd.grad(u, side_grid, grad_outputs=ones, create_graph=True)[0][:, 1:2]
        v_x = torch.autograd.grad(v, side_grid, grad_outputs=ones, create_graph=True)[0][:, 1:2]

        u = u.reshape(-1, 2) # so that u[:, 0] corresponds to x=-5 and u[:, 1] corresponds to x=5
        v = v.reshape(-1, 2)
        u_x = u_x.reshape(-1, 2)
        v_x = v_x.reshape(-1, 2)
        return u, v, u_x, v_x

    def _compute_loss(self, h: torch.Tensor, X_f: torch.Tensor) -> torch.Tensor:
        # PDE: i u_t + 0.5 u_xx + |u|^2 u = 0, u = u[0] + i u[1]
        u = h[:, 0:1] # Re(u)
        v = h[:, 1:2] # Im(u)
        u_t, v_t, u_xx, v_xx = self._compute_derivatives(u, v, X_f)

        h2 = u.pow(2) + v.pow(2)
        Re_res = (-v_t + 0.5 * u_xx + u * h2).pow(2)
        Im_res = (u_t + 0.5 * v_xx + v * h2).pow(2)
        L_pde = (Re_res + Im_res).mean()

        # Boundary conditions: h(BC_grid) = BC_h
        L_bc = (self.model(self.BC_grid) - self.BC_h).pow(2).mean()

        # Symmetry: h(t, -x) = h(t, x), h_x(t, -x) = h_x(t, x)
        u_b, v_b, u_x_b, v_x_b = self._boundaries()
        L_sym1 = (u_b[:, 0] - u_b[:, 1]).pow(2).mean() + (v_b[:, 0] - v_b[:, 1]).pow(2).mean()
        L_sym2 = (u_x_b[:, 0] + u_x_b[:, 1]).pow(2).mean() + (v_x_b[:, 0] + v_x_b[:, 1]).pow(2).mean()

        return L_pde + self.config.BC_w * (L_bc + L_sym1 + L_sym2)

    def _forward(self, X_f_base: torch.Tensor):
        """
        Creates the collocation tensor & evaluates model)
        Return (h, X_f) where X_f is a fresh leaf tensor with grad enabled.
        """
        X_f = X_f_base.clone().detach().requires_grad_(True)
        h = self.model(X_f)
        return h, X_f

    # ------------------------------------------------------------------ #
    #  Training loops
    # ------------------------------------------------------------------ #

    def _train_adam(self, X_f_base: torch.Tensor, max_epochs: int, epoch_offset: int = 0):
        """Standard Adam training loop. Returns the last epoch index (global)."""
        optimizer = self._make_adam()
        cfg = self.config

        for epoch in range(1, max_epochs + 1):
            global_epoch = epoch_offset + epoch

            h, X_f = self._forward(X_f_base)
            loss = self._compute_loss(h, X_f)

            self.loss_history.append(loss.item())
            self._update_best(loss)

            if loss.item() < cfg.min_loss:
                print(f"[Adam] Converged at epoch {global_epoch} with loss {loss.item():.10f}")
                return global_epoch

            if global_epoch % cfg.save_every == 0:
                self.save(global_epoch, final=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_epoch % cfg.print_every == 0:
                print(f"[Adam]   Epoch {global_epoch:5d} | Loss: {loss.item():.10f}")

        return epoch_offset + max_epochs

    def _train_lbfgs(self, X_f_base: torch.Tensor, max_epochs: int, epoch_offset: int = 0):
        """L-BFGS training loop.  Each `optimizer.step(closure)` counts as one epoch."""
        optimizer = self._make_lbfgs()
        cfg = self.config

        for epoch in range(1, max_epochs + 1):
            global_epoch = epoch_offset + epoch

            # L-BFGS requires a closure that re-evaluates the model + loss
            current_loss = [None]

            def closure():
                optimizer.zero_grad()
                h, X_f = self._forward(X_f_base)
                loss = self._compute_loss(h, X_f)
                loss.backward()
                current_loss[0] = loss.item()
                return loss

            optimizer.step(closure)

            self.loss_history.append(current_loss[0])
            # Need to recompute to get the actual latest loss after the step
            with torch.no_grad():
                h_eval, X_f_eval = self._forward(X_f_base)
            # We already recorded the loss from the closure's last evaluation
            self._update_best_from_value(current_loss[0])

            if current_loss[0] < cfg.min_loss:
                print(f"[L-BFGS] Converged at epoch {global_epoch} with loss {current_loss[0]:.10f}")
                return global_epoch

            if global_epoch % cfg.save_every == 0:
                self.save(global_epoch, final=False)

            if global_epoch % cfg.print_every == 0:
                print(f"[L-BFGS] Epoch {global_epoch:5d} | Loss: {current_loss[0]:.10f}")

            # Early stop, usually in a local minimum
            if epoch >= cfg.save_every + 1:
                recent = self.loss_history[-cfg.save_every - 1 :]
                m = np.mean(recent)
                if m != 0 and np.std(recent) / abs(m) < 1e-6:
                    print(f"[L-BFGS] Plateau at epoch {epoch}")
                    return epoch

        return epoch_offset + max_epochs

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    def train(self):
        cfg = self.config

        # Create collocation points
        sample = qmc.LatinHypercube(d=2).random(n=cfg.N_points)
        sample = qmc.scale(sample, [cfg.t_l, cfg.x_l], [cfg.t_r, cfg.x_r])
        X_f_base = torch.tensor(sample, dtype=torch.float32,
                                device=self.device).requires_grad_(True)

        if cfg.optimizer == "adam":
            last_epoch = self._train_adam(X_f_base, cfg.max_epochs)

        elif cfg.optimizer == "lbfgs":
            last_epoch = self._train_lbfgs(X_f_base, cfg.max_epochs)

        elif cfg.optimizer == "adam+lbfgs":
            adam_epochs = min(cfg.adam_epochs, cfg.max_epochs)
            lbfgs_epochs = cfg.max_epochs - adam_epochs

            print(f"=== Phase 1: Adam for {adam_epochs} epochs ===")
            last_epoch = self._train_adam(X_f_base, adam_epochs)

            if lbfgs_epochs > 0 and self.best_loss >= cfg.min_loss:
                print(f"=== Phase 2: L-BFGS for up to {lbfgs_epochs} epochs ===")
                last_epoch = self._train_lbfgs(X_f_base, lbfgs_epochs, epoch_offset=last_epoch)

        else:
            raise ValueError(
                f"Unknown optimizer '{cfg.optimizer}'. Choose 'adam', 'lbfgs', or 'adam+lbfgs'."
            )

        self.save(last_epoch, final=True)

    def _update_best(self, loss: torch.Tensor):
        self._update_best_from_value(loss.item())

    def _update_best_from_value(self, loss_val: float):
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

    # ------------------------------------------------------------------ #
    #  Saving
    # ------------------------------------------------------------------ #

    @property
    def _dir(self) -> str:
        cfg = self.config
        return os.path.join("models", f"N{cfg.neurons_per_layer}_L{cfg.layers}")

    def _fname(self, epoch, final: bool) -> str:
        cfg = self.config
        if final:
            return f"PINN_E{epoch}_t{cfg.t_l:.2f}-{cfg.t_r:.2f}_x{cfg.x_l:.2f}-{cfg.x_r:.2f}{cfg.comment}"
        else:
            return f"PINN_checkpoint_t{cfg.t_l:.2f}-{cfg.t_r:.2f}_x{cfg.x_l:.2f}-{cfg.x_r:.2f}{cfg.comment}"

    def save(self, epoch, final):
        cfg = self.config
        os.makedirs(self._dir, exist_ok=True)
        base = os.path.join(self._dir, self._fname(epoch, final))

        torch.save(self.best_state_dict, f"{base}.pth")

        with open(f"{base}_history.csv", "w") as f:
            f.write("epoch,loss\n")
            for i, loss_val in enumerate(self.loss_history):
                f.write(f"{i},{loss_val}\n")

        with open(f"{base}_param.json", "w") as f:
            json.dump({
                "loss": self.best_loss,
                "epoch": len(self.loss_history),
                "optimizer": cfg.optimizer,
                "neurons_per_layer": cfg.neurons_per_layer,
                "layers": cfg.layers,
                "learning_rate": cfg.learning_rate,
                "lbfgs_lr": cfg.lbfgs_lr,
                "lbfgs_max_iter": cfg.lbfgs_max_iter,
                "lbfgs_history_size": cfg.lbfgs_history_size,
                "adam_epochs": cfg.adam_epochs,
                "max_epochs": cfg.max_epochs,
                "min_loss": cfg.min_loss,
                "t_l": cfg.t_l,
                "t_r": cfg.t_r,
                "x_l": cfg.x_l,
                "x_r": cfg.x_r,
                "N_points": cfg.N_points,
                "comment": cfg.comment,
                "save_every": cfg.save_every,
                "print_every": cfg.print_every,
                "param": cfg.param,
                "BC_grid": cfg.BC_grid.tolist(),
                "BC_h": cfg.BC_h.tolist(),
                "BC_w": cfg.BC_w,
            }, f, indent=4)


if __name__ == "__main__":
    # Boundary conditions u(t=0, x) = sech(x) for x in [-5, 5]
    x_l, x_r = -5.0, 5.0
    x = qmc.LatinHypercube(d=1).random(n=50)
    x = qmc.scale(x, x_l, x_r)
    u = 2 / np.cosh(x)
    BD_grid = np.hstack((np.zeros_like(x), x))
    BD_h = np.hstack((u, np.zeros_like(u)))

    config = PINNConfig(
        neurons_per_layer=32,
        layers=3,
        t_l=0.0,
        t_r=np.pi / 2,
        x_l=x_l,
        x_r=x_r,
        N_points=20_000,
        N_periodic_BC=50,

        # --- Optimizer choice ---
        optimizer="lbfgs",       # "adam", "lbfgs", or "adam+lbfgs"

        # L-BFGS settings
        lbfgs_lr=1.0,
        lbfgs_max_iter=20,
        lbfgs_history_size=100,
        lbfgs_line_search_fn="strong_wolfe",

        max_epochs=100_000,
        min_loss=1e-8,
        BC_grid=BD_grid,
        BC_h=BD_h,
        comment="_ibiliv",

        print_every=10,
        save_every=100,
    )

    solver = GPESolver(config)
    total_params = sum(p.numel() for p in solver.model.parameters())
    trainable_params = sum(p.numel() for p in solver.model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    solver.train()
