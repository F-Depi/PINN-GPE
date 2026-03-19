from functions import *


def multiple_runs(domain, model_config, train_config, load_model=None, N_runs=5):
    comment = train_config.comment
    runs_best_loss = float("inf")
    losses = []
    for i in range(N_runs):
        print(f"\n=== Starting run {i+1}/{N_runs} ===")
        train_config.comment = comment + f"-run{i+1}"
        solver = GPESolver(domain, model_config, train_config, load_model)
        solver.train()
        best_loss = solver.best_loss
        losses.append(best_loss)
        save_base = solver.save_base
        if best_loss < runs_best_loss:
            best_base = save_base.replace(f"-run{i+1}", "-best")
            os.rename(f"{save_base}.pth", f"{best_base}.pth")
            os.rename(f"{save_base}_history.csv", f"{best_base}_history.csv")
            os.rename(f"{save_base}_param.json", f"{best_base}_param.json")
            runs_best_loss = best_loss
        else:
            os.remove(f"{save_base}.pth")
            os.remove(f"{save_base}_history.csv")
            os.remove(f"{save_base}_param.json")

    train_config.comment = comment  # reset comment to original for next set of runs
    final_base = best_base.replace("-best", "")
    os.rename(f"{best_base}.pth", f"{final_base}.pth")
    os.rename(f"{best_base}_history.csv", f"{final_base}_history.csv")
    os.rename(f"{best_base}_param.json", f"{final_base}_param.json")
    print(f"\n=== All runs completed ===")
    print(f"Best loss across runs: {runs_best_loss:.3e}")
    print(f"Losses for all runs: {', '.join(f'{l:.3e}' for l in losses)}")


""" First Double Well """
#r = 3.0
#domain = DomainConfig(
#        input_dim=1,
#        x_l=[-r],
#        x_r=[r],
#        N_points=1_000,
#        V_ext = lambda x: (x**2 - 1.0)**2,
#        Na = 1,
#        BC_grid=np.array([[r], [r]]),  # shape: (2, 1)
#        BC_h=np.array([[0.0], [0.0]]),  # shape: (2, 1)
#        norm_weight=1.0,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=8,
#        layers=2,
#        positive=True,
#        gaussian_init=True,
#        gaussian_sigma=1,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment="_dw1"
#    )
#
#multiple_runs(domain, model_config, train_config, N_runs=5)


""" Symm doule well when things started goint south"""
#r = 3.0
#domain = DomainConfig(
#        V_ext = lambda x: (x**2 - 2.0)**2,
#        Na = 100,
#
#        input_dim=1,
#        x_l=[-r],
#        x_r=[r],
#        N_points=2_000,
#        BC_grid=np.array([[r], [r]]),  # shape: (2, 1)
#        BC_h=np.array([[0.0], [0.0]]),  # shape: (2, 1)
#        norm_weight=1.0,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=16,
#        layers=2,
#        positive=True,
#        gaussian_init=True,
#        gaussian_sigma=1,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment="_dw2_gauss"
#    )
#
#multiple_runs(domain, model_config, train_config, N_runs=5)
#
## Then retrain with starting from the high-g model, and lowering the Na
#path_high_g = "models/1D_N16_L2_pos/PINN_Na100.0_x-3.00-3.00_dw2_gauss.pth"
#model_config.gaussian_init = False
#domain.Na = 1.0
#train_config.comment = "_dw2_high_g"
#
#multiple_runs(domain, model_config, train_config, path_high_g, N_runs=5)


""" Asymm doule well with the same approach """
#r = 3.0
#domain = DomainConfig(
#        V_ext = lambda x: (x**2 - 1.0)**2 + x / 6.0,
#        Na = 100,
#
#        input_dim=1,
#        x_l=[-r],
#        x_r=[r],
#        N_points=2_000,
#        BC_grid=np.array([[r], [r]]),  # shape: (2, 1)
#        BC_h=np.array([[0.0], [0.0]]),  # shape: (2, 1)
#        norm_weight=1.0,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=16,
#        layers=2,
#        positive=True,
#        gaussian_init=True,
#        gaussian_sigma=1,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment="_dw2_as6_gauss"
#    )
#
#multiple_runs(domain, model_config, train_config, N_runs=5)
#
## Then retrain with starting from the high-g model, and lowering the Na
#path_high_g = "models/1D_N16_L2_pos/PINN_Na100.0_x-3.00-3.00_dw2_as6_gauss.pth"
#model_config.gaussian_init = False
#domain.Na = 1.0
#train_config.comment = "_dw2_as6_high_g"
#
#multiple_runs(domain, model_config, train_config, path_high_g, N_runs=5)


"""
                        Harmonic in 2D
    (x[:,0:1]**2 + x[:,1:2]**2) / 2.0   harmonic
"""
#r = 6.0
#N_b = 100
#
#x_l = [-r, -r];     x_r = [r, r]
## Boundary conditions u(x_l) = u(x_r) = 0
#BD_grid = np.array([x_l, x_r])  # shape: (2, 1)
#BD_h = np.array([[0.0], [0.0]])  # shape: (2, 1)
#BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)
#
#domain = DomainConfig(
#        V_ext = lambda x: (x[:,0:1]**2 + x[:,1:2]**2) / 2.0,
#        Na = 0,
#
#        input_dim=2,
#        x_l=x_l,
#        x_r=x_r,
#        N_points=20_000,
#        BC_grid=BD_grid,
#        BC_h=BD_h,
#        BC_w = 1.0,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=32,
#        layers=3,
#        positive=True,
#        gaussian_init=True,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment="_harmonic"
#    )
#
#multiple_runs(domain, model_config, train_config, N_runs=5)


"""
                    Asymmetrical double well in 2D 
x[:,0:1]**4 + (x[:,1:2]**2 - 2.0)**2 + x[:,0:1] * x[:,1:2] + x[:,0:1]/4 dw2_as4
"""
#r = 3.0
#N_b = 100
#a = 2.0
#b = 6.0
#x_l = [-r, -r];     x_r = [r, r]
#BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)
#domain = DomainConfig(
#        V_ext = lambda x: x[:,0:1]**4 + (x[:,1:2]**2 - a)**2 + x[:,0:1] * x[:,1:2] + x[:,1:2] / b,
#        Na = 40,
#
#        input_dim=2,
#        x_l=x_l,
#        x_r=x_r,
#        N_points=20_000,
#        BC_grid=BD_grid,
#        BC_h=BD_h,
#        norm_weight=1.0,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=32,
#        layers=3,
#        positive=True,
#        gaussian_init=True,
#        gaussian_sigma=1,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment=f"_dw{a:.0f}_as{b:.0f}_gauss"
#    )
#
#multiple_runs(domain, model_config, train_config, N_runs=2)
#
## Then retrain with starting from the high-g model, and lowering the Na
#path_high_g = f"models/2D_N32_L3_pos/PINN_Na40.0_x-3.00_-3.00-3.00_3.00_dw{a:.0f}_as{b:.0f}_gauss.pth"
#model_config.gaussian_init = False
#domain.Na = 1.0
#train_config.comment = f"_dw{a:.0f}_as{b:.0f}_high_g"
#
#multiple_runs(domain, model_config, train_config, path_high_g, N_runs=5)


"""
            Penning trap in 2D
"""
#r = 6.0
#N_b = 100
#gamma = 4.0
#x_l = [-r, -r];     x_r = [r, r]
#BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)
#domain = DomainConfig(
#        V_ext = lambda x: 0.5 * (x[:,0:1]**2 + x[:,1:2]**2) + gamma * x[:,0:1]**2 * x[:,1:2]**2,
#        Na = 1,
#
#        input_dim=2,
#        x_l=x_l,
#        x_r=x_r,
#        N_points=20_000,
#        BC_grid=BD_grid,
#        BC_h=BD_h,
#        norm_weight=1.0,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=32,
#        layers=3,
#        positive=True,
#        gaussian_init=True,
#        gaussian_sigma=1,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment=f"_penning_gamma{gamma:.0f}_gauss"
#    )
#
#multiple_runs(domain, model_config, train_config, N_runs=1)


"""
                    Time to go crazy
"""
#d_eps = 0.3
#eps = 9.7 + d_eps
#while eps < 10.00001:
#    r = 3.0
#    N_b = 100
#    a = 2.0
#    n = 5.0
#    x_l = [-r, -r];     x_r = [r, r]
#    BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)
#    domain = DomainConfig(
#            V_ext = lambda x: ((x[:,0:1]**2 + x[:,1:2]**2) - a)**2 + eps*np.cos(n*np.atan2(x[:,1:2], x[:,0:1])),
#            Na = 1,
#
#            input_dim=2,
#            x_l=x_l,
#            x_r=x_r,
#            N_points=40_000,
#            BC_grid=BD_grid,
#            BC_h=BD_h,
#            norm_weight=1.0,
#        )
#
#    model_config = ModelConfig(
#            neurons_per_layer=32,
#            layers=3,
#            positive=True,
#            gaussian_init=False,
#            gaussian_sigma=1,
#        )
#
#    train_config = TrainConfig(
#            # The other defaults are good
#            optimizer="lbfgs",
#            comment=f"_crazy_a{a:.1f}_eps{eps:.1f}_n{n:.0f}_var",
#            save_every=10,
#        )
#
#    load_model = f"models/2D_N32_L3_pos/PINN_Na1.0_x-3.00_-3.00-3.00_3.00_crazy_a2.0_eps{eps-d_eps:.1f}_n5_var.pth"
#    mu = json.load(open(load_model.replace(".pth", "_param.json"), "r"))["mu"]
#
#    N_attempt = 0
#    while N_attempt < 5:
#        solver = GPESolver(domain, model_config, train_config, load_model, mu)
#        solver.train()
#        if solver.best_loss < 1e-5:
#            eps += d_eps
#            break
#        else:
#            N_attempt += 1
#    if N_attempt >= 5:
#        print(f"\n=== Failed to converge for eps={eps:.1f} after {n} attempts ===")
#        break
#

## Then retrain with starting from the high-g model, and lowering the Na
#train_config.comment = train_config.comment.replace("_gauss", "_high_g")
#path_high_g = f"models/2D_N32_L3_pos/PINN_Na{domain.Na:.1f}_x-{r:.2f}_-{r:.2f}-{r:.2f}_{r:.2f}{train_config.comment}.pth"
#model_config.gaussian_init = False
#domain.Na = 1.0
#print(f"\n=== Starting retrain with Na={domain.Na:.1f} ===")
#
#solver = GPESolver(domain, model_config, train_config)
#solver.train()


"""
            Quasicrystal potential in 2D
            V(x,y) = V0 * sum_{k=0}^{4} cos(q_k . r) + 0.5 * omega^2 * (x^2 + y^2)
"""
r = 6.0
N_b = 0
V0 = 1.0
omega = 0.1
d = 2.0

def quasicrystal(x):
    V = 0.5 * omega * (x[:,0:1]**2 + x[:,1:2]**2)
    for k in range(5):
        angle = 2 * np.pi * k / 5
        qx = 2 * np.pi / d * np.cos(angle)
        qy = 2 * np.pi / d * np.sin(angle)
        V = V + V0 * np.cos(qx * x[:,0:1] + qy * x[:,1:2])
    return V

x_l = [-r, -r];     x_r = [r, r]
domain = DomainConfig(
        V_ext = quasicrystal,
        Na = 1.0,

        input_dim=2,
        x_l=x_l,
        x_r=x_r,
        N_points=40_000,
        BC_grid=np.array([]),
        BC_h=np.array([]),
        norm_weight=1.0,
    )

model_config = ModelConfig(
        neurons_per_layer=32,
        layers=3,
        positive=True,
        gaussian_init=False,
        gaussian_sigma=1,
    )

train_config = TrainConfig(
        optimizer="adam+lbfgs",
        adam_epochs=5000,
        comment=f"_quasicrystal_V0{V0:.1f}_d{d:.1f}_omega{omega:.1f}_more_points",
        save_every=10,
    )

load_model = f"models/2D_N32_L3_pos/PINN_Na1.0_x-6.00_-6.00-6.00_6.00_quasicrystal_V01.0_d2.0_omega0.1.pth"
mu = json.load(open(load_model.replace(".pth", "_param.json"), "r"))["mu"]
solver = GPESolver(domain, model_config, train_config, load_model, mu)
solver.train()
#
## Lower Na by 10% iteratively up to 1
#model_config.gaussian_init = False
#Na = 1000
#prev_Na = Na
#for ii in range(65):
#    if ii == 0: continue
#    loss = 1
#    attempt = 0
#    Na *= 0.9
#    load_model = f"models/2D_N32_L3_pos/PINN_Na{prev_Na:.1f}_x-6.00_-6.00-6.00_6.00_quasicrystal_V01.0_d2.0_omega0.1.pth"
#    mu = json.load(open(load_model.replace(".pth", "_param.json"), "r"))["mu"]
#    domain.Na = Na
#    while loss > 1e-4 and attempt < 5:
#        solver = GPESolver(domain, model_config, train_config, load_model, mu)
#        solver.train()
#        loss = solver.best_loss
#        attempt += 1
#    prev_Na = Na
#    if Na < 1.0:
#        print(f"\n=== Reached Na={Na:.1f} with loss={loss:.3e} ===")
#        break
