from functions import *


def multiple_runs(domain, model_config, train_config, N_runs=5):
    comment = train_config.comment
    runs_best_loss = float("inf")
    losses = []
    for i in range(N_runs):
        print(f"\n=== Starting run {i+1}/5 ===")
        train_config.comment = comment + f"-run{i+1}"
        solver = GPESolver(domain, model_config, train_config)
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

    final_base = best_base.replace("-best", "")
    os.rename(f"{best_base}.pth", f"{final_base}.pth")
    os.rename(f"{best_base}_history.csv", f"{final_base}_history.csv")
    os.rename(f"{best_base}_param.json", f"{final_base}_param.json")
    print(f"\n=== All runs completed ===")
    print(f"Best loss across runs: {runs_best_loss:.3e}")
    print(f"Losses for all runs: {', '.join(f'{l:.3e}' for l in losses)}")


"""
1D potentials:
    (x**2 - 1.0)**2                 dw1
    (x**2 - 2.0)**2                 dw2
    (x**2 - 2.0)**2 + x / 6.0       dw2_as6
"""

#r = 8
#
#domain = DomainConfig(
#        input_dim=1,
#        N_eigenstates=6,
#        x_l=[-r],
#        x_r=[r],
#        N_points=3000,
#        V_ext = lambda x: 0.5 * x**2,
#        Na = 0,
#        BC_grid=np.array([[r], [r]]),  # shape: (2, 1)
#        BC_h=np.array([[0.0], [0.0]]),  # shape: (2, 1)
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=16,
#        layers=3,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment="_harmonic_eigen"
#    )
#
##gpe_solver = GPESolver(domain, model_config, train_config)
##gpe_solver.train()
#
#multiple_runs(domain, model_config, train_config, N_runs=5)


"""
2D potentials:
    (x[:,0:1]**2 + x[:,1:2]**2) / 2.0   harmonic
"""
#r = 5.0
#N_b = 100
#
#x_l = [-r, -r];     x_r = [r, r]
#BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)
#
#domain = DomainConfig(
#        V_ext = lambda x: 0.5 * (x[:,0:1]**2 + x[:,1:2]**2),
#        Na = 1.0,
#
#        input_dim=2,
#        N_eigenstates=5,
#        x_l=x_l,
#        x_r=x_r,
#        N_points=20_000,
#        BC_grid=BD_grid,
#        BC_h=BD_h,
#    )
#
#model_config = ModelConfig(
#        neurons_per_layer=32,
#        layers=4,
#        positive=False,
#    )
#
#train_config = TrainConfig(
#        # The other defaults are good
#        optimizer="lbfgs",
#        comment="_harmonic_eigen"
#    )
#
#GPE_solver = GPESolver(domain, model_config, train_config)
#GPE_solver.train()


"""
2D potentials:
    x[:,0:1]**4 + (x[:,1:2]**2 - 2.0)**2 + x[:,0:1] * x[:,1:2] + x[:,0:1]/6  dw2_as6
"""
r = 3.0
N_b = 100

x_l = [-r, -r];     x_r = [r, r]
BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)

domain = DomainConfig(
        V_ext = lambda x: x[:,0:1]**4 + (x[:,1:2]**2 - 2.0)**2 + x[:,0:1] * x[:,1:2] + x[:,0:1]/6,
        Na = 0.0,

        input_dim=2,
        N_eigenstates=5,
        x_l=x_l,
        x_r=x_r,
        N_points=20_000,
        BC_grid=BD_grid,
        BC_h=BD_h,
    )

model_config = ModelConfig(
        neurons_per_layer=32,
        layers=4,
        positive=False,
    )

train_config = TrainConfig(
        # The other defaults are good
        optimizer="lbfgs",
        comment="_dw2_as6_eigen"
    )

GPE_solver = GPESolver(domain, model_config, train_config)
GPE_solver.train()
