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


""" l = general vortex """
l = 6
r = 5.0
g = 10.0
vec = np.linspace(-r, r, 100)
ones = np.ones_like(vec)

left   = np.column_stack((-r * ones, vec))
right  = np.column_stack(( r * ones, vec))
top    = np.column_stack((vec,  r * ones))
bottom = np.column_stack((vec, -r * ones))
BC = np.vstack([left, right, top, bottom])

# compute real and immaginary parts
z = (BC[:, 0] + 1j * BC[:, 1])
z /= np.abs(z)
z_l = z ** l

BC_u = z_l.real.reshape(-1, 1)
BC_v = z_l.imag.reshape(-1, 1)
BC_h = np.hstack([BC_u, BC_v])


domain = DomainConfig(
    input_dim=2,
    x_l=[-r, -r],
    x_r=[r, r],
    N_points=20_000,
    Na = g,
    B = 0.0,
    BC = BC,
    BC_h = BC_h,
    BC_w = 1.0,
)

model_config = ModelConfig(
        neurons_per_layer=32,
        layers=4,
    )

train_config = TrainConfig(
        # The other defaults are good
        optimizer="lbfgs",
        comment=f"_l{l:.0f}",
        save_every=100,
    )

path = "models/N32_L4/PINN_Na10.0_x-5.00_-5.00-5.00_5.00_l6-run2.pth"
GPE_solver = GPESolver(domain, model_config, train_config, load_model=path)
GPE_solver.train()
