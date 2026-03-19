from functions import *


def multiple_runs(domain, model_config, train_config, N_runs=5):
    comment = train_config.comment
    runs_best_loss = float("inf")
    losses = []
    best_base = None
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
    train_config.comment = comment  # reset comment to original for next set of runs

    if best_base is not None:
        final_base = best_base.replace("-best", "")
        os.rename(f"{best_base}.pth", f"{final_base}.pth")
        os.rename(f"{best_base}_history.csv", f"{final_base}_history.csv")
        os.rename(f"{best_base}_param.json", f"{final_base}_param.json")
        print(f"\n=== All runs completed ===")
        print(f"Best loss across runs: {runs_best_loss:.3e}")
        print(f"Losses for all runs: {', '.join(f'{l:.3e}' for l in losses)}")
    else:
        print(f"\n=== All runs completed ===")
        print(f"No successful runs found.")
        print(f"Losses for all runs: {', '.join(f'{l:.3e}' for l in losses)}")


x_l = [1e-7]
x_r = [6.0]

domain = DomainConfig(
        input_dim=1,
        x_l=x_l,
        x_r=x_r,
        N_points=1_000,
        V_ext = lambda x: 0.5 * x**2,
        Na = 0.0,
        BC_grid=np.array([x_l, x_r]),  # shape: (2, 1)
        BC_h=np.array([[0.0], [0.0]]),  # shape: (2, 1)
    )

model_config = ModelConfig(
        neurons_per_layer=2,
        layers=1,
        positive=False,
    )

train_config = TrainConfig(
        # The other defaults are good
        optimizer="lbfgs",
        comment=""
    )

multiple_runs(domain, model_config, train_config, N_runs=5)
