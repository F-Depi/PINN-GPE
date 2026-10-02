import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# import matplotlib
# matplotlib.use('module://matplotlib-backend-kitty')

SMALL_SIZE = 13
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize


def plot_NN(NpL, layers, x_l, x_r, Na, comment, save=False):
    model = nn.Sequential(
        nn.Linear(1, NpL),
        nn.Tanh(),
        *[
            layer
            for _ in range(layers - 1)
            for layer in (nn.Linear(NpL, NpL), nn.Tanh())
        ],
        nn.Linear(NpL, 1),
    )

    dir = f"models/N{NpL}_L{layers}"
    name = f"PINN_Na{Na:.1f}_x{x_l:.2f}-{x_r:.2f}{comment}"
    model.load_state_dict(torch.load(f"{dir}/{name}.pth"))
    model.eval()  # important for inference

    param = json.load(open(f"{dir}/{name}_param.json"))

    # Step 3: Plot and compare with exact solution
    if Na in [1, 10]:
        data = np.loadtxt(f"data/GPE_Na{Na}_numerov.csv", delimiter=",", skiprows=1)
        x = data[:, 0]
        u_exact = data[:, 1]

    elif Na == 100:
        data = np.loadtxt(f"data/GPE_Na{Na}_numerov.csv", delimiter=",", skiprows=1)
        x_num = data[:, 0]
        u_num = data[:, 1]
        data = np.loadtxt(f"data/GPE_Na{Na}_variational.csv", delimiter=",", skiprows=1)
        x = data[:, 0]
        u_exact = data[:, 1]

    elif Na == 0:
        x = np.linspace(x_l, x_r, 1000)
        u_exact = 2 * np.pi ** (-1 / 4) * x * np.exp(-(x**2) / 2)
    else:
        print(f"Exact solution for Na={Na} not available.")
        x = np.linspace(x_l, x_r, int(x_r - x_l) * 1000)
        u_exact = np.zeros_like(x)

    u_pred = (
        model(torch.tensor(x, dtype=torch.float32).reshape(-1, 1))
        .detach()
        .numpy()
        .flatten()
    )
    u_pred = u_pred * np.sign(u_pred[len(u_pred) // 2])  # Ensure correct sign

    L2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    print(f"N{NpL}L{layers}, loss = {param['loss']:.2e}, L2 error = {L2_error:.2e}")

    loss = np.loadtxt(f"{dir}/{name}_history.csv", delimiter=",", skiprows=1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss[:, 0], loss[:, 1])
    ax[0].set_yscale("log")
    ax[0].set_xticks(ticks=(np.linspace(0, len(loss[:, 0]), 6).astype(int)))
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid()
    ax[0].set_title(f"Loss={param['loss']:.2e}")

    ax[1].plot(x, u_pred, label="PINN Prediction", linewidth=2)
    if Na == 100:
        ax[1].plot(
            x, u_exact, label="Variational Solution", linestyle="dashed", linewidth=2
        )
        # ax[1].plot(
        #    x_num, u_num, label="Numerov Solution", linestyle="dotted", linewidth=2
        # )
    else:
        ax[1].plot(x, u_exact, label="Numerov Solution", linestyle="dashed")
    ax[1].legend()
    ax[1].set_xlabel("r")
    ax[1].set_ylabel("u(r)")
    ax[1].grid()
    ax[1].set_title(f"L2 Error={L2_error:.2e}")

    fig.suptitle(
        rf"Na = {Na}, $\mu={param['mu']:.3f}$, N={NpL}, L={layers}",
        fontsize=BIGGER_SIZE,
    )
    if save:
        plt.savefig(f"{dir}/{name}.png", dpi=300)
        plt.savefig(f"../report/Figures/ex1_N{NpL}L{layers}_{name}.eps")
    else:
        plt.show()


def build_model(NpL, layers):
    return nn.Sequential(
        nn.Linear(1, NpL),
        nn.Tanh(),
        *[m for _ in range(layers - 1) for m in (nn.Linear(NpL, NpL), nn.Tanh())],
        nn.Linear(NpL, 1),
    )


def load_pinn(NpL, layers, x_l, x_r, Na, comment):
    folder = f"models/N{NpL}_L{layers}"
    name = f"PINN_Na{Na:.1f}_x{x_l:.2f}-{x_r:.2f}{comment}"
    model = build_model(NpL, layers)
    model.load_state_dict(torch.load(f"{folder}/{name}.pth"))
    model.eval()
    with open(f"{folder}/{name}_param.json") as f:
        param = json.load(f)
    return model, param


def load_exact(Na, x_l, x_r):
    """Returns (x grid, reference solution, label)."""
    if Na in (1, 10):
        data = np.loadtxt(f"data/GPE_Na{Na}_numerov.csv", delimiter=",", skiprows=1)
        return data[:, 0], data[:, 1], "Numerov Solution"
    if Na == 100:
        data = np.loadtxt(f"data/GPE_Na{Na}_variational.csv", delimiter=",", skiprows=1)
        return data[:, 0], data[:, 1], "Variational Solution"
    if Na == 0:
        x = np.linspace(x_l, x_r, 1000)
        return x, 2 * np.pi ** (-1 / 4) * x * np.exp(-(x**2) / 2), "Analytical Solution"
    print(f"Exact solution for Na={Na} not available.")
    x = np.linspace(x_l, x_r, max(int(x_r - x_l), 1) * 1000)
    return x, np.zeros_like(x), None


def plot_panel(ax, NpL, layers, x_l, x_r, Na, comment):
    model, param = load_pinn(NpL, layers, x_l, x_r, Na, comment)
    x, u_exact, label = load_exact(Na, x_l, x_r)

    with torch.no_grad():
        u_pred = model(torch.tensor(x, dtype=torch.float32).reshape(-1, 1)).numpy().flatten()
    u_pred *= np.sign(u_pred[len(u_pred) // 2])  # fix overall sign

    L2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    print(f"N{NpL}L{layers}, loss = {param['loss']:.2e}, L2 error = {L2_error:.2e}")

    ax.plot(x, u_pred, label="PINN Prediction", linewidth=2)
    if label is not None:
        ax.plot(x, u_exact, label=label, linestyle="dashed", linewidth=2)
    ax.legend()
    ax.set_xlabel("r")
    ax.set_ylabel("u(r)")
    ax.grid()
    ax.set_title(
        rf"Na = {Na}, $\mu={param['mu']:.3f}$, N={NpL}, L={layers}"
        f"\nL2 Error={L2_error:.2e}"
    )


def plot_2in1(NpL_l, layers_l, x_l_l, x_r_l, Na_l, comment_l,
              NpL_r, layers_r, x_l_r, x_r_r, Na_r, comment_r,
              save=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    plot_panel(ax[0], NpL_l, layers_l, x_l_l, x_r_l, Na_l, comment_l)
    plot_panel(ax[1], NpL_r, layers_r, x_l_r, x_r_r, Na_r, comment_r)
    #fig.tight_layout()
    if save:
        plt.savefig(f"../report/Figures/{save}.eps")
    else:
        plt.show()


save = False

# for N in [4, 8, 16, 32]:
#    for L in [1, 2, 3, 4]:
#        plot_NN(NpL=N, layers=L, x_l=1e-6, x_r=6, Na=10, comment="", save=save)

# for N in [2, 3, 4, 8, 16, 32]:
#    for L in [1, 2, 3, 4]:
#        plot_NN(NpL=N, layers=L, x_l=1e-7, x_r=6, Na=0, comment="", save=save)

# for N_points in [10, 100, 500, 1_000, 5000, 10_000, 20_000]:
#    plot_NN(NpL=16, layers=3, x_l=1e-7, x_r=6, Na=0, comment=f"_N_points{N_points}", save=save)


# plot_NN(NpL=1, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
# plot_NN(NpL=2, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
# plot_NN(NpL=3, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
# plot_NN(NpL=4, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
# plot_NN(NpL=4, layers=1, x_l=1e-6, x_r=6, Na=1, comment="", save=save)
# plot_NN(NpL=4, layers=1, x_l=1e-6, x_r=8, Na=100, comment="", save=save)
# plot_NN(NpL=8, layers=2, x_l=1e-6, x_r=8, Na=100, comment="", save=save)
plot_2in1(NpL_l=4, layers_l=1, x_l_l=1e-6, x_r_l=6, Na_l=10, comment_l="",
          NpL_r=8, layers_r=2, x_l_r=1e-6, x_r_r=8, Na_r=100, comment_r="",
          save='ex1_Na10vsNa100')
