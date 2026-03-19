import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('module://matplotlib-backend-kitty')
import json

SMALL_SIZE = 13
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize


def plot_NN(NpL, layers, x_l, x_r, Na, comment, save=False):
    model = nn.Sequential(
        nn.Linear(1, NpL),
        nn.Tanh(),
        *[
            layer
            for _ in range(layers - 1)
            for layer in (nn.Linear(NpL, NpL), nn.Tanh())
        ],
        nn.Linear(NpL, 1)
    )

    dir = f"models/N{NpL}_L{layers}"
    name = f"PINN_Na{Na:.1f}_x{x_l:.2f}-{x_r:.2f}{comment}"
    model.load_state_dict(torch.load(f"{dir}/{name}.pth"))
    model.eval()  # important for inference

    param = json.load(open(f"{dir}/{name}_param.json"))

    # Step 3: Plot and compare with exact solution
    if Na in[1, 10]:
        data = np.loadtxt(f"data/GPE_Na{Na}_numerov.csv", delimiter=',', skiprows=1)
        x = data[:, 0]
        u_exact = data[:, 1]
    elif Na == 100:
        data = np.loadtxt(f"data/GPE_Na{Na}_numerov.csv", delimiter=',', skiprows=1)
        x_num = data[:, 0]
        u_num = data[:, 1]
        data = np.loadtxt(f"data/GPE_Na{Na}_variational.csv", delimiter=',', skiprows=1)
        x = data[:, 0]
        u_exact = data[:, 1]

    elif Na == 0:
        x = np.linspace(x_l, x_r, 1000)
        u_exact = 2 * np.pi**(-1/4) * x * np.exp(-x**2 / 2)
    else:
        print(f"Exact solution for Na={Na} not available.")
        x = np.linspace(x_l, x_r, int(x_r - x_l) * 1000)
        u_exact = np.zeros_like(x)

    u_pred = model(torch.tensor(x, dtype=torch.float32).reshape(-1, 1)).detach().numpy().flatten()
    u_pred = u_pred * np.sign(u_pred[len(u_pred) // 2])  # Ensure correct sign

    L2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    print(f"N{NpL}L{layers}, loss = {param["loss"]:.2e}, L2 error = {L2_error:.2e}")
    return

    loss = np.loadtxt(f"{dir}/{name}_history.csv", delimiter=',', skiprows=1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss[:, 0], loss[:, 1])
    ax[0].set_yscale('log')
    ax[0].set_xticks(ticks=(np.linspace(0, len(loss[:,0]), 6).astype(int)))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].grid()
    ax[0].set_title(f'Loss={param["loss"]:.2e}')

    ax[1].plot(x, u_pred, label='PINN Prediction', linewidth=2)
    if Na == 100:
        ax[1].plot(x, u_exact, label='Variational Solution', linestyle='dashed', linewidth=2)
        ax[1].plot(x_num, u_num, label='Numerov Solution', linestyle='dotted', linewidth=2)
    else:
        ax[1].plot(x, u_exact, label='Numerov Solution', linestyle='dashed')
    ax[1].legend()
    ax[1].set_xlabel('r')
    ax[1].set_ylabel('u(r)')
    ax[1].grid()
    ax[1].set_title(f'L2 Error={L2_error:.2e}')

    fig.suptitle(fr'Na = {Na}, $\mu={param["mu"]:.3f}$, N={NpL}, L={layers}',
                 fontsize=BIGGER_SIZE)
    if save:
        plt.savefig(f"{dir}/{name}.png", dpi=300)
        plt.savefig(f"../report/Figures/ex1_N{NpL}L{layers}_{name}.eps")
    else:
        plt.show()


save = False

#for N in [4, 8, 16, 32]:
#    for L in [1, 2, 3, 4]:
#        plot_NN(NpL=N, layers=L, x_l=1e-6, x_r=6, Na=10, comment="", save=save)

for N in [2, 3, 4, 8, 16, 32]:
    for L in [1, 2, 3, 4]:
        plot_NN(NpL=N, layers=L, x_l=1e-7, x_r=6, Na=0, comment="", save=save)

#for N_points in [10, 100, 500, 1_000, 5000, 10_000, 20_000]:
#    plot_NN(NpL=16, layers=3, x_l=1e-7, x_r=6, Na=0, comment=f"_N_points{N_points}", save=save)


#plot_NN(NpL=1, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
#plot_NN(NpL=2, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
#plot_NN(NpL=3, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
#plot_NN(NpL=4, layers=1, x_l=1e-6, x_r=6, Na=10, comment="", save=save)
#plot_NN(NpL=4, layers=1, x_l=1e-6, x_r=6, Na=1, comment="", save=save)
#plot_NN(NpL=4, layers=1, x_l=1e-6, x_r=8, Na=100, comment="", save=save)
#plot_NN(NpL=8, layers=2, x_l=1e-6, x_r=8, Na=100, comment="", save=save)
