from functions import build_model, ModelConfig

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.colors import PowerNorm
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


def get_model(NpL, layers, dim, x_l, x_r, Na, comment="", positive=False):
    dir_ = f"models/{dim:.0f}D_N{NpL}_L{layers}"
    if positive:
        dir_ += "_pos"
    name = f"PINN_Na{Na:.1f}_x{'_'.join(f'{v:.2f}' for v in x_l)}-{'_'.join(f'{v:.2f}' for v in x_r)}{comment}"
    fpath = f"{dir_}/{name}"

    model_config = ModelConfig(
            neurons_per_layer=NpL,
            layers=layers,
            positive=positive,
        )
    model = build_model(model_config, dim, "cpu", load_path=f"{fpath}.pth") 

    param = json.load(open(f"{fpath}_param.json"))

    return fpath, model, param


def plot_GPE_2D(NpL, layers, x_l, x_r, Na, comment="", positive=True, save=False):

    fpath, model, param = get_model(NpL, layers, 2, x_l, x_r, Na, comment, positive)

    n = 720
    xs = np.linspace(x_l[0], x_r[0], n)
    ys = np.linspace(x_l[1], x_r[1], n)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.from_numpy(
        np.stack([XX.ravel(), YY.ravel()], axis=1)
    ).float()

    with torch.no_grad():
        h = model(grid).numpy().reshape(n, n)

    h_exact = np.exp(-10 * grid[:, 0].numpy()**2) * np.exp(-10 * grid[:, 1].numpy()**2)
    L2_error = np.sqrt(np.mean((h - h_exact.reshape(n, n))**2) / np.mean(h_exact.reshape(n, n)**2))
    print(f"L2 error of the 2D solution: {L2_error:.2e}")


    plt.figure()
    im = plt.imshow(h, extent=[x_l[0], x_r[0], x_l[1], x_r[1]], origin="lower", cmap="hot", aspect="equal")
    plt.colorbar(im)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(fr'N{NpL}L{layers}, Loss$={param["loss"]:.2e}$, $g = {Na}$, $\mu={param["mu"]:.3f}$')
    plt.tight_layout()

    # Plot a slice at y=0
    h_slice = h[n//2, :]
    plt.figure()
    plt.plot(xs, h_slice)
    plt.plot(xs, h_exact[n//2 * n:(n//2 + 1) * n], label="Exact", linestyle="dashed")
    plt.xlabel('x')
    plt.ylabel('h(x, 0)')
    plt.title(fr'Slice at $y=0$, N{NpL}L{layers}, Loss$={param["loss"]:.2e}$, $g = {Na}$, $\mu={param["mu"]:.3f}$')


    if save:
        plt.savefig(f"{fpath}.png", dpi=300)
    else:
        plt.show()



save = False

plot_GPE_2D(NpL=16, layers=2, x_l=[-1, -1], x_r=[1, 1], Na=0, comment="", save=save)


