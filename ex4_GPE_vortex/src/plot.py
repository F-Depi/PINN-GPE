from functions import build_model, ModelConfig

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


def get_model(NpL, layers, dim, x_l, x_r, Na, comment=""):
    dir = f"models/N{NpL}_L{layers}"
    name = f"PINN_Na{Na:.1f}_x{'_'.join(f'{v:.2f}' for v in x_l)}-{'_'.join(f'{v:.2f}' for v in x_r)}{comment}"
    fpath = f"{dir}/{name}"

    model_config = ModelConfig(
            neurons_per_layer=NpL,
            layers=layers,
        )
    model = build_model(model_config, dim, "cpu", load_path=f"{fpath}.pth") 

    param = json.load(open(f"{fpath}_param.json"))

    return fpath, model, param


def plot_GPE_2D(NpL, layers, x_l, x_r, Na, comment="", save=False, full=False):

    fpath, model, param = get_model(NpL, layers, 2, x_l, x_r, Na, comment)

    n = 200
    xs = np.linspace(x_l[0], x_r[0], n)
    ys = np.linspace(x_l[1], x_r[1], n)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.from_numpy(
        np.stack([XX.ravel(), YY.ravel()], axis=1)
    ).float()

    with torch.no_grad():
        h = model(grid).numpy().reshape(n, n, -1)
    u = h[:, :, 0]
    v = h[:, :, 1]

    if full:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if full:
        u_max = np.max(np.abs(u))
        im0 = axes[0,0].imshow(
            u, extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
            origin="lower", cmap="RdBu_r", aspect="equal",
            vmin=-u_max, vmax=u_max 
        )
        fig.colorbar(im0, ax=axes[0,0])
        axes[0,0].set_title(r'Re{$\psi(x,y)$}')
        axes[0,0].set_xlabel('x')
        axes[0,0].set_ylabel('y')

        v_max = np.max(np.abs(v))
        im1 = axes[0,1].imshow(
            v, extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
            origin="lower", cmap="RdBu_r", aspect="equal",
            vmin=-v_max, vmax=v_max 
        )
        fig.colorbar(im1, ax=axes[0,1])
        axes[0,1].set_title(r'Im{$\psi(x,y)$}')
        axes[0,1].set_xlabel('x')
        axes[0,1].set_ylabel('y')
        
        ax1 = axes[1, 0]
        ax2 = axes[1, 1]
    else:
        ax1 = axes[0]
        ax2 = axes[1]

    h2 = u**2 + v**2
    im2 = ax1.imshow(
        h2, extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
        origin="lower", cmap="hot", aspect="equal",
    )
    fig.colorbar(im2, ax=ax1)
    ax1.set_title(r'$|\psi(x,y)|^2$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    phase = np.arctan2(v, u) / np.pi  # Normalize phase to [-1, 1]
    im3 = ax2.imshow(
        phase, extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
        origin="lower", cmap="hsv", aspect="equal",
        )
    fig.colorbar(im3, ax=ax2)
    ax2.set_title(r'arg{$\psi(x,y) / \pi$}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    fig.suptitle(fr'N{NpL}L{layers}, Loss$={param["loss"]:.2e}$, '
               + fr'$g = {Na}$, $\mu={param["mu"]:.3f}$')

    plt.tight_layout()
    if save:
        full_char = "_full" if full else ""
        plt.savefig(f"{fpath}{full_char}.png", dpi=300)
        plt.savefig(f"../report/Figures/ex4_N{NpL}L{layers}_{fpath.split('/')[-1]}{full_char}.eps")
    else:
        plt.show()


save = False
full = False

#plot_GPE_2D(NpL=32, layers=4, x_l=[-5, -5], x_r=[5, 5], Na=1, comment="_l1", save=save, full=full) 
#plot_GPE_2D(NpL=32, layers=4, x_l=[-5, -5], x_r=[5, 5], Na=1, comment="_l2", save=save, full=full) 
#plot_GPE_2D(NpL=32, layers=4, x_l=[-7, -7], x_r=[7, 7], Na=1, comment="_l3", save=save, full=full) 
#plot_GPE_2D(NpL=32, layers=4, x_l=[-10, -10], x_r=[10, 10], Na=1, comment="_l6", save=save, full=full) 


#plot_GPE_2D(NpL=32, layers=4, x_l=[-5, -5], x_r=[5, 5], Na=10, comment="_l1", save=save, full=full) 
#plot_GPE_2D(NpL=32, layers=4, x_l=[-5, -5], x_r=[5, 5], Na=10, comment="_l6", save=save, full=full) 

