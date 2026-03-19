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
            gaussian_init=False,
        )
    model = build_model(model_config, dim, "cpu", load_path=f"{fpath}.pth") 

    param = json.load(open(f"{fpath}_param.json"))

    return fpath, model, param


def plot_GPE_1D(NpL, layers, x_l, x_r, Na, comment="", positive=True, save=False):

    fpath, model, param = get_model(NpL, layers, 1, [x_l], [x_r], Na, comment, positive)

    xx = torch.from_numpy(np.linspace(x_l, x_r, 1000).reshape(-1, 1)).float()
    h = model(xx).detach().numpy().squeeze()
    max_h = np.argmax(np.abs(h))
    if h[max_h] < 0:
        h = -h

    x = np.array(param["x"]).squeeze()
    V_ext = np.array(param["V_ext"]).squeeze()
    
    upper_bound = 1.2
    lower_bound = min(h.min(), V_ext.min())
    lower_bound -= (upper_bound - lower_bound) * 0.05

    plt.figure()
    plt.plot(xx.numpy(), h, label=r'$\psi(x)$', linewidth=2)
    plt.plot(x, V_ext, '. ', markersize=2, label=r'$V_{\rm ext}(x)$')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel(r'$\psi(x), ~ V_{\rm ext}(x)$')
    plt.ylim(lower_bound, upper_bound)
    plt.grid()
    plt.title(fr'N{NpL}L{layers}, Loss$={param["loss"]:.2e}$'
              +'\n' + fr'$g = {Na}$, $\mu={param["mu"]:.3f}$')
    plt.tight_layout()
    if save:
        plt.savefig(f"{fpath}.png", dpi=300)
        fname = f"../report/Figures/ex3_1D_N{NpL}L{layers}"
        if positive:
            fname += f"_pos"
        plt.savefig(f"{fname}_g{Na}{comment}.eps")
    else:
        plt.show()


def plot_GPE_2D(NpL, layers, x_l, x_r, Na, comment="", positive=True, save=False):

    fpath, model, param = get_model(NpL, layers, 2, x_l, x_r, Na, comment, positive)

    x = np.array(param["x"])
    V_ext = np.array(param["V_ext"]).squeeze()

    n = 720
    xs = np.linspace(x_l[0], x_r[0], n)
    ys = np.linspace(x_l[1], x_r[1], n)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.from_numpy(
        np.stack([XX.ravel(), YY.ravel()], axis=1)
    ).float()

    with torch.no_grad():
        h = model(grid).numpy().reshape(n, n)
    #h *= h

    from scipy.interpolate import griddata
    from scipy.ndimage import minimum_filter, maximum_filter

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── V_ext scatter plot ──────────────────────────────────────────────────
    norm = mcolors.SymLogNorm(
        linthresh=0.5,
        linscale=1.0,
        vmin=V_ext.min(),
        vmax=V_ext.max()
    )

    #sc = axes[0].scatter(x[:, 0], x[:, 1], c=V_ext, cmap="viridis", s=2, norm=norm)
    #cbar = fig.colorbar(sc, ax=axes[0])
    #n_ticks = 8
    #ticks = norm.inverse(np.linspace(0, 1, n_ticks))
    #cbar.set_ticks(ticks) # pyright: ignore
    #cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    V_grid = griddata(x[:, :2], V_ext, (XX, YY), method="linear")
    V_grid = np.nan_to_num(V_grid, nan=np.nanmax(V_ext))
    im0 = axes[0].imshow(V_grid, extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
                            origin="lower", cmap="viridis", aspect="equal")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(r'$V_{\rm ext}(x,y)$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # ── Local minima of V_ext ───────────────────────────────────────────────
    #V_grid = griddata(x[:, :2], V_ext, (XX, YY), method='linear')
    #V_grid = np.nan_to_num(V_grid, nan=np.nanmax(V_grid))

    #neighborhood = 40
    #local_V_minima = (V_grid == minimum_filter(V_grid, size=neighborhood))
    #local_V_minima &= (V_grid < np.nanpercentile(V_grid, 20))

    #min_coords = np.argwhere(local_V_minima)
    #min_px = [xs[ix] for _, ix in min_coords]
    #min_py = [ys[iy] for iy, _ in min_coords]
    #min_vals = [V_grid[iy, ix] for iy, ix in min_coords]

    #for px, py, val in zip(min_px, min_py, min_vals):
    #    axes[0].scatter(px, py, s=100,  marker='v', edgecolors='white',
    #                    linewidths=2, label=f"{val:.2f}")
    #axes[0].legend(loc='upper right', title='Minima')

    # ── h plot ────────────────────────────────────────────────────────────
    im1 = axes[1].imshow(h, extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
                         origin="lower", cmap="hot", aspect="equal")
                         #norm=PowerNorm(gamma=0.2))
    cbar = fig.colorbar(im1, ax=axes[1])
    #cbar.set_ticks(np.geomspace(h.min(), h.max(), 5))

    axes[1].set_title(r'$\psi(x,y)$')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    # ── Local maxima of h ───────────────────────────────────────────────────
    #local_h_maxima = (h == maximum_filter(h, size=neighborhood))
    #local_h_maxima &= (h > np.percentile(h, 80))

    #max_coords = np.argwhere(local_h_maxima)
    #max_px = [xs[ix] for _, ix in max_coords]
    #max_py = [ys[iy] for iy, _ in max_coords]
    #max_vals = [h[iy, ix] for iy, ix in max_coords]

    #for px, py, val in zip(max_px, max_py, max_vals):
    #    axes[1].scatter(px, py, s=100, marker='^', edgecolors='white',
    #                    linewidths=2, label=f"{val:.2f}")
    #axes[1].legend(loc='upper right', title='Maxima')

    fig.suptitle(
            fr'N{NpL}L{layers}, Loss$={param["loss"]:.2e}$, '
            + fr'$g = {Na}$, $\mu={param["mu"]:.3f}$'
    )

    plt.tight_layout()
    if save:
        plt.savefig(f"{fpath}.png", dpi=300)
        plt.savefig(f"../report/Figures/ex3_2D_N{NpL}L{layers}_g{Na}{comment}.eps")
    else:
        plt.show()


def plot_gaussian(NpL, layers, D, sigma, r, comment="", save=False):
    path = f"models/N{NpL}_L{layers}/gaussian_{D:.0f}D_sigma{sigma:.2f}_r{r:.1f}.pth"
    model_config = ModelConfig(
            neurons_per_layer=NpL,
            layers=layers,
            positive=True,
            gaussian_init=False,
        )
    model = build_model(model_config, D, "cpu", load_path=path)

    r = 3.0

    if D == 1:
        xx = torch.from_numpy(np.linspace(-r, r, 1000).reshape(-1, 1)).float()
        h = model(xx).detach().numpy().squeeze()
        h2 = h**2

        plt.figure()
        plt.plot(xx.numpy(), h2, label=r'$|\psi(x)|^2$')
        plt.plot(xx.numpy(), h, label=r'$\psi(x)$')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel(r'$|\psi(x)|^2$')
        plt.ylim(-0.1, 1.1)
        plt.grid()
        plt.title(fr'Gaussian in {D}D with $\sigma={sigma}$' + '\n'
                  + f'N={NpL}, L={layers}')
        if save:
            plt.savefig(f"{path[:-4]}.png", dpi=300)
        else:
            plt.show()
    elif D == 2:
        # color map
        n = 400
        xs = np.linspace(-r, r, n)
        ys = np.linspace(-r, r, n)
        XX, YY = np.meshgrid(xs, ys)
        grid = torch.from_numpy(
            np.stack([XX.ravel(), YY.ravel()], axis=1)
        ).float()
        with torch.no_grad():
            h = model(grid).numpy().reshape(n, n)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(
            h, extent=[-r, r, -r, r],
            origin="lower", cmap="hot", aspect="equal"
        )
        plt.colorbar(im)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(fr'Gaussian in {D}D with $\sigma={sigma}$' + '\n'
                  + f'N={NpL}, L={layers}')
        if save:
            plt.savefig(f"{path[:-4]}.png", dpi=300)
        else:
            plt.show()


save = False

#plot_gaussian(NpL=32, layers=4, D=2, sigma=1.0, r=10.0, save=save)
#plot_gaussian(NpL=16, layers=3, D=1, sigma=1.0, r=10.0, save=save)
#plot_gaussian(NpL=16, layers=3, D=1, sigma=2.0, r=10.0, save=save)

""" new """
#plot_GPE_1D(NpL=4, layers=2, x_l=-3, x_r=3, Na=1, positive=False, comment="_dw1", save=save)
#plot_GPE_1D(NpL=4, layers=2, x_l=-3, x_r=3, Na=1, positive=False, comment="_dw2", save=save)
#plot_GPE_1D(NpL=4, layers=2, x_l=-3, x_r=3, Na=1, positive=True, comment="_dw2", save=save)
#plot_GPE_1D(NpL=4, layers=2, x_l=-3, x_r=3, Na=1, positive=True, comment="_dw2_gauss", save=save)
#plot_GPE_1D(NpL=16, layers=2, x_l=-3, x_r=3, Na=100, positive=True, comment="_dw2_gauss", save=save)
#plot_GPE_1D(NpL=16, layers=2, x_l=-3, x_r=3, Na=1,   positive=True, comment="_dw2_high_g", save=save)
#plot_GPE_1D(NpL=16, layers=2, x_l=-3, x_r=3, Na=100, positive=True, comment="_dw1_as6_gauss", save=save)
#plot_GPE_1D(NpL=16, layers=2, x_l=-3, x_r=3, Na=1,   positive=True, comment="_dw1_as6_high_g", save=save)


#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=0, comment="_harmonic", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1, comment="_harmonic", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=40, comment="_dw2_as4_gauss", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=1, comment="_dw2_as4_high_g", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=40, comment="_dw2_as6_gauss", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=1, comment="_dw2_as6_high_g", save=save)

#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=1, comment="_crazy_a2.0_eps0.4_n5_high_g", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=1, comment="_crazy_a2.0_eps1.0_n5_high_g", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-3, -3], x_r=[3, 3], Na=1, comment="_crazy_a2.0_eps10.0_n5_var", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1, comment="_penning_gamma4_gauss", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1000, comment="_quasicrystal_V01.0_d2.0_omega0.3", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=100, comment="_quasicrystal_V01.0_d2.0_omega0.3", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1000, comment="_quasicrystal_V01.0_d2.0_omega0.3", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1000, comment="_quasicrystal_V01.0_d2.0_omega0.1", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=106.6, comment="_quasicrystal_V01.0_d2.0_omega0.1", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=10.5, comment="_quasicrystal_V01.0_d2.0_omega0.1", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=5, comment="_quasicrystal_V01.0_d2.0_omega0.1", save=save)
#plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1, comment="_quasicrystal_V01.0_d2.0_omega0.1", save=save)
plot_GPE_2D(NpL=32, layers=3, x_l=[-6, -6], x_r=[6, 6], Na=1, comment="_quasicrystal_V01.0_d2.0_omega0.1_more_points", save=save)

