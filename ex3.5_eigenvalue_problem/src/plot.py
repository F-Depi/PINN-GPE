import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from functions import build_model, ModelConfig
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


def plot_multiGPE_1D(NpL, layers, x_l, x_r, Na, N_eigenstates, comment="", save=False):
    """Plot multiple eigenstates for a 1D GPE solution."""

    fpath, model, param = get_model(NpL, layers, 1, [x_l], [x_r], Na, comment)

    xx = torch.from_numpy(np.linspace(x_l, x_r, 1000).reshape(-1, 1)).float()

    N_plt_x = int(np.ceil(np.sqrt(N_eigenstates)))
    N_plt_y = int(np.ceil(N_eigenstates / N_plt_x))
    fig, axes = plt.subplots(N_plt_y, N_plt_x, figsize=(14, 6), sharey=True)

    # Ensure axes is always 2D
    axes = np.array(axes).reshape(N_plt_y, N_plt_x)

    mu_list = param["mu"] if isinstance(param["mu"], list) else [param["mu"]]

    # Sort eigenstates by their chemical potential
    eigenstate_indices = list(range(N_eigenstates))
    index_sorted = sorted(eigenstate_indices, key=lambda i: mu_list[i])
    mu_list_sorted = [mu_list[i] for i in index_sorted]

    for plot_idx, n in enumerate(index_sorted):
        N_col = torch.full((len(xx),), n, dtype=torch.long)
        with torch.no_grad():
            h = model(xx, N_col).numpy().squeeze()
        argmax_idx = np.argmax(np.abs(h))
        if h[argmax_idx] < 0:
            h = -h

        ax = axes[plot_idx // N_plt_x, plot_idx % N_plt_x]
        ax.plot(xx.numpy(), h, label=r'$\psi(x)$', linewidth=2)

        if "x" in param and "V_ext" in param:
            x_pts = np.array(param["x"])
            V_pts = np.array(param["V_ext"])

            # Take the block of collocation points belonging to eigenstate n
            block_size = len(x_pts) // N_eigenstates
            x_block = x_pts[n * block_size:(n + 1) * block_size]
            V_block = V_pts[n * block_size:(n + 1) * block_size]
            ax.plot(x_block, V_block, '.', markersize=2, label='$V_{ext}$')

        mu_n = mu_list_sorted[plot_idx] if plot_idx < len(mu_list_sorted) else float('nan')

        ax.set_title(fr'$\mu_{{n={n}}}={mu_n:.3f}$')
        ax.set_xlabel('x')
        ax.set_xlim(x_l, x_r)
        ax.set_ylim(-1, 1)
        ax.grid()
        ax.legend(loc='lower right')

        if plot_idx == 0:
            ax.set_ylabel(r'$\psi(x)$')

    fig.suptitle(fr'N{NpL}L{layers}, Loss={param["loss"]:.2e}, g = {Na}',
                 fontsize=BIGGER_SIZE)
    plt.tight_layout()

    if save:
        plt.savefig(f"{fpath}.png", dpi=300)
        plt.savefig(f"../report/Figures/ex3.5_N{NpL}L{layers}_{fpath.split('/')[-1]}.eps")
    else:
        plt.show()


from scipy.interpolate import griddata
from scipy.ndimage import minimum_filter

def plot_multiGPE_2D(NpL, layers, x_l, x_r, Na, N_eigenstates,
                     comment="", positive=True, save=False):
    """Plot multiple 2D GPE eigenstates together with V_ext and its minima."""

    fpath, model, param = get_model(
        NpL, layers, 2, x_l, x_r, Na, comment, positive
    )

    x_pts = np.array(param["x"])
    V_pts = np.array(param["V_ext"]).squeeze()

    mu_list = param["mu"] if isinstance(param["mu"], list) else [param["mu"]]

    # Sort eigenstates by chemical potential (like 1D version)
    eigenstate_indices = list(range(N_eigenstates))
    index_sorted = sorted(eigenstate_indices, key=lambda i: mu_list[i])
    mu_sorted = [mu_list[i] for i in index_sorted]

    n_grid = 200
    xs = np.linspace(x_l[0], x_r[0], n_grid)
    ys = np.linspace(x_l[1], x_r[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)

    grid = torch.from_numpy(
        np.stack([XX.ravel(), YY.ravel()], axis=1)
    ).float()

    block_size = len(x_pts) // N_eigenstates
    x_block = x_pts[:block_size]
    V_block = V_pts[:block_size]

    # ── Local minima of V_ext ───────────────────────────────────────────────
    V_grid = griddata(x_block[:, :2], V_block, (XX, YY), method="linear")
    V_grid = np.nan_to_num(V_grid, nan=np.nanmax(V_grid))

    neighborhood = 20
    local_min = (V_grid == minimum_filter(V_grid, size=neighborhood))
    local_min &= (V_grid < np.nanpercentile(V_grid, 20))

    min_coords = np.argwhere(local_min)
    min_px = [xs[ix] for _, ix in min_coords]
    min_py = [ys[iy] for iy, _ in min_coords]
    # Extract specific values for each local minimum
    min_vals = [V_grid[iy, ix] for iy, ix in min_coords]

    N_total = N_eigenstates + 1  # +1 for V_ext
    N_plt_x = int(np.ceil(np.sqrt(N_total)))
    N_plt_y = int(np.ceil(N_total / N_plt_x))

    fig, axes = plt.subplots(
        N_plt_y, N_plt_x,
        figsize=(5 * N_plt_x, 4 * N_plt_y)
    )

    axes = np.array(axes).reshape(N_plt_y, N_plt_x)
    axes_flat = axes.flatten()

    ax_v = axes_flat[0]

    # ── V_ext scatter plot ──────────────────────────────────────────────────
    norm = mcolors.SymLogNorm(
        linthresh=0.5,
        linscale=1.0,
        vmin=V_block.min(),
        vmax=V_block.max()
    )

    sc = ax_v.scatter(
        x_block[:, 0], x_block[:, 1],
        c=V_block, cmap="viridis", s=2, norm=norm
    )

    # Replaced default colorbar with formatted ticks
    cbar = fig.colorbar(sc, ax=ax_v)
    n_ticks = 8
    ticks = norm.inverse(np.linspace(0, 1, n_ticks))
    cbar.set_ticks(ticks) # pyright: ignore
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Plot each individual minimum and add it to the legend
    for px, py, val in zip(min_px, min_py, min_vals):
        ax_v.scatter(px, py, s=120, marker='v', edgecolors='white',
                     linewidths=2, label=f"{val:.2f}")

    ax_v.set_title(r'$V_{\rm ext}(x,y)$')
    ax_v.set_xlabel('x')
    ax_v.set_ylabel('y')
    ax_v.legend(loc="upper right", title='Minima')
    
    # Match the other subplots
    ax_v.set_aspect('equal')
    ax_v.set_xlim(x_l[0], x_r[0])
    ax_v.set_ylim(x_l[1], x_r[1])

    # ── h plot ──────────────────────────────────────────────────────────────
    for plot_idx, n in enumerate(index_sorted):

        ax = axes_flat[plot_idx + 1]

        N_col = torch.full((len(grid),), n, dtype=torch.long)

        with torch.no_grad():
            h = model(grid, N_col).numpy().reshape(n_grid, n_grid)

        vmax_val = np.max(np.abs(h))

        im = ax.imshow(
            h,
            extent=[x_l[0], x_r[0], x_l[1], x_r[1]],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=-vmax_val,
            vmax=vmax_val
        )

        fig.colorbar(im, ax=ax)

        # Overlay V_ext minima on eigenstates
        if len(min_px) > 0:
            ax.scatter(min_px, min_py,
                       marker='v', s=120,
                       edgecolors='white', linewidths=2)

        mu_n = mu_sorted[plot_idx]
        ax.set_title(fr'$n={n}$, $\mu={mu_n:.3f}$')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    for i in range(N_total, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(fr'N{NpL}L{layers}, Loss={param["loss"]:.2e}, g = {Na}',
                 fontsize=BIGGER_SIZE)

    plt.tight_layout()

    if save:
        plt.savefig(f"{fpath}_multi2D.png", dpi=300)
        plt.savefig(f"../report/Figures/ex3.5_N{NpL}L{layers}_{fpath.split('/')[-1]}.eps")
    else:
        plt.show()


def compare_ground_state(N0, L0, x_l0, x_r0, g0, comment0,
                         N, L, x_l, x_r, g, comment, save=False):
    """Compare the ground state wave functions of two different GPE solutions."""

    # ex3 model
    fpath0 = f"../ex3_GPE_double_well/models/1D_N{N0}_L{L0}_pos/PINN_Na{g0:.1f}_x{x_l0:.2f}-{x_r0:.2f}{comment0}.pth"
    layers = [nn.Linear(1, N0), nn.Tanh(),]
    for _ in range(L0 - 1):
        layers += [nn.Linear(N0, N0), nn.Tanh()]
    layers.append(nn.Linear(N0, 1))
    layers.append(nn.Softplus())
    model0 = nn.Sequential(*layers)
    model0.load_state_dict(torch.load(fpath0, map_location="cpu"))
    xx0 = torch.from_numpy(np.linspace(x_l0, x_r0, 1000).reshape(-1, 1)).float()

    # ex3.5 model with multiple eigenstates
    _, model, param = get_model(N, L, 1, [x_l], [x_r], g, comment)
    mu_list0 = param["mu"]
    mu_min_idx = np.argmin(mu_list0)
    mu_min = mu_list0[mu_min_idx]

    xx0_t = torch.tensor(xx0, dtype=torch.float32)
    n_GS = torch.full((len(xx0),), float(mu_min_idx), dtype=torch.long)
    with torch.no_grad():
        h0 = model0(xx0_t).numpy().squeeze()
        h = model(xx0_t, n_GS).numpy().squeeze()
    max_idx = np.argmax(np.abs(h))
    if h[max_idx] < 0:
        h = -h
    
    L2_loss = np.sqrt(np.mean((h - h0) ** 2))

    plt.figure()
    plt.plot(xx0, h0, linewidth=2, label=f'GS from high g')
    plt.plot(xx0, h, '--', linewidth=2, label=fr'GS eigen, $\mu = {{{mu_min:.4f}}}$')
    plt.title('Comparison of Ground State Wave Functions' +'\n' + fr'L2 Loss = {L2_loss:.2e}')
    plt.xlabel('x')
    plt.ylabel(r'$\psi(x)$')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"../report/Figures/ex3.5_1D{comment0}_comparison.eps")
    else:
        plt.show()



save = False

""" These go in the report """
#plot_multiGPE_1D(NpL=16, layers=3, x_l=-8,   x_r=8,   Na=0, N_eigenstates=6, comment="_harmonic_eigen", save=save)
#plot_multiGPE_1D(NpL=16, layers=3, x_l=-8,   x_r=8,   Na=1, N_eigenstates=6, comment="_harmonic_eigen", save=save)
#plot_multiGPE_1D(NpL=16, layers=3, x_l=-3.5, x_r=3.5, Na=1, N_eigenstates=6, comment="_dw2_eigen",      save=save)
#plot_multiGPE_1D(NpL=16, layers=3, x_l=-3.5, x_r=3.5, Na=1, N_eigenstates=6, comment="_dw1_as6_eigen",  save=save)
#
#compare_ground_state(N0=16, L0=2, x_l0=-3,  x_r0=3,  g0=1, comment0="_dw2_high_g",
#                     N=16,  L=3,  x_l=-3.5, x_r=3.5,  g=1,  comment="_dw2_eigen", save=save)
#compare_ground_state(N0=16, L0=2, x_l0=-3,  x_r0=3,  g0=1, comment0="_dw1_as6_high_g",
#                     N=16,  L=3,  x_l=-3.5, x_r=3.5,  g=1,  comment="_dw1_as6_eigen", save=save)


#plot_multiGPE_2D(NpL=32, layers=4, x_l=[-5, -5], x_r=[5, 5], Na=0, N_eigenstates=5, comment="_harmonic_eigen", save=save)
#plot_multiGPE_2D(NpL=32, layers=4, x_l=[-5, -5], x_r=[5, 5], Na=1, N_eigenstates=5, comment="_harmonic_eigen", save=save)
#plot_multiGPE_2D(NpL=32, layers=4, x_l=[-3, -3], x_r=[3, 3], Na=1, N_eigenstates=5, comment="_dw2_as6_eigen", save=save)
plot_multiGPE_2D(NpL=32, layers=4, x_l=[-3, -3], x_r=[3, 3], Na=0, N_eigenstates=5, comment="_dw2_as6_eigen", save=save)
#plot_multiGPE_2D(NpL=64, layers=5, x_l=[-3, -3], x_r=[3, 3], Na=1, N_eigenstates=5, comment="_dw2_as6_eigen", save=save)


""" Attempts with a bigger NN, failed to provide better results """
#plot_multiGPE_1D(NpL=32, layers=4, x_l=-3.5, x_r=3.5, Na=1, N_eigenstates=6, comment="_dw2_eigen", save=save)
#plot_multiGPE_1D(NpL=32, layers=4, x_l=-3.5, x_r=3.5, Na=1, N_eigenstates=6, comment="_dw1_as6_eigen", save=save)
#compare_ground_state(N0=16, L0=2, x_l0=-3,  x_r0=3,  g0=1, comment0="_dw2_high_g",
#                     N=32,  L=4,  x_l=-3.5, x_r=3.5,  g=1,  comment="_dw2_eigen", save=save)
#compare_ground_state(N0=16, L0=2, x_l0=-3,  x_r0=3,  g0=1, comment0="_dw1_as6_high_g",
#                     N=32,  L=4,  x_l=-3.5, x_r=3.5,  g=1,  comment="_dw1_as6_eigen", save=save)


""" Random """
#plot_multiGPE_2D(NpL=16, layers=4, x_l=[-4, -4], x_r=[4, 4], Na=-0.2, N_eigenstates=5, comment="_ciambella", save=save)
#plot_multiGPE_2D(NpL=16, layers=4, x_l=[-4, -4], x_r=[4, 4], Na=50, N_eigenstates=5, comment="_ciambella", save=save)
