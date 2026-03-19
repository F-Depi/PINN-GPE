import torch
import scipy.io
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


def plot_GPE_time(NpL, layers, epochs, t_l, t_r, x_l, x_r, comment, save=False):
    model = nn.Sequential(
        nn.Linear(2, NpL),
        nn.Tanh(),
        *[
            layer
            for _ in range(layers - 1)
            for layer in (nn.Linear(NpL, NpL), nn.Tanh())
        ],
        nn.Linear(NpL, 2)
    )

    dir = f"models/N{NpL}_L{layers}"
    if epochs == -1:
        name = f"PINN_checkpoint_t{t_l:.2f}-{t_r:.2f}_x{x_l:.2f}-{x_r:.2f}{comment}"
    else:
        name = f"PINN_E{epochs}_t{t_l:.2f}-{t_r:.2f}_x{x_l:.2f}-{x_r:.2f}{comment}"
    model.load_state_dict(torch.load(f"{dir}/{name}.pth"))
    model.eval()  # important for inference

    param = json.load(open(f"{dir}/{name}_param.json"))


    
    ## Exact data from https://github.com/maziarraissi/PINNs/blob/master/main/Data/NLS.mat
    data = scipy.io.loadmat('data/NLS.mat')
    tt = data['tt'].flatten()
    xx = data['x'].flatten()

    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2).T

    # Correct meshgrid
    T, X = np.meshgrid(tt, xx, indexing='ij')

    grid = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))
    grid = torch.tensor(grid, dtype=torch.float32)

    h = model(grid).detach()
    h2 = (h[:, 0:1].pow(2) + h[:, 1:2].pow(2)).reshape(T.shape).numpy()
    h_mod = np.sqrt(h2)

    L2_error = np.linalg.norm(h_mod - Exact_h) / np.linalg.norm(Exact_h)
    print(f"L2 error: {L2_error:.2e}")

    xx_flat = xx.flatten()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(xx_flat, h_mod[100, :], linewidth=4, label=f't={tt[100]:.2f}')
    ax[0].plot(xx_flat, Exact_h[100, :], '--', linewidth=4, label=f'Exact t={tt[100]:.2f}')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlabel('x')
    ax[0].set_ylabel(r'$|h(t, x)|$')
    ax[0].set_title(f"t={tt[100]:.2f}, L2 error={np.linalg.norm(h_mod[100, :] - Exact_h[100, :]) / np.linalg.norm(Exact_h[100, :]):.2e}")

    ax[1].plot(xx_flat, h_mod[75, :], linewidth=4, label=f't={tt[75]:.2f}')
    ax[1].plot(xx_flat, Exact_h[75, :], '--', linewidth=4, label=f'Exact t={tt[75]:.2f}')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlabel('x')
    ax[1].set_ylabel(r'$|h(t, x)|$')
    ax[1].set_title(f"t={tt[75]:.2f}, L2 error={np.linalg.norm(h_mod[75, :] - Exact_h[75, :]) / np.linalg.norm(Exact_h[75, :]):.2e}")

    #plt.plot(x, h_mod[2, :], linewidth=4, label=f't={t[2]:.2f}')
    #plt.plot(x, Exact_h[:, 125], '--', linewidth=4, label=f'Exact t={t[0]:.2f}')
    fig.suptitle(fr'N={NpL}, L={layers}, E={param["epoch"]}, Loss={param["loss"]:.2e}, L2 error={L2_error:.2e}', fontsize=BIGGER_SIZE)
    if save:
        plt.savefig(f"{dir}/{name}_slice.png", dpi=300)
        plt.savefig(f"../report/Figures/ex2_N{NpL}L{layers}_{name}_slice.eps")
    else:
        plt.show()


    h_mod = h_mod[4:, :]  # Exclude the first 4 time points used for line plots

    plt.figure(figsize=(10, 6))
    plt.imshow(h_mod.T, extent=(t_l, t_r, x_l, x_r), aspect='auto', origin='lower')
    plt.colorbar(label=r'$|h(t, x)|$')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(fr'N={NpL}, L={layers}, E={param["epoch"]}, Loss={param["loss"]:.2e}, L2 error={L2_error:.2e}')
    if save:
        plt.savefig(f"{dir}/{name}_heatmap.png", dpi=300)
        plt.savefig(f"../report/Figures/ex2_N{NpL}L{layers}_{name}_heatmap.eps")
    else:
        plt.show()


save = False

#plot_GPE_time(NpL=8, layers=3, epochs=66000, t_l=0, t_r=1.57, x_l=-5, x_r=5, comment="", save=save)
#plot_GPE_time(NpL=8, layers=3, epochs=1400, t_l=0, t_r=1.57, x_l=-5, x_r=5, comment="_lbfgs", save=save)
#plot_GPE_time(NpL=16, layers=3, epochs=1000, t_l=0, t_r=1.57, x_l=-5, x_r=5, comment="_lbfgs", save=save)
plot_GPE_time(NpL=32, layers=3, epochs=1900, t_l=0, t_r=1.57, x_l=-5, x_r=5, comment="_lbfgs", save=save)
