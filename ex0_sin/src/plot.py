import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SMALL_SIZE = 14
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize


def plot(N, L, E, x=None, comment="", save=False):

    if x == None:
        x = [-2.0, 10.0]
        compute_error = False
    else:
        compute_error = True

    print(compute_error)
    layers = [nn.Linear(1, N), nn.Tanh()]
    for _ in range(L - 1):
        layers += [nn.Linear(N, N), nn.Tanh()]

    layers.append(nn.Linear(N, 1))
    model = nn.Sequential(*layers)

    model.load_state_dict(torch.load(f'models/PINN_N{N}_L{L}_E{E}{comment}.pth', map_location='cpu'))

    x = torch.linspace(x[0], x[1], 1000).reshape(-1, 1)
    with torch.no_grad():
        u = model(x).cpu().numpy()

    u_exact = torch.sin(x).cpu().numpy()

    title = f"PINN Solution for N={N}, L={L}"
    if comment:
        title += f" ({comment.strip('_')})"
    if compute_error:
        L2_error = torch.sqrt(torch.mean((torch.tensor(u) - torch.tensor(u_exact)) ** 2)).item()
        title += f"\nL2 Error: {L2_error:.2e}"

    plt.figure(figsize=(8, 5))
    plt.plot(x.cpu().numpy(), u, label="PINN Solution")
    plt.plot(x.cpu().numpy(), u_exact, label="Exact Solution", linestyle='dashed')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f'figures/PINN_N{N}_L{L}_E{E}.png', dpi=300)
    else:
        plt.show()

    plt.close()

save = False

plot(100, 3, 100000, save=save)
plot(100, 4, 100000, save=save)
plot(16,  2, 5000,   save=save)
plot(16,  3, 10000,  save=save)
plot(32,  2, 5000,   save=save)
plot(4,   1, 10000,  save=save)
plot(4,   1, 100000, save=save)
plot(4,   2, 5000,   save=save)
plot(4,   3, 5000,   save=save)
plot(4,   4, 5000,   save=save)
plot(512, 3, 1000,   save=save)
plot(512, 3, 10000,  save=save)
plot(6,   1, 50000,  save=save)
plot(8,   1, 50000,  save=save)
plot(8,   2, 5000,   save=save)
plot(4,   4, 10000,  comment="_larger",            save=save)
plot(4,   4, 10000,  comment="_larger_morepoints", save=save)
plot(4,   4, 5000,   comment="_larger",            save=save)
plot(16,  3, 10000,  x=[-5, 5], comment="_L1.69e-06_xl-5.0_xr5.0", save=save)
plot(16,  3, 10000,  x=[0, 5],  comment="_L7.26e-01_xl0.0_xr5.0",  save=save)
plot(8,   3, 100000, x=[0, 10], comment="_L1.93e-05_xl0.0_xr5.0",  save=save)
