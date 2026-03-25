from functions import *


r = 1.0
N_b = 100

x_l = [-r, -r];     x_r = [r, r]
BD_grid, BD_h = make_bc_grid_2d(x_l, x_r, n=N_b)

domain = DomainConfig(
        V_ext = lambda x: 0,
        Na = 0,

        input_dim=2,
        x_l=x_l,
        x_r=x_r,
        N_points=10_000,
        BC_grid=BD_grid,
        BC_h=BD_h,
        BC_w = 1.0,
    )

model_config = ModelConfig(
        neurons_per_layer=16,
        layers=2,
        positive=True,
    )

train_config = TrainConfig(
        # The other defaults are good
        optimizer="lbfgs",
        comment="_weird"
    )

load_model = "models/2D_N16_L2_pos/PINN_Na0.0_x-1.00_-1.00-1.00_1.00.pth"
GPESolver(domain, model_config, train_config, load_model).train()
