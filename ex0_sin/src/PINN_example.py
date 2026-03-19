import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
we start by trying to solve
u''(x) + u(x) = 0
u(0) = 0
u(2π) = 0

=> u(x) = sin(x)
"""

neurons_per_layer = 100
layers = 4

# ROCm uses the 'cuda' namespace in PyTorch. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

model = nn.Sequential(
    nn.Linear(1, neurons_per_layer),
    nn.Tanh(),      
    nn.Linear(neurons_per_layer, neurons_per_layer),
    nn.Tanh(),      
    nn.Linear(neurons_per_layer, neurons_per_layer),
    nn.Tanh(),      
    nn.Linear(neurons_per_layer, neurons_per_layer),
    nn.Tanh(),      
    nn.Linear(neurons_per_layer, 1)
)
model.to(device)

# Optimizer: adjusts the network weights to minimize the loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Boundary conditions
x_left = 0
y_left = 0
x_right = 7 / 2 * torch.pi
y_right = - 1

# Weight for the boundary condition losses, this is a hyperparameter 
# that can be tuned to balance the importance of the PDE loss and the
# boundary condition losses.
BC_w = 1

# Training loop
epoches = 100000
err = 1e-6
for epoch in range(epoches):
    
    # Point grid to sample the prediction of the NN
    x = torch.linspace(x_left, x_right, 10000).reshape(-1, 1).to(device)
    x.requires_grad_(True)
    
    # Evaluate the network: u(x)
    u = model(x)
    
    # --- Compute derivatives using autodiff ---
    
    # First derivative: du/dx
    du_dx = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Second derivative: d²u/dx²
    d2u_dx2 = torch.autograd.grad(
        du_dx, x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]
    
    # --- PDE loss: u'' + u = 0 ---
    residual = d2u_dx2 + u
    L_pde = residual.pow(2).mean()
    
    # --- Boundary condition losses ---
    L_bc1 = (model(torch.ones(1, 1, device=device) * x_left) - y_left).pow(2).mean()
    L_bc2 = (model(torch.ones(1, 1, device=device) * x_right) - y_right).pow(2).mean()
    
    # --- Total loss ---
    # Here weight are applied to the different losses, this is because the PDE
    # loss comes from multiple points and could be much larger than the boundary
    # losses. Usually is adjusted if we see the NN is behaving weirdly.
    loss = L_pde + BC_w * L_bc1 + BC_w * L_bc2

    if loss < err:
        break
    
    # --- Backpropagation: PyTorch computes all gradients ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")

torch.save(model.state_dict(), f'example_models/PINN_N{neurons_per_layer}_L{layers}_E{epoches}.pth')

