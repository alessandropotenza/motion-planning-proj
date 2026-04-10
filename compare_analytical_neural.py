# Add cdf/2Dexamples to path
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
script_dir = script_dir + "/cdf" + "/2Dexamples"
sys.path.append(script_dir)


import robot_plot2D
import torch
import math
from cdf import CDF2D
from primitives2D_torch import Circle
from nn_cdf import Train_CDF, inference
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mlp import MLPRegression
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_cdf = Train_CDF(device)
# train_cdf.train(input_dim=4, 
#           hidden_dim=[256, 256, 128, 128, 128], 
#           output_dim=1, 
#           activate=torch.nn.ReLU, 
#           batch_size=100,
#           learning_rate=0.01, 
#           weight_decay=1e-5, 
#           save_path='./2Dexamples', 
#           device=device,
#       epochs=1000)



net = torch.load(os.path.join(script_dir, 'model.pth'), weights_only=False)
net.eval()

obstacles = [
    # Circle(center=torch.tensor([-3.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-3.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-3.0, 1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-2.0, -2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-2.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-2.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-2.0, 1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-2.0, 2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, -3.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, -2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, 1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, 2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([-1.0, 3.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, -4.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, -3.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, -2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, 1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, 2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, 3.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([0.0, 4.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, -3.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, -2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, 1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, 2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([1.0, 3.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([2.0, -2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([2.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([2.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([2.0, 1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([2.0, 2.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([3.0, -1.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([3.0, 0.0]), radius=0.01, device=device),
    # Circle(center=torch.tensor([3.0, 1.0]), radius=0.01, device=device),
]
obstacle_points = []
samples = 1
for obs in obstacles:
    points = obs.sample_surface(samples)
    obstacle_points.append(points)
obstacle_points = torch.cat(obstacle_points, dim=0)

# obstacle_points = torch.tensor(
#     [
#         # [0.0,-2.0],
#         # [0.0,3.5],
#         [2.0, 2.0],
#     ],
#     device=device
# )

### Plotting the obstacle points
plt.xlim(-4, 4)
plt.ylim(-4, 4)
### Draw circle of radius 4 at center
circle = plt.Circle((0, 0), 4, color='black', fill=False, linestyle='--')
plt.gca().add_artist(circle)

plt.scatter(obstacle_points[:, 0].cpu().numpy(), obstacle_points[:, 1].cpu().numpy(), s=1)
plt.title('Obstacle Surface Points')
plt.xlabel('x')
plt.ylabel('y')
# plt.axis('equal')
plt.show()

n = train_cdf.cdf.nbData
q = train_cdf.cdf.create_grid_torch(n).to(device)
q.requires_grad = True

q_proj = q.clone()
for i in range(100):
    c_dist, grad = inference(obstacle_points, q_proj, net)
    c_dist = c_dist.reshape(len(obstacle_points), n, n)
    c_dist_min, min_idx = c_dist.min(dim=0)

    grad = grad.reshape(len(obstacle_points), n, n, 2)
    grad = grad.permute(1, 2, 0, 3)  # (n, n, len(obstacle_points), 2)
    min_idx_exp = min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
    grad_min = grad.gather(2, min_idx_exp).squeeze(2)

    q_proj = train_cdf.cdf.projection(
        q_proj,
        c_dist_min.reshape(-1),
        grad_min.reshape(-1, 2),
    )
    q_proj = ((q_proj+np.pi) % (2*np.pi)) - np.pi  # Wrap around to keep within [-pi, pi]
    # print(q_proj.shape)
# plot

import matplotlib.pyplot as plt
c_dist_raw, grad = inference(obstacle_points, q, net)
c_dist_reordered = c_dist_raw.reshape(len(obstacle_points), n, n)
c_dist = c_dist_reordered.min(dim=0)[0]
plt.contourf(q[:,0].detach().cpu().numpy().reshape(n,n), q[:,1].detach().cpu().numpy().reshape(n,n), c_dist.cpu().detach().numpy().reshape(n,n), levels=20)
plt.scatter(q_proj[:,0].detach().cpu().numpy(), q_proj[:,1].detach().cpu().numpy(), c='r', s=1)
plt.title('CDF')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot q_proj in task space with obstacle points - draw full robots
obs_np = obstacle_points.detach().cpu().numpy()
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(obs_np[:,0], obs_np[:,1], s=50, c='blue', label='obstacle points', marker='x')

# Sample a subset of q_proj configurations to avoid overcrowding
subset_indices = torch.randperm(len(q_proj))[:100]  # random 100 configurations
# for i in subset_indices:
#     q_config = q_proj[i].unsqueeze(0).detach().cpu().numpy()
robot_plot2D.plot_2d_manipulators(joint_angles_batch=q_proj[subset_indices].detach().cpu().numpy(), ax=ax, color='red', show_start_end=False, show_eef_traj=False, alpha=0.1)

ax.set_title('Projected configurations in task space')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
ax.legend()
plt.show()






# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cdf = CDF2D(device)

# fig, ax = plt.subplots(figsize=(10, 8))  # Create the third plot

cdf.plot_cdf(ax,obstacles)
q_proj_analytic = q.clone()
for i in range(1000):
    c_dist, grad = cdf.calculate_cdf(q_proj_analytic, obstacles, "online_computation", return_grad=True)
    
    # c_dist shape: (num_queries,), grad shape: (num_queries, 2)
    q_proj_analytic = q_proj_analytic - 0.1 * c_dist.unsqueeze(-1) * grad  # Gradient descent step with learning rate

# Plot q_proj_analytic in task space with obstacle points - draw full robots (analytical CDF)
obs_np = obstacle_points.detach().cpu().numpy()
# fig, ax = plt.subplots(figsize=(8,8))
# ax.scatter(obs_np[:,0], obs_np[:,1], s=50, c='blue', label='obstacle points', marker='x')

# Sample a subset of q_proj_analytic configurations to avoid overcrowding
subset_indices_cdf = torch.randperm(len(q_proj_analytic))[:100]  # random 100 configurations
robot_plot2D.plot_2d_manipulators(joint_angles_batch=q_proj_analytic[subset_indices_cdf].detach().cpu().numpy(), ax=ax, color='green', show_start_end=False, show_eef_traj=False, alpha=0.1)

ax.set_title('Analytical CDF Projected configurations in task space')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
ax.legend()
plt.show()

d_result = cdf.calculate_cdf(cdf.Q_grid.clone().detach().requires_grad_(True), obstacles, "online_computation", return_grad=True)
d = d_result[0].detach().cpu().numpy()
grad = d_result[1].detach().cpu().numpy()

# Create a new figure for C-space CDF plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal', 'box')  # Make sure the pixels are square
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi) 

ax.set_title('Configuration space', size=30)  # Add a title to your plot
ax.set_xlabel('q1', size=20)
ax.set_ylabel('q2', size=20)
axis_limits = (-np.pi, np.pi)  # Set the limits for both axes to be the same
ax.set_xlim(axis_limits)
ax.set_ylim(axis_limits)
ax.tick_params(axis='both', labelsize=20)

# ax.contour(cdf.q0, cdf.q1, d.reshape(cdf.nbData, cdf.nbData), levels=[0], linewidths=6, colors='black', alpha=1.0)
ct = ax.contourf(cdf.q0, cdf.q1, d.reshape(cdf.nbData, cdf.nbData), levels=8, linewidths=1, cmap='coolwarm')
# ax.clabel(ct, inline=False, fontsize=15, colors='black', fmt='%.1f')
ax.scatter(q_proj_analytic[:, 0].detach().cpu().numpy(), q_proj_analytic[:, 1].detach().cpu().numpy(), c='green', s=5, alpha=0.6, label='CDF projections')
ax.legend(fontsize=12)

plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
### Set x and y limits to be +- pi, and make sure the aspect ratio is equal
ax.set_aspect('equal', 'box')  # Make sure the pixels are square
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi) 
title = ""
for i, obs in enumerate(obstacles):
    title += f"C{i}: {obs.center.cpu().numpy()}, R{i}: {obs.radius}, "

ax.legend(fontsize=12)
ax.set_title(title, size=10) # Add a title to your plot
ax.scatter(q_proj[:, 0].detach().cpu().numpy(), q_proj[:, 1].detach().cpu().numpy(), c='red', s=5, alpha=0.6, label='NN CDF projections')
ax.scatter(q_proj_analytic[:, 0].detach().cpu().numpy(), q_proj_analytic[:, 1].detach().cpu().numpy(), c='green', s=5, alpha=0.6, label='CDF projections')
ax.legend(fontsize=12)
### Save figure to dir figs, and ensure figure has unique
figs_dir = "./figs"
os.makedirs(figs_dir, exist_ok=True)
num = os.listdir(figs_dir)
fig_path = os.path.join(figs_dir, f'projection_plot_{len(num)}.png')
fig.savefig(fig_path)
plt.show()


