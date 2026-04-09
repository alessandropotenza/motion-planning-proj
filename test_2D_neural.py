"""
Minimal working example: Query configuration space distance field
"""
import torch
import math
from cdf import CDF2D
from primitives2D_torch import Circle
from nn_cdf import Train_CDF, inference
import os

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



script_dir = os.path.dirname(os.path.realpath(__file__))
net = torch.load(os.path.join(script_dir, 'model.pth'), weights_only=False)
net.eval()

obstacles = [
    Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device),
    Circle(center=torch.tensor([-2.0, -2.0]), radius=0.4, device=device),
]
obstacle_points = []
for obs in obstacles:
    points = obs.sample_surface(100)
    obstacle_points.append(points)
obstacle_points = torch.cat(obstacle_points, dim=0)

# obstacle_points = torch.tensor(
#     [
#         # [0.0,-2.0],
#         [0.0,3.5],
#         # [2.0, 2.0],
#     ],
#     device=device
# )

### Plotting the obstacle points
import matplotlib.pyplot as plt
plt.scatter(obstacle_points[:, 0].cpu().numpy(), obstacle_points[:, 1].cpu().numpy(), s=1)
plt.title('Obstacle Surface Points')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

n = train_cdf.cdf.nbData
q = train_cdf.cdf.create_grid_torch(n).to(device)
q.requires_grad = True

q_proj = q.clone()
for i in range(10):
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
import robot_plot2D
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
