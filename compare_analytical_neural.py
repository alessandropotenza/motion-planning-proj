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

model_name = 'model_ee.pth.10000ee'
mode = "ee"  # "ee" = end effector only | "wb" = whole body
# mode = "wb"

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


# model_name = 'model.pth'
net = torch.load(os.path.join(script_dir, model_name), weights_only=False)
net.eval()

obstacles = [
    Circle(center=torch.tensor([-3.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-3.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-3.0, 1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-2.0, -2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-2.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-2.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-2.0, 1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-2.0, 2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, -3.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, -2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, 1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, 2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([-1.0, 3.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, -4.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, -3.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, -2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, 1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, 2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, 3.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([0.0, 4.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, -3.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, -2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, 1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, 2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([1.0, 3.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([2.0, -2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([2.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([2.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([2.0, 1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([2.0, 2.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([3.0, -1.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([3.0, 0.0]), radius=0.01, device=device),
    Circle(center=torch.tensor([3.0, 1.0]), radius=0.01, device=device),
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

# Skip individual plots - we'll create a combined figure below

n = train_cdf.cdf.nbData
q_base = train_cdf.cdf.create_grid_torch(n).to(device)

# Create output directory for figures
os.makedirs('figs', exist_ok=True)

# ================================================================================
# PROCESS EACH OBSTACLE INDIVIDUALLY
# ================================================================================

for obs_idx, current_obstacle in enumerate(obstacles):
    print(f"\n{'='*80}")
    print(f"Processing obstacle {obs_idx + 1}/{len(obstacles)}: Center={current_obstacle.center.cpu().numpy()}, Radius={current_obstacle.radius}")
    print(f"{'='*80}")
    
    # Get obstacle points for this single obstacle
    current_obstacle_points = current_obstacle.sample_surface(1)
    
    q = q_base.clone().requires_grad_(True)
    
    # Neural network projection
    q_proj = q.clone()
    for i in range(100):
        c_dist, grad = inference(current_obstacle_points, q_proj, net)
        c_dist = c_dist.reshape(len(current_obstacle_points), n, n)
        c_dist_min, min_idx = c_dist.min(dim=0)

        grad = grad.reshape(len(current_obstacle_points), n, n, 2)
        grad = grad.permute(1, 2, 0, 3)
        min_idx_exp = min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        grad_min = grad.gather(2, min_idx_exp).squeeze(2)

        q_proj = train_cdf.cdf.projection(
            q_proj,
            c_dist_min.reshape(-1),
            grad_min.reshape(-1, 2),
        )
        q_proj = ((q_proj+np.pi) % (2*np.pi)) - np.pi

    # Get neural network C-space field
    q_temp = q_base.clone().requires_grad_(True)
    c_dist_raw, grad = inference(current_obstacle_points, q_temp, net)
    c_dist_reordered = c_dist_raw.reshape(len(current_obstacle_points), n, n)
    c_dist_nn = c_dist_reordered.min(dim=0)[0]

    # Analytical CDF projection
    cdf = CDF2D(device)
    current_obstacle_list = [current_obstacle]
    
    # Clear any cached zero-level sets to ensure fresh computation for each obstacle
    if hasattr(cdf, 'q_0_level_set'):
        delattr(cdf, 'q_0_level_set')
    if hasattr(cdf, 'q_0_level_set_eef'):
        delattr(cdf, 'q_0_level_set_eef')
    
    if mode == "ee":
        # For end effector mode, compute distance field based on EE collisions in C-space
        q_proj_analytic = q_base.clone().requires_grad_(True)
        for i in range(1000):
            c_dist, grad = cdf.calculate_cdf_eef(q_proj_analytic, current_obstacle_list, "online_computation", return_grad=True)
            q_proj_analytic = q_proj_analytic - 0.1 * c_dist.unsqueeze(-1) * grad
            q_proj_analytic = q_proj_analytic.detach().requires_grad_(True)

        # Get analytical C-space field for EE
        q_grid_with_grad = cdf.Q_grid.clone().requires_grad_(True)
        d_analytical = cdf.calculate_cdf_eef(q_grid_with_grad, current_obstacle_list, "online_computation", return_grad=False).detach().cpu().numpy()

    else:  # mode == "wb"
        # Original whole body approach
        q_proj_analytic = q_base.clone().requires_grad_(True)
        for i in range(1000):
            c_dist, grad = cdf.calculate_cdf(q_proj_analytic, current_obstacle_list, "online_computation", return_grad=True)
            q_proj_analytic = q_proj_analytic - 0.1 * c_dist.unsqueeze(-1) * grad
            q_proj_analytic = q_proj_analytic.detach().requires_grad_(True)

        # Get analytical C-space field  
        q_grid_with_grad = cdf.Q_grid.clone().requires_grad_(True)
        d_analytical = cdf.calculate_cdf(q_grid_with_grad, current_obstacle_list, "online_computation", return_grad=False).detach().cpu().numpy()

    # ================================================================================
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Obstacle Surface Points
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    circle = plt.Circle((0, 0), 4, color='black', fill=False, linestyle='--', linewidth=2)
    ax1.add_artist(circle)
    ax1.scatter(current_obstacle_points[:, 0].cpu().numpy(), current_obstacle_points[:, 1].cpu().numpy(), s=100, alpha=0.8, color='red', marker='o')
    ax1.set_title('Obstacle Surface Points', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('y', fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Neural Network C-space CDF with projections
    ax2 = fig.add_subplot(gs[0, 1])
    contour2 = ax2.contourf(q[:, 0].detach().cpu().numpy().reshape(n, n), q[:, 1].detach().cpu().numpy().reshape(n, n), c_dist_nn.detach().cpu().numpy().reshape(n, n), levels=15, cmap='RdYlBu_r')
    ax2.scatter(q_proj[:, 0].detach().cpu().numpy(), q_proj[:, 1].detach().cpu().numpy(), c='red', s=2, alpha=0.4, label='NN projections')
    ax2.set_title('Neural Network CDF (C-space)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('q1', fontsize=10)
    ax2.set_ylabel('q2', fontsize=10)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_aspect('equal', 'box')
    ax2.legend(loc='upper right', fontsize=9)
    plt.colorbar(contour2, ax=ax2)

    # Plot 3: Analytical C-space CDF with projections
    ax3 = fig.add_subplot(gs[0, 2])
    contour3 = ax3.contourf(cdf.q0, cdf.q1, d_analytical.reshape(cdf.nbData, cdf.nbData), levels=15, cmap='RdYlBu_r')
    ax3.contour(cdf.q0, cdf.q1, d_analytical.reshape(cdf.nbData, cdf.nbData), levels=[0], linewidths=3, colors='black', alpha=0.8)
    plt.colorbar(contour3, ax=ax3)
    
    ax3.scatter(q_proj_analytic[:, 0].detach().cpu().numpy(), q_proj_analytic[:, 1].detach().cpu().numpy(), c='green', s=2, alpha=0.4, label='Analytical projections')
    ax3.set_title(f'Analytical CDF (C-space) - {mode.upper()}', fontsize=12, fontweight='bold')
    ax3.set_xlabel('q1', fontsize=10)
    ax3.set_ylabel('q2', fontsize=10)
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_aspect('equal', 'box')
    ax3.legend(loc='upper right', fontsize=9)

    # Plot 4: Neural Network Task Space (row 2, cols 0-1)
    ax4 = fig.add_subplot(gs[1, 0:2])
    obs_np = current_obstacle_points.detach().cpu().numpy()
    ax4.scatter(obs_np[:, 0], obs_np[:, 1], s=200, c='red', label='obstacle points', marker='o', linewidths=2, edgecolors='darkred')
    ax4.set_xlim(-4, 4)
    ax4.set_ylim(-4, 4)
    circle4 = plt.Circle((0, 0), 4, color='black', fill=False, linestyle='--', linewidth=2)
    ax4.add_artist(circle4)
    subset_indices = torch.randperm(len(q_proj))[:50]
    robot_plot2D.plot_2d_manipulators(joint_angles_batch=q_proj[subset_indices].detach().cpu().numpy(), ax=ax4, color='red', show_start_end=False, show_eef_traj=False, alpha=0.08)
    ax4.set_title('NN Projected Configs (Task Space)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x', fontsize=10)
    ax4.set_ylabel('y', fontsize=10)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # Plot 5: Analytical Task Space (row 2, col 2)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(obs_np[:, 0], obs_np[:, 1], s=200, c='red', label='obstacle points', marker='o', linewidths=2, edgecolors='darkred')
    ax5.set_xlim(-4, 4)
    ax5.set_ylim(-4, 4)
    circle5 = plt.Circle((0, 0), 4, color='black', fill=False, linestyle='--', linewidth=2)
    ax5.add_artist(circle5)
    subset_indices_cdf = torch.randperm(len(q_proj_analytic))[:50]
    robot_plot2D.plot_2d_manipulators(joint_angles_batch=q_proj_analytic[subset_indices_cdf].detach().cpu().numpy(), ax=ax5, color='green', show_start_end=False, show_eef_traj=False, alpha=0.08)
    ax5.set_title('Analytical Projected Configs (Task Space)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('x', fontsize=10)
    ax5.set_ylabel('y', fontsize=10)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    # Plot 6: Combined comparison in C-space (row 3, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    # contour6 = ax6.contourf(cdf.q0, cdf.q1, d_analytical.reshape(cdf.nbData, cdf.nbData), levels=15, cmap='coolwarm', alpha=0.8)
    # ax6.contour(cdf.q0, cdf.q1, d_analytical.reshape(cdf.nbData, cdf.nbData), levels=[0], linewidths=4, colors='black', alpha=1.0, linestyles='solid')
    ax6.scatter(q_proj[:, 0].detach().cpu().numpy(), q_proj[:, 1].detach().cpu().numpy(), c='red', s=3, alpha=0.3, label='NN Projections')
    ax6.scatter(q_proj_analytic[:, 0].detach().cpu().numpy(), q_proj_analytic[:, 1].detach().cpu().numpy(), c='lime', s=3, alpha=0.3, label='Analytical Projections')
    ax6.set_title('Comparison: Neural Network vs Analytical CDF Projections', fontsize=14, fontweight='bold')
    ax6.set_xlabel('q1', fontsize=12)
    ax6.set_ylabel('q2', fontsize=12)
    ax6.set_xlim(-np.pi, np.pi)
    ax6.set_ylim(-np.pi, np.pi)
    ax6.set_aspect('equal', 'box')
    ax6.legend(loc='upper right', fontsize=11, framealpha=0.95)
    # plt.colorbar(contour6, ax=ax6, label='Distance to obstacles')

    obstacle_center = current_obstacle.center.cpu().numpy()
    mode_label = "End Effector" if mode == "ee" else "Whole Body"
    plt.suptitle(f'Obstacle {obs_idx + 1}: Center=({obstacle_center[0]:.2f}, {obstacle_center[1]:.2f}) - Mode: {mode_label}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save the figure for this obstacle
    from pathlib import Path
    out_dir = Path('figs/comparison')
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f'{obs_idx:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()
    
    # Clear memory
    del q_proj, q_proj_analytic, c_dist_nn, d_analytical, cdf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'='*80}")
print(f"✓ All {len(obstacles)} obstacles processed successfully!")
print(f"✓ Figures saved to figs/ directory")
print(f"{'='*80}")
