# 7DOF Panda Robot - Neural Network vs Analytical CDF Comparison
# Add frankaemika path
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
frankaemika_dir = os.path.join(script_dir, "cdf", "frankaemika")
rdf_dir = os.path.join(script_dir, "cdf")

# Clear any existing paths and add frankaemika first
sys.path = [frankaemika_dir, rdf_dir] + [p for p in sys.path if frankaemika_dir not in p and rdf_dir not in p]

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import time
import copy

# Monkey-patch torch.load to handle weights_only issues with legacy models
_original_torch_load = torch.load
def patched_torch_load(f, *args, **kwargs):
    try:
        return _original_torch_load(f, *args, **kwargs)
    except Exception as e:
        if "weights_only" in str(e):
            return _original_torch_load(f, *args, weights_only=False, **{k: v for k, v in kwargs.items() if k != 'weights_only'})
        raise

torch.load = patched_torch_load

# Now import from frankaemika (must add RDF path first)
sys.path.insert(0, os.path.join(rdf_dir, 'RDF'))

from mlp import MLPRegression
from nn_cdf import CDF
from data_generator import DataGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================================================================================
# INITIALIZE MODELS AND ROBOT
# ================================================================================

# Initialize data generator (which includes robot and BP-SDF collision checker)
data_gen = DataGenerator(device)
print("✓ Data generator initialized")

# Load pre-trained neural network model
model = MLPRegression(
    input_dims=10,  # 7 DOF q + 3 position x
    output_dims=1,  # distance
    mlp_layers=[1024, 512, 256, 128, 128],
    skips=[],
    act_fn=torch.nn.ReLU,
    nerf=True
)
model_path = os.path.join(frankaemika_dir, 'model_dict.pt')
model.load_state_dict(torch.load(model_path)[49900])
model.to(device)
model.eval()
print("✓ Neural network model loaded")

# ================================================================================
# CREATE OBSTACLE POINTS IN TASK SPACE
# ================================================================================

# Grid of obstacle points in task space (3D)
n_obstacles = 5
obstacle_points_list = []

# Create a grid of obstacles in reachable space
grid_x = np.linspace(-0.3, 0.3, n_obstacles)
grid_y = np.linspace(-0.3, 0.3, n_obstacles)
grid_z = np.linspace(0.2, 0.7, 2)

for x in grid_x:
    for y in grid_y[:2]:  # Limit to 10 obstacles for faster computation
        for z in grid_z:
            obstacle_points_list.append(np.array([x, y, z]))

obstacle_points_list = obstacle_points_list[:10]  # Limit to 10 obstacles

print(f"✓ Created {len(obstacle_points_list)} obstacle points")

# ================================================================================
# CONFIGURATION SPACE SAMPLING
# ================================================================================

# Sample configuration space grid for visualization (reduced size for 7DOF)
n_samples_per_joint = 8
q_dims = []
for i in range(7):
    q_dims.append(np.linspace(data_gen.panda.theta_min[i].item(), data_gen.panda.theta_max[i].item(), n_samples_per_joint))

# Create a random sampling instead of grid for 7DOF (grid would be 8^7 too large)
n_config_samples = 400  # Reduced from 2500 for memory constraints
q_samples = torch.zeros(n_config_samples, 7, device=device)
for i in range(7):
    q_samples[:, i] = torch.rand(n_config_samples, device=device) * (data_gen.panda.theta_max[i] - data_gen.panda.theta_min[i]) + data_gen.panda.theta_min[i]

print(f"✓ Sampled {n_config_samples} configurations")

# ================================================================================
# PROCESS EACH OBSTACLE
# ================================================================================

os.makedirs('figs/7dof_comparison', exist_ok=True)

for obs_idx, obstacle_point in enumerate(obstacle_points_list):
    print(f"\n{'='*80}")
    print(f"Processing obstacle {obs_idx + 1}/{len(obstacle_points_list)}: Position={obstacle_point}")
    print(f"{'='*80}")
    
    obstacle_torch = torch.tensor(obstacle_point, device=device, dtype=torch.float32).unsqueeze(0)
    
    # ================================================================================
    # NEURAL NETWORK INFERENCE
    # ================================================================================
    
    # Compute NN distance for all sampled configurations
    q_samples_copy = q_samples.clone().detach().requires_grad_(False)
    
    # Create input: [q (7D), x (3D)]
    nn_inputs = torch.cat([
        q_samples_copy,
        obstacle_torch.expand(n_config_samples, 3)
    ], dim=1)  # Shape: (n_samples, 10)
    
    with torch.no_grad():
        nn_distances = model(nn_inputs).squeeze()  # Shape: (n_samples,)
    
    # Gradient descent projection for NN
    q_proj_nn = q_samples.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([q_proj_nn], lr=0.01)
    
    for step in range(100):
        optimizer.zero_grad()
        nn_inputs_proj = torch.cat([q_proj_nn, obstacle_torch.expand(len(q_proj_nn), 3)], dim=1)
        distances_proj = model(nn_inputs_proj).squeeze()
        loss = torch.sum(torch.abs(distances_proj))
        loss.backward()
        optimizer.step()
        
        # Clamp to valid range
        with torch.no_grad():
            for i in range(7):
                q_proj_nn.data[:, i].clamp_(data_gen.panda.theta_min[i], data_gen.panda.theta_max[i])
    
    q_proj_nn = q_proj_nn.detach()
    # ================================================================================
    # ANALYTICAL INFERENCE (DATA-DRIVEN CDF)
    # ================================================================================
    
    # For analytical case, just compute distances at sample points (no projection needed)
    # since we don't have the full CDF data
    analytic_distances = []
    for q_config in torch.split(q_samples, 20):  # Reduced batch size
        pose = torch.eye(4, device=device).unsqueeze(0).expand(len(q_config), 4, 4).float()
        d_sdf, _ = data_gen.bp_sdf.get_whole_body_sdf_batch(
            obstacle_torch.expand(len(q_config), 3),
            pose,
            q_config,
            data_gen.model,
            use_derivative=False
        )
        d_sdf = d_sdf.min(dim=1)[0]
        analytic_distances.extend(d_sdf.detach().cpu().numpy().tolist())
    
    analytic_distances = np.array(analytic_distances)
    
    # Simple projection for analytical: move configs with negative distance towards zero
    q_proj_analytic = q_samples.clone()
    for step in range(20):
        pose = torch.eye(4, device=device).unsqueeze(0).expand(len(q_proj_analytic), 4, 4).float()
        d_sdf, _ = data_gen.bp_sdf.get_whole_body_sdf_batch(
            obstacle_torch.expand(len(q_proj_analytic), 3),
            pose,
            q_proj_analytic,
            data_gen.model,
            use_derivative=False
        )
        d_sdf = d_sdf.min(dim=1)[0]
        
        # Move configs with negative distance in random direction
        mask = (d_sdf < 0).cpu().numpy()
        if mask.any():
            with torch.no_grad():
                direction = torch.randn_like(q_proj_analytic)
                direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)
                q_proj_analytic[mask] += 0.02 * direction[mask]
                
                # Clamp
                for i in range(7):
                    q_proj_analytic[:, i].clamp_(data_gen.panda.theta_min[i], data_gen.panda.theta_max[i])
    
    # Final array conversion
    analytic_distances = np.array(analytic_distances) if isinstance(analytic_distances, list) else analytic_distances
    nn_distances_clean = nn_distances.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: NN Distance Field (first two DOF)
    ax1 = fig.add_subplot(gs[0, 0])
    q1_samples = q_samples[:, 0].detach().cpu().numpy()
    q2_samples = q_samples[:, 1].detach().cpu().numpy()
    scatter1 = ax1.scatter(q1_samples, q2_samples, c=nn_distances_clean, s=20, cmap='RdYlBu_r', alpha=0.6)
    ax1.scatter(q_proj_nn[:, 0].detach().cpu().numpy(), q_proj_nn[:, 1].detach().cpu().numpy(), 
                c='red', s=50, marker='x', linewidths=2, label='NN projections')
    ax1.set_title('NN Distance Field (q0-q1)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('q0 (rad)', fontsize=10)
    ax1.set_ylabel('q1 (rad)', fontsize=10)
    ax1.legend(fontsize=9)
    plt.colorbar(scatter1, ax=ax1, label='Distance')
    
    # Plot 2: Analytical Distance Field (first two DOF)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(q1_samples, q2_samples, c=analytic_distances, s=20, cmap='RdYlBu_r', alpha=0.6)
    ax2.scatter(q_proj_analytic[:, 0].detach().cpu().numpy(), q_proj_analytic[:, 1].detach().cpu().numpy(),
                c='green', s=50, marker='x', linewidths=2, label='Analytical projections')
    ax2.set_title('Analytical Distance Field (q0-q1)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('q0 (rad)', fontsize=10)
    ax2.set_ylabel('q1 (rad)', fontsize=10)
    ax2.legend(fontsize=9)
    plt.colorbar(scatter2, ax=ax2, label='Distance')
    
    # Plot 3: Comparison of Projections
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(q1_samples, q2_samples, c='gray', s=10, alpha=0.2, label='Config space')
    ax3.scatter(q_proj_nn[:, 0].detach().cpu().numpy(), q_proj_nn[:, 1].detach().cpu().numpy(),
                c='red', s=40, marker='x', linewidths=2, label='NN projections', alpha=0.7)
    ax3.scatter(q_proj_analytic[:, 0].detach().cpu().numpy(), q_proj_analytic[:, 1].detach().cpu().numpy(),
                c='green', s=40, marker='+', linewidths=2, label='Analytical projections', alpha=0.7)
    ax3.set_title('Projection Comparison (q0-q1)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('q0 (rad)', fontsize=10)
    ax3.set_ylabel('q1 (rad)', fontsize=10)
    ax3.legend(fontsize=9)
    
    # Plot 4: Distance Distribution - NN
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(nn_distances_clean, bins=30, color='red', alpha=0.7, edgecolor='black')
    ax4.set_title('NN Distance Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Distance', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero distance')
    ax4.legend(fontsize=9)
    
    # Plot 5: Distance Distribution - Analytical
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(analytic_distances, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax5.set_title('Analytical Distance Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Distance', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero distance')
    ax5.legend(fontsize=9)
    
    # Plot 6: Distance Correlation
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(nn_distances_clean, analytic_distances, alpha=0.5, s=20)
    min_val = min(nn_distances_clean.min(), analytic_distances.min())
    max_val = max(nn_distances_clean.max(), analytic_distances.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='y=x')
    ax6.set_title('Distance Correlation', fontsize=12, fontweight='bold')
    ax6.set_xlabel('NN Distance', fontsize=10)
    ax6.set_ylabel('Analytical Distance', fontsize=10)
    ax6.legend(fontsize=9)
    
    obstacle_str = f"({obstacle_point[0]:.2f}, {obstacle_point[1]:.2f}, {obstacle_point[2]:.2f})"
    plt.suptitle(f'7DOF Panda: Obstacle {obs_idx + 1} at {obstacle_str}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = f'figs/7dof_comparison/obstacle_{obs_idx:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()
    
    # Clear memory
    del q_proj_nn, q_proj_analytic
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'='*80}")
print(f"✓ All {len(obstacle_points_list)} obstacles processed successfully!")
print(f"✓ Figures saved to figs/7dof_comparison/ directory")
print(f"{'='*80}")
