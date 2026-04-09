"""
Minimal working example: Query configuration space distance field
"""
import torch
import math
from cdf import CDF2D
from primitives2D_torch import Circle

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize CDF
cdf = CDF2D(device)
print("CDF2D initialized")

# Define obstacles in task space
obstacles = [
    Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device),
    Circle(center=torch.tensor([-2.0, -2.0]), radius=0.4, device=device),
]
print(f"Created {len(obstacles)} obstacles")

# Query configuration space distance
# Query a few configurations
q_queries = torch.tensor([
    [0.0, 0.0],      # origin
    [1.0, 1.0],      # arbitrary config
    [math.pi/2, 0.0], # q1 = 90 degrees
], device=device, requires_grad=False)

print("\nQuerying distances:")
print("-" * 50)

# Get distances using CDF
print("\nUsing CDF (Configuration Distance Field):")
cdf_distances = cdf.calculate_cdf(q_queries, obstacles, method='online_computation')
for i, (q, d) in enumerate(zip(q_queries, cdf_distances)):
    print(f"  Config {i}: {q.cpu().numpy()} -> distance: {d.item():.4f}")

# Advanced: Get distances with gradients
print("\nWith gradient information (SDF):")
q_test = torch.tensor([[0.5, 0.5]], device=device, requires_grad=True)
sdf_dist, sdf_grad = cdf.inference_sdf(q_test, obstacles, return_grad=True)
print(f"  Config: {q_test[0].cpu().detach().numpy()}")
print(f"  Distance: {sdf_dist.item():.4f}")
print(f"  Gradient: {sdf_grad[0].cpu().detach().numpy()}")

print("\n" + "=" * 50)
print("✓ Minimal example completed successfully!")
