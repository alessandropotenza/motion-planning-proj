import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2Dexamples'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og

# Import robot and primitives
from robot2D_torch import Robot2D
from primitives2D_torch import Circle
import robot_plot2D

from nn_cdf import Train_CDF, inference

script_dir = os.path.dirname(os.path.realpath(__file__))
inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(os.path.join(script_dir, 'model.pth'), weights_only=False)
net = net.to(inference_device)
net.eval()





def is_state_valid(state, robot, obstacle_points):
    """Check if a state (joint configuration) is collision-free"""
    # Extract joint angles from OMPL state
    q1 = state[0]
    q2 = state[1]
    q = torch.tensor([[q1, q2]], dtype=torch.float32)

    # Get end-effector position and joint positions
    eef_pos = robot.forward_kinematics_eef(q)
    joint_positions = robot.forward_kinematics_all_joints(q)

    # Check collision with obstacles
    MIN_DIST = 0.05  # Minimum distance threshold for collision
    robot_link_points = robot.surface_points_sampler(q).reshape(-1, 2)
    for point in robot_link_points:
        dist = torch.min(torch.norm(point.unsqueeze(0) - obstacle_points, dim=1))
        if dist < MIN_DIST:
            return False
    return True

class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, robot, obstacle_points):
        super().__init__(si)
        self.robot = robot
        self.obstacle_points = obstacle_points
    
    def isValid(self, state):
        return is_state_valid(state, self.robot, self.obstacle_points)

def plan_with_rrt_star(robot, obstacle_points, start_config, goal_config):
    """Plan a path using OMPL RRT*"""

    # Create state space
    space = ob.RealVectorStateSpace(2)

    # Set bounds (-pi to pi for both joints)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-np.pi)
    bounds.setHigh(np.pi)
    space.setBounds(bounds)

    # Create space information
    si = ob.SpaceInformation(space)

    # Set state validity checker
    validity_checker = ValidityChecker(si, robot, obstacle_points)
    si.setStateValidityChecker(validity_checker)

    # Set start and goal
    start = space.allocState()
    start[0] = start_config[0]
    start[1] = start_config[1]

    goal = space.allocState()
    goal[0] = goal_config[0]
    goal[1] = goal_config[1]

    # Create problem definition
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # Create planner (RRT*)
    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)

    # Set planner parameters
    planner.setRange(0.1)  # Step size

    # Solve
    solved = planner.solve(10.0)  # 10 second timeout

    if solved:
        # Get the path
        path = pdef.getSolutionPath()
        print(f"Found solution with {path.getStateCount()} states")

        # Convert path to numpy array
        path_states = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            path_states.append([state[0], state[1]])
        path_states = np.array(path_states)

        return path_states
    else:
        print("No solution found")
        return None
    
def custom_planning(robot, obstacle_points, start_config, goal_config):
    """
    Custom planner using neural network-guided sampling instead of uniform random sampling.
    Samples configurations preferentially in collision-free regions using CDF guidance.
    """
    
    planning_device = "cpu"
    max_iterations = 1000
    step_size = 0.1
    
    # Convert start and goal to tensors
    start_q = torch.tensor([start_config], dtype=torch.float32, device=planning_device)
    goal_q = torch.tensor([goal_config], dtype=torch.float32, device=planning_device)
    
    # Initialize tree/graph with start configuration
    nodes = [start_q.clone()]
    edges = {}  # Graph edges for path reconstruction
    
    print(f"Starting custom planning with NN-guided sampling...")
    
    # Main planning loop
    for iteration in range(max_iterations):
        # TODO: Implement custom sampling strategy
        # Option 1: Sample random configs and project them to collision-free regions using CDF gradient
        # Option 2: Sample near existing nodes and project
        # Option 3: Use informed sampling based on distance field
        
        # Sample a random configuration in [-pi, pi]^2
        q_rand = torch.rand(1, 2, device=planning_device) * 2 * np.pi - np.pi
        
        # TODO: Get distance to obstacles using neural network
        # d_rand = get_distance_to_obstacles(robot, q_rand, obstacle_points)
        
        # TODO: If collision, project towards free space using CDF gradient
        # If d_rand < threshold, apply gradient-based projection
        
        # TODO: Find nearest node in tree
        # distances_to_rand = [torch.norm(node - q_rand) for node in nodes]
        # nearest_idx = torch.argmin(torch.stack(distances_to_rand))
        # q_near = nodes[nearest_idx]
        
        # TODO: Steer from q_near towards q_rand
        # q_new = steer(q_near, q_rand, step_size)
        
        # TODO: Check if q_new is collision-free
        # if is_state_valid(q_new, robot, obstacle_points):
        #     nodes.append(q_new)
        #     edges[len(nodes)-1] = nearest_idx
        
        # TODO: Check if q_new is close to goal
        # if torch.norm(q_new - goal_q) < step_size:
        #     nodes.append(goal_q)
        #     edges[len(nodes)-1] = len(nodes)-2
        #     print(f"Goal reached at iteration {iteration}")
        #     break
        
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}: {len(nodes)} nodes in tree")
    
    # TODO: Reconstruct path from start to goal using edges
    # path = reconstruct_path(edges, start_idx=0, goal_idx=len(nodes)-1)
    
    # TODO: Smooth path (optional)
    
    path = None  # Placeholder
    print(f"Custom planning finished with {len(nodes)} nodes")
    
    return path

def visualize_path(robot, obstacle_points, path, start_config, goal_config):
    """Visualize the planned path in both task space and configuration space"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Task space plot (left)
    # Plot obstacles points
    for obs in obstacle_points:
        ax1.plot(obs[0].cpu().numpy(), obs[1].cpu().numpy(), 'kx', markersize=5, label='Obstacle' if 'Obstacle' not in ax1.get_legend_handles_labels()[1] else "")

    # Plot start configuration (blue)
    robot_plot2D.plotArm(
        ax=ax1,
        a=np.array(start_config),
        d=robot.link_length[0].cpu().numpy(),
        p=np.array([0.0, 0.0]),
        sz=0.05,
        facecol='blue',
        edgecol='blue',
        robot_base=True
    )

    # Plot goal configuration (green)
    robot_plot2D.plotArm(
        ax=ax1,
        a=np.array(goal_config),
        d=robot.link_length[0].cpu().numpy(),
        p=np.array([0.0, 0.0]),
        sz=0.05,
        facecol='green',
        edgecol='green',
        robot_base=True
    )

    if path is not None:
        # Plot intermediate configurations along the path
        for i in range(0, len(path), max(1, len(path)//20)):  # Plot every 20th point or so
            robot_plot2D.plotArm(
                ax=ax1,
                a=path[i],
                d=robot.link_length[0].cpu().numpy(),
                p=np.array([0.0, 0.0]),
                sz=0.03,
                facecol='red',
                edgecol='red',
                alpha=0.3,
                robot_base=False
            )

    # Set axis limits for task space
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect('equal')
    ax1.set_title('Task Space (Workspace)')
    ax1.grid(True)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Configuration space plot (right)
    if path is not None:
        # Plot the path trajectory in joint space
        path_array = np.array(path)
        ax2.plot(path_array[:, 0], path_array[:, 1], 'r-', linewidth=2, alpha=0.7, label='Path')
        ax2.plot(path_array[:, 0], path_array[:, 1], 'ro', markersize=3, alpha=0.5)

    # Plot start and goal points
    ax2.plot(start_config[0], start_config[1], 'bo', markersize=10, label='Start')
    ax2.plot(goal_config[0], goal_config[1], 'go', markersize=10, label='Goal')

    # Set joint limits
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_aspect('equal')
    ax2.set_title('Configuration Space (Joint Space)')
    ax2.grid(True)
    ax2.set_xlabel('Joint 1 (radians)')
    ax2.set_ylabel('Joint 2 (radians)')

    # Add joint limit lines
    ax2.axhline(y=-np.pi, color='k', linestyle='--', alpha=0.5, label='Joint limits')
    ax2.axhline(y=np.pi, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=-np.pi, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(x=np.pi, color='k', linestyle='--', alpha=0.5)

    # Add legend to configuration space plot
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ompl_path.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ompl_path.png")
    plt.show()

def get_distance_to_obstacles(robot, q, obstacle_points):
    """Compute the minimum distance from the robot (at configuration q) to the obstacles using neural network"""
    # Move data to inference device (CUDA)
    q_infer = q.to(inference_device).clone().detach().requires_grad_(True)
    obstacle_points_infer = obstacle_points.to(inference_device)
    
    # Compute distance using neural network
    c_dist, grad = inference(obstacle_points_infer, q_infer, net)
    
    # Take minimum distance (across all obstacle points)
    c_dist_min = c_dist.min()
    
    # Move result back to CPU if needed
    return c_dist_min.cpu() if c_dist_min.device.type == 'cuda' else c_dist_min

def main():
    # Set device
    # RRT planning always uses CPU
    planning_device = "cpu"
    
    print(f"Using device for RRT: {planning_device}")
    print(f"Using device for inference: {inference_device}")

    # Create robot on CPU for RRT planning
    robot = Robot2D(
        num_joints=2,
        # init_states=torch.tensor([[0.0, 0.0]]),
        init_states=torch.tensor([[0.0, -np.pi]]),
        link_length=torch.tensor([[2.0, 2.0]]).float(),
        device=planning_device
    )

    # Define obstacles (simple circles)
    obstacles = [
        # Circle(center=torch.tensor([2.5, 2.5]), radius=0.5, device=device),
        # Circle(center=torch.tensor([0.0, 2.5]), radius=0.3, device=device),
        # Circle(center=torch.tensor([-2.5, 0.0]), radius=0.4, device=device)
        Circle(center=torch.tensor([3.0, 3.0]), radius=0.5, device=planning_device),
        Circle(center=torch.tensor([0.0, 3.5]), radius=0.3, device=planning_device),
        Circle(center=torch.tensor([-3.0, -1.5]), radius=0.4, device=planning_device)
    ]
    obstacle_points = []
    for obs in obstacles:
        points = obs.sample_surface(100)
        obstacle_points.append(points)
    obstacle_points = torch.cat(obstacle_points, dim=0)

    d = get_distance_to_obstacles(robot, torch.tensor([[0.0, 0.0]], device=planning_device), obstacle_points)
    print(f"Distance to obstacles at start config: {d.item():.4f}")

    # Define start and goal configurations
    start_config = [0.0, 0.0]  # Robot pointing right
    goal_config = [-5*np.pi/4, -np.pi/4]  # Different configuration

    print("Planning path with OMPL RRT*...")
    path = plan_with_rrt_star(robot, obstacle_points, start_config, goal_config)

    if path is not None:
        print(f"Path found with {len(path)} waypoints")
        print("Visualizing...")

        # Visualize
        visualize_path(robot, obstacle_points, path, start_config, goal_config)
    else:
        print("No path found")

if __name__ == "__main__":
    
    main()