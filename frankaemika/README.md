# Franka Panda: joint-space RRT* (vanilla and CDF-guided)

This folder contains the original CDF / PyBullet demos plus a **self-contained 7-DoF motion-planning stack**: RRT* and CDF-guided RRT* in configuration space, **collision checking without PyBullet** (optional **Pinocchio + hpp-fcl**), and an optional **PyBullet playback** that only mirrors the planned path and obstacles.

Planning logic is aligned with the 2D reference in `../2Dexamples/` (`rrt_star_2d.py`, `cdf_guided_rrtstar.py`).

---

## New and planning-related files

| File | Purpose |
|------|---------|
| `plan_and_demo_franka.py` | **Entry point**: builds scene + collision checker, runs vanilla and/or CDF-guided RRT*, prints stats, optional PyBullet demo (feasible path or best-effort tree path). |
| `rrt_star_franka.py` | **Vanilla RRT*** and shared `RRTStarFrankaBase` loop (same statistics shape as the 2D eval script). Helpers: `path_to_tree_node_nearest_goal`, `print_stats_franka`, `path_waypoint_cost`. |
| `cdf_guided_rrtstar_franka.py` | **CDF-guided RRT***: loads `model_dict.pt` MLP, oracle points on obstacle surfaces, soft-min CDF queries, projection sampling and safety-shell edge projection (mirrors 2D `CDF_RRTStar`). |
| `franka_kinematics.py` | **URDF-based FK** for the 7 arm joints (and flange offset): parses `panda_urdf/panda.urdf`, exposes `fk_link_origins`, `fk_flange_position`, default joint limits. |
| `workspace_obstacles.py` | **Task-space obstacles**: `SphereObstacle`, `BoxObstacle`, analytic SDFs (NumPy / Torch), `build_demo_obstacles(scene)` for named scenes. |
| `sphere_arm_collision.py` | **Default collision checker**: conservative **sphere soup** on the arm vs obstacle union SDFs; implements `workspace_margin` for CDF gating. |
| `pin_fcl_collision.py` | **Optional checker**: **Pinocchio** kinematics + **hpp-fcl** distances from **URDF collision meshes** to the same obstacles. See `requirements-franka-pinocchio.txt` for install and ROS vs conda notes. |
| `requirements-franka-pinocchio.txt` | Optional dependency hints for the `pin` collision backend. |

**Not new** but used by the planners: `mlp.py` (MLP for CDF), `panda_urdf/` (meshes + URDF), `pybullet_panda_sim.py` (demo robot only).

---

## Dependencies

- **Always:** `numpy`, `torch`, `pytorch-minimize` (project `requirements.txt` style).
- **Vanilla RRT* + sphere collision:** no extra ML artifacts.
- **CDF-guided mode:** `frankaemika/model_dict.pt` (same checkpoint convention as `mp_ik.py`; default load uses iteration **49900** inside `cdf_guided_rrtstar_franka.py`).
- **PyBullet demo (`--demo`):** `pybullet`, `pybullet_data`, local `pybullet_panda_sim.py`.

### Pinocchio (`--collision-backend pin`)

Use a **conda** environment for Pinocchio + hpp-fcl; pip-only setups often conflict with NumPy ABI or ROS-shipped packages.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda if needed, then create and activate an env (Python **3.10** is a good choice):

   ```bash
   conda create -n venv python=3.10
   conda activate venv
   ```

2. From the **repository root** (where `requirements.txt` lives), install the project’s Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install Pinocchio (and its hpp-fcl stack) from conda-forge **into the same env**:

   ```bash
   conda install pinocchio -c conda-forge
   ```

4. **Do not source ROS** in the same shell before running the planner (e.g. avoid `source /opt/ros/jazzy/setup.bash` or `source /opt/ros/humble/setup.bash` and comment out these lines if they're in .bashrc). ROS prepends its `site-packages` and you may load an older Pinocchio built against NumPy 1, which breaks under NumPy 2 or shadows conda’s build. Use a **clean terminal**: `conda activate venv` only, then run `plan_and_demo_franka.py`. If you already sourced ROS, open a new shell or run `unset PYTHONPATH` after activating conda.

More detail and troubleshooting: `requirements-franka-pinocchio.txt`.

---

## Quick start

From this directory:

```bash
# Vanilla RRT*, sphere collision, print stats only
python plan_and_demo_franka.py --mode vanilla --collision-backend sphere

# Same + progress every 100 iters + PyBullet playback if a path is found
python plan_and_demo_franka.py --mode vanilla --log-every 100 --demo

# Compare vanilla vs CDF (needs model_dict.pt)
python plan_and_demo_franka.py --mode both --scene demo_table --max-iters 5000

# Accurate collision (optional install); do not mix broken ROS pinocchio on PYTHONPATH
python plan_and_demo_franka.py --collision-backend pin --mode vanilla --scene sparse
```

---

## `plan_and_demo_franka.py` command-line reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `both` | `vanilla`: uniform sampling RRT* only. `cdf`: CDF-guided RRT* only (requires `model_dict.pt`). `both`: run vanilla then CDF with the same start/goal and seeds. |
| `--scene` | `demo_table` | Obstacle layout: `demo_table` (plate + two spheres), `sparse` (one sphere), `pillar_and_box` (box + sphere). Defined in `workspace_obstacles.build_demo_obstacles`. |
| `--seed` | `1` | RNG seed for NumPy, Torch, and planner. |
| `--max-iters` | `4000` | Maximum RRT* iterations per planner run. |
| `--step-size` | `0.12` | Maximum joint-space extension length along one tree edge (radians L2 per expand). |
| `--goal-threshold` | `0.35` | Joint-space L2 distance below which the tree may connect to the goal configuration. |
| `--goal-bias` | `0.06` | Probability (vanilla) of sampling the goal instead of uniform `q`. CDF branch uses `0.10` internally for its planner instance. |
| `--neighbor-radius` | `0.65` | RRT* rewire neighborhood radius in joint space. |
| `--edge-resolution` | `0.06` | Collision checking resolution along straight-line segments in `q` (smaller = more samples, slower). |
| `--demo` | off | After planning, open **PyBullet** and replay a dense interpolation of the **chosen** joint path. Obstacles are recreated for **visualization only** (same geometry as the analytic scene). A **green sphere** marks the goal flange pose from FK. |
| `--demo-best-effort` | off | With `--demo`: if **no** feasible path exists, animate the tree path from start to the node **closest to the goal** in joint L2 (not a certified solution path). |
| `--device` | auto | Torch device for CDF net, e.g. `cuda:0` or `cpu`. Default: `cuda:0` if CUDA is available else `cpu`. |
| `--auto-start-goal` | off | If set, or if built-in default start/goal are in collision, sample random collision-free start/goal pairs until one is found. |
| `--collision-backend` | `sphere` | `sphere`: `SphereArmCollisionChecker` (URDF FK + sphere proxies). `pin`: `PinFclCollisionChecker` (URDF collision meshes + hpp-fcl). |
| `--urdf-path` | built-in | Override path to `panda.urdf` when using `pin` (default: `panda_urdf/panda.urdf` next to this README). |
| `--log-every` | `0` | If `N > 0`, print one progress line every `N` iterations (`tree_nodes`, rejects, rewires, goal connected yet, elapsed seconds). `0` disables. |

Run `python plan_and_demo_franka.py --help` for the same list in the shell.

---

## Design notes

- **Separation:** Collision / feasibility uses only the chosen **checker** + `workspace_obstacles`. **PyBullet** is not used for accept/reject during planning.
- **CDF vs collision:** The neural CDF biases samples and can project endpoints; **feasibility** still comes from `is_state_free` / `is_edge_free` on the checker.
- **Which path is demoed:** If `--mode both` and both succeed, the **CDF** path is shown; otherwise the first successful path; with `--demo-best-effort`, a failed run can still visualize the closest tree path.

For the original paper demos (`mp_ik.py`, `throw_wik.py`, `data.pt`, Google Drive assets), see the repository root `readme.md`.
