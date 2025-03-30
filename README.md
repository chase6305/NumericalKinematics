# NumericalKinematics

A robotic kinematics solver library providing efficient implementations based on Pinocchio and PyTorch.

## Features

- Support forward and inverse kinematics solving
- Two solver implementations:
  - Pinocchio Solver: Based on high-performance C++ robotics dynamics library
  - PyTorch Solver: Numerical solver implemented with PyTorch
- Unified solver manager interface
- Support joint limits and elbow pose constraints
- Custom weighted IK solving

## Requirements

```bash
pin=>=2.7.0
torch>=2.0.1
pytorch_kinematics_ms>=0.7.3
```

## Quick Start

1. Create solver manager:

```python
from kinematics import SolverManager

solver_manager = SolverManager()
```

2. Get solver instance:

```python
# Get Pinocchio solver
solver = solver_manager.get_instance("PinocchioSolver")
solver = solver(urdf_path="robot.urdf", end_link_name="ee_link")

# Get PyTorch solver
solver = solver_manager.get_instance("PytorchSolver") 
solver = solver(urdf_path="robot.urdf", end_link_name="ee_link")
```

3. Use solver:

```python
# Forward kinematics
ee_pose = solver.get_fk(qpos)

# Inverse kinematics
success, joint_positions = solver.get_ik(target_pose, joint_seed)
```

## Main APIs

### Base Solver

- `get_fk()`: Calculate forward kinematics
- `get_ik()`: Calculate inverse kinematics
- `set_iteration_params()`: Set iteration parameters
- `set_position_limits()`: Set joint limits
- `set_tcp()`: Set tool center point

### Pinocchio Solver

- Additional dynamics computation features:
  - `compute_generalized_mass_matrix()`: Calculate generalized mass matrix
  - `compute_coriolis_matrix()`: Calculate Coriolis matrix
  - `compute_inverse_dynamics()`: Calculate inverse dynamics
  - `compute_forward_dynamics()`: Calculate forward dynamics

### PyTorch Solver

- PyTorch-based numerical solving implementation
- GPU acceleration support
- Batch kinematics computation support

## Notes

1. Pinocchio solver only supports Linux systems
2. PyTorch solver requires PyTorch version >=2.0.1
3. Joint limits and TCP settings must be completed before computation
