# Configuration Guide

This directory contains configuration files for training and testing the Biped Robot environment.

## `test_config.yaml`

This file defines the hyperparameters and environment settings for testing and training.

### Structure

#### `train` Section
General training parameters:
- `seed`: Random seed for reproducibility.
- `steps`: Total training steps.
- `max_episode_steps`: Maximum steps per episode.
- `sim_frequency`: Simulation frequency (Hz).
- `use_history`: Enable observation history.
- `history_size`: Number of past frames to stack.
- `actor_obs` / `critic_obs`: Observation types ("normal", "privileged").

#### `env_config` Section
Specific environment configuration mirroring `BipedEnvCfg`:

- **Objective**: `objective` ("walk", "target", "balance").
- **Mirroring**: `enable_mirroring` and `mirror_joint_indices`.
- **Rewards**: `reward_weights` dictionary defining the scale of each reward component.
- **Commands**: `commands` dictionary defining velocity ranges for `lin_vel_x`, `lin_vel_y`, and `ang_vel_z`.
- **Randomization**:
  - `enable_perturbations`: Enable random pushes.
  - `push_interval_s`: Interval for pushes.
  - `push_vel_range`: Magnitude of pushes.
  - `enable_physics_randomization`: Enable mass/friction randomization.
  - `events`: Detailed configuration for reset and randomization events (`reset_base`, `reset_robot_joints`, `push_robot`, `randomize_mass`, `randomize_friction`).
- **Noise**: `observation_noise_model` and `action_noise_model` (Gaussian mean and std).
- **Termination**: `termination_height` threshold.

### Usage

Modify `test_config.yaml` to experiment with different reward functions, randomization strategies, or command ranges without changing the code.
