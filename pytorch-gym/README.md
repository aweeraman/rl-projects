# PyTorch Gym

A reinforcement learning project implementing Deep Q-Network (DQN) for the CartPole-v1 environment using PyTorch and Gymnasium.

## Features

- DQN implementation with experience replay and target networks
- Reward shaping for improved training performance
- TensorBoard integration for training metrics visualization
- Model checkpointing and validation
- Configurable hyperparameters
- Reproducible training with seed management

## Requirements

- Python >=3.13
- PyTorch >=2.7.1
- Gymnasium >=1.2.0
- NumPy >=2.3.2
- TensorBoard >=2.20.0
- Pygame >=2.6.1

## Installation

```bash
uv sync
```

## Usage

### Training

Train the DQN model on CartPole-v1:

```bash
python cartpole.py
```

### Testing

Test a trained model with visualization:

```bash
python cartpole.py --test
```

## Configuration

Key hyperparameters can be modified at the top of `cartpole.py`:

- `EPISODES`: Number of training episodes (default: 1000)
- `GAMMA`: Discount factor (default: 0.99)
- `LR`: Learning rate (default: 1e-3)
- `BATCH_SIZE`: Training batch size (default: 128)
- `EPS_START/EPS_END/EPS_DECAY`: Epsilon-greedy exploration parameters

## Reward Shaping

The implementation includes several reward shaping techniques:

- Early termination penalty for episodes ending below threshold
- Bonus rewards for staying alive longer
- Action duplication for high-reward experiences
- Maximum reward capping

## TensorBoard Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir=runs
```

Tracked metrics include:
- Episode rewards
- Training loss
- Q-values
- Epsilon decay
- Step counts

## Model Persistence

The best performing model is automatically saved to `dqn_cartpole.pth` based on average reward over the last 100 episodes.