import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

# Hyperparameters
SEED = 24
EPISODES = 1000
INTERVAL = 100
GAMMA = 0.99
LR = 1e-3
TRAIN_INTERVAL = 5
BATCH_SIZE = 128
MEM_SIZE = 10000
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

# Reward shaping
EARLY_TERMINATION_PENALTY = -50
ACTION_DUPLICATION_FACTOR = 5
MIN_REWARD_THRESHOLD = 300
BONUS_FOR_STAYING_ALIVE = .9
PREFER_ACTIONS_WITH_REWARDS_OVER = 200
SOLVED_STEPS = 500

# Validation
VALIDATION_EPISODES = 3

# CONFIG
ENV = "CartPole-v1"
MODEL_PATH = "dqn_cartpole.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)
        
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Reward shaping to prefer actions with high rewards
        if reward > PREFER_ACTIONS_WITH_REWARDS_OVER:
            for _ in range(ACTION_DUPLICATION_FACTOR):
                self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

def select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state)
            return q_values.argmax().item()

def train_step(step_count, model, target_model, optimizer, replay_buffer, writer):
    if (len(replay_buffer) < BATCH_SIZE):
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    # Q(s, a) from the current model
    current_q = model(states).gather(1, actions)

    # Q target using target network (no gradients)
    with torch.no_grad():
        max_next_q = target_model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + GAMMA * max_next_q * (1 - dones)
    
    loss = nn.MSELoss()(current_q, target_q)

    if step_count % TRAIN_INTERVAL == 0:
        writer.add_scalar("Loss", loss.item(), step_count)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    print(f"----- Training DQN on {ENV} -----")
    
    env = gym.make(ENV)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    state_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore

    writer = SummaryWriter(log_dir="runs/dqn_cartpole")

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(MEM_SIZE)

    epsilon = EPS_START

    rewards = []
    best_reward = -float('inf')

    for episode in range(1, EPISODES + 1):
        
        state, _ = env.reset() # type: ignore
        step_count = 0
        total_reward = 0

        done = False
        while not done:
            step_count += 1
            action = select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Reward shaping
            if not done:
                reward += BONUS_FOR_STAYING_ALIVE # type: ignore
            if done and total_reward < MIN_REWARD_THRESHOLD:
                reward += EARLY_TERMINATION_PENALTY #type: ignore

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward # type: ignore

            if step_count % TRAIN_INTERVAL == 0:
                train_step(step_count, policy_net, target_net, optimizer, replay_buffer, writer)

        writer.add_scalar("Reward/Total", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        rewards.append(total_reward)

        # Epsilon decay
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Logging
        if len(rewards) >= INTERVAL and episode % INTERVAL == 0:
            avg_reward = np.mean(rewards[-INTERVAL:])
            model_saved = False

            if (avg_reward > best_reward):
                best_reward = avg_reward

                torch.save({
                        'model_state_dict': policy_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'episode': episode,
                        'epsilon': epsilon,
                        'rewards': rewards,
                }, MODEL_PATH)

                model_saved = True

            print(f"Episode: {episode:>4}, Average Reward: {avg_reward:7.3f}, Epsilon: {epsilon:6.3f} {'[ Saved ]':<6}" if model_saved else
                f"Episode: {episode:>4}, Average Reward: {avg_reward:7.3f}, Epsilon: {epsilon:6.3f} {'':<6}")

    writer.flush()
    writer.close()
    env.close()

def validate(model, env_name, episodes=10, render=False, seed=None):
    env = gym.make(env_name, render_mode="human" if render else None)
    
    if seed is None:
        seed = random.randint(0, 10000)

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    model.eval()

    total_rewards = []
    total_steps = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed) # type: ignore
        done = False
        total_reward = 0
        steps = 0

        while not done:
            steps += 1
            if render:
                env.render()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward #type: ignore
            state = next_state
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        print(f"Validation Episode {episode}: Reward = {total_reward}")

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    print(f"Validation Average Reward over {episodes} episodes: {avg_reward:.3f}")

    if (avg_steps == SOLVED_STEPS):
        print(f"{ENV} is solved!");

    env.close()

def load_and_validate_model(model_path, render=False, seed=None):
    print(f"----- Validating DQN on {ENV} -----")
    
    if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Please train the model first.")
            exit(1)

    model = DQN(4, 2).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    validate(model, ENV, episodes=VALIDATION_EPISODES, render=render, seed=seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test the trained model')
    args = parser.parse_args()

    try:
        if (args.test):
            load_and_validate_model(MODEL_PATH, render=True, seed=None)

        else:

            # Reproducibility
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            train()
            load_and_validate_model(MODEL_PATH, render=False, seed=SEED)
    except KeyboardInterrupt:
        print(f"Training/validation interrupted by user. Exiting...")
        exit(0)
