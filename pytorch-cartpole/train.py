import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from models import DQNAgent

# Hyperparameters
EPISODES = 5000
EARLY_TERMINATION_PENALTY = -50
BONUS_FOR_STAYING_ALIVE = 0.9
MIN_REWARD_THRESHOLD = 300
SOLVED_THRESHOLD = 700
PROGRESS_PRINT_INTERVAL = 100
RUNNING_AVERAGE_WINDOW = 100
TEST_EPISODES = 5
DEFAULT_TEST_EPISODES = 10
SOLVED_EPISODES_WINDOW = 100

# Exploration parameters
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Learning parameters
LEARNING_RATE = 5e-4

def train_cartpole():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, lr=LEARNING_RATE, epsilon=EPSILON_START, 
                     epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN)
    scores = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Custom reward shaping for better learning
            if done and total_reward < MIN_REWARD_THRESHOLD:
                reward = EARLY_TERMINATION_PENALTY
            else:
                reward = reward + BONUS_FOR_STAYING_ALIVE
            
            agent.remember(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        agent.replay()
        scores.append(total_reward)
        
        # Print progress
        if episode % PROGRESS_PRINT_INTERVAL == 0:
            avg_score = np.mean(scores[-RUNNING_AVERAGE_WINDOW:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Early stopping if solved
        if len(scores) >= SOLVED_EPISODES_WINDOW and np.mean(scores[-SOLVED_EPISODES_WINDOW:]) >= SOLVED_THRESHOLD:
            print(f"Environment solved in {episode} episodes!")
            break
    
    env.close()
    return agent, scores

def test_agent(agent, episodes=TEST_EPISODES, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state, training=False)  # No exploration
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()
    print(f"Average test score: {np.mean(scores):.2f}")
    return scores

def plot_scores(scores):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    running_avg = [np.mean(scores[max(0, i-RUNNING_AVERAGE_WINDOW):i+1]) for i in range(len(scores))]
    plt.plot(running_avg)
    plt.title('Running Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.axhline(y=195, color='r', linestyle='--', label='Solved threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training DQN on CartPole-v1...")
    agent, training_scores = train_cartpole()
    
    print("\nTesting trained agent...")
    test_scores = test_agent(agent, episodes=DEFAULT_TEST_EPISODES)

    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'cartpole_dqn.pth')
    print("Model saved as 'cartpole_dqn.pth'")

    print("\nPlotting training progress...")
    plot_scores(training_scores)
