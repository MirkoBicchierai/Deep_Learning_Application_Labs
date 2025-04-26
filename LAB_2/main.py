# Standard imports.
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F

# Plus one non standard one -- we need this to sample from policies.
from torch.distributions import Categorical

# This *might* help PyGame crash less...
import pygame
_ = pygame.init()


# Given an environment, observation, and policy, sample from pi(a | obs). Returns the
# selected action and the log probability of that action (needed for policy gradient).
def select_action(env, obs, policy):
    dist = Categorical(policy(obs))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))


# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array. There's probably a better way to do this.
def compute_returns(rewards, gamma):
    return np.flip(np.cumsum([gamma ** (i + 1) * r for (i, r) in enumerate(rewards)][::-1]), 0).copy()


# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=500):
    # Collect just about everything.
    observations = []
    actions = []
    log_probs = []
    rewards = []

    # Reset the environment and start the episode.
    (obs, info) = env.reset()
    for i in range(maxlen):
        # Get the current observation, run the policy and select an action.
        obs = torch.tensor(obs)
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        # Advance the episode by executing the selected action.
        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


# A simple, but generic, policy network with one hidden layer.
class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s


def reinforce(policy, env, env_render=None, gamma=0.99, num_episodes=10,
              eval_interval=100, eval_episodes=10):
    """
    REINFORCE algorithm with improved evaluation metrics.

    Args:
        policy: The policy network to train
        env: Training environment
        env_render: Environment for rendering (optional)
        gamma: Discount factor
        num_episodes: Number of training episodes
        eval_interval: Evaluate the agent every N episodes
        eval_episodes: Number of episodes to run during evaluation

    Returns:
        dict: A dictionary containing training history and evaluation metrics
    """
    # The only non-vanilla part: we use Adam instead of SGD.
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)

    # Track metrics in lists
    running_rewards = [0.0]
    evaluation_history = []

    # The main training loop.
    policy.train()
    for episode in range(num_episodes):
        # Run an episode of the environment, collect everything needed for policy update.
        (observations, actions, log_probs, rewards) = run_episode(env, policy)

        # Compute the discounted reward for every step of the episode.
        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)

        # Keep a running average of total discounted rewards for the whole episode.
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])

        # Standardize returns.
        returns = (returns - returns.mean()) / returns.std()

        # Make an optimization step
        opt.zero_grad()
        loss = (-log_probs * returns).mean()
        loss.backward()
        opt.step()

        # Evaluate the agent every eval_interval episodes
        if episode > 0 and episode % eval_interval == 0:
            eval_metrics = evaluate_policy(policy, env, eval_episodes)
            evaluation_history.append({
                'episode': episode,
                'avg_total_reward': eval_metrics['avg_total_reward'],
                'avg_episode_length': eval_metrics['avg_episode_length']
            })

            print(f'Episode {episode}:')
            print(f'  Running reward: {running_rewards[-1]:.2f}')
            print(f'  Evaluation - Avg reward: {eval_metrics["avg_total_reward"]:.2f}, '
                  f'Avg length: {eval_metrics["avg_episode_length"]:.2f}')

            # Render an episode if env_render is provided
            if env_render:
                policy.eval()
                run_episode(env_render, policy)
                policy.train()

    # Final evaluation
    final_metrics = evaluate_policy(policy, env, eval_episodes)
    evaluation_history.append({
        'episode': num_episodes,
        'avg_total_reward': final_metrics['avg_total_reward'],
        'avg_episode_length': final_metrics['avg_episode_length']
    })

    # Return metrics history
    policy.eval()
    return {
        'running_rewards': running_rewards,
        'evaluation_history': evaluation_history
    }


def evaluate_policy(policy, env, num_episodes):
    """
    Evaluate a policy by running it for multiple episodes and collecting metrics.

    Args:
        policy: The policy to evaluate
        env: The environment to evaluate in
        num_episodes: Number of episodes to run

    Returns:
        dict: A dictionary containing evaluation metrics
    """
    policy.eval()  # Set policy to evaluation mode

    total_rewards = []
    episode_lengths = []

    for _ in range(num_episodes):
        observations, actions, _, rewards = run_episode(env, policy)
        total_reward = sum(rewards)
        episode_length = len(rewards)

        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    # Calculate average metrics
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)

    policy.train()  # Set policy back to training mode
    return {
        'avg_total_reward': avg_total_reward,
        'avg_episode_length': avg_episode_length,
        'all_rewards': total_rewards,
        'all_episode_lengths': episode_lengths
    }


def analyze_performance(metrics_history):
    """
    Analyze the performance of the agent using the collected metrics.

    Args:
        metrics_history: Dictionary containing training and evaluation metrics
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract metrics from history
    episodes = [entry['episode'] for entry in metrics_history['evaluation_history']]
    avg_rewards = [entry['avg_total_reward'] for entry in metrics_history['evaluation_history']]
    avg_lengths = [entry['avg_episode_length'] for entry in metrics_history['evaluation_history']]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot average total reward
    ax1.plot(episodes, avg_rewards, 'b-', label='Avg Total Reward')
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('Average Total Reward')
    ax1.set_title('Agent Performance: Average Total Reward')
    ax1.grid(True)

    # Plot average episode length
    ax2.plot(episodes, avg_lengths, 'r-', label='Avg Episode Length')
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Average Episode Length')
    ax2.set_title('Agent Performance: Average Episode Length')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Print final performance
    final_metrics = metrics_history['evaluation_history'][-1]
    print("\nFinal Agent Performance:")
    print(f"After {final_metrics['episode']} training episodes:")
    print(f"  - Average Total Reward: {final_metrics['avg_total_reward']:.2f}")
    print(f"  - Average Episode Length: {final_metrics['avg_episode_length']:.2f}")

def main():
    # Instantiate a (rendering) CartPole environment.
    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    pygame.display.init()  # Might help PyGame not crash...

    # Make a policy network and run a few episodes to see how well random initialization works.
    policy = PolicyNet(env_render)
    for _ in range(10):
        run_episode(env_render, policy)

    # If you don't close the environment, the PyGame window stays visible.
    env_render.close()

    # Again we pray PyGame doesn't crash...
    pygame.display.quit()

    # In the new version of Gymnasium you need different environments for rendering and no rendering.
    # Here we instaintiate two versions of cartpole, one that animates the episodes (which slows everything
    # down), and another that does not animate.

    env = gymnasium.make('CartPole-v1')
    env_render = None  # gymnasium.make('CartPole-v1', render_mode='human')

    # PyGame, please don't crash.
    pygame.display.init()

    # Make a policy network.
    policy = PolicyNet(env)

    metrics = reinforce(
        policy=policy,
        env=env,
        env_render=env,
        gamma=0.99,
        num_episodes=1000,
        eval_interval=100,  # Evaluate every 100 episodes
        eval_episodes=10  # Run 10 episodes during each evaluation
    )

    # Analyze the results
    analyze_performance(metrics)


    # Close up everything
    # env_render.close()
    env.close()
    pygame.display.quit()  # Fingers crossed...

    env_render = gymnasium.make('CartPole-v1', render_mode='human')
    for _ in range(10):
        run_episode(env_render, policy)
    env_render.close()


if __name__ == '__main__':
    main()