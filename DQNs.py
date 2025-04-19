import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from blackjack import BlackjackEnv
import os
import csv

"""
            Trains a Deep Q-Network (DQN) on the given environment.
            Args:
                env (object): The environment to train the DQN on. Must have `reset` and `step` methods.
                policy_net (torch.nn.Module): The policy network used to predict Q-values.
                target_net (torch.nn.Module): The target network used for stable Q-value updates.
                buffer_capacity (int, optional): Maximum capacity of the replay buffer. Defaults to 20000.
                batch_size (int, optional): Number of samples per training batch. Defaults to 128.
                gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
                learning_rate (float, optional): Learning rate for the optimizer. Defaults to 5e-5.
                epsilon_start (float, optional): Initial value of epsilon for epsilon-greedy policy. Defaults to 1.0.
                epsilon_min (float, optional): Minimum value of epsilon for epsilon-greedy policy. Defaults to 0.05.
                epsilon_decay (float, optional): Decay rate of epsilon per step. Defaults to 0.9998.
                num_episodes (int, optional): Total number of episodes to train. Defaults to 25000.
                target_update_freq (int, optional): Frequency (in steps) to update the target network. Defaults to 2000.
                early_stop_patience (int, optional): Number of evaluation batches without improvement before early stopping. Defaults to 20.
            Returns:
                None
            Side Effects:
                - Trains the policy network using the DQN algorithm.
                - Logs average rewards every 100 episodes.
                - Optionally stops training early if no improvement is observed.
                - Plots the average rewards over time.
            Notes:
                - The replay buffer stores transitions (state, action, reward, next_state, done) for training.
                - The target network is updated periodically to stabilize training.
                - The function uses an epsilon-greedy policy for exploration.
"""

class DQNNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=2, dropout_p=0.2):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout_p)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drop2(x)
        return self.output(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

def generate_strategy_csv(policy_net, filename_prefix="DQN_strategy"):
    player_sums = list(range(4, 22))
    dealer_cards = list(range(1, 11))
    output_dir = "strategyTable"
    os.makedirs(output_dir, exist_ok=True)

    for usable in [False, True]:
        table = []
        header = ["Player Sum \\ Dealer Card"] + [str(dealer) for dealer in dealer_cards]
        table.append(header)

        for player_sum in player_sums:
            row = [str(player_sum)]
            for dealer in dealer_cards:
                state = np.array([player_sum, dealer, int(usable)], dtype=np.float32)
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()
                row.append(action)
            table.append(row)

        suffix = "soft" if usable else "hard"
        filename = os.path.join(output_dir, f"{filename_prefix}_{suffix}.csv")
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(table)

        print(f"Strategy table ({suffix}) saved to: {filename}")

def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return random.choice(action_space)
    else:
        policy_net.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        policy_net.train()
        return torch.argmax(q_values).item()

def plot_rewards(episodes, avg_rewards):
    plt.figure(figsize=(12,6))
    plt.plot(episodes, avg_rewards, label='Average Reward (100-episode)')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('DQN Blackjack Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_dqn(env, policy_net, target_net, buffer_capacity=20000,
              batch_size=128, gamma=0.99,
              learning_rate=5e-5,
              epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9994,
              num_episodes=25000, target_update_freq=2000,
              early_stop_patience=100, min_improvement_delta=0.01):

    replay_buffer = ReplayBuffer(buffer_capacity)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    epsilon = epsilon_start
    total_steps = 0
    best_avg_reward = -float('inf')
    no_improve_batches = 0

    rewards_history = []
    avg_rewards_log = []

    for episode in range(1, num_episodes + 1):
        state_dict = env.reset()
        state = np.array([state_dict["player_sum"], state_dict["dealer_card"], int(state_dict["usable_ace"])])
        done = False
        episode_reward = 0

        while not done:
            action = select_action(state, policy_net, epsilon, [0, 1])
            action_name = ["hit", "stand"][action]

            next_state_dict, reward, done = env.step(action_name)
            next_state = np.array([
                next_state_dict["player_sum"], next_state_dict["dealer_card"],
                int(next_state_dict["usable_ace"])
            ])

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones.astype(int))

                q_pred = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0]
                    q_target = rewards + gamma * q_next * (1 - dones)

                loss = loss_fn(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-300:])
            avg_rewards_log.append(avg_reward)
            print(f"Episode {episode} | Steps: {total_steps} | Epsilon: {epsilon:.4f} | AvgReward(100): {avg_reward:.4f}")

            if avg_reward > best_avg_reward + min_improvement_delta:
                best_avg_reward = avg_reward
                no_improve_batches = 0
            else:
                no_improve_batches += 1

            if no_improve_batches >= early_stop_patience:
                print(f" Early stopping triggered at episode {episode} | Best AvgReward: {best_avg_reward:.4f}")
                break

    plot_rewards(np.arange(100, episode + 1, 100), avg_rewards_log)
    print(" Training complete.")
    generate_strategy_csv(policy_net, filename_prefix="DQN_strategy")

    

def evaluate_policy(env, policy_net, num_episodes=10000):
    """
    Evaluates the learned DQN policy without exploration.
    
    Args:
        env (object): Blackjack environment
        policy_net (nn.Module): Trained DQN policy network
        num_episodes (int): Number of games to play for evaluation

    Returns:
        dict: Win, loss, draw counts and win rate
    """
    results = {"win": 0, "loss": 0, "draw": 0}
    
    for _ in range(num_episodes):
        state_dict = env.reset()
        state = np.array([state_dict["player_sum"], state_dict["dealer_card"], int(state_dict["usable_ace"])])
        done = False
        
        while not done:
            with torch.no_grad():
                action = torch.argmax(policy_net(torch.FloatTensor(state).unsqueeze(0))).item()
            action_name = ["hit", "stand"][action]
            next_state_dict, reward, done = env.step(action_name)
            state = np.array([next_state_dict["player_sum"], next_state_dict["dealer_card"], int(next_state_dict["usable_ace"])])

        if reward > 0:
            results["win"] += 1
        elif reward < 0:
            results["loss"] += 1
        else:
            results["draw"] += 1

    total = sum(results.values())
    win_rate = results["win"] / total
    print(f"\nPolicy Evaluation over {total} episodes:")
    print(f" Wins  : {results['win']} ({results['win']/total:.2%})")
    print(f" Losses: {results['loss']} ({results['loss']/total:.2%})")
    print(f" Draws : {results['draw']} ({results['draw']/total:.2%})")
    print(f" Win Rate: {win_rate:.2%}")
    return results

   
if __name__ == "__main__":
    env = BlackjackEnv(
        num_decks=6,
        dealer_hits_soft_17=True,
        allow_double=True,
        allow_split=True,
        allow_surrender=False
    )

    state_dim = 3
    action_dim = 2

    policy_net = DQNNetwork(input_dim=state_dim, hidden_dim=256, output_dim=action_dim)
    target_net = DQNNetwork(input_dim=state_dim, hidden_dim=256, output_dim=action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    train_dqn(
        env=env,
        policy_net=policy_net,
        target_net=target_net,
        buffer_capacity=20000,
        batch_size=128,
        gamma=0.99,
        learning_rate=5e-5,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.994,
        num_episodes=25000,
        target_update_freq=5000,
        early_stop_patience=50
    )
    
    # Final evaluation of the learned policy
    evaluate_policy(env, policy_net, num_episodes=10000)
    
