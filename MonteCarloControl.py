"""
MonteCarloControl.py

Enhanced Monte Carlo Control algorithm for Blackjack with adaptive stopping and epsilon decay.

Key Features:
- First-Visit Monte Carlo Control with ε-greedy policy using decay.
- Adaptive early stopping based on policy stability.
- Metrics tracking (average reward and policy changes) every batch.
- Visualization of learning progress.

Dependencies:
- Python standard libraries, NumPy, Matplotlib
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os 

def monte_carlo_first_visit_control(env, num_episodes=1000000, epsilon=0.1, gamma=1.0, 
                                    batch_size=5000, patience=10):
    """
    First-Visit Monte Carlo Control with ε-decay and adaptive stopping.

    Parameters:
        env: Instance of your Blackjack environment.
        num_episodes: Total episodes to train (default 1M for research quality).
        epsilon: Initial exploration probability (e.g., 0.1).
        gamma: Discount factor (typically 1.0 for Blackjack).
        batch_size: Number of episodes per logging/evaluation batch.
        patience: Number of consecutive batches with no policy changes before early stopping.

    Returns:
        policy: dict mapping state (player_sum, dealer_card, usable_ace) to best action.
        Q: nested dict mapping state -> action -> Q-value estimate.
    """
    actions_list = ["hit", "stand"]
    
    # Initialize dictionaries for Q-values, returns sum, and returns count.
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}
    
    # Initialize dictionary entries for every possible state.
    for player_total in range(4, 22):
        for dealer in range(1, 11):
            for usable in [False, True]:
                state = (player_total, dealer, usable)
                Q[state] = {action: 0.0 for action in actions_list}
                returns_sum[state] = {action: 0.0 for action in actions_list}
                returns_count[state] = {action: 0 for action in actions_list}
                policy[state] = random.choice(actions_list)
                
    def epsilon_greedy_action(state, legal_actions, eps):
        """Return action using ε-greedy selection from current policy."""
        if random.random() < eps:
            return random.choice(legal_actions)
        return policy[state]
    
    # Metric tracking dictionaries.
    episode_log = {"avg_reward": [], "policy_changes": [], "episodes_seen": []}
    no_policy_change_batches = 0
    batch_rewards = []
    policy_old = None  # To store policy snapshot at beginning of each batch.

    # Main loop: run episodes and update Q and policy.
    for episode in range(1, num_episodes + 1):
        # Epsilon decay: decay linearly from initial epsilon to a minimum of 0.01.
        current_epsilon = max(0.01, epsilon * (1 - episode / num_episodes))
        
        episode_history = []  # To store tuples: (state, action, reward)
        state_dict = env.reset()  # Reset environment.
        state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])
        done = False
        
        while not done:
            # Get legal actions from state and filter for actions in actions_list.
            legal_actions = [a for a in state_dict["legal_actions"] if a in actions_list]
            action = epsilon_greedy_action(state, legal_actions, current_epsilon)
            state_dict, reward, done = env.step(action)
            episode_history.append((state, action, reward))
            if not done:
                state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])
        
        # Use final reward from the episode (terminal reward) for logging.
        final_reward = episode_history[-1][2]
        batch_rewards.append(final_reward)
        
        # Process episode history in reverse order for first-visit updates.
        visited_state_actions = set()
        G = 0  # Cumulative return initialization.
        for t in range(len(episode_history) - 1, -1, -1):
            s, a, r = episode_history[t]
            G = r + gamma * G  # For blackjack gamma is usually 1.0.
            if (s, a) not in visited_state_actions:  # First-visit condition.
                visited_state_actions.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]
                best_action = max(Q[s], key=Q[s].get)
                policy[s] = best_action

        # At the start of each new batch, capture the current policy.
        if episode % batch_size == 1:
            policy_old = policy.copy()
        
        # At the end of each batch, log metrics.
        if episode % batch_size == 0:
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            # Count how many states changed compared to policy_old.
            changes = sum(1 for s in policy if policy[s] != policy_old.get(s, None))
            episode_log["episodes_seen"].append(episode)
            episode_log["avg_reward"].append(avg_reward)
            episode_log["policy_changes"].append(changes)
            print(f"Episode: {episode}, Avg Reward = {avg_reward:.4f}, Policy Changes = {changes}, Epsilon = {current_epsilon:.4f}")
            
            # Reset batch rewards for next batch.
            batch_rewards = []
            
            if changes == 0:
                no_policy_change_batches += 1
            else:
                no_policy_change_batches = 0
            
            # Early stopping if no policy changes persist for 'patience' batches.
            if no_policy_change_batches >= patience:
                print(f"Early stopping triggered after {episode} episodes due to no policy changes for {patience} batches.")
                break

    # Plot learning progress.
    plot_results(episode_log)
    
    return policy, Q

def plot_results(episode_log):
    episodes = episode_log["episodes_seen"]
    avg_rewards = episode_log["avg_reward"]
    policy_changes = episode_log["policy_changes"]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward', color='tab:blue')
    ax1.plot(episodes, avg_rewards, '-o', color='tab:blue', label='Avg Reward')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Policy Changes', color='tab:red')
    ax2.plot(episodes, policy_changes, '-s', color='tab:red', label='Policy Changes')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Monte Carlo Control Learning Progress')
    plt.tight_layout()
    plt.show()

def generate_strategy_csv(learned_policy, filename_prefix="strategy_McControl"):
    player_sums = list(range(4, 22))
    dealer_cards = list(range(1, 11))
    
    output_dir = "strategyTable"

    for usable in [False, True]:
        table = []

        # Header row
        header = ["Player Sum \\ Dealer Card"] + [str(dealer) for dealer in dealer_cards]
        table.append(header)

        for player_sum in player_sums:
            row = [str(player_sum)]
            for dealer in dealer_cards:
                state = (player_sum, dealer, usable)
                action = learned_policy.get(state, "N/A")
                row.append(action)
            table.append(row)

        # Save CSV
        suffix = "soft" if usable else "hard"
        filename = os.path.join(output_dir, f"{filename_prefix}_{suffix}.csv")
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(table)

        print(f"Strategy table ({suffix}) saved to: {filename}")


# Example Usage:
if __name__ == "__main__":
    from blackjack import BlackjackEnv  # Ensure correct import from your environment file.
    
    env = BlackjackEnv(
        num_decks=6,
        dealer_hits_soft_17=True,
        allow_double=True,
        allow_split=True,
        allow_surrender=False
    )
    
    learned_policy, Q_values = monte_carlo_first_visit_control(
        env, num_episodes=1000000, epsilon=0.1, gamma=1.0, batch_size=5000, patience=10
    )
    
    generate_strategy_csv(learned_policy)
    
    # Print a few sample state policies for inspection.
    sample_states = [(16, 10, False), (12, 3, True), (18, 7, False)]
    for state in sample_states:
        print(f"State {state}: Best action = {learned_policy.get(state, 'N/A')}")
