import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os 

def monte_carlo_weighted_off_policy_control(env, num_episodes=1000000, gamma=1.0, batch_size=5000, patience=10):
    
    """
    Performs Weighted Importance Sampling Off-Policy Monte Carlo Control to estimate the optimal policy
    for a given environment. This algorithm uses a behavior policy to generate episodes and updates
    the target policy greedily based on the action-value function (Q).
    Args:
        num_episodes (int, optional): The total number of episodes to run. Defaults to 1,000,000.
        gamma (float, optional): The discount factor for future rewards. Defaults to 1.0.
        batch_size (int, optional): The number of episodes per batch for logging and evaluation. Defaults to 5000.
        patience (int, optional): The number of consecutive batches with no policy changes before early stopping. Defaults to 10.
    Returns:
        tuple: A tuple containing:
            - policy (dict): The learned target policy mapping states to actions.
            - Q (dict): The action-value function mapping states to action-value pairs.
    Notes:
        - The behavior policy is uniform random over legal actions.
        - The target policy is updated greedily based on the action-value function.
        - Early stopping is triggered if no policy changes occur for a specified number of batches.
        - The function logs average rewards and policy changes per batch for analysis.
        
    """
    
    actions_list = ["hit", "stand"]
    Q = {}
    C = {}  # cumulative weights
    policy = {}

    # Initialize Q, C, and policy
    for player_total in range(4, 22):
        for dealer in range(1, 11):
            for usable in [False, True]:
                state = (player_total, dealer, usable)
                Q[state] = {a: 0.0 for a in actions_list}
                C[state] = {a: 0.0 for a in actions_list}
                policy[state] = random.choice(actions_list)

    def behavior_policy(_state, legal_actions):
        return random.choice(legal_actions)

    def target_prob(s, a):
        return 1.0 if policy[s] == a else 0.0

    episode_log = {"avg_reward": [], "policy_changes": [], "episodes_seen": []}
    no_policy_change_batches = 0
    batch_rewards = []
    policy_old = None

    for episode in range(1, num_episodes + 1):
        state_dict = env.reset()
        state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])
        episode_history = []
        done = False

        while not done:
            legal_actions = [a for a in state_dict["legal_actions"] if a in actions_list]
            action = behavior_policy(state, legal_actions)
            state_dict, reward, done = env.step(action)
            episode_history.append((state, action, reward))
            if not done:
                state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])

        batch_rewards.append(episode_history[-1][2])  # final reward

        G = 0
        W = 1.0  # Importance weight
        for t in range(len(episode_history) - 1, -1, -1):
            s, a, r = episode_history[t]
            G = gamma * G + r

            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            # Update target policy greedily
            best_action = max(Q[s], key=Q[s].get)
            policy[s] = best_action

            if policy[s] != a:
                break  # early termination of importance sampling

            W *= 1.0 / 0.5  # uniform behavior policy → 0.5 for any action

        if episode % batch_size == 1:
            policy_old = policy.copy()

        if episode % batch_size == 0:
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            changes = sum(1 for s in policy if policy[s] != policy_old.get(s, None))
            episode_log["episodes_seen"].append(episode)
            episode_log["avg_reward"].append(avg_reward)
            episode_log["policy_changes"].append(changes)
            print(f"Episode: {episode}, Avg Reward = {avg_reward:.4f}, Policy Changes = {changes}")
            batch_rewards = []

            if changes == 0:
                no_policy_change_batches += 1
            else:
                no_policy_change_batches = 0

            if no_policy_change_batches >= patience:
                print(f"Early stopping triggered after {episode} episodes (no policy changes for {patience} batches).")
                break

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

def generate_strategy_csv(learned_policy, filename_prefix="offpolicy_strategy_McControl"):
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
        
if __name__ == "__main__":
    from blackjack import BlackjackEnv
    
    env = BlackjackEnv(
        num_decks=6,
        dealer_hits_soft_17=True,
        allow_double=True,
        allow_split=True,
        allow_surrender=False
    )

    learned_policy, Q_values = monte_carlo_weighted_off_policy_control(
    env, num_episodes=1000000, gamma=1.0, batch_size=5000, patience=10
)

    # Print sample actions
    sample_states = [(16, 10, False), (12, 3, True), (18, 7, False)]
    for state in sample_states:
        print(f"State {state}: Best action = {learned_policy.get(state, 'N/A')}")

    # Export CSV
    generate_strategy_csv(learned_policy, filename_prefix="strategy_weighted_offpolicy")

