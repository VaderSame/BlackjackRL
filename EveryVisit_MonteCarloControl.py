import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import os 

def monte_carlo_every_visit_control(env, num_episodes=1000000, epsilon=0.1, gamma=1.0,
                                    batch_size=5000, patience=10):
    """
    Every-Visit Monte Carlo Control with ε-greedy policy improvement and decay.
    Updates Q-values for *every* occurrence of (state, action) in each episode.
    """
    actions_list = ["hit", "stand"]
    Q = {}
    returns_sum = {}
    returns_count = {}
    policy = {}

    # Initialize Q, returns, policy
    for player_total in range(4, 22):
        for dealer in range(1, 11):
            for usable in [False, True]:
                state = (player_total, dealer, usable)
                Q[state] = {action: 0.0 for action in actions_list}
                returns_sum[state] = {action: 0.0 for action in actions_list}
                returns_count[state] = {action: 0 for action in actions_list}
                policy[state] = random.choice(actions_list)

    def epsilon_greedy_action(state, legal_actions, eps):
        if random.random() < eps:
            return random.choice(legal_actions)
        return policy[state]

    # Metric tracking
    episode_log = {"avg_reward": [], "policy_changes": [], "episodes_seen": []}
    no_policy_change_batches = 0
    batch_rewards = []
    policy_old = None

    for episode in range(1, num_episodes + 1):
        current_epsilon = max(0.01, epsilon * (1 - episode / num_episodes))
        episode_history = []
        state_dict = env.reset()
        state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])
        done = False

        while not done:
            legal_actions = [a for a in state_dict["legal_actions"] if a in actions_list]
            action = epsilon_greedy_action(state, legal_actions, current_epsilon)
            state_dict, reward, done = env.step(action)
            episode_history.append((state, action, reward))
            if not done:
                state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])

        final_reward = episode_history[-1][2]
        batch_rewards.append(final_reward)

        # Every-visit: update for every (state, action) in the episode
        G = 0
        for t in range(len(episode_history) - 1, -1, -1):
            s, a, r = episode_history[t]
            G = r + gamma * G
            returns_sum[s][a] += G
            returns_count[s][a] += 1
            Q[s][a] = returns_sum[s][a] / returns_count[s][a]
            best_action = max(Q[s], key=Q[s].get)
            policy[s] = best_action

        if episode % batch_size == 1:
            policy_old = policy.copy()

        if episode % batch_size == 0:
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            changes = sum(1 for s in policy if policy[s] != policy_old.get(s, None))
            episode_log["episodes_seen"].append(episode)
            episode_log["avg_reward"].append(avg_reward)
            episode_log["policy_changes"].append(changes)
            print(f"Episode: {episode}, Avg Reward = {avg_reward:.4f}, Policy Changes = {changes}, Epsilon = {current_epsilon:.4f}")
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

def generate_strategy_csv(learned_policy, filename_prefix="strategy_EveryVisit_McControl"):
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
    
    # Run Every-Visit MC
    learned_policy, Q_values = monte_carlo_every_visit_control(
        env, num_episodes=1000000, epsilon=0.1, gamma=1.0, batch_size=5000, patience=10
    )

    # Sample state check
    sample_states = [(16, 10, False), (12, 3, True), (18, 7, False)]
    for state in sample_states:
        print(f"State {state}: Best action = {learned_policy.get(state, 'N/A')}")

    # Optionally export strategy tables
    generate_strategy_csv(learned_policy, filename_prefix="strategy_every_visit")

    