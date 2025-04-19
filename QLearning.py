import random
import numpy as np
import matplotlib.pyplot as plt
import os , csv

def q_learning_control(env, num_episodes=1000000, alpha=0.1, gamma=1.0, 
                       epsilon=0.1, batch_size=5000, patience=10):

    actions_list = ["hit", "stand"]
    Q = {}
    policy = {}

    # Initialize Q-table and policy randomly
    for player_total in range(4, 22):
        for dealer in range(1, 11):
            for usable in [False, True]:
                state = (player_total, dealer, usable)
                Q[state] = {a: 0.0 for a in actions_list}
                policy[state] = random.choice(actions_list)

    def epsilon_greedy(state, legal_actions, eps):
        if random.random() < eps:
            return random.choice(legal_actions)
        return policy[state]

    episode_log = {"avg_reward": [], "policy_changes": [], "episodes_seen": []}
    no_policy_change_batches = 0
    batch_rewards = []
    policy_old = None

    for episode in range(1, num_episodes + 1):
        state_dict = env.reset()
        state = (state_dict["player_sum"], state_dict["dealer_card"], state_dict["usable_ace"])
        done = False

        while not done:
            legal_actions = [a for a in state_dict["legal_actions"] if a in actions_list]
            action = epsilon_greedy(state, legal_actions, epsilon)

            next_state_dict, reward, done = env.step(action)

            next_state = (next_state_dict["player_sum"],
                          next_state_dict["dealer_card"],
                          next_state_dict["usable_ace"]) if not done else None

            # Q-update using max over next state's actions
            max_q_next = max(Q[next_state].values()) if next_state else 0
            Q[state][action] += alpha * (reward + gamma * max_q_next - Q[state][action])

            # Policy improvement
            policy[state] = max(Q[state], key=Q[state].get)
            state = next_state if not done else None
            state_dict = next_state_dict if not done else None

        # Log final reward of the episode
        batch_rewards.append(reward)

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

def generate_strategy_csv(learned_policy, filename_prefix="Qlearning_strategy"):
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

    learned_policy, Q_values = q_learning_control(
        env, num_episodes=1000000, alpha=0.1, gamma=1.0,
        epsilon=0.1, batch_size=5000, patience=10
    )

    sample_states = [(16, 10, False), (12, 3, True), (18, 7, False)]
    for state in sample_states:
        print(f"State {state}: Best action = {learned_policy.get(state, 'N/A')}")

    generate_strategy_csv(learned_policy, filename_prefix="strategy_q_learning")
