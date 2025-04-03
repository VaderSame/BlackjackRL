from blackjack import BlackjackEnv
import random

def run_random_agent(env, num_episodes=100000):
    total_reward = 0
    wins = 0
    for i in range(num_episodes):
        state = env.reset()
        done = False
        # Run a single episode
        while not done:
            # Retrieve legal actions from the state
            legal_actions = state["legal_actions"]
             # Filter out "split" if it's not implemented
            legal_actions = [a for a in legal_actions if a != "split"]
            # Choose an action at random
            action = random.choice(legal_actions)
            # Apply the action; get the next state, reward, and done flag
            state, reward, done = env.step(action)
        total_reward += reward
        if reward > 0:
            wins += 1
    avg_reward = total_reward / num_episodes
    win_rate = (wins / num_episodes) * 100
    print(f"Random Agent: Average Reward = {avg_reward:.3f}, Win Rate = {win_rate:.2f}%")
    return avg_reward, win_rate

env = BlackjackEnv(num_decks=6, 
                   dealer_hits_soft_17=True, 
                   allow_double=True, 
                   allow_split=True, 
                   allow_surrender=False)
run_random_agent(env, num_episodes=100000)
