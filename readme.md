# Blackjack Optimization using Reinforcement Learning

## Project Overview
This project implements and compares different reinforcement learning approaches to optimize playing strategies in Blackjack. We explore three main algorithms:
- Deep Q-Networks (DQN)
- Monte Carlo Methods
- Proximal Policy Optimization (PPO)

## Requirements
```
python>=3.8
torch
gymnasium
numpy
pandas
matplotlib
```

## Implementation Details

### 1. Deep Q-Network (DQN)
- State-action value approximation using neural networks
- Experience replay for improved learning stability
- Target network for reducing overestimation bias

### 2. Monte Carlo Methods
- Episode-based learning
- Policy evaluation using returns
- First-visit Monte Carlo implementation

### 3. PPO (Proximal Policy Optimization)
- Policy gradient method
- Clipped objective function
- Value function estimation

## Project Structure
```
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   ├── monte_carlo_agent.py
│   │   └── ppo_agent.py
│   ├── environment/
│   │   └── blackjack_env.py
│   └── utils/
├── notebooks/
├── results/
└── README.md
```

## Running the Project
```bash
# Example commands to run different algorithms
python src/main.py --agent dqn
python src/main.py --agent monte_carlo
python src/main.py --agent ppo
```

## Results
Comparison of algorithms based on:
- Average rewards
- Learning speed
- Policy stability
- Win rate against the dealer

## References
- Sutton & Barto: Reinforcement Learning: An Introduction
- PPO: https://arxiv.org/abs/1707.06347
- DQN: https://www.nature.com/articles/nature14236