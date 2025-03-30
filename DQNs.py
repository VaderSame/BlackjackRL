"""
This is going to be the DQN network for blackjack environment

A neural network model that serves as the Deep Q-Network (DQN) for playing Blackjack.
Takes the current state of the game as input and outputs Q-values for each possible action.

Attributes:
    fc1 (nn.Linear): First fully connected layer with input size matching state dimensions
    fc2 (nn.Linear): Second fully connected layer
    fc3 (nn.Linear): Output layer with size matching number of possible actions in Blackjack

Methods:
    forward(x): Forward pass through the network that processes state input to action values

Returns:
    torch.Tensor: Q-values for each possible action in the Blackjack environment
"""