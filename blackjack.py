"""
This is a Blackjack game logic code.

This class represents a standard playing card with a suit and rank.
The card can be either face up or face down, affecting how it is displayed.

Attributes:
    suit (str): The suit of the card ('Hearts', 'Diamonds', 'Clubs', 'Spades')
    rank (str): The rank of the card ('Ace', '2', '3', ..., 'King')
    is_face_up (bool): Flag indicating if the card is face up (True) or face down (False)
"""

import random 

class BlackjackEnv: 
    """
    Parameters:
        num_decks (int): Number of standard 52-card decks shuffled together (typically 6 in casinos)
        dealer_hits_soft_17 (bool): If True, dealer hits on soft 17 (e.g., Ace+6)
        allow_double (bool): If True, player can double down (double bet for one more card)
        allow_split (bool): If True, player can split pairs into two hands
        allow_surrender (bool): If True, player can surrender and lose half their bet
        blackjack_payout (float): Payout multiplier for a natural blackjack (typically 1.5)
        """
        
    def __init__(self,
             num_decks=6,
             dealer_hits_soft_17=True,
             allow_double=True,
             allow_split=True,
             allow_surrender=False,
             blackjack_payout=1.5):
    
        # Game rules and parameters
        self.num_decks = num_decks
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.allow_double = allow_double
        self.allow_split = allow_split
        self.allow_surrender = allow_surrender
        self.blackjack_payout = blackjack_payout

        # Initialize the deck of cards and start a new round immediately
        self.reset_deck()
        self.reset()
        
    def reset_deck(self):
        """
        Resets and shuffles the deck of cards.

        This method creates a new deck with standard playing card values (1-10),
        where 10 appears four times (representing 10, Jack, Queen, King).
        The deck is multiplied by 4 (representing four suits) and by the number
        of decks specified in self.num_decks. The deck is then randomly shuffled.

        Returns:
            None
        """
        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4 * self.num_decks
        random.shuffle(self.deck)
    
    def draw_card(self):
        """
        Draws a single card from the deck.
        Removes and returns the top card from the deck using a pop operation.
        Returns:
            Card: A card object representing the drawn card from the top of the deck.
        """
        return self.deck.pop()
    
    def usable_ace(self, hand):
        """
        Check if the hand contains a usable ace (an ace that can be counted as 11 without busting).

        Args:
            hand (list): A list of integers representing the card values in the hand.

        Returns:
            bool: True if there is an ace (value 1) that can be counted as 11 without the total exceeding 21,
                  False otherwise.
        """
        return 1 in hand and sum(hand) + 10 <= 21

    def hand_sum(self, hand):
        """
        Calculate the sum of cards in a hand, considering Aces.

        This function adds up all card values in the hand and checks for usable Aces.
        If there is a usable Ace, it adds an additional 10 to the total (since Aces
        can count as 1 or 11).

        Args:
            hand (list): A list of integer values representing cards in the hand

        Returns:
            int: The total sum of the hand, accounting for any usable Aces
        """
        total = sum(hand)
        return total + 10 if self.usable_ace(hand) else total

    def is_blackjack(self, hand):
        """
        Check if a given hand is a blackjack (an ace and a 10-value card).

        Args:
            hand (list): A list of card values representing the player's or dealer's hand.

        Returns:
            bool: True if the hand is a blackjack (contains exactly an ace and a 10-value card),
                  False otherwise.

        Note:
            - The hand must contain exactly two cards
            - One card must be an ace (value 1)
            - The other card must be a 10-value card (10, Jack, Queen, or King)
        """
        return sorted(hand) == [1,10]

    def reset(self):
        """
        Resets the game state for a new round of Blackjack.
        This method performs the following actions:
        - Checks if the deck has fewer than 15 cards remaining and resets the deck if necessary.
        - Initializes the player's hand with two cards drawn from the deck.
        - Initializes the dealer's hand with two cards drawn from the deck.
        - Resets game-related flags and variables:
            - `done`: Indicates whether the game is over (set to False).
            - `bet`: The player's current bet (set to 1).
            - `double_used`: Tracks if the player has used the double-down option (set to False).
            - `surrendered`: Tracks if the player has surrendered (set to False).
        - Returns the current state of the game.
        Returns:
            object: The current state of the game, as determined by the `get_state` method.
        """
        if len(self.deck) < 15:
            self.reset_deck()

        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card(), self.draw_card()]
        
        self.done = False
        self.bet = 1
        self.double_used = False
        self.surrendered = False

        return self.get_state()

    def get_state(self):
        """
        Retrieve the current state of the Blackjack game.

        Returns:
            dict: A dictionary containing the following keys:
                - "player_sum" (int): The total sum of the player's hand.
                - "dealer_card" (int): The value of the dealer's face-up card.
                - "usable_ace" (bool): Indicates whether the player has a usable ace (an ace counted as 11 without busting).
                - "legal_actions" (list): A list of legal actions the player can take in the current state.
        """
        return {
            "player_sum": self.hand_sum(self.player_hand),
            "dealer_card": self.dealer_hand[0],
            "usable_ace": self.usable_ace(self.player_hand),
            "legal_actions": self.get_legal_actions()
        }

    def get_legal_actions(self):
        """
        Determines the legal actions available to the player based on the current game state.

        This function evaluates the player's hand and game rules to return a list of actions 
        that the player is allowed to take. The default actions are "hit" and "stand". 
        Additional actions such as "double", "split", and "surrender" may be included 
        depending on the following conditions:
        
        - "double": Allowed if the game rules permit doubling down and the player has exactly two cards.
        - "split": Allowed if the game rules permit splitting and the player's two cards are of the same rank.
        - "surrender": Allowed if the game rules permit surrendering and the player has exactly two cards.

        Returns:
            list: A list of strings representing the legal actions the player can take.
        """
        actions = ["hit", "stand"]
        if len(self.player_hand) == 2:
            if self.allow_double:
                actions.append("double")
            if self.allow_split and self.player_hand[0] == self.player_hand[1]:
                actions.append("split")
            if self.allow_surrender:
                actions.append("surrender")
        return actions
    
    def step(self, action):
        """
        Executes a step in the Blackjack game environment based on the given action.
        Args:
            action (str): The action to be performed by the player. Must be one of the legal actions
                          returned by `get_legal_actions()`. Possible actions include:
                          - "hit": Draw a card and add it to the player's hand. If the hand value exceeds 21, the game ends.
                          - "stand": End the player's turn and let the dealer play.
                          - "double": Double the player's bet, draw one card, and end the turn. If the hand value is 21 or less, the dealer plays.
                          - "split": (Not implemented) Intended for splitting a pair into two separate hands.
                          - "surrender": Forfeit the game and lose half the bet.
        Returns:
            tuple: A tuple containing:
                - state (object): The current state of the game after the action.
                - reward (float): The reward calculated based on the outcome of the action.
                - done (bool): A flag indicating whether the game has ended.
        Raises:
            AssertionError: If the provided action is not in the list of legal actions.
            NotImplementedError: If the "split" action is attempted, as it is not implemented.
        """
        assert action in self.get_legal_actions(), "Illegal action attempted!"

        if action == "hit":
            self.player_hand.append(self.draw_card())
            if self.hand_sum(self.player_hand) > 21:
                self.done = True

        elif action == "stand":
            self.done = True
            self.play_dealer()

        elif action == "double":
            self.bet *= 2
            self.player_hand.append(self.draw_card())
            self.done = True
            self.double_used = True
            if self.hand_sum(self.player_hand) <= 21:
                self.play_dealer()

        elif action == "split":
            # Implement split logic later if needed
            raise NotImplementedError("Split not implemented yet.")

        elif action == "surrender":
            self.bet *= -0.5
            self.surrendered = True
            self.done = True

        return self.get_state(), self.calculate_reward(), self.done

    def play_dealer(self):
        """
        Simulates the dealer's turn in a game of Blackjack.

        The dealer will continue drawing cards based on the following rules:
        - If the dealer's total hand value is less than 17, they will draw another card.
        - If the dealer's total hand value is exactly 17 and they have a usable ace 
          (making it a "soft 17"), they will draw another card only if the game rules 
          allow the dealer to hit on soft 17 (self.dealer_hits_soft_17 is True).
        - Otherwise, the dealer will stop drawing cards.

        This function modifies the dealer's hand in place by appending new cards 
        drawn from the deck until the dealer's turn is complete.
        """
        while True:
            dealer_total = self.hand_sum(self.dealer_hand)
            soft = self.usable_ace(self.dealer_hand)
            if dealer_total < 17 or (dealer_total == 17 and soft and self.dealer_hits_soft_17):
                self.dealer_hand.append(self.draw_card())
            else:
                break

    def calculate_reward(self):
        """
        Calculate the reward for the current state of the game.
        This function determines the reward based on the player's and dealer's hands,
        the game state, and whether the player has surrendered. The reward is calculated
        as follows:
        - If the game is not finished (`done` is False), the reward is 0.
        - If the player has surrendered, the reward is negative half of the bet.
        - If the player's hand total exceeds 21, the reward is negative the bet.
        - If the dealer's hand total exceeds 21 or the player's hand total is greater
          than the dealer's, the reward is the bet. If the player has a blackjack
          (a two-card hand totaling 21), the reward is the bet multiplied by the
          blackjack payout.
        - If the player's hand total is less than the dealer's, the reward is negative
          the bet.
        - If the player's hand total equals the dealer's, the reward is 0 (a push).
        Returns:
            int: The calculated reward for the current state of the game.
        """
        if not self.done:
            return 0

        player_total = self.hand_sum(self.player_hand)
        dealer_total = self.hand_sum(self.dealer_hand)

        if self.surrendered:
            return self.bet  # already negative half bet
        
        if player_total > 21:
            return -self.bet
        if dealer_total > 21 or player_total > dealer_total:
            if self.is_blackjack(self.player_hand) and len(self.player_hand) == 2:
                return self.bet * self.blackjack_payout
            return self.bet
        if player_total < dealer_total:
            return -self.bet
        return 0  # Push
    
        
    


        
 
        
        
