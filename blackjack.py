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
import numpy as np
import matplotlib.pyplot as plt

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

        
    


        
 
        
        
