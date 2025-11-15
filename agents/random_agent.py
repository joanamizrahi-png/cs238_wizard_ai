"""
Random agent that makes random decisions
"""
import random
from typing import List
from game.deck import Card, Suit

class RandomAgent:
    """Agent that plays randomly"""
    
    def __init__(self, name: str = "Random"):
        self.name = name
    
    def bid(self, hand: List[Card], trump_suit, round_num, player_idx, bids_so_far) -> int:
        """Make a random bid between 0 and number of cards"""
        return random.randint(0, round_num)
    
    def play(self, hand: List[Card], valid_cards: List[Card], **kwargs) -> Card:
        """Play a random valid card"""
        return random.choice(valid_cards)