"""
Card and deck representations for Wizard
"""
from enum import Enum
from dataclasses import dataclass
from typing import List
import random

class Suit(Enum):
    HEARTS = 'H'
    DIAMONDS = 'D'
    CLUBS = 'C'
    SPADES = 'S'
    WIZARD = 'W'
    JESTER = 'J'

@dataclass(frozen=True)
class Card:
    """Represents a single card"""
    suit: Suit
    rank: int  # 1-13 for normal cards, 0 for Wizard/Jester
    
    def __str__(self):
        if self.suit == Suit.WIZARD:
            return "Wizard"
        elif self.suit == Suit.JESTER:
            return "Jester"
        else:
            rank_names = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
            rank_str = rank_names.get(self.rank, str(self.rank))
            return f"{rank_str}{self.suit.value}"
    
    def __repr__(self):
        return str(self)

class Deck:
    """Creates and manages the Wizard deck"""
    
    @staticmethod
    def create_standard_deck() -> List[Card]:
        """Create a standard 60-card Wizard deck"""
        cards = []
        
        # Add 4 Jesters
        for _ in range(4):
            cards.append(Card(Suit.JESTER, 0))
        
        # Add 4 Wizards
        for _ in range(4):
            cards.append(Card(Suit.WIZARD, 0))
        
        # Add standard 52 cards (1-13 in each suit)
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            for rank in range(1, 14):  # 1 = Ace, 13 = King
                cards.append(Card(suit, rank))
        
        return cards
    
    @staticmethod
    def shuffle_and_deal(num_players: int, round_num: int) -> tuple:
        """
        Shuffle deck and deal cards for a round
        
        Returns:
            (player_hands, trump_card, remaining_deck)
        """
        deck = Deck.create_standard_deck()
        random.shuffle(deck)
        
        # Deal cards
        hands = [[] for _ in range(num_players)]
        card_idx = 0
        
        for _ in range(round_num):  # Each player gets 'round_num' cards
            for player_idx in range(num_players):
                hands[player_idx].append(deck[card_idx])
                card_idx += 1
        
        # Trump card (if cards remain)
        trump_card = deck[card_idx] if card_idx < len(deck) else None
        
        return hands, trump_card, deck[card_idx+1:]