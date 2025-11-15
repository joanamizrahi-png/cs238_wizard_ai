"""
Game rules for Wizard
"""
from game.deck import Card, Suit
from typing import List, Optional

class WizardRules:
    """Implements Wizard game rules"""
    
    @staticmethod
    def get_trump_suit(trump_card: Optional[Card]) -> Optional[Suit]:
        """Determine trump suit from trump card"""
        if trump_card is None:
            return None  # Last round, no trump
        elif trump_card.suit == Suit.JESTER:
            return None  # Jester = no trump
        elif trump_card.suit == Suit.WIZARD:
            # In real game, dealer chooses. For now, default to Hearts
            return Suit.HEARTS  # TODO: Let dealer choose
        else:
            return trump_card.suit
    
    @staticmethod
    def get_valid_plays(hand: List[Card], led_suit: Optional[Suit]) -> List[Card]:
        """
        Get valid cards a player can play
        
        Rules:
        - Must follow suit if possible
        - Can always play Wizard or Jester
        - If can't follow suit, can play anything
        """
        # First trick or leading - can play anything
        if led_suit is None:
            return hand.copy()
        
        # Can always play Wizard or Jester
        same_suit = [c for c in hand if c.suit == led_suit]
        
        # If you have the led suit, must play it (or Wizard/Jester)
        if same_suit:
            wizards_jesters = [c for c in hand if c.suit in [Suit.WIZARD, Suit.JESTER]]
            return same_suit + wizards_jesters
        
        # If you don't have led suit, can play anything
        return hand.copy()
    
    @staticmethod
    def determine_trick_winner(
        cards_played: List[tuple[int, Card]],  # (player_idx, card)
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit]
    ) -> int:
        """
        Determine which player won the trick
        
        Returns: player_idx of winner
        """
        # Rule 1: First Wizard played wins
        for player_idx, card in cards_played:
            if card.suit == Suit.WIZARD:
                return player_idx
        
        # Rule 2: If all Jesters, first Jester wins
        if all(card.suit == Suit.JESTER for _, card in cards_played):
            return cards_played[0][0]
        
        # Filter out Jesters (they can't win unless all Jesters)
        non_jesters = [(p, c) for p, c in cards_played if c.suit != Suit.JESTER]
        
        # Rule 3: Highest trump wins
        if trump_suit:
            trumps = [(p, c) for p, c in non_jesters if c.suit == trump_suit]
            if trumps:
                winner = max(trumps, key=lambda x: x[1].rank)
                return winner[0]
        
        # Rule 4: Highest card of led suit wins
        led_cards = [(p, c) for p, c in non_jesters if c.suit == led_suit]
        if led_cards:
            winner = max(led_cards, key=lambda x: x[1].rank)
            return winner[0]
        
        # Shouldn't reach here, but just in case
        return cards_played[0][0]
    
    @staticmethod
    def score_round(bid: int, tricks_won: int) -> int:
        """
        Calculate score for a player in a round
        
        Correct bid: 20 + 10*n
        Wrong bid: -10 per trick off
        """
        if bid == tricks_won:
            return 20 + 10 * bid
        else:
            return -10 * abs(bid - tricks_won)