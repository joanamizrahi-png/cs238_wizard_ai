"""
Heuristic agent that uses simple rules
"""
from typing import List, Optional
from game.deck import Card, Suit

class HeuristicAgent:
    """Agent that uses hand-crafted heuristics"""
    
    def __init__(self, name: str = "Heuristic"):
        self.name = name
    
    def bid(self, hand: List[Card], trump_suit: Optional[Suit], round_num: int, 
        player_idx: int, bids_so_far: List) -> int:
        """
        Bid based on hand strength - MORE CONSERVATIVE
        """
        likely_tricks = 0
        
        # Wizards always win (100% confidence)
        wizards = [c for c in hand if c.suit == Suit.WIZARD]
        likely_tricks += len(wizards)
        
        # High trump cards (be more conservative)
        if trump_suit:
            trumps = [c for c in hand if c.suit == trump_suit]
            # Only count Kings and Aces of trump
            high_trumps = [c for c in trumps if c.rank >= 13 or c.rank == 1]
            likely_tricks += len(high_trumps) * 0.6  # 60% chance
        
        # Aces in non-trump suits (lower confidence - might be trumped)
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            if suit != trump_suit:
                aces = [c for c in hand if c.suit == suit and c.rank == 1]
                likely_tricks += len(aces) * 0.3  # Only 30% chance
        
        # Be conservative: round down and reduce
        bid = int(likely_tricks * 0.5)  # Very conservative
        
        # Additional conservative adjustment: avoid bidding 0 unless very weak
        if bid == 0 and len(wizards) == 0 and round_num > 1:
            # If we have any high card, bid at least 1
            high_cards = [c for c in hand if c.rank >= 11 and c.suit != Suit.JESTER]
            if high_cards:
                bid = 1
        
        # Clamp to valid range
        return max(0, min(bid, round_num))
    
    def play(self, hand: List[Card], valid_cards: List[Card], 
             current_trick: List[tuple[int, Card]], trump_suit: Optional[Suit],
             led_suit: Optional[Suit], player_idx: int, state) -> Card:
        """
        Play a card based on current situation
        
        Strategy:
        - If trying to win: play high cards
        - If trying to avoid: play low cards
        - Consider trump and led suit
        """
        my_bid = state.bids[player_idx]
        my_tricks_won = state.tricks_won[player_idx]
        tricks_remaining = state.round_num - len([t for t in state.tricks_won if t > 0])
        
        # Determine if we want to win this trick
        want_to_win = my_tricks_won < my_bid
        
        if want_to_win:
            return self._play_to_win(valid_cards, current_trick, trump_suit, led_suit)
        else:
            return self._play_to_lose(valid_cards, current_trick, trump_suit, led_suit)
    
    def _play_to_win(self, valid_cards: List[Card], current_trick: List, 
                     trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> Card:
        """Try to win the trick"""
        
        # If we have a Wizard, play it!
        wizards = [c for c in valid_cards if c.suit == Suit.WIZARD]
        if wizards:
            return wizards[0]
        
        # If we're leading, play highest card
        if not current_trick:
            # Prefer high trump or high cards
            if trump_suit:
                trumps = [c for c in valid_cards if c.suit == trump_suit]
                if trumps:
                    return max(trumps, key=lambda c: c.rank)
            
            # Otherwise highest card
            non_jesters = [c for c in valid_cards if c.suit != Suit.JESTER]
            if non_jesters:
                return max(non_jesters, key=lambda c: c.rank)
            return valid_cards[0]
        
        # If following, try to beat current winning card
        return self._try_to_beat_trick(valid_cards, current_trick, trump_suit, led_suit)
    
    def _play_to_lose(self, valid_cards: List[Card], current_trick: List,
                      trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> Card:
        """Try to avoid winning the trick"""
        
        # Never play Wizards when trying to lose
        non_wizards = [c for c in valid_cards if c.suit != Suit.WIZARD]
        if non_wizards:
            valid_cards = non_wizards
        
        # Play Jesters if possible
        jesters = [c for c in valid_cards if c.suit == Suit.JESTER]
        if jesters:
            return jesters[0]
        
        # Play lowest card
        return min(valid_cards, key=lambda c: c.rank if c.suit != Suit.JESTER else -1)
    
    def _try_to_beat_trick(self, valid_cards: List[Card], current_trick: List,
                          trump_suit: Optional[Suit], led_suit: Optional[Suit]) -> Card:
        """Try to play a card that beats the current trick"""
        
        # Figure out what's currently winning
        current_winner_card = self._get_current_winner(current_trick, led_suit, trump_suit)
        
        # Try to beat it
        if current_winner_card.suit == Suit.WIZARD:
            # Can't beat a Wizard, play lowest
            return min(valid_cards, key=lambda c: c.rank if c.suit != Suit.JESTER else -1)
        
        # Try to play higher card of the same category
        if trump_suit and current_winner_card.suit == trump_suit:
            # Need higher trump
            higher_trumps = [c for c in valid_cards 
                           if c.suit == trump_suit and c.rank > current_winner_card.rank]
            if higher_trumps:
                return min(higher_trumps, key=lambda c: c.rank)  # Lowest winning card
            
            # Can't beat, play trump anyway or play lowest
            trumps = [c for c in valid_cards if c.suit == trump_suit]
            if trumps:
                return min(trumps, key=lambda c: c.rank)
        
        # Try trump if we have it
        if trump_suit:
            trumps = [c for c in valid_cards if c.suit == trump_suit]
            if trumps:
                return min(trumps, key=lambda c: c.rank)  # Lowest trump
        
        # Try to beat with higher card of led suit
        if led_suit:
            higher_led = [c for c in valid_cards 
                         if c.suit == led_suit and c.rank > current_winner_card.rank]
            if higher_led:
                return min(higher_led, key=lambda c: c.rank)
        
        # Can't win, play lowest card
        return min(valid_cards, key=lambda c: c.rank if c.suit != Suit.JESTER else -1)
    
    def _get_current_winner(self, current_trick: List[tuple[int, Card]], 
                           led_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Determine which card is currently winning the trick"""
        from game.rules import WizardRules
        
        if not current_trick:
            return None
        
        winner_idx = WizardRules.determine_trick_winner(
            current_trick, led_suit, trump_suit
        )
        
        # Find the winning card
        for player_idx, card in current_trick:
            if player_idx == winner_idx:
                return card
        
        return current_trick[0][1]  # Shouldn't reach here