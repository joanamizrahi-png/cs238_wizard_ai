"""
Simplified MCTS agent for initial testing
"""
import random
from typing import List, Optional
from game.deck import Card, Suit
from game.simulator import GameState
from agents.heuristic_agent import HeuristicAgent

class SimpleMCTSAgent:
    """Simplified MCTS agent that uses Monte Carlo rollouts"""
    
    def __init__(self, name: str = "SimpleMCTS", num_simulations: int = 100):
        self.name = name
        self.num_simulations = num_simulations
        self.heuristic = HeuristicAgent()
    
    def bid(self, hand: List[Card], trump_suit: Optional[Suit], round_num: int,
            player_idx: int, bids_so_far: List) -> int:
        """Use heuristic for bidding"""
        return self.heuristic.bid(hand, trump_suit, round_num, player_idx, bids_so_far)
    
    def play(self, hand: List[Card], valid_cards: List[Card],
             current_trick: List[tuple[int, Card]], trump_suit: Optional[Suit],
             led_suit: Optional[Suit], player_idx: int, state: GameState) -> Card:
        """
        Choose card using simple Monte Carlo evaluation
        
        For each valid card:
        - Simulate playing it
        - Estimate value
        - Choose best
        """
        if len(valid_cards) == 1:
            return valid_cards[0]
        
        card_values = {}
        
        for card in valid_cards:
            total_value = 0.0
            
            # Run simulations for this card
            for _ in range(self.num_simulations):
                value = self._simulate_card(card, state, player_idx)
                total_value += value
            
            card_values[card] = total_value / self.num_simulations
        
        # Choose best card
        best_card = max(card_values.keys(), key=lambda c: card_values[c])
        return best_card
    
    def _simulate_card(self, card: Card, state: GameState, player_idx: int) -> float:
        """
        Simulate playing a card and estimate value
        
        Simple heuristic value:
        - Positive if helps meet bid
        - Negative if hurts bid chances
        """
        my_bid = state.bids[player_idx]
        my_tricks = state.tricks_won[player_idx]
        tricks_left = state.round_num - sum(1 for t in state.tricks_won if t > 0)
        
        # Estimate if this card will win the trick
        will_win = self._estimate_win_probability(card, state, player_idx)
        
        # Value based on whether we want to win
        want_to_win = my_tricks < my_bid
        
        if want_to_win and will_win:
            return 10  # Good - helps meet bid
        elif not want_to_win and not will_win:
            return 10  # Good - avoids extra trick
        elif want_to_win and not will_win:
            return -5  # Bad - needed to win
        else:  # not want_to_win and will_win
            return -10  # Bad - unwanted trick
    
    def _estimate_win_probability(self, card: Card, state: GameState, 
                                   player_idx: int) -> bool:
        """Roughly estimate if this card will win the trick"""
        from game.deck import Suit
        
        # Wizard always wins
        if card.suit == Suit.WIZARD:
            return True
        
        # Jester rarely wins
        if card.suit == Suit.JESTER:
            return False
        
        # High cards more likely to win
        # High trump very likely
        if state.trump_suit and card.suit == state.trump_suit and card.rank >= 10:
            return True
        
        # Aces somewhat likely
        if card.rank == 1:  # Ace
            return random.random() > 0.3
        
        # High cards maybe
        if card.rank >= 11:
            return random.random() > 0.5
        
        return False