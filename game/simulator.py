"""
Main Wizard game simulator
"""
from typing import List, Optional
from dataclasses import dataclass
from game.deck import Deck, Card, Suit
from game.rules import WizardRules

@dataclass
class GameState:
    """Represents the current state of the game"""
    num_players: int
    round_num: int
    hands: List[List[Card]]
    trump_card: Optional[Card]
    trump_suit: Optional[Suit]
    bids: List[Optional[int]]  # None if not yet bid
    tricks_won: List[int]
    scores: List[int]
    current_trick: List[tuple[int, Card]]  # Cards played in current trick
    trick_leader: int  # Who leads the current trick
    
    def copy(self):
        """Create a copy of the game state"""
        return GameState(
            num_players=self.num_players,
            round_num=self.round_num,
            hands=[hand.copy() for hand in self.hands],
            trump_card=self.trump_card,
            trump_suit=self.trump_suit,
            bids=self.bids.copy(),
            tricks_won=self.tricks_won.copy(),
            scores=self.scores.copy(),
            current_trick=self.current_trick.copy(),
            trick_leader=self.trick_leader
        )

class WizardGame:
    """Manages a full game of Wizard"""
    
    def __init__(self, num_players: int = 4):
        """
        Initialize a new game
        
        Args:
            num_players: Number of players (3-6)
        """
        if num_players < 3 or num_players > 6:
            raise ValueError("Wizard requires 3-6 players")
        
        self.num_players = num_players
        self.total_rounds = 60 // num_players  # 20 for 3p, 15 for 4p, etc.
        self.scores = [0] * num_players
        self.state = None
        self.verbose = False  # Set to True for debug output
    
    def play_full_game(self, agents: List) -> List[int]:
        """
        Play a complete game with the given agents
        
        Args:
            agents: List of agent objects (must have bid() and play() methods)
        
        Returns:
            Final scores for each player
        """
        assert len(agents) == self.num_players
        
        for round_num in range(1, self.total_rounds + 1):
            self._play_round(round_num, agents)
            
            if self.verbose:
                print(f"\n=== After Round {round_num} ===")
                print(f"Scores: {self.scores}")
        
        return self.scores
    
    def _play_round(self, round_num: int, agents: List):
        """Play a single round"""
        
        # Deal cards
        hands, trump_card, _ = Deck.shuffle_and_deal(self.num_players, round_num)
        trump_suit = WizardRules.get_trump_suit(trump_card)
        
        # Initialize round state
        self.state = GameState(
            num_players=self.num_players,
            round_num=round_num,
            hands=hands,
            trump_card=trump_card,
            trump_suit=trump_suit,
            bids=[None] * self.num_players,
            tricks_won=[0] * self.num_players,
            scores=self.scores.copy(),
            current_trick=[],
            trick_leader=0  # Player 0 leads first trick
        )
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}")
            print(f"Trump: {trump_card} ({trump_suit})")
        
        # Bidding phase
        self._bidding_phase(agents)
        
        # Playing phase (multiple tricks)
        for trick_num in range(round_num):
            self._play_trick(agents)
        
        # Score the round
        self._score_round()
    
    def _bidding_phase(self, agents: List):
        """Handle bidding for all players"""
        
        if self.verbose:
            print(f"\n--- Bidding Phase ---")
        
        for player_idx in range(self.num_players):
            # Agent makes bid decision
            bid = agents[player_idx].bid(
                hand=self.state.hands[player_idx],
                trump_suit=self.state.trump_suit,
                round_num=self.state.round_num,
                player_idx=player_idx,
                bids_so_far=self.state.bids[:player_idx]
            )
            
            # Validate bid
            if not (0 <= bid <= self.state.round_num):
                raise ValueError(f"Invalid bid {bid} for round {self.state.round_num}")
            
            self.state.bids[player_idx] = bid
            
            if self.verbose:
                print(f"Player {player_idx} bids {bid}")
        
        if self.verbose:
            print(f"Total bids: {sum(self.state.bids)}/{self.state.round_num} tricks")
    
    def _play_trick(self, agents: List):
        """Play a single trick"""
        
        self.state.current_trick = []
        led_suit = None
        
        if self.verbose:
            print(f"\n--- Trick (Leader: Player {self.state.trick_leader}) ---")
        
        # Each player plays a card
        for i in range(self.num_players):
            player_idx = (self.state.trick_leader + i) % self.num_players
            
            # Get valid plays
            valid_cards = WizardRules.get_valid_plays(
                self.state.hands[player_idx],
                led_suit
            )
            
            # Agent chooses card
            card = agents[player_idx].play(
                hand=self.state.hands[player_idx],
                valid_cards=valid_cards,
                current_trick=self.state.current_trick,
                trump_suit=self.state.trump_suit,
                led_suit=led_suit,
                player_idx=player_idx,
                state=self.state
            )
            
            # Validate card choice
            if card not in valid_cards:
                raise ValueError(f"Invalid card play: {card} not in {valid_cards}")
            
            # Remove card from hand
            self.state.hands[player_idx].remove(card)
            
            # Add to trick
            self.state.current_trick.append((player_idx, card))
            
            # First card determines led suit (unless it's Jester or Wizard)
            if i == 0 and card.suit not in [Suit.JESTER, Suit.WIZARD]:
                led_suit = card.suit
            elif i == 1 and led_suit is None and card.suit not in [Suit.JESTER, Suit.WIZARD]:
                # If first card was Jester, second card sets suit
                led_suit = card.suit
            
            if self.verbose:
                print(f"  Player {player_idx} plays {card}")
        
        # Determine winner
        winner = WizardRules.determine_trick_winner(
            self.state.current_trick,
            led_suit,
            self.state.trump_suit
        )
        
        self.state.tricks_won[winner] += 1
        self.state.trick_leader = winner  # Winner leads next trick
        
        if self.verbose:
            print(f"  → Player {winner} wins the trick!")
            print(f"  Tricks won so far: {self.state.tricks_won}")
    
    def _score_round(self):
        """Calculate and apply scores for the round"""
        
        if self.verbose:
            print(f"\n--- Scoring ---")
        
        for player_idx in range(self.num_players):
            bid = self.state.bids[player_idx]
            won = self.state.tricks_won[player_idx]
            round_score = WizardRules.score_round(bid, won)
            
            self.scores[player_idx] += round_score
            
            if self.verbose:
                result = "✓" if bid == won else "✗"
                print(f"Player {player_idx}: bid {bid}, won {won} → {round_score:+d} {result}")