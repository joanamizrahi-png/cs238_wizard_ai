"""
Monte Carlo Tree Search agent for Wizard
"""
import random
import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from game.deck import Card, Suit
from game.simulator import GameState
from game.rules import WizardRules

@dataclass
class MCTSNode:
    """Node in the MCTS search tree"""
    state: GameState
    parent: Optional['MCTSNode'] = None
    action: Optional[Card] = None  # Action that led to this node
    children: List['MCTSNode'] = None
    visits: int = 0
    total_value: float = 0.0
    untried_actions: List[Card] = None
    player_idx: int = 0  # Which player this node is deciding for
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = []
    
    @property
    def value(self) -> float:
        """Average value of this node"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def ucb1(self, exploration_constant: float = 1.41) -> float:
        """UCB1 value for action selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (round over)"""
        # Round is over if all hands are empty
        return all(len(hand) == 0 for hand in self.state.hands)


class MCTSAgent:
    """MCTS agent with determinization for hidden information"""
    
    def __init__(self, name: str = "MCTS", num_simulations: int = 1000, 
                 num_determinizations: int = 5, exploration_constant: float = 1.41):
        """
        Initialize MCTS agent
        
        Args:
            name: Agent name
            num_simulations: Number of MCTS simulations per decision
            num_determinizations: Number of opponent hand samples
            exploration_constant: UCB exploration parameter (√2 ≈ 1.41)
        """
        self.name = name
        self.num_simulations = num_simulations
        self.num_determinizations = num_determinizations
        self.exploration_constant = exploration_constant
        
        # For rollouts, we'll use simple heuristics
        from agents.heuristic_agent import HeuristicAgent
        self.rollout_agent = HeuristicAgent("Rollout")
    
    def bid(self, hand: List[Card], trump_suit: Optional[Suit], round_num: int,
            player_idx: int, bids_so_far: List) -> int:
        """
        Choose bid using MCTS
        
        For now, use heuristic bidding (MCTS for bidding is complex)
        TODO: Can improve by running MCTS over bid choices
        """
        # Simple heuristic for bidding (can improve later)
        return self.rollout_agent.bid(hand, trump_suit, round_num, player_idx, bids_so_far)
    
    def play(self, hand: List[Card], valid_cards: List[Card],
             current_trick: List[tuple[int, Card]], trump_suit: Optional[Suit],
             led_suit: Optional[Suit], player_idx: int, state: GameState) -> Card:
        """
        Choose card to play using MCTS with determinization
        """
        # Run MCTS over multiple determinizations
        action_values = {card: 0.0 for card in valid_cards}
        
        for _ in range(self.num_determinizations):
            # Sample possible opponent hands
            determinized_state = self._determinize_state(state, player_idx)
            
            # Run MCTS on this determinization
            best_card, value = self._mcts_search(determinized_state, player_idx, valid_cards)
            action_values[best_card] += value
        
        # Choose action with highest average value
        best_card = max(action_values.keys(), key=lambda c: action_values[c])
        return best_card
    
    def _determinize_state(self, state: GameState, player_idx: int) -> GameState:
        """
        Sample a possible world consistent with observations
        
        We know:
        - Our own hand
        - Cards already played
        - Trump card
        
        We don't know:
        - Opponent hands
        """
        # Create copy of state
        det_state = state.copy()
        
        # Collect unknown cards
        my_hand = state.hands[player_idx]
        all_cards = set(range(60))  # Simplified: use card IDs
        
        # For simplicity, just randomly deal unknown cards to opponents
        # TODO: Weight by bids and cards played (more sophisticated)
        unknown_cards = []
        
        # Get all cards
        from game.deck import Deck
        deck = Deck.create_standard_deck()
        
        # Remove known cards
        known_cards = set(my_hand)
        for hand in state.hands:
            if hand != my_hand:
                known_cards.update(hand)
        
        # Cards played in current trick
        for _, card in state.current_trick:
            known_cards.add(card)
        
        if state.trump_card:
            known_cards.add(state.trump_card)
        
        # Unknown cards
        unknown_cards = [c for c in deck if c not in known_cards]
        random.shuffle(unknown_cards)
        
        # Redistribute to opponents
        idx = 0
        for i in range(state.num_players):
            if i != player_idx:
                hand_size = len(state.hands[i])
                det_state.hands[i] = unknown_cards[idx:idx + hand_size]
                idx += hand_size
        
        return det_state
    
    def _mcts_search(self, state: GameState, player_idx: int, 
                     valid_cards: List[Card]) -> Tuple[Card, float]:
        """
        Run MCTS from current state
        
        Returns:
            (best_card, value)
        """
        # Create root node
        root = MCTSNode(
            state=state.copy(),
            player_idx=player_idx,
            untried_actions=valid_cards.copy()
        )
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_state = state.copy()
            
            # Selection: traverse tree using UCB
            while not node.is_terminal() and node.is_fully_expanded():
                node = self._select_child(node)
                # Apply action to simulation state
                if node.action:
                    sim_state = self._apply_action(sim_state, node.action, node.player_idx)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node, sim_state)
                if node.action:
                    sim_state = self._apply_action(sim_state, node.action, node.player_idx)
            
            # Simulation: rollout to end of round
            value = self._simulate(sim_state, player_idx)
            
            # Backpropagation: update values
            self._backpropagate(node, value)
        
        # Choose best action (most visited)
        if not root.children:
            return valid_cards[0], 0.0
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action, best_child.value
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB value"""
        return max(node.children, key=lambda c: c.ucb1(self.exploration_constant))
    
    def _expand(self, node: MCTSNode, state: GameState) -> MCTSNode:
        """Expand node by trying an untried action"""
        action = node.untried_actions.pop()
        
        # Create child node
        child_state = self._apply_action(state.copy(), action, node.player_idx)
        child = MCTSNode(
            state=child_state,
            parent=node,
            action=action,
            player_idx=node.player_idx,
            untried_actions=self._get_valid_actions(child_state, node.player_idx)
        )
        
        node.children.append(child)
        return child
    
    def _apply_action(self, state: GameState, card: Card, player_idx: int) -> GameState:
        """
        Apply a card play action to the state
        
        Simplified version - just for simulation purposes
        """
        # This is a simplified state transition
        # For full game simulation, you'd need to track trick progression
        
        # Remove card from hand
        if card in state.hands[player_idx]:
            state.hands[player_idx].remove(card)
        
        return state
    
    def _get_valid_actions(self, state: GameState, player_idx: int) -> List[Card]:
        """Get valid cards for current player"""
        # Simplified - in real version, would check suit following rules
        return state.hands[player_idx].copy()
    
    def _simulate(self, state: GameState, player_idx: int) -> float:
        """
        Simulate rest of round using rollout policy
        
        Returns reward for player_idx
        """
        # Simplified simulation: use heuristic agent for rollout
        # In real version, would complete the round and calculate score
        
        # For now, return a simple heuristic value based on current state
        my_bid = state.bids[player_idx]
        my_tricks = state.tricks_won[player_idx]
        
        # Estimate final score
        # Positive if on track, negative if off track
        if my_bid == my_tricks:
            return 20 + 10 * my_bid
        else:
            return -10 * abs(my_bid - my_tricks)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree"""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent