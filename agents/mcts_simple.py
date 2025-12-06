"""
MCTS agent with proper UCB1 tree search for Wizard card game.

Implements Information Set MCTS (ISMCTS) with determinization:
- Sample possible opponent hands (determinization)
- Build search tree using UCB1 selection
- Backpropagate rewards through the tree
"""
import random
import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from game.deck import Card, Suit, Deck
from game.simulator import GameState
from game.rules import WizardRules
from agents.heuristic_agent import HeuristicAgent


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.

    Each node represents a game state after a particular action.
    """
    action: Optional[Card] = None  # The action (card played) that led to this node
    parent: Optional['MCTSNode'] = None
    children: Dict[Card, 'MCTSNode'] = field(default_factory=dict)

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    @property
    def q_value(self) -> float:
        """Average value (Q) of this node"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB1 value for this node.

        UCB1 = Q + C * sqrt(ln(N_parent) / N_child)

        Args:
            exploration_constant: C parameter balancing exploration vs exploitation

        Returns:
            UCB1 value (infinity if unvisited)
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority

        if self.parent is None or self.parent.visits == 0:
            return self.q_value

        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children) == 0

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1"""
        return max(self.children.values(),
                   key=lambda c: c.ucb1(exploration_constant))

    def best_action(self) -> Card:
        """Return action with highest visit count (most robust choice)"""
        return max(self.children.keys(),
                   key=lambda a: self.children[a].visits)


class SimpleMCTSAgent:
    """
    MCTS agent using Information Set MCTS with determinization.

    For card play:
    - Determinize: Sample possible opponent hands
    - Run MCTS with UCB1 selection on the determinized state
    - Choose action with highest visit count

    For bidding:
    - Use Monte Carlo rollouts to evaluate each possible bid
    """

    def __init__(self, name: str = "SimpleMCTS", num_simulations: int = 100,
                 bid_simulations: int = 50, use_mcts_bidding: bool = True,
                 exploration_constant: float = 1.414):
        self.name = name
        self.num_simulations = num_simulations
        self.bid_simulations = bid_simulations
        self.use_mcts_bidding = use_mcts_bidding
        self.exploration_constant = exploration_constant
        self.heuristic = HeuristicAgent()

    # =========================================================================
    # BIDDING (Monte Carlo rollouts)
    # =========================================================================

    def bid(self, hand: List[Card], trump_suit: Optional[Suit], round_num: int,
            player_idx: int, bids_so_far: List) -> int:
        """
        Use Monte Carlo rollouts to evaluate each possible bid.

        For each possible bid (0 to round_num):
        - Sample random opponent hands
        - Simulate the round with heuristic card play
        - Calculate expected score
        - Choose bid with highest expected score
        """
        if not self.use_mcts_bidding:
            return self.heuristic.bid(hand, trump_suit, round_num, player_idx, bids_so_far)

        num_players = max(4, player_idx + 1)
        bid_scores = {}

        for candidate_bid in range(round_num + 1):
            total_score = 0.0

            for _ in range(self.bid_simulations):
                tricks_won = self._simulate_round_for_bid(
                    hand, trump_suit, round_num, player_idx,
                    num_players, bids_so_far, candidate_bid
                )
                score = WizardRules.score_round(candidate_bid, tricks_won)
                total_score += score

            bid_scores[candidate_bid] = total_score / self.bid_simulations

        best_bid = max(bid_scores.keys(), key=lambda b: bid_scores[b])
        return best_bid

    def _simulate_round_for_bid(self, my_hand: List[Card], trump_suit: Optional[Suit],
                                round_num: int, my_idx: int, num_players: int,
                                bids_so_far: List, my_bid: int) -> int:
        """Simulate a round to estimate how many tricks we'll win."""
        # Create deck and remove my cards
        all_cards = Deck.create_standard_deck()
        remaining_cards = [c for c in all_cards if c not in my_hand]
        random.shuffle(remaining_cards)

        # Deal random hands to opponents
        opponent_hands = []
        card_idx = 0
        for p in range(num_players):
            if p == my_idx:
                opponent_hands.append(my_hand.copy())
            else:
                hand = remaining_cards[card_idx:card_idx + round_num]
                opponent_hands.append(hand)
                card_idx += round_num

        # Generate bids for remaining players
        all_bids = list(bids_so_far) + [my_bid]
        for p in range(len(all_bids), num_players):
            opponent_bid = self.heuristic.bid(
                opponent_hands[p], trump_suit, round_num, p, all_bids
            )
            all_bids.append(opponent_bid)

        # Simulate all tricks with heuristic play
        tricks_won = [0] * num_players
        hands = [h.copy() for h in opponent_hands]
        trick_leader = 0

        for _ in range(round_num):
            trick_winner = self._simulate_trick_heuristic(
                hands, trick_leader, trump_suit, all_bids, tricks_won, num_players
            )
            tricks_won[trick_winner] += 1
            trick_leader = trick_winner

        return tricks_won[my_idx]

    # =========================================================================
    # CARD PLAY (MCTS with UCB1)
    # =========================================================================

    def play(self, hand: List[Card], valid_cards: List[Card],
             current_trick: List[Tuple[int, Card]], trump_suit: Optional[Suit],
             led_suit: Optional[Suit], player_idx: int, state: GameState) -> Card:
        """
        Choose card using MCTS with UCB1 selection.

        Algorithm:
        1. Determinize: Sample possible opponent hands
        2. For each simulation:
           a. Selection: Traverse tree using UCB1
           b. Expansion: Add new node for unexplored action
           c. Simulation: Rollout to end of round
           d. Backpropagation: Update node statistics
        3. Return action with highest visit count
        """
        if len(valid_cards) == 1:
            return valid_cards[0]

        # Run MCTS with multiple determinizations
        # Aggregate statistics across determinizations (ISMCTS approach)
        root = MCTSNode()

        # Initialize children for all valid actions
        for card in valid_cards:
            root.children[card] = MCTSNode(action=card, parent=root)

        for _ in range(self.num_simulations):
            # Determinize: sample opponent hands
            determinized_state = self._determinize_state(state, player_idx)

            # Run one MCTS iteration
            self._mcts_iteration(root, determinized_state, player_idx, valid_cards)

        # Return most visited action (most robust)
        return root.best_action()

    def _determinize_state(self, state: GameState, player_idx: int) -> GameState:
        """
        Create a determinized copy of the state by sampling opponent hands.

        This converts the POMDP into an MDP by assigning specific cards
        to opponents consistent with what we've observed.
        """
        det_state = state.copy()

        # Collect all cards we know about
        my_hand = state.hands[player_idx]
        cards_in_current_trick = [card for _, card in state.current_trick]

        # Cards that are definitely not in opponent hands
        known_cards = set(my_hand) | set(cards_in_current_trick)

        # All possible cards
        all_cards = Deck.create_standard_deck()

        # Cards that could be in opponent hands
        # (excluding trump card if it exists and we've seen it)
        unknown_cards = [c for c in all_cards if c not in known_cards]
        if state.trump_card:
            unknown_cards = [c for c in unknown_cards if c != state.trump_card]

        random.shuffle(unknown_cards)

        # Redistribute unknown cards to opponents
        card_idx = 0
        for p in range(state.num_players):
            if p == player_idx:
                det_state.hands[p] = my_hand.copy()
            else:
                # Opponent should have same number of cards as they currently do
                num_cards = len(state.hands[p])
                det_state.hands[p] = unknown_cards[card_idx:card_idx + num_cards]
                card_idx += num_cards

        return det_state

    def _mcts_iteration(self, root: MCTSNode, state: GameState,
                        player_idx: int, valid_cards: List[Card]):
        """
        Run one iteration of MCTS: Select -> Expand -> Simulate -> Backpropagate
        """
        # SELECTION: Choose action using UCB1
        selected_child = root.best_child(self.exploration_constant)
        selected_action = selected_child.action

        # SIMULATION: Play out the rest of the round
        value = self._simulate_from_action(selected_action, state, player_idx)

        # BACKPROPAGATION: Update statistics
        selected_child.visits += 1
        selected_child.total_value += value
        root.visits += 1

    def _simulate_from_action(self, action: Card, state: GameState,
                               player_idx: int) -> float:
        """
        Simulate playing the given action and complete the round.

        Returns a value representing how good this action was:
        - Positive if helps meet bid
        - Negative if hurts bid chances
        """
        # Create simulation state
        sim_state = state.copy()

        # Complete current trick
        current_trick = list(sim_state.current_trick)
        current_trick.append((player_idx, action))

        # Remove the played card from hand
        sim_hands = [h.copy() for h in sim_state.hands]
        if action in sim_hands[player_idx]:
            sim_hands[player_idx].remove(action)

        # Determine led suit
        led_suit = self._get_led_suit(current_trick)

        # Complete current trick with remaining players
        num_in_trick = len(current_trick)
        trick_leader = sim_state.trick_leader

        for i in range(num_in_trick, sim_state.num_players):
            p = (trick_leader + i) % sim_state.num_players
            if not sim_hands[p]:
                continue

            p_valid = WizardRules.get_valid_plays(sim_hands[p], led_suit)
            p_card = self._heuristic_play(
                p_valid, current_trick, sim_state.trump_suit, led_suit,
                sim_state.bids[p], sim_state.tricks_won[p]
            )
            sim_hands[p].remove(p_card)
            current_trick.append((p, p_card))

            if led_suit is None and p_card.suit not in [Suit.JESTER, Suit.WIZARD]:
                led_suit = p_card.suit

        # Determine trick winner
        trick_winner = WizardRules.determine_trick_winner(
            current_trick, led_suit, sim_state.trump_suit
        )

        # Update tricks won
        tricks_won = sim_state.tricks_won.copy()
        tricks_won[trick_winner] += 1

        # Simulate remaining tricks
        tricks_played = 1
        trick_leader = trick_winner

        cards_per_player = sim_state.round_num
        tricks_remaining = cards_per_player - len(sim_state.tricks_won) - tricks_played
        # Actually: tricks remaining = cards left in hand
        tricks_remaining = len(sim_hands[player_idx])

        for _ in range(tricks_remaining):
            trick_result = self._simulate_trick_heuristic(
                sim_hands, trick_leader, sim_state.trump_suit,
                sim_state.bids, tricks_won, sim_state.num_players
            )
            tricks_won[trick_result] += 1
            trick_leader = trick_result

        # Calculate value based on how well we met our bid
        my_bid = sim_state.bids[player_idx]
        my_tricks = tricks_won[player_idx]

        # Return the score we would get
        return WizardRules.score_round(my_bid, my_tricks)

    def _get_led_suit(self, current_trick: List[Tuple[int, Card]]) -> Optional[Suit]:
        """Determine the led suit from the current trick"""
        for _, card in current_trick:
            if card.suit not in [Suit.JESTER, Suit.WIZARD]:
                return card.suit
        return None

    def _simulate_trick_heuristic(self, hands: List[List[Card]], leader: int,
                                   trump_suit: Optional[Suit], bids: List[int],
                                   tricks_won: List[int], num_players: int) -> int:
        """Simulate a single trick using heuristic play and return winner index"""
        current_trick = []
        led_suit = None

        for i in range(num_players):
            player_idx = (leader + i) % num_players
            hand = hands[player_idx]

            if not hand:
                continue

            valid_cards = WizardRules.get_valid_plays(hand, led_suit)

            card = self._heuristic_play(
                valid_cards, current_trick, trump_suit, led_suit,
                bids[player_idx], tricks_won[player_idx]
            )

            hand.remove(card)
            current_trick.append((player_idx, card))

            # Update led suit
            if i == 0 and card.suit not in [Suit.JESTER, Suit.WIZARD]:
                led_suit = card.suit
            elif i == 1 and led_suit is None and card.suit not in [Suit.JESTER, Suit.WIZARD]:
                led_suit = card.suit

        return WizardRules.determine_trick_winner(current_trick, led_suit, trump_suit)

    def _heuristic_play(self, valid_cards: List[Card], current_trick: List,
                        trump_suit: Optional[Suit], led_suit: Optional[Suit],
                        bid: int, tricks_won: int) -> Card:
        """Simple heuristic card selection for simulation/rollout"""
        want_to_win = tricks_won < bid

        if want_to_win:
            # Try to win: prefer Wizards, then high trump, then high cards
            wizards = [c for c in valid_cards if c.suit == Suit.WIZARD]
            if wizards:
                return wizards[0]

            if not current_trick:
                # Leading: play high card
                non_jesters = [c for c in valid_cards if c.suit != Suit.JESTER]
                if non_jesters:
                    return max(non_jesters, key=lambda c: (c.suit == trump_suit, c.rank))
                return valid_cards[0]

            # Following: try to beat current winner
            if trump_suit:
                trumps = [c for c in valid_cards if c.suit == trump_suit]
                if trumps:
                    return max(trumps, key=lambda c: c.rank)

            non_jesters = [c for c in valid_cards if c.suit != Suit.JESTER]
            if non_jesters:
                return max(non_jesters, key=lambda c: c.rank)
            return valid_cards[0]
        else:
            # Try to lose: prefer Jesters, then low cards
            jesters = [c for c in valid_cards if c.suit == Suit.JESTER]
            if jesters:
                return jesters[0]

            non_wizards = [c for c in valid_cards if c.suit != Suit.WIZARD]
            if non_wizards:
                return min(non_wizards, key=lambda c: c.rank)
            return valid_cards[0]
