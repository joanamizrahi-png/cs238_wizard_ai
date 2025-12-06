"""
MCTS agent with belief state (particle filtering) for opponent hand inference.

Uses proper MCTS with UCB1 for card play decisions, combined with:
1. Particles weighted by bid consistency (Gaussian weighting)
2. Particles updated during card play (cards revealed are eliminated)
3. Card play consistency checking (did opponent follow suit correctly?)
"""
import random
import math
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from game.deck import Card, Suit, Deck
from game.simulator import GameState
from game.rules import WizardRules
from agents.heuristic_agent import HeuristicAgent


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    action: Optional[Card] = None
    parent: Optional['MCTSNode'] = None
    children: Dict[Card, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    total_value: float = 0.0

    @property
    def q_value(self) -> float:
        """Average value (Q) of this node"""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value: Q + C * sqrt(ln(N_parent) / N_child)"""
        if self.visits == 0:
            return float('inf')
        if self.parent is None or self.parent.visits == 0:
            return self.q_value
        return self.q_value + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1"""
        return max(self.children.values(),
                   key=lambda c: c.ucb1(exploration_constant))

    def best_action(self) -> Card:
        """Return action with highest visit count (most robust)"""
        return max(self.children.keys(),
                   key=lambda a: self.children[a].visits)


class BeliefMCTSAgent:
    """
    MCTS agent that uses particle filtering to model opponent hands.

    Combines:
    - MCTS with UCB1 for card play decisions
    - Particle filtering for belief state over opponent hands
    - Belief updates from both bids and card plays
    """

    def __init__(self, name: str = "BeliefMCTS", num_simulations: int = 100,
                 bid_simulations: int = 50, num_particles: int = 100,
                 use_play_updates: bool = True, exploration_constant: float = 1.414):
        self.name = name
        self.num_simulations = num_simulations
        self.bid_simulations = bid_simulations
        self.num_particles = num_particles
        self.use_play_updates = use_play_updates
        self.exploration_constant = exploration_constant
        self.heuristic = HeuristicAgent()

        # Belief state: particles representing possible opponent hands
        self.particles: List[Tuple[List[List[Card]], float]] = []

        # Track cards played this round (for belief updates)
        self.cards_played_this_round: List[Tuple[int, Card, Optional[Suit]]] = []
        self.current_round: int = -1

    # =========================================================================
    # BIDDING (Monte Carlo rollouts with belief-weighted sampling)
    # =========================================================================

    def bid(self, hand: List[Card], trump_suit: Optional[Suit], round_num: int,
            player_idx: int, bids_so_far: List) -> int:
        """
        Use Monte Carlo rollouts with belief-weighted sampling to evaluate bids.
        """
        num_players = max(4, player_idx + 1)

        # Reset round tracking
        if round_num != self.current_round:
            self.current_round = round_num
            self.cards_played_this_round = []

        # Generate particles (weighted samples of opponent hands)
        self._generate_particles(
            hand, trump_suit, round_num, player_idx, num_players, bids_so_far
        )

        bid_scores = {}

        for candidate_bid in range(round_num + 1):
            total_score = 0.0
            total_weight = 0.0

            for _ in range(self.bid_simulations):
                # Sample a particle according to weights
                sampled_hands, weight = self._sample_particle()

                tricks_won = self._simulate_round_for_bid(
                    sampled_hands, trump_suit, round_num, player_idx,
                    num_players, bids_so_far, candidate_bid
                )

                score = WizardRules.score_round(candidate_bid, tricks_won)
                total_score += score * weight
                total_weight += weight

            if total_weight > 0:
                bid_scores[candidate_bid] = total_score / total_weight
            else:
                bid_scores[candidate_bid] = 0

        best_bid = max(bid_scores.keys(), key=lambda b: bid_scores[b])
        return best_bid

    def _generate_particles(self, my_hand: List[Card], trump_suit: Optional[Suit],
                            round_num: int, my_idx: int, num_players: int,
                            bids_so_far: List):
        """Generate weighted particles based on observed bids."""
        self.particles = []

        all_cards = Deck.create_standard_deck()
        remaining_cards = [c for c in all_cards if c not in my_hand]

        for _ in range(self.num_particles):
            shuffled = remaining_cards.copy()
            random.shuffle(shuffled)

            hands = []
            card_idx = 0
            for p in range(num_players):
                if p == my_idx:
                    hands.append(my_hand.copy())
                else:
                    hand = shuffled[card_idx:card_idx + round_num]
                    hands.append(hand)
                    card_idx += round_num

            # Calculate weight based on observed bids
            weight = self._calculate_particle_weight(
                hands, trump_suit, round_num, bids_so_far
            )
            self.particles.append((hands, weight))

        # Normalize weights
        total_weight = sum(w for _, w in self.particles)
        if total_weight > 0:
            self.particles = [(h, w / total_weight) for h, w in self.particles]
        else:
            uniform_weight = 1.0 / len(self.particles)
            self.particles = [(h, uniform_weight) for h, _ in self.particles]

    def _calculate_particle_weight(self, hands: List[List[Card]],
                                   trump_suit: Optional[Suit], round_num: int,
                                   bids_so_far: List) -> float:
        """
        Calculate likelihood weight based on observed bids.
        Uses Gaussian weighting: closer to expected bid = higher weight.
        """
        if not bids_so_far:
            return 1.0

        weight = 1.0
        for player_idx, observed_bid in enumerate(bids_so_far):
            if observed_bid is None:
                continue

            expected_bid = self._estimate_expected_bid(
                hands[player_idx], trump_suit, round_num
            )
            diff = abs(observed_bid - expected_bid)
            weight *= math.exp(-0.5 * diff * diff)

        return weight

    def _estimate_expected_bid(self, hand: List[Card], trump_suit: Optional[Suit],
                               round_num: int) -> float:
        """Estimate expected bid for a hand (based on heuristic logic)."""
        likely_tricks = 0.0

        # Wizards always win
        wizards = [c for c in hand if c.suit == Suit.WIZARD]
        likely_tricks += len(wizards)

        # High trump cards
        if trump_suit:
            trumps = [c for c in hand if c.suit == trump_suit]
            high_trumps = [c for c in trumps if c.rank >= 13 or c.rank == 1]
            likely_tricks += len(high_trumps) * 0.6

        # Aces in non-trump suits
        for suit in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]:
            if suit != trump_suit:
                aces = [c for c in hand if c.suit == suit and c.rank == 1]
                likely_tricks += len(aces) * 0.3

        return likely_tricks * 0.5  # Conservative factor

    def _sample_particle(self) -> Tuple[List[List[Card]], float]:
        """Sample a particle according to weights."""
        r = random.random()
        cumulative = 0.0

        for hands, weight in self.particles:
            cumulative += weight
            if r <= cumulative:
                return hands, weight

        return self.particles[-1]

    def _simulate_round_for_bid(self, hands: List[List[Card]], trump_suit: Optional[Suit],
                                round_num: int, my_idx: int, num_players: int,
                                bids_so_far: List, my_bid: int) -> int:
        """Simulate a round to estimate tricks won."""
        all_bids = list(bids_so_far) + [my_bid]
        for p in range(len(all_bids), num_players):
            opponent_bid = self.heuristic.bid(
                hands[p], trump_suit, round_num, p, all_bids
            )
            all_bids.append(opponent_bid)

        tricks_won = [0] * num_players
        sim_hands = [h.copy() for h in hands]
        trick_leader = 0

        for _ in range(round_num):
            trick_winner = self._simulate_trick_heuristic(
                sim_hands, trick_leader, trump_suit, all_bids, tricks_won, num_players
            )
            tricks_won[trick_winner] += 1
            trick_leader = trick_winner

        return tricks_won[my_idx]

    # =========================================================================
    # CARD PLAY (MCTS with UCB1 + belief-weighted determinization)
    # =========================================================================

    def play(self, hand: List[Card], valid_cards: List[Card],
             current_trick: List[Tuple[int, Card]], trump_suit: Optional[Suit],
             led_suit: Optional[Suit], player_idx: int, state: GameState) -> Card:
        """
        Choose card using MCTS with UCB1 and belief-weighted sampling.
        """
        # Update beliefs based on cards played in current trick
        if self.use_play_updates:
            self._update_beliefs_from_trick(current_trick, led_suit, player_idx, state)

        if len(valid_cards) == 1:
            return valid_cards[0]

        # Run MCTS with belief-weighted determinizations
        root = MCTSNode()

        # Initialize children for all valid actions
        for card in valid_cards:
            root.children[card] = MCTSNode(action=card, parent=root)

        for _ in range(self.num_simulations):
            # Determinize using belief-weighted sampling
            determinized_state = self._determinize_state_with_belief(state, player_idx)

            # Run one MCTS iteration
            self._mcts_iteration(root, determinized_state, player_idx)

        # Return most visited action
        return root.best_action()

    def _determinize_state_with_belief(self, state: GameState, player_idx: int) -> GameState:
        """
        Create a determinized state by sampling from belief distribution.
        """
        det_state = state.copy()

        if self.particles:
            # Sample from belief distribution
            sampled_hands, _ = self._sample_particle()

            # Use sampled hands but adjust for cards already played
            for p in range(state.num_players):
                if p == player_idx:
                    det_state.hands[p] = state.hands[player_idx].copy()
                else:
                    # Filter out cards that have been played
                    played_cards = set(card for _, card, _ in self.cards_played_this_round)
                    det_state.hands[p] = [c for c in sampled_hands[p] if c not in played_cards]
        else:
            # Fallback to uniform sampling
            my_hand = state.hands[player_idx]
            cards_in_trick = [card for _, card in state.current_trick]
            known_cards = set(my_hand) | set(cards_in_trick)

            all_cards = Deck.create_standard_deck()
            unknown_cards = [c for c in all_cards if c not in known_cards]
            if state.trump_card:
                unknown_cards = [c for c in unknown_cards if c != state.trump_card]

            random.shuffle(unknown_cards)

            card_idx = 0
            for p in range(state.num_players):
                if p == player_idx:
                    det_state.hands[p] = my_hand.copy()
                else:
                    num_cards = len(state.hands[p])
                    det_state.hands[p] = unknown_cards[card_idx:card_idx + num_cards]
                    card_idx += num_cards

        return det_state

    def _mcts_iteration(self, root: MCTSNode, state: GameState, player_idx: int):
        """Run one MCTS iteration: Select -> Simulate -> Backpropagate"""
        # Selection
        selected_child = root.best_child(self.exploration_constant)
        selected_action = selected_child.action

        # Simulation
        value = self._simulate_from_action(selected_action, state, player_idx)

        # Backpropagation
        selected_child.visits += 1
        selected_child.total_value += value
        root.visits += 1

    def _simulate_from_action(self, action: Card, state: GameState,
                              player_idx: int) -> float:
        """Simulate playing action and complete the round."""
        sim_state = state.copy()
        current_trick = list(sim_state.current_trick)
        current_trick.append((player_idx, action))

        sim_hands = [h.copy() for h in sim_state.hands]
        if action in sim_hands[player_idx]:
            sim_hands[player_idx].remove(action)

        led_suit = self._get_led_suit(current_trick)

        # Complete current trick
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

        tricks_won = sim_state.tricks_won.copy()
        tricks_won[trick_winner] += 1

        # Simulate remaining tricks
        trick_leader = trick_winner
        tricks_remaining = len(sim_hands[player_idx])

        for _ in range(tricks_remaining):
            trick_result = self._simulate_trick_heuristic(
                sim_hands, trick_leader, sim_state.trump_suit,
                sim_state.bids, tricks_won, sim_state.num_players
            )
            tricks_won[trick_result] += 1
            trick_leader = trick_result

        # Return score
        my_bid = sim_state.bids[player_idx]
        my_tricks = tricks_won[player_idx]
        return WizardRules.score_round(my_bid, my_tricks)

    def _get_led_suit(self, current_trick: List[Tuple[int, Card]]) -> Optional[Suit]:
        """Determine led suit from current trick."""
        for _, card in current_trick:
            if card.suit not in [Suit.JESTER, Suit.WIZARD]:
                return card.suit
        return None

    # =========================================================================
    # BELIEF UPDATES FROM CARD PLAY
    # =========================================================================

    def _update_beliefs_from_trick(self, current_trick: List[Tuple[int, Card]],
                                   led_suit: Optional[Suit], my_idx: int,
                                   state: GameState):
        """
        Update particle weights based on cards played.

        When an opponent plays a card:
        1. They must have that card (eliminate inconsistent particles)
        2. If they didn't follow suit, they must not have that suit
        """
        if not self.particles:
            return

        for opp_idx, card in current_trick:
            if opp_idx == my_idx:
                continue

            play_key = (opp_idx, card, led_suit)
            if play_key in self.cards_played_this_round:
                continue
            self.cards_played_this_round.append(play_key)

            # Update particle weights
            new_particles = []
            for hands, weight in self.particles:
                new_weight = weight * self._card_play_consistency(
                    hands[opp_idx], card, led_suit
                )
                if new_weight > 0:
                    updated_hands = [h.copy() for h in hands]
                    if card in updated_hands[opp_idx]:
                        updated_hands[opp_idx].remove(card)
                    new_particles.append((updated_hands, new_weight))

            if len(new_particles) < self.num_particles // 4:
                self._regenerate_particles_with_constraints(state, my_idx)
            elif new_particles:
                total_weight = sum(w for _, w in new_particles)
                if total_weight > 0:
                    self.particles = [(h, w / total_weight) for h, w in new_particles]

    def _card_play_consistency(self, hand: List[Card], played_card: Card,
                               led_suit: Optional[Suit]) -> float:
        """Check if playing this card is consistent with the hand."""
        if played_card not in hand:
            return 0.0

        if led_suit is not None and played_card.suit != led_suit:
            if played_card.suit not in [Suit.WIZARD, Suit.JESTER]:
                has_led_suit = any(c.suit == led_suit for c in hand
                                   if c.suit not in [Suit.WIZARD, Suit.JESTER])
                if has_led_suit:
                    return 0.0

        return 1.0

    def _regenerate_particles_with_constraints(self, state: GameState, my_idx: int):
        """Regenerate particles when too many are eliminated."""
        num_players = state.num_players
        my_hand = state.hands[my_idx]
        round_num = state.round_num

        played_cards: Set[Card] = set()
        for opp_idx, card, _ in self.cards_played_this_round:
            played_cards.add(card)

        my_cards = set(my_hand)
        all_cards = Deck.create_standard_deck()
        remaining_cards = [c for c in all_cards
                          if c not in my_cards and c not in played_cards]

        self.particles = []
        for _ in range(self.num_particles):
            shuffled = remaining_cards.copy()
            random.shuffle(shuffled)

            hands = []
            card_idx = 0
            for p in range(num_players):
                if p == my_idx:
                    hands.append(my_hand.copy())
                else:
                    cards_played_by_p = sum(1 for (idx, _, _) in self.cards_played_this_round
                                            if idx == p)
                    cards_remaining = round_num - cards_played_by_p
                    hand = shuffled[card_idx:card_idx + cards_remaining]
                    hands.append(hand)
                    card_idx += cards_remaining

            weight = self._calculate_particle_weight(
                hands, state.trump_suit, round_num,
                [b for b in state.bids if b is not None]
            )
            self.particles.append((hands, weight))

        total_weight = sum(w for _, w in self.particles)
        if total_weight > 0:
            self.particles = [(h, w / total_weight) for h, w in self.particles]
        else:
            uniform = 1.0 / len(self.particles)
            self.particles = [(h, uniform) for h, _ in self.particles]

    # =========================================================================
    # HEURISTIC HELPERS
    # =========================================================================

    def _simulate_trick_heuristic(self, hands: List[List[Card]], leader: int,
                                  trump_suit: Optional[Suit], bids: List[int],
                                  tricks_won: List[int], num_players: int) -> int:
        """Simulate a single trick using heuristic play."""
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

            if i == 0 and card.suit not in [Suit.JESTER, Suit.WIZARD]:
                led_suit = card.suit
            elif i == 1 and led_suit is None and card.suit not in [Suit.JESTER, Suit.WIZARD]:
                led_suit = card.suit

        return WizardRules.determine_trick_winner(current_trick, led_suit, trump_suit)

    def _heuristic_play(self, valid_cards: List[Card], current_trick: List,
                        trump_suit: Optional[Suit], led_suit: Optional[Suit],
                        bid: int, tricks_won: int) -> Card:
        """Simple heuristic card selection for simulation."""
        want_to_win = tricks_won < bid

        if want_to_win:
            wizards = [c for c in valid_cards if c.suit == Suit.WIZARD]
            if wizards:
                return wizards[0]

            if not current_trick:
                non_jesters = [c for c in valid_cards if c.suit != Suit.JESTER]
                if non_jesters:
                    return max(non_jesters, key=lambda c: (c.suit == trump_suit, c.rank))
                return valid_cards[0]

            if trump_suit:
                trumps = [c for c in valid_cards if c.suit == trump_suit]
                if trumps:
                    return max(trumps, key=lambda c: c.rank)

            non_jesters = [c for c in valid_cards if c.suit != Suit.JESTER]
            if non_jesters:
                return max(non_jesters, key=lambda c: c.rank)
            return valid_cards[0]
        else:
            jesters = [c for c in valid_cards if c.suit == Suit.JESTER]
            if jesters:
                return jesters[0]

            non_wizards = [c for c in valid_cards if c.suit != Suit.WIZARD]
            if non_wizards:
                return min(non_wizards, key=lambda c: c.rank)
            return valid_cards[0]
