"""
Comparison Study: Belief-based MCTS vs Uniform Sampling MCTS

This script compares:
1. BeliefMCTS: Uses particle filtering to weight opponent hand samples
2. SimpleMCTS: Uses uniform random sampling
3. Heuristic: Baseline

The key question: Does modeling opponent hands based on observed bids help?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.simulator import WizardGame
from agents.heuristic_agent import HeuristicAgent
from agents.mcts_simple import SimpleMCTSAgent
from agents.belief_mcts_agent import BeliefMCTSAgent


def run_belief_comparison(num_games: int = 20):
    """
    Compare belief-based MCTS against uniform sampling MCTS.
    """
    print("=" * 60)
    print("BELIEF STATE COMPARISON: Particle Filtering vs Uniform Sampling")
    print("=" * 60)
    print(f"\nRunning {num_games} games with 4 agents:")
    print("  - BeliefMCTS: Particle filtering weighted by observed bids")
    print("  - SimpleMCTS: Uniform random sampling (current best)")
    print("  - Heuristic1: Baseline")
    print("  - Heuristic2: Baseline (for 4 players)")
    print("\nThis tests whether belief modeling improves performance.\n")

    # Track results
    wins = {"BeliefMCTS": 0, "SimpleMCTS": 0, "Heuristic1": 0, "Heuristic2": 0}
    total_scores = {"BeliefMCTS": 0, "SimpleMCTS": 0, "Heuristic1": 0, "Heuristic2": 0}

    for game_num in range(num_games):
        agents = [
            BeliefMCTSAgent("BeliefMCTS", num_simulations=100, bid_simulations=50, num_particles=100),
            SimpleMCTSAgent("SimpleMCTS", num_simulations=100, bid_simulations=50, use_mcts_bidding=True),
            HeuristicAgent("Heuristic1"),
            HeuristicAgent("Heuristic2")
        ]

        game = WizardGame(num_players=4)
        game.verbose = False
        final_scores = game.play_full_game(agents)

        # Track scores
        for i, agent in enumerate(agents):
            total_scores[agent.name] += final_scores[i]

        # Determine winner
        winner_idx = final_scores.index(max(final_scores))
        winner_name = agents[winner_idx].name
        wins[winner_name] += 1

        print(f"Game {game_num + 1:2d}: "
              f"Belief={final_scores[0]:4d}, "
              f"Simple={final_scores[1]:4d}, "
              f"Heur1={final_scores[2]:4d}, "
              f"Heur2={final_scores[3]:4d} | "
              f"Winner: {winner_name}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nWin Rates:")
    for name in ["BeliefMCTS", "SimpleMCTS", "Heuristic1", "Heuristic2"]:
        pct = 100 * wins[name] / num_games
        print(f"  {name:12s}: {wins[name]:2d}/{num_games} ({pct:5.1f}%)")

    print("\nAverage Scores:")
    for name in ["BeliefMCTS", "SimpleMCTS", "Heuristic1", "Heuristic2"]:
        avg = total_scores[name] / num_games
        print(f"  {name:12s}: {avg:+7.1f}")

    # Key comparison
    print("\n" + "-" * 60)
    print("KEY COMPARISON: Belief State vs Uniform Sampling")
    print("-" * 60)

    belief_avg = total_scores["BeliefMCTS"] / num_games
    simple_avg = total_scores["SimpleMCTS"] / num_games
    improvement = belief_avg - simple_avg

    print(f"\n  BeliefMCTS (particle filtering): {belief_avg:+.1f} avg score")
    print(f"  SimpleMCTS (uniform sampling):   {simple_avg:+.1f} avg score")
    print(f"  -----------------------------------------")
    print(f"  Difference: {improvement:+.1f} points")

    belief_wins = wins["BeliefMCTS"]
    simple_wins = wins["SimpleMCTS"]
    print(f"\n  BeliefMCTS win rate: {100*belief_wins/num_games:.1f}%")
    print(f"  SimpleMCTS win rate: {100*simple_wins/num_games:.1f}%")

    print("\nComparison complete!")

    return total_scores, wins


def run_head_to_head(num_games: int = 30):
    """
    Direct head-to-head: 2 BeliefMCTS vs 2 SimpleMCTS
    """
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD: 2 BeliefMCTS vs 2 SimpleMCTS")
    print("=" * 60)
    print(f"\nRunning {num_games} games...\n")

    belief_wins = 0
    simple_wins = 0
    belief_total = 0
    simple_total = 0

    for game_num in range(num_games):
        agents = [
            BeliefMCTSAgent("Belief1", num_simulations=100, bid_simulations=50, num_particles=100),
            SimpleMCTSAgent("Simple1", num_simulations=100, bid_simulations=50, use_mcts_bidding=True),
            BeliefMCTSAgent("Belief2", num_simulations=100, bid_simulations=50, num_particles=100),
            SimpleMCTSAgent("Simple2", num_simulations=100, bid_simulations=50, use_mcts_bidding=True),
        ]

        game = WizardGame(num_players=4)
        game.verbose = False
        final_scores = game.play_full_game(agents)

        # Aggregate by type
        belief_score = final_scores[0] + final_scores[2]
        simple_score = final_scores[1] + final_scores[3]

        belief_total += belief_score
        simple_total += simple_score

        if belief_score > simple_score:
            belief_wins += 1
        elif simple_score > belief_score:
            simple_wins += 1

        winner = "Belief" if belief_score > simple_score else "Simple" if simple_score > belief_score else "Tie"
        print(f"Game {game_num + 1:2d}: Belief={belief_score:4d}, Simple={simple_score:4d} | {winner}")

    print("\n" + "-" * 60)
    print("HEAD-TO-HEAD RESULTS")
    print("-" * 60)
    print(f"\n  BeliefMCTS wins: {belief_wins}/{num_games} ({100*belief_wins/num_games:.1f}%)")
    print(f"  SimpleMCTS wins: {simple_wins}/{num_games} ({100*simple_wins/num_games:.1f}%)")
    print(f"  Ties: {num_games - belief_wins - simple_wins}")
    print(f"\n  BeliefMCTS avg total: {belief_total/num_games:+.1f}")
    print(f"  SimpleMCTS avg total: {simple_total/num_games:+.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare belief-based MCTS vs uniform sampling")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games to run (default: 20)")
    parser.add_argument("--head-to-head", action="store_true",
                        help="Run head-to-head comparison (2v2)")
    args = parser.parse_args()

    run_belief_comparison(args.games)

    if args.head_to_head:
        run_head_to_head(args.games)
