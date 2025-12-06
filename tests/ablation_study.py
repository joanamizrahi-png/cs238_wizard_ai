"""
Ablation Study: Compare MCTS bidding vs Heuristic bidding

This script runs tournaments to isolate the impact of MCTS bidding.
We compare:
1. MCTS agent with MCTS bidding (full agent)
2. MCTS agent with heuristic bidding (card play only)
3. Pure heuristic agent (baseline)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.simulator import WizardGame
from agents.heuristic_agent import HeuristicAgent
from agents.mcts_simple import SimpleMCTSAgent


def run_ablation_study(num_games: int = 20):
    """
    Run ablation study comparing bidding strategies.

    All 4 players use different strategies to see head-to-head performance.
    """
    print("=" * 60)
    print("ABLATION STUDY: MCTS Bidding vs Heuristic Bidding")
    print("=" * 60)
    print(f"\nRunning {num_games} games with 4 agents:")
    print("  - MCTS_full: MCTS card play + MCTS bidding")
    print("  - MCTS_heur: MCTS card play + heuristic bidding")
    print("  - Heuristic: Heuristic card play + heuristic bidding")
    print("  - Heuristic2: Same as Heuristic (for 4 players)")
    print("\nThis isolates the impact of MCTS bidding specifically.\n")

    # Track results
    wins = {"MCTS_full": 0, "MCTS_heur": 0, "Heuristic": 0, "Heuristic2": 0}
    total_scores = {"MCTS_full": 0, "MCTS_heur": 0, "Heuristic": 0, "Heuristic2": 0}
    all_scores = {"MCTS_full": [], "MCTS_heur": [], "Heuristic": [], "Heuristic2": []}

    for game_num in range(num_games):
        # Create agents
        agents = [
            SimpleMCTSAgent("MCTS_full", num_simulations=100, bid_simulations=50, use_mcts_bidding=True),
            SimpleMCTSAgent("MCTS_heur", num_simulations=100, bid_simulations=50, use_mcts_bidding=False),
            HeuristicAgent("Heuristic"),
            HeuristicAgent("Heuristic2")
        ]

        game = WizardGame(num_players=4)
        game.verbose = False
        final_scores = game.play_full_game(agents)

        # Track scores
        for i, agent in enumerate(agents):
            total_scores[agent.name] += final_scores[i]
            all_scores[agent.name].append(final_scores[i])

        # Determine winner
        winner_idx = final_scores.index(max(final_scores))
        winner_name = agents[winner_idx].name
        wins[winner_name] += 1

        # Progress output
        print(f"Game {game_num + 1:2d}: "
              f"MCTS_full={final_scores[0]:4d}, "
              f"MCTS_heur={final_scores[1]:4d}, "
              f"Heuristic={final_scores[2]:4d}, "
              f"Heuristic2={final_scores[3]:4d} | "
              f"Winner: {winner_name}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nWin Rates:")
    for name in ["MCTS_full", "MCTS_heur", "Heuristic", "Heuristic2"]:
        pct = 100 * wins[name] / num_games
        print(f"  {name:12s}: {wins[name]:2d}/{num_games} ({pct:5.1f}%)")

    print("\nAverage Scores:")
    for name in ["MCTS_full", "MCTS_heur", "Heuristic", "Heuristic2"]:
        avg = total_scores[name] / num_games
        print(f"  {name:12s}: {avg:+7.1f}")

    # Key comparison
    print("\n" + "-" * 60)
    print("KEY COMPARISON: Impact of MCTS Bidding")
    print("-" * 60)

    mcts_full_avg = total_scores["MCTS_full"] / num_games
    mcts_heur_avg = total_scores["MCTS_heur"] / num_games
    improvement = mcts_full_avg - mcts_heur_avg

    print(f"\n  MCTS + MCTS bidding:      {mcts_full_avg:+.1f} avg score")
    print(f"  MCTS + heuristic bidding: {mcts_heur_avg:+.1f} avg score")
    print(f"  -----------------------------------------")
    print(f"  Improvement from MCTS bidding: {improvement:+.1f} points")

    mcts_full_wins = wins["MCTS_full"]
    mcts_heur_wins = wins["MCTS_heur"]
    print(f"\n  MCTS bidding win rate: {100*mcts_full_wins/num_games:.1f}%")
    print(f"  Heur bidding win rate: {100*mcts_heur_wins/num_games:.1f}%")

    print("\nAblation study complete!")

    return all_scores, wins


def run_extended_study(num_games: int = 50):
    """Run a larger study for more stable statistics."""
    print("\n" + "=" * 60)
    print(f"EXTENDED ABLATION STUDY ({num_games} games)")
    print("=" * 60)

    return run_ablation_study(num_games)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games to run (default: 20)")
    args = parser.parse_args()

    run_ablation_study(args.games)
