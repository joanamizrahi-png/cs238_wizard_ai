"""
MCTS Card Play Study: Does MCTS card play help vs pure heuristic?

Both agents use heuristic bidding, so the only difference is card play strategy.
This isolates whether MCTS tree search for card play provides any benefit.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.simulator import WizardGame
from agents.heuristic_agent import HeuristicAgent
from agents.mcts_simple import SimpleMCTSAgent


def run_mcts_cardplay_study(num_games: int = 50):
    """
    Compare MCTS card play vs heuristic card play.
    Both use heuristic bidding to isolate the card play difference.
    """
    print("=" * 60)
    print("MCTS CARD PLAY STUDY: Does MCTS help for card play?")
    print("=" * 60)
    print(f"\nRunning {num_games} games with 4 agents:")
    print("  - MCTS_play: MCTS card play + heuristic bidding")
    print("  - Heuristic1: Heuristic card play + heuristic bidding")
    print("  - Heuristic2: Heuristic (baseline)")
    print("  - Heuristic3: Heuristic (baseline)")
    print("\nThis tests whether MCTS card play alone provides benefit.\n")

    # Track results
    wins = {"MCTS_play": 0, "Heuristic1": 0, "Heuristic2": 0, "Heuristic3": 0}
    total_scores = {"MCTS_play": 0, "Heuristic1": 0, "Heuristic2": 0, "Heuristic3": 0}

    for game_num in range(num_games):
        agents = [
            SimpleMCTSAgent("MCTS_play", num_simulations=100, bid_simulations=50, use_mcts_bidding=False),
            HeuristicAgent("Heuristic1"),
            HeuristicAgent("Heuristic2"),
            HeuristicAgent("Heuristic3")
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
              f"MCTS={final_scores[0]:4d}, "
              f"H1={final_scores[1]:4d}, "
              f"H2={final_scores[2]:4d}, "
              f"H3={final_scores[3]:4d} | "
              f"Winner: {winner_name}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nWin Rates:")
    for name in ["MCTS_play", "Heuristic1", "Heuristic2", "Heuristic3"]:
        pct = 100 * wins[name] / num_games
        print(f"  {name:12s}: {wins[name]:2d}/{num_games} ({pct:5.1f}%)")

    print("\nAverage Scores:")
    for name in ["MCTS_play", "Heuristic1", "Heuristic2", "Heuristic3"]:
        avg = total_scores[name] / num_games
        print(f"  {name:12s}: {avg:+7.1f}")

    # Key comparison
    print("\n" + "-" * 60)
    print("KEY FINDING: MCTS Card Play vs Heuristic")
    print("-" * 60)

    mcts_avg = total_scores["MCTS_play"] / num_games
    heur_avg = (total_scores["Heuristic1"] + total_scores["Heuristic2"] + total_scores["Heuristic3"]) / (3 * num_games)

    print(f"\n  MCTS card play avg:      {mcts_avg:+.1f}")
    print(f"  Heuristic avg:           {heur_avg:+.1f}")
    print(f"  Difference:              {mcts_avg - heur_avg:+.1f}")

    mcts_wins = wins["MCTS_play"]
    expected_wins = num_games / 4  # 25% if random
    print(f"\n  MCTS win rate: {100*mcts_wins/num_games:.1f}% (expected 25% if no advantage)")

    return total_scores, wins


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MCTS card play vs heuristic")
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games to run (default: 50)")
    args = parser.parse_args()

    run_mcts_cardplay_study(args.games)
