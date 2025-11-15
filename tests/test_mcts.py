import sys
sys.path.append('..')

from game.simulator import WizardGame
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.mcts_simple import SimpleMCTSAgent

def test_mcts_single_game():
    """Test MCTS in a single game"""
    
    agents = [
        SimpleMCTSAgent("MCTS", num_simulations=50),
        HeuristicAgent("Heuristic"),
        RandomAgent("Random1"),
        RandomAgent("Random2")
    ]
    
    game = WizardGame(num_players=4)
    game.verbose = True
    
    scores = game.play_full_game(agents)
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS:")
    for i, agent in enumerate(agents):
        print(f"{agent.name}: {scores[i]} points")
    
    winner_idx = scores.index(max(scores))
    print(f"\nWinner: {agents[winner_idx].name}")
    
    print("\n✅ MCTS test complete!")

def test_mcts_vs_heuristic():
    """Run multiple games: MCTS vs Heuristic"""
    
    num_games = 10
    wins = {"MCTS": 0, "Heuristic": 0, "Random": 0}
    scores = {"MCTS": [], "Heuristic": [], "Random": []}
    
    print(f"Running {num_games} games: MCTS vs Heuristic vs Random...")
    print("(This may take a minute...)\n")
    
    for game_num in range(num_games):
        agents = [
            SimpleMCTSAgent("MCTS", num_simulations=100),
            HeuristicAgent("Heuristic"),
            RandomAgent("Random1"),
            RandomAgent("Random2")
        ]
        
        game = WizardGame(num_players=4)
        game.verbose = False
        final_scores = game.play_full_game(agents)
        
        # Track results
        scores["MCTS"].append(final_scores[0])
        scores["Heuristic"].append(final_scores[1])
        scores["Random"].append((final_scores[2] + final_scores[3]) / 2)
        
        winner_idx = final_scores.index(max(final_scores))
        if winner_idx == 0:
            wins["MCTS"] += 1
        elif winner_idx == 1:
            wins["Heuristic"] += 1
        else:
            wins["Random"] += 1
        
        print(f"Game {game_num+1}: MCTS={final_scores[0]}, "
              f"Heuristic={final_scores[1]}, "
              f"Random avg={(final_scores[2]+final_scores[3])/2:.1f} | "
              f"Winner: {agents[winner_idx].name}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS OVER {num_games} GAMES:")
    print(f"\nWin rates:")
    print(f"  MCTS:      {wins['MCTS']}/{num_games} ({100*wins['MCTS']/num_games:.1f}%)")
    print(f"  Heuristic: {wins['Heuristic']}/{num_games} ({100*wins['Heuristic']/num_games:.1f}%)")
    print(f"  Random:    {wins['Random']}/{num_games} ({100*wins['Random']/num_games:.1f}%)")
    
    print(f"\nAverage scores:")
    print(f"  MCTS:      {sum(scores['MCTS'])/num_games:.1f}")
    print(f"  Heuristic: {sum(scores['Heuristic'])/num_games:.1f}")
    print(f"  Random:    {sum(scores['Random'])/num_games:.1f}")
    
    print("\n✅ Tournament complete!")

if __name__ == "__main__":
    print("Testing single game with verbose output...\n")
    test_mcts_single_game()
    
    print("\n" + "="*50 + "\n")
    
    print("Running tournament...\n")
    test_mcts_vs_heuristic()