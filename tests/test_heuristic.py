import sys
sys.path.append('..')

from game.simulator import WizardGame
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent

def test_heuristic_vs_random():
    """Test heuristic agent against random agents"""
    
    agents = [
        HeuristicAgent("Heuristic"),
        RandomAgent("Random1"),
        RandomAgent("Random2"),
        RandomAgent("Random3")
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
    
    print("\n✅ Heuristic vs Random test complete!")

def test_multiple_games():
    """Run multiple games to see average performance"""
    
    num_games = 10
    wins = {"Heuristic": 0, "Random": 0}
    total_scores = {"Heuristic": 0, "Random": 0}
    
    print(f"Running {num_games} games...")
    
    for game_num in range(num_games):
        agents = [
            HeuristicAgent("Heuristic"),
            RandomAgent("Random1"),
            RandomAgent("Random2"),
            RandomAgent("Random3")
        ]
        
        game = WizardGame(num_players=4)
        game.verbose = False
        scores = game.play_full_game(agents)
        
        # Track heuristic performance
        if scores[0] == max(scores):
            wins["Heuristic"] += 1
        else:
            wins["Random"] += 1
        
        total_scores["Heuristic"] += scores[0]
        total_scores["Random"] += sum(scores[1:]) / 3  # Average random score
        
        print(f"Game {game_num+1}: Heuristic={scores[0]}, "
              f"Random avg={sum(scores[1:])/3:.1f}")
    
    print(f"\n{'='*50}")
    print(f"RESULTS OVER {num_games} GAMES:")
    print(f"Heuristic wins: {wins['Heuristic']}/{num_games} "
          f"({100*wins['Heuristic']/num_games:.1f}%)")
    print(f"Heuristic avg score: {total_scores['Heuristic']/num_games:.1f}")
    print(f"Random avg score: {total_scores['Random']/num_games:.1f}")
    
    print("\n✅ Multiple games test complete!")

if __name__ == "__main__":
    print("Testing single game...")
    test_heuristic_vs_random()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing multiple games...")
    test_multiple_games()