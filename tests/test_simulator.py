import sys
sys.path.append('..')

from game.simulator import WizardGame
from agents.random_agent import RandomAgent

def test_full_game():
    """Test playing a complete game"""
    
    # Create 4 random agents
    agents = [RandomAgent(f"Player{i}") for i in range(4)]
    
    # Play a game
    game = WizardGame(num_players=4)
    game.verbose = True  # See what's happening
    
    final_scores = game.play_full_game(agents)
    
    print(f"\n{'='*50}")
    print("GAME OVER!")
    print(f"Final Scores: {final_scores}")
    print(f"Winner: Player {final_scores.index(max(final_scores))} with {max(final_scores)} points")
    
    # Basic sanity checks
    assert len(final_scores) == 4
    assert all(isinstance(score, int) for score in final_scores)
    print("\n✅ Game completed successfully!")

def test_quick_game():
    """Test a game without verbose output"""
    agents = [RandomAgent(f"P{i}") for i in range(4)]
    game = WizardGame(num_players=4)
    game.verbose = False
    
    scores = game.play_full_game(agents)
    print(f"Quick game result: {scores}")
    print("✅ Quick game works!")

if __name__ == "__main__":
    print("Testing full game with verbose output...")
    test_full_game()
    
    print("\n" + "="*50)
    print("\nTesting quick game...")
    test_quick_game()