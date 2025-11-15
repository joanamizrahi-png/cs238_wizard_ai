import sys
sys.path.append('..')

from game.rules import WizardRules
from game.deck import Card, Suit

def test_wizard_wins():
    """Wizard should always win"""
    cards = [
        (0, Card(Suit.HEARTS, 13)),  # King of Hearts
        (1, Card(Suit.WIZARD, 0)),    # Wizard
        (2, Card(Suit.HEARTS, 1)),    # Ace of Hearts
    ]
    
    winner = WizardRules.determine_trick_winner(
        cards, led_suit=Suit.HEARTS, trump_suit=None
    )
    assert winner == 1
    print("✓ Wizard wins trick")

def test_trump_beats_led_suit():
    """Trump card should beat higher card of led suit"""
    cards = [
        (0, Card(Suit.HEARTS, 13)),  # King of Hearts (led)
        (1, Card(Suit.SPADES, 2)),   # 2 of Spades (trump)
    ]
    
    winner = WizardRules.determine_trick_winner(
        cards, led_suit=Suit.HEARTS, trump_suit=Suit.SPADES
    )
    assert winner == 1
    print("✓ Trump beats led suit")

def test_scoring():
    """Test score calculation"""
    assert WizardRules.score_round(bid=3, tricks_won=3) == 50  # 20 + 30
    assert WizardRules.score_round(bid=0, tricks_won=0) == 20
    assert WizardRules.score_round(bid=3, tricks_won=1) == -20  # Off by 2
    assert WizardRules.score_round(bid=2, tricks_won=5) == -30  # Off by 3
    print("✓ Scoring works correctly")

if __name__ == "__main__":
    test_wizard_wins()
    test_trump_beats_led_suit()
    test_scoring()
    print("\n✅ All rules tests passed!")