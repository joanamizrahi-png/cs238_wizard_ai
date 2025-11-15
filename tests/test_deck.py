import sys
sys.path.append('..')

from game.deck import Deck, Card, Suit

def test_deck_creation():
    """Test that deck has 60 cards"""
    deck = Deck.create_standard_deck()
    assert len(deck) == 60
    print("✓ Deck has 60 cards")

def test_deal_round_1():
    """Test dealing for round 1 (4 players, 1 card each)"""
    hands, trump, remaining = Deck.shuffle_and_deal(num_players=4, round_num=1)
    
    assert len(hands) == 4
    assert all(len(hand) == 1 for hand in hands)
    assert trump is not None
    print("✓ Round 1 dealing works")
    print(f"  Example hands: {hands}")
    print(f"  Trump: {trump}")

def test_deal_round_15():
    """Test dealing for final round (4 players, 15 cards each)"""
    hands, trump, remaining = Deck.shuffle_and_deal(num_players=4, round_num=15)
    
    assert len(hands) == 4
    assert all(len(hand) == 15 for hand in hands)
    assert trump is None  # All cards dealt
    print("✓ Round 15 dealing works (no trump)")

if __name__ == "__main__":
    test_deck_creation()
    test_deal_round_1()
    test_deal_round_15()
    print("\n✅ All deck tests passed!")