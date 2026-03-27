import pytest
from engine.hand_evaluator import Eval7Evaluator

def test_eval7_card_conversion():
    """Ensure our 0-51 integers correctly map to eval7 strings."""
    # 0 = 2c, 51 = As
    c0 = Eval7Evaluator.int_to_eval7(0)
    assert str(c0) == "2c"
    
    c51 = Eval7Evaluator.int_to_eval7(51)
    assert str(c51) == "As"
    
    c_invalid = Eval7Evaluator.int_to_eval7(-1)
    assert c_invalid is None

def test_eval7_equity():
    """Test 1000-runout EV calculator for Aces vs Kings."""
    # Hero: As(51), Ac(48)
    hero = [51, 48]
    # Villain: Ks(47), Kc(44)
    villain = [47, 44]
    
    # Board: 2h(2), 7d(21), Jc(36)
    board = [2, 21, 36]
    
    equities = Eval7Evaluator.get_equity([hero, villain], board, runouts=1000)
    
    assert len(equities) == 2
    assert sum(equities) == pytest.approx(1.0)
    # Aces dominate Kings on dry board (~91%)
    assert equities[0] > 0.80
    assert equities[1] < 0.20

def test_eval7_showdown():
    """Test instant exact showdown calculator."""
    # Board: Ah(50), Kh(46), Qh(42), Jh(38), 2c(0)
    board = [50, 46, 42, 38, 0]
    
    # Hero: Th(34), 9h(30) -> Straight Flush
    hero = [34, 30]
    # Villain: As(51), Ac(48) -> Three of a kind
    villain = [51, 48]
    
    winners = Eval7Evaluator.get_showdown_winners([hero, villain], board)
    
    # Hero should unconditionally win
    assert winners == [0]

def test_eval7_showdown_split_pot():
    """Test exact showdown identifying a split pot."""
    # Board: As, Ks, Qs, Js, Ts (Royal flush on board)
    # 51, 47, 43, 39, 35
    board = [51, 47, 43, 39, 35]
    
    hero = [0, 1]      # 2c, 2d
    villain = [2, 3]   # 2h, 2s
    
    winners = Eval7Evaluator.get_showdown_winners([hero, villain], board)
    
    # Both players play the board for a Royal Flush, so they split
    assert winners == [0, 1]
