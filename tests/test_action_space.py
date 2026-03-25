import pytest
import torch

from engine.game_state import GameState, Action, ActionType
from model.action_space import get_sizing_mask, POT_FRACTIONS

def test_sizing_mask_preflop_deep():
    """Test standard preflop spot with deep stacks."""
    # 2 players, 100bb each, bb=2
    state = GameState(num_players=2, stacks=[100.0, 100.0], big_blind=2.0, small_blind=1.0)
    state.post_blinds()
    # P0 (SB) is to act. Pot is 3 (1+2). Current bet is 2.
    # Min raise is 4 (current bet 2 + min raise amount 2).
    # Stack remaining is 99. Max raise is 100.
    
    mask = get_sizing_mask(state)
    assert mask.shape == (10,)
    assert mask.dtype == torch.bool
    
    # Let's check which buckets are legal.
    # target = 2 + frac * 3
    # min_raise = 4. 4 <= 2 + frac*3 => frac >= 2/3 (0.66)
    for i, frac in enumerate(POT_FRACTIONS):
        if frac < 0:
            assert mask[i] == True, "All-in should always be legal if raising is legal"
        else:
            target = state.current_bet + frac * state.pot
            is_legal = (target >= 4.0 - 1e-5) and (target <= 100.0 + 1e-5)
            assert mask[i] == is_legal, f"Bucket {frac} legality mismatch. Target = {target}"

def test_sizing_mask_short_stack():
    """Test spot where stack is too short for large bet sizes."""
    # Player 0 has 10 chips, Player 1 has 1000 chips. BB=2.
    state = GameState(num_players=2, stacks=[10.0, 1000.0], big_blind=2.0, small_blind=1.0)
    state.post_blinds()
    # P0 to act. Pot = 3. Current bet = 2. Stack = 9. Total = 10.
    # Max raise is 10.
    mask = get_sizing_mask(state)
    
    for i, frac in enumerate(POT_FRACTIONS):
        if frac < 0:
            assert mask[i] == True  # All in
        else:
            target = state.current_bet + frac * state.pot
            # If target > 10, should be masked!
            if target > 10.0 + 1e-5:
                assert mask[i] == False, f"Target {target} should be illegal since stack is 10"

def test_sizing_mask_forced_all_in():
    """Test spot where min_raise is greater than stack size."""
    # Player 0 has 3 chips, Player 1 has 10 chips. BB=2.
    state = GameState(num_players=2, stacks=[3.0, 10.0], big_blind=2.0, small_blind=1.0)
    state.post_blinds()
    # P0 posted 1 (2 left). BB posted 2. Pot = 3.
    # P0 max raise is 3 (all-in).
    # But current_bet = 2. Min raise = 4 (which is > 3).
    # Therefore NO sizing multiplier is mathematically >= min_raise AND <= max_raise, EXCEPT all-in.
    mask = get_sizing_mask(state)
    
    # We expect all buckets to fail typical constraints, but Fallback/All-in MUST be True.
    assert mask[-1] == True, "All-in bucket must be legal even if it's a short all-in"
    
    # Standard check: None of the other buckets should realistically be allowed,
    # OR if they evaluate, they shouldn't mistakenly trigger the mask.
    for i in range(len(POT_FRACTIONS) - 1):
        assert mask[i] == False, f"Bucket index {i} should be illegal because stack is too small"

def test_evaluator_amount_clamping():
    """
    Simulate the Evaluator decoding logic to ensure no amount exceeds bounds.
    (This exact logic exists in `poker_agent.py` and `evaluator.py`)
    """
    state = GameState(num_players=2, stacks=[10.0, 100.0], big_blind=2.0, small_blind=1.0)
    state.post_blinds()
    # P0 (dealer) has 10 total.
    
    legal_types = state.get_legal_actions()
    assert ActionType.RAISE in legal_types
    
    min_r = state.get_min_raise_to() # 4
    max_r = state.get_max_raise_to() # 10
    
    # Suppose model selected the 2.0x pot bucket (index 8).
    # Target = 2 + 2.0 * 3 = 8
    frac = 2.0
    rt = state.current_bet + frac * state.pot
    assert rt == 8.0
    
    # Clamp
    amount = max(min_r, min(rt, max_r))
    assert amount == 8.0
    
    # Now suppose pot explodes and model selected 2.0x pot which is 20 chips!
    state.pot = 15.0
    rt = state.current_bet + frac * state.pot
    assert rt == 32.0  # 2 + 2*15
    # Clamp
    amount = max(min_r, min(rt, max_r))
    assert amount == 10.0  # Clamped exactly to max_raise (All In!)
    
    # If the model explicitly chose the All-In bucket (frac = -1.0)
    amount_all_in = max_r
    assert amount_all_in == 10.0

def test_engine_all_in_limit():
    """Verify the Engine accepts an exact max_raise as a legal action without issues."""
    state = GameState(num_players=2, stacks=[100.0, 100.0], big_blind=2.0, small_blind=1.0)
    state.post_blinds()
    
    max_r = state.get_max_raise_to()
    assert max_r == 100.0
    
    # Make the action
    state.apply_action(Action(ActionType.RAISE, amount=max_r))
    
    assert state.players[0].is_all_in
    assert state.pot == 102.0 # 100 from P0 + 2 from P1 BB
