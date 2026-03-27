import pytest
from engine.dealer import Dealer
from engine.game_state import Action, ActionType, Street

def apply_actions(dealer, actions):
    for a in actions:
        dealer.apply_action(a)

def test_ev_fold_preflop():
    """Ensure EV perfectly mirrors physical profit if the hand ends early (No Showdown)."""
    dealer = Dealer(num_players=2, stacks=[100, 100])
    dealer.start_hand()
    
    # SB (P1) acts first preflop if Heads-Up
    apply_actions(dealer, [Action(ActionType.FOLD)])
    
    assert dealer.is_hand_over()
    results = dealer.get_results()
    
    # No equity calculations should happen, it should just map 1:1 over folding
    assert results['profit'] == [-0.5, 0.5]
    assert results['ev_profit'] == [-0.5, 0.5]

def test_ev_all_in_preflop():
    """Ensure EV dynamically simulates the remaining board instead of picking a random physical winner."""
    dealer = Dealer(num_players=2, stacks=[100, 100])
    dealer.start_hand()
    
    # Rig the deck:
    # P0: As Ac (51, 48)
    # P1: Ks Kc (47, 44)
    dealer.state.players[0].hole_cards = (51, 48)
    dealer.state.players[1].hole_cards = (47, 44)
    
    # P1 (SB) raises 100, P0 (BB) calls
    apply_actions(dealer, [
        Action(ActionType.RAISE, 100.0),
        Action(ActionType.CALL)
    ])
    
    results = dealer.get_results()
    assert dealer.state.street == Street.SHOWDOWN
    
    # Player 1 (Aces) should have roughly 82% EV on a 200bb pot.
    # Player 0 (Kings) should have roughly 18% EV.
    ev_prof = results['ev_profit']
    act_prof = results['profit']
    
    # Aces EV Profit: (0.82 * 200) - 100 = ~64
    assert ev_prof[0] > 50
    # Kings EV Profit: (0.18 * 200) - 100 = ~-64
    assert ev_prof[1] < -50
    
    # Physical profit must remain 100% pure casino luck for the UI to function
    assert (act_prof == [100.0, -100.0] or act_prof == [-100.0, 100.0])

def test_ev_side_pots_complex():
    """Ensure EV isolates perfectly into main and side pots dynamically for multiple eligible subsets."""
    # 3 Players. BTN (P0) has 10. SB (P1) has 50. BB (P2) has 100.
    dealer = Dealer(num_players=3, stacks=[10, 50, 100])
    dealer.start_hand()
    
    # Rig the deck (P0=Aces, P1=Kings, P2=Queens)
    dealer.state.players[0].hole_cards = (51, 48)  # Aces
    dealer.state.players[1].hole_cards = (47, 44)  # Kings
    dealer.state.players[2].hole_cards = (43, 40)  # Queens
    
    # Action Preflop:
    # BTN (P0) acts first, Shoves 10.
    # SB (P1) acts next, Shoves 50.
    # BB (P2) acts next, Calls 50.
    apply_actions(dealer, [
        Action(ActionType.RAISE, 10.0),
        Action(ActionType.RAISE, 50.0),
        Action(ActionType.CALL)
    ])
    
    results = dealer.get_results()
    
    # EV Math:
    # Pot 1 (30bb): P0, P1, P2. P0 rules. P0 gets +EV. P1/P2 lose EV.
    # Pot 2 (80bb): P1, P2. P1 rules. P1 gets massive +EV. P2 loses EV entirely.
    
    ev = results['ev_profit']
    
    assert ev[0] > 0   # Aces inherently positive EV over 30bb pot
    assert ev[1] > 0   # Kings intrinsically positive EV because it dominates Queens in the massive 80bb side-pot
    assert ev[2] < 0   # Queens lose EV in all directions
    
    # The sum of all EV profits must always equal 0 (chips merely change hands)
    assert sum(ev) == pytest.approx(0.0)
