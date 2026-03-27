# import eval7
# # 1. Convert your cards to the C++ format
# hero_hand = [eval7.Card("As"), eval7.Card("Ac")]
# villain_hand = [eval7.Card("Ks"), eval7.Card("Kc")]
# board = [eval7.Card("2h"), eval7.Card("7d"), eval7.Card("Jc")]
# # 2. Ask the C++ backend to run 10,000 Monte Carlo runouts instantly
# hero_equity = eval7.py_hand_vs_range_monte_carlo(
#     hero_hand, 
#     eval7.HandRange("KK"), # Opponent's hand as a range
#     board, 
#     10000 
# )
# print(hero_equity) 

import random
import eval7
def get_multiway_equity_eval7(known_hands, known_board, runouts=1000):
    """
    known_hands: List of hands, e.g., [[eval7.Card('As'), eval7.Card('Ac')], ...]
    known_board: List of community cards currently dealt, e.g., [eval7.Card('2h'), ...]
    Returns: A list of win equities corresponding to each player.
    """
    wins = [0.0] * len(known_hands)
    
    # 1. Build a completely fresh C-optimized Deck
    deck = eval7.Deck()
    
    # 2. Collect and remove "Dead Cards"
    dead_cards = known_board.copy()
    for hand in known_hands:
        dead_cards.extend(hand)
        
    for c in dead_cards:
        deck.cards.remove(c)
        
    cards_needed = 5 - len(known_board)
    deck_list = deck.cards  # Cache list for fast random sampling
    
    for _ in range(runouts):
        # 3. Deal randomly until there are exactly 5 community cards
        if cards_needed > 0:
            simulated_board = known_board + random.sample(deck_list, cards_needed)
        else:
            simulated_board = known_board
            
        # 4. Evaluate all players. Note: In eval7, HIGHER score is better!
        best_score = -1
        winners = []
        for i, hand in enumerate(known_hands):
            # Combine the 5 board cards and 2 hole cards into 7 and evaluate
            score = eval7.evaluate(simulated_board + hand)
            
            if score > best_score:
                best_score = score
                winners = [i]
            elif score == best_score:
                winners.append(i) # Tie (split pot)
                
        # 5. Award the win (dividing evenly if it's a split pot)
        for tied_winner in winners:
            wins[tied_winner] += 1.0 / len(winners)
            
    return [w / runouts for w in wins]
# --- EXAMPLE USAGE: 3-WAY ALL-IN WITH EVAL7 ---
hero      = [eval7.Card("As"), eval7.Card("Ac")]
villain_1 = [eval7.Card("Kh"), eval7.Card("Kd")]
villain_2 = [eval7.Card("8s"), eval7.Card("9s")]
flop_board = [eval7.Card("2s"), eval7.Card("7s"), eval7.Card("Jc")]
equities = get_multiway_equity_eval7([hero, villain_1, villain_2], flop_board, runouts=1000)
print(f"Hero Equity:      {equities[0]:.2%}")
print(f"Villain 1 Equity: {equities[1]:.2%}")
print(f"Villain 2 Equity: {equities[2]:.2%}")