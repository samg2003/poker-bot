#!/usr/bin/env python3
"""Monte Carlo equity evaluator using eval7."""

import eval7
import random
import sys
from itertools import combinations


def parse_cards(s: str) -> list[eval7.Card]:
    """Parse a string like 'AhKs' into a list of eval7.Card objects."""
    cards = []
    s = s.strip()
    if not s:
        return cards
    i = 0
    while i < len(s):
        rank = s[i].upper()
        suit = s[i + 1].lower()
        cards.append(eval7.Card(rank + suit))
        i += 2
    return cards


CHECKPOINTS = [10, 20, 50, 100, 250, 500, 1000, 10000]


def monte_carlo_equity(hands: list[list[eval7.Card]], board: list[eval7.Card],
                       num_simulations: int = 100_000) -> list[float]:
    """Run Monte Carlo simulation to estimate equity for each hand."""
    num_players = len(hands)
    wins = [0.0] * num_players

    # Build the deck minus known cards
    all_cards = eval7.Deck().cards
    dead = set()
    for h in hands:
        for c in h:
            dead.add(c)
    for c in board:
        dead.add(c)
    remaining = [c for c in all_cards if c not in dead]

    cards_needed = 5 - len(board)

    # Build set of checkpoints that fall within our sim count
    checkpoints = sorted(set(c for c in CHECKPOINTS if c < num_simulations))

    # Header for convergence table
    player_labels = [f"P{i+1}" for i in range(num_players)]
    header = f"  {'Sims':>8}  " + "  ".join(f"{p:>7}" for p in player_labels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for sim in range(1, num_simulations + 1):
        # Draw remaining community cards
        drawn = random.sample(remaining, cards_needed)
        full_board = board + drawn

        # Evaluate each hand
        scores = []
        for h in hands:
            scores.append(eval7.evaluate(h + full_board))

        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        share = 1.0 / len(winners)
        for w in winners:
            wins[w] += share

        # Print at checkpoints
        if sim in checkpoints:
            eqs = [w / sim * 100 for w in wins]
            row = f"  {sim:>8,}  " + "  ".join(f"{e:>6.2f}%" for e in eqs)
            print(row)

    # Final result
    equities = [w / num_simulations * 100 for w in wins]
    row = f"  {num_simulations:>8,}  " + "  ".join(f"{e:>6.2f}%" for e in equities)
    print(row)

    return equities


def main():
    print("=" * 50)
    print("  Monte Carlo Poker Equity Calculator (eval7)")
    print("=" * 50)

    # Number of players
    num_players = int(input("\nHow many players? "))

    # Get the board first so we can exclude those cards from random deals
    raw_board = input("Board (empty for preflop, e.g. Ah2sKd): ").strip()
    board = parse_cards(raw_board)
    if len(board) not in (0, 3, 4, 5):
        print(f"Error: board must have 0, 3, 4, or 5 cards, got {len(board)}")
        sys.exit(1)

    # Track used cards for random dealing
    used_cards = set(board)

    # Get each player's hand (empty = random)
    hands = []
    random_players = []
    for i in range(num_players):
        raw = input(f"Player {i + 1} hand (e.g. AhKs, empty for random): ").strip()
        if not raw:
            # Deal random hand from remaining cards
            remaining = [c for c in eval7.Deck().cards if c not in used_cards]
            hand = random.sample(remaining, 2)
            used_cards.update(hand)
            hands.append(hand)
            random_players.append(i)
        else:
            hand = parse_cards(raw)
            if len(hand) != 2:
                print(f"Error: expected 2 cards, got {len(hand)}")
                sys.exit(1)
            used_cards.update(hand)
            hands.append(hand)

    # Number of simulations
    raw_sims = input("Simulations (default 100000): ").strip()
    num_sims = int(raw_sims) if raw_sims else 100_000

    # Determine street
    street_names = {0: "Pre-flop", 3: "Flop", 4: "Turn", 5: "River"}
    street = street_names[len(board)]

    print(f"\nStreet: {street}")
    for i, h in enumerate(hands):
        print(f"  Player {i + 1}: {h[0]} {h[1]}")
    if board:
        print(f"  Board: {' '.join(str(c) for c in board)}")
    print(f"  Running {num_sims:,} simulations...\n")

    equities = monte_carlo_equity(hands, board, num_sims)
    print()


if __name__ == "__main__":
    main()
