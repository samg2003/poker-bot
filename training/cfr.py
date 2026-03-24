"""
Vanilla Counterfactual Regret Minimization (CFR) for Kuhn Poker.

This validates that our game logic produces the correct Nash equilibrium.
The known equilibrium value for Player 1 in Kuhn poker is -1/18 ≈ -0.0556.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Tuple

from engine.kuhn_poker import KuhnState, JACK, QUEEN, KING


class InfoSetData:
    """Tracks regrets and strategy for one information set."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions

    def get_strategy(self) -> List[float]:
        """
        Get current strategy via regret matching.
        Positive regrets are normalized; if none, use uniform.
        """
        strategy = [max(0, r) for r in self.regret_sum]
        total = sum(strategy)
        if total > 0:
            return [s / total for s in strategy]
        return [1.0 / self.num_actions] * self.num_actions

    def get_average_strategy(self) -> List[float]:
        """
        Get average strategy over all iterations.
        This converges to Nash equilibrium.
        """
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / self.num_actions] * self.num_actions


class KuhnCFR:
    """
    CFR solver for Kuhn Poker.

    Usage:
        cfr = KuhnCFR()
        cfr.train(iterations=10000)
        strategy = cfr.get_final_strategy()
        game_value = cfr.get_game_value()
    """

    def __init__(self):
        self.info_sets: Dict[str, InfoSetData] = {}
        self.game_value_sum = 0.0
        self.iterations = 0

    def _get_info_set(self, key: str, num_actions: int) -> InfoSetData:
        if key not in self.info_sets:
            self.info_sets[key] = InfoSetData(num_actions)
        return self.info_sets[key]

    def train(self, iterations: int = 10000) -> float:
        """
        Run CFR for the given number of iterations.
        Returns the average game value for Player 1.
        """
        cards = [JACK, QUEEN, KING]

        for _ in range(iterations):
            # Deal all possible card permutations
            for p1_card, p2_card in permutations(cards, 2):
                state = KuhnState((p1_card, p2_card))
                value = self._cfr(state, [1.0, 1.0])
                self.game_value_sum += value
                self.iterations += 1

        return self.game_value_sum / self.iterations

    def _cfr(self, state: KuhnState, reach_probs: List[float]) -> float:
        """
        Recursive CFR traversal.

        Returns the expected value for Player 1.
        reach_probs: probability of reaching this state for each player.
        """
        if state.is_terminal:
            return state.get_payoff(0)  # payoff for P1

        player = state.current_player
        actions = state.get_actions()
        info_key = state.info_set_key()

        info_set = self._get_info_set(info_key, len(actions))
        strategy = info_set.get_strategy()

        # Accumulate strategy (weighted by reach probability)
        for i, s in enumerate(strategy):
            info_set.strategy_sum[i] += reach_probs[player] * s

        # Compute value of each action
        action_values = []
        node_value = 0.0

        for i, action in enumerate(actions):
            next_state = state.apply(action)

            # Update reach probability for the current player
            new_reach = list(reach_probs)
            new_reach[player] *= strategy[i]

            action_value = self._cfr(next_state, new_reach)
            action_values.append(action_value)
            node_value += strategy[i] * action_value

        # Compute and accumulate counterfactual regrets
        opponent = 1 - player
        for i, action_value in enumerate(action_values):
            if player == 0:
                regret = action_value - node_value
            else:
                regret = -(action_value - node_value)  # flip for P2

            info_set.regret_sum[i] += reach_probs[opponent] * regret

        return node_value

    def get_game_value(self) -> float:
        """Average game value for Player 1."""
        if self.iterations == 0:
            return 0.0
        return self.game_value_sum / self.iterations

    def get_final_strategy(self) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        Get the converged Nash equilibrium strategy.
        Returns dict mapping info_set_key → (actions, probabilities).
        """
        result = {}
        for key, info_set in sorted(self.info_sets.items()):
            avg_strategy = info_set.get_average_strategy()

            # Determine actions for this info set
            history = key.split(':')[1]
            dummy_state = KuhnState((0, 0), history)
            actions = dummy_state.get_actions()

            result[key] = (actions, avg_strategy)
        return result

    def print_strategy(self) -> None:
        """Print the converged strategy in a readable format."""
        strategy = self.get_final_strategy()
        print(f"\nGame value for P1: {self.get_game_value():.4f}")
        print(f"(Nash equilibrium: -0.0556)\n")
        print(f"{'Info Set':<10} {'Actions':<15} {'Strategy'}")
        print("-" * 50)
        for key, (actions, probs) in strategy.items():
            action_strs = [f"{a}={p:.3f}" for a, p in zip(actions, probs)]
            print(f"{key:<10} {str(actions):<15} {', '.join(action_strs)}")


if __name__ == '__main__':
    cfr = KuhnCFR()
    print("Training CFR on Kuhn Poker...")
    value = cfr.train(iterations=10000)
    cfr.print_strategy()
