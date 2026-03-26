"""
Lightweight Search — System 2 deep reasoning for complex poker decisions.

When the policy network (System 1) is uncertain, the search module builds
a small game subtree from the current state forward, runs CFR with the
policy network as a leaf evaluator, and returns a refined action distribution.

Key design choices (vs Pluribus-scale search):
- Small subtree: current street + 1 more (not full game tree)
- Policy network evaluates leaves (no Monte Carlo rollout)
- Opponent ranges from our encoder (no need to re-derive)
- Only 50-100 CFR iterations
- Only triggers on complex spots, not every decision

Usage:
    searcher = SearchEngine(policy_network, opponent_encoder, range_estimator)
    should = searcher.should_search(policy_output, game_state)
    if should:
        refined = searcher.search(game_state, player_idx, ...)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.action_space import ActionIndex, NUM_ACTION_TYPES
from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from model.stat_tracker import NUM_STAT_FEATURES
from search.range_estimator import RangeEstimator, NUM_COMBOS, uniform_range


# =============================================================================
# Search configuration
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for the search engine."""
    # Trigger conditions
    min_pot_bb: float = 20.0          # minimum pot (in BB) to trigger search
    entropy_threshold: float = 1.0     # policy entropy threshold
    max_streets_remaining: int = 2     # don't search too deep

    # Search parameters
    num_iterations: int = 100          # CFR iterations
    max_actions_per_node: int = 5      # discretize raise sizes to limit tree
    exploration_bonus: float = 0.3     # regret matching exploration

    # Raise size buckets (as fraction of pot)
    raise_sizes: Tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 1.5)


# =============================================================================
# Search tree node
# =============================================================================

class SearchNode:
    """A node in the search tree."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

        # CFR accumulators
        self.regret_sum = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions
        self.visit_count = 0

    def get_strategy(self) -> List[float]:
        """Get current strategy via regret matching."""
        strategy = [0.0] * self.num_actions
        positive_sum = sum(max(r, 0) for r in self.regret_sum)

        if positive_sum > 0:
            for i in range(self.num_actions):
                strategy[i] = max(self.regret_sum[i], 0) / positive_sum
        else:
            # Uniform
            for i in range(self.num_actions):
                strategy[i] = 1.0 / self.num_actions

        return strategy

    def get_average_strategy(self) -> List[float]:
        """Get the average strategy (converges to Nash)."""
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / self.num_actions] * self.num_actions


# =============================================================================
# Abstracted game state for search
# =============================================================================

@dataclass
class SearchState:
    """
    Abstracted game state for search tree traversal.

    Simplified from full GameState to make tree building tractable.
    """
    pot: float
    stacks: List[float]          # stack per player
    bets: List[float]            # current bet per player this street
    board: List[int]             # community cards
    street: int                  # 0=preflop, 1=flop, 2=turn, 3=river
    current_player: int          # whose turn
    is_terminal: bool = False
    folded: List[bool] = None    # who has folded
    num_players: int = 2

    def __post_init__(self):
        if self.folded is None:
            self.folded = [False] * self.num_players

    def get_actions(self, raise_sizes: Tuple[float, ...] = (0.5, 1.0)) -> List[str]:
        """Get abstracted legal actions."""
        if self.is_terminal:
            return []

        actions = []
        p = self.current_player
        to_call = max(self.bets) - self.bets[p]

        if to_call <= 0:
            actions.append('check')
        else:
            actions.append('fold')
            actions.append('call')

        # Discretized raise sizes
        if self.stacks[p] > to_call:
            for frac in raise_sizes:
                raise_amount = self.pot * frac
                total_raise = max(self.bets) + raise_amount
                if total_raise <= self.stacks[p] + self.bets[p]:
                    actions.append(f'raise_{frac}')

            # All-in is always an option
            actions.append('allin')

        return actions

    def apply(self, action: str) -> 'SearchState':
        """Apply an action, return new state."""
        new = SearchState(
            pot=self.pot,
            stacks=list(self.stacks),
            bets=list(self.bets),
            board=list(self.board),
            street=self.street,
            current_player=self.current_player,
            folded=list(self.folded),
            num_players=self.num_players,
        )

        p = new.current_player

        if action == 'fold':
            new.folded[p] = True
            # Check if only one player left
            active = sum(1 for f in new.folded if not f)
            if active <= 1:
                new.is_terminal = True
        elif action == 'check':
            pass  # no change
        elif action == 'call':
            to_call = max(new.bets) - new.bets[p]
            actual = min(to_call, new.stacks[p])
            new.bets[p] += actual
            new.stacks[p] -= actual
            new.pot += actual
        elif action.startswith('raise_'):
            frac = float(action.split('_')[1])
            raise_amount = new.pot * frac
            total = max(new.bets) + raise_amount
            additional = total - new.bets[p]
            actual = min(additional, new.stacks[p])
            new.bets[p] += actual
            new.stacks[p] -= actual
            new.pot += actual
        elif action == 'allin':
            actual = new.stacks[p]
            new.bets[p] += actual
            new.stacks[p] = 0
            new.pot += actual

        # Advance player
        new.current_player = (p + 1) % new.num_players
        while new.folded[new.current_player] or new.stacks[new.current_player] <= 0:
            new.current_player = (new.current_player + 1) % new.num_players
            if new.current_player == p:
                break

        # Check if street is over (simplified: everyone has matched highest bet)
        all_matched = all(
            new.bets[i] >= max(new.bets) or new.folded[i] or new.stacks[i] == 0
            for i in range(new.num_players)
        )
        if all_matched and action not in ('fold',):
            # Check if it's a call or check closing the action
            active_not_allin = sum(
                1 for i in range(new.num_players)
                if not new.folded[i] and new.stacks[i] > 0
            )
            if new.street >= 3 or active_not_allin <= 1:
                new.is_terminal = True
            else:
                # Advance street
                new.street += 1
                new.bets = [0.0] * new.num_players
                new.current_player = 0
                while new.folded[new.current_player]:
                    new.current_player = (new.current_player + 1) % new.num_players

        return new

    @property
    def info_key(self) -> str:
        """Information set key for CFR."""
        return f"p{self.current_player}_s{self.street}_pot{self.pot:.0f}_bets{'_'.join(f'{b:.0f}' for b in self.bets)}"


# =============================================================================
# Search Engine
# =============================================================================

class SearchEngine:
    """
    Lightweight search engine (System 2).

    Builds a small game subtree and runs CFR with the policy network
    as a leaf evaluator.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        range_estimator: Optional[RangeEstimator] = None,
        config: Optional[SearchConfig] = None,
    ):
        self.policy = policy
        self.opponent_encoder = opponent_encoder
        self.range_estimator = range_estimator
        self.config = config or SearchConfig()
        self.nodes: Dict[str, SearchNode] = {}

    def should_search(
        self,
        action_probs: torch.Tensor,
        pot_bb: float,
        street: int,
    ) -> bool:
        """
        Determine if search should be triggered.

        Search is expensive (~200ms) — only use for complex, high-stakes spots.
        """
        # Entropy of action distribution
        entropy = -(action_probs * (action_probs + 1e-8).log()).sum().item()

        return (
            pot_bb >= self.config.min_pot_bb
            and entropy >= self.config.entropy_threshold
            and (3 - street) <= self.config.max_streets_remaining
        )

    @torch.no_grad()
    def evaluate_leaf(self, state: SearchState, player: int) -> float:
        """
        Evaluate a leaf node using the policy network's value head.

        Returns expected value for `player` at this state.
        """
        self.policy.eval()

        # Build minimal inputs for the policy network
        hole_cards = torch.tensor([[0, 1]], dtype=torch.long)  # placeholder
        board = list(state.board)
        while len(board) < 5:
            board.append(-1)
        community = torch.tensor([board], dtype=torch.long)

        bb = 1.0
        numeric = torch.tensor([[
            state.pot / (100 * bb),
            state.stacks[player] / (100 * bb),
            state.bets[player] / (100 * bb),
            float(player),
            state.street / 3.0,
            state.num_players / 9.0,
            sum(1 for f in state.folded if not f) / 9.0,
            max(state.bets) / (100 * bb),
            0.0,  # min_raise, dummy 0.0 for leaf evaluation
            max(0.0, max(state.bets) - state.bets[player]) / (100 * bb),  # amount_to_call
        ]], dtype=torch.float32)

        opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
        opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)
        own_stats = torch.zeros(1, NUM_STAT_FEATURES)

        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embed,
            opponent_stats=opp_stats,
            own_stats=own_stats,
        )

        return output.value[0, 0].item()

    def _cfr_traverse(
        self,
        state: SearchState,
        hero: int,
        depth: int = 0,
        max_depth: int = 10,
    ) -> float:
        """
        Traverse the search tree using CFR.

        Returns expected value for hero.
        """
        if state.is_terminal or depth >= max_depth:
            return self.evaluate_leaf(state, hero)

        actions = state.get_actions(self.config.raise_sizes)
        if not actions:
            return self.evaluate_leaf(state, hero)

        num_actions = len(actions)
        key = state.info_key

        # Get or create node
        if key not in self.nodes:
            self.nodes[key] = SearchNode(num_actions)
        node = self.nodes[key]

        strategy = node.get_strategy()
        action_values = [0.0] * num_actions

        # Traverse each action
        for i, action in enumerate(actions):
            next_state = state.apply(action)
            action_values[i] = self._cfr_traverse(next_state, hero, depth + 1, max_depth)

        # Node value under current strategy
        node_value = sum(strategy[i] * action_values[i] for i in range(num_actions))

        # Update regrets and strategy sums
        if state.current_player == hero:
            for i in range(num_actions):
                node.regret_sum[i] += action_values[i] - node_value
        node.visit_count += 1

        for i in range(num_actions):
            node.strategy_sum[i] += strategy[i]

        return node_value

    def search(
        self,
        pot: float,
        stacks: List[float],
        board: List[int],
        street: int,
        hero: int,
        num_iterations: Optional[int] = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Run search from the current game state.

        Args:
            pot: current pot size
            stacks: stack per player
            board: community cards
            street: current street (0-3)
            hero: our player index

        Returns:
            (actions, probabilities) — refined action distribution
        """
        iterations = num_iterations or self.config.num_iterations
        self.nodes.clear()

        # Build initial search state
        search_state = SearchState(
            pot=pot,
            stacks=list(stacks),
            bets=[0.0] * len(stacks),
            board=list(board),
            street=street,
            current_player=hero,
            num_players=len(stacks),
        )

        # Run CFR iterations
        for _ in range(iterations):
            self._cfr_traverse(search_state, hero, max_depth=6)

        # Get average strategy from the root node
        root_key = search_state.info_key
        if root_key in self.nodes:
            node = self.nodes[root_key]
            actions = search_state.get_actions(self.config.raise_sizes)
            avg_strategy = node.get_average_strategy()
            return actions, avg_strategy
        else:
            # Fallback
            actions = search_state.get_actions(self.config.raise_sizes)
            uniform = [1.0 / len(actions)] * len(actions)
            return actions, uniform

    def get_search_stats(self) -> Dict:
        """Return stats about the last search."""
        return {
            'num_nodes': len(self.nodes),
            'total_visits': sum(n.visit_count for n in self.nodes.values()),
        }
