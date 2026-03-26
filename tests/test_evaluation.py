"""
Tests for Phase 5: agent interface and evaluation framework.
"""

import pytest
import torch

from agent.poker_agent import PokerAgent, ActionResult
from agent.config import AgentConfig
from model.action_space import ActionIndex
from model.stat_tracker import HandRecord
from evaluation.evaluator import Evaluator, BenchmarkResult, EvalResults
from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder


# =============================================================================
# Agent Tests
# =============================================================================

class TestPokerAgent:
    @pytest.fixture
    def agent(self):
        config = AgentConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            enable_search=False,
        )
        return PokerAgent.from_config(config)

    def test_get_action(self, agent):
        """Agent returns valid action."""
        result = agent.get_action(
            hole_cards=(0, 1),
            community_cards=[10, 20, 30],
            numeric_features=[0.5, 1.0, 0.0] + [0.0]*9 + [0.0] + [1.0, 0.0, 0.0, 0.0] + [0.22, 0.22, 0.0, 0.0, 0.0, 2.0],
            opponent_ids=[1],
        )
        assert isinstance(result, ActionResult)
        assert 0 <= result.action_type < 4
        assert result.bet_sizing == -1.0 or 0 <= result.bet_sizing <= 2.0
        assert abs(result.action_probs.sum().item() - 1.0) < 1e-4

    def test_action_mask(self, agent):
        """Agent respects action mask."""
        mask = [True, False, True, False]  # only fold + call
        result = agent.get_action(
            hole_cards=(0, 1),
            community_cards=[],
            numeric_features=[0.5, 1.0, 0.0] + [0.0]*9 + [0.0] + [1.0, 0.0, 0.0, 0.0] + [0.22, 0.22, 0.0, 0.0, 0.0, 2.0],
            opponent_ids=[1],
            action_mask=mask,
        )
        # Check and raise should have near-zero prob
        assert result.action_probs[ActionIndex.CHECK] < 1e-5
        assert result.action_probs[ActionIndex.RAISE] < 1e-5

    def test_observe_action(self, agent):
        """Observing actions updates opponent history."""
        assert 1 not in agent._action_sequences
        agent.observe_action(
            player_id=1, action_type=ActionIndex.RAISE,
            bet_size_frac=0.75, pot_size=10.0, stack_size=100.0, street=1,
        )
        assert 1 in agent._action_sequences
        assert len(agent._action_sequences[1]) == 1

    def test_with_opponent_history(self, agent):
        """Agent works with accumulated opponent history."""
        # Build some opponent history
        for _ in range(5):
            agent.observe_action(1, ActionIndex.RAISE, 0.5, 10, 100, 0)

        result = agent.get_action(
            hole_cards=(0, 1),
            community_cards=[],
            numeric_features=[0.5, 1.0, 0.0] + [0.0]*9 + [0.0] + [1.0, 0.0, 0.0, 0.0] + [0.22, 0.22, 0.0, 0.0, 0.0, 2.0],
            opponent_ids=[1],
        )
        assert isinstance(result, ActionResult)

    def test_reset_opponent(self, agent):
        agent.observe_action(1, ActionIndex.CALL, 0, 10, 100, 0)
        agent.reset_opponent(1)
        assert 1 not in agent._action_sequences

    def test_reset_all(self, agent):
        agent.observe_action(1, ActionIndex.CALL, 0, 10, 100, 0)
        agent.observe_action(2, ActionIndex.FOLD, 0, 10, 100, 0)
        agent.reset_all()
        assert len(agent._action_sequences) == 0

    def test_param_count(self, agent):
        count = agent.get_param_count()
        assert count > 0

    def test_no_opponents(self, agent):
        """Agent works with no opponents (solo, edge case)."""
        result = agent.get_action(
            hole_cards=(0, 1),
            community_cards=[10, 20, 30, 40, 45],
            numeric_features=[1.0, 0.5, 0.3] + [0.0]*9 + [0.0] + [0.0, 0.0, 0.0, 1.0] + [0.22, 0.11, 0.5, 0.2, 0.2, 0.5],
            opponent_ids=[],
        )
        assert isinstance(result, ActionResult)


# =============================================================================
# Evaluator Tests
# =============================================================================

class TestEvaluator:
    @pytest.fixture
    def evaluator(self):
        policy = PolicyNetwork(embed_dim=32, opponent_embed_dim=32,
                               num_cross_attn_heads=2, num_cross_attn_layers=1)
        encoder = OpponentEncoder(embed_dim=32, num_layers=1, num_heads=2)
        return Evaluator(policy, encoder, num_hands=50)

    def test_model_consistency(self, evaluator):
        result = evaluator.benchmark_model_consistency()
        assert isinstance(result, BenchmarkResult)
        assert result.passed

    def test_value_head_sanity(self, evaluator):
        result = evaluator.benchmark_value_head()
        assert isinstance(result, BenchmarkResult)
        assert result.passed

    def test_gto_symmetry(self, evaluator):
        result = evaluator.benchmark_gto_symmetry()
        assert isinstance(result, BenchmarkResult)
        # Untrained model — just check it runs without error

    def test_exploitation_benchmark(self, evaluator):
        result = evaluator.benchmark_exploitation()
        assert isinstance(result, BenchmarkResult)

    def test_run_all_benchmarks(self, evaluator):
        results = evaluator.run_all_benchmarks()
        assert isinstance(results, EvalResults)
        assert len(results.benchmarks) == 6
        # Model consistency and value head should pass even untrained
        assert results.benchmarks[0].passed  # consistency
        assert results.benchmarks[1].passed  # value head

    def test_eval_results_summary(self, evaluator):
        results = evaluator.run_all_benchmarks()
        summary = results.summary()
        assert "EVALUATION RESULTS" in summary
        assert "Pass rate" in summary


class TestBenchmarkResult:
    def test_pass(self):
        r = BenchmarkResult("Test", True, "metric", 0.5, 1.0)
        assert "PASS" in str(r)

    def test_fail(self):
        r = BenchmarkResult("Test", False, "metric", 2.0, 1.0)
        assert "FAIL" in str(r)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
