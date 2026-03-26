"""
Tests for Phase 6: deployment infrastructure.
"""

import json
import os
import pytest
import shutil
import tempfile
import torch

from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from deployment.checkpoint import CheckpointManager, CheckpointMetadata
from deployment.inference import InferenceEngine, InferenceStats


# =============================================================================
# Checkpoint Tests
# =============================================================================

class TestCheckpointManager:
    @pytest.fixture
    def models(self):
        policy = PolicyNetwork(embed_dim=32, opponent_embed_dim=32,
                               num_cross_attn_heads=2, num_cross_attn_layers=1)
        encoder = OpponentEncoder(embed_dim=32, num_layers=1, num_heads=2)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(encoder.parameters()), lr=1e-3
        )
        return policy, encoder, optimizer

    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def metadata(self):
        return CheckpointMetadata(
            version="test_v001",
            created_at="2024-01-01T00:00:00",
            epoch=10,
            stage="Test Stage",
            total_hands=5000,
            avg_reward=0.5,
            loss=0.1,
            test_count=154,
        )

    def test_save_and_load(self, models, tmp_dir, metadata):
        policy, encoder, optimizer = models
        mgr = CheckpointManager(tmp_dir)

        # Save
        path = mgr.save(policy, encoder, optimizer, metadata)
        assert (path / 'policy.pt').exists()
        assert (path / 'opponent_encoder.pt').exists()
        assert (path / 'metadata.json').exists()

        # Load into fresh models
        policy2 = PolicyNetwork(embed_dim=32, opponent_embed_dim=32,
                                num_cross_attn_heads=2, num_cross_attn_layers=1)
        encoder2 = OpponentEncoder(embed_dim=32, num_layers=1, num_heads=2)
        loaded_meta = mgr.load(policy2, encoder2)

        assert loaded_meta.version == "test_v001"
        assert loaded_meta.epoch == 10

        # Verify weights match
        for p1, p2 in zip(policy.parameters(), policy2.parameters()):
            assert torch.allclose(p1, p2)

    def test_save_best(self, models, tmp_dir, metadata):
        policy, encoder, optimizer = models
        mgr = CheckpointManager(tmp_dir)

        # First save — should be best
        path = mgr.save_best(policy, encoder, optimizer, metadata)
        assert path is not None

        # Save worse — should NOT replace
        worse = CheckpointMetadata(
            version="worse", created_at="2024-01-02",
            epoch=20, stage="Test", total_hands=10000,
            avg_reward=0.2, loss=0.5, test_count=154,
        )
        path2 = mgr.save_best(policy, encoder, optimizer, worse)
        assert path2 is None  # not saved

    def test_version_tracking(self, models, tmp_dir, metadata):
        policy, encoder, optimizer = models
        mgr = CheckpointManager(tmp_dir)

        mgr.save(policy, encoder, optimizer, metadata, tag='v001')
        metadata.version = "test_v002"
        metadata.epoch = 20
        mgr.save(policy, encoder, optimizer, metadata, tag='v002')

        versions = mgr.list_versions()
        assert len(versions) == 2

    def test_metadata_roundtrip(self, metadata):
        d = metadata.to_dict()
        restored = CheckpointMetadata.from_dict(d)
        assert restored.version == metadata.version
        assert restored.epoch == metadata.epoch
        assert restored.avg_reward == metadata.avg_reward

    def test_load_nonexistent_raises(self, models, tmp_dir):
        policy, encoder, _ = models
        mgr = CheckpointManager(tmp_dir)
        with pytest.raises(FileNotFoundError):
            mgr.load(policy, encoder, tag='nonexistent')


# =============================================================================
# Inference Engine Tests
# =============================================================================

class TestInferenceEngine:
    @pytest.fixture
    def engine(self):
        policy = PolicyNetwork(embed_dim=32, opponent_embed_dim=32,
                               num_cross_attn_heads=2, num_cross_attn_layers=1)
        encoder = OpponentEncoder(embed_dim=32, num_layers=1, num_heads=2)
        return InferenceEngine(policy, encoder)

    def test_basic_inference(self, engine):
        hole = torch.tensor([[0, 1]])
        community = torch.tensor([[-1, -1, -1, -1, -1]])
        numeric = torch.randn(1, 10).to(engine.device)

        output = engine.infer(hole, community, numeric)
        assert output.action_type_probs.shape == (1, 4)
        assert abs(output.action_type_probs.sum().item() - 1.0) < 1e-4

    def test_latency_tracking(self, engine):
        hole = torch.tensor([[0, 1]])
        community = torch.tensor([[-1, -1, -1, -1, -1]])
        numeric = torch.randn(1, 10).to(engine.device)
        engine.infer(hole, community, numeric)
        assert engine.stats.num_calls == 1
        assert engine.stats.avg_latency_ms > 0

    def test_batch_inference(self, engine):
        inputs = []
        for _ in range(4):
            inputs.append({
                'hole_cards': torch.randint(0, 52, (2,)),
                'community_cards': torch.full((5,), -1),
                'numeric_features': torch.randn(10).to(engine.device),
            })

        outputs = engine.infer_batch(inputs)
        assert len(outputs) == 4
        for out in outputs:
            assert out.action_type_probs.shape[1] == 4

    def test_benchmark(self, engine):
        stats = engine.benchmark(num_iterations=10)
        assert stats.num_calls == 10
        assert stats.avg_latency_ms > 0
        assert stats.min_latency_ms <= stats.max_latency_ms

    def test_optimize(self, engine):
        engine.optimize()
        assert engine._is_optimized
        # Still works after optimization
        hole = torch.tensor([[0, 1]])
        community = torch.tensor([[-1, -1, -1, -1, -1]])
        numeric = torch.randn(1, 10).to(engine.device)
        output = engine.infer(hole, community, numeric)
        assert output.action_type_probs.shape == (1, 4)

    def test_model_size(self, engine):
        size = engine.get_model_size_mb()
        assert size > 0

    def test_stats_summary(self, engine):
        summary = engine.stats.summary()
        assert "No inference calls" in summary

        engine.benchmark(num_iterations=5)
        summary = engine.stats.summary()
        assert "5 calls" in summary


class TestInferenceStats:
    def test_empty(self):
        stats = InferenceStats()
        assert stats.avg_latency_ms == 0
        assert stats.num_calls == 0

    def test_record(self):
        stats = InferenceStats()
        stats.record(1.0)
        stats.record(2.0)
        stats.record(3.0)
        assert stats.num_calls == 3
        assert abs(stats.avg_latency_ms - 2.0) < 1e-6
        assert stats.min_latency_ms == 1.0
        assert stats.max_latency_ms == 3.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
