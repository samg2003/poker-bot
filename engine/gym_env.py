"""
OpenAI Gymnasium wrapper for NLHE Poker engine.
Allows massively parallel CPU simulation using gym.vector.AsyncVectorEnv.

Runs a single Poker Game per environment on CPU. 
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Any, Tuple

from training.nlhe_trainer import NLHETrainingConfig, NLHESelfPlayTrainer, TableState, OPP_GAME_STATE_DIM, PROFILE_DIM, ACTION_FEATURE_DIM, MAX_HAND_ACTIONS

class PokerGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_dict: dict, policy_state_dict: dict, opp_enc_state_dict: dict, frozen_pool_states: list):
        super().__init__()
        
        self.config = NLHETrainingConfig(**config_dict)
        self.config.device = 'cpu'
        torch.set_num_threads(1)  # CRITICAL: Prevent OpenMP thread contention across N parallel workers
        self.trainer = NLHESelfPlayTrainer(self.config)
        
        # Load weights locally to avoid IPC locking issues
        self.trainer.policy.load_state_dict(policy_state_dict)
        self.trainer.opponent_encoder.load_state_dict(opp_enc_state_dict)
        for p_idx, p_state in enumerate(frozen_pool_states):
            self.trainer._frozen_models[p_idx].load_state_dict(p_state)
            
        self.table = TableState()
        self.generator = None
        self.max_opps = self.config.max_players - 1
        self._stashed_opp_embed = None
        
        # Action space: Dict allowing PyTorch log_probs to flow back from Central GPU
        self.action_space = spaces.Dict({
            'action': spaces.MultiDiscrete([5, 10]),
            'probs': spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32),
            'value': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'sizing_probs': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        })
        
        # Observation space matches padded state dicts perfectly
        self.observation_space = spaces.Dict({
            'hole_cards': spaces.Box(low=0, high=52, shape=(2,), dtype=np.int64),
            'community_cards': spaces.Box(low=-1, high=52, shape=(5,), dtype=np.int64),
            'numeric_features': spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32),
            'opponent_stats': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_opps, 30), dtype=np.float32),
            'own_stats': spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32),
            'opponent_game_state': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_opps, OPP_GAME_STATE_DIM), dtype=np.float32),
            'hand_action_seq': spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_HAND_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32),
            'hand_action_len': spaces.Box(low=0, high=MAX_HAND_ACTIONS, shape=(1,), dtype=np.int64),
            'actor_profiles_seq': spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_HAND_ACTIONS, PROFILE_DIM), dtype=np.float32),
            'hero_profile': spaces.Box(low=-np.inf, high=np.inf, shape=(PROFILE_DIM,), dtype=np.float32),
            'opponent_profiles': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_opps, PROFILE_DIM), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8), 
            'sizing_mask': spaces.Box(low=0, high=1, shape=(10,), dtype=np.int8),
            'opponent_embeddings': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_opps, self.config.opponent_embed_dim), dtype=np.float32),
        })

    def update_weights(self, policy_sd, opp_enc_sd, pool_sds):
        self.trainer.policy.load_state_dict(policy_sd)
        self.trainer.opponent_encoder.load_state_dict(opp_enc_sd)
        for p_idx, p_state in enumerate(pool_sds):
            # Ensure the frozen model slot exists
            if p_idx not in self.trainer._frozen_models:
                import copy
                self.trainer._frozen_models[p_idx] = copy.deepcopy(self.trainer._frozen_template)
            self.trainer._frozen_models[p_idx].load_state_dict(p_state)

    def _pad_obs(self, state_dict: dict) -> dict:
        """Pads generator state to gym shape (removing batch=1)."""
        # Distribute Opponent Encoder inference to the CPU parallel workers!
        if 'opponent_embeddings' not in state_dict:
            cached = state_dict['_cached_embeds']
            for opid, history in state_dict['_uncached_opp_histories']:
                if not history:
                    emb = self.trainer.opponent_encoder.encode_empty(1, device="cpu")
                else:
                    seq = torch.stack(history)[-self.trainer.opponent_encoder.max_seq_len:]
                    with torch.no_grad():
                        emb = self.trainer.opponent_encoder(seq.unsqueeze(0).to("cpu"))
                cached[opid] = emb.detach()
                self.table.opp_embed_cache[opid] = emb.detach()
                
            ordered = [cached[opid] for opid in state_dict['_opp_ids']]
            if not ordered:
                emb_t = self.trainer.opponent_encoder.encode_empty(1, device="cpu").unsqueeze(1)
            else:
                emb_t = torch.cat(ordered, dim=0).unsqueeze(0)
            state_dict['opponent_embeddings'] = emb_t

        out = {}
        # Save opp_embed to be restored later
        if 'opponent_embeddings' in state_dict:
            self._stashed_opp_embed = state_dict['opponent_embeddings'].detach().cpu()
            
        for k, v in state_dict.items():
            if k.startswith('_'): continue
            
            if isinstance(v, torch.Tensor):
                v_cpu = v.cpu()
                if k not in ['hero_profile', 'opponent_profiles']:
                    v_sq = v_cpu.squeeze(0)  # remove batch=1
                else:
                    v_sq = v_cpu
                
                # Pad opponents dim
                if k in ['opponent_stats', 'opponent_game_state', 'opponent_profiles', 'opponent_embeddings']:
                    curr_opps = v_sq.shape[0]
                    pad_amt = self.max_opps - curr_opps
                    if pad_amt > 0:
                        pad_shape = list(v_sq.shape)
                        pad_shape[0] = pad_amt
                        zeros = torch.zeros(pad_shape, dtype=v_sq.dtype, device=v_sq.device)
                        v_sq = torch.cat([v_sq, zeros], dim=0)
                
                if k in ['action_mask', 'sizing_mask']:
                    # Ensure mask length is correct (action_mask is 5, sizing is 10)
                    v_sq = v_sq.to(torch.int8)
                
                out[k] = v_sq.numpy()
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.trainer.rng.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self._pending_experiences = getattr(self, '_pending_experiences', [])
        
        while True:
            self.generator = self.trainer._play_hand_gen(self.table)
            obs, reward, done, truncated, info = self._fast_forward(injected_tuple=None)
            if not done:
                return obs, info
            else:
                # The hand natively ended before the Hero could act.
                # We must stash its experiences and restart to provide a valid starting state for Gym.
                if 'experiences' in info:
                    self._pending_experiences.extend(info['experiences'])

    def step(self, action_dict):
        """Action dict contains central PyTorch answers."""
        # Note action_dict['action'] = [action_idx, sizing_idx] (not used directly in generator logic right now)
        # We just need to forward the GPU values back to the generator
        
        probs = torch.tensor(action_dict['probs'])
        value = float(action_dict['value'][0])
        sizing_probs = action_dict['sizing_probs'].tolist()
        
        injected_tuple = (probs, value, sizing_probs, self._stashed_opp_embed)
        return self._fast_forward(injected_tuple)

    def _fast_forward(self, injected_tuple=None):
        try:
            if injected_tuple is None:
                state_dict = next(self.generator)
            else:
                state_dict = self.generator.send(injected_tuple)
                
            # Auto-play Opponents CPU
            while 'frozen_idx' in state_dict:
                op_idx = state_dict['frozen_idx']
                frozen = self.trainer._frozen_models.get(op_idx, self.trainer._frozen_template)
                
                # Pad state dynamically to local batch=1
                obs_padded = self._pad_obs(state_dict)
                batch = {k: torch.tensor(v).unsqueeze(0) for k, v in obs_padded.items()}
                
                # Create mask
                curr_opps = state_dict['opponent_embeddings'].shape[1]
                m = torch.ones(1, self.max_opps, dtype=torch.bool)
                m[0, :curr_opps] = False
                batch['opponent_mask'] = m
                
                # Cast Gym int8 masks back to torch.bool for PyTorch masked_fill
                batch['action_mask'] = batch['action_mask'].to(torch.bool)
                batch['sizing_mask'] = batch['sizing_mask'].to(torch.bool)
                
                with torch.no_grad():
                    output = frozen(**batch)
                    
                opp_probs = output.action_type_probs[0]
                opp_val = output.value[0, 0].item() if hasattr(output, 'value') else 0.0
                opp_sizing = torch.softmax(output.bet_size_logits[0], dim=-1).tolist()
                opp_emb_tup = state_dict['opponent_embeddings']
                
                state_dict = self.generator.send((opp_probs, opp_val, opp_sizing, opp_emb_tup))

            # Hit a Hero node
            obs = self._pad_obs(state_dict)
            return obs, 0.0, False, False, {}

        except StopIteration as e:
            # Trajectory finished natively!
            all_player_exps = e.value[0]
            
            flat_experiences = []
            for pexp_list in all_player_exps:
                flat_experiences.extend(pexp_list)
            
            # Convert PyTorch tensors to Numpy to avoid Mac OS torch_shm_manager crash over IPC!
            np_experiences = getattr(self, '_pending_experiences', [])
            for exp in flat_experiences:
                d = exp._asdict()
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        d[k] = v.numpy()
                np_experiences.append(d)
            self._pending_experiences = []
                
            return self.observation_space.sample(), 0.0, True, False, {'experiences': np_experiences}
