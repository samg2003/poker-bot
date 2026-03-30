"""
PPO Updater — pure gradient computation for PPO training.

Extracted from NLHESelfPlayTrainer. Functions are stateless —
they take experiences + config and return losses.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.distributions import Categorical

from model.action_space import ActionIndex, POT_FRACTIONS


def compute_gae(
    trajectory: list,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    v_res_alpha: float = 1.0,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute GAE with street-scoped reward shaping.

    Key design choices (poker-specific):
    - r=0 for all intermediate steps (cost embedded in equity_x_pot via -bet_total)
    - At street boundaries: bootstrap from V(end_of_street) instead of V(next_street)
      to prevent new-card variance from contaminating current street's advantages
    - GAE lambda chain resets at street boundaries — each street's advantages
      accumulate independently (no river variance in preflop advantages)
    - v_res_alpha scales V_res influence on policy: 0.0=pure equity, 1.0=full V_res
    - Value training always uses alpha=1.0 (full V_res) regardless of v_res_alpha

    V(s) = equity_x_pot(s) + alpha * V_res(s)
    Returns (advantages, returns_for_value_training, returns_alpha_scaled).
    """
    n = len(trajectory)
    if n == 0:
        return [], [], []

    # Alpha-scaled values for advantages (policy)
    alpha = v_res_alpha
    values = [exp.equity_x_pot + alpha * exp.value for exp in trajectory]
    # Full values for value training (always alpha=1)
    full_values = [exp.equity_x_pot + exp.value for exp in trajectory]
    terminal_reward = trajectory[-1].reward

    # TD errors with street-scoped bootstrapping (using alpha-scaled values for policy)
    deltas = []
    for t in range(n):
        if t == n - 1:
            # Terminal: actual profit
            deltas.append(terminal_reward - values[t])
        elif trajectory[t].street_idx != trajectory[t + 1].street_idx:
            # Street boundary: bootstrap from V(end_of_street) BEFORE new card
            v_end = trajectory[t].end_street_equity_x_pot + alpha * trajectory[t].v_res_end_of_street
            deltas.append(v_end - values[t])
        else:
            # Same street: r=0, normal TD bootstrap
            deltas.append(gamma * values[t + 1] - values[t])

    # Full TD errors for value training (alpha=1)
    full_deltas = []
    for t in range(n):
        if t == n - 1:
            full_deltas.append(terminal_reward - full_values[t])
        elif trajectory[t].street_idx != trajectory[t + 1].street_idx:
            v_end_full = trajectory[t].end_street_equity_x_pot + trajectory[t].v_res_end_of_street
            full_deltas.append(v_end_full - full_values[t])
        else:
            full_deltas.append(gamma * full_values[t + 1] - full_values[t])

    # GAE with per-street reset for policy advantages
    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1 or trajectory[t].street_idx != trajectory[t + 1].street_idx:
            gae = deltas[t]
        else:
            gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    # GAE with per-street reset for value training (alpha=1)
    full_advantages = [0.0] * n
    full_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1 or trajectory[t].street_idx != trajectory[t + 1].street_idx:
            full_gae = full_deltas[t]
        else:
            full_gae = full_deltas[t] + gamma * gae_lambda * full_gae
        full_advantages[t] = full_gae

    # Returns for value training: G_t = A_t + V(s_t) with alpha=1
    full_returns = [full_advantages[t] + full_values[t] for t in range(n)]

    return advantages, full_returns, [advantages[t] + values[t] for t in range(n)]


def precompute_ppo_data(experiences: list, device: torch.device) -> dict:
    """Stack all experience tensors into batched format for PPO mini-batching.

    Returns a dict of batched tensors on CPU (moved to device during loss computation).
    """
    n = len(experiences)

    # Stack observation tensors
    hole_cards = torch.cat([e.hole_cards for e in experiences], dim=0)
    community = torch.cat([e.community_cards for e in experiences], dim=0)
    numeric = torch.cat([e.numeric_features for e in experiences], dim=0)
    own_stats = torch.cat([e.own_stats for e in experiences], dim=0)
    action_masks = torch.cat([e.action_mask for e in experiences], dim=0)
    sizing_masks = torch.cat([e.sizing_mask for e in experiences], dim=0)

    # Opponent tensors — pad to max num opponents
    max_opp = max(e.opponent_embeddings.shape[1] for e in experiences)
    embed_dim = experiences[0].opponent_embeddings.shape[-1]
    stat_dim = experiences[0].opponent_stats.shape[-1]

    opp_embeds = torch.zeros(n, max_opp, embed_dim)
    opp_stats_t = torch.zeros(n, max_opp, stat_dim)
    from model.policy_network import OPP_GAME_STATE_DIM, PROFILE_DIM
    opp_gs = torch.zeros(n, max_opp, OPP_GAME_STATE_DIM)
    opp_masks = torch.ones(n, max_opp, dtype=torch.bool)
    opp_profiles = torch.zeros(n, max_opp, PROFILE_DIM)

    for i, e in enumerate(experiences):
        k = e.opponent_embeddings.shape[1]
        opp_embeds[i, :k] = e.opponent_embeddings[0, :k]
        opp_stats_t[i, :k] = e.opponent_stats[0, :k]
        opp_gs[i, :k] = e.opponent_game_state[0, :k]
        opp_masks[i, :k] = False
        opp_profiles[i, :k] = e.opponent_profiles[0, :k]

    # Hand history tensors
    from model.policy_network import MAX_HAND_ACTIONS
    ha_feat_dim = experiences[0].hand_action_seq.shape[-1]
    hand_action_seqs = torch.zeros(n, MAX_HAND_ACTIONS, ha_feat_dim)
    hand_action_lens = torch.zeros(n, dtype=torch.long)
    actor_profiles_seqs = torch.zeros(n, MAX_HAND_ACTIONS, PROFILE_DIM)
    hero_profiles = torch.zeros(n, PROFILE_DIM)

    for i, e in enumerate(experiences):
        seq_len = min(e.hand_action_seq.shape[1], MAX_HAND_ACTIONS)
        hand_action_seqs[i, :seq_len] = e.hand_action_seq[0, :seq_len]
        hand_action_lens[i] = e.hand_action_len.item() if e.hand_action_len.dim() == 0 else e.hand_action_len[0].item()
        ap_len = min(e.actor_profiles_seq.shape[1], MAX_HAND_ACTIONS)
        actor_profiles_seqs[i, :ap_len] = e.actor_profiles_seq[0, :ap_len]
        hero_profiles[i] = e.hero_profile[0]

    # Scalar tensors
    action_t = torch.tensor([e.action_idx for e in experiences], dtype=torch.long)
    sizing_t = torch.tensor([e.sizing_idx for e in experiences], dtype=torch.long)
    old_log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32)
    old_action_log_probs = torch.tensor([e.action_log_prob for e in experiences], dtype=torch.float32)
    old_sizing_log_probs = torch.tensor([e.sizing_log_prob for e in experiences], dtype=torch.float32)
    equity_x_pot = torch.tensor([e.equity_x_pot for e in experiences], dtype=torch.float32)

    return {
        'hole_cards': hole_cards,
        'community': community,
        'numeric': numeric,
        'own_stats': own_stats,
        'action_masks': action_masks,
        'sizing_masks': sizing_masks,
        'opp_embeds': opp_embeds,
        'opp_stats': opp_stats_t,
        'opp_gs': opp_gs,
        'opp_masks': opp_masks,
        'opp_profiles': opp_profiles,
        'hand_action_seqs': hand_action_seqs,
        'hand_action_lens': hand_action_lens,
        'actor_profiles_seqs': actor_profiles_seqs,
        'hero_profiles': hero_profiles,
        'action_t': action_t,
        'sizing_t': sizing_t,
        'old_log_probs': old_log_probs,
        'old_action_log_probs': old_action_log_probs,
        'old_sizing_log_probs': old_sizing_log_probs,
        'equity_x_pot': equity_x_pot,
    }


def compute_ppo_loss(
    policy,
    data: dict,
    indices: List[int],
    device: torch.device,
    ppo_clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.005,
    remove_clip: bool = False,
    kl_beta: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Compute PPO loss on a mini-batch. Returns (total_loss, action_loss, sizing_loss, value_loss)."""
    idx = torch.tensor(indices, dtype=torch.long)

    # Slice all tensors — move to device
    hole_cards = data['hole_cards'][idx].to(device)
    community = data['community'][idx].to(device)
    numeric = data['numeric'][idx].to(device)
    own_stats = data['own_stats'][idx].to(device)
    action_masks = data['action_masks'][idx].to(device)
    sizing_masks = data['sizing_masks'][idx].to(device)
    opp_embeds = data['opp_embeds'][idx].to(device)
    opp_stats = data['opp_stats'][idx].to(device)
    opp_gs = data['opp_gs'][idx].to(device)
    opp_masks = data['opp_masks'][idx].to(device)
    hand_action_seqs = data['hand_action_seqs'][idx].to(device)
    hand_action_lens = data['hand_action_lens'][idx].to(device)
    actor_profiles_seqs = data['actor_profiles_seqs'][idx].to(device)
    hero_profiles = data['hero_profiles'][idx].to(device)
    opp_profiles = data['opp_profiles'][idx].to(device)
    action_t = data['action_t'][idx].to(device)
    sizing_t = data['sizing_t'][idx].to(device)
    old_action_log_probs = data['old_action_log_probs'][idx].to(device)
    old_sizing_log_probs = data['old_sizing_log_probs'][idx].to(device)
    gae_advantages = data['gae_advantages'][idx].to(device)
    gae_returns = data['gae_returns'][idx].to(device)

    # Forward pass
    output = policy(
        hole_cards=hole_cards,
        community_cards=community,
        numeric_features=numeric,
        opponent_embeddings=opp_embeds,
        opponent_stats=opp_stats,
        own_stats=own_stats,
        opponent_game_state=opp_gs,
        action_mask=action_masks,
        sizing_mask=sizing_masks,
        opponent_mask=opp_masks,
        hand_action_seq=hand_action_seqs,
        hand_action_len=hand_action_lens,
        actor_profiles_seq=actor_profiles_seqs,
        hero_profile=hero_profiles,
        opponent_profiles=opp_profiles,
    )

    # Decoupled PPO: action type and sizing get independent credit
    dist = Categorical(output.action_type_probs)
    action_log_probs = dist.log_prob(action_t)
    action_entropy = dist.entropy().mean()

    is_raise = (action_t == ActionIndex.RAISE)

    # Action head ratio
    action_ratio = torch.exp(action_log_probs - old_action_log_probs)

    if remove_clip:
        approx_kl = old_action_log_probs - action_log_probs
        action_surr = action_ratio * gae_advantages
        action_loss = -action_surr.mean() + (kl_beta * approx_kl.mean())
    else:
        surr1 = action_ratio * gae_advantages
        surr2 = torch.clamp(action_ratio, 1 - ppo_clip, 1 + ppo_clip) * gae_advantages
        action_surr = torch.min(surr1, surr2)
        dual_clip_bound = torch.where(
            gae_advantages < 0, 3.0 * gae_advantages,
            torch.full_like(gae_advantages, -float('inf'))
        )
        action_loss = -torch.max(action_surr, dual_clip_bound).mean()

    # Sizing head PPO — only on raise experiences
    if is_raise.any():
        raise_logits = output.bet_size_logits[is_raise]
        raise_sizing_t = sizing_t[is_raise]
        raise_old_slp = old_sizing_log_probs[is_raise]
        raise_adv = gae_advantages[is_raise]

        sizing_dist = Categorical(logits=raise_logits)
        sizing_log_probs = sizing_dist.log_prob(raise_sizing_t)
        sizing_entropy = sizing_dist.entropy().mean()

        sizing_ratio = torch.exp(sizing_log_probs - raise_old_slp)

        if remove_clip:
            s_approx_kl = raise_old_slp - sizing_log_probs
            s_surr = sizing_ratio * raise_adv
            sizing_loss = -s_surr.mean() + (kl_beta * s_approx_kl.mean())
        else:
            s_surr1 = sizing_ratio * raise_adv
            s_surr2 = torch.clamp(sizing_ratio, 1 - ppo_clip, 1 + ppo_clip) * raise_adv
            s_surr = torch.min(s_surr1, s_surr2)
            raise_dual_clip = torch.where(
                raise_adv < 0, 3.0 * raise_adv,
                torch.full_like(raise_adv, -float('inf'))
            )
            sizing_loss = -torch.max(s_surr, raise_dual_clip).mean()
    else:
        sizing_loss = torch.tensor(0.0, device=device)
        sizing_entropy = torch.tensor(0.0, device=device)

    entropy = action_entropy + sizing_entropy

    value_pred = output.value.squeeze(-1)
    equity_x_pot_t = data['equity_x_pot'][idx].to(device)
    residual_returns = gae_returns - equity_x_pot_t
    value_loss = torch.nn.functional.smooth_l1_loss(value_pred, residual_returns)

    loss = (
        action_loss
        + sizing_loss
        + value_coef * value_loss
        - entropy_coef * entropy
    )

    loss.backward()

    return loss.item(), action_loss.item(), sizing_loss.item(), value_loss.item()


def count_action_distribution(experiences: list) -> Dict[str, float]:
    """Compute action choice rates WHEN each action was legal (conditional %).

    Returns dict with keys: fold, check, call, raise, allin, deep_ai
    """
    allin_idx = len(POT_FRACTIONS) - 1

    chosen = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'allin': 0}
    legal = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'allin': 0}
    deep_raise_count = 0
    deep_allin_count = 0

    for e in experiences:
        mask = e.action_mask.squeeze(0)
        if mask[ActionIndex.FOLD]:
            legal['fold'] += 1
        if mask[ActionIndex.CHECK]:
            legal['check'] += 1
        if mask[ActionIndex.CALL]:
            legal['call'] += 1
        if mask[ActionIndex.RAISE]:
            legal['raise'] += 1
            legal['allin'] += 1

        if e.action_idx == ActionIndex.FOLD:
            chosen['fold'] += 1
        elif e.action_idx == ActionIndex.CHECK:
            chosen['check'] += 1
        elif e.action_idx == ActionIndex.CALL:
            chosen['call'] += 1
        elif e.action_idx == ActionIndex.RAISE:
            if e.sizing_idx == allin_idx:
                chosen['allin'] += 1
            else:
                chosen['raise'] += 1
            if e.hero_stack_bb > 50:
                deep_raise_count += 1
                if e.sizing_idx == allin_idx:
                    deep_allin_count += 1

    rates = {}
    for k in chosen:
        rates[k] = (chosen[k] / legal[k] * 100) if legal[k] > 0 else 0.0
    rates['deep_ai'] = (deep_allin_count / max(deep_raise_count, 1)) * 100
    return rates
