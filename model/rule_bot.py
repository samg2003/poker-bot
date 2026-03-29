import torch
import torch.nn as nn
import torch.nn.functional as F
from model.policy_network import ActionOutput
from model.action_space import NUM_ACTION_TYPES

class RuleBasedBot(nn.Module):
    """
    A hardcoded, Tight-Aggressive (TAG) poker bot implemented as a PyTorch Module.
    This allows it to be perfectly drop-in compatible with the PyTorch batching pipeline
    used in both the trainer and the UI API.
    """
    def __init__(self):
        super().__init__()
        # Dummy parameter so it can be moved to GPU seamlessly
        self.dummy = nn.Parameter(torch.zeros(1))

    def _eval_preflop(self, c1, c2):
        """Vectorized preflop hand strength evaluation [0.0 - 1.0]."""
        r1 = c1 // 4
        r2 = c2 // 4
        is_pair = (r1 == r2)
        is_suited = (c1 % 4) == (c2 % 4)
        max_r = torch.max(r1, r2)
        min_r = torch.min(r1, r2)

        # Baseline Chen formula approximation
        strength = torch.zeros_like(r1, dtype=torch.float32)
        
        # Pairs: 0.5 to 1.0 (AA=1.0)
        strength = torch.where(is_pair, 0.5 + (r1 * 0.04), strength)
        
        # Non-pairs
        high_card_val = max_r * 0.03
        gap = max_r - min_r
        gap_penalty = torch.clamp((gap - 1) * 0.02, min=0.0)
        suited_bonus = torch.where(is_suited, 0.08, 0.0)
        
        non_pair_str = 0.2 + high_card_val + suited_bonus - gap_penalty
        strength = torch.where(~is_pair, non_pair_str, strength)
        
        # Premium hands (AK, AQ) manual bump
        is_premium_unpaired = (~is_pair) & (min_r >= 10)
        strength = torch.where(is_premium_unpaired, strength + 0.15, strength)
        
        return torch.clamp(strength, 0.0, 1.0)

    def _eval_postflop(self, c1, c2, board):
        """
        Extremely basic vectorized postflop heuristic. 
        Detects pairs with the board.
        """
        r1 = c1 // 4
        r2 = c2 // 4
        
        # Mask out absent cards (-1)
        valid_board = board >= 0
        board_ranks = torch.where(valid_board, board // 4, torch.full_like(board, -1))
        
        # Check for pair with board
        hit_r1 = (board_ranks == r1.unsqueeze(-1)).any(dim=-1)
        hit_r2 = (board_ranks == r2.unsqueeze(-1)).any(dim=-1)
        hit_pair = hit_r1 | hit_r2
        
        # Check for pocket pair overcards
        is_pocket_pair = (r1 == r2)
        max_board = torch.max(board_ranks, dim=-1)[0]
        overpair = is_pocket_pair & (r1 > max_board)
        
        strength = torch.full_like(r1, 0.2, dtype=torch.float32) # Default weak
        strength = torch.where(hit_pair, 0.6 + (torch.max(r1, r2) * 0.01), strength)
        strength = torch.where(overpair, 0.8 + (r1 * 0.01), strength)
        
        return torch.clamp(strength, 0.0, 1.0)

    def forward(
        self,
        hole_cards,            
        community_cards,        
        numeric_features,      
        action_mask,   
        sizing_mask,   
        **kwargs
    ):
        batch_size = hole_cards.shape[0]
        device = hole_cards.device
        
        c1 = hole_cards[:, 0]
        c2 = hole_cards[:, 1]
        
        # Detect street by how many board cards are visible
        valid_board_count = (community_cards >= 0).sum(dim=-1)
        is_preflop = (valid_board_count < 3)
        
        # Calculate heuristics
        pre_str = self._eval_preflop(c1, c2)
        post_str = self._eval_postflop(c1, c2, community_cards)
        
        hand_strength = torch.where(is_preflop, pre_str, post_str)
        
        # Base logical logits
        # 0=Fold, 1=Check, 2=Call, 3=Raise
        logits = torch.full((batch_size, 4), -10.0, device=device)
        
        # Strategy Logic (TAG)
        # Fold is always slightly viable fallback depending on strength
        logits[:, 0] = (1.0 - hand_strength) * 5.0
        
        # If strength > 0.4, checking/calling becomes good
        can_continue = hand_strength > 0.4
        logits[:, 1] = torch.where(can_continue, 5.0 * hand_strength, logits[:, 1])
        logits[:, 2] = torch.where(can_continue, 4.0 * hand_strength, logits[:, 2])
        
        # If strength > 0.65, raising becomes prime
        can_raise = hand_strength > 0.65
        logits[:, 3] = torch.where(can_raise, 10.0 * hand_strength, logits[:, 3])
        
        # Apply strict action mask
        logits = logits.masked_fill(~action_mask.bool(), float('-inf'))
        
        # Sizing logic: Pick standard raise sizes (1.5x, 2.0x pot) if strong
        sz_logits = torch.full((batch_size, 10), -10.0, device=device)
        
        # Bucket 7 is 1.5x, Bucket 8 is 2.0x, Bucket 9 is All-in
        sz_logits[:, 7] = 5.0
        sz_logits[:, 8] = 3.0
        sz_logits[:, 9] = torch.where(hand_strength > 0.85, 2.0, -10.0) # Only shove absolute nuts
        
        sz_logits = sz_logits.masked_fill(~sizing_mask.bool(), float('-inf'))
        
        probs = F.softmax(logits, dim=-1)
        
        # Ensure we return valid logits even if sizing is all blocked
        sz_probs = F.softmax(sz_logits, dim=-1)
        sz_probs = torch.where(torch.isnan(sz_probs), torch.zeros_like(sz_probs), sz_probs)
        
        return ActionOutput(
            action_type_logits=logits,
            action_type_probs=probs,
            bet_size_logits=sz_logits,
            value=torch.zeros(batch_size, 1, device=device)
        )
