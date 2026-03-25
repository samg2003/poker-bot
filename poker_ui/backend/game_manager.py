import random
import torch
import copy
from typing import Dict, List, Optional, Any

from engine.dealer import Dealer
from engine.game_state import GameState, Action, ActionType, Street
from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from model.stat_tracker import StatTracker, NUM_STAT_FEATURES, HandRecord
from model.action_space import ActionIndex, NUM_ACTION_TYPES, encode_action
from training.personality import SituationalPersonality, PersonalityModifier, detect_situations

class TimelineSnapshot:
    def __init__(
        self,
        game_state: GameState,
        action: Optional[Action],
        god_mode: Dict[int, Any],  # Dictionary mapping pid -> AI's predicted probabilities and EV
    ):
        # Deep copy game state
        self.game_state = copy.deepcopy(game_state)
        self.action = action
        self.god_mode = god_mode

class GameManager:
    """Manages the interactive web poker game against the AI."""
    def __init__(self, policy: PolicyNetwork, encoder: OpponentEncoder, human_seat: int = 0):
        self.policy = policy
        self.opponent_encoder = encoder
        self.rng = random.Random()
        self.human_seat = human_seat
        
        self.dealer: Optional[Dealer] = None
        self.game_state: Optional[GameState] = None
        self.personalities: List[Optional[SituationalPersonality]] = []
        self.hand_records: List[HandRecord] = []
        
        # Persistence across hands
        self.stat_tracker = StatTracker()
        self.action_histories: Dict[int, List[torch.Tensor]] = {}
        
        # Hand Timeline
        self.timeline: List[TimelineSnapshot] = []

    # ─────────────────────────────────────────────────────────
    # Evaluator Tracking Logic (Copied to avoid coupling)
    # ─────────────────────────────────────────────────────────
    def _action_to_type_idx(self, action: Action) -> int:
        mapping = {
            ActionType.FOLD: ActionIndex.FOLD,
            ActionType.CHECK: ActionIndex.CHECK,
            ActionType.CALL: ActionIndex.CALL,
            ActionType.RAISE: ActionIndex.RAISE,
            ActionType.ALL_IN: ActionIndex.RAISE,
        }
        return mapping.get(action.action_type, ActionIndex.FOLD)

    def _record_action(self, player_id: int, action_type: int,
                       bet_frac: float, pot: float, stack: float, street: int):
        if player_id not in self.action_histories:
            self.action_histories[player_id] = []
        token = encode_action(action_type, bet_frac, pot, stack, street)
        self.action_histories[player_id].append(token)
        if len(self.action_histories[player_id]) > 16:
            self.action_histories[player_id] = self.action_histories[player_id][-16:]

    def _get_opponent_embedding(self, player_id: int) -> torch.Tensor:
        if hasattr(self, '_current_embed_cache') and player_id in self._current_embed_cache:
            return self._current_embed_cache[player_id]

        history = self.action_histories.get(player_id, [])
        if not history:
            return self.opponent_encoder.encode_empty(1).detach()
        seq = torch.stack(history).unsqueeze(0)
        with torch.no_grad():
            return self.opponent_encoder(seq).detach()

    def _get_all_opponent_embeddings(self, hero_id: int, num_players: int) -> torch.Tensor:
        opp_embeds = []
        for pid in range(num_players):
            if pid == hero_id: continue
            opp_embeds.append(self._get_opponent_embedding(pid))
        if not opp_embeds:
            return self.opponent_encoder.encode_empty(1).unsqueeze(1).detach()
        return torch.cat(opp_embeds, dim=0).unsqueeze(0).detach()

    def _get_opponent_stats(self, hero_id: int, num_players: int) -> torch.Tensor:
        stats = []
        for pid in range(num_players):
            if pid == hero_id: continue
            stats.append(self.stat_tracker.get_stats(pid))
        if not stats:
            return torch.zeros(1, 1, NUM_STAT_FEATURES)
        return torch.stack(stats).unsqueeze(0)

    # ─────────────────────────────────────────────────────────
    # API Methods
    # ─────────────────────────────────────────────────────────
    def start_new_hand(self, num_players: int = 6):
        """Initializes a new hand and generates bot personalities."""
        self.personalities = []
        for pid in range(num_players):
            if pid == self.human_seat:
                self.personalities.append(None)
            else:
                # Randomize personality to make it interesting
                roll = self.rng.random()
                if roll < 0.6:
                    base = PersonalityModifier.gto()
                    base.name = "GTO"
                elif roll < 0.8:
                    base = PersonalityModifier.nit()
                    base.name = "Nit"
                else:
                    base = PersonalityModifier.maniac()
                    base.name = "Maniac"
                self.personalities.append(SituationalPersonality(base=base))

        self.dealer = Dealer(
            num_players=num_players,
            stacks=[100.0] * num_players,
            small_blind=0.5,
            big_blind=1.0,
            dealer_button=self.rng.randint(0, num_players - 1),
            seed=self.rng.randint(0, 2**31)
        )
        self.game_state = self.dealer.start_hand()
        self.hand_records = [HandRecord() for _ in range(num_players)]
        self.timeline = []
        
        # Take AI snapshot before first action
        self._take_snapshot(None)

    def _take_snapshot(self, last_action: Optional[Action]):
        """Records the current game state and the AI's God Mode evaluations."""
        terminal = self.dealer.is_hand_over()
        if terminal:
            self.timeline.append(TimelineSnapshot(self.game_state, last_action, {}))
            return

        # Precompute individual opponent embeddings to avoid O(N^2) encoder passes
        self._current_embed_cache = {}
        for pid in range(len(self.game_state.players)):
            self._current_embed_cache[pid] = self._get_opponent_embedding(pid)

        # Calculate God Mode for all active players
        god_mode = {}
        for pid, p in enumerate(self.game_state.players):
            if p.is_active:
                god_mode[pid] = self._evaluate_seat(pid)
                
        self.timeline.append(TimelineSnapshot(self.game_state, last_action, god_mode))
        self._current_embed_cache.clear()

    def _evaluate_seat(self, pid: int) -> Dict[str, Any]:
        """Runs the neural network exactly as if it was in the given seat."""
        p = self.game_state.players[pid]
        num_players = len(self.game_state.players)
        
        hole = torch.tensor([list(p.hole_cards)], dtype=torch.long)
        board = list(self.game_state.board)
        while len(board) < 5: board.append(-1)
        community = torch.tensor([board[:5]], dtype=torch.long)

        pot = self.game_state.pot / 100.0
        own_stack = p.stack / 100.0
        own_bet = p.bet_this_street / 100.0
        rel_pos = (pid - self.game_state.dealer_button) % num_players
        position = rel_pos / max(num_players - 1, 1)

        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        street_val = street_map.get(self.game_state.street, 0) / 3.0

        num_active = sum(1 for pp in self.game_state.players if pp.is_active)
        current_bet = self.game_state.current_bet / 100.0
        min_raise = self.game_state.min_raise / 100.0

        numeric = torch.tensor([[
            pot, own_stack, own_bet, position, street_val,
            num_players / 9.0, num_active / 9.0,
            current_bet, min_raise,
        ]], dtype=torch.float32)

        legal_types = self.game_state.get_legal_actions()
        mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool)
        for at in legal_types:
            if at == ActionType.FOLD: mask[0, ActionIndex.FOLD] = True
            elif at == ActionType.CHECK: mask[0, ActionIndex.CHECK] = True
            elif at == ActionType.CALL: mask[0, ActionIndex.CALL] = True
            elif at in (ActionType.RAISE, ActionType.ALL_IN): mask[0, ActionIndex.RAISE] = True

        opp_embed = self._get_all_opponent_embeddings(pid, num_players)
        opp_stats = self._get_opponent_stats(pid, num_players)
        own_stats = self.stat_tracker.get_stats(pid).unsqueeze(0)

        with torch.no_grad():
            out = self.policy(
                hole, community, numeric, opp_embed, opp_stats, own_stats, action_mask=mask
            )

        probs = out.action_type_probs[0].tolist()
        sizing = out.bet_sizing[0, 0].item()
        ev = out.value[0, 0].item()

        # Apply personality if it is a bot (for accurate visualization of what the bot will actually do)
        personality = self.personalities[pid]
        if personality is not None:
            situations = detect_situations(street=street_map.get(self.game_state.street, 0))
            probs_tensor = personality.apply(torch.tensor(probs), situations, hand_strength=0.5)
            probs = probs_tensor.tolist()

        return {
            'probs': probs,
            'sizing': sizing,
            'ev': ev,
            'legal_actions': [at.name for at in legal_types]
        }

    def process_action(self, action: Action):
        """Applies an action to the dealer and updates tracking."""
        pid = self.game_state.current_player_idx
        p = self.game_state.players[pid]
        
        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        street_integer = street_map.get(self.game_state.street, 0)

        self.dealer.apply_action(action)

        # Record History
        pot_base = self.game_state.pot
        bet_frac = action.amount / max(pot_base, 1.0) if action.amount > 0 else 0.0
        self._record_action(pid, self._action_to_type_idx(action), bet_frac, pot_base, p.stack, street_integer)
        
        if action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
            self.hand_records[pid].vpip = True
        if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            self.hand_records[pid].pfr = True

        # Take Snapshot
        self._take_snapshot(action)

        # End of Hand
        terminal = self.dealer.is_hand_over()
        if terminal:
            results = self.dealer.get_results()
            for i in range(len(self.game_state.players)):
                self.hand_records[i].result = results['profit'][i]
                self.hand_records[i].went_to_showdown = (
                    self.game_state.street == Street.SHOWDOWN and not self.game_state.players[i].is_folded
                )
                self.stat_tracker.record_hand(i, self.hand_records[i])

    def step_ai(self) -> bool:
        """Executes the next AI action if it's not the human's turn. Returns True if an action was taken."""
        terminal = self.dealer.is_hand_over()
        if terminal:
            return False
            
        pid = self.game_state.current_player_idx
        if pid == self.human_seat:
            return False

        # Get the God Mode evaluation we already calculated for this seat in the snapshot
        god_mode = self.timeline[-1].god_mode[pid]
        probs = god_mode['probs']
        sizing = god_mode['sizing']
        legal_types = self.game_state.get_legal_actions()
        
        # Sample action
        from torch.distributions import Categorical
        dist = Categorical(torch.tensor(probs))
        a_idx = dist.sample().item()

        # Map to Engine Action
        if a_idx == ActionIndex.FOLD and ActionType.FOLD in legal_types:
            act = Action(ActionType.FOLD)
        elif a_idx == ActionIndex.CHECK and ActionType.CHECK in legal_types:
            act = Action(ActionType.CHECK)
        elif a_idx == ActionIndex.CALL and ActionType.CALL in legal_types:
            act = Action(ActionType.CALL)
        elif a_idx == ActionIndex.RAISE and ActionType.RAISE in legal_types:
            min_r = self.game_state.get_min_raise_to()
            max_r = self.game_state.get_max_raise_to()
            rt = min_r + sizing * (max_r - min_r)
            act = Action(ActionType.RAISE, amount=max(min_r, min(rt, max_r)))
        else:
            if ActionType.CHECK in legal_types: act = Action(ActionType.CHECK)
            elif ActionType.CALL in legal_types: act = Action(ActionType.CALL)
            else: act = Action(ActionType.FOLD)

        self.process_action(act)
        return True
