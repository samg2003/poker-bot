import random
import torch
import copy
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from engine.dealer import Dealer
from engine.game_state import GameState, Action, ActionType, Street
from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from model.stat_tracker import StatTracker, NUM_STAT_FEATURES, HandRecord
from model.action_space import ActionIndex, NUM_ACTION_TYPES, encode_action
from training.personality import SituationalPersonality, PersonalityModifier, detect_situations

# ─── Bot name pool ───
BOT_NAMES = [
    "Ace", "Blitz", "Cobra", "Dice", "Edge", "Fox", "Ghost", "Hawk",
    "Ice", "Joker", "King", "Luna", "Maverick", "Neon", "Onyx", "Phantom",
    "Queen", "Raven", "Shadow", "Titan", "Viper", "Wolf", "Zen", "Storm",
]

@dataclass
class SeatInfo:
    """Persistent info about a player sitting at the table."""
    occupied: bool = False
    is_human: bool = False
    name: str = ""
    personality: Optional[SituationalPersonality] = None
    personality_name: str = "Bot"
    stack: float = 100.0
    hands_remaining: int = 30      # How many more hands before leaving
    uid: int = 0                    # Unique ID for stat tracking
    sitting_out: bool = False       # Sat out this hand (just joined mid-hand)


class TimelineSnapshot:
    def __init__(
        self,
        game_state: GameState,
        action: Optional[Action],
        god_mode: Dict[int, Any],
        seat_map: List[int],     # Maps engine player idx → table seat idx
    ):
        self.game_state = copy.deepcopy(game_state)
        self.action = action
        self.god_mode = god_mode
        self.seat_map = list(seat_map)


class GameManager:
    """Manages an interactive web poker game with a persistent 9-seat table."""
    MAX_SEATS = 9

    def __init__(self, policy: PolicyNetwork, encoder: OpponentEncoder, human_seat: int = 0):
        self.policy = policy
        self.opponent_encoder = encoder
        self.rng = random.Random()
        self.human_seat = human_seat
        
        self.dealer: Optional[Dealer] = None
        self.game_state: Optional[GameState] = None
        self.hand_records: List[HandRecord] = []

        # Persistent lobby state
        self.seats: List[SeatInfo] = [SeatInfo() for _ in range(self.MAX_SEATS)]
        self._next_uid = 1
        self.hand_count = 0
        self.dealer_button_seat = 0  # Tracks dealer across variable player counts
        self.total_buyin = 100.0     # Track hero's total money invested

        # Persistence across hands (keyed by uid)
        self.stat_tracker = StatTracker()
        self.action_histories: Dict[int, List[torch.Tensor]] = {}

        # Current hand mapping
        self.seat_map: List[int] = []       # engine idx → table seat idx
        self.engine_map: Dict[int, int] = {}  # table seat idx → engine idx
        
        # Hand Timeline
        self.timeline: List[TimelineSnapshot] = []

        # Initialize human seat
        self._seat_human()

    # ─────────────────────────────────────────────────────────
    # Lobby Management
    # ─────────────────────────────────────────────────────────
    def _seat_human(self):
        s = self.seats[self.human_seat]
        s.occupied = True
        s.is_human = True
        s.name = "Hero"
        s.personality = None
        s.personality_name = "Human"
        s.stack = 100.0
        s.hands_remaining = 999999
        s.uid = 0
        self.total_buyin = 100.0

    def _random_personality(self) -> tuple:
        roll = self.rng.random()
        if roll < 0.90:
            base = PersonalityModifier.gto()
            return SituationalPersonality(base=base), "GTO"
        elif roll < 0.92:
            base = PersonalityModifier.tag()
            return SituationalPersonality(base=base), "TAG"
        elif roll < 0.94:
            base = PersonalityModifier.nit()
            return SituationalPersonality(base=base), "Nit"
        elif roll < 0.96:
            base = PersonalityModifier.lag()
            return SituationalPersonality(base=base), "LAG"
        elif roll < 0.97:
            base = PersonalityModifier.maniac()
            return SituationalPersonality(base=base), "Maniac"
        elif roll < 0.985:
            base = PersonalityModifier.calling_station()
            return SituationalPersonality(base=base), "Station"
        else:
            base = PersonalityModifier.fish()
            return SituationalPersonality(base=base), "Fish"

    def _random_buyin(self) -> float:
        """Most players buy in at 100bb, some short-stack, rare deep-stack."""
        roll = self.rng.random()
        if roll < 0.65:
            return 100.0
        elif roll < 0.80:
            # Short stack: 30-60bb
            return round(self.rng.uniform(30, 60), 1)
        elif roll < 0.92:
            # Medium: 60-100bb
            return round(self.rng.uniform(60, 100), 1)
        else:
            # Deep stack: 150-250bb
            return round(self.rng.uniform(150, 250), 1)

    def _seat_bot(self, seat_idx: int):
        """Seat a new bot at the given index."""
        s = self.seats[seat_idx]
        personality, pname = self._random_personality()
        name = self.rng.choice(BOT_NAMES)
        
        s.occupied = True
        s.is_human = False
        s.name = name
        s.personality = personality
        s.personality_name = pname
        s.stack = self._random_buyin()
        s.hands_remaining = max(8, int(self.rng.gauss(30, 12)))
        s.uid = self._next_uid
        s.sitting_out = False
        self._next_uid += 1

    def _unseat(self, seat_idx: int):
        """Remove a bot from the table."""
        s = self.seats[seat_idx]
        s.occupied = False
        s.is_human = False
        s.name = ""
        s.personality = None
        s.personality_name = ""
        s.stack = 0
        s.hands_remaining = 0
        s.uid = 0
        s.sitting_out = False

    def _active_count(self) -> int:
        return sum(1 for s in self.seats if s.occupied)

    def _lobby_churn(self):
        """Between hands: bots leave and join to simulate a real cash game."""
        # --- Departures ---
        for i, s in enumerate(self.seats):
            if i == self.human_seat or not s.occupied:
                continue
            s.hands_remaining -= 1
            # Leave if out of hands or busted
            if s.hands_remaining <= 0 or s.stack < 1.0:
                self._unseat(i)
                continue
            # Small random chance to leave early
            if self._active_count() > 4 and self.rng.random() < 0.03:
                self._unseat(i)

        # --- Arrivals ---
        target = self.rng.choices([5, 6, 7, 8], weights=[15, 45, 30, 10])[0]
        empty_seats = [i for i in range(self.MAX_SEATS) if not self.seats[i].occupied]
        self.rng.shuffle(empty_seats)

        while self._active_count() < target and empty_seats:
            seat_idx = empty_seats.pop()
            self._seat_bot(seat_idx)

        # Ensure at least 2 players (hero + 1 bot)
        if self._active_count() < 2:
            for i in empty_seats:
                self._seat_bot(i)
                break

    def _build_seat_map(self) -> List[int]:
        """Build mapping from engine player index to table seat index."""
        return [i for i in range(self.MAX_SEATS) if self.seats[i].occupied]

    # ─────────────────────────────────────────────────────────
    # Evaluator Tracking Logic
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

    def _record_action(self, uid: int, action_type: int,
                       bet_frac: float, pot: float, stack: float, street: int):
        if uid not in self.action_histories:
            self.action_histories[uid] = []
        token = encode_action(action_type, bet_frac, pot, stack, street)
        self.action_histories[uid].append(token)
        if len(self.action_histories[uid]) > 16:
            self.action_histories[uid] = self.action_histories[uid][-16:]

    def _get_opponent_embedding(self, uid: int) -> torch.Tensor:
        if hasattr(self, '_current_embed_cache') and uid in self._current_embed_cache:
            return self._current_embed_cache[uid]

        history = self.action_histories.get(uid, [])
        if not history:
            return self.opponent_encoder.encode_empty(1).detach()
        seq = torch.stack(history).unsqueeze(0)
        with torch.no_grad():
            return self.opponent_encoder(seq).detach()

    def _get_all_opponent_embeddings(self, hero_uid: int, active_uids: List[int]) -> torch.Tensor:
        opp_embeds = []
        for uid in active_uids:
            if uid == hero_uid: continue
            opp_embeds.append(self._get_opponent_embedding(uid))
        if not opp_embeds:
            return self.opponent_encoder.encode_empty(1).unsqueeze(1).detach()
        return torch.cat(opp_embeds, dim=0).unsqueeze(0).detach()

    def _get_opponent_stats(self, hero_uid: int, active_uids: List[int]) -> torch.Tensor:
        stats = []
        for uid in active_uids:
            if uid == hero_uid: continue
            stats.append(self.stat_tracker.get_stats(uid))
        if not stats:
            return torch.zeros(1, 1, NUM_STAT_FEATURES)
        return torch.stack(stats).unsqueeze(0)

    # ─────────────────────────────────────────────────────────
    # API Methods
    # ─────────────────────────────────────────────────────────
    def start_new_hand(self):
        """Run lobby churn, then deal a new hand to all seated players."""
        self.hand_count += 1

        # First hand: fill the table
        if self.hand_count == 1:
            for i in range(self.MAX_SEATS):
                if i != self.human_seat and not self.seats[i].occupied:
                    self._seat_bot(i)
            # Trim to ~6
            occupied = [i for i in range(self.MAX_SEATS) if self.seats[i].occupied and i != self.human_seat]
            self.rng.shuffle(occupied)
            while self._active_count() > 6 and occupied:
                self._unseat(occupied.pop())
        else:
            self._lobby_churn()

        # Build engine mappings
        self.seat_map = self._build_seat_map()
        self.engine_map = {seat: eng for eng, seat in enumerate(self.seat_map)}
        num_engine_players = len(self.seat_map)

        # Auto top-up hero if busted to avoid engine crash
        if self.seats[self.human_seat].stack < 1.0:
            self.buy_in()

        # Build stacks and personalities arrays for the engine
        stacks = [self.seats[s].stack for s in self.seat_map]

        # Advance dealer button to next occupied seat
        self.dealer_button_seat = self._next_occupied_seat(self.dealer_button_seat)
        dealer_engine_idx = self.engine_map[self.dealer_button_seat]

        self.dealer = Dealer(
            num_players=num_engine_players,
            stacks=stacks,
            small_blind=0.5,
            big_blind=1.0,
            dealer_button=dealer_engine_idx,
            seed=self.rng.randint(0, 2**31)
        )
        self.game_state = self.dealer.start_hand()
        self.hand_records = [HandRecord() for _ in range(num_engine_players)]
        self.timeline = []

        # Per-hand stat tracking state
        self._pf_raise_count = 0
        self._pf_callers_after_raise = 0
        self._pf_aggressor = -1
        self._pf_has_raise = False
        self._player_checked_this_street = [False] * num_engine_players
        self._first_bet_this_street_by = -1
        self._current_street = Street.PREFLOP

        # Sync stacks back (blinds have been posted)
        for eng_idx, seat_idx in enumerate(self.seat_map):
            self.seats[seat_idx].stack = self.game_state.players[eng_idx].stack

        self._take_snapshot(None)

    def _next_occupied_seat(self, current_seat: int) -> int:
        """Find the next occupied seat clockwise."""
        for offset in range(1, self.MAX_SEATS + 1):
            idx = (current_seat + offset) % self.MAX_SEATS
            if self.seats[idx].occupied:
                return idx
        return current_seat

    def _take_snapshot(self, last_action: Optional[Action]):
        """Records the current game state and the AI's God Mode evaluations."""
        terminal = self.dealer.is_hand_over()
        if terminal:
            self.timeline.append(TimelineSnapshot(self.game_state, last_action, {}, self.seat_map))
            return

        self._current_embed_cache = {}
        active_uids = [self.seats[s].uid for s in self.seat_map]
        for uid in active_uids:
            self._current_embed_cache[uid] = self._get_opponent_embedding(uid)

        god_mode = {}
        for eng_idx, seat_idx in enumerate(self.seat_map):
            p = self.game_state.players[eng_idx]
            if p.is_active:
                god_mode[eng_idx] = self._evaluate_seat(eng_idx)

        self.timeline.append(TimelineSnapshot(self.game_state, last_action, god_mode, self.seat_map))
        self._current_embed_cache.clear()

    def _evaluate_seat(self, eng_idx: int) -> Dict[str, Any]:
        """Runs the neural network as if it was in the given engine seat."""
        p = self.game_state.players[eng_idx]
        seat_idx = self.seat_map[eng_idx]
        seat_info = self.seats[seat_idx]
        num_engine_players = len(self.seat_map)
        active_uids = [self.seats[s].uid for s in self.seat_map]

        hole = torch.tensor([list(p.hole_cards)], dtype=torch.long)
        board = list(self.game_state.board)
        while len(board) < 5: board.append(-1)
        community = torch.tensor([board[:5]], dtype=torch.long)

        pot = self.game_state.pot / 100.0
        own_stack = p.stack / 100.0
        own_bet = p.bet_this_street / 100.0
        rel_pos = (eng_idx - self.game_state.dealer_button) % num_engine_players
        position = rel_pos / max(num_engine_players - 1, 1)

        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        street_val = street_map.get(self.game_state.street, 0) / 3.0

        num_active = sum(1 for pp in self.game_state.players if pp.is_active)
        current_bet = self.game_state.current_bet / 100.0
        min_raise = self.game_state.min_raise / 100.0
        amount_to_call = max(0.0, current_bet - own_bet)

        numeric = torch.tensor([[
            pot, own_stack, own_bet, position, street_val,
            num_engine_players / 9.0, num_active / 9.0,
            current_bet, min_raise, amount_to_call
        ]], dtype=torch.float32)

        legal_types = self.game_state.get_legal_actions()
        mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool)
        for at in legal_types:
            if at == ActionType.FOLD: mask[0, ActionIndex.FOLD] = True
            elif at == ActionType.CHECK: mask[0, ActionIndex.CHECK] = True
            elif at == ActionType.CALL: mask[0, ActionIndex.CALL] = True
            elif at in (ActionType.RAISE, ActionType.ALL_IN): mask[0, ActionIndex.RAISE] = True

        opp_embed = self._get_all_opponent_embeddings(seat_info.uid, active_uids)
        opp_stats = self._get_opponent_stats(seat_info.uid, active_uids)
        own_stats = self.stat_tracker.get_stats(seat_info.uid).unsqueeze(0)

        from model.action_space import get_sizing_mask
        sizing_mask = get_sizing_mask(self.game_state).unsqueeze(0)

        with torch.no_grad():
            out = self.policy(
                hole, community, numeric, opp_embed, opp_stats, own_stats, action_mask=mask, sizing_mask=sizing_mask
            )

        probs = out.action_type_probs[0].tolist()
        sizing_logits = out.bet_size_logits[0]
        sizing_probs = torch.softmax(sizing_logits, dim=-1).tolist()
        ev = out.value[0, 0].item()

        # Apply personality for bots
        if seat_info.personality is not None:
            situations = detect_situations(
                street=street_map.get(self.game_state.street, 0),
                is_facing_raise=(self.game_state.current_bet > 0
                                 and p.bet_this_street < self.game_state.current_bet),
            )
            # Approximate hand strength
            c1, c2 = p.hole_cards
            r1, r2 = c1 // 4, c2 // 4
            if self.game_state.street == Street.PREFLOP:
                if r1 == r2:
                    hs = 0.55 + r1 * 0.035
                elif max(r1, r2) >= 10:
                    hs = 0.45 + max(r1, r2) * 0.02
                elif (c1 % 4) == (c2 % 4):
                    hs = 0.3
                else:
                    hs = 0.15 + max(r1, r2) * 0.01
            else:
                paired_board = any(r1 == (bc // 4) or r2 == (bc // 4)
                                   for bc in self.game_state.board if bc >= 0)
                if paired_board:
                    hs = 0.65 + max(r1, r2) * 0.02
                elif max(r1, r2) >= 10:
                    hs = 0.4
                else:
                    hs = 0.2
            probs_tensor = seat_info.personality.apply(
                torch.tensor(probs), situations,
                hand_strength=hs,
                is_facing_raise=(self.game_state.current_bet > 0
                                 and p.bet_this_street < self.game_state.current_bet),
            )
            probs = probs_tensor.tolist()
            
            sizing_tensor = seat_info.personality.apply_sizing(
                torch.tensor(sizing_probs), situations,
                hand_strength=hs
            )
            sizing_probs = sizing_tensor.tolist()

        import math
        action_evs = {}
        T = 0.5  # Soft-Q temperature
        bb = max(self.game_state.big_blind, 1.0)
        
        for i, name in enumerate(["FOLD", "CHECK", "CALL", "RAISE"]):
            prob = probs[i]
            # Verify if action is legal
            is_legal = any(at.name == name for at in legal_types)
            if name == "RAISE" and not is_legal and any(at.name == "ALL_IN" for at in legal_types):
                name = "ALL_IN"
                is_legal = True
                
            if is_legal:
                if name == "FOLD":
                    action_evs[name] = -(p.bet_total / bb)
                elif prob > 1e-4:
                    action_evs[name] = ev + T * math.log(prob)
                else:
                    action_evs[name] = ev - 5.0

        return {
            'probs': probs,
            'sizing': sizing_probs,
            'ev': ev,
            'action_evs': action_evs,
            'legal_actions': [at.name for at in legal_types]
        }

    def process_action(self, action: Action):
        """Applies an action to the dealer and updates tracking."""
        eng_idx = self.game_state.current_player_idx
        seat_idx = self.seat_map[eng_idx]
        seat_info = self.seats[seat_idx]
        p = self.game_state.players[eng_idx]
        num_engine_players = len(self.seat_map)

        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        pre_street = self.game_state.street

        # --- Comprehensive stat tracking BEFORE apply_action ---
        bet_frac = action.amount / max(self.game_state.pot, 1.0) if action.amount > 0 else 0.0

        if action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
            self.hand_records[eng_idx].vpip = True
        if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            self.hand_records[eng_idx].pfr = True

        # Preflop stats
        if pre_street == Street.PREFLOP:
            if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                self._pf_raise_count += 1
                if self._pf_raise_count >= 2:
                    self.hand_records[eng_idx].three_bet = True
                if self._pf_raise_count == 1 and self._pf_callers_after_raise > 0:
                    self.hand_records[eng_idx].squeeze = True
                self._pf_aggressor = eng_idx
                self._pf_has_raise = True
                self._pf_callers_after_raise = 0
            elif action.action_type == ActionType.CALL:
                if self._pf_has_raise:
                    self.hand_records[eng_idx].cold_call = True
                    self._pf_callers_after_raise += 1
                else:
                    self.hand_records[eng_idx].limp = True

        # Post-flop stats
        if pre_street in (Street.FLOP, Street.TURN, Street.RIVER):
            st_idx = street_map.get(pre_street, 0) - 1
            if 0 <= st_idx < 3:
                if action.action_type == ActionType.CHECK:
                    self._player_checked_this_street[eng_idx] = True
                if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                    if self._player_checked_this_street[eng_idx]:
                        self.hand_records[eng_idx].check_raise[st_idx] = True
                if action.action_type in (ActionType.RAISE, ActionType.ALL_IN) and self._first_bet_this_street_by == -1:
                    self._first_bet_this_street_by = eng_idx
                    if eng_idx == self._pf_aggressor:
                        self.hand_records[eng_idx].cbet[st_idx] = True
                if action.action_type == ActionType.FOLD and self._first_bet_this_street_by == self._pf_aggressor and self._pf_aggressor != -1:
                    self.hand_records[eng_idx].fold_to_cbet[st_idx] = True
                elif action.action_type == ActionType.CALL and self._first_bet_this_street_by == self._pf_aggressor and self._pf_aggressor != -1:
                    self.hand_records[eng_idx].fold_to_cbet[st_idx] = False

        if action.amount > 0 and action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
            self.hand_records[eng_idx].bet_sizes.append(bet_frac)
        self.hand_records[eng_idx].total_wagered += (action.amount if action.amount else 0.0)

        self.dealer.apply_action(action)

        # Record History (keyed by uid)
        pot_base = self.game_state.pot
        self._record_action(seat_info.uid, self._action_to_type_idx(action), bet_frac, pot_base, p.stack, street_map.get(pre_street, 0))

        # Street transition
        if self.game_state.street != self._current_street:
            self._current_street = self.game_state.street
            self._player_checked_this_street = [False] * num_engine_players
            self._first_bet_this_street_by = -1

        self._take_snapshot(action)

        # End of Hand — sync stacks back to seats
        terminal = self.dealer.is_hand_over()
        if terminal:
            results = self.dealer.get_results()
            for eng_i in range(len(self.game_state.players)):
                self.hand_records[eng_i].result = results['profit'][eng_i]
                pf_actions = [a for a in self.game_state.action_history if a.player_idx == eng_i]
                self.hand_records[eng_i].saw_flop = not any(a.action_type == ActionType.FOLD for a in pf_actions) and self.game_state.street.value > Street.PREFLOP.value
                self.hand_records[eng_i].was_pf_aggressor = (self._pf_aggressor == eng_i)
                self.hand_records[eng_i].went_to_showdown = (
                    self.game_state.street == Street.SHOWDOWN and not self.game_state.players[eng_i].is_folded
                )
                self.hand_records[eng_i].won_at_showdown = (
                    self.hand_records[eng_i].went_to_showdown and eng_i in results.get('winners', [])
                )
                si = self.seats[self.seat_map[eng_i]]
                self.stat_tracker.record_hand(si.uid, self.hand_records[eng_i])
                # Update persistent stack
                si.stack = self.game_state.players[eng_i].stack

    def step_ai(self) -> bool:
        """Executes the next AI action if it's not the human's turn."""
        terminal = self.dealer.is_hand_over()
        if terminal:
            return False

        eng_idx = self.game_state.current_player_idx
        seat_idx = self.seat_map[eng_idx]

        if seat_idx == self.human_seat:
            return False

        god_mode = self.timeline[-1].god_mode.get(eng_idx)
        if god_mode is None:
            return False
        
        probs = god_mode['probs']
        sizing_probs = god_mode['sizing']
        legal_types = self.game_state.get_legal_actions()

        from torch.distributions import Categorical
        dist = Categorical(torch.tensor(probs))
        a_idx = dist.sample().item()

        sizing_idx = 0
        if a_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
            s_dist = Categorical(torch.tensor(sizing_probs))
            sizing_idx = s_dist.sample().item()

        # Map to Engine Action
        if a_idx == ActionIndex.FOLD and ActionType.FOLD in legal_types:
            act = Action(ActionType.FOLD)
        elif a_idx == ActionIndex.CHECK and ActionType.CHECK in legal_types:
            act = Action(ActionType.CHECK)
        elif a_idx == ActionIndex.CALL and ActionType.CALL in legal_types:
            act = Action(ActionType.CALL)
        elif a_idx == ActionIndex.RAISE and ActionType.RAISE in legal_types:
            from model.action_space import POT_FRACTIONS
            frac = POT_FRACTIONS[sizing_idx]
            if frac < 0:
                act = Action(ActionType.ALL_IN, amount=self.game_state.get_max_raise_to())
            else:
                min_r = self.game_state.get_min_raise_to()
                max_r = self.game_state.get_max_raise_to()
                rt = frac * self.game_state.pot
                act = Action(ActionType.RAISE, amount=max(min_r, min(rt, max_r)))
        else:
            if ActionType.CHECK in legal_types: act = Action(ActionType.CHECK)
            elif ActionType.CALL in legal_types: act = Action(ActionType.CALL)
            else: act = Action(ActionType.FOLD)

        self.process_action(act)
        return True

    def get_table_info(self) -> List[Dict[str, Any]]:
        """Returns info about all 9 seats for the frontend."""
        result = []
        for i, s in enumerate(self.seats):
            result.append({
                'seat_idx': i,
                'occupied': s.occupied,
                'name': s.name,
                'personality': s.personality_name,
                'stack': s.stack,
                'is_human': s.is_human,
            })
        return result

    def buy_in(self) -> float:
        """Top up the hero's stack to 100bb. Returns the amount added."""
        s = self.seats[self.human_seat]
        if s.stack >= 100.0:
            return 0.0
        added = 100.0 - s.stack
        s.stack = 100.0
        self.total_buyin += added
        return added

    def reset_session(self):
        """Full reset — clear all players, stats, and start fresh."""
        self.rng = random.Random()  # Reseed for new players
        self.seats = [SeatInfo() for _ in range(self.MAX_SEATS)]
        self._next_uid = 1
        self.hand_count = 0
        self.dealer_button_seat = 0
        self.total_buyin = 100.0
        self.stat_tracker = StatTracker()
        self.action_histories = {}
        self.seat_map = []
        self.engine_map = {}
        self.timeline = []
        self.dealer = None
        self.game_state = None
        self._seat_human()
