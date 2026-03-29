import os
import sys
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Ensure imports work from parent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from deployment.checkpoint import CheckpointManager
from engine.game_state import Action, ActionType, Street
from poker_ui.backend.game_manager import GameManager, TimelineSnapshot

def _sanitize_floats(obj):
    """Recursively replace NaN/inf floats with 0.0 so JSON serialization works."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_floats(x) for x in obj]
    return obj

app = FastAPI(title="Goated Poker AI")

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
game_manager = None

class ActionRequest(BaseModel):
    action_type: str
    amount: float = 0.0

@app.on_event("startup")
def load_model():
    global game_manager
    print("Loading latest checkpoint...")
    
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    import torch

    policy = PolicyNetwork(
        embed_dim=embed_dim,
        opponent_embed_dim=embed_dim,
        num_cross_attn_heads=num_heads,
        num_cross_attn_layers=num_layers,
    )
    encoder = OpponentEncoder(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'checkpoints')
    try:
        policy.load_state_dict(
            torch.load(os.path.join(ckpt_dir, 'latest', 'policy.pt'), map_location='cpu', weights_only=True)
        )
        encoder.load_state_dict(
            torch.load(os.path.join(ckpt_dir, 'latest', 'opponent_encoder.pt'), map_location='cpu', weights_only=True)
        )
        policy.eval()
        encoder.eval()
        print("✓ Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        
    game_manager = GameManager(policy, encoder, human_seat=0)
    print("Ready!")


def _serialize_snapshot(snap: TimelineSnapshot):
    gs = snap.game_state
    seat_map = snap.seat_map
    
    # Map street to string
    street_str = "PREFLOP"
    if gs.street == Street.FLOP: street_str = "FLOP"
    elif gs.street == Street.TURN: street_str = "TURN"
    elif gs.street == Street.RIVER: street_str = "RIVER"
    elif gs.street == Street.SHOWDOWN: street_str = "SHOWDOWN"

    # Build full 9-seat player array (with empty seats)
    players = []
    for table_seat in range(game_manager.MAX_SEATS):
        seat_info = game_manager.seats[table_seat]
        if table_seat in game_manager.engine_map:
            eng_idx = game_manager.engine_map[table_seat]
            p = gs.players[eng_idx]
            players.append({
                "id": table_seat,
                "occupied": True,
                "name": seat_info.name,
                "stack": p.stack,
                "bet": p.bet_this_street,
                "is_active": p.is_active,
                "is_all_in": p.is_all_in,
                "is_folded": p.is_folded,
                "hole_cards": list(p.hole_cards) if p.hole_cards else [],
                "personality": seat_info.personality_name,
                "is_human": seat_info.is_human,
            })
        else:
            players.append({
                "id": table_seat,
                "occupied": False,
                "name": "",
                "stack": 0,
                "bet": 0,
                "is_active": False,
                "is_all_in": False,
                "is_folded": True,
                "hole_cards": [],
                "personality": "",
                "is_human": False,
            })

    # Map engine current_player back to table seat
    current_table_seat = seat_map[gs.current_player_idx] if gs.current_player_idx < len(seat_map) else -1

    # Map dealer/SB/BB from engine indices to table seats
    dealer_table = seat_map[gs.dealer_button] if gs.dealer_button < len(seat_map) else -1
    sb_table = seat_map[gs._sb_idx()] if gs._sb_idx() < len(seat_map) else -1
    bb_table = seat_map[gs._bb_idx()] if gs._bb_idx() < len(seat_map) else -1

    # Remap god_mode from engine indices to table seat indices
    god_mode_remapped = {}
    for eng_idx, data in snap.god_mode.items():
        if eng_idx < len(seat_map):
            god_mode_remapped[seat_map[eng_idx]] = _sanitize_floats(data)

    terminal = game_manager.dealer.is_hand_over() if game_manager.dealer else False
    results = None
    if terminal and game_manager.dealer:
        eng_results = game_manager.dealer.get_results()
        winners = [seat_map[i] for i in eng_results['winners'] if i < len(seat_map)]
        profits = {seat_map[i]: eng_results['profit'][i] for i in range(len(eng_results['profit'])) if i < len(seat_map)}
        # pot_won is profit + amount invested
        pot_won = {}
        for i in range(len(gs.players)):
            if i < len(seat_map):
                pot_won[seat_map[i]] = eng_results['profit'][i] + gs.players[i].bet_total
                
        results = {
            "winners": winners,
            "profits": profits,
            "pot_won": pot_won
        }

    # Calculate side pots for display
    side_pots_info = []
    pots = gs.calculate_side_pots()
    if pots:
        for p_amt, eligible in pots:
            mapped_eligible = [seat_map[i] for i in eligible if i < len(seat_map)]
            side_pots_info.append({"amount": p_amt, "eligible": mapped_eligible})

    return {
        "pot": gs.pot,
        "side_pots": side_pots_info,
        "board": gs.board,
        "street": street_str,
        "current_player": current_table_seat,
        "is_terminal": terminal,
        "results": results,
        "dealer_button": dealer_table,
        "small_blind": sb_table,
        "big_blind": bb_table,
        "legal_actions": [a.name for a in gs.get_legal_actions()],
        "min_raise": gs.get_min_raise_to(),
        "max_raise": gs.get_max_raise_to(),
        "current_bet": gs.current_bet,
        "players": players,
        "god_mode": god_mode_remapped,
        "last_action": {
            "type": snap.action.action_type.name,
            "amount": snap.action.amount
        } if snap.action else None,
        "hand_count": game_manager.hand_count,
        "total_buyin": game_manager.total_buyin,
        "hero_stack": game_manager.seats[game_manager.human_seat].stack,
    }

@app.get("/api/start")
def start_game():
    global game_manager
    game_manager.start_new_hand()
    return {"status": "started", "timeline_length": len(game_manager.timeline)}

@app.get("/api/state")
def get_state():
    global game_manager
    if not game_manager.timeline:
        return {"error": "Game not started"}
    return {
        "snapshot": _serialize_snapshot(game_manager.timeline[-1]),
        "timeline_index": len(game_manager.timeline) - 1,
        "total_steps": len(game_manager.timeline)
    }

@app.get("/api/timeline/{idx}")
def get_timeline(idx: int):
    global game_manager
    if idx < 0 or idx >= len(game_manager.timeline):
        raise HTTPException(status_code=404, detail="Timeline index out of bounds")
    return {
        "snapshot": _serialize_snapshot(game_manager.timeline[idx]),
        "timeline_index": idx,
        "total_steps": len(game_manager.timeline)
    }

@app.get("/api/step")
def step_ai():
    global game_manager
    if not game_manager.timeline:
        raise HTTPException(status_code=400, detail="Game not started")
        
    took_action = game_manager.step_ai()
    return {"took_action": took_action, "state": get_state()}

@app.get("/api/step_until_human")
def step_until_human():
    """Batch-run ALL AI actions until it's the human's turn or the hand is over."""
    global game_manager
    if not game_manager.timeline:
        raise HTTPException(status_code=400, detail="Game not started")
    
    actions = game_manager.step_all_ai()
    return {
        "actions": actions,
        "state": get_state(),
    }

@app.post("/api/action")
def human_action(req: ActionRequest):
    global game_manager
    gs = game_manager.game_state
    
    terminal = game_manager.dealer.is_hand_over() if game_manager.dealer else True
    if terminal:
        raise HTTPException(status_code=400, detail="Hand is over")
    
    # Check it's the human's turn using seat map
    eng_idx = gs.current_player_idx
    table_seat = game_manager.seat_map[eng_idx]
    if table_seat != game_manager.human_seat:
        raise HTTPException(status_code=400, detail="Not your turn")
        
    atype_map = {
        "FOLD": ActionType.FOLD,
        "CHECK": ActionType.CHECK,
        "CALL": ActionType.CALL,
        "RAISE": ActionType.RAISE,
        "ALL_IN": ActionType.ALL_IN
    }
    
    atype = atype_map.get(req.action_type.upper())
    if atype not in gs.get_legal_actions():
        raise HTTPException(status_code=400, detail=f"Illegal action: {req.action_type}")
        
    amount = req.amount
    if atype == ActionType.RAISE:
        if amount < gs.get_min_raise_to() or amount > gs.get_max_raise_to():
            amount = max(gs.get_min_raise_to(), min(amount, gs.get_max_raise_to()))

    game_manager.process_action(Action(atype, amount=amount))
    return get_state()

@app.get("/api/table")
def get_table():
    """Returns info about all 9 seats."""
    global game_manager
    return {"seats": game_manager.get_table_info()}

@app.get("/api/buyin")
def buy_in():
    """Top up hero stack to 100bb."""
    global game_manager
    added = game_manager.buy_in()
    return {"added": added, "total_buyin": game_manager.total_buyin, "stack": game_manager.seats[game_manager.human_seat].stack}

@app.get("/api/reset")
def reset_session():
    """Full reset — clear everything and start fresh."""
    global game_manager
    game_manager.reset_session()
    return {"status": "reset"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
