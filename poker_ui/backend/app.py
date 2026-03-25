import os
import sys
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
    
    # Needs to match the trainer dimensions, we'll try to auto-detect or default to the user's running config
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    
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
    mgr = CheckpointManager(ckpt_dir)
    try:
        mgr.load(policy, encoder, tag='latest')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        
    game_manager = GameManager(policy, encoder, human_seat=0)
    print("Ready!")

def _serialize_snapshot(snap: TimelineSnapshot):
    gs = snap.game_state
    
    # Map street to string
    street_str = "PREFLOP"
    if gs.street == Street.FLOP: street_str = "FLOP"
    elif gs.street == Street.TURN: street_str = "TURN"
    elif gs.street == Street.RIVER: street_str = "RIVER"
    elif gs.street == Street.SHOWDOWN: street_str = "SHOWDOWN"

    return {
        "pot": gs.pot,
        "board": gs.board,
        "street": street_str,
        "current_player": gs.current_player_idx,
        "is_terminal": game_manager.dealer.is_hand_over() if game_manager.dealer else False,
        "dealer_button": gs.dealer_button,
        "small_blind": gs._sb_idx(),
        "big_blind": gs._bb_idx(),
        "legal_actions": [a.name for a in gs.get_legal_actions()],
        "min_raise": gs.get_min_raise_to(),
        "max_raise": gs.get_max_raise_to(),
        "current_bet": gs.current_bet,
        "players": [
            {
                "id": pid,
                "stack": p.stack,
                "bet": p.bet_this_street,
                "is_active": p.is_active,
                "is_all_in": p.is_all_in,
                "is_folded": p.is_folded,
                "hole_cards": list(p.hole_cards) if p.hole_cards else [],
                "personality": getattr(game_manager.personalities[pid].base, 'name', 'Bot') if game_manager.personalities[pid] else "Human",
            } for pid, p in enumerate(gs.players)
        ],
        "god_mode": snap.god_mode,
        "last_action": {
            "type": snap.action.action_type.name,
            "amount": snap.action.amount
        } if snap.action else None
    }

@app.get("/api/start")
def start_game(num_players: int = 6):
    global game_manager
    game_manager.start_new_hand(num_players=num_players)
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

@app.post("/api/action")
def human_action(req: ActionRequest):
    global game_manager
    gs = game_manager.game_state
    
    terminal = game_manager.dealer.is_hand_over() if game_manager.dealer else True
    if terminal or gs.current_player_idx != game_manager.human_seat:
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
            # Auto-clamp instead of crashing
            amount = max(gs.get_min_raise_to(), min(amount, gs.get_max_raise_to()))

    game_manager.process_action(Action(atype, amount=amount))
    return get_state()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
