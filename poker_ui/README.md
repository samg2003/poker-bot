# Poker AI — Web UI

Interactive web application for playing No-Limit Texas Hold'em against the trained AI model. Features real-time "God Mode" visualization of the AI's internal decision-making.

## Architecture

```
poker_ui/
├── backend/
│   ├── app.py            # FastAPI server (REST API)
│   └── game_manager.py   # Game orchestration, AI inference, timeline snapshots
└── frontend/
    └── src/
        ├── App.jsx                    # Main app shell, API calls, AI step loop
        ├── components/
        │   ├── PokerTable.jsx         # Table felt, seats, cards, board
        │   ├── ActionBar.jsx          # Fold/Check/Call + raise slider/presets
        │   ├── GodModePanel.jsx       # EV, action probabilities, sizing histogram
        │   └── TimelineScrubber.jsx   # Hand history scrubber (prev/next/live)
        ├── utils.js                   # Card formatting helpers
        └── index.css                  # Full design system
```

## Quick Start

### 1. Start the Backend
```bash
cd poker_ui/backend
python3 app.py
# Runs on http://127.0.0.1:8000
```

### 2. Start the Frontend
```bash
cd poker_ui/frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### 3. Play
Open `http://localhost:5173` in your browser. Click **Start New Hand** to begin.

## Features

| Feature | Description |
|---------|-------------|
| **Poker Table** | Oval felt with dynamic seat positioning (2–9 players), dealer/SB/BB chips, community cards |
| **Action Bar** | Fold, Check/Call (with amount), raise slider + presets (Min, ½ Pot, Pot, All-In) |
| **God Mode** | Real-time EV estimate, Fold/Call/Raise probability bars, 10-bucket sizing histogram |
| **Timeline Scrubber** | Step forward/backward through hand history, arrow key support |
| **Personalities** | AI opponents randomly assigned GTO, Nit, or Maniac styles |

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/start?num_players=6` | Start a new hand |
| `GET` | `/api/state` | Get current game state |
| `GET` | `/api/step` | Execute next AI action |
| `POST` | `/api/action` | Submit human action `{ action_type, amount }` |
| `GET` | `/api/timeline/{idx}` | View a specific timeline snapshot |

## Card Encoding

Cards are integers `0–51` using the formula: `card = rank × 4 + suit`

- **Rank:** 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
- **Suit:** 0=♣, 1=♦, 2=♥, 3=♠

## Requirements

- Python 3.9+ with `fastapi`, `uvicorn`, `torch`
- Node.js 18+ for the frontend (Vite + React)
- A trained model checkpoint in `checkpoints/` (loaded automatically on startup)
