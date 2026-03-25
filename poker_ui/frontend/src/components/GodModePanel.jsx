import React from 'react'

const SIZING_LABELS = ['10%', '25%', '33%', '50%', '66%', '75%', '100%', '150%', '200%', 'ALL-IN'];
const POT_FRACTIONS = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0, -1.0];

export default function GodModePanel({ gameState, selectedSeat }) {
  if (!gameState || !gameState.god_mode || !gameState.players[selectedSeat]) {
    return (
      <div className="god-mode-panel glass-panel">
        <div className="dashboard-header">
          <h2>God Mode 👁️</h2>
        </div>
        <p className="subtitle">Select a player seat to view the AI's real-time expected value.</p>
      </div>
    )
  }

  const p = gameState.players[selectedSeat]
  const evData = gameState.god_mode[selectedSeat]
  
  let pf = 0, pc = 0, pr = 0
  let evText = '-- bb'
  let sizingProbs = Array(10).fill(0)

  if (evData && !p.is_folded) {
    evText = `${evData.ev > 0 ? '+' : ''}${evData.ev.toFixed(2)} bb`
    pf = (evData.probs[0] * 100).toFixed(1)
    pc = (evData.probs[1] * 100 + evData.probs[2] * 100).toFixed(1)
    pr = (evData.probs[3] * 100).toFixed(1)
    
    if (evData.sizing && evData.sizing.length === 10) {
      sizingProbs = evData.sizing
    }
  }

  // Build sorted sizing entries with bb amounts
  const pot = gameState.pot || 0
  const currentBet = gameState.current_bet || 0
  const totalProb = sizingProbs.reduce((a, b) => a + b, 0)

  const sizingEntries = sizingProbs.map((prob, idx) => {
    const frac = POT_FRACTIONS[idx]
    const label = SIZING_LABELS[idx]
    const normalized = totalProb > 0 ? prob / totalProb : 0
    
    let bbAmount
    if (frac < 0) {
      // All-in: use the player's remaining stack + current street bet
      bbAmount = p.stack + (p.bet || 0)
    } else {
      bbAmount = currentBet + frac * pot
    }
    
    return { label, normalized, bbAmount, idx }
  })

  // Sort descending by normalized probability
  const sorted = [...sizingEntries].sort((a, b) => b.normalized - a.normalized)

  return (
    <div className="god-mode-panel glass-panel">
      <div className="dashboard-header">
        <h2>God Mode 👁️</h2>
        <span className="pulse-indicator"></span>
      </div>
      
      <div className="selected-player-info">
        <h3>Seat {selectedSeat} ({p.personality})</h3>
        <span className="tag">Tracked Seat</span>
      </div>

      <hr />

      <div className="ev-display">
        <span className="ev-label">Expected Value</span>
        <span className="ev-amount">{evText}</span>
      </div>

      <div className="prob-bars">
        <div className="prob-row">
          <span className="prob-label text-fold">Fold</span>
          <div className="bar-bg"><div className="bar bg-fold" style={{ width: `${pf}%` }}></div></div>
          <span className="prob-val">{pf}%</span>
        </div>
        <div className="prob-row">
          <span className="prob-label text-call">Check/Call</span>
          <div className="bar-bg"><div className="bar bg-call" style={{ width: `${pc}%` }}></div></div>
          <span className="prob-val">{pc}%</span>
        </div>
        <div className="prob-row">
          <span className="prob-label text-raise">Raise</span>
          <div className="bar-bg"><div className="bar bg-raise" style={{ width: `${pr}%` }}></div></div>
          <span className="prob-val">{pr}%</span>
        </div>
      </div>

      <div className="sizing-display">
        <span className="ev-label">Sizing Distribution (If Raising)</span>
        <div className="sizing-list">
          {sorted.map((entry) => {
            const pct = (entry.normalized * 100).toFixed(1)
            if (entry.normalized < 0.001) return null
            return (
              <div key={entry.idx} className="sizing-row">
                <span className="sizing-label">{entry.label}</span>
                <div className="sizing-bar-bg">
                  <div className="sizing-bar" style={{ width: `${pct}%` }}></div>
                </div>
                <span className="sizing-pct">{pct}%</span>
                <span className="sizing-bb">{entry.bbAmount.toFixed(1)} bb</span>
              </div>
            )
          })}
        </div>
      </div>
      
      <small className="disclaimer">These probabilities flow from the raw Policy Network evaluation, modified by the agent's baseline personality matrix.</small>
    </div>
  )
}
