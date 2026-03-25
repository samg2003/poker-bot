import React from 'react'

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
  
  let pf = 0, pc = 0, pr = 0, sizingHtml = '--'
  let evText = '-- bb'

  if (evData && !p.is_folded) {
    evText = `${evData.ev > 0 ? '+' : ''}${evData.ev.toFixed(2)} bb`
    pf = (evData.probs[0] * 100).toFixed(1)
    pc = (evData.probs[1] * 100 + evData.probs[2] * 100).toFixed(1)
    pr = (evData.probs[3] * 100).toFixed(1)
    
    const minr = gameState.min_raise
    const maxr = gameState.max_raise
    const target = minr + evData.sizing * (maxr - minr)
    sizingHtml = `${target.toFixed(2)} bb`
  }

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
        <span className="ev-label">If Raising, Target Size:</span>
        <span className="ev-amount">{sizingHtml}</span>
      </div>
      
      <small className="disclaimer">These probabilities flow from the raw Policy Network evaluation, modified by the agent's baseline personality matrix.</small>
    </div>
  )
}
