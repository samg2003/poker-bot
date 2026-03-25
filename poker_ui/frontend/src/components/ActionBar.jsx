import React, { useState, useEffect } from 'react'

export default function ActionBar({ gameState, isLive, onSubmitAction }) {
  const [raiseAmount, setRaiseAmount] = useState(0)

  useEffect(() => {
    if (gameState && gameState.min_raise) {
      setRaiseAmount(gameState.min_raise)
    }
  }, [gameState])

  if (!isLive || !gameState || gameState.is_terminal || gameState.players[0].personality !== "Human" || gameState.current_player !== 0) {
    return <div className="player-action-bar glass-panel hidden"></div>
  }

  const minr = gameState.min_raise
  const maxr = gameState.max_raise
  const legal = gameState.legal_actions || []

  // Pre-calculate sizing buttons for 1/2 pot, full pot, all-in
  const potSize = gameState.pot
  const potRef = Math.max(minr, Math.min(potSize, maxr))
  const halfPotRef = Math.max(minr, Math.min(Math.floor(potSize / 2), maxr))
  
  const handleRaiseSubmit = () => {
    onSubmitAction('RAISE', raiseAmount)
  }

  return (
    <div className="player-action-bar glass-panel">
      
      {/* Fold / Check / Call */}
      <div className="primary-actions">
        <button 
          className="btn action-fold" 
          onClick={() => onSubmitAction('FOLD')}
        >
          Fold
        </button>
        
        {legal.includes('CHECK') ? (
          <button className="btn action-call" onClick={() => onSubmitAction('CHECK')}>
            Check
          </button>
        ) : (
          <button className="btn action-call" onClick={() => onSubmitAction('CALL')}>
            Call
          </button>
        )}
      </div>

      {/* Advanced Raise Controls */}
      <div className="raise-actions">
        <div className="preset-buttons">
          <button className="btn preset" onClick={() => setRaiseAmount(minr)}>Min</button>
          <button className="btn preset" onClick={() => setRaiseAmount(halfPotRef)}>1/2 Pot</button>
          <button className="btn preset" onClick={() => setRaiseAmount(potRef)}>Pot</button>
          <button className="btn preset" onClick={() => setRaiseAmount(maxr)}>All-In</button>
        </div>
        
        <div className="slider-group">
          <input 
            type="range" 
            min={minr} 
            max={maxr} 
            value={raiseAmount} 
            onChange={(e) => setRaiseAmount(Number(e.target.value))}
            className="raise-slider"
          />
        </div>

        <div className="submit-group">
          <div className="input-prefix">
            <span className="bb-label">bb</span>
            <input 
              type="number" 
              className="custom-raise-input"
              value={raiseAmount.toFixed(2)}
              onChange={(e) => setRaiseAmount(Math.min(maxr, Math.max(minr, Number(e.target.value))))}
              step="0.5"
            />
          </div>
          <button className="btn action-raise" onClick={handleRaiseSubmit}>
            Raise To
          </button>
        </div>
      </div>

    </div>
  )
}
