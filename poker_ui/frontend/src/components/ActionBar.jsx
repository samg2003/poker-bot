import React, { useState, useEffect } from 'react'

export default function ActionBar({ gameState, isLive, onSubmitAction }) {
  const [raiseAmount, setRaiseAmount] = useState(0)

  const minr = gameState?.min_raise ?? 0
  const maxr = gameState?.max_raise ?? 0
  const legal = gameState?.legal_actions || []
  const potSize = gameState?.pot ?? 0
  const currentBet = gameState?.current_bet ?? 0

  const heroBet = gameState?.players?.[0]?.bet ?? 0
  const callAmount = currentBet > heroBet ? (currentBet - heroBet) : 0

  const canRaise = legal.includes('RAISE') || legal.includes('ALL_IN')
  const isOpeningBet = currentBet === 0

  useEffect(() => {
    if (minr > 0) {
      setRaiseAmount(minr)
    }
  }, [minr])

  if (!isLive || !gameState || gameState.is_terminal || gameState.players[0].personality !== "Human" || gameState.current_player !== 0) {
    return <div className="player-action-bar glass-panel hidden"></div>
  }

  const clampRaise = (val) => Math.max(minr, Math.min(val, maxr))

  const presetTo = (fraction) => {
    setRaiseAmount(clampRaise(Math.round(currentBet + fraction * potSize)))
  }

  const handleInputChange = (e) => {
    const raw = e.target.value
    if (raw === '') { setRaiseAmount(minr); return }
    setRaiseAmount(clampRaise(parseFloat(raw)))
  }

  const handleRaiseSubmit = () => {
    onSubmitAction('RAISE', clampRaise(raiseAmount))
  }

  return (
    <div className="player-action-bar glass-panel">
      
      {/* Fold / Check / Call */}
      <div className="primary-actions">
        <button className="btn action-fold" onClick={() => onSubmitAction('FOLD')}>Fold</button>
        {legal.includes('CHECK') ? (
          <button className="btn action-call" onClick={() => onSubmitAction('CHECK')}>Check</button>
        ) : (
          <button className="btn action-call" onClick={() => onSubmitAction('CALL')}>
            Call {callAmount > 0 ? callAmount.toFixed(1) : ''}
          </button>
        )}
      </div>

      {/* Raise Controls — presets + typed input, no slider */}
      {canRaise && (
        <div className="raise-actions">
          <div className="preset-buttons">
            <button className="btn preset" onClick={() => setRaiseAmount(minr)}>Min</button>
            <button className="btn preset" onClick={() => presetTo(0.33)}>⅓ Pot</button>
            <button className="btn preset" onClick={() => presetTo(0.5)}>½ Pot</button>
            <button className="btn preset" onClick={() => presetTo(0.75)}>¾ Pot</button>
            <button className="btn preset" onClick={() => presetTo(1.0)}>Pot</button>
            <button className="btn preset" onClick={() => presetTo(2.0)}>2× Pot</button>
            <button className="btn preset" onClick={() => setRaiseAmount(maxr)}>All-In</button>
          </div>

          <div className="submit-group">
            <div className="input-prefix">
              <span className="bb-label">bb</span>
              <input 
                type="number" 
                className="custom-raise-input"
                value={Math.round(raiseAmount * 100) / 100}
                onChange={handleInputChange}
                step="0.5"
                min={minr}
                max={maxr}
              />
            </div>
            <button className="btn action-raise" onClick={handleRaiseSubmit}>
              {isOpeningBet ? 'Bet' : 'Raise To'}
            </button>
          </div>
        </div>
      )}

    </div>
  )
}
