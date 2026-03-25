import React, { useState } from 'react'
import { formatCard, isRedSuit } from '../utils'

export default function PokerTable({ gameState, selectedSeat, onSelectSeat }) {
  const [revealedSeats, setRevealedSeats] = useState(new Set())

  const handleDoubleClick = (seatIdx) => {
    setRevealedSeats(prev => {
      const next = new Set(prev)
      if (next.has(seatIdx)) {
        next.delete(seatIdx)
      } else {
        next.add(seatIdx)
      }
      return next
    })
  }

  if (!gameState) {
    return (
      <div className="poker-table">
        <div className="felt"></div>
        <div className="pot-area">
          <span className="pot-amount">Pot: 0 bb</span>
          <span className="board-street">PREFLOP</span>
        </div>
      </div>
    )
  }

  const { pot, street, board, players, current_player, is_terminal, dealer_button, small_blind, big_blind } = gameState
  
  return (
    <div className="poker-table">
      <div className="felt"></div>
      
      <div className="pot-area">
        <span className="pot-amount">Pot: {pot.toFixed(2)} bb</span>
        <span className="board-street">{street}</span>
      </div>

      <div className="board-cards">
        {board.filter(c => c !== -1).map((c, i) => (
          <div key={i} className={`card ${isRedSuit(c) ? 'red' : ''}`}>
            {formatCard(c)}
          </div>
        ))}
      </div>

      <div className="seats-container">
        {players.map((p, i) => {
          // Skip empty seats - render a faded empty slot
          if (!p.occupied) {
            return (
              <div key={i} className={`seat seat-${i} empty-seat`}>
                <div className="seat-name">Empty</div>
              </div>
            )
          }

          const isTurn = current_player === p.id && !is_terminal
          const isActiveSeat = selectedSeat === p.id
          const isHero = p.is_human
          const isRevealed = revealedSeats.has(p.id)
          
          const isWinner = is_terminal && gameState.results?.winners?.includes(p.id)
          const profit = gameState.results?.profits?.[p.id] || 0
          
          const showCards = isHero || isRevealed || street === 'SHOWDOWN'
          const displayName = isHero 
            ? 'Hero' 
            : isRevealed 
              ? `${p.name} (${p.personality})` 
              : p.name

          return (
            <div 
              key={i} 
              className={`seat seat-${i} ${isTurn ? 'turn-active' : ''} ${p.is_folded ? 'folded' : ''} ${isActiveSeat ? 'selected' : ''} ${isHero ? 'hero' : ''} ${isRevealed ? 'revealed' : ''} ${isWinner ? 'winner' : ''}`}
              onClick={() => onSelectSeat(p.id)}
              onDoubleClick={() => handleDoubleClick(p.id)}
            >
              {isWinner && profit > 0 && (
                <div className="profit-flyout">+{profit.toFixed(1)} bb</div>
              )}
              <div className="seat-name">{displayName}</div>
              <div className="seat-stack">{p.stack.toFixed(2)} bb</div>
              {p.bet > 0 && <div className="seat-bet">{p.bet.toFixed(2)}</div>}
              {dealer_button === p.id && <div className="dealer-btn">D</div>}
              {small_blind === p.id && <div className="sb-btn">SB</div>}
              {big_blind === p.id && <div className="bb-btn">BB</div>}
              
              <div className="seat-cards">
                {showCards && p.hole_cards && p.hole_cards.length === 2 ? (
                  <>
                    <div className={`card ${isRedSuit(p.hole_cards[0]) ? 'red' : ''}`}>{formatCard(p.hole_cards[0])}</div>
                    <div className={`card ${isRedSuit(p.hole_cards[1]) ? 'red' : ''}`}>{formatCard(p.hole_cards[1])}</div>
                  </>
                ) : (
                  <>
                    <div className="card hidden-card"></div>
                    <div className="card hidden-card"></div>
                  </>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
