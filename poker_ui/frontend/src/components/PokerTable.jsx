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
          const isTurn = current_player === i && !is_terminal
          const isActiveSeat = selectedSeat === i
          const isHero = i === 0
          const isRevealed = revealedSeats.has(i)
          
          // Show cards for Hero always, for others only if revealed via double-click (or showdown)
          const showCards = isHero || isRevealed || street === 'SHOWDOWN'
          // Show personality only if revealed
          const displayName = isHero 
            ? 'Hero' 
            : isRevealed 
              ? `Seat ${i} (${p.personality})` 
              : `Seat ${i}`

          return (
            <div 
              key={i} 
              className={`seat seat-${i} ${isTurn ? 'turn-active' : ''} ${p.is_folded ? 'folded' : ''} ${isActiveSeat ? 'selected' : ''} ${isHero ? 'hero' : ''} ${isRevealed ? 'revealed' : ''}`}
              onClick={() => onSelectSeat(i)}
              onDoubleClick={() => handleDoubleClick(i)}
            >
              <div className="seat-name">{displayName}</div>
              <div className="seat-stack">{p.stack.toFixed(2)} bb</div>
              {p.bet > 0 && <div className="seat-bet">{p.bet.toFixed(2)}</div>}
              {dealer_button === i && <div className="dealer-btn">D</div>}
              {small_blind === i && <div className="sb-btn">SB</div>}
              {big_blind === i && <div className="bb-btn">BB</div>}
              
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
