import React from 'react'
import { formatCard } from '../utils'

export default function PokerTable({ gameState, selectedSeat, onSelectSeat }) {
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
          <div key={i} className={`card ${[1, 2].includes(Math.floor(c / 13)) ? 'red' : ''}`}>
            {formatCard(c)}
          </div>
        ))}
      </div>

      <div className="seats-container">
        {players.map((p, i) => {
          const isTurn = current_player === i && !is_terminal
          const isActiveSeat = selectedSeat === i

          return (
            <div 
              key={i} 
              className={`seat seat-${i} ${isTurn ? 'turn-active' : ''} ${p.is_folded ? 'folded' : ''} ${isActiveSeat ? 'selected' : ''} ${i===0 ? 'hero' : ''}`}
              onClick={() => onSelectSeat(i)}
            >
              <div className="seat-name">Seat {i} ({p.personality})</div>
              <div className="seat-stack">{p.stack.toFixed(2)} bb</div>
              {p.bet > 0 && <div className="seat-bet">{p.bet.toFixed(2)}</div>}
              {dealer_button === i && <div className="dealer-btn">D</div>}
              {small_blind === i && <div className="sb-btn">SB</div>}
              {big_blind === i && <div className="bb-btn">BB</div>}
              
              <div className="seat-cards">
                {p.hole_cards && p.hole_cards.length === 2 ? (
                  <>
                    <div className={`card ${[1, 2].includes(Math.floor(p.hole_cards[0] / 13)) ? 'red' : ''}`}>{formatCard(p.hole_cards[0])}</div>
                    <div className={`card ${[1, 2].includes(Math.floor(p.hole_cards[1] / 13)) ? 'red' : ''}`}>{formatCard(p.hole_cards[1])}</div>
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
