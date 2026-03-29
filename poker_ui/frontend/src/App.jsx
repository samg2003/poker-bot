import { useState, useEffect, useRef } from 'react'
import PokerTable from './components/PokerTable'
import ActionBar from './components/ActionBar'
import GodModePanel from './components/GodModePanel'
import TimelineScrubber from './components/TimelineScrubber'
import { playActionSound, playNewHandSound, playNavigateSound } from './sounds'
import './index.css'

const API_URL = 'http://127.0.0.1:8000/api'

function App() {
  const [gameState, setGameState] = useState(null)
  const [timelineIdx, setTimelineIdx] = useState(0)
  const [totalSteps, setTotalSteps] = useState(0)
  const [selectedSeat, setSelectedSeat] = useState(0)
  const [loading, setLoading] = useState(true)
  const [logs, setLogs] = useState([])
  
  const log = (msg) => {
    setLogs(prev => [msg, ...prev].slice(0, 50))
  }

  const apiGet = async (route) => {
    try {
      const res = await fetch(`${API_URL}${route}`)
      if (!res.ok) throw new Error(await res.text())
      return await res.json()
    } catch (e) {
      console.error(e)
      return null
    }
  }

  const apiPost = async (route, body) => {
    try {
      const res = await fetch(`${API_URL}${route}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      if (!res.ok) throw new Error(await res.text())
      return await res.json()
    } catch (e) {
      console.error(e)
      return null
    }
  }

  const refreshState = async () => {
    const data = await apiGet('/state')
    if (data && data.snapshot) {
      setGameState(data.snapshot)
      setTimelineIdx(data.timeline_index)
      setTotalSteps(data.total_steps)
    }
  }

  const startHand = async () => {
    setLoading(true)
    log("--- New Hand Started ---")
    playNewHandSound()
    const data = await apiGet('/start')
    if (data) await refreshState()
    setLoading(false)
  }

  const fetchTimeline = async (idx) => {
    if (idx < 0 || idx >= totalSteps) return
    cancelledRef.current = true  // Stop any running animation
    playNavigateSound()
    const data = await apiGet(`/timeline/${idx}`)
    if (data && data.snapshot) {
      setGameState(data.snapshot)
      setTimelineIdx(data.timeline_index)
      setTotalSteps(data.total_steps)
      if (data.snapshot.last_action) {
        playActionSound(data.snapshot.last_action.type)
      }
    }
  }

  // ─── Robust AI stepping: batch on server, animate on client ───
  const isAnimatingRef = useRef(false)
  const cancelledRef = useRef(false)

  const runAISteps = async () => {
    // Guard: only one animation loop at a time
    if (isAnimatingRef.current) return
    isAnimatingRef.current = true
    cancelledRef.current = false

    try {
      // 1. Ask server to batch-run all AI actions
      const data = await apiGet('/step_until_human')
      if (!data) return

      const actions = data.actions || []
      const finalState = data.state
      if (!finalState?.snapshot) return

      const newTotalSteps = finalState.total_steps
      const startIdx = newTotalSteps - actions.length

      // 2. Animate each step with delays
      for (let i = 0; i < actions.length; i++) {
        if (cancelledRef.current) break  // User navigated away

        const tIdx = startIdx + i
        const snapData = await apiGet(`/timeline/${tIdx}`)
        if (!snapData?.snapshot) break

        setGameState(snapData.snapshot)
        setTimelineIdx(snapData.timeline_index)
        setTotalSteps(newTotalSteps)

        // Sound + log
        const act = actions[i]
        playActionSound(act.action?.type || '')
        const amt = act.action?.amount
        const seatPlayer = snapData.snapshot.players?.find(p => p.id === act.actor_seat)
        const name = seatPlayer?.name || `Seat ${act.actor_seat}`
        log(`[${name}] ${act.action?.type} ${amt != null && amt > 0 ? amt.toFixed(2) : ''}`)

        // Street transition = longer delay
        const delay = act.street_changed ? 1500 : 600
        await new Promise(r => setTimeout(r, delay))
      }

      // 3. Show final live state (unless user is browsing history)
      if (!cancelledRef.current) {
        setGameState(finalState.snapshot)
        setTimelineIdx(finalState.timeline_index)
        setTotalSteps(finalState.total_steps)
      }

    } catch (e) {
      console.error('runAISteps error:', e)
      await refreshState()
    } finally {
      isAnimatingRef.current = false
    }
  }

  // Kick off AI stepping whenever gameState changes and it's not human's turn
  useEffect(() => {
    if (!gameState) return
    if (gameState.is_terminal) return
    if (isAnimatingRef.current) return  // Don't re-trigger during animation

    const currentPlayer = gameState.players?.find(p => p.id === gameState.current_player)
    if (currentPlayer?.is_human) return

    // It's an AI's turn — run the batch stepping
    runAISteps()
  }, [gameState?.is_terminal, gameState?.current_player, gameState?.hand_count])

  useEffect(() => {
    const init = async () => {
      const state = await apiGet('/state')
      if (!state || state.error === "Game not started") {
        await startHand()
      } else {
        setGameState(state.snapshot)
        setTimelineIdx(state.timeline_index)
        setTotalSteps(state.total_steps)
        setLoading(false)
      }
    }
    init()
  }, [])

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger if user is typing in the custom raise input
      if (e.target.tagName === 'INPUT') return;
      
      if (e.key === 'ArrowLeft') {
        fetchTimeline(timelineIdx - 1)
      } else if (e.key === 'ArrowRight') {
        fetchTimeline(timelineIdx + 1)
      } else if (e.key === 'Enter' && gameState?.is_terminal) {
        startHand()
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [timelineIdx, totalSteps, gameState])

  const submitAction = async (type, amount = 0) => {
    const heroPlayer = gameState?.players?.find(p => p.is_human)
    if (!gameState || gameState.is_terminal || !heroPlayer || gameState.current_player !== heroPlayer.id) return
    if (timelineIdx < totalSteps - 1) {
      alert("You are viewing the past. Click 'Return to Live Action' to play.")
      return
    }
    log(`[Hero] played ${type} ${amount > 0 ? amount.toFixed(2) : ''}`)
    playActionSound(type)
    const data = await apiPost('/action', { action_type: type, amount })
    if (data && data.snapshot) {
      setGameState(data.snapshot)
      setTimelineIdx(data.timeline_index)
      setTotalSteps(data.total_steps)
    }
  }

  return (
    <div className="app-container">
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Evaluating Neural Network...</p>
        </div>
      )}

      {/* Left Sidebar */}
      <div className="sidebar glass-panel">
        <h2>Poker AI</h2>
        {gameState && (
          <div style={{fontSize: '12px', color: '#8b949e', marginBottom: '12px'}}>
            Hand #{gameState.hand_count || 0} &middot; {gameState.players.filter(p => p.occupied).length} players
          </div>
        )}
        <button className="btn primary block" onClick={startHand}>Start New Hand</button>

        {gameState && (
          <div style={{background: 'rgba(0,0,0,0.4)', padding: '12px', borderRadius: '10px', marginTop: '12px', border: '1px solid rgba(255,255,255,0.05)'}}>
            <div style={{fontSize: '11px', color: '#8b949e', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '8px', fontWeight: 600}}>Session P&L</div>
            <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px'}}>
              <span style={{color: '#8b949e'}}>Total Buy-in</span>
              <span style={{color: '#f85149', fontWeight: 700}}>{(gameState.total_buyin || 100).toFixed(1)} bb</span>
            </div>
            <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px'}}>
              <span style={{color: '#8b949e'}}>Stack</span>
              <span style={{color: '#58a6ff', fontWeight: 700}}>{(gameState.hero_stack || 0).toFixed(1)} bb</span>
            </div>
            <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '13px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '6px', marginTop: '6px'}}>
              <span style={{color: '#fff', fontWeight: 700}}>Net P&L</span>
              {(() => {
                const pnl = (gameState.hero_stack || 0) - (gameState.total_buyin || 100)
                return <span style={{color: pnl >= 0 ? '#2ea043' : '#f85149', fontWeight: 800}}>{pnl >= 0 ? '+' : ''}{pnl.toFixed(1)} bb</span>
              })()}
            </div>
            {gameState.is_terminal && (gameState.hero_stack || 0) < 100 && (
              <button className="btn secondary block" style={{marginTop: '10px', fontSize: '12px'}} onClick={async () => {
                await apiGet('/buyin')
                await refreshState()
                log('[Hero] bought in for 100bb')
              }}>
                Top Up to 100bb
              </button>
            )}
          </div>
        )}
        
        <button className="btn text" style={{marginTop: '8px', fontSize: '11px', width: '100%', color: '#8b949e'}} onClick={async () => {
          await apiGet('/reset')
          setGameState(null)
          setGameLog([])
          setTimelineIdx(0)
          setTotalSteps(0)
          log('--- Session Reset ---')
          await startHand()
        }}>
          Reset Session
        </button>
        <TimelineScrubber 
          currentIndex={timelineIdx} 
          totalSteps={totalSteps}
          onPrev={() => fetchTimeline(timelineIdx - 1)}
          onNext={() => fetchTimeline(timelineIdx + 1)}
          onLive={refreshState}
        />

        <div className="game-log-container">
          <h3>Action Log</h3>
          <ul className="game-log">
            {logs.map((l, i) => <li key={i} className={l.includes('[Hero]') ? 'hero' : ''}>{l}</li>)}
          </ul>
        </div>
      </div>

      {/* Main Area */}
      <div className="main-table-area">
        <PokerTable 
          gameState={gameState} 
          selectedSeat={selectedSeat}
          onSelectSeat={setSelectedSeat}
        />
        
        <ActionBar 
          gameState={gameState}
          isLive={timelineIdx === totalSteps - 1}
          onSubmitAction={submitAction}
        />
      </div>

      {/* Right Sidebar */}
      <GodModePanel 
        gameState={gameState}
        selectedSeat={selectedSeat}
      />
    </div>
  )
}

export default App
