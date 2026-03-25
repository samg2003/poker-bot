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

  const stepTimeoutRef = useRef(null)
  const isSteppingRef = useRef(false)

  const timelineIdxRef = useRef(timelineIdx)
  const totalStepsRef = useRef(totalSteps)
  const gameStateRef = useRef(gameState)
  useEffect(() => { timelineIdxRef.current = timelineIdx }, [timelineIdx])
  useEffect(() => { totalStepsRef.current = totalSteps }, [totalSteps])
  useEffect(() => { gameStateRef.current = gameState }, [gameState])

  const stepAI = async () => {
    if (timelineIdxRef.current < totalStepsRef.current - 1) return
    const gs = gameStateRef.current
    if (!gs || gs.is_terminal) return
    const currentSeat = gs.current_player
    const currentPlayer = gs.players.find(p => p.id === currentSeat)
    if (!currentPlayer || currentPlayer.is_human) return
    if (isSteppingRef.current) return
    
    isSteppingRef.current = true
    stepTimeoutRef.current = setTimeout(async () => {
      try {
        const data = await apiGet('/step')
        if (data && data.took_action) {
          setGameState(data.state.snapshot)
          setTimelineIdx(data.state.timeline_index)
          setTotalSteps(data.state.total_steps)
          if (data.state.snapshot.last_action) {
            const actionType = data.state.snapshot.last_action.type
            playActionSound(actionType)
            const amt = data.state.snapshot.last_action.amount
            log(`[Bot] played ${actionType} ${amt != null ? amt.toFixed(2) : ''}`)
          }
        }
      } catch (e) {
        console.error('stepAI error:', e)
      } finally {
        isSteppingRef.current = false
      }
    }, 600)
  }

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
    if (gameState && !gameState.is_terminal) {
      stepAI()
    }
    return () => {
      if (stepTimeoutRef.current) clearTimeout(stepTimeoutRef.current)
    }
  }, [gameState, timelineIdx, totalSteps])

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
