import { useState, useEffect } from 'react'
import PokerTable from './components/PokerTable'
import ActionBar from './components/ActionBar'
import GodModePanel from './components/GodModePanel'
import TimelineScrubber from './components/TimelineScrubber'
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
    const data = await apiGet('/start')
    if (data) await refreshState()
    setLoading(false)
  }

  const fetchTimeline = async (idx) => {
    if (idx < 0 || idx >= totalSteps) return
    const data = await apiGet(`/timeline/${idx}`)
    if (data && data.snapshot) {
      setGameState(data.snapshot)
      setTimelineIdx(data.timeline_index)
      setTotalSteps(data.total_steps)
    }
  }

  const stepAI = async () => {
    if (timelineIdx < totalSteps - 1) return
    if (!gameState || gameState.is_terminal || gameState.players[gameState.current_player].personality === 'Human') return
    
    // AI turn
    setTimeout(async () => {
      const data = await apiGet('/step')
      if (data && data.took_action) {
        setGameState(data.state.snapshot)
        setTimelineIdx(data.state.timeline_index)
        setTotalSteps(data.state.total_steps)
        if (data.state.snapshot.last_action) {
          log(`[Bot] played ${data.state.snapshot.last_action.type} ${data.state.snapshot.last_action.amount.toFixed(2)}`)
        }
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
  }, [gameState, timelineIdx, totalSteps])

  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger if user is typing in the custom raise input
      if (e.target.tagName === 'INPUT') return;
      
      if (e.key === 'ArrowLeft') {
        fetchTimeline(timelineIdx - 1)
      } else if (e.key === 'ArrowRight') {
        fetchTimeline(timelineIdx + 1)
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [timelineIdx, totalSteps])

  const submitAction = async (type, amount = 0) => {
    if (!gameState || gameState.is_terminal || gameState.current_player !== 0) return
    if (timelineIdx < totalSteps - 1) {
      alert("You are viewing the past. Click 'Return to Live Action' to play.")
      return
    }
    log(`[Hero] played ${type} ${amount > 0 ? amount.toFixed(2) : ''}`)
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
        <h2>DeepMind Poker Bot</h2>
        <button className="btn primary block" onClick={startHand}>Start New Hand</button>
        
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
