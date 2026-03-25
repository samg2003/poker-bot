import React from 'react'

export default function TimelineScrubber({ currentIndex, totalSteps, onPrev, onNext, onLive }) {
  return (
    <div className="hand-timeline-controls">
      <h3>Hand History Scrubber</h3>
      <div className="timeline-row">
        <button className="btn secondary" onClick={onPrev} disabled={currentIndex <= 0}>
          ⟵ Prev
        </button>
        <span className="timeline-text">
          Step {Math.max(1, currentIndex + 1)}/{Math.max(1, totalSteps)}
        </span>
        <button className="btn secondary" onClick={onNext} disabled={currentIndex >= totalSteps - 1}>
          Next ⟶
        </button>
      </div>
      <button className="btn text w-full" onClick={onLive}>
        Return to Live Action
      </button>
    </div>
  )
}
