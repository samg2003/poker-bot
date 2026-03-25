// Wizard Poker — mystical sound effects via Web Audio API
let audioCtx = null

function ctx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  return audioCtx
}

// --- Primitives ---

function tone(freq, dur, type = 'sine', vol = 0.12, delay = 0) {
  try {
    const c = ctx(), t = c.currentTime + delay
    const osc = c.createOscillator()
    const gain = c.createGain()
    osc.type = type
    osc.frequency.setValueAtTime(freq, t)
    gain.gain.setValueAtTime(vol, t)
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur)
    osc.connect(gain).connect(c.destination)
    osc.start(t)
    osc.stop(t + dur)
  } catch (e) {}
}

function sweep(startFreq, endFreq, dur, type = 'sine', vol = 0.1, delay = 0) {
  try {
    const c = ctx(), t = c.currentTime + delay
    const osc = c.createOscillator()
    const gain = c.createGain()
    osc.type = type
    osc.frequency.setValueAtTime(startFreq, t)
    osc.frequency.exponentialRampToValueAtTime(endFreq, t + dur)
    gain.gain.setValueAtTime(vol, t)
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur)
    osc.connect(gain).connect(c.destination)
    osc.start(t)
    osc.stop(t + dur)
  } catch (e) {}
}

function noise(dur = 0.03, vol = 0.1, hp = 3000, delay = 0) {
  try {
    const c = ctx(), t = c.currentTime + delay
    const buf = c.createBuffer(1, c.sampleRate * dur, c.sampleRate)
    const d = buf.getChannelData(0)
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length)
    const src = c.createBufferSource()
    src.buffer = buf
    const filter = c.createBiquadFilter()
    filter.type = 'highpass'
    filter.frequency.value = hp
    const gain = c.createGain()
    gain.gain.setValueAtTime(vol, t)
    gain.gain.exponentialRampToValueAtTime(0.001, t + dur)
    src.connect(filter).connect(gain).connect(c.destination)
    src.start(t)
  } catch (e) {}
}

// --- Wizard Poker Sounds ---

// Fold: cards sliding away — descending whoosh
export function playFoldSound() {
  sweep(800, 200, 0.2, 'sine', 0.1)
  noise(0.08, 0.06, 2000, 0.05)
}

// Check: magical tap — two bright pings
export function playCheckSound() {
  tone(1200, 0.06, 'sine', 0.1)
  tone(1600, 0.08, 'sine', 0.08, 0.07)
}

// Call: chip stack with a warm chime
export function playCallSound() {
  tone(600, 0.1, 'triangle', 0.1)
  tone(800, 0.12, 'triangle', 0.08, 0.08)
  noise(0.02, 0.08, 4000, 0.0)
  noise(0.02, 0.06, 5000, 0.04)
  noise(0.02, 0.05, 4500, 0.08)
}

// Raise: ascending magical power-up
export function playRaiseSound() {
  sweep(300, 900, 0.15, 'sawtooth', 0.07)
  tone(700, 0.08, 'triangle', 0.1, 0.1)
  tone(900, 0.08, 'triangle', 0.1, 0.16)
  tone(1200, 0.12, 'triangle', 0.12, 0.22)
  // Chip scatter
  for (let i = 0; i < 4; i++) {
    noise(0.02, 0.08, 3500 + Math.random() * 2000, i * 0.04)
  }
}

// All-In: dramatic energy blast
export function playAllInSound() {
  sweep(100, 600, 0.25, 'sawtooth', 0.1)
  sweep(200, 1200, 0.3, 'square', 0.06, 0.05)
  tone(800, 0.15, 'triangle', 0.12, 0.2)
  tone(1000, 0.15, 'triangle', 0.12, 0.28)
  tone(1400, 0.2, 'triangle', 0.15, 0.35)
  // Big chip cascade
  for (let i = 0; i < 8; i++) {
    noise(0.025, 0.1, 3000 + Math.random() * 3000, 0.05 + i * 0.03)
  }
}

// New Hand: mystical card deal — quick sparkle sequence
export function playNewHandSound() {
  tone(523, 0.06, 'sine', 0.06)
  tone(659, 0.06, 'sine', 0.06, 0.08)
  tone(784, 0.06, 'sine', 0.08, 0.16)
  tone(1047, 0.1, 'sine', 0.1, 0.24)
  // Card flick sounds
  noise(0.015, 0.05, 6000, 0.0)
  noise(0.015, 0.05, 6000, 0.1)
  noise(0.015, 0.05, 6000, 0.2)
}

// Timeline Navigate: soft page-turn whoosh
export function playNavigateSound() {
  sweep(400, 800, 0.08, 'sine', 0.06)
  noise(0.04, 0.05, 5000, 0.0)
}

export function playActionSound(actionType) {
  switch (actionType?.toUpperCase()) {
    case 'FOLD': playFoldSound(); break
    case 'CHECK': playCheckSound(); break
    case 'CALL': playCallSound(); break
    case 'RAISE': playRaiseSound(); break
    case 'ALL_IN': playAllInSound(); break
    default: playCheckSound()
  }
}
