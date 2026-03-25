// Poker sound effects using Web Audio API — chip clinks and bass kicks
let audioCtx = null

function getCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)()
  return audioCtx
}

// --- Noise generator for chip/click sounds ---
function playNoiseBurst(duration = 0.04, volume = 0.2, highpass = 3000) {
  try {
    const ctx = getCtx()
    const bufferSize = ctx.sampleRate * duration
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate)
    const data = buffer.getChannelData(0)
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize) // decaying noise
    }
    const source = ctx.createBufferSource()
    source.buffer = buffer

    const hp = ctx.createBiquadFilter()
    hp.type = 'highpass'
    hp.frequency.value = highpass

    const gain = ctx.createGain()
    gain.gain.setValueAtTime(volume, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration)

    source.connect(hp).connect(gain).connect(ctx.destination)
    source.start(ctx.currentTime)
  } catch (e) {}
}

// Chip clink = two quick high-frequency noise bursts
function chipClink(volume = 0.2) {
  playNoiseBurst(0.03, volume, 4000)
  setTimeout(() => playNoiseBurst(0.025, volume * 0.7, 5000), 40)
}

// Multiple chips = rapid burst of clinks
function chipStack(count = 3, volume = 0.15) {
  for (let i = 0; i < count; i++) {
    setTimeout(() => playNoiseBurst(0.025, volume * (0.6 + Math.random() * 0.4), 3500 + Math.random() * 2000), i * 35)
  }
}

// --- Bass/Kick drum for navigation ---
function playKick(volume = 0.3) {
  try {
    const ctx = getCtx()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.type = 'sine'
    osc.frequency.setValueAtTime(150, ctx.currentTime)
    osc.frequency.exponentialRampToValueAtTime(30, ctx.currentTime + 0.12)

    gain.gain.setValueAtTime(volume, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.15)

    osc.connect(gain).connect(ctx.destination)
    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + 0.15)

    // Add a click transient
    playNoiseBurst(0.01, volume * 0.5, 1000)
  } catch (e) {}
}

// --- Exported sounds ---

export function playFoldSound() {
  // Single soft chip toss
  playNoiseBurst(0.05, 0.1, 2000)
}

export function playCheckSound() {
  // Two quick taps (knocking the table)
  chipClink(0.12)
}

export function playCallSound() {
  // A few chips hitting the felt
  chipStack(3, 0.15)
}

export function playRaiseSound() {
  // More aggressive chip stack
  chipStack(5, 0.2)
}

export function playAllInSound() {
  // Big chip shove — lots of chips
  chipStack(8, 0.25)
}

export function playNewHandSound() {
  // Card shuffle / dealing sound
  setTimeout(() => playNoiseBurst(0.02, 0.08, 6000), 0)
  setTimeout(() => playNoiseBurst(0.02, 0.08, 6000), 60)
  setTimeout(() => playNoiseBurst(0.02, 0.1, 5000), 120)
  setTimeout(() => playNoiseBurst(0.02, 0.1, 5000), 180)
}

export function playNavigateSound() {
  playKick(0.25)
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
