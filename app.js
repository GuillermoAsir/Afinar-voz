// app.js
// High-level: device selection, audio graph, real-time pitch detection (autocorrelation), note mapping, UI updates.

// DOM
const micSelect = document.getElementById('micSelect');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const refToneToggle = document.getElementById('refToneToggle');
const errorBox = document.getElementById('errorBox');
const secureContextWarning = document.getElementById('secureContextWarning');
const audioActive = document.getElementById('audioActive');

const noteNameEl = document.getElementById('noteName');
const freqHzEl = document.getElementById('freqHz');
const centsEl = document.getElementById('cents');
const needleEl = document.getElementById('needle');
const pianoEl = document.getElementById('piano');
const historyCanvas = document.getElementById('history');
const hctx = historyCanvas.getContext('2d');

// State
let audioContext = null;
let analyser = null;
let sourceNode = null;
let hpFilter = null;
let lpFilter = null;
let compressor = null;
let refOsc = null;

let mediaStream = null;
let devices = [];
let running = false;

const sampleBuffer = new Float32Array(4096);
let animationId = null;

let smoothedFreq = 0;
const smoothAlpha = 0.2;

const MIN_RMS = 0.01;
const MIN_FREQUENCY = 80;
const MAX_FREQUENCY = 1500;

// Utilities — note math
const A4 = 440;
const NOTE_NAMES = ['Do', 'Do♯/Re♭', 'Re', 'Re♯/Mi♭', 'Mi', 'Fa', 'Fa♯/Sol♭', 'Sol', 'Sol♯/La♭', 'La', 'La♯/Si♭', 'Si'];

function freqToMidi(freq) {
  return Math.round(12 * Math.log2(freq / A4) + 69);
}
function midiToFreq(midi) {
  return A4 * Math.pow(2, (midi - 69) / 12);
}
function midiToNoteName(midi) {
  const name = NOTE_NAMES[midi % 12];
  const octave = Math.floor(midi / 12) - 1;
  return `${name}${octave}`;
}
function centsOff(freq, target) {
  return Math.round(1200 * Math.log2(freq / target));
}

// UI helpers
function setError(msg) {
  errorBox.textContent = msg || '';
  errorBox.classList.toggle('hidden', !msg);
}
function setAudioActive(active) {
  audioActive.classList.toggle('active', !!active);
}
function setSecureContextWarning() {
  const secure = window.isSecureContext || location.hostname === 'localhost';
  secureContextWarning.classList.toggle('hidden', secure);
}

// Device enumeration
async function getDevices() {
  const list = await navigator.mediaDevices.enumerateDevices();
  devices = list.filter(d => d.kind === 'audioinput');
  micSelect.innerHTML = '';
  devices.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Microphone ${i + 1}`;
    micSelect.appendChild(opt);
  });
  if (devices.length === 0) {
    setError('No audio input devices found.');
  }
}

// Audio graph
function buildGraph() {
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 4096;
  analyser.smoothingTimeConstant = 0;

  hpFilter = audioContext.createBiquadFilter();
  hpFilter.type = 'highpass';
  hpFilter.frequency.value = MIN_FREQUENCY - 10;

  lpFilter = audioContext.createBiquadFilter();
  lpFilter.type = 'lowpass';
  lpFilter.frequency.value = MAX_FREQUENCY + 200;

  compressor = audioContext.createDynamicsCompressor();
  compressor.threshold.value = -30;
  compressor.knee.value = 20;
  compressor.ratio.value = 3;
  compressor.attack.value = 0.003;
  compressor.release.value = 0.25;

  sourceNode.connect(hpFilter);
  hpFilter.connect(lpFilter);
  lpFilter.connect(compressor);
  compressor.connect(analyser);
}

function destroyStream() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (sourceNode) sourceNode.disconnect();
  if (hpFilter) hpFilter.disconnect();
  if (lpFilter) lpFilter.disconnect();
  if (compressor) compressor.disconnect();
  sourceNode = hpFilter = lpFilter = compressor = analyser = null;
  setAudioActive(false);
}

// Start/Stop
async function start(deviceId) {
  setError('');
  try {
    audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    const constraints = {
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: audioContext.sampleRate
      }
    };
    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    buildGraph();
    running = true;
    setAudioActive(true);
    loop();
  } catch (err) {
    setError(`Microphone access failed: ${err.name || 'Error'}`);
    running = false;
    setAudioActive(false);
  }
}

function stop() {
  running = false;
  destroyStream();
}

// Pitch detection — autocorrelation with peak interpolation
function detectPitchAutoCorr(buf, sampleRate) {
  let rms = 0;
  for (let i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / buf.length);
  if (rms < MIN_RMS) return null;

  const size = buf.length;
  const c = new Float32Array(size);
  for (let lag = 0; lag < size; lag++) {
    let sum = 0;
    for (let i = 0; i < size - lag; i++) sum += buf[i] * buf[i + lag];
    c[lag] = sum;
  }

  let d = 0;
  while (d < size - 1 && c[d] > c[d + 1]) d++;
  let max = -1, maxPos = -1;
  for (let i = d; i < size; i++) {
    if (c[i] > max) {
      max = c[i];
      maxPos = i;
    }
  }
  if (maxPos <= 0) return null;

  const y1 = c[maxPos - 1], y2 = c[maxPos], y3 = c[maxPos + 1] || y2;
  const shift = (y3 - y1) / (2 * (2 * y2 - y1 - y3));
  const peak = maxPos + shift;

  const freq = sampleRate / peak;
  if (freq < MIN_FREQUENCY || freq > MAX_FREQUENCY) return null;
  return freq;
}

// Loop
function loop() {
  animationId = requestAnimationFrame(loop);
  analyser.getFloatTimeDomainData(sampleBuffer);
  const freq = detectPitchAutoCorr(sampleBuffer, audioContext.sampleRate);
  if (!freq) {
    noteNameEl.textContent = '—';
    freqHzEl.textContent = '0.0 Hz';
    centsEl.textContent = '—';
    moveNeedle(0);
    highlightPiano(null);
    pushHistory(null);
    return;
  }

  smoothedFreq = smoothedFreq ? (smoothAlpha * freq + (1 - smoothAlpha) * smoothedFreq) : freq;

  const midi = freqToMidi(smoothedFreq);
  const targetFreq = midiToFreq(midi);
  const name = midiToNoteName(midi);
  const cents = centsOff(smoothedFreq, targetFreq);

  noteNameEl.textContent = name;
  freqHzEl.textContent = `${smoothedFreq.toFixed(1)} Hz`;
  centsEl.textContent = `${cents} cents`;

  moveNeedle(cents);
  highlightPiano(midi);
  pushHistory(smoothedFreq);
}

// Needle mapping
function moveNeedle(cents) {
  const clamped = Math.max(-50, Math.min(50, cents || 0));
  const deg = (clamped / 50) * 25;
  needleEl.style.transform = `translateX(-50%) rotate(${deg}deg)`;
}

// Piano
function buildPiano() {
  const startMidi = 48; // C3
  const endMidi = 84;   // C6
  for (let m = startMidi; m <= endMidi; m++) {
    const name = NOTE_NAMES[m % 12];
    const isBlack = name.includes('♯');
    const key = document.createElement('div');
    key.className = `key ${isBlack ? 'black' : 'white'}`;
    key.dataset.midi = String(m);
    pianoEl.appendChild(key);
  }
}
function highlightPiano(midi) {
  const keys = pianoEl.querySelectorAll('.key');
  keys.forEach(k => k.classList.remove('active'));
  if (midi == null) return;
  const key = pianoEl.querySelector(`.key[data-midi="${midi}"]`);
  if (key) key.classList.add('active');
}

// History graph
const history = [];
const HISTORY_MAX = 800;
function pushHistory(freq) {
  history.push(freq);
  if (history.length > HISTORY_MAX) history.shift();
  drawHistory();
}
function drawHistory() {
  hctx.clearRect(0, 0, historyCanvas.width, historyCanvas.height);
  hctx.strokeStyle = '#6cf3ff';
  hctx.lineWidth = 2;
  hctx.beginPath();
  let first = true;
  for (let i = 0; i < history.length; i++) {
    const f = history[i];
    const x = (i / HISTORY_MAX) * historyCanvas.width;
    const y = (1 - mapFreq(f)) * historyCanvas.height;
    if (f == null) continue;
    if (first) { hctx.moveTo(x, y); first = false; } else { hctx.lineTo(x, y); }
  }
  hctx.stroke();
  hctx.strokeStyle = '#2a2f4a';
  hctx.beginPath();
  hctx.moveTo(0, historyCanvas.height / 2);
  hctx.lineTo(historyCanvas.width, historyCanvas.height / 2);
  hctx.stroke();
}
function mapFreq(f) {
  if (!f || f < MIN_FREQUENCY) return 0;
  const clamped = Math.min(MAX_FREQUENCY, Math.max(MIN_FREQUENCY, f));
  return (clamped - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY);
}

// Reference tone
function setRefTone(on) {
  if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
  if (on) {
    if (!refOsc) {
      refOsc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      gain.gain.value = 0.08;
      refOsc.frequency.value = A4;
      refOsc.connect(gain).connect(audioContext.destination);
      refOsc.start();
    }
  } else {
    if (refOsc) {
      refOsc.stop();
      refOsc.disconnect();
      refOsc = null;
    }
  }
}

// Events
startBtn.addEventListener('click', async () => {
  await start(micSelect.value || undefined);
});
stopBtn.addEventListener('click', () => {
  stop();
});
refToneToggle.addEventListener('change', (e) => {
  setRefTone(e.target.checked);
});
micSelect.addEventListener('change', async (e) => {
  if (!running) return;
  destroyStream();
  await start(e.target.value);
});

// Boot
async function boot() {
  setSecureContextWarning();
  buildPiano();

  try {
    // Initial permission request on load to reveal device labels
    await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    // Some browsers require a user gesture; allow Start button to proceed
  }

  await getDevices();

  navigator.mediaDevices.addEventListener('devicechange', async () => {
    await getDevices();
  });
}

boot();