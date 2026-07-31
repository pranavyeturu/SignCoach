import React from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, BarChart3, Camera, GraduationCap, Hand, Radio, RefreshCcw, Shuffle } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/detect';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const MODES = [
  { id: 'practice', label: 'Practice', icon: GraduationCap },
  { id: 'quiz', label: 'Quiz', icon: Shuffle },
  { id: 'freestyle', label: 'Freestyle', icon: Hand },
  { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
];

const emptyStats = () => ({
  startedAt: Date.now(), attempts: 0, correct: 0, incorrect: 0,
  streak: 0, bestStreak: 0, confidenceTotal: 0, byLetter: {},
});

function App() {
  const videoRef = React.useRef(null);
  const overlayRef = React.useRef(null);
  const captureRef = React.useRef(null);
  const socketRef = React.useRef(null);
  const streamRef = React.useRef(null);
  const timerRef = React.useRef(null);
  const inFlightRef = React.useRef(false);
  const confirmationRef = React.useRef(false);
  const modeRef = React.useRef('practice');
  const targetRef = React.useRef('A');

  const [cameraState, setCameraState] = React.useState('idle');
  const [backend, setBackend] = React.useState({ state: 'checking', modelAvailable: false });
  const [dataset, setDataset] = React.useState(null);
  const [result, setResult] = React.useState(null);
  const [fps, setFps] = React.useState(0);
  const [mode, setMode] = React.useState('practice');
  const [target, setTarget] = React.useState('A');
  const [successNotice, setSuccessNotice] = React.useState('');
  const [stats, setStats] = React.useState(() => {
    try { return JSON.parse(localStorage.getItem('signcoach-progress')) || emptyStats(); }
    catch { return emptyStats(); }
  });
  const frameCounterRef = React.useRef({ count: 0, startedAt: performance.now() });

  React.useEffect(() => { checkBackend(); return stopCamera; }, []);
  React.useEffect(() => { drawOverlay(result); }, [result]);
  React.useEffect(() => { localStorage.setItem('signcoach-progress', JSON.stringify(stats)); }, [stats]);
  React.useEffect(() => { modeRef.current = mode; targetRef.current = target; }, [mode, target]);

  React.useEffect(() => {
    if (
      !result?.confirmed ||
      result.target !== target ||
      confirmationRef.current === `${mode}:${target}` ||
      !['practice', 'quiz'].includes(mode)
    ) return;
    confirmationRef.current = `${mode}:${target}`;
    recordAttempt(target, true, result.confidence);
    if (mode === 'quiz') {
      setSuccessNotice(`Correct! ${target} confirmed — next letter…`);
      window.setTimeout(() => {
        setTarget(randomLetter(target));
        setSuccessNotice('');
      }, 700);
    }
  }, [result, mode, target]);

  async function checkBackend() {
    try {
      const [healthResponse, datasetResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`), fetch(`${API_BASE}/api/dataset`),
      ]);
      const health = await healthResponse.json();
      setBackend({
        state: health.mediapipeAvailable ? 'ready' : 'missing-mediapipe',
        modelAvailable: health.modelAvailable,
        modelError: health.modelError,
      });
      setDataset(await datasetResponse.json());
    } catch { setBackend({ state: 'offline', modelAvailable: false }); }
  }

  async function startCamera() {
    setCameraState('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      connectSocket();
      timerRef.current = window.setInterval(captureFrame, 125);
      setCameraState('running');
    } catch (error) {
      setCameraState('blocked');
      setResult({ handDetected: false, message: error?.message || 'Camera is unavailable.', landmarks: [] });
    }
  }

  function stopCamera() {
    window.clearInterval(timerRef.current);
    timerRef.current = null;
    inFlightRef.current = false;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState('idle');
    setFps(0);
    clearOverlay();
  }

  function connectSocket() {
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      inFlightRef.current = false;
      setResult(JSON.parse(event.data));
      updateFps();
    };
    socket.onclose = () => { inFlightRef.current = false; };
  }

  function captureFrame() {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (!video || !socket || socket.readyState !== WebSocket.OPEN || inFlightRef.current) return;
    const canvas = captureRef.current;
    canvas.width = 640;
    canvas.height = Math.round(640 * (video.videoHeight / video.videoWidth || 0.75));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    inFlightRef.current = true;
    const activeTarget = ['practice', 'quiz'].includes(modeRef.current) ? targetRef.current : null;
    socket.send(JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.72), target: activeTarget }));
  }

  function updateFps() {
    const counter = frameCounterRef.current;
    counter.count += 1;
    const elapsed = performance.now() - counter.startedAt;
    if (elapsed >= 1000) {
      setFps(Math.round((counter.count * 1000) / elapsed));
      frameCounterRef.current = { count: 0, startedAt: performance.now() };
    }
  }

  function recordAttempt(letter, correct, confidence) {
    setStats((current) => {
      const item = current.byLetter[letter] || { attempts: 0, correct: 0 };
      const streak = correct ? current.streak + 1 : 0;
      return {
        ...current,
        attempts: current.attempts + 1,
        correct: current.correct + (correct ? 1 : 0),
        incorrect: current.incorrect + (correct ? 0 : 1),
        streak,
        bestStreak: Math.max(current.bestStreak, streak),
        confidenceTotal: current.confidenceTotal + confidence,
        byLetter: { ...current.byLetter, [letter]: {
          attempts: item.attempts + 1, correct: item.correct + (correct ? 1 : 0),
        } },
      };
    });
  }

  function skipAttempt() {
    if (!['practice', 'quiz'].includes(mode)) return;
    recordAttempt(target, false, result?.confidence || 0);
    if (mode === 'quiz') setTarget(randomLetter(target));
  }

  function selectMode(nextMode) {
    confirmationRef.current = false;
    setSuccessNotice('');
    setMode(nextMode);
    if (nextMode === 'quiz') setTarget(randomLetter());
  }

  function drawOverlay(current) {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    canvas.width = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!current?.handDetected) return;
    context.lineWidth = 3;
    context.strokeStyle = '#55d6be';
    for (const [start, end] of current.connections || []) {
      context.beginPath();
      context.moveTo((1 - current.landmarks[start].x) * canvas.width, current.landmarks[start].y * canvas.height);
      context.lineTo((1 - current.landmarks[end].x) * canvas.width, current.landmarks[end].y * canvas.height);
      context.stroke();
    }
    current.landmarks.forEach((point) => {
      context.fillStyle = '#fff';
      context.beginPath();
      context.arc((1 - point.x) * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
      context.fill();
    });
  }

  function clearOverlay() {
    const canvas = overlayRef.current;
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }

  const isRunning = cameraState === 'running';
  const accuracy = stats.attempts ? Math.round((stats.correct / stats.attempts) * 100) : 0;
  const showTarget = ['practice', 'quiz'].includes(mode);

  return (
    <main className="appShell">
      <header className="topbar">
        <div><p className="eyebrow">Realtime ASL tutor</p><h1>SignCoach AI</h1></div>
        <nav>{MODES.map(({ id, label, icon: Icon }) => (
          <button key={id} className={mode === id ? 'tab active' : 'tab'} onClick={() => selectMode(id)}>
            <Icon size={17} />{label}
          </button>
        ))}</nav>
      </header>

      {mode === 'dashboard' ? (
        <Dashboard stats={stats} accuracy={accuracy} onReset={() => setStats(emptyStats())} />
      ) : (
        <section className="workspace">
          <div className="cameraPane">
            <div className="videoStage">
              <video ref={videoRef} className="videoFeed" playsInline muted />
              <canvas ref={overlayRef} className="overlay" />
              {!isRunning && <div className="cameraEmpty"><Hand size={48} /><span>Camera is {cameraState === 'blocked' ? 'unavailable' : 'off'}</span></div>}
              {showTarget && <div className="targetBadge"><span>Show</span><strong>{target}</strong></div>}
              {result?.prediction && <div className="predictionBadge">{result.prediction} · {Math.round(result.confidence * 100)}%</div>}
            </div>
            <div className="toolbar">
              <button className="primaryButton" onClick={isRunning ? stopCamera : startCamera}>
                <Camera size={18} />{isRunning ? 'Stop Camera' : 'Start Camera'}
              </button>
              {showTarget && <button className="secondaryButton" onClick={skipAttempt}>Skip / incorrect</button>}
              <button className="iconButton" onClick={checkBackend} title="Refresh backend"><RefreshCcw size={18} /></button>
            </div>
          </div>

          <aside className="sidePanel">
            <div><p className="eyebrow">{mode} mode</p><h2>{showTarget ? `Practice the letter ${target}` : 'Sign any supported letter'}</h2></div>
            {mode === 'practice' && <div className="letterGrid">{LETTERS.map((letter) => (
              <button key={letter} className={target === letter ? 'selected' : ''} onClick={() => setTarget(letter)}>{letter}</button>
            ))}</div>}
            <Status icon={<Radio size={18} />} label="Backend" value={backendLabel(backend)} />
            <Status icon={<Camera size={18} />} label="Camera" value={cameraState} />
            <Status icon={<Activity size={18} />} label="Inference" value={`${fps} FPS`} />
            <div className={`feedback ${result?.confirmed || successNotice ? 'good' : 'warn'}`}>
              {successNotice || statusFor(cameraState, backend, result)}
            </div>
            <div className="holdTrack"><span style={{ width: `${(result?.holdProgress || 0) * 100}%` }} /></div>
            <div className="metricsGrid">
              <Metric label="Prediction" value={result?.prediction || '—'} />
              <Metric label="Confidence" value={`${Math.round((result?.confidence || 0) * 100)}%`} />
              <Metric label="Attempt score" value={result?.attemptScore || 0} />
              <Metric label="Session accuracy" value={`${accuracy}%`} />
            </div>
            <p className="modelNote">{backend.modelAvailable ? 'Trained classifier loaded.' : backend.modelError || 'Classifier unavailable.'}</p>
            <p className="datasetNote">{dataset?.image_count || 0} local dataset images · {dataset?.classes?.length || 0} classes</p>
          </aside>
        </section>
      )}
      <canvas ref={captureRef} className="hiddenCanvas" />
    </main>
  );
}

function Dashboard({ stats, accuracy, onReset }) {
  const rows = Object.entries(stats.byLetter).sort((a, b) => {
    const accuracyA = a[1].correct / a[1].attempts;
    const accuracyB = b[1].correct / b[1].attempts;
    return accuracyA - accuracyB;
  });
  const minutes = Math.max(1, Math.round((Date.now() - stats.startedAt) / 60000));
  return <section className="dashboard">
    <div className="summaryGrid">
      <Metric label="Accuracy" value={`${accuracy}%`} />
      <Metric label="Attempts" value={stats.attempts} />
      <Metric label="Correct" value={stats.correct} />
      <Metric label="Best streak" value={stats.bestStreak} />
      <Metric label="Current streak" value={stats.streak} />
      <Metric label="Session" value={`${minutes} min`} />
    </div>
    <div className="progressCard">
      <div className="cardHeading"><div><p className="eyebrow">Saved on this device</p><h2>Per-letter progress</h2></div>
        <button className="secondaryButton" onClick={onReset}>Reset progress</button></div>
      {rows.length ? rows.map(([letter, value]) => {
        const percent = Math.round((value.correct / value.attempts) * 100);
        return <div className="progressRow" key={letter}><strong>{letter}</strong><div><span style={{ width: `${percent}%` }} /></div><span>{percent}% · {value.attempts} tries</span></div>;
      }) : <p className="emptyState">Complete or skip practice attempts to build your dashboard.</p>}
    </div>
  </section>;
}

function Status({ icon, label, value }) {
  return <div className="statusRow"><div className="statusLabel">{icon}<span>{label}</span></div><strong>{value}</strong></div>;
}
function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}
function backendLabel(backend) {
  if (backend.state === 'ready') return backend.modelAvailable ? 'Ready' : 'Needs model';
  if (backend.state === 'missing-mediapipe') return 'Install deps';
  return backend.state;
}
function statusFor(camera, backend, result) {
  if (backend.state === 'offline') return 'Start the backend on port 8000.';
  if (backend.state === 'missing-mediapipe') return 'Install backend dependencies with Python 3.10–3.12.';
  if (!backend.modelAvailable) return backend.modelError || 'Train the classifier to enable predictions.';
  if (camera === 'idle') return 'Start the camera when you are ready.';
  if (camera === 'blocked') return result?.message || 'Camera permission was denied.';
  return result?.message || 'Waiting for a hand.';
}
function randomLetter(exclude) {
  const choices = LETTERS.filter((letter) => letter !== exclude);
  return choices[Math.floor(Math.random() * choices.length)];
}

createRoot(document.getElementById('root')).render(<App />);
