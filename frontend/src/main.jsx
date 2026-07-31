import React from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, BarChart3, Camera, GraduationCap, Hand, Radio, RefreshCcw, Shuffle, SlidersHorizontal } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/detect';
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const MODES = [
  { id: 'practice', label: 'Practice', icon: GraduationCap },
  { id: 'quiz', label: 'Quiz', icon: Shuffle },
  { id: 'freestyle', label: 'Freestyle', icon: Hand },
  { id: 'calibration', label: 'Calibration', icon: SlidersHorizontal },
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
  const resultRef = React.useRef(null);

  const [cameraState, setCameraState] = React.useState('idle');
  const [backend, setBackend] = React.useState({ state: 'checking', modelAvailable: false });
  const [dataset, setDataset] = React.useState(null);
  const [result, setResult] = React.useState(null);
  const [fps, setFps] = React.useState(0);
  const [mode, setMode] = React.useState('practice');
  const [target, setTarget] = React.useState('A');
  const [successNotice, setSuccessNotice] = React.useState('');
  const [successOverlay, setSuccessOverlay] = React.useState(null);
  const [calibration, setCalibration] = React.useState({ total: 0, byLetter: {} });
  const [calibrationStatus, setCalibrationStatus] = React.useState('Capture 12-20 samples for letters that are failing.');
  const [capturingCalibration, setCapturingCalibration] = React.useState(false);
  const [trainingCalibration, setTrainingCalibration] = React.useState(false);
  const [stats, setStats] = React.useState(() => {
    try { return JSON.parse(localStorage.getItem('signcoach-progress')) || emptyStats(); }
    catch { return emptyStats(); }
  });
  const frameCounterRef = React.useRef({ count: 0, startedAt: performance.now() });

  React.useEffect(() => { checkBackend(); checkCalibration(); return stopCamera; }, []);
  React.useEffect(() => { drawOverlay(result); }, [result]);
  React.useEffect(() => { resultRef.current = result; }, [result]);
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
    setSuccessNotice(`Success! ${target} confirmed. Next letter…`);
    setSuccessOverlay({ letter: target, key: Date.now() });
    window.setTimeout(() => {
      confirmationRef.current = false;
      setTarget(mode === 'practice' ? nextLetter(target) : randomLetter(target));
      setSuccessNotice('');
      setSuccessOverlay(null);
    }, 900);
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

  async function checkCalibration() {
    try {
      const response = await fetch(`${API_BASE}/api/calibration`);
      setCalibration(await response.json());
    } catch {
      setCalibrationStatus('Start the backend to load calibration samples.');
    }
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
      const item = current.byLetter[letter] || { attempts: 0, correct: 0, incorrect: 0, confidenceTotal: 0 };
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
          attempts: item.attempts + 1,
          correct: item.correct + (correct ? 1 : 0),
          incorrect: (item.incorrect || 0) + (correct ? 0 : 1),
          confidenceTotal: (item.confidenceTotal || 0) + confidence,
          lastConfidence: confidence,
          lastResult: correct ? 'correct' : 'incorrect',
        } },
      };
    });
  }

  function skipAttempt() {
    if (!['practice', 'quiz'].includes(mode)) return;
    recordAttempt(target, false, result?.confidence || 0);
    confirmationRef.current = false;
    setSuccessNotice('');
    setSuccessOverlay(null);
    setTarget(randomLetter(target));
  }

  function selectMode(nextMode) {
    confirmationRef.current = false;
    setSuccessNotice('');
    setSuccessOverlay(null);
    setMode(nextMode);
    if (nextMode === 'quiz') setTarget(randomLetter());
  }

  async function captureCalibrationBurst() {
    if (capturingCalibration) return;
    if (!result?.handDetected || !result?.landmarks?.length) {
      setCalibrationStatus('Start the camera and hold the target sign clearly in frame.');
      return;
    }

    setCapturingCalibration(true);
    setCalibrationStatus(`Capturing ${target} samples. Hold steady.`);
    let saved = 0;
    for (let index = 0; index < 12; index += 1) {
      await wait(160);
      const current = resultRef.current;
      if (!current?.handDetected || !current?.landmarks?.length) continue;
      const response = await fetch(`${API_BASE}/api/calibration/sample`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: target, landmarks: current.landmarks }),
      });
      const summary = await response.json();
      if (!summary.error) {
        saved += 1;
        setCalibration(summary);
      }
    }
    setCapturingCalibration(false);
    setCalibrationStatus(saved ? `Saved ${saved} ${target} samples. Retrain when ready.` : 'No samples saved. Keep your hand visible and try again.');
  }

  async function trainCalibrationModel() {
    setTrainingCalibration(true);
    setCalibrationStatus('Retraining model with calibration samples. This may take a moment.');
    try {
      const response = await fetch(`${API_BASE}/api/calibration/train`, { method: 'POST' });
      const payload = await response.json();
      setCalibration(payload.calibration || calibration);
      setBackend((current) => ({
        ...current,
        modelAvailable: payload.modelAvailable,
        modelError: payload.modelError,
      }));
      const accuracyText = payload.metrics?.accuracy == null ? '' : ` Accuracy ${Math.round(payload.metrics.accuracy * 100)}%.`;
      setCalibrationStatus(payload.modelAvailable ? `Calibrated model loaded.${accuracyText}` : payload.modelError || 'Retraining failed.');
    } catch (error) {
      setCalibrationStatus(error?.message || 'Retraining failed.');
    } finally {
      setTrainingCalibration(false);
    }
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
  const showTarget = ['practice', 'quiz', 'calibration'].includes(mode);

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
              {result?.handDetected && <div className="handBadge"><span>Physical hand</span><strong>{result.handedness || 'Unknown'}</strong></div>}
              {result?.prediction && <div className="predictionBadge">{result.prediction} · {Math.round(result.confidence * 100)}%</div>}
              {successOverlay && <div className="successOverlay" key={successOverlay.key}><span>Success</span><strong>{successOverlay.letter}</strong></div>}
            </div>
            <div className="toolbar">
              <button className="primaryButton" onClick={isRunning ? stopCamera : startCamera}>
                <Camera size={18} />{isRunning ? 'Stop Camera' : 'Start Camera'}
              </button>
              {['practice', 'quiz'].includes(mode) && <button className="secondaryButton" onClick={skipAttempt}>Skip / incorrect</button>}
              {mode === 'calibration' && <button className="secondaryButton" onClick={captureCalibrationBurst} disabled={capturingCalibration}>{capturingCalibration ? 'Capturing…' : 'Capture 12'}</button>}
              <button className="iconButton" onClick={checkBackend} title="Refresh backend"><RefreshCcw size={18} /></button>
            </div>
          </div>

          <aside className="sidePanel">
            <div><p className="eyebrow">{mode} mode</p><h2>{showTarget ? `Practice the letter ${target}` : 'Sign any supported letter'}</h2></div>
            {['practice', 'calibration'].includes(mode) && <ReferenceCard letter={target} />}
            {['practice', 'calibration'].includes(mode) && <div className="letterGrid">{LETTERS.map((letter) => (
              <button key={letter} className={target === letter ? 'selected' : ''} onClick={() => setTarget(letter)}>{letter}</button>
            ))}</div>}
            {mode === 'calibration' && <CalibrationPanel
              calibration={calibration}
              status={calibrationStatus}
              target={target}
              capturing={capturingCalibration}
              training={trainingCalibration}
              onCapture={captureCalibrationBurst}
              onTrain={trainCalibrationModel}
            />}
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
              <Metric label="Physical hand" value={result?.handedness || '—'} />
              <Metric label="MediaPipe raw" value={result?.rawHandedness || '—'} />
              <Metric label="Attempt score" value={result?.attemptScore || 0} />
              <Metric label="Session accuracy" value={`${accuracy}%`} />
            </div>
            <p className="handNote">Physical hand corrects MediaPipe's selfie-camera handedness assumption. MediaPipe raw is shown for debugging.</p>
            {result?.adjustmentReason && <p className="handNote">Adjusted to {result.adjustedPrediction}: {result.adjustmentReason}.</p>}
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
  const rows = LETTERS.map((letter) => {
    const value = stats.byLetter[letter] || {};
    const attempts = value.attempts || 0;
    const correct = value.correct || 0;
    const incorrect = value.incorrect ?? Math.max(0, attempts - correct);
    const letterAccuracy = attempts ? Math.round((correct / attempts) * 100) : 0;
    const averageConfidence = attempts ? Math.round(((value.confidenceTotal || 0) / attempts) * 100) : 0;
    return { letter, attempts, correct, incorrect, accuracy: letterAccuracy, averageConfidence, lastResult: value.lastResult };
  });
  const attemptedRows = rows.filter((row) => row.attempts > 0);
  const focusRows = attemptedRows
    .filter((row) => row.incorrect > 0 || row.accuracy < 80)
    .sort((a, b) => (a.accuracy - b.accuracy) || (b.incorrect - a.incorrect) || (b.attempts - a.attempts))
    .slice(0, 5);
  const strongestRows = attemptedRows
    .filter((row) => row.correct > 0)
    .sort((a, b) => (b.accuracy - a.accuracy) || (b.attempts - a.attempts))
    .slice(0, 5);
  const untouchedRows = rows.filter((row) => row.attempts === 0).slice(0, 8);
  const averageConfidence = stats.attempts ? Math.round((stats.confidenceTotal / stats.attempts) * 100) : 0;
  const coverage = Math.round((attemptedRows.length / LETTERS.length) * 100);
  const minutes = Math.max(1, Math.round((Date.now() - stats.startedAt) / 60000));
  const recommendation = focusRows.length
    ? `Practice ${focusRows.map((row) => row.letter).join(', ')} next.`
    : untouchedRows.length
      ? `Try unpracticed letters: ${untouchedRows.map((row) => row.letter).join(', ')}.`
      : 'Coverage is complete. Keep rotating through quiz mode.';
  return <section className="dashboard">
    <div className="summaryGrid">
      <Metric label="Accuracy" value={`${accuracy}%`} />
      <Metric label="Attempts" value={stats.attempts} />
      <Metric label="Correct" value={stats.correct} />
      <Metric label="Avg confidence" value={`${averageConfidence}%`} />
      <Metric label="Best streak" value={stats.bestStreak} />
      <Metric label="Coverage" value={`${coverage}%`} />
    </div>

    <div className="insightBand">
      <div><p className="eyebrow">Next best action</p><h2>{recommendation}</h2></div>
      <Metric label="Session" value={`${minutes} min`} />
      <Metric label="Current streak" value={stats.streak} />
    </div>

    <div className="dashboardGrid">
      <InsightCard title="Focus letters" eyebrow="Lowest accuracy or recent mistakes" rows={focusRows} empty="No weak letters yet." />
      <InsightCard title="Strongest letters" eyebrow="Best current performance" rows={strongestRows} empty="Complete correct attempts to see strengths." />
    </div>

    <div className="progressCard">
      <div className="cardHeading"><div><p className="eyebrow">Saved on this device</p><h2>Letter mastery map</h2></div>
        <button className="secondaryButton" onClick={onReset}>Reset progress</button></div>
      <div className="masteryGrid">
        {rows.map((row) => <div className={`masteryCell ${masteryClass(row)}`} key={row.letter}>
          <strong>{row.letter}</strong>
          <span>{row.attempts ? `${row.accuracy}%` : '—'}</span>
        </div>)}
      </div>
    </div>

    <div className="progressCard">
      <div className="cardHeading"><div><p className="eyebrow">Detailed progress</p><h2>Per-letter attempts</h2></div></div>
      {attemptedRows.length ? attemptedRows.map((row) => (
        <div className="progressRow" key={row.letter}>
          <strong>{row.letter}</strong>
          <div><span style={{ width: `${row.accuracy}%` }} /></div>
          <span>{row.accuracy}% · {row.correct}/{row.attempts} · {row.averageConfidence}% conf</span>
        </div>
      )) : <p className="emptyState">Complete or skip practice attempts to build your dashboard.</p>}
    </div>
  </section>;
}

function InsightCard({ title, eyebrow, rows, empty }) {
  return <div className="progressCard insightCard">
    <div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div>
    {rows.length ? rows.map((row) => (
      <div className="insightRow" key={row.letter}>
        <strong>{row.letter}</strong>
        <span>{row.accuracy}% accuracy</span>
        <span>{row.incorrect} misses</span>
      </div>
    )) : <p className="emptyState compact">{empty}</p>}
  </div>;
}

function masteryClass(row) {
  if (!row.attempts) return 'empty';
  if (row.accuracy >= 85 && row.attempts >= 2) return 'strong';
  if (row.accuracy >= 60) return 'mixed';
  return 'focus';
}

function ReferenceCard({ letter }) {
  const [failed, setFailed] = React.useState(false);
  React.useEffect(() => setFailed(false), [letter]);

  return <div className="referenceCard">
    <div className="referenceHeader">
      <span>Reference</span>
      <strong>{letter}</strong>
    </div>
    {failed ? (
      <div className="referenceMissing">{letter}</div>
    ) : (
      <img src={`${API_BASE}/api/reference/${letter}`} alt={`ASL reference for ${letter}`} onError={() => setFailed(true)} />
    )}
  </div>;
}

function CalibrationPanel({ calibration, status, target, capturing, training, onCapture, onTrain }) {
  const targetCount = calibration.byLetter?.[target] || 0;
  return <div className="calibrationPanel">
    <div className="calibrationStats">
      <Metric label={`${target} samples`} value={targetCount} />
      <Metric label="Total samples" value={calibration.total || 0} />
    </div>
    <div className="calibrationActions">
      <button className="primaryButton" onClick={onCapture} disabled={capturing || training}>
        {capturing ? 'Capturing…' : 'Capture 12'}
      </button>
      <button className="secondaryButton" onClick={onTrain} disabled={training || !(calibration.total || 0)}>
        {training ? 'Retraining…' : 'Retrain model'}
      </button>
    </div>
    <p>{status}</p>
  </div>;
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
function nextLetter(current) {
  const index = LETTERS.indexOf(current);
  return LETTERS[(index + 1) % LETTERS.length];
}
function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

createRoot(document.getElementById('root')).render(<App />);
