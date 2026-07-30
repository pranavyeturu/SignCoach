import React from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, Camera, Hand, Radio, RefreshCcw } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://127.0.0.1:8000';
const WS_URL = 'ws://127.0.0.1:8000/ws/detect';
const FRAME_INTERVAL_MS = 125;

const LANDMARK_COLORS = {
  wrist: '#f8fafc',
  thumb: '#ff8a65',
  index: '#4dd0e1',
  middle: '#81c784',
  ring: '#ffd54f',
  pinky: '#ba68c8',
};

const FINGER_GROUPS = [
  { name: 'thumb', points: [1, 2, 3, 4] },
  { name: 'index', points: [5, 6, 7, 8] },
  { name: 'middle', points: [9, 10, 11, 12] },
  { name: 'ring', points: [13, 14, 15, 16] },
  { name: 'pinky', points: [17, 18, 19, 20] },
];

function App() {
  const videoRef = React.useRef(null);
  const overlayRef = React.useRef(null);
  const captureRef = React.useRef(null);
  const socketRef = React.useRef(null);
  const streamRef = React.useRef(null);
  const timerRef = React.useRef(null);
  const inFlightRef = React.useRef(false);

  const [cameraState, setCameraState] = React.useState('idle');
  const [backendState, setBackendState] = React.useState('checking');
  const [dataset, setDataset] = React.useState(null);
  const [result, setResult] = React.useState(null);
  const [fps, setFps] = React.useState(0);
  const frameCounterRef = React.useRef({ count: 0, startedAt: performance.now() });

  React.useEffect(() => {
    checkBackend();
    return stopCamera;
  }, []);

  React.useEffect(() => {
    drawOverlay(result);
  }, [result]);

  async function checkBackend() {
    try {
      const [healthResponse, datasetResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`),
        fetch(`${API_BASE}/api/dataset`),
      ]);
      const health = await healthResponse.json();
      const datasetSummary = await datasetResponse.json();
      setBackendState(health.mediapipeAvailable ? 'ready' : 'missing-mediapipe');
      setDataset(datasetSummary);
    } catch {
      setBackendState('offline');
    }
  }

  async function startCamera() {
    setCameraState('starting');
    setResult(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      connectSocket();
      timerRef.current = window.setInterval(captureFrame, FRAME_INTERVAL_MS);
      setCameraState('running');
    } catch (error) {
      setCameraState('blocked');
      setResult({
        handDetected: false,
        message: error?.message || 'Camera is unavailable.',
        landmarks: [],
        connections: [],
      });
    }
  }

  function stopCamera() {
    window.clearInterval(timerRef.current);
    timerRef.current = null;
    inFlightRef.current = false;

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

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

    socket.onclose = () => {
      inFlightRef.current = false;
    };
  }

  function captureFrame() {
    const video = videoRef.current;
    const socket = socketRef.current;
    if (!video || !socket || socket.readyState !== WebSocket.OPEN || inFlightRef.current) {
      return;
    }

    const canvas = captureRef.current;
    const maxWidth = 640;
    const ratio = video.videoHeight / video.videoWidth || 0.75;
    canvas.width = maxWidth;
    canvas.height = Math.round(maxWidth * ratio);
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    inFlightRef.current = true;
    socket.send(JSON.stringify({ image: canvas.toDataURL('image/jpeg', 0.72) }));
  }

  function updateFps() {
    const state = frameCounterRef.current;
    state.count += 1;
    const now = performance.now();
    const elapsed = now - state.startedAt;
    if (elapsed >= 1000) {
      setFps(Math.round((state.count * 1000) / elapsed));
      frameCounterRef.current = { count: 0, startedAt: now };
    }
  }

  function drawOverlay(currentResult) {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = video.getBoundingClientRect();
    canvas.width = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);

    if (!currentResult?.handDetected) return;

    const points = currentResult.landmarks;
    context.lineWidth = 3;
    context.lineCap = 'round';

    for (const [start, end] of currentResult.connections || []) {
      const color = colorForConnection(start, end);
      context.strokeStyle = color;
      context.beginPath();
      context.moveTo(toCanvasX(points[start].x, canvas.width), points[start].y * canvas.height);
      context.lineTo(toCanvasX(points[end].x, canvas.width), points[end].y * canvas.height);
      context.stroke();
    }

    points.forEach((point, index) => {
      context.fillStyle = colorForPoint(index);
      context.beginPath();
      context.arc(toCanvasX(point.x, canvas.width), point.y * canvas.height, index === 0 ? 6 : 4.5, 0, Math.PI * 2);
      context.fill();
    });

    if (currentResult.boundingBox) {
      const box = currentResult.boundingBox;
      context.strokeStyle = '#ffffff';
      context.lineWidth = 2;
      context.setLineDash([7, 7]);
      context.strokeRect(
        (1 - box.x - box.width) * canvas.width,
        box.y * canvas.height,
        box.width * canvas.width,
        box.height * canvas.height,
      );
      context.setLineDash([]);
    }
  }

  function clearOverlay() {
    const canvas = overlayRef.current;
    if (!canvas) return;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }

  const isRunning = cameraState === 'running';
  const statusLabel = statusFor(cameraState, backendState, result);

  return (
    <main className="appShell">
      <section className="workspace">
        <div className="cameraPane">
          <div className="videoStage">
            <video ref={videoRef} className="videoFeed" playsInline muted />
            <canvas ref={overlayRef} className="overlay" />
            {cameraState !== 'running' && (
              <div className="cameraEmpty">
                <Hand size={48} strokeWidth={1.7} />
                <span>{cameraState === 'blocked' ? 'Camera unavailable' : 'Camera is off'}</span>
              </div>
            )}
          </div>
          <div className="toolbar">
            <button className="primaryButton" onClick={isRunning ? stopCamera : startCamera}>
              <Camera size={18} />
              <span>{isRunning ? 'Stop Camera' : 'Start Camera'}</span>
            </button>
            <button className="iconButton" onClick={checkBackend} title="Refresh backend status">
              <RefreshCcw size={18} />
            </button>
          </div>
        </div>

        <aside className="sidePanel">
          <div>
            <p className="eyebrow">SignCoach AI</p>
            <h1>Phase 1 Landmark Check</h1>
          </div>

          <StatusRow icon={<Radio size={18} />} label="Backend" value={backendLabel(backendState)} />
          <StatusRow icon={<Camera size={18} />} label="Camera" value={cameraLabel(cameraState)} />
          <StatusRow icon={<Activity size={18} />} label="Inference" value={`${fps} FPS`} />

          <div className={`feedback ${result?.handDetected ? 'good' : 'warn'}`}>
            <span>{statusLabel}</span>
          </div>

          <div className="metricsGrid">
            <Metric label="Hand" value={result?.handDetected ? 'Detected' : 'None'} />
            <Metric label="Handedness" value={result?.handedness || '-'} />
            <Metric label="Framing" value={`${Math.round((result?.framingScore || 0) * 100)}%`} />
            <Metric label="Landmarks" value={result?.landmarks?.length || 0} />
          </div>

          <div className="datasetPanel">
            <h2>Dataset</h2>
            <p>{dataset?.image_count || 0} images across {dataset?.classes?.length || 0} classes</p>
            <div className="classList">
              {(dataset?.classes || []).map((className) => (
                <span key={className}>{className}</span>
              ))}
            </div>
          </div>
        </aside>
      </section>
      <canvas ref={captureRef} className="hiddenCanvas" />
    </main>
  );
}

function StatusRow({ icon, label, value }) {
  return (
    <div className="statusRow">
      <div className="statusLabel">
        {icon}
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function backendLabel(state) {
  if (state === 'ready') return 'Ready';
  if (state === 'missing-mediapipe') return 'Install deps';
  if (state === 'offline') return 'Offline';
  return 'Checking';
}

function cameraLabel(state) {
  if (state === 'running') return 'Running';
  if (state === 'starting') return 'Starting';
  if (state === 'blocked') return 'Blocked';
  return 'Idle';
}

function statusFor(cameraState, backendState, result) {
  if (backendState === 'offline') return 'Start the backend on port 8000.';
  if (backendState === 'missing-mediapipe') return 'Backend is running, but MediaPipe is not installed.';
  if (cameraState === 'idle') return 'Start the camera to test hand landmark detection.';
  if (cameraState === 'starting') return 'Requesting camera access.';
  if (cameraState === 'blocked') return result?.message || 'Camera permission was denied or unavailable.';
  return result?.message || 'Waiting for the first frame.';
}

function colorForPoint(index) {
  if (index === 0) return LANDMARK_COLORS.wrist;
  const group = FINGER_GROUPS.find((entry) => entry.points.includes(index));
  return LANDMARK_COLORS[group?.name] || '#ffffff';
}

function colorForConnection(start, end) {
  if (start === 0 || end === 0) return LANDMARK_COLORS.wrist;
  return colorForPoint(end);
}

function toCanvasX(normalizedX, width) {
  return (1 - normalizedX) * width;
}

createRoot(document.getElementById('root')).render(<App />);
