import React, { useState, useEffect, useRef } from 'react';
import {
  Shield,
  ShieldAlert,
  Play,
  Pause,
  Settings,
  Terminal,
  Compass,
  Battery,
  AlertTriangle,
  Zap,
  Activity,
  Cpu,
  RotateCcw,
  Trash2,
  Navigation,
  Crosshair,
  Sliders,
  Video
} from 'lucide-react';

function App() {
  // Navigation tabs: 'dashboard', 'settings', 'training'
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // Pipeline / Telemetry status state
  const [status, setStatus] = useState({
    pipeline_running: false,
    paused: false,
    guidance_active: false,
    follow_target_enabled: false,
    active_controller: 'FOLLOW_TARGET',
    lock_state: 'ACQUISITION',
    target_id: null,
    target_distance: 0.0,
    fps: 0.0,
    warnings: [],
    telemetry: {},
    current_cmd: { vx: 0.0, vy: 0.0, vz: 0.0, yaw_rate: 0.0 },
    system_mode: 'simulation',
    camera_source: '0'
  });
  
  // Connection states
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [takeoffAlt, setTakeoffAlt] = useState(3.0);
  
  // Config state
  const [config, setConfig] = useState(null);
  const [configLoading, setConfigLoading] = useState(false);
  
  // Task/Training WebSocket state
  const [activeTask, setActiveTask] = useState(null);
  const [logs, setLogs] = useState([]);
  const [taskStatus, setTaskStatus] = useState('idle'); // idle, running, completed, error
  const socketRef = useRef(null);
  const terminalEndRef = useRef(null);

  // Available cameras for webcam mode
  const [availableCameras, setAvailableCameras] = useState([]);

  // Fetch available cameras
  useEffect(() => {
    const getCameras = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true }); // Request permission to get labels
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setAvailableCameras(videoDevices);
      } catch (err) {
        console.error("Error fetching cameras:", err);
      }
    };
    if (activeTab === 'settings' || status.system_mode === 'webcam') {
      getCameras();
    }
  }, [activeTab, status.system_mode]);

  // Poll status from FastAPI backend
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/status');
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          setIsBackendConnected(true);
        } else {
          setIsBackendConnected(false);
        }
      } catch (err) {
        setIsBackendConnected(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 400);
    return () => clearInterval(interval);
  }, []);

  // Auto scroll terminal logs
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Load config on mount/tab change
  const fetchConfig = async () => {
    setConfigLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/config');
      if (res.ok) {
        const data = await res.json();
        setConfig(data);
      }
    } catch (err) {
      console.error('Failed to load settings:', err);
    } finally {
      setConfigLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'settings') {
      fetchConfig();
    }
  }, [activeTab]);

  // Handle setting updates
  const handleConfigChange = (section, key, value) => {
    // Parse numeric fields properly
    let parsedValue = value;
    if (value !== '' && !isNaN(value) && typeof value !== 'boolean') {
      parsedValue = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
    } else if (value === 'true') {
      parsedValue = true;
    } else if (value === 'false') {
      parsedValue = false;
    }

    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: parsedValue
      }
    }));
  };

  const saveConfig = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (res.ok) {
        alert('Configuration saved! Restarting tracking pipeline to apply changes...');
        await fetch('http://127.0.0.1:8000/api/pipeline/restart', { method: 'POST' });
      } else {
        alert('Failed to save configuration.');
      }
    } catch (err) {
      alert('Error connecting to backend: ' + err.message);
    }
  };

  // Telemetry API control triggers
  const triggerControlAction = async (action, additional = {}) => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/control/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...additional })
      });
      if (!res.ok) {
        const err = await res.json();
        console.error('Control error:', err.detail);
      }
    } catch (err) {
      console.error('Failed to execute command:', err);
    }
  };

  const triggerToggle = async (target) => {
    try {
      await fetch(`http://127.0.0.1:8000/api/control/toggle/${target}`, { method: 'POST' });
    } catch (err) {
      console.error(`Failed to toggle ${target}:`, err);
    }
  };

  const triggerControllerSelect = async (controller) => {
    try {
      await fetch('http://127.0.0.1:8000/api/control/controller', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ controller })
      });
    } catch (err) {
      console.error('Failed to update guidance law:', err);
    }
  };

  const triggerPipelineRestart = async () => {
    try {
      await fetch('http://127.0.0.1:8000/api/pipeline/restart', { method: 'POST' });
    } catch (err) {
      console.error('Failed to restart tracking pipeline:', err);
    }
  };

  const triggerModeSelect = async (mode) => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/pipeline/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      if (res.ok) {
        console.log("Successfully set system mode to", mode);
      } else {
        const err = await res.json();
        alert('Failed to set mode: ' + (err.detail || 'unknown error'));
      }
    } catch (err) {
      console.error('Failed to change system mode:', err);
    }
  };

  // WebSockets Task/Training triggers
  const startTask = (taskName) => {
    if (socketRef.current) {
      socketRef.current.close();
    }

    setLogs([]);
    setActiveTask(taskName);
    setTaskStatus('running');

    // Create websocket
    const wsUrl = `ws://127.0.0.1:8000/api/ws/run_task`;
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      // Send task to start
      socket.send(JSON.stringify({ task: taskName }));
      setLogs([{ type: 'status', message: `[WEBAPI] Establishing WebSocket connection for ${taskName}...` }]);
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'log') {
        setLogs((prev) => [...prev, { type: 'log', message: data.message }]);
      } else if (data.type === 'status') {
        setLogs((prev) => [...prev, { type: 'status', message: data.message }]);
      } else if (data.type === 'done') {
        setLogs((prev) => [...prev, { type: 'status', message: data.message }]);
        setTaskStatus('completed');
        socket.close();
      } else if (data.type === 'error') {
        setLogs((prev) => [...prev, { type: 'error', message: `ERROR: ${data.message}` }]);
        setTaskStatus('error');
        socket.close();
      }
    };

    socket.onclose = () => {
      socketRef.current = null;
      logger.info('WebSocket connection closed.');
    };

    socket.onerror = (err) => {
      setLogs((prev) => [...prev, { type: 'error', message: 'WebSocket communication error.' }]);
      setTaskStatus('error');
    };
  };

  const stopActiveTask = () => {
    if (socketRef.current) {
      socketRef.current.close();
      setLogs((prev) => [...prev, { type: 'error', message: '[WEBAPI] Subprocess session interrupted by user.' }]);
      setTaskStatus('idle');
      setActiveTask(null);
    }
  };

  const clearTerminal = () => {
    setLogs([]);
  };

  // Helper colors mapping for status badges
  const getLockStateClass = (state) => {
    switch (state) {
      case 'LOCKED': return 'locked';
      case 'REACQUISITION': return 'acquisition';
      case 'LOST': return 'lost';
      default: return 'acquisition';
    }
  };

  return (
    <div className="dashboard-container">
      {/* Top Header */}
      <header className="header">
        <div className="header-title-container">
          <Shield className="status-dot active" style={{ width: '26px', height: '26px', color: 'var(--primary)' }} />
          <h1>ORCUS AUTO-GUIDED DECK</h1>
        </div>

        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {isBackendConnected && (
            <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', padding: '2px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
              <button
                onClick={() => triggerModeSelect('simulation')}
                style={{
                  background: (status.system_mode === 'simulation' || !status.system_mode) ? 'var(--primary)' : 'none',
                  color: (status.system_mode === 'simulation' || !status.system_mode) ? '#000' : 'var(--text-muted)',
                  border: 'none', padding: '5px 12px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s'
                }}
              >
                Simulation
              </button>
              <button
                onClick={() => triggerModeSelect('webcam')}
                style={{
                  background: status.system_mode === 'webcam' ? 'var(--primary)' : 'none',
                  color: status.system_mode === 'webcam' ? '#000' : 'var(--text-muted)',
                  border: 'none', padding: '5px 12px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s'
                }}
              >
                Webcam
              </button>
              {status.system_mode === 'webcam' && availableCameras.length > 0 && (
                <select
                  value={status.camera_source || "0"}
                  onChange={async (e) => {
                    const newSource = e.target.value;
                    try {
                      const res = await fetch('http://127.0.0.1:8000/api/config');
                      const currentConfig = await res.json();
                      currentConfig.camera.source = newSource;
                      await fetch('http://127.0.0.1:8000/api/config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(currentConfig)
                      });
                      await fetch('http://127.0.0.1:8000/api/pipeline/restart', { method: 'POST' });
                    } catch(err) { console.error('Failed to change camera:', err); }
                  }}
                  style={{
                    background: 'none', border: 'none', color: 'var(--primary)', 
                    fontSize: '0.7rem', fontWeight: 600, outline: 'none', cursor: 'pointer', 
                    maxWidth: '120px', marginLeft: '2px', marginRight: '4px'
                  }}
                  title="Select Camera"
                >
                  {availableCameras.map((cam, idx) => (
                    <option key={cam.deviceId || idx} value={idx.toString()} style={{ background: '#111827', color: '#fff' }}>
                      {cam.label || `Camera ${idx}`}
                    </option>
                  ))}
                </select>
              )}
              <button
                onClick={() => triggerModeSelect('hardware')}
                style={{
                  background: status.system_mode === 'hardware' ? 'var(--primary)' : 'none',
                  color: status.system_mode === 'hardware' ? '#000' : 'var(--text-muted)',
                  border: 'none', padding: '5px 12px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s'
                }}
              >
                Hardware
              </button>
            </div>
          )}

          <div className="header-status-badge">
            <span className={`status-dot ${isBackendConnected ? 'active' : 'danger'}`}></span>
            Backend: {isBackendConnected ? 'Connected' : 'Offline'}
          </div>
          
          <div className="header-status-badge">
            <span className={`status-dot ${status.pipeline_running ? 'active' : 'danger'}`}></span>
            Tracker: {status.pipeline_running ? `${status.fps.toFixed(1)} FPS` : 'Stopped'}
          </div>
          
          <div className="header-status-badge">
            <span className={`status-dot ${status.telemetry.is_connected ? 'active' : 'danger'}`}></span>
            FC Link: {status.telemetry.is_connected ? (status.telemetry.current_mode || 'GUIDED') : 'Disconnected'}
          </div>
        </div>
      </header>

      {/* Main Tabs Selection */}
      <div className="tab-headers" style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '8px' }}>
        <button
          className={`tab-header ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <Activity style={{ display: 'inline', marginRight: '6px', width: '16px' }} /> Dashboard
        </button>
        <button
          className={`tab-header ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          <Sliders style={{ display: 'inline', marginRight: '6px', width: '16px' }} /> Settings
        </button>
        <button
          className={`tab-header ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          <Terminal style={{ display: 'inline', marginRight: '6px', width: '16px' }} /> Training Workshop
        </button>
      </div>

      {/* Warnings Banner */}
      {status.warnings && status.warnings.length > 0 && (
        <div className="warnings-widget">
          <div className="warnings-header">
            <ShieldAlert style={{ width: '18px', height: '18px' }} /> ACTIVE FAILSAFES & ALERTS
          </div>
          {status.warnings.map((w, idx) => (
            <div key={idx} className="warning-line">
              <AlertTriangle style={{ width: '14px', height: '14px', color: 'var(--danger)' }} /> {w}
            </div>
          ))}
        </div>
      )}

      {/* Content Area */}
      {activeTab === 'dashboard' && (
        <div className="dashboard-grid">
          {/* Column 1: Video Feed */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div className="glass-card" style={{ padding: '15px' }}>
              <div className="card-title">
                <Video style={{ color: 'var(--primary)', width: '18px' }} /> Visual Tracking & HUD Feed
              </div>
              <div className="video-container">
                {status.pipeline_running ? (
                  <img
                    src="http://127.0.0.1:8000/api/video_feed"
                    alt="Processed Frame Feed"
                    className="video-feed-img"
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                ) : (
                  <div className="video-placeholder">
                    <Video style={{ width: '48px', height: '48px', opacity: 0.3 }} />
                    <p>Pipeline Offline. Start the tracker to stream video.</p>
                    <button className="btn primary" onClick={triggerPipelineRestart}>
                      Initialize Tracking Pipeline
                    </button>
                  </div>
                )}

                {/* Video HUD badge overlay */}
                <div className="video-hud-overlay">
                  <div className="hud-badge-container">
                    <div className={`hud-badge ${getLockStateClass(status.lock_state)}`}>
                      Lock: {status.lock_state}
                    </div>
                    {status.target_id && (
                      <div className="hud-badge locked">
                        ID: {status.target_id} | Range: {status.target_distance.toFixed(1)}m
                      </div>
                    )}
                  </div>
                </div>

                {/* Floating Bottom HUD Controls */}
                {status.pipeline_running && (
                  <div className="video-hud-controls-bottom">
                    <div className="hud-switches">
                      <label className="switch-label">
                        <input
                          type="checkbox"
                          className="switch-input"
                          checked={status.guidance_active}
                          onChange={() => triggerToggle('guidance')}
                        />
                        <span className="switch-slider"></span>
                        Auto Guidance
                      </label>

                      <label className="switch-label">
                        <input
                          type="checkbox"
                          className="switch-input"
                          checked={status.paused}
                          onChange={() => triggerToggle('pause')}
                        />
                        <span className="switch-slider"></span>
                        Pause Pipeline
                      </label>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <button
                        className="btn"
                        style={{
                          background: status.active_controller === 'FOLLOW' && status.follow_target_enabled ? 'var(--primary)' : 'rgba(59, 130, 246, 0.1)',
                          color: status.active_controller === 'FOLLOW' && status.follow_target_enabled ? '#000' : 'var(--primary)',
                          border: '1px solid var(--primary)',
                          padding: '6px 16px',
                          fontSize: '0.8rem',
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                        onClick={async () => {
                          const isCurrentlyActive = status.active_controller === 'FOLLOW' && status.follow_target_enabled;
                          if (isCurrentlyActive) {
                            await fetch('http://127.0.0.1:8000/api/control/set_follow', {
                              method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false })
                            });
                          } else {
                            await fetch('http://127.0.0.1:8000/api/control/controller', {
                              method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ controller: 'FOLLOW' })
                            });
                            await fetch('http://127.0.0.1:8000/api/control/set_follow', {
                              method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true })
                            });
                          }
                        }}
                      >
                        <Navigation style={{ width: '14px', display: 'inline', marginRight: '6px', verticalAlign: 'middle' }} />
                        FOLLOW
                      </button>

                      <button
                        className="btn"
                        style={{
                          background: status.active_controller === 'DESTROY' && status.follow_target_enabled ? 'var(--danger)' : 'rgba(239, 68, 68, 0.1)',
                          color: status.active_controller === 'DESTROY' && status.follow_target_enabled ? '#000' : 'var(--danger)',
                          border: '1px solid var(--danger)',
                          padding: '6px 16px',
                          fontSize: '0.8rem',
                          fontWeight: 700,
                          cursor: 'pointer'
                        }}
                        onClick={async () => {
                          const isCurrentlyActive = status.active_controller === 'DESTROY' && status.follow_target_enabled;
                          if (isCurrentlyActive) {
                            await fetch('http://127.0.0.1:8000/api/control/set_follow', {
                              method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: false })
                            });
                          } else {
                            await fetch('http://127.0.0.1:8000/api/control/controller', {
                              method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ controller: 'DESTROY' })
                            });
                            await fetch('http://127.0.0.1:8000/api/control/set_follow', {
                              method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true })
                            });
                          }
                        }}
                      >
                        <Crosshair style={{ width: '14px', display: 'inline', marginRight: '6px', verticalAlign: 'middle' }} />
                        DESTROY
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ── Below-Video Status Info Bar ── */}
            <div className="glass-card" style={{ padding: '14px 18px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Lock Status</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 700, color: status.lock_state === 'LOCKED' ? 'var(--success)' : status.lock_state === 'LOST' ? 'var(--danger)' : 'var(--warning)' }}>
                    {status.lock_state || 'ACQUISITION'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Target</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 700, color: status.target_id ? 'var(--primary)' : 'var(--text-muted)' }}>
                    {status.target_id ? `ID ${status.target_id} · ${status.target_distance.toFixed(1)} m` : 'No Target'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Loop FPS</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 700, color: status.fps > 20 ? 'var(--success)' : status.fps > 10 ? 'var(--warning)' : 'var(--danger)' }}>
                    {status.fps.toFixed(1)} FPS
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Control Law</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--primary)' }}>
                    {status.active_controller === 'DESTROY' || status.active_controller === 'PN_GUIDANCE' ? 'Prop. Nav.' : status.active_controller === 'FOLLOW' ? 'Hybrid Follow' : status.active_controller === 'DIRECT_PURSUIT' ? 'Direct Pursuit' : 'Follow Target'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>MAVLink</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: status.telemetry.is_connected ? 'var(--success)' : 'var(--danger)' }}>
                    {status.telemetry.is_connected ? `${(status.system_mode || '').toUpperCase()} · ${status.telemetry.current_mode || '—'}` : 'DISCONNECTED'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Follow Switch</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: status.follow_target_enabled ? 'var(--success)' : 'var(--text-muted)' }}>
                    {status.follow_target_enabled ? 'ENABLED (ACTIVE)' : 'DISABLED (OBS)'}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Vel Cmd (m/s)</span>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: 'var(--primary)' }}>
                    Vx{(status.current_cmd?.vx ?? 0).toFixed(1)} Vy{(status.current_cmd?.vy ?? 0).toFixed(1)} Vz{(status.current_cmd?.vz ?? 0).toFixed(1)}
                  </span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>Auto Guidance</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: status.guidance_active ? 'var(--success)' : 'var(--text-muted)' }}>
                    {status.guidance_active ? '⚡ ACTIVE' : '○ STANDBY'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Column 2: Flight Control & Telemetry */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {/* Flight Command Deck */}
            <div className="glass-card">
              <div className="card-title">
                <Compass style={{ color: 'var(--primary)', width: '18px' }} /> Flight Control Deck
              </div>
              <div className="control-deck">
                <div className="button-grid">
                  <button 
                    className="btn" 
                    style={{ background: status.telemetry.is_armed ? 'var(--success)' : 'rgba(16, 185, 129, 0.2)', color: status.telemetry.is_armed ? '#000' : 'var(--success)', border: '1px solid var(--success)' }}
                    onClick={() => triggerControlAction('arm')}
                  >
                    ARM VEHICLE
                  </button>
                  <button 
                    className="btn" 
                    style={{ background: !status.telemetry.is_armed ? 'var(--danger)' : 'rgba(239, 68, 68, 0.2)', color: !status.telemetry.is_armed ? '#000' : 'var(--danger)', border: '1px solid var(--danger)' }}
                    onClick={() => triggerControlAction('disarm')}
                  >
                    DISARM VEHICLE
                  </button>
                </div>

                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <button
                    className="btn primary"
                    style={{ flexGrow: 1 }}
                    onClick={() => triggerControlAction('takeoff', { altitude: takeoffAlt })}
                  >
                    TAKEOFF
                  </button>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <label style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-muted)' }}>ALT (M):</label>
                    <input
                      type="number"
                      step="0.5"
                      style={{ width: '60px', background: '#111827', color: '#fff', border: '1px solid var(--border-muted)', padding: '5px', borderRadius: '4px' }}
                      value={takeoffAlt}
                      onChange={(e) => setTakeoffAlt(parseFloat(e.target.value))}
                    />
                  </div>
                </div>

                <div className="button-grid">
                  <button className="btn warning" onClick={() => triggerControlAction('land')}>
                    LAND
                  </button>
                  <button className="btn" onClick={() => triggerControlAction('rtl')}>
                    RTL
                  </button>
                </div>

                <button className="btn estop" onClick={() => triggerControlAction('emergency_stop')}>
                  EMERGENCY STOP (HOVER)
                </button>
              </div>
            </div>

            {/* MAVLink Telemetry stats */}
            <div className="glass-card">
              <div className="card-title">
                <Activity style={{ color: 'var(--primary)', width: '18px' }} /> Live Telemetry Stats
              </div>
              <div className="telemetry-grid">
                <div className="telemetry-item">
                  <div className="telemetry-label">Status</div>
                  <div className={`telemetry-value ${status.telemetry.is_connected ? 'active' : 'inactive'}`}>
                    {status.telemetry.is_connected ? 'LINK OK' : 'LINK LOST'}
                  </div>
                </div>

                <div className="telemetry-item">
                  <div className="telemetry-label">Armed</div>
                  <div className={`telemetry-value ${status.telemetry.is_armed ? 'active' : 'inactive'}`}>
                    {status.telemetry.is_armed ? 'ARMED' : 'DISARMED'}
                  </div>
                </div>

                <div className="telemetry-item">
                  <div className="telemetry-label">Battery</div>
                  <div className="telemetry-value">
                    {status.telemetry.battery_voltage || 0.0} V
                  </div>
                  <div className="progress-bar-container">
                    <div 
                      className={`progress-bar ${
                        (status.telemetry.battery_voltage || 0) > 15 ? 'success' : (status.telemetry.battery_voltage || 0) > 14.2 ? 'warning' : 'danger'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, ((status.telemetry.battery_voltage || 11.1) - 10.5) / 6.3 * 100))}%` }}
                    ></div>
                  </div>
                </div>

                <div className="telemetry-item">
                  <div className="telemetry-label">Altitude</div>
                  <div className="telemetry-value">
                    {status.telemetry.altitude || 0.0} m
                  </div>
                </div>

                <div className="telemetry-item">
                  <div className="telemetry-label">GPS Fix</div>
                  <div className="telemetry-value" style={{ fontSize: '0.8rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {status.telemetry.gps_lock || 'NO_GPS'}
                  </div>
                </div>

                <div className="telemetry-item">
                  <div className="telemetry-label">YPR Angles</div>
                  <div className="telemetry-value" style={{ fontSize: '0.8rem' }}>
                    R:{status.telemetry.roll}° P:{status.telemetry.pitch}°
                  </div>
                </div>

                <div className="telemetry-item" style={{ gridColumn: 'span 2' }}>
                  <div className="telemetry-label">NED Local Pos</div>
                  <div className="telemetry-value" style={{ fontSize: '0.85rem' }}>
                    X: {status.telemetry.local_x}m | Y: {status.telemetry.local_y}m | Z: {status.telemetry.local_z}m
                  </div>
                </div>
              </div>
            </div>

            {/* Velocity Commands HUD */}
            <div className="glass-card" style={{ padding: '15px' }}>
              <div className="card-title">
                <Crosshair style={{ color: 'var(--primary)', width: '18px' }} /> Active Guidance Commands
              </div>
              <div className="telemetry-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <div className="telemetry-item">
                  <div className="telemetry-label">Target VX (m/s)</div>
                  <div className="telemetry-value" style={{ color: 'var(--primary)' }}>
                    {status.current_cmd.vx?.toFixed(2) || '0.00'}
                  </div>
                </div>
                <div className="telemetry-item">
                  <div className="telemetry-label">Target VY (m/s)</div>
                  <div className="telemetry-value" style={{ color: 'var(--primary)' }}>
                    {status.current_cmd.vy?.toFixed(2) || '0.00'}
                  </div>
                </div>
                <div className="telemetry-item">
                  <div className="telemetry-label">Target VZ (m/s)</div>
                  <div className="telemetry-value" style={{ color: 'var(--primary)' }}>
                    {status.current_cmd.vz?.toFixed(2) || '0.00'}
                  </div>
                </div>
                <div className="telemetry-item">
                  <div className="telemetry-label">Yaw Rate (rad/s)</div>
                  <div className="telemetry-value" style={{ color: 'var(--primary)' }}>
                    {status.current_cmd.yaw_rate?.toFixed(2) || '0.00'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'settings' && (
        <div className="glass-card">
          <div className="card-title">
            <Sliders style={{ color: 'var(--primary)', width: '18px' }} /> System Parameters & Configurations
          </div>
          {configLoading ? (
            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
              Loading settings.yaml...
            </div>
          ) : config ? (
            <div className="tab-container">
              <div className="settings-grid">
                {/* YOLO Configuration */}
                <h3 style={{ gridColumn: 'span 2', fontSize: '0.9rem', color: 'var(--primary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px', margin: '10px 0 5px 0' }}>YOLO Detection Model</h3>
                <div className="form-group">
                  <label>Model Weights Path</label>
                  <input
                    type="text"
                    className="form-input"
                    value={config.yolo.model_path}
                    onChange={(e) => handleConfigChange('yolo', 'model_path', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Confidence Threshold</label>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    className="form-input"
                    value={config.yolo.confidence_threshold}
                    onChange={(e) => handleConfigChange('yolo', 'confidence_threshold', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Inference Image Size</label>
                  <input
                    type="number"
                    step="32"
                    className="form-input"
                    value={config.yolo.imgsz}
                    onChange={(e) => handleConfigChange('yolo', 'imgsz', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Hardware Device</label>
                  <select
                    className="form-input"
                    value={config.yolo.device}
                    onChange={(e) => handleConfigChange('yolo', 'device', e.target.value)}
                  >
                    <option value="cuda">NVIDIA GPU (CUDA)</option>
                    <option value="cpu">Processor (CPU)</option>
                  </select>
                </div>

                {/* Tracker Configuration */}
                <h3 style={{ gridColumn: 'span 2', fontSize: '0.9rem', color: 'var(--primary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px', margin: '15px 0 5px 0' }}>ByteTrack Association</h3>
                <div className="form-group">
                  <label>Association Track Threshold</label>
                  <input
                    type="number"
                    step="0.05"
                    className="form-input"
                    value={config.tracker.track_threshold}
                    onChange={(e) => handleConfigChange('tracker', 'track_threshold', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Max Optical Flow Fallback Frames</label>
                  <input
                    type="number"
                    className="form-input"
                    value={config.tracker.lk_max_fallback_frames}
                    onChange={(e) => handleConfigChange('tracker', 'lk_max_fallback_frames', e.target.value)}
                  />
                </div>

                {/* Guidance Configurations */}
                <h3 style={{ gridColumn: 'span 2', fontSize: '0.9rem', color: 'var(--primary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px', margin: '15px 0 5px 0' }}>Pursuit Guidance Law</h3>
                <div className="form-group">
                  <label>Proportional Nav Gain (N)</label>
                  <input
                    type="number"
                    step="0.5"
                    className="form-input"
                    value={config.guidance.nav_constant}
                    onChange={(e) => handleConfigChange('guidance', 'nav_constant', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Desired Follow Standoff (m)</label>
                  <input
                    type="number"
                    step="0.5"
                    className="form-input"
                    value={config.guidance.desired_follow_distance}
                    onChange={(e) => handleConfigChange('guidance', 'desired_follow_distance', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Max XY Speed Command (m/s)</label>
                  <input
                    type="number"
                    className="form-input"
                    value={config.guidance.max_speed_xy}
                    onChange={(e) => handleConfigChange('guidance', 'max_speed_xy', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Target Loss Timeout (s)</label>
                  <input
                    type="number"
                    className="form-input"
                    value={config.guidance.target_loss_timeout}
                    onChange={(e) => handleConfigChange('guidance', 'target_loss_timeout', e.target.value)}
                  />
                </div>

                {/* MAVLink / Hardware Connections */}
                <h3 style={{ gridColumn: 'span 2', fontSize: '0.9rem', color: 'var(--primary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px', margin: '15px 0 5px 0' }}>MAVLink Connection (SpeedyBee)</h3>
                <div className="form-group">
                  <label>Link Type</label>
                  <select
                    className="form-input"
                    value={config.mavlink.connection_type === 'udp' ? 'wireless' : 'wired'}
                    onChange={(e) => {
                      if (e.target.value === 'wired') {
                        handleConfigChange('mavlink', 'connection_type', 'serial');
                      } else {
                        handleConfigChange('mavlink', 'connection_type', 'udp');
                        handleConfigChange('mavlink', 'baudrate', 115200);
                      }
                    }}
                  >
                    <option value="wired">Wired (USB / Serial)</option>
                    <option value="wireless">Wireless (UDP / WiFi)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>MAVLink Baudrate</label>
                  <input
                    type="number"
                    className="form-input"
                    value={config.mavlink.baudrate || 115200}
                    onChange={(e) => handleConfigChange('mavlink', 'baudrate', e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Serial COM Port / Dev</label>
                  <input
                    type="text"
                    className="form-input"
                    value={config.mavlink.serial_port}
                    onChange={(e) => handleConfigChange('mavlink', 'serial_port', e.target.value)}
                    disabled={config.mavlink.connection_type !== 'serial'}
                  />
                </div>
                <div className="form-group">
                  <label>UDP Target Address</label>
                  <input
                    type="text"
                    className="form-input"
                    value={config.mavlink.udp_address}
                    onChange={(e) => handleConfigChange('mavlink', 'udp_address', e.target.value)}
                    disabled={config.mavlink.connection_type !== 'udp'}
                  />
                </div>
                <div className="form-group">
                  <label>UDP Port</label>
                  <input
                    type="number"
                    className="form-input"
                    value={config.mavlink.udp_port}
                    onChange={(e) => handleConfigChange('mavlink', 'udp_port', e.target.value)}
                    disabled={config.mavlink.connection_type !== 'udp'}
                  />
                </div>

                {/* Camera / Video Source */}
                <h3 style={{ gridColumn: 'span 2', fontSize: '0.9rem', color: 'var(--primary)', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px', margin: '15px 0 5px 0' }}>Video Acquisition Source</h3>
                <div className="form-group">
                  <label>System Operating Mode</label>
                  <select
                    className="form-input"
                    value={config.system.mode}
                    onChange={(e) => handleConfigChange('system', 'mode', e.target.value)}
                  >
                    <option value="simulation">SITL Simulation (Virtual Target)</option>
                    <option value="webcam">Local Webcam</option>
                    <option value="hardware">Hardware Camera Deployment</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Camera Index / RTSP URL / File path</label>
                  {config.system.mode === 'webcam' ? (
                    <select
                      className="form-input"
                      value={config.camera.source}
                      onChange={(e) => handleConfigChange('camera', 'source', e.target.value)}
                    >
                      {availableCameras.map((camera, index) => (
                        <option key={camera.deviceId} value={index.toString()}>
                          {camera.label || `Camera ${index}`}
                        </option>
                      ))}
                      <option value="custom">Custom URL / File</option>
                    </select>
                  ) : (
                    <input
                      type="text"
                      className="form-input"
                      value={config.camera.source}
                      onChange={(e) => handleConfigChange('camera', 'source', e.target.value)}
                    />
                  )}
                  {config.system.mode === 'webcam' && config.camera.source === 'custom' && (
                    <input
                      type="text"
                      className="form-input"
                      style={{ marginTop: '10px' }}
                      placeholder="Enter Custom Index or URL"
                      onChange={(e) => handleConfigChange('camera', 'source', e.target.value)}
                    />
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '15px' }}>
                <button className="btn" onClick={fetchConfig}>
                  <RotateCcw style={{ width: '16px' }} /> Discard
                </button>
                <button className="btn primary" onClick={saveConfig}>
                  Save Settings & Restart Pipeline
                </button>
              </div>
            </div>
          ) : (
            <div style={{ padding: '20px', color: 'var(--danger)' }}>
              Failed to load settings file from backend.
            </div>
          )}
        </div>
      )}

      {activeTab === 'training' && (
        <div className="glass-card">
          <div className="card-title">
            <Terminal style={{ color: 'var(--primary)', width: '18px' }} /> Preprocessing & Model Training Workshop
          </div>
          
          <div className="tab-container">
            <p style={{ margin: '0 0 10px 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Manage dataset preparation pipelines and launch custom YOLOv8 model training. Click any card below to launch the script on the backend host machine.
            </p>

            <div className="training-deck">
              <div className="task-list">
                <button
                  className={`task-btn ${activeTask === 'preprocess' ? 'active' : ''}`}
                  onClick={() => startTask('preprocess')}
                  disabled={taskStatus === 'running'}
                >
                  <span className="task-title">1. Format Dataset</span>
                  <span className="task-desc">Merge custom COCO datasets, resize to 640x640, and restructure directories.</span>
                </button>

                <button
                  className={`task-btn ${activeTask === 'augment' ? 'active' : ''}`}
                  onClick={() => startTask('augment')}
                  disabled={taskStatus === 'running'}
                >
                  <span className="task-title">2. Data Augmentations</span>
                  <span className="task-desc">Apply weather simulations, motion blur, and sensor noise to images.</span>
                </button>

                <button
                  className={`task-btn ${activeTask === 'enhance' ? 'active' : ''}`}
                  onClick={() => startTask('enhance')}
                  disabled={taskStatus === 'running'}
                >
                  <span className="task-title">3. SAHI Slice Tiling</span>
                  <span className="task-desc">Apply local contrast (CLAHE) and slice high-resolution images into overlapping tiles.</span>
                </button>

                <button
                  className={`task-btn ${activeTask === 'auto_label' ? 'active' : ''}`}
                  onClick={() => startTask('auto_label')}
                  disabled={taskStatus === 'running'}
                >
                  <span className="task-title">4. Auto-Labeling</span>
                  <span className="task-desc">Run pseudo-label generator script on unannotated categories.</span>
                </button>

                <button
                  className={`task-btn ${activeTask === 'train_strict' ? 'active' : ''}`}
                  onClick={() => startTask('train_strict')}
                  disabled={taskStatus === 'running'}
                >
                  <span className="task-title">5. Train Strict YOLO</span>
                  <span className="task-desc">Train model with cosine learning rate scheduler and small-target optimization.</span>
                </button>

                <button
                  className={`task-btn ${activeTask === 'evaluate' ? 'active' : ''}`}
                  onClick={() => startTask('evaluate')}
                  disabled={taskStatus === 'running'}
                >
                  <span className="task-title">6. Evaluate Test Split</span>
                  <span className="task-desc">Analyze false positives, calculate precision/recall metrics, and test against birds/noise.</span>
                </button>
              </div>

              {/* Terminal Logs window */}
              <div className="terminal-card">
                <div className="terminal-header">
                  <div className="terminal-title">
                    TERMINAL: {activeTask ? `${activeTask.toUpperCase()} RUNNING` : 'IDLE'}
                  </div>
                  <div className="terminal-controls">
                    {taskStatus === 'running' && (
                      <button className="btn danger" style={{ padding: '2px 8px', fontSize: '0.7rem' }} onClick={stopActiveTask}>
                        STOP TASK
                      </button>
                    )}
                    <button className="btn" style={{ padding: '2px 8px', fontSize: '0.7rem' }} onClick={clearTerminal}>
                      <Trash2 style={{ width: '12px', height: '12px' }} /> Clear
                    </button>
                    <span className="terminal-dot red"></span>
                    <span className="terminal-dot yellow"></span>
                    <span className="terminal-dot green"></span>
                  </div>
                </div>

                <div className="terminal-body">
                  {logs.length === 0 ? (
                    <div className="terminal-empty">Terminal output is empty. Start a task to display logs.</div>
                  ) : (
                    logs.map((log, idx) => (
                      <div key={idx} className={`terminal-log-line ${log.type}`}>
                        {log.message}
                      </div>
                    ))
                  )}
                  <div ref={terminalEndRef} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
