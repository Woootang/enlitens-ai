/**
 * Enlitens AI Monitoring Dashboard - Vibrant Modern Redesign
 * Colorful, interactive dashboard with charts and visualizations
 * 
 * Uses React from global scope (loaded via script tag)
 */

const { useState, useEffect, useRef, useMemo } = React;

const Dashboard = () => {
  // State Management
  const [theme] = useState('dark');
  const [logs, setLogs] = useState([]);
  const [isPaused, setIsPaused] = useState(false);
  const [stats, setStats] = useState({
    documentsProcessed: 0,
    totalDocuments: 345,
    errors: 0,
    warnings: 0,
    qualityScore: 0,
    logsCount: 0
  });
  const [currentDocument, setCurrentDocument] = useState({
    title: 'Initializing...',
    stage: 'idle',
    progress: 0,
    lastUpdate: null
  });
  const [activeView, setActiveView] = useState('overview');
  const [logFilters, setLogFilters] = useState({ INFO: true, WARN: true, ERROR: true });
  
  const eventSourceRef = useRef(null);
  const logsEndRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Initialize WebSocket connection
  useEffect(() => {
    if (isPaused) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const ws = new WebSocket(wsUrl);
    eventSourceRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          handleLogUpdate(data);
        } else if (data.type === 'stats') {
          // Update stats from WebSocket
          if (data.stats) {
            setStats(prev => ({ ...prev, ...data.stats }));
          }
        }
      } catch (error) {
        console.error('Failed to parse WebSocket data:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
      // Attempt to reconnect after 5 seconds
      setTimeout(() => {
        if (!isPaused) {
          setIsPaused(true);
          setTimeout(() => setIsPaused(false), 100);
        }
      }, 5000);
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [isPaused]);

  // Auto-scroll logs
  useEffect(() => {
    if (autoScroll && !isPaused && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll, isPaused]);

  // Poll stats from API
  useEffect(() => {
    const pollStats = async () => {
      try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        if (data) {
          setStats(prev => ({
            ...prev,
            documentsProcessed: data.documentsProcessed || 0,
            errors: data.errors || 0,
            warnings: data.warnings || 0,
            qualityScore: data.qualityScore || 0,
            logsCount: data.totalLogs || 0
          }));
          
          if (data.currentDocument) {
            setCurrentDocument({
              title: data.currentDocument.title || 'Initializing...',
              stage: data.currentDocument.stage || 'idle',
              progress: data.currentDocument.progress || 0,
              lastUpdate: data.currentDocument.lastUpdate || null
            });
          }
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      }
    };

    pollStats();
    const interval = setInterval(pollStats, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleLogUpdate = (data) => {
    const { type, message, timestamp, level } = data;
    
    const logEntry = {
      id: Date.now() + Math.random(),
      timestamp: timestamp || new Date().toISOString(),
      level: level || 'INFO',
      message: message || JSON.stringify(data)
    };

    setLogs(prev => [...prev.slice(-999), logEntry]);
    
    // Update stats based on log level
    if (level === 'ERROR') {
      setStats(prev => ({ ...prev, errors: prev.errors + 1 }));
    } else if (level === 'WARN') {
      setStats(prev => ({ ...prev, warnings: prev.warnings + 1 }));
    }
  };

  const formatTime = (dateString) => {
    if (!dateString) return '--';
    const date = new Date(dateString);
    return date.toLocaleTimeString();
  };

  const filteredLogs = useMemo(() => {
    return logs.filter(log => logFilters[log.level]);
  }, [logs, logFilters]);

  return (
    <div className="dashboard-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">Enlitens AI</h1>
          <p className="sidebar-subtitle">Monitoring Dashboard</p>
        </div>
        <nav className="nav-menu">
          <ul>
            <li className="nav-item">
              <a href="#" className={`nav-link ${activeView === 'overview' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveView('overview'); }}>
                <span className="nav-icon">📊</span>
                <span>Overview</span>
              </a>
            </li>
            <li className="nav-item">
              <a href="#" className={`nav-link ${activeView === 'logs' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveView('logs'); }}>
                <span className="nav-icon">📋</span>
                <span>Live Logs</span>
              </a>
            </li>
            <li className="nav-item">
              <a href="#" className={`nav-link ${activeView === 'pipeline' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveView('pipeline'); }}>
                <span className="nav-icon">🔄</span>
                <span>Agent Pipeline</span>
              </a>
            </li>
            <li className="nav-item">
              <a href="#" className={`nav-link ${activeView === 'quality' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveView('quality'); }}>
                <span className="nav-icon">✅</span>
                <span>Quality</span>
              </a>
            </li>
            <li className="nav-item">
              <a href="#" className={`nav-link ${activeView === 'foreman' ? 'active' : ''}`} onClick={(e) => { e.preventDefault(); setActiveView('foreman'); }}>
                <span className="nav-icon">🤖</span>
                <span>Foreman AI</span>
              </a>
            </li>
          </ul>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Status Hero Bar */}
        <section className="status-hero">
          <div className="status-hero-content">
            <h2 className="status-title">Current Document</h2>
            <div className="status-meta">
              <span className="status-badge">{currentDocument.title}</span>
              <span className="status-badge">Stage: {currentDocument.stage}</span>
              <span className="status-badge">Last Update: {formatTime(currentDocument.lastUpdate)}</span>
            </div>
            <div className="progress-container">
              <div className="progress-label">
                <span>Processing Progress</span>
                <span>{Math.round(currentDocument.progress)}%</span>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${currentDocument.progress}%` }}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card documents">
            <div className="stat-header">
              <div className="stat-icon">📄</div>
            </div>
            <div className="stat-value">{stats.documentsProcessed}</div>
            <div className="stat-label">Documents Processed</div>
            <div className="stat-change">of {stats.totalDocuments} total</div>
          </div>

          <div className="stat-card errors">
            <div className="stat-header">
              <div className="stat-icon">⚠️</div>
            </div>
            <div className="stat-value">{stats.errors}</div>
            <div className="stat-label">Errors</div>
            <div className="stat-change" style={{ color: stats.errors > 0 ? '#f87171' : '#34d399' }}>
              {stats.errors > 0 ? '⚠️ Issues detected' : '✓ All clear'}
            </div>
          </div>

          <div className="stat-card warnings">
            <div className="stat-header">
              <div className="stat-icon">🔔</div>
            </div>
            <div className="stat-value">{stats.warnings}</div>
            <div className="stat-label">Warnings</div>
            <div className="stat-change">monitoring</div>
          </div>

          <div className="stat-card quality">
            <div className="stat-header">
              <div className="stat-icon">⭐</div>
            </div>
            <div className="stat-value">{stats.qualityScore || '--'}%</div>
            <div className="stat-label">Quality Score</div>
            <div className="stat-change">
              {stats.qualityScore >= 90 ? '🟢 Excellent' : 
               stats.qualityScore >= 75 ? '🟢 Good' : 
               stats.qualityScore >= 60 ? '🟡 Fair' : '🔴 Poor'}
            </div>
          </div>
        </div>

        {/* Controls Panel */}
        <div className="controls-panel">
          <button className="btn btn-secondary" onClick={() => setIsPaused(!isPaused)}>
            <span className="btn-icon">{isPaused ? '▶️' : '⏸️'}</span>
            <span>{isPaused ? 'Resume' : 'Pause'}</span>
          </button>
          <button className="btn btn-secondary" onClick={() => setLogs([])}>
            <span className="btn-icon">🗑️</span>
            <span>Clear Logs</span>
          </button>
          <button className="btn btn-primary" onClick={() => {
            const dataStr = JSON.stringify(logs, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'enlitens-logs.json';
            a.click();
          }}>
            <span className="btn-icon">💾</span>
            <span>Export Logs</span>
          </button>
        </div>

        {/* Logs Section */}
        {activeView === 'logs' && (
          <div className="logs-section">
            <div className="logs-header">
              <h3 className="logs-title">Live Processing Logs</h3>
              <div className="logs-filters">
                <button 
                  className={`filter-btn ${logFilters.INFO ? 'active' : ''}`}
                  onClick={() => setLogFilters(prev => ({ ...prev, INFO: !prev.INFO }))}
                >
                  INFO
                </button>
                <button 
                  className={`filter-btn ${logFilters.WARN ? 'active' : ''}`}
                  onClick={() => setLogFilters(prev => ({ ...prev, WARN: !prev.WARN }))}
                >
                  WARN
                </button>
                <button 
                  className={`filter-btn ${logFilters.ERROR ? 'active' : ''}`}
                  onClick={() => setLogFilters(prev => ({ ...prev, ERROR: !prev.ERROR }))}
                >
                  ERROR
                </button>
              </div>
            </div>
            <div className="logs-container">
              {filteredLogs.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                  No logs to display. Waiting for processing to start...
                </div>
              ) : (
                filteredLogs.map(log => (
                  <div key={log.id} className={`log-entry ${log.level.toLowerCase()}`}>
                    <span className="log-timestamp">{formatTime(log.timestamp)}</span>
                    <span className={`log-level ${log.level.toLowerCase()}`}>{log.level}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}

        {/* Overview View */}
        {activeView === 'overview' && (
          <div className="logs-section">
            <div className="logs-header">
              <h3 className="logs-title">System Overview</h3>
            </div>
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
              <p>Real-time processing dashboard with live metrics and visualizations.</p>
              <p style={{ marginTop: '1rem' }}>Select a view from the sidebar to see detailed information.</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

// Render Dashboard
ReactDOM.render(<Dashboard />, document.getElementById('root'));
