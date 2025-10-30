import React, { useState, useEffect, useRef, useMemo } from 'react';

/**
 * Enlitens AI Monitoring Dashboard - Redesign
 *
 * A vibrant, neurodivergent-friendly monitoring interface that displays
 * real-time processing logs, stats, and AI conversation from the document
 * processing pipeline.
 *
 * Features:
 * - Real-time SSE log streaming
 * - Dark/Light theme toggle
 * - Reduced motion accessibility
 * - Filterable log levels (INFO/WARN/ERROR)
 * - Live statistics and progress tracking
 * - Foreman AI chat interface
 * - Export and control functions
 */

// ============================================================================
// Main Dashboard Component
// ============================================================================

const Dashboard = () => {
  // =========== STATE MANAGEMENT ===========

  // Theme and accessibility
  const [theme, setTheme] = useState('dark');
  const [motionReduced, setMotionReduced] = useState(false);

  // Logs and streaming
  const [logs, setLogs] = useState([]);
  const [isPaused, setIsPaused] = useState(false);
  const [logFilters, setLogFilters] = useState({
    INFO: true,
    WARN: true,
    ERROR: true
  });

  // Statistics
  const [stats, setStats] = useState({
    documentsProcessed: 0,
    errors: 0,
    warnings: 0,
    qualityScore: 0
  });

  // Current document tracking
  const [currentDocument, setCurrentDocument] = useState({
    title: 'Initializing...',
    stage: 'idle',
    progress: 0,
    lastUpdate: null
  });

  // Active view tab
  const [activeView, setActiveView] = useState('logs');

  // Foreman AI chat
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      sender: 'foreman',
      text: 'Hello! I\'m Foreman AI. Ask me anything about the processing pipeline.',
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState('');

  // =========== REFS ===========
  const logsEndRef = useRef(null);
  const eventSourceRef = useRef(null);
  const logsContainerRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // =========== EFFECTS ===========

  // Initialize theme from system preference
  useEffect(() => {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('enlitens-theme') || (prefersDark ? 'dark' : 'light');
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  // Check for reduced motion preference
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    setMotionReduced(prefersReducedMotion);
  }, []);

  // SSE Connection for real-time logs
  useEffect(() => {
    if (isPaused) {
      // Close connection when paused
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    // Establish SSE connection
    const eventSource = new EventSource('/stream');
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleLogUpdate(data);
      } catch (error) {
        console.error('Failed to parse SSE data:', error);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
      // Attempt to reconnect after 5 seconds
      eventSource.close();
      setTimeout(() => {
        if (!isPaused) {
          // Trigger reconnection by toggling state
          setIsPaused(true);
          setTimeout(() => setIsPaused(false), 100);
        }
      }, 5000);
    };

    return () => {
      eventSource.close();
    };
  }, [isPaused]);

  // Auto-scroll logs when new entries arrive
  useEffect(() => {
    if (autoScroll && !isPaused && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: motionReduced ? 'auto' : 'smooth' });
    }
  }, [logs, autoScroll, isPaused, motionReduced]);

  // =========== HANDLERS ===========

  /**
   * Process incoming log updates from SSE stream
   */
  const handleLogUpdate = (data) => {
    const { type, message, timestamp, level, stage, progress, metadata } = data;

    // Create log entry
    const logEntry = {
      id: Date.now() + Math.random(),
      level: level || 'INFO',
      message: message || '',
      timestamp: timestamp ? new Date(timestamp) : new Date(),
      stage: stage || null,
      metadata: metadata || {}
    };

    // Add to logs
    setLogs(prevLogs => [...prevLogs, logEntry]);

    // Update stats based on log level
    if (level === 'ERROR') {
      setStats(prev => ({ ...prev, errors: prev.errors + 1 }));
    } else if (level === 'WARN') {
      setStats(prev => ({ ...prev, warnings: prev.warnings + 1 }));
    }

    // Update current document if stage/progress info is present
    if (stage || progress !== undefined) {
      setCurrentDocument(prev => ({
        ...prev,
        stage: stage || prev.stage,
        progress: progress !== undefined ? progress : prev.progress,
        lastUpdate: new Date()
      }));
    }

    // Extract document processing updates
    if (message.includes('Processing document:')) {
      const titleMatch = message.match(/Processing document: (.+)/);
      if (titleMatch) {
        setCurrentDocument(prev => ({
          ...prev,
          title: titleMatch[1],
          stage: 'processing',
          progress: 0
        }));
      }
    }

    // Track document completion
    if (message.includes('completed') || message.includes('finished')) {
      setStats(prev => ({ ...prev, documentsProcessed: prev.documentsProcessed + 1 }));
      setCurrentDocument(prev => ({ ...prev, progress: 100 }));
    }

    // Update quality score if present
    if (metadata?.quality_score) {
      setStats(prev => ({ ...prev, qualityScore: metadata.quality_score }));
    }
  };

  /**
   * Toggle theme between dark and light
   */
  const toggleTheme = () => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('enlitens-theme', newTheme);
  };

  /**
   * Toggle reduced motion preference
   */
  const toggleMotion = () => {
    setMotionReduced(!motionReduced);
  };

  /**
   * Toggle log level filter
   */
  const toggleLogFilter = (level) => {
    setLogFilters(prev => ({
      ...prev,
      [level]: !prev[level]
    }));
  };

  /**
   * Clear all logs
   */
  const clearLogs = () => {
    if (window.confirm('Clear all logs?')) {
      setLogs([]);
      setStats({
        documentsProcessed: 0,
        errors: 0,
        warnings: 0,
        qualityScore: 0
      });
    }
  };

  /**
   * Export logs to JSON file
   */
  const exportLogs = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      stats,
      currentDocument,
      logs: logs.map(log => ({
        level: log.level,
        message: log.message,
        timestamp: log.timestamp.toISOString(),
        stage: log.stage,
        metadata: log.metadata
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `enlitens-logs-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  /**
   * Send message to Foreman AI
   */
  const sendChatMessage = () => {
    if (!chatInput.trim()) return;

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: chatInput,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');

    // Simulate Foreman AI response (in production, this would call an API)
    setTimeout(() => {
      const responses = [
        'I\'m analyzing the current processing pipeline status...',
        'Based on the logs, everything looks good!',
        'Let me check those error messages for you.',
        'The quality score is within expected parameters.',
        'Document processing is progressing smoothly.'
      ];

      const foremanMessage = {
        id: Date.now() + 1,
        sender: 'foreman',
        text: responses[Math.floor(Math.random() * responses.length)],
        timestamp: new Date()
      };

      setChatMessages(prev => [...prev, foremanMessage]);
    }, 1000);
  };

  // =========== COMPUTED VALUES ===========

  /**
   * Filter logs based on selected log levels
   */
  const filteredLogs = useMemo(() => {
    return logs.filter(log => logFilters[log.level]);
  }, [logs, logFilters]);

  /**
   * Format timestamp for display
   */
  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // =========== RENDER ===========

  return (
    <main className={`dashboard ${motionReduced ? 'motion-reduced' : ''}`}>

      {/* STATUS BAR - Hero section with gradient */}
      <StatusBar
        currentDocument={currentDocument}
        theme={theme}
        motionReduced={motionReduced}
      />

      {/* GRID PANELS - Three cards: Views, Stats, Controls */}
      <section className="grid-panels">

        {/* Views Panel - Tabbed navigation */}
        <ViewsPanel
          activeView={activeView}
          onViewChange={setActiveView}
        />

        {/* Stats Panel - Metric tiles */}
        <StatsPanel stats={stats} />

        {/* Controls Panel - Action buttons and toggles */}
        <ControlsPanel
          isPaused={isPaused}
          onTogglePause={() => setIsPaused(!isPaused)}
          onClearLogs={clearLogs}
          onExportLogs={exportLogs}
          theme={theme}
          onToggleTheme={toggleTheme}
          motionReduced={motionReduced}
          onToggleMotion={toggleMotion}
          autoScroll={autoScroll}
          onToggleAutoScroll={() => setAutoScroll(!autoScroll)}
        />

      </section>

      {/* GRID ROW - Timeline chart + Foreman AI */}
      <section className="grid-row">

        {/* Timeline Panel - Charts and visualizations */}
        <TimelinePanel
          logs={filteredLogs}
          stats={stats}
        />

        {/* Foreman Panel - AI chat interface */}
        <ForemanPanel
          messages={chatMessages}
          input={chatInput}
          onInputChange={setChatInput}
          onSendMessage={sendChatMessage}
        />

      </section>

      {/* LOGS PANEL - Full-width log stream */}
      <LogsPanel
        logs={filteredLogs}
        filters={logFilters}
        onToggleFilter={toggleLogFilter}
        logsEndRef={logsEndRef}
        logsContainerRef={logsContainerRef}
        isPaused={isPaused}
      />

    </main>
  );
};

// ============================================================================
// StatusBar Component - Hero section with gradient and progress
// ============================================================================

const StatusBar = ({ currentDocument, theme, motionReduced }) => {
  const { title, stage, progress, lastUpdate } = currentDocument;

  // Determine status chip variant based on stage
  const getStatusChip = () => {
    const stageMap = {
      idle: { label: 'Idle', class: 'chip-outline' },
      initializing: { label: 'Initializing...', class: 'chip-glow' },
      processing: { label: 'Processing', class: 'chip-glow' },
      completed: { label: 'Completed', class: '' }
    };

    return stageMap[stage] || stageMap.idle;
  };

  const statusChip = getStatusChip();

  return (
    <section className="status-bar">
      <div className="status-card">
        <div className="status-header">
          <h1 className="heading-xl">{title}</h1>
          <span className={`chip ${statusChip.class}`}>
            {statusChip.label}
          </span>
        </div>

        <div className="status-meta">
          <span className="label-sm">Progress: {Math.round(progress)}%</span>
          <span className="label-sm">
            Last update: {lastUpdate ? formatTime(lastUpdate) : '--'}
          </span>
          <span className="label-sm">Stage: {stage}</span>
        </div>

        {/* Progress bar with gradient fill */}
        <div className="progress-gradient">
          <div
            className="progress-fill"
            style={{
              width: `${progress}%`,
              transition: motionReduced ? 'none' : 'width 0.5s ease-out'
            }}
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin="0"
            aria-valuemax="100"
          />
        </div>
      </div>
    </section>
  );
};

// ============================================================================
// ViewsPanel Component - Tabbed navigation
// ============================================================================

const ViewsPanel = ({ activeView, onViewChange }) => {
  const views = [
    { id: 'logs', label: 'Live Logs' },
    { id: 'pipeline', label: 'Agent Pipeline' },
    { id: 'quality', label: 'Quality' },
    { id: 'json', label: 'JSON Viewer' },
    { id: 'foreman', label: 'Foreman' },
    { id: 'stats', label: 'Statistics' }
  ];

  return (
    <article className="panel panel-glass">
      <h2 className="panel-title">Views</h2>
      <div className="tabs-container">
        {views.map(view => (
          <button
            key={view.id}
            className={`tab-pill ${activeView === view.id ? 'active' : ''}`}
            onClick={() => onViewChange(view.id)}
            aria-pressed={activeView === view.id}
          >
            {view.label}
          </button>
        ))}
      </div>
      <div className="panel-content">
        <p className="label-sm">
          {views.find(v => v.id === activeView)?.label} view content will appear here.
        </p>
      </div>
    </article>
  );
};

// ============================================================================
// StatsPanel Component - Metric tiles with color coding
// ============================================================================

const StatsPanel = ({ stats }) => {
  const statTiles = [
    {
      label: 'Documents',
      value: stats.documentsProcessed,
      color: 'forest-teal',
      icon: '📄'
    },
    {
      label: 'Errors',
      value: stats.errors,
      color: 'sunset-orange',
      icon: '⚠️'
    },
    {
      label: 'Warnings',
      value: stats.warnings,
      color: 'lemon-glow',
      icon: '⚡'
    },
    {
      label: 'Quality',
      value: `${Math.round(stats.qualityScore * 100)}%`,
      color: 'deep-indigo',
      icon: '✨'
    }
  ];

  return (
    <article className="panel panel-glass">
      <h2 className="panel-title">Quick Stats</h2>
      <div className="stats-tiles">
        {statTiles.map((tile, index) => (
          <div
            key={index}
            className={`stat-tile stat-tile-${tile.color}`}
            role="status"
            aria-label={`${tile.label}: ${tile.value}`}
          >
            <span className="stat-icon" aria-hidden="true">{tile.icon}</span>
            <div className="stat-content">
              <span className="stat-value">{tile.value}</span>
              <span className="stat-label">{tile.label}</span>
            </div>
          </div>
        ))}
      </div>
    </article>
  );
};

// ============================================================================
// ControlsPanel Component - Action buttons and toggles
// ============================================================================

const ControlsPanel = ({
  isPaused,
  onTogglePause,
  onClearLogs,
  onExportLogs,
  theme,
  onToggleTheme,
  motionReduced,
  onToggleMotion,
  autoScroll,
  onToggleAutoScroll
}) => {
  return (
    <article className="panel panel-glass">
      <h2 className="panel-title">Controls</h2>

      {/* Primary action buttons */}
      <div className="controls-actions">
        <button
          className="btn-gradient"
          onClick={onTogglePause}
          aria-pressed={isPaused}
        >
          {isPaused ? '▶️ Resume' : '⏸️ Pause'}
        </button>

        <button
          className="btn-gradient"
          onClick={onClearLogs}
        >
          🗑️ Clear
        </button>

        <button
          className="btn-gradient"
          onClick={onExportLogs}
        >
          💾 Export
        </button>
      </div>

      {/* Toggles and settings */}
      <div className="controls-toggles">
        <label className="toggle-pill">
          <input
            type="checkbox"
            checked={theme === 'dark'}
            onChange={onToggleTheme}
            aria-label="Toggle dark mode"
          />
          <span className="toggle-label">
            {theme === 'dark' ? '🌙 Dark' : '☀️ Light'}
          </span>
        </label>

        <label className="toggle-pill">
          <input
            type="checkbox"
            checked={motionReduced}
            onChange={onToggleMotion}
            aria-label="Toggle reduced motion"
          />
          <span className="toggle-label">
            {motionReduced ? '🚶 Reduced Motion' : '🏃 Motion'}
          </span>
        </label>

        <label className="toggle-pill">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={onToggleAutoScroll}
            aria-label="Toggle auto-scroll"
          />
          <span className="toggle-label">
            {autoScroll ? '📜 Auto-scroll' : '📌 Manual'}
          </span>
        </label>
      </div>
    </article>
  );
};

// ============================================================================
// TimelinePanel Component - Charts and visualizations placeholder
// ============================================================================

const TimelinePanel = ({ logs, stats }) => {
  // Calculate log activity over time for visualization
  const logActivity = useMemo(() => {
    const activity = { INFO: 0, WARN: 0, ERROR: 0 };
    logs.forEach(log => {
      activity[log.level] = (activity[log.level] || 0) + 1;
    });
    return activity;
  }, [logs]);

  return (
    <article className="panel panel-glass timeline-panel">
      <h2 className="panel-title">Processing Timeline</h2>

      <div className="timeline-content">
        {/* Placeholder for chart visualization */}
        <div className="chart-placeholder">
          <p className="label-sm">Timeline chart will be rendered here</p>

          {/* Simple text-based activity display */}
          <div className="activity-summary">
            <div className="activity-bar">
              <div
                className="activity-segment activity-info"
                style={{ width: `${(logActivity.INFO / logs.length * 100) || 0}%` }}
                title={`INFO: ${logActivity.INFO}`}
              />
              <div
                className="activity-segment activity-warn"
                style={{ width: `${(logActivity.WARN / logs.length * 100) || 0}%` }}
                title={`WARN: ${logActivity.WARN}`}
              />
              <div
                className="activity-segment activity-error"
                style={{ width: `${(logActivity.ERROR / logs.length * 100) || 0}%` }}
                title={`ERROR: ${logActivity.ERROR}`}
              />
            </div>

            <div className="activity-legend">
              <span className="legend-item">
                <span className="legend-color legend-info"></span>
                INFO: {logActivity.INFO}
              </span>
              <span className="legend-item">
                <span className="legend-color legend-warn"></span>
                WARN: {logActivity.WARN}
              </span>
              <span className="legend-item">
                <span className="legend-color legend-error"></span>
                ERROR: {logActivity.ERROR}
              </span>
            </div>
          </div>
        </div>
      </div>
    </article>
  );
};

// ============================================================================
// ForemanPanel Component - AI chat interface
// ============================================================================

const ForemanPanel = ({ messages, input, onInputChange, onSendMessage }) => {
  const chatEndRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSendMessage();
    }
  };

  return (
    <article className="panel panel-glass foreman-panel">
      <h2 className="panel-title">🤖 Foreman AI</h2>

      <div className="chat-container">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`chat-bubble chat-bubble-${msg.sender}`}
          >
            <div className="chat-message">{msg.text}</div>
            <div className="chat-timestamp">
              {formatTime(msg.timestamp)}
            </div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-container">
        <input
          type="text"
          className="chat-input"
          placeholder="Ask Foreman about the pipeline..."
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyPress={handleKeyPress}
          aria-label="Chat message input"
        />
        <button
          className="btn-gradient btn-send"
          onClick={onSendMessage}
          disabled={!input.trim()}
          aria-label="Send message"
        >
          📤
        </button>
      </div>
    </article>
  );
};

// ============================================================================
// LogsPanel Component - Filterable log stream
// ============================================================================

const LogsPanel = ({
  logs,
  filters,
  onToggleFilter,
  logsEndRef,
  logsContainerRef,
  isPaused
}) => {
  return (
    <section className="panel panel-glass logs-masonry">
      <div className="logs-header">
        <h2 className="panel-title">Live Processing Logs</h2>

        {/* Log level filters */}
        <div className="logs-filters">
          {['INFO', 'WARN', 'ERROR'].map(level => (
            <label key={level} className="filter-checkbox">
              <input
                type="checkbox"
                checked={filters[level]}
                onChange={() => onToggleFilter(level)}
                aria-label={`Show ${level} logs`}
              />
              <span className={`filter-label filter-${level.toLowerCase()}`}>
                {level}
              </span>
            </label>
          ))}
        </div>

        {isPaused && (
          <span className="chip chip-outline">⏸️ Paused</span>
        )}
      </div>

      <div className="log-stream" ref={logsContainerRef}>
        {logs.length === 0 ? (
          <div className="log-empty">
            <p className="label-sm">No logs to display. Waiting for events...</p>
          </div>
        ) : (
          logs.map(log => (
            <LogEntry key={log.id} log={log} />
          ))
        )}
        <div ref={logsEndRef} />
      </div>
    </section>
  );
};

// ============================================================================
// LogEntry Component - Individual log message
// ============================================================================

const LogEntry = ({ log }) => {
  const getLevelIcon = (level) => {
    const icons = {
      INFO: 'ℹ️',
      WARN: '⚠️',
      ERROR: '🚨'
    };
    return icons[level] || 'ℹ️';
  };

  return (
    <div className={`log-entry log-entry-${log.level.toLowerCase()}`}>
      <span className="log-icon" aria-hidden="true">
        {getLevelIcon(log.level)}
      </span>
      <span className="log-timestamp">{formatTime(log.timestamp)}</span>
      <span className={`log-level log-level-${log.level.toLowerCase()}`}>
        {log.level}
      </span>
      {log.stage && (
        <span className="log-stage">[{log.stage}]</span>
      )}
      <span className="log-message">{log.message}</span>
    </div>
  );
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a Date object to HH:MM:SS string
 */
const formatTime = (date) => {
  if (!date || !(date instanceof Date)) return '--:--:--';

  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

// ============================================================================
// Export
// ============================================================================

export default Dashboard;
