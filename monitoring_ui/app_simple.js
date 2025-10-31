/**
 * Simple Vanilla JS Dashboard - No React dependencies
 * This will definitely work while we debug React issues
 */

(function() {
  'use strict';
  
  // State
  let logs = [];
  let stats = {
    documentsProcessed: 0,
    totalDocuments: 345,
    errors: 0,
    warnings: 0,
    qualityScore: 0
  };
  let currentDocument = {
    title: 'Initializing...',
    stage: 'idle',
    progress: 0,
    lastUpdate: null
  };
  let isPaused = false;
  let ws = null;
  
  // Initialize
  function init() {
    console.log('Dashboard init starting...');
    const root = document.getElementById('root');
    if (!root) {
      console.error('ERROR: root element not found!');
      document.body.innerHTML = '<div style="padding: 2rem; color: red;"><h1>Error: Root element not found</h1><p>Check browser console for details.</p></div>';
      return;
    }
    console.log('Root element found, rendering...');
    render();
    console.log('Connecting WebSocket...');
    connectWebSocket();
    console.log('Starting stats polling...');
    pollStats();
    console.log('Dashboard initialized successfully!');
  }
  
  // Render dashboard
  function render() {
    const root = document.getElementById('root');
    if (!root) {
      console.error('Cannot render: root element not found');
      return;
    }
    console.log('Rendering dashboard...');
    root.innerHTML = `
      <div class="dashboard-container">
        <!-- Sidebar -->
        <aside class="sidebar">
          <div class="sidebar-header">
            <h1 class="sidebar-title">Enlitens AI</h1>
            <p class="sidebar-subtitle">Monitoring Dashboard</p>
          </div>
          <nav class="nav-menu">
            <ul>
              <li class="nav-item">
                <a href="#" class="nav-link active" onclick="return false;">
                  <span class="nav-icon">📊</span>
                  <span>Overview</span>
                </a>
              </li>
              <li class="nav-item">
                <a href="#" class="nav-link" onclick="return false;">
                  <span class="nav-icon">📋</span>
                  <span>Live Logs</span>
                </a>
              </li>
            </ul>
          </nav>
        </aside>
        
        <!-- Main Content -->
        <main class="main-content">
          <!-- Status Hero -->
          <section class="status-hero">
            <div class="status-hero-content">
              <h2 class="status-title">Current Document</h2>
              <div class="status-meta">
                <span class="status-badge" id="doc-title">${currentDocument.title}</span>
                <span class="status-badge">Stage: <span id="doc-stage">${currentDocument.stage}</span></span>
                <span class="status-badge">Last Update: <span id="doc-update">--</span></span>
              </div>
              <div class="progress-container">
                <div class="progress-label">
                  <span>Processing Progress</span>
                  <span id="progress-text">${Math.round(currentDocument.progress)}%</span>
                </div>
                <div class="progress-bar">
                  <div class="progress-fill" id="progress-fill" style="width: ${currentDocument.progress}%"></div>
                </div>
              </div>
            </div>
          </section>
          
          <!-- Stats Grid -->
          <div class="stats-grid">
            <div class="stat-card documents">
              <div class="stat-header">
                <div class="stat-icon">📄</div>
              </div>
              <div class="stat-value" id="stat-docs">${stats.documentsProcessed}</div>
              <div class="stat-label">Documents Processed</div>
              <div class="stat-change">of ${stats.totalDocuments} total</div>
            </div>
            
            <div class="stat-card errors">
              <div class="stat-header">
                <div class="stat-icon">⚠️</div>
              </div>
              <div class="stat-value" id="stat-errors">${stats.errors}</div>
              <div class="stat-label">Errors</div>
              <div class="stat-change" style="color: ${stats.errors > 0 ? '#f87171' : '#34d399'}">
                ${stats.errors > 0 ? '⚠️ Issues detected' : '✓ All clear'}
              </div>
            </div>
            
            <div class="stat-card warnings">
              <div class="stat-header">
                <div class="stat-icon">🔔</div>
              </div>
              <div class="stat-value" id="stat-warnings">${stats.warnings}</div>
              <div class="stat-label">Warnings</div>
              <div class="stat-change">monitoring</div>
            </div>
            
            <div class="stat-card quality">
              <div class="stat-header">
                <div class="stat-icon">⭐</div>
              </div>
              <div class="stat-value" id="stat-quality">${stats.qualityScore || '--'}%</div>
              <div class="stat-label">Quality Score</div>
              <div class="stat-change" id="quality-status">--</div>
            </div>
          </div>
          
          <!-- Controls -->
          <div class="controls-panel">
            <button class="btn btn-secondary" onclick="togglePause()">
              <span class="btn-icon">${isPaused ? '▶️' : '⏸️'}</span>
              <span>${isPaused ? 'Resume' : 'Pause'}</span>
            </button>
            <button class="btn btn-secondary" onclick="clearLogs()">
              <span class="btn-icon">🗑️</span>
              <span>Clear Logs</span>
            </button>
            <button class="btn btn-primary" onclick="exportLogs()">
              <span class="btn-icon">💾</span>
              <span>Export Logs</span>
            </button>
          </div>
          
          <!-- Logs Section -->
          <div class="logs-section">
            <div class="logs-header">
              <h3 class="logs-title">Live Processing Logs</h3>
              <div class="logs-filters">
                <button class="filter-btn active">INFO</button>
                <button class="filter-btn active">WARN</button>
                <button class="filter-btn active">ERROR</button>
              </div>
            </div>
            <div class="logs-container" id="logs-container">
              <div style="text-align: center; padding: 2rem; color: var(--text-muted);">
                Waiting for logs...
              </div>
            </div>
          </div>
        </main>
      </div>
    `;
  }
  
  // Update UI elements
  function updateUI() {
    const docTitle = document.getElementById('doc-title');
    const docStage = document.getElementById('doc-stage');
    const docUpdate = document.getElementById('doc-update');
    const progressText = document.getElementById('progress-text');
    const progressFill = document.getElementById('progress-fill');
    const statDocs = document.getElementById('stat-docs');
    const statErrors = document.getElementById('stat-errors');
    const statWarnings = document.getElementById('stat-warnings');
    const statQuality = document.getElementById('stat-quality');
    const qualityStatus = document.getElementById('quality-status');
    
    if (docTitle) docTitle.textContent = currentDocument.title;
    if (docStage) docStage.textContent = currentDocument.stage;
    if (docUpdate) docUpdate.textContent = formatTime(currentDocument.lastUpdate);
    if (progressText) progressText.textContent = Math.round(currentDocument.progress) + '%';
    if (progressFill) progressFill.style.width = currentDocument.progress + '%';
    if (statDocs) statDocs.textContent = stats.documentsProcessed;
    if (statErrors) statErrors.textContent = stats.errors;
    if (statWarnings) statWarnings.textContent = stats.warnings;
    if (statQuality) statQuality.textContent = (stats.qualityScore || '--') + '%';
    if (qualityStatus) {
      const score = stats.qualityScore || 0;
      qualityStatus.textContent = score >= 90 ? '🟢 Excellent' : 
                                   score >= 75 ? '🟢 Good' : 
                                   score >= 60 ? '🟡 Fair' : '🔴 Poor';
    }
    
    updateLogs();
  }
  
  // Update logs display
  function updateLogs() {
    const container = document.getElementById('logs-container');
    if (!container) return;
    
    if (logs.length === 0) {
      container.innerHTML = '<div style="text-align: center; padding: 2rem; color: var(--text-muted);">No logs to display. Waiting for processing to start...</div>';
      return;
    }
    
    container.innerHTML = logs.slice(-100).map(log => `
      <div class="log-entry ${log.level.toLowerCase()}">
        <span class="log-timestamp">${formatTime(log.timestamp)}</span>
        <span class="log-level ${log.level.toLowerCase()}">${log.level}</span>
        <span class="log-message">${escapeHtml(log.message)}</span>
      </div>
    `).join('');
    
    container.scrollTop = container.scrollHeight;
  }
  
  // Format time
  function formatTime(dateString) {
    if (!dateString) return '--';
    try {
      const date = new Date(dateString);
      return date.toLocaleTimeString();
    } catch {
      return '--';
    }
  }
  
  // Escape HTML
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  
  // Connect WebSocket
  function connectWebSocket() {
    if (isPaused) return;
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        addLog('INFO', 'WebSocket connected successfully');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'log') {
            addLog(data.level || 'INFO', data.message || JSON.stringify(data));
          }
        } catch (error) {
          console.error('Failed to parse WebSocket data:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addLog('ERROR', 'WebSocket connection error');
      };
      
      ws.onclose = () => {
        console.log('WebSocket closed');
        addLog('WARN', 'WebSocket disconnected. Reconnecting...');
        setTimeout(connectWebSocket, 5000);
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      addLog('ERROR', 'Failed to create WebSocket connection');
    }
  }
  
  // Poll stats
  async function pollStats() {
    try {
      const response = await fetch('/api/stats');
      const data = await response.json();
      
      if (data) {
        stats.documentsProcessed = data.documents_processed || 0;
        stats.totalDocuments = data.total_documents || 345;
        stats.errors = data.recent_errors?.length || 0;
        stats.warnings = data.recent_warnings?.length || 0;
        stats.qualityScore = data.quality_metrics?.avg_quality_score || 0;
        
        if (data.current_document) {
          currentDocument.title = data.current_document || 'Initializing...';
        }
        if (data.progress_percentage !== undefined) {
          currentDocument.progress = data.progress_percentage;
        }
        if (data.time_on_document_seconds) {
          currentDocument.lastUpdate = new Date().toISOString();
        }
        
        updateUI();
      }
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
    
    setTimeout(pollStats, 2000);
  }
  
  // Add log
  function addLog(level, message) {
    logs.push({
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      level: level,
      message: message
    });
    
    if (logs.length > 1000) {
      logs = logs.slice(-1000);
    }
    
    if (level === 'ERROR') stats.errors++;
    if (level === 'WARN') stats.warnings++;
    
    updateLogs();
  }
  
  // Toggle pause
  window.togglePause = function() {
    isPaused = !isPaused;
    if (isPaused && ws) {
      ws.close();
    } else {
      connectWebSocket();
    }
    render();
  };
  
  // Clear logs
  window.clearLogs = function() {
    logs = [];
    updateLogs();
  };
  
  // Export logs
  window.exportLogs = function() {
    const dataStr = JSON.stringify(logs, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'enlitens-logs.json';
    a.click();
  };
  
  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

