/**
 * Enlitens AI - Enhanced Monitoring Dashboard
 * Features: Real-time logs, AI chat assistant, JSON viewer, quality tracking
 */

// State
let logs = [];
let stats = {
  docs: 0,
  total: 0,
  errors: 0,
  warnings: 0,
  quality: 0,
  progress: 0
};
let ws = null;
let isPaused = false;
let charts = {};
let latestJSON = null;

// Initialize
async function init() {
  console.log('Initializing Enhanced Dashboard...');
  
  // Setup charts
  setupCharts();
  
  // Connect WebSocket
  connectWebSocket();
  
  // Start polling
  pollStats();
  setInterval(pollStats, 2000);
  
  // Load initial JSON
  refreshJSON();
  
  console.log('Dashboard ready!');
}

// WebSocket connection
function connectWebSocket() {
  if (isPaused) return;
  
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  
  try {
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      updateStatus('✅ Connected');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'log') {
          addLog(data.level || 'INFO', data.message || JSON.stringify(data), data.timestamp);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket data:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      updateStatus('❌ Connection Error');
    };
    
    ws.onclose = () => {
      console.log('WebSocket closed');
      updateStatus('⚠️ Reconnecting...');
      setTimeout(connectWebSocket, 5000);
    };
  } catch (error) {
    console.error('Failed to create WebSocket:', error);
    updateStatus('❌ Connection Failed');
  }
}

// Poll stats API
async function pollStats() {
  try {
    const response = await fetch('/api/stats');
    const data = await response.json();
    
    // Update stats
    stats.docs = data.documents_processed || 0;
    stats.total = data.total_documents || 0;
    stats.errors = data.recent_errors?.length || 0;
    stats.warnings = data.recent_warnings?.length || 0;
    stats.quality = Math.round(data.quality_metrics?.avg_quality_score || 0);
    stats.progress = Math.round(data.progress_percentage || 0);
    
    // Update UI
    document.getElementById('metric-docs').textContent = stats.docs;
    document.getElementById('metric-docs-total').textContent = `of ${stats.total} total`;
    document.getElementById('metric-errors').textContent = stats.errors;
    document.getElementById('metric-errors-status').textContent = stats.errors > 0 ? '⚠️ Issues detected' : '✓ All clear';
    document.getElementById('metric-warnings').textContent = stats.warnings;
    document.getElementById('metric-quality').textContent = stats.quality ? stats.quality + '%' : '--';
    document.getElementById('metric-quality-status').textContent = getQualityLabel(stats.quality);
    
    // Update current document
    if (data.current_document) {
      document.getElementById('current-doc').innerHTML = `
        <strong>${data.current_document}</strong>
        <p style="margin-top:0.5rem;font-size:0.875rem;color:var(--text-secondary);">
          Processing time: ${Math.round(data.time_on_document_seconds || 0)}s<br>
          Progress: ${stats.progress}%
        </p>
      `;
    }
    
    // Update charts
    updateCharts(data);
    
    updateStatus('✅ Connected');
  } catch (error) {
    console.error('Failed to fetch stats:', error);
    updateStatus('❌ API Error');
  }
}

// Setup charts
function setupCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#ffffff' }
      }
    },
    scales: {
      y: {
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      x: {
        ticks: { color: '#ffffff' },
        grid: { color: 'rgba(255,255,255,0.1)' }
      }
    }
  };
  
  // Progress Chart
  charts.progress = new Chart(document.getElementById('progressChart'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Documents Processed',
        data: [],
        borderColor: '#05F2C7',
        backgroundColor: 'rgba(5,242,199,0.1)',
        tension: 0.4
      }]
    },
    options: commonOptions
  });
  
  // Quality Chart
  charts.quality = new Chart(document.getElementById('qualityChart'), {
    type: 'bar',
    data: {
      labels: ['Excellent', 'Good', 'Fair', 'Poor'],
      datasets: [{
        label: 'Quality Distribution',
        data: [0, 0, 0, 0],
        backgroundColor: ['#7CFC00', '#05F2C7', '#FFF700', '#FF4502']
      }]
    },
    options: commonOptions
  });
  
  // Agents Chart
  charts.agents = new Chart(document.getElementById('agentsChart'), {
    type: 'doughnut',
    data: {
      labels: [],
      datasets: [{
        label: 'Agent Status',
        data: [],
        backgroundColor: ['#7CFC00', '#05F2C7', '#FFF700', '#FF4502', '#C105F5']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#ffffff' }
        }
      }
    }
  });
}

// Update charts
function updateCharts(data) {
  const now = new Date().toLocaleTimeString();
  
  // Progress chart
  if (charts.progress.data.labels.length > 20) {
    charts.progress.data.labels.shift();
    charts.progress.data.datasets[0].data.shift();
  }
  charts.progress.data.labels.push(now);
  charts.progress.data.datasets[0].data.push(stats.docs);
  charts.progress.update('none');
  
  // Agents chart
  if (data.agent_status) {
    const completed = Object.values(data.agent_status).filter(s => s === 'completed').length;
    const running = Object.values(data.agent_status).filter(s => s === 'running').length;
    const pending = Object.values(data.agent_status).filter(s => s === 'pending').length;
    
    charts.agents.data.labels = ['Completed', 'Running', 'Pending'];
    charts.agents.data.datasets[0].data = [completed, running, pending];
    charts.agents.update('none');
  }
}

// Add log entry
function addLog(level, message, timestamp) {
  logs.push({ level, message, timestamp: timestamp || new Date().toISOString(), id: Date.now() });
  
  if (logs.length > 500) logs.shift();
  
  updateLogs();
}

// Update logs display
function updateLogs() {
  const container = document.getElementById('logs');
  if (!container) return;
  
  if (logs.length === 0) {
    container.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-muted);">No logs yet...</div>';
    return;
  }
  
  container.innerHTML = logs.slice(-100).reverse().map(log => {
    const time = new Date(log.timestamp).toLocaleTimeString();
    return `
      <div class="log-entry ${log.level.toLowerCase()}">
        <span style="color:var(--text-muted);min-width:80px;">${time}</span>
        <span style="font-weight:600;min-width:80px;color:${getLevelColor(log.level)}">${log.level}</span>
        <span>${escapeHtml(log.message)}</span>
      </div>
    `;
  }).join('');
}

// AI Chat
async function sendMessage() {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;
  
  input.value = '';
  
  // Add user message
  const messagesDiv = document.getElementById('chat-messages');
  messagesDiv.innerHTML += `
    <div class="message user">
      <strong>You</strong>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  
  // Send to AI
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        context: {
          stats: stats,
          logs: logs.slice(-10),
          latestJSON: latestJSON
        }
      })
    });
    
    const data = await response.json();
    
    // Add assistant response
    messagesDiv.innerHTML += `
      <div class="message assistant">
        <strong>Assistant</strong>
        <p>${escapeHtml(data.response || 'Sorry, I could not process that.')}</p>
      </div>
    `;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  } catch (error) {
    console.error('Chat error:', error);
    messagesDiv.innerHTML += `
      <div class="message assistant">
        <strong>Assistant</strong>
        <p style="color:var(--status-error);">Error: Could not reach AI service.</p>
      </div>
    `;
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }
}

// JSON Viewer
async function refreshJSON() {
  try {
    const response = await fetch('/api/knowledge-base');
    const data = await response.json();
    latestJSON = data;
    
    const viewer = document.getElementById('json-viewer');
    viewer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
  } catch (error) {
    console.error('JSON load error:', error);
    document.getElementById('json-viewer').innerHTML = '<p style="color:var(--status-error);">Error loading JSON</p>';
  }
}

function downloadJSON() {
  if (!latestJSON) return;
  const blob = new Blob([JSON.stringify(latestJSON, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'enlitens-output.json';
  a.click();
}

// Utility functions
function switchTab(event, tabId) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(tabId).classList.add('active');
}

function clearLogs() {
  logs = [];
  updateLogs();
}

function exportLogs() {
  const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'enlitens-logs.json';
  a.click();
}

function togglePause() {
  isPaused = !isPaused;
  const btn = document.getElementById('pause-btn');
  btn.textContent = isPaused ? 'Resume' : 'Pause';
  if (isPaused && ws) {
    ws.close();
  } else {
    connectWebSocket();
  }
}

function updateStatus(text) {
  document.getElementById('status').textContent = text;
}

function getQualityLabel(score) {
  if (score >= 90) return '🟢 Excellent';
  if (score >= 75) return '🟢 Good';
  if (score >= 60) return '🟡 Fair';
  if (score > 0) return '🔴 Poor';
  return 'Analyzing...';
}

function getLevelColor(level) {
  const colors = {
    'ERROR': '#f87171',
    'WARN': '#fbbf24',
    'WARNING': '#fbbf24',
    'INFO': '#60a5fa',
    'DEBUG': '#a78bfa'
  };
  return colors[level] || '#60a5fa';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Start
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

