// Enlitens AI Enhanced Monitor - Client-Side Application

// Global State
let ws = null;
let isPaused = false;
let logs = [];
let stats = {
    totalLogs: 0,
    errors: 0,
    warnings: 0,
    infos: 0,
    debugs: 0
};
let processingState = null;
let statusUpdateInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Enlitens AI Enhanced Monitor...');
    setupWebSocket();
    setupEventListeners();
    setupViewSwitching();
    initializeWelcomeContent();
    startStatusPolling();
});

// WebSocket Connection
function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        updateConnectionStatus(true);
        showToast('Connected to monitoring server', 'success');
    };

    ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleMessage(message);
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    };

    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        updateConnectionStatus(false);
        showToast('Connection error', 'error');
    };

    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        updateConnectionStatus(false);
        showToast('Disconnected from server', 'warning');

        // Attempt reconnection after 3 seconds
        setTimeout(() => {
            console.log('🔄 Attempting to reconnect...');
            setupWebSocket();
        }, 3000);
    };
}

// Start polling for status updates
function startStatusPolling() {
    // Poll stats endpoint every 2 seconds
    statusUpdateInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/stats');
            if (response.ok) {
                const data = await response.json();
                processingState = data;
                updateProcessingStatus(data);
                updateAgentPipeline(data);
                updateQualityMetrics(data);
            }
        } catch (error) {
            console.error('Failed to fetch stats:', error);
        }
    }, 2000);
}

// Update processing status bar
function updateProcessingStatus(data) {
    const currentDoc = document.getElementById('currentDoc');
    const progressPct = document.getElementById('progressPct');
    const timeOnDoc = document.getElementById('timeOnDoc');
    const lastUpdate = document.getElementById('lastUpdate');
    const docsProcessed = document.getElementById('docsProcessed');

    if (currentDoc) {
        const docName = data.current_document || 'Waiting...';
        currentDoc.textContent = docName.length > 40 ? docName.substring(0, 40) + '...' : docName;
    }

    if (progressPct) {
        progressPct.textContent = `${data.progress_percentage?.toFixed(1) || 0}%`;
    }

    if (timeOnDoc) {
        const seconds = data.time_on_document_seconds;
        if (seconds !== null && seconds !== undefined) {
            timeOnDoc.textContent = formatDuration(seconds);
        } else {
            timeOnDoc.textContent = '--';
        }
    }

    if (lastUpdate) {
        const lastLogSecs = data.last_log_seconds_ago;
        if (lastLogSecs !== null && lastLogSecs !== undefined) {
            lastUpdate.textContent = `${lastLogSecs.toFixed(0)}s ago`;
        } else {
            lastUpdate.textContent = '--';
        }
    }

    if (docsProcessed) {
        docsProcessed.textContent = `${data.documents_processed || 0}/${data.total_documents || 0}`;
    }
}

// Update agent pipeline visualization
function updateAgentPipeline(data) {
    const supervisorStatus = document.getElementById('supervisorStatus');
    const supervisorStage = document.getElementById('supervisorStage');
    const agentList = document.getElementById('agentList');

    if (supervisorStatus) {
        const status = data.current_document ? 'Running' : 'Idle';
        supervisorStatus.textContent = status;
        supervisorStatus.style.color = data.current_document ? '#10b981' : '#6b7280';
    }

    if (supervisorStage) {
        const stage = data.supervisor_stack && data.supervisor_stack.length > 0
            ? data.supervisor_stack[data.supervisor_stack.length - 1]
            : 'None';
        supervisorStage.textContent = stage;
    }

    if (agentList && data.agent_pipeline) {
        if (data.agent_pipeline.length === 0) {
            agentList.innerHTML = '<p class="placeholder">No agents active yet...</p>';
        } else {
            agentList.innerHTML = '';
            data.agent_pipeline.forEach(agentName => {
                const status = data.agent_status[agentName] || 'idle';
                const card = document.createElement('div');
                card.className = 'agent-card';

                const statusColor = {
                    'running': '#f6ad55',
                    'completed': '#10b981',
                    'failed': '#ef4444',
                    'idle': '#6b7280'
                }[status] || '#6b7280';

                const statusIcon = {
                    'running': '⚙️',
                    'completed': '✅',
                    'failed': '❌',
                    'idle': '⏸️'
                }[status] || '•';

                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>${agentName}</strong>
                        <span style="color: ${statusColor};">${statusIcon} ${status}</span>
                    </div>
                `;
                agentList.appendChild(card);
            });
        }
    }
}

// Update quality metrics
function updateQualityMetrics(data) {
    if (data.quality_metrics) {
        const metrics = data.quality_metrics;

        // Update hallucination metrics if on quality view
        const hallucinationMetrics = document.getElementById('hallucinationMetrics');
        if (hallucinationMetrics) {
            const metricRows = hallucinationMetrics.querySelectorAll('.metric-row .metric-value');
            if (metricRows.length >= 3) {
                metricRows[0].textContent = metrics.citation_verified || 0;
                metricRows[1].textContent = metrics.validation_failures || 0;
                metricRows[2].textContent = metrics.empty_fields || 0;
            }
        }

        // Update quality dashboard grid
        updateQualityDashboard(data);
    }

    // Calculate and update quality score
    const qualityScore = calculateQualityScore(data);
    const qualityScoreEl = document.getElementById('qualityScore');
    if (qualityScoreEl) {
        qualityScoreEl.textContent = qualityScore !== '--' ? `${qualityScore}%` : '--';
    }
}

// Update quality dashboard with comprehensive metrics
function updateQualityDashboard(data) {
    const grid = document.getElementById('qualityMetricsGrid');
    if (!grid) return;

    const metrics = data.quality_metrics || {};
    const qualityScore = calculateQualityScore(data);

    // Determine quality level
    let qualityClass, qualityLabel;
    if (qualityScore === '--' || qualityScore >= 90) {
        qualityClass = 'quality-excellent';
        qualityLabel = 'EXCELLENT';
    } else if (qualityScore >= 75) {
        qualityClass = 'quality-good';
        qualityLabel = 'GOOD';
    } else if (qualityScore >= 60) {
        qualityClass = 'quality-fair';
        qualityLabel = 'FAIR';
    } else {
        qualityClass = 'quality-poor';
        qualityLabel = 'POOR';
    }

    const precision = metrics.precision_at_3;
    const recall = metrics.recall_at_3;
    const faithfulness = metrics.faithfulness;
    const hallucinationRate = metrics.hallucination_rate;
    const layerFailures = metrics.layer_failures || [];

    grid.innerHTML = `
        <div class="metric-card">
            <div class="metric-header">
                <h3>Overall Quality</h3>
                <div class="metric-icon">📊</div>
            </div>
            <div class="metric-value-lg">${qualityScore}${qualityScore !== '--' ? '%' : ''}</div>
            <span class="quality-indicator ${qualityClass}">${qualityLabel}</span>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Citations Verified</h3>
                <div class="metric-icon">✅</div>
            </div>
            <div class="metric-value-lg" style="color: #10b981;">${metrics.citation_verified || 0}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Source verification checks</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Precision @ 3</h3>
                <div class="metric-icon">📐</div>
            </div>
            <div class="metric-value-lg" style="color: #10b981;">${formatPercent(precision)}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Supported key findings</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Recall @ 3</h3>
                <div class="metric-icon">📊</div>
            </div>
            <div class="metric-value-lg" style="color: #667eea;">${formatPercent(recall)}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Recovered ground-truth claims</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Faithfulness</h3>
                <div class="metric-icon">🔒</div>
            </div>
            <div class="metric-value-lg" style="color: #2563eb;">${formatPercent(faithfulness)}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Average semantic agreement</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Hallucination Rate</h3>
                <div class="metric-icon">🧠</div>
            </div>
            <div class="metric-value-lg" style="color: #ef4444;">${formatPercent(hallucinationRate)}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Flagged critical findings</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Validation Failures</h3>
                <div class="metric-icon">⚠️</div>
            </div>
            <div class="metric-value-lg" style="color: #ef4444;">${metrics.validation_failures || 0}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Quality check failures</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Empty Fields</h3>
                <div class="metric-icon">📝</div>
            </div>
            <div class="metric-value-lg" style="color: #f6ad55;">${metrics.empty_fields || 0}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Missing data fields</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Layer Failures</h3>
                <div class="metric-icon">🧩</div>
            </div>
            <div class="metric-value-lg" style="color: #f97316;">${layerFailures.length}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">${layerFailures.join(', ') || 'All layers passing'}</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Processing Progress</h3>
                <div class="metric-icon">📈</div>
            </div>
            <div class="metric-value-lg">${data.progress_percentage?.toFixed(0) || 0}%</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">
                ${data.documents_processed || 0} / ${data.total_documents || 0} documents
            </p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Active Agents</h3>
                <div class="metric-icon">🤖</div>
            </div>
            <div class="metric-value-lg" style="color: #667eea;">${data.active_agents?.length || 0}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">
                ${data.active_agents?.join(', ') || 'None active'}
            </p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Recent Errors</h3>
                <div class="metric-icon">❌</div>
            </div>
            <div class="metric-value-lg" style="color: #ef4444;">${data.recent_errors?.length || 0}</div>
            <p style="margin-top: 0.5rem; color: #6b7280;">In current session</p>
        </div>

        <div class="metric-card">
            <div class="metric-header">
                <h3>Processing Time</h3>
                <div class="metric-icon">⏱️</div>
            </div>
            <div class="metric-value-lg" style="color: #667eea;">
                ${formatDuration(data.time_on_document_seconds || 0)}
            </div>
            <p style="margin-top: 0.5rem; color: #6b7280;">Current document</p>
        </div>
    `;
}

// Handle incoming messages
function handleMessage(message) {
    if (message.type === 'log') {
        handleLogMessage(message);
    } else if (message.type === 'foreman_response') {
        handleForemanResponse(message);
    } else if (message.type === 'quality_update') {
        handleQualityUpdate(message);
    }
}

// Handle log messages
function handleLogMessage(logData) {
    if (isPaused) return;

    logs.push(logData);
    stats.totalLogs++;

    // Update stats based on level
    switch (logData.level) {
        case 'ERROR':
        case 'CRITICAL':
            stats.errors++;
            break;
        case 'WARNING':
            stats.warnings++;
            break;
        case 'INFO':
            stats.infos++;
            break;
        case 'DEBUG':
            stats.debugs++;
            break;
    }

    updateStats();
    appendLogToUI(logData);

    // Auto-scroll to bottom
    const container = document.getElementById('logsContainer');
    if (container && container.scrollHeight - container.scrollTop < container.clientHeight + 100) {
        container.scrollTop = container.scrollHeight;
    }
}

// Append log to UI
function appendLogToUI(logData) {
    const container = document.getElementById('logsContainer');
    if (!container) return;

    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${logData.level}`;

    const timestamp = new Date(logData.timestamp).toLocaleTimeString();

    logEntry.innerHTML = `
        <div class="log-header">
            <span class="log-level ${logData.level}">${getLogIcon(logData.level)} ${logData.level}</span>
            <span class="log-time">${timestamp}</span>
        </div>
        <div class="log-message">${escapeHtml(logData.message)}</div>
        ${logData.agent_name ? `<div class="log-meta">Agent: ${logData.agent_name}</div>` : ''}
        ${logData.document_id ? `<div class="log-meta">Document: ${logData.document_id}</div>` : ''}
    `;

    // Remove welcome message if it exists
    const welcome = container.querySelector('.log-entry.welcome');
    if (welcome && stats.totalLogs > 0) {
        welcome.remove();
    }

    container.appendChild(logEntry);

    // Keep only last 500 logs in DOM
    const entries = container.querySelectorAll('.log-entry');
    if (entries.length > 500) {
        entries[0].remove();
    }
}

// Initialize dynamic content in static placeholders
function initializeWelcomeContent() {
    const welcomeTime = document.getElementById('welcomeLogTime');
    if (welcomeTime) {
        welcomeTime.textContent = new Date().toLocaleTimeString();
    }

    const foremanTime = document.getElementById('foremanWelcomeTime');
    if (foremanTime) {
        foremanTime.textContent = new Date().toLocaleTimeString();
    }
}

// Get log level icon
function getLogIcon(level) {
    const icons = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    };
    return icons[level] || '•';
}

// Update connection status
function updateConnectionStatus(connected) {
    const statusIndicator = document.getElementById('connectionStatus');
    const statusText = document.getElementById('statusText');
    const pulse = statusIndicator?.querySelector('.pulse');

    if (connected) {
        if (statusText) statusText.textContent = 'Connected';
        if (pulse) pulse.classList.add('connected');
    } else {
        if (statusText) statusText.textContent = 'Disconnected';
        if (pulse) pulse.classList.remove('connected');
    }
}

// Update stats display
function updateStats() {
    const totalLogsEl = document.getElementById('totalLogs');
    const totalErrorsEl = document.getElementById('totalErrors');
    const totalWarningsEl = document.getElementById('totalWarnings');

    if (totalLogsEl) totalLogsEl.textContent = stats.totalLogs;
    if (totalErrorsEl) totalErrorsEl.textContent = stats.errors;
    if (totalWarningsEl) totalWarningsEl.textContent = stats.warnings;
}

// Calculate quality score
function calculateQualityScore(data) {
    if (!data || !data.total_documents || data.total_documents === 0) return '--';

    const metrics = data.quality_metrics || {};
    const errors = data.recent_errors?.length || 0;
    const warnings = data.recent_warnings?.length || 0;

    // Start with 100
    let score = 100;

    // Deduct for validation failures (5 points each)
    score -= (metrics.validation_failures || 0) * 5;

    // Deduct for errors (10 points each, max 50)
    score -= Math.min(errors * 10, 50);

    // Deduct for warnings (3 points each, max 20)
    score -= Math.min(warnings * 3, 20);

    // Deduct for empty fields (2 points each, max 20)
    score -= Math.min((metrics.empty_fields || 0) * 2, 20);

    // Precision/recall/faithfulness adjustments when available
    const precision = typeof metrics.precision_at_3 === 'number' ? metrics.precision_at_3 : null;
    const recall = typeof metrics.recall_at_3 === 'number' ? metrics.recall_at_3 : null;
    const faithfulness = typeof metrics.faithfulness === 'number' ? metrics.faithfulness : null;
    const hallucinationRate = typeof metrics.hallucination_rate === 'number' ? metrics.hallucination_rate : null;

    if (precision !== null) score += Math.round((precision - 0.75) * 40);
    if (recall !== null) score += Math.round((recall - 0.75) * 30);
    if (faithfulness !== null) score += Math.round((faithfulness - 0.8) * 40);
    if (hallucinationRate !== null) score -= Math.round(hallucinationRate * 60);

    // Bonus for citations verified (1 point each, max 20)
    score += Math.min((metrics.citation_verified || 0), 20);

    return Math.max(0, Math.min(100, Math.round(score)));
}

function formatPercent(value, decimals = 1) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    return `${(value * 100).toFixed(decimals)}%`;
}

// Format duration in seconds to human-readable
function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || isNaN(seconds)) return '--';

    if (seconds < 60) {
        return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
}

// Clear logs
function clearLogs() {
    if (confirm('Clear all logs? This cannot be undone.')) {
        logs = [];
        stats = {
            totalLogs: 0,
            errors: 0,
            warnings: 0,
            infos: 0,
            debugs: 0
        };

        const container = document.getElementById('logsContainer');
        if (container) {
            container.innerHTML = `
                <div class="log-entry welcome">
                    <div class="log-header">
                        <span class="log-level">INFO</span>
                        <span class="log-time">${new Date().toLocaleTimeString()}</span>
                    </div>
                    <div class="log-message">
                        ✨ Logs cleared. Waiting for new messages...
                    </div>
                </div>
            `;
        }

        updateStats();
        showToast('Logs cleared', 'info');
    }
}

// Pause/Resume stream
function pauseStream() {
    isPaused = !isPaused;
    const pauseText = document.getElementById('pauseText');
    if (pauseText) {
        pauseText.textContent = isPaused ? 'Resume' : 'Pause';
    }
    showToast(isPaused ? 'Stream paused' : 'Stream resumed', 'info');
}

// Export logs
function exportLogs() {
    const dataStr = JSON.stringify(logs, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

    const exportFileDefaultName = `enlitens_logs_${new Date().toISOString()}.json`;

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();

    showToast('Logs exported', 'success');
}

// Refresh knowledge base JSON
async function refreshKnowledgeBase() {
    const viewer = document.getElementById('jsonViewer');
    if (!viewer) return;

    viewer.innerHTML = '<p class="placeholder">Loading knowledge base...</p>';

    try {
        const response = await fetch('/api/knowledge-base');
        if (response.ok) {
            const data = await response.json();
            displayJSON(data, viewer);
            showToast('Knowledge base loaded', 'success');
        } else {
            const error = await response.json();
            viewer.innerHTML = `<p class="placeholder" style="color: #ef4444;">Error: ${error.error || 'Failed to load'}</p>`;
            showToast('Failed to load knowledge base', 'error');
        }
    } catch (error) {
        viewer.innerHTML = `<p class="placeholder" style="color: #ef4444;">Error: ${error.message}</p>`;
        showToast('Failed to load knowledge base', 'error');
    }
}

// Display JSON with syntax highlighting
function displayJSON(data, container) {
    const jsonString = JSON.stringify(data, null, 2);
    const highlighted = syntaxHighlightJSON(jsonString);
    container.innerHTML = `<pre class="json-tree">${highlighted}</pre>`;
}

// Syntax highlight JSON
function syntaxHighlightJSON(json) {
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

// Apply filters
function applyFilters() {
    const levelFilter = document.getElementById('levelFilter')?.value || 'all';
    const searchFilter = document.getElementById('searchFilter')?.value.toLowerCase() || '';

    const entries = document.querySelectorAll('.log-entry:not(.welcome)');
    entries.forEach(entry => {
        const level = entry.classList.contains('DEBUG') ? 'DEBUG' :
                     entry.classList.contains('INFO') ? 'INFO' :
                     entry.classList.contains('WARNING') ? 'WARNING' :
                     entry.classList.contains('ERROR') ? 'ERROR' :
                     entry.classList.contains('CRITICAL') ? 'CRITICAL' : '';

        const text = entry.textContent.toLowerCase();

        const levelMatch = levelFilter === 'all' || level === levelFilter;
        const searchMatch = searchFilter === '' || text.includes(searchFilter);

        entry.style.display = (levelMatch && searchMatch) ? 'block' : 'none';
    });
}

// Send message to Foreman AI
function sendMessage() {
    const input = document.getElementById('chatInput');
    if (!input) return;

    const query = input.value.trim();
    if (!query) return;

    // Add user message to chat
    addChatMessage('user', query);

    // Send to server
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'foreman_query',
            query: query
        }));

        // Add thinking message
        addChatMessage('foreman', '🤔 Analyzing your question with AI...', 'thinking');
    } else {
        addChatMessage('foreman', '⚠️ Not connected to server. Please wait for reconnection.', 'error');
    }

    input.value = '';
}

// Handle Foreman response from server
function handleForemanResponse(data) {
    // Remove thinking message
    const messages = document.getElementById('chatMessages');
    if (messages) {
        const thinking = Array.from(messages.children).find(msg =>
            msg.classList.contains('thinking')
        );
        if (thinking) {
            thinking.remove();
        }
    }

    addChatMessage('foreman', data.response);
}

// Add message to chat
function addChatMessage(sender, text, extraClass = '') {
    const messages = document.getElementById('chatMessages');
    if (!messages) return;

    const message = document.createElement('div');
    message.className = `message ${sender}`;
    if (extraClass) {
        message.classList.add(extraClass);
    }

    const avatar = sender === 'foreman' ? '🏗️' : '👤';
    const name = sender === 'foreman' ? 'Foreman AI' : 'You';
    const timestamp = new Date().toLocaleTimeString();

    message.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <strong>${name}</strong>
                <span class="message-time">${timestamp}</span>
            </div>
            <div class="message-text">${escapeHtml(text).replace(/\n/g, '<br>')}</div>
        </div>
    `;

    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
}

// Handle quality updates
function handleQualityUpdate(data) {
    console.log('Quality update received:', data);
    if (processingState) {
        processingState.quality_metrics = { ...processingState.quality_metrics, ...data };
        updateQualityMetrics(processingState);
    }
}

// Setup event listeners
function setupEventListeners() {
    // Chat input enter key
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
}

// Setup view switching
function setupViewSwitching() {
    const navItems = document.querySelectorAll('.nav-item');
    const views = document.querySelectorAll('.view');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Remove active class from all nav items
            navItems.forEach(nav => nav.classList.remove('active'));

            // Add active class to clicked item
            item.classList.add('active');

            // Hide all views
            views.forEach(view => view.classList.remove('active'));

            // Show selected view
            const viewName = item.getAttribute('data-view') + 'View';
            const targetView = document.getElementById(viewName);
            if (targetView) {
                targetView.classList.add('active');
            }
        });
    });
}

// Show toast notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Export functions for inline onclick handlers
window.clearLogs = clearLogs;
window.pauseStream = pauseStream;
window.exportLogs = exportLogs;
window.applyFilters = applyFilters;
window.sendMessage = sendMessage;
window.refreshKnowledgeBase = refreshKnowledgeBase;
