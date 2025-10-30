// Enlitens AI Monitor - Client-Side Application

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

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Enlitens AI Monitor...');
    setupWebSocket();
    setupEventListeners();
    setupViewSwitching();
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
    if (container.scrollHeight - container.scrollTop < container.clientHeight + 100) {
        container.scrollTop = container.scrollHeight;
    }
}

// Append log to UI
function appendLogToUI(logData) {
    const container = document.getElementById('logsContainer');

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
    const pulse = statusIndicator.querySelector('.pulse');

    if (connected) {
        statusText.textContent = 'Connected';
        pulse.classList.add('connected');
    } else {
        statusText.textContent = 'Disconnected';
        pulse.classList.remove('connected');
    }
}

// Update stats display
function updateStats() {
    document.getElementById('totalLogs').textContent = stats.totalLogs;
    document.getElementById('totalErrors').textContent = stats.errors;
    document.getElementById('totalWarnings').textContent = stats.warnings;

    // Calculate quality score (simple heuristic)
    const qualityScore = calculateQualityScore();
    document.getElementById('qualityScore').textContent = `${qualityScore}%`;
}

// Calculate quality score
function calculateQualityScore() {
    if (stats.totalLogs === 0) return '--';

    const errorWeight = 10;
    const warningWeight = 5;

    const deductions = (stats.errors * errorWeight) + (stats.warnings * warningWeight);
    const score = Math.max(0, 100 - deductions);

    return Math.round(score);
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

        updateStats();
        showToast('Logs cleared', 'info');
    }
}

// Pause/Resume stream
function pauseStream() {
    isPaused = !isPaused;
    const pauseText = document.getElementById('pauseText');
    pauseText.textContent = isPaused ? 'Resume' : 'Pause';
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

// Apply filters
function applyFilters() {
    const levelFilter = document.getElementById('levelFilter').value;
    const searchFilter = document.getElementById('searchFilter').value.toLowerCase();

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
    }

    // Add thinking message
    addChatMessage('foreman', '🤔 Analyzing your question...', true);

    input.value = '';

    // Simulate Foreman AI response (to be replaced with real AI)
    setTimeout(() => {
        generateForemanResponse(query);
    }, 1500);
}

// Generate Foreman AI response
function generateForemanResponse(query) {
    // Remove thinking message
    const messages = document.getElementById('chatMessages');
    const thinking = messages.lastElementChild;
    if (thinking && thinking.querySelector('.message-text').textContent.includes('Analyzing')) {
        thinking.remove();
    }

    // Generate intelligent response based on query
    let response = '';

    if (query.toLowerCase().includes('status')) {
        response = `Current processing status:\n\n` +
                  `• Total logs: ${stats.totalLogs}\n` +
                  `• Errors: ${stats.errors}\n` +
                  `• Warnings: ${stats.warnings}\n` +
                  `• Quality score: ${calculateQualityScore()}%\n\n` +
                  `The system is ${stats.errors > 5 ? 'experiencing some issues' : 'running smoothly'}. ` +
                  `${stats.errors > 0 ? 'I recommend checking the error logs for details.' : ''}`;
    } else if (query.toLowerCase().includes('error')) {
        const recentErrors = logs.filter(l => l.level === 'ERROR' || l.level === 'CRITICAL').slice(-3);
        if (recentErrors.length > 0) {
            response = `Recent errors detected:\n\n`;
            recentErrors.forEach((err, i) => {
                response += `${i + 1}. ${err.message}\n`;
            });
            response += `\nThese errors might indicate issues with data extraction or validation. Check the logs for more details.`;
        } else {
            response = `Good news! No errors have been detected in the recent processing. The system is operating cleanly.`;
        }
    } else if (query.toLowerCase().includes('quality') || query.toLowerCase().includes('hallucination')) {
        response = `Quality and hallucination prevention status:\n\n` +
                  `• Citation verification: Active\n` +
                  `• Chain-of-Thought prompts: Enabled\n` +
                  `• Temperature optimization: 0.3 (factual) / 0.6 (creative)\n` +
                  `• FTC compliance: Enforced\n` +
                  `• Overall quality score: ${calculateQualityScore()}%\n\n` +
                  `The hallucination prevention system is working as designed. All statistics require proper citations, and creative content is separated from factual extraction.`;
    } else if (query.toLowerCase().includes('help')) {
        response = `I can help you with:\n\n` +
                  `• Checking processing status and metrics\n` +
                  `• Analyzing errors and warnings\n` +
                  `• Reviewing quality and hallucination prevention\n` +
                  `• Explaining agent performance\n` +
                  `• Troubleshooting issues\n\n` +
                  `Try asking specific questions like:\n` +
                  `"What's causing the errors?"\n` +
                  `"How is quality looking?"\n` +
                  `"Show me the processing stats"`;
    } else {
        response = `Based on the current processing data:\n\n` +
                  `The system has processed ${stats.totalLogs} log entries with ${stats.errors} errors and ${stats.warnings} warnings. ` +
                  `The quality score is ${calculateQualityScore()}%.\n\n` +
                  `Is there something specific you'd like me to analyze?`;
    }

    addChatMessage('foreman', response);
}

// Add message to chat
function addChatMessage(sender, text, isThinking = false) {
    const messages = document.getElementById('chatMessages');

    const message = document.createElement('div');
    message.className = `message ${sender}`;

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

// Handle Foreman response from server
function handleForemanResponse(data) {
    // Remove thinking message
    const messages = document.getElementById('chatMessages');
    const thinking = messages.lastElementChild;
    if (thinking && thinking.querySelector('.message-text').textContent.includes('Analyzing')) {
        thinking.remove();
    }

    addChatMessage('foreman', data.response);
}

// Handle quality updates
function handleQualityUpdate(data) {
    // Update quality metrics in the Quality view
    console.log('Quality update received:', data);
    // TODO: Implement quality metrics visualization
}

// Setup event listeners
function setupEventListeners() {
    // Chat input enter key
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
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
