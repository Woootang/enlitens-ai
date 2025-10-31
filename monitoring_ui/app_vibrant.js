/**
 * Enlitens AI Command Center (Version 3)
 * ---------------------------------------------------------------
 * Comprehensive monitoring experience featuring:
 *  - Modern modular layout with run control strip and analytics grid
 *  - Chart.js visualisations for throughput, quality and agent success
 *  - GSAP powered micro-interactions with reduced-motion awareness
 *  - Conversational Foreman AI console backed by /api/foreman/query
 *  - Collapsible live log drawer and document inspector utilities
 *
 *  This bundle relies on React, Chart.js and GSAP provided globally
 *  via script tags inside monitoring_ui/index.html.
 */

const {
  useState,
  useEffect,
  useReducer,
  useMemo,
  useRef,
  useCallback,
  useContext
} = React;

const DashboardContext = React.createContext(null);

const defaultStats = {
  documentsProcessed: 0,
  totalDocuments: 0,
  errors: 0,
  warnings: 0,
  qualityScore: 0,
  successRate: 0,
  avgTimePerDoc: 0,
  queueDepth: 0,
  throughputHistory: [],
  currentDocument: {
    title: 'Awaiting pipeline start…',
    stage: 'idle',
    progress: 0,
    lastUpdate: null
  },
  knowledgeBase: {
    processed: 0,
    remaining: 0,
    averageLatency: 0
  },
  systemHealth: {
    gpu: { utilisation: 0, memory: 0 },
    cpu: { load: 0, temperature: 0 },
    memory: { used: 0, total: 0 }
  }
};

const defaultQuality = {
  citationAccuracy: 0,
  validationPassRate: 0,
  hallucinationRate: 0,
  faithfulness: 0,
  factualDrift: 0,
  flaggedFindings: 0
};

const initialState = {
  stats: defaultStats,
  agentPipeline: [],
  agentPerformance: [],
  qualityMetrics: defaultQuality,
  incidents: [],
  documents: [],
  logs: [],
  suggestedActions: [],
  lastUpdated: null
};

function dashboardReducer(state, action) {
  switch (action.type) {
    case 'SET_STATS':
      return {
        ...state,
        stats: {
          ...state.stats,
          ...action.payload,
          currentDocument: {
            ...state.stats.currentDocument,
            ...(action.payload.currentDocument || {})
          },
          systemHealth: {
            ...state.stats.systemHealth,
            ...(action.payload.systemHealth || {})
          },
          knowledgeBase: {
            ...state.stats.knowledgeBase,
            ...(action.payload.knowledgeBase || {})
          }
        },
        lastUpdated: new Date().toISOString()
      };
    case 'SET_PIPELINE':
      return { ...state, agentPipeline: action.payload || [] };
    case 'SET_AGENT_PERFORMANCE':
      return { ...state, agentPerformance: action.payload || [] };
    case 'SET_QUALITY_METRICS':
      return {
        ...state,
        qualityMetrics: { ...state.qualityMetrics, ...(action.payload || {}) }
      };
    case 'SET_INCIDENTS':
      return { ...state, incidents: action.payload || [] };
    case 'SET_DOCUMENTS':
      return { ...state, documents: action.payload || [] };
    case 'SET_SUGGESTED_ACTIONS':
      return { ...state, suggestedActions: action.payload || [] };
    case 'APPEND_LOG': {
      const newLogs = [...state.logs, action.payload];
      return { ...state, logs: newLogs.slice(-2000) };
    }
    case 'BATCH_LOGS':
      return { ...state, logs: action.payload.slice(-2000) };
    default:
      return state;
  }
}

function usePrefersReducedMotion() {
  const [prefers, setPrefers] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const listener = (event) => setPrefers(event.matches);
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', listener);
    } else {
      mediaQuery.addListener(listener);
    }
    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', listener);
      } else {
        mediaQuery.removeListener(listener);
      }
    };
  }, []);

  return prefers;
}

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '—';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${minutes}m ${remaining.toFixed(0)}s`;
}

function formatPercent(value) {
  if (value === undefined || value === null) return '—';
  return `${Math.max(0, Math.min(100, value)).toFixed(1)}%`;
}

function normaliseStatsPayload(data = {}) {
  const documentsProcessed = data.documentsProcessed ?? data.documents_processed ?? data.docs ?? 0;
  const totalDocuments = data.totalDocuments ?? data.total_documents ?? data.total ?? 0;
  const recentErrors = Array.isArray(data.recent_errors) ? data.recent_errors : [];
  const recentWarnings = Array.isArray(data.recent_warnings) ? data.recent_warnings : [];

  const knowledgeBase = data.knowledgeBase || data.kb || {
    processed: documentsProcessed,
    remaining: Math.max(totalDocuments - documentsProcessed, 0),
    averageLatency: data.average_latency || data.avg_time_per_doc || 0
  };

  const currentDocument = data.currentDocument || (data.current_document
    ? {
        title: data.current_document,
        stage: (Array.isArray(data.active_agents) && data.active_agents.length > 0)
          ? `Agent ${data.active_agents[0]}`
          : data.status || (documentsProcessed === 0 ? 'idle' : 'processing'),
        progress: data.progress_percentage || 0,
        elapsedSeconds: data.time_on_document_seconds || 0,
        lastUpdate: data.last_log_seconds_ago
          ? new Date(Date.now() - data.last_log_seconds_ago * 1000).toISOString()
          : null
      }
    : {});

  const systemHealth = data.systemHealth || data.health || data.system_health || defaultStats.systemHealth;
  const qualityMetrics = data.qualityMetrics || data.quality_metrics || {};

  return {
    documentsProcessed,
    totalDocuments,
    errors: data.errors ?? recentErrors.length ?? 0,
    warnings: data.warnings ?? recentWarnings.length ?? 0,
    qualityScore: data.qualityScore
      ?? data.quality
      ?? qualityMetrics.avg_quality_score
      ?? 0,
    successRate: data.successRate ?? data.success ?? 0,
    avgTimePerDoc: data.avgTimePerDoc ?? data.averageDocumentTime ?? data.avg_time_per_doc ?? 0,
    queueDepth: data.queueDepth ?? data.queue_depth ?? data.pending ?? 0,
    throughputHistory: data.throughputHistory ?? data.throughput ?? [],
    currentDocument,
    knowledgeBase,
    systemHealth
  };
}

function normaliseQualityMetrics(metrics = {}) {
  const source = metrics.qualityMetrics || metrics.quality_metrics || metrics;
  if (!source || typeof source !== 'object') return {};
  return {
    citationAccuracy: source.citationAccuracy ?? source.citation_accuracy ?? source.precision_at_3 ?? 0,
    validationPassRate: source.validationPassRate ?? source.validation_pass_rate ?? 0,
    hallucinationRate: source.hallucinationRate ?? source.hallucination_rate ?? 0,
    faithfulness: source.faithfulness ?? 0,
    factualDrift: source.factualDrift ?? source.factual_drift ?? 0,
    flaggedFindings:
      source.flaggedFindings
      ?? source.validation_failures
      ?? (Array.isArray(source.layer_failures) ? source.layer_failures.length : 0)
  };
}

function normaliseAgentPipeline(pipeline) {
  if (!pipeline) return [];
  if (Array.isArray(pipeline)) {
    return pipeline.map((entry) =>
      typeof entry === 'string'
        ? { agent: entry, status: 'running', start: null, end: null, retries: 0 }
        : entry
    );
  }
  if (pipeline.supervisor && Array.isArray(pipeline.supervisor.agents)) {
    return pipeline.supervisor.agents.map((agent) => ({
      agent: agent.name,
      status: agent.status,
      duration: agent.performance?.avg_time,
      retries: agent.performance?.failures || 0
    }));
  }
  return [];
}

function normaliseAgentPerformance(performance) {
  if (!performance) return [];
  if (Array.isArray(performance)) return performance;
  return Object.entries(performance).map(([name, stats]) => ({
    name,
    success: stats.successes ?? stats.executions ?? 0,
    failed: stats.failures ?? 0
  }));
}

function normaliseIncidents(data = {}) {
  if (Array.isArray(data.incidents)) return data.incidents;
  const incidents = [];
  if (Array.isArray(data.recent_errors)) {
    data.recent_errors.forEach((item) => {
      incidents.push({
        id: `error-${item.timestamp || Math.random()}`,
        level: 'ERROR',
        title: item.message,
        timestamp: item.timestamp
      });
    });
  }
  if (Array.isArray(data.recent_warnings)) {
    data.recent_warnings.forEach((item) => {
      incidents.push({
        id: `warn-${item.timestamp || Math.random()}`,
        level: 'WARNING',
        title: item.message,
        timestamp: item.timestamp
      });
    });
  }
  return incidents;
}

// -------------------------------------
// Charts
// -------------------------------------

function useChart(ref, configFactory, deps) {
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !window.Chart) return undefined;
    const chart = new window.Chart(canvas, configFactory());
    return () => chart.destroy();
  }, deps);
}

const ProgressDonut = ({ progress }) => {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const value = Math.max(0, Math.min(100, progress || 0));

  useEffect(() => {
    if (!canvasRef.current || !window.Chart) return undefined;
    const ctx = canvasRef.current.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, canvasRef.current.width, 0);
    gradient.addColorStop(0, '#C105F5');
    gradient.addColorStop(0.5, '#FF6B3B');
    gradient.addColorStop(1, '#05F2C7');

    if (chartRef.current) {
      chartRef.current.data.datasets[0].data = [value, 100 - value];
      chartRef.current.data.datasets[0].backgroundColor = [gradient, 'rgba(255,255,255,0.08)'];
      chartRef.current.update();
      return undefined;
    }

    chartRef.current = new window.Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Complete', 'Remaining'],
        datasets: [
          {
            data: [value, 100 - value],
            backgroundColor: [gradient, 'rgba(255,255,255,0.08)'],
            borderWidth: 0
          }
        ]
      },
      options: {
        cutout: '75%',
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        }
      }
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, [value]);

  return (
    <div className="progress-donut">
      <canvas ref={canvasRef} />
      <span className="progress-value">{value.toFixed(0)}%</span>
    </div>
  );
};

const ThroughputChart = ({ history }) => {
  const canvasRef = useRef(null);
  const labels = history.map((entry) => entry.label || entry.time || '');
  const values = history.map((entry) => entry.value || entry.documents || 0);

  useChart(
    canvasRef,
    () => ({
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Docs / hour',
            data: values,
            fill: true,
            tension: 0.35,
            backgroundColor: 'rgba(193,5,245,0.1)',
            borderColor: '#C105F5',
            borderWidth: 2,
            pointBackgroundColor: '#05F2C7'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            ticks: { color: 'var(--color-text-muted)' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            beginAtZero: true,
            ticks: { color: 'var(--color-text-muted)' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    }),
    [labels.join('|'), values.join('|')]
  );

  return <canvas ref={canvasRef} aria-label="Document throughput trend" />;
};

const AgentPerformanceChart = ({ data }) => {
  const canvasRef = useRef(null);
  const labels = data.map((agent) => agent.name || agent.agent || 'Agent');
  const success = data.map((agent) => agent.success || agent.successCount || 0);
  const failed = data.map((agent) => agent.failed || agent.errorCount || 0);

  useChart(
    canvasRef,
    () => ({
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Success',
            data: success,
            backgroundColor: 'rgba(124,252,0,0.8)'
          },
          {
            label: 'Retries / Failed',
            data: failed,
            backgroundColor: 'rgba(248,113,113,0.8)'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: 'var(--color-text-muted)' }
          }
        },
        scales: {
          x: {
            stacked: true,
            ticks: { color: 'var(--color-text-muted)' },
            grid: { display: false }
          },
          y: {
            stacked: true,
            beginAtZero: true,
            ticks: { color: 'var(--color-text-muted)' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          }
        }
      }
    }),
    [labels.join('|'), success.join('|'), failed.join('|')]
  );

  return <canvas ref={canvasRef} aria-label="Agent performance chart" />;
};

const QualityRadarChart = ({ metrics }) => {
  const canvasRef = useRef(null);
  const values = [
    metrics.citationAccuracy || 0,
    metrics.validationPassRate || 0,
    metrics.faithfulness || 0,
    Math.max(0, 100 - (metrics.hallucinationRate || 0)),
    Math.max(0, 100 - (metrics.factualDrift || 0))
  ];

  useChart(
    canvasRef,
    () => ({
      type: 'radar',
      data: {
        labels: [
          'Citation accuracy',
          'Validation pass rate',
          'Faithfulness',
          'Hallucination control',
          'Factual grounding'
        ],
        datasets: [
          {
            label: 'Quality mix',
            data: values,
            backgroundColor: 'rgba(5,242,199,0.2)',
            borderColor: '#05F2C7',
            pointBackgroundColor: '#C105F5',
            pointRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: { display: false },
            grid: { color: 'rgba(255,255,255,0.1)' },
            angleLines: { color: 'rgba(255,255,255,0.1)' }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    }),
    values
  );

  return <canvas ref={canvasRef} aria-label="Quality radar" />;
};

// -------------------------------------
// UI Components
// -------------------------------------

const ControlBar = () => {
  const { state } = useContext(DashboardContext);
  const { stats } = state;
  const doc = stats.currentDocument || {};
  const prefersReducedMotion = usePrefersReducedMotion();
  const controlRef = useRef(null);

  useEffect(() => {
    if (prefersReducedMotion || !window.gsap) return;
    const ctx = window.gsap.context(() => {
      window.gsap.from('.control-pill', {
        opacity: 0,
        y: 12,
        duration: 0.6,
        stagger: 0.08,
        ease: 'power3.out'
      });
    }, controlRef);
    return () => ctx && ctx.revert();
  }, [prefersReducedMotion, stats.documentsProcessed, stats.errors, stats.warnings]);

  return (
    <section className="control-bar" ref={controlRef}>
      <div className="control-bar__headline">
        <h1 className="title">Enlitens AI Command Center</h1>
        <span className={`status-dot status-${doc.stage || 'idle'}`}> {doc.stage || 'idle'} </span>
      </div>
      <div className="control-bar__pills">
        <article className="control-pill">
          <header>Current document</header>
          <strong>{doc.title || 'Idle queue'}</strong>
          <small>Last update • {doc.lastUpdate ? new Date(doc.lastUpdate).toLocaleTimeString() : '—'}</small>
        </article>
        <article className="control-pill">
          <header>Progress</header>
          <ProgressDonut progress={doc.progress || 0} />
        </article>
        <article className="control-pill">
          <header>Runtime</header>
          <strong>{formatDuration(doc.elapsedSeconds || doc.elapsed)}</strong>
          <small>Average per doc • {formatDuration(stats.avgTimePerDoc)}</small>
        </article>
        <article className="control-pill">
          <header>Knowledge base</header>
          <strong>{stats.knowledgeBase.processed || 0} / {stats.knowledgeBase.processed + (stats.knowledgeBase.remaining || 0)}</strong>
          <small>Remaining • {stats.knowledgeBase.remaining || 0}</small>
        </article>
      </div>
    </section>
  );
};

const OverviewGrid = ({ onSelectDocument }) => {
  const { state } = useContext(DashboardContext);
  const prefersReducedMotion = usePrefersReducedMotion();
  const gridRef = useRef(null);

  useEffect(() => {
    if (prefersReducedMotion || !window.gsap) return;
    const ctx = window.gsap.context(() => {
      window.gsap.from('.metric-card', {
        opacity: 0,
        y: 24,
        duration: 0.7,
        stagger: 0.06,
        ease: 'power3.out'
      });
    }, gridRef);
    return () => ctx && ctx.revert();
  }, [prefersReducedMotion, state.stats.documentsProcessed, state.stats.errors]);

  const successRate = state.stats.successRate || (state.stats.documentsProcessed
    ? ((state.stats.documentsProcessed - state.stats.errors) / Math.max(state.stats.documentsProcessed, 1)) * 100
    : 0);

  const throughputHistory = state.stats.throughputHistory || [];
  const documents = state.documents.slice(0, 6);

  return (
    <section className="overview-grid" ref={gridRef}>
      <div className="metric-card kpi">
        <header>
          <span className="metric-label">Documents processed</span>
          <span className="metric-value">{state.stats.documentsProcessed}</span>
        </header>
        <footer>
          <span className="metric-subtext">Total in corpus • {state.stats.totalDocuments || '—'}</span>
        </footer>
      </div>
      <div className="metric-card kpi">
        <header>
          <span className="metric-label">Success rate</span>
          <span className="metric-value">{formatPercent(successRate)}</span>
        </header>
        <footer>
          <span className="metric-subtext">Errors • {state.stats.errors} | Warnings • {state.stats.warnings}</span>
        </footer>
      </div>
      <div className="metric-card kpi">
        <header>
          <span className="metric-label">Queue depth</span>
          <span className="metric-value">{state.stats.queueDepth || 0}</span>
        </header>
        <footer>
          <span className="metric-subtext">Average latency • {formatDuration(state.stats.knowledgeBase.averageLatency)}</span>
        </footer>
      </div>
      <div className="metric-card kpi">
        <header>
          <span className="metric-label">Quality score</span>
          <span className="metric-value">{formatPercent(state.stats.qualityScore)}</span>
        </header>
        <footer>
          <span className="metric-subtext">Validation pass rate • {formatPercent(state.qualityMetrics.validationPassRate)}</span>
        </footer>
      </div>

      <div className="metric-card span-2">
        <h2>Throughput trend</h2>
        <ThroughputChart history={throughputHistory} />
      </div>

      <div className="metric-card">
        <h2>Knowledge base coverage</h2>
        <div className="coverage-grid">
          <div>
            <span className="coverage-value">{state.stats.knowledgeBase.processed || 0}</span>
            <span className="coverage-label">Processed entries</span>
          </div>
          <div>
            <span className="coverage-value">{state.stats.knowledgeBase.remaining || 0}</span>
            <span className="coverage-label">Remaining</span>
          </div>
          <div>
            <span className="coverage-value">{formatDuration(state.stats.avgTimePerDoc)}</span>
            <span className="coverage-label">Avg time / doc</span>
          </div>
        </div>
      </div>

      <div className="metric-card">
        <h2>System health</h2>
        <ul className="system-health">
          <li>
            <span>GPU utilisation</span>
            <div className="progress-line">
              <span style={{ width: `${Math.min(state.stats.systemHealth.gpu.utilisation || 0, 100)}%` }} />
            </div>
            <strong>{formatPercent(state.stats.systemHealth.gpu.utilisation)}</strong>
          </li>
          <li>
            <span>CPU load</span>
            <div className="progress-line">
              <span style={{ width: `${Math.min(state.stats.systemHealth.cpu.load || 0, 100)}%` }} />
            </div>
            <strong>{formatPercent(state.stats.systemHealth.cpu.load)}</strong>
          </li>
          <li>
            <span>Memory</span>
            <div className="progress-line">
              <span style={{ width: `${Math.min(((state.stats.systemHealth.memory.used || 0) / Math.max(state.stats.systemHealth.memory.total || 1, 1)) * 100, 100)}%` }} />
            </div>
            <strong>
              {(state.stats.systemHealth.memory.used || 0).toFixed(1)} / {(state.stats.systemHealth.memory.total || 0).toFixed(1)} GB
            </strong>
          </li>
        </ul>
      </div>

      <div className="metric-card span-2">
        <h2>Recent documents</h2>
        <table className="documents-table">
          <thead>
            <tr>
              <th>Title</th>
              <th>Stage</th>
              <th>Elapsed</th>
              <th>Validation</th>
            </tr>
          </thead>
          <tbody>
            {documents.length === 0 && (
              <tr>
                <td colSpan="4" className="empty-state">No documents yet</td>
              </tr>
            )}
            {documents.map((doc) => (
              <tr key={doc.id || doc.title} onClick={() => onSelectDocument(doc)}>
                <td>{doc.title}</td>
                <td>
                  <span className={`badge badge-${doc.status || 'pending'}`}>
                    {doc.status || 'pending'}
                  </span>
                </td>
                <td>{formatDuration(doc.elapsedSeconds || doc.duration)}</td>
                <td>{formatPercent(doc.validationScore)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="metric-card">
        <h2>Incidents</h2>
        <ul className="incident-list">
          {state.incidents.length === 0 && (
            <li className="empty-state">Clean run • no incidents</li>
          )}
          {state.incidents.map((incident) => (
            <li key={incident.id || incident.timestamp}>
              <span className={`badge badge-${incident.level || 'info'}`}>{incident.level || 'info'}</span>
              <div>
                <strong>{incident.title || incident.message}</strong>
                <small>{new Date(incident.timestamp || Date.now()).toLocaleTimeString()}</small>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
};

const PipelineView = ({ onSelectDocument }) => {
  const { state } = useContext(DashboardContext);
  const timelineRef = useRef(null);
  const prefersReducedMotion = usePrefersReducedMotion();

  useEffect(() => {
    if (prefersReducedMotion || !window.gsap) return;
    const ctx = window.gsap.context(() => {
      window.gsap.from('.timeline-row', {
        opacity: 0,
        x: -20,
        duration: 0.6,
        stagger: 0.05,
        ease: 'power2.out'
      });
    }, timelineRef);
    return () => ctx && ctx.revert();
  }, [prefersReducedMotion, state.agentPipeline]);

  const pipeline = state.agentPipeline || [];
  const now = Date.now();
  const minStart = pipeline.reduce((acc, step) => {
    const start = new Date(step.startTime || step.start || now).getTime();
    return Math.min(acc, start);
  }, now);
  const maxEnd = pipeline.reduce((acc, step) => {
    const end = new Date(step.endTime || step.end || now).getTime();
    return Math.max(acc, end);
  }, now + 1000);
  const total = Math.max(maxEnd - minStart, 1);

  return (
    <section className="pipeline-view" ref={timelineRef}>
      <header className="section-heading">
        <div>
          <h2>Agent pipeline timeline</h2>
          <p>Stacked view of agent execution windows with retry counts.</p>
        </div>
      </header>
      <div className="timeline">
        {pipeline.length === 0 && <div className="empty-state">No pipeline telemetry yet.</div>}
        {pipeline.map((step) => {
          const start = new Date(step.startTime || step.start || now).getTime();
          const end = new Date(step.endTime || step.end || now).getTime();
          const width = ((end - start) / total) * 100;
          const offset = ((start - minStart) / total) * 100;
          return (
            <article className="timeline-row" key={`${step.agent || step.name}-${start}`}>
              <div className="timeline-meta">
                <span className="agent-name">{step.agent || step.name}</span>
                <span className="agent-stage">{step.stage || step.status}</span>
              </div>
              <div className="timeline-track">
                <span
                  className={`timeline-bar status-${step.status || 'running'}`}
                  style={{ width: `${Math.max(width, 5)}%`, left: `${offset}%` }}
                />
              </div>
              <div className="timeline-stats">
                <span>{formatDuration(step.durationSeconds || step.duration)}</span>
                <span className="retry">Retries • {step.retries || 0}</span>
              </div>
            </article>
          );
        })}
      </div>

      <div className="metric-card span-2">
        <h2>Agent performance breakdown</h2>
        <AgentPerformanceChart data={state.agentPerformance} />
      </div>

      <div className="metric-card span-2">
        <h2>Recent document states</h2>
        <table className="documents-table">
          <thead>
            <tr>
              <th>Document</th>
              <th>Active agent</th>
              <th>Status</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {state.documents.length === 0 && (
              <tr>
                <td colSpan="4" className="empty-state">No documents yet</td>
              </tr>
            )}
            {state.documents.map((doc) => (
              <tr key={doc.id || doc.title} onClick={() => onSelectDocument(doc)}>
                <td>{doc.title}</td>
                <td>{doc.activeAgent || doc.lastAgent || '—'}</td>
                <td><span className={`badge badge-${doc.status || 'pending'}`}>{doc.status || 'pending'}</span></td>
                <td>{formatDuration(doc.elapsedSeconds || doc.duration)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

const QualityView = () => {
  const { state } = useContext(DashboardContext);
  return (
    <section className="quality-view">
      <header className="section-heading">
        <div>
          <h2>Quality intelligence</h2>
          <p>Faithfulness, citation accuracy and validation throughput.</p>
        </div>
      </header>
      <div className="metric-card span-2">
        <QualityRadarChart metrics={state.qualityMetrics} />
      </div>
      <div className="metric-card">
        <h3>Validation outcomes</h3>
        <ul className="quality-list">
          <li>
            <span>Validation pass rate</span>
            <strong>{formatPercent(state.qualityMetrics.validationPassRate)}</strong>
          </li>
          <li>
            <span>Citation accuracy</span>
            <strong>{formatPercent(state.qualityMetrics.citationAccuracy)}</strong>
          </li>
          <li>
            <span>Hallucination rate</span>
            <strong>{formatPercent(state.qualityMetrics.hallucinationRate)}</strong>
          </li>
        </ul>
      </div>
      <div className="metric-card">
        <h3>Flagged findings</h3>
        <p className="quality-note">
          {state.qualityMetrics.flaggedFindings || 0} findings routed to manual review.
        </p>
        <div className="quality-progress">
          <span style={{ width: `${Math.min((state.qualityMetrics.flaggedFindings || 0) * 10, 100)}%` }} />
        </div>
      </div>
    </section>
  );
};

const ForemanConsole = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Foreman online. Ask about pipeline status, triage incidents, or request next actions.'
    }
  ]);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim() || isSending) return;
    const userMessage = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsSending(true);
    setError(null);

    try {
      const response = await fetch('/api/foreman/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content })
      });

      if (!response.ok) {
        throw new Error(`Request failed with ${response.status}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        const reply = data.reply || data.response || data.message || 'Foreman acknowledged the request.';
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: reply, context: data.context || null }
        ]);
      } else if (response.body && response.body.getReader) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let result = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          result += decoder.decode(value, { stream: true });
        }
        result += decoder.decode();
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: result.trim() || 'Foreman completed the request.' }
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: 'Foreman completed the request.' }
        ]);
      }
    } catch (err) {
      console.error('Foreman query failed', err);
      setError('Foreman is offline. Showing fallback guidance.');
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'I am operating in fallback mode. Please verify agent logs and consider re-running validation for the last processed document.'
        }
      ]);
    } finally {
      setIsSending(false);
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const promptSuggestions = [
    'Summarise current pipeline health',
    'Which agents are retrying the most?',
    'Highlight documents needing manual validation',
    'Generate next best actions'
  ];

  return (
    <section className="foreman-console">
      <header className="section-heading">
        <div>
          <h2>Foreman AI supervisor</h2>
          <p>Conversational command deck for triage and strategy.</p>
        </div>
      </header>
      <div className="foreman-body">
        <aside className="foreman-suggestions">
          <h3>Quick prompts</h3>
          <ul>
            {promptSuggestions.map((suggestion) => (
              <li key={suggestion}>
                <button type="button" onClick={() => setInput(suggestion)}>
                  {suggestion}
                </button>
              </li>
            ))}
          </ul>
          <div className="voice-ready">
            <button type="button" aria-label="Voice input coming soon" disabled>
              🎙️
            </button>
            <span>Voice input coming soon</span>
          </div>
        </aside>
        <div className="foreman-chat">
          <div className="messages">
            {messages.map((msg, index) => (
              <article key={index} className={`message message-${msg.role}`}>
                <div className="message-role">{msg.role === 'assistant' ? 'Foreman' : 'You'}</div>
                <p>{msg.content}</p>
                {msg.context && <pre>{JSON.stringify(msg.context, null, 2)}</pre>}
              </article>
            ))}
            {isSending && <div className="message message-assistant loading">Foreman is thinking…</div>}
          </div>
          <footer className="composer">
            <textarea
              ref={inputRef}
              placeholder="Ask the Foreman to audit quality, triage incidents, or plan next actions"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              rows="3"
            />
            <div className="composer-actions">
              {error && <span className="error-text">{error}</span>}
              <button type="button" className="send-button" onClick={sendMessage} disabled={isSending}>
                {isSending ? 'Sending…' : 'Send'}
              </button>
            </div>
          </footer>
        </div>
      </div>
    </section>
  );
};

const LogsDrawer = () => {
  const { state, drawerOpen, setDrawerOpen } = useContext(DashboardContext);
  const [filters, setFilters] = useState({ INFO: true, WARNING: true, ERROR: true });
  const endRef = useRef(null);

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.logs.length, drawerOpen]);

  const toggleFilter = (level) => {
    const key = (level || '').toUpperCase();
    setFilters((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const filteredLogs = state.logs.filter((entry) => {
    const level = (entry.level || 'INFO').toUpperCase();
    return filters[level] ?? true;
  });

  return (
    <aside className={`log-drawer ${drawerOpen ? 'open' : 'collapsed'}`}>
      <button className="log-toggle" onClick={() => setDrawerOpen(!drawerOpen)}>
        {drawerOpen ? 'Close logs' : 'Open logs'}
      </button>
      <div className="log-filters">
        {['INFO', 'WARNING', 'ERROR'].map((level) => (
          <button
            key={level}
            className={`filter-chip ${filters[level] ? 'active' : ''}`}
            onClick={() => toggleFilter(level)}
          >
            {level}
          </button>
        ))}
      </div>
      <div className="log-stream">
        {filteredLogs.map((log) => {
          const level = (log.level || 'INFO').toUpperCase();
          const timestamp = log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '—';
          return (
            <div key={log.id || `${level}-${timestamp}-${log.message}`} className={`log-line level-${level}`}>
              <span className="timestamp">{timestamp}</span>
              <span className="level">{level}</span>
              <span className="message">{log.message}</span>
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </aside>
  );
};

const DocumentInspector = ({ documentData, onClose }) => {
  if (!documentData) return null;
  const metadata = documentData.metadata || {};
  const agents = documentData.agentOutputs || documentData.agents || [];
  return (
    <div className="inspector-overlay" role="dialog" aria-modal="true">
      <div className="inspector">
        <header>
          <h2>{documentData.title || 'Document details'}</h2>
          <button onClick={onClose} aria-label="Close inspector">×</button>
        </header>
        <section className="inspector-section">
          <h3>Metadata</h3>
          <pre>{JSON.stringify(metadata, null, 2)}</pre>
        </section>
        <section className="inspector-section">
          <h3>Agent outputs</h3>
          {agents.length === 0 && <p>No agent outputs recorded.</p>}
          {agents.map((agent) => (
            <article key={agent.agent || agent.name} className="agent-output">
              <header>
                <strong>{agent.agent || agent.name}</strong>
                <span>{formatPercent(agent.validationScore)}</span>
              </header>
              <pre>{JSON.stringify(agent.output || agent.summary, null, 2)}</pre>
            </article>
          ))}
        </section>
        {documentData.validation && (
          <section className="inspector-section">
            <h3>Validation notes</h3>
            <pre>{JSON.stringify(documentData.validation, null, 2)}</pre>
          </section>
        )}
      </div>
    </div>
  );
};

const Navigation = ({ activeView, onChangeView }) => {
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'pipeline', label: 'Agent pipeline' },
    { id: 'quality', label: 'Quality' },
    { id: 'foreman', label: 'Foreman AI' }
  ];

  return (
    <nav className="primary-nav" aria-label="Dashboard view switcher">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={activeView === tab.id ? 'active' : ''}
          onClick={() => onChangeView(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
};

const Sidebar = ({ theme, onToggleTheme }) => {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <h2>Enlitens</h2>
        <p>Neuro-informed knowledge base pipeline</p>
      </div>
      <div className="theme-toggle">
        <span>Theme</span>
        <button onClick={onToggleTheme}>{theme === 'dark' ? 'Switch to light' : 'Switch to dark'}</button>
      </div>
      <div className="sidebar-notes">
        <h3>Suggested actions</h3>
        <ul>
          <li>Validate the latest neuroscience upload</li>
          <li>Review documents flagged for manual QA</li>
          <li>Schedule overnight GPU usage audit</li>
        </ul>
      </div>
    </aside>
  );
};

const DashboardApp = () => {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem('enlitens-theme');
    if (stored) return stored;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });
  const [activeView, setActiveView] = useState('overview');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [selectedDocument, setSelectedDocument] = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('enlitens-theme', theme);
  }, [theme]);

  useEffect(() => {
    let isMounted = true;
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error('Failed to fetch stats');
        const data = await response.json();
        if (!isMounted) return;

        const statsPayload = normaliseStatsPayload(data);
        dispatch({ type: 'SET_STATS', payload: statsPayload });

        const pipelinePayload = normaliseAgentPipeline(data.agentPipeline || data.agent_pipeline);
        dispatch({ type: 'SET_PIPELINE', payload: pipelinePayload });

        const agentPerformancePayload = normaliseAgentPerformance(data.agentPerformance || data.agent_performance);
        dispatch({ type: 'SET_AGENT_PERFORMANCE', payload: agentPerformancePayload });

        const qualityPayload = normaliseQualityMetrics(data.qualityMetrics || data.quality_metrics);
        dispatch({ type: 'SET_QUALITY_METRICS', payload: qualityPayload });

        const incidentsPayload = normaliseIncidents(data);
        dispatch({ type: 'SET_INCIDENTS', payload: incidentsPayload });

        const documentsPayload = data.documents || data.recent_documents || [];
        dispatch({ type: 'SET_DOCUMENTS', payload: documentsPayload });
        if (data.suggestedActions) {
          dispatch({ type: 'SET_SUGGESTED_ACTIONS', payload: data.suggestedActions });
        }
      } catch (error) {
        console.warn('Stat polling failed', error);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

    useEffect(() => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'log') {
            dispatch({
              type: 'APPEND_LOG',
              payload: {
                id: payload.id || `${Date.now()}-${Math.random()}`,
                timestamp: payload.timestamp || new Date().toISOString(),
                level: (payload.level || 'INFO').toUpperCase(),
                message: payload.message || ''
              }
            });
          }
          if (payload.type === 'stats' && payload.stats) {
            dispatch({ type: 'SET_STATS', payload: normaliseStatsPayload(payload.stats) });
          }
          if (payload.type === 'pipeline') {
            dispatch({ type: 'SET_PIPELINE', payload: normaliseAgentPipeline(payload.data) });
          }
          if (payload.type === 'quality') {
            dispatch({ type: 'SET_QUALITY_METRICS', payload: normaliseQualityMetrics(payload.data) });
          }
          if (payload.type === 'agentPerformance') {
            dispatch({ type: 'SET_AGENT_PERFORMANCE', payload: normaliseAgentPerformance(payload.data) });
          }
          if (payload.type === 'documents') {
            dispatch({ type: 'SET_DOCUMENTS', payload: payload.data || [] });
          }
          if (payload.type === 'incidents') {
            dispatch({ type: 'SET_INCIDENTS', payload: payload.data || [] });
          }
        } catch (err) {
          console.error('Failed to process websocket message', err);
        }
      };

      return () => ws.close();
    }, []);

  const contextValue = useMemo(
    () => ({
      state,
      dispatch,
      drawerOpen,
      setDrawerOpen
    }),
    [state, drawerOpen]
  );

  return (
    <DashboardContext.Provider value={contextValue}>
      <div className={`dashboard-shell theme-${theme}`}>
        <Sidebar theme={theme} onToggleTheme={() => setTheme(theme === 'dark' ? 'light' : 'dark')} />
        <main className="main-area">
          <ControlBar />
          <Navigation activeView={activeView} onChangeView={setActiveView} />
          {activeView === 'overview' && <OverviewGrid onSelectDocument={setSelectedDocument} />}
          {activeView === 'pipeline' && <PipelineView onSelectDocument={setSelectedDocument} />}
          {activeView === 'quality' && <QualityView />}
          {activeView === 'foreman' && <ForemanConsole />}
        </main>
        <LogsDrawer />
      </div>
      <DocumentInspector documentData={selectedDocument} onClose={() => setSelectedDocument(null)} />
    </DashboardContext.Provider>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<DashboardApp />);
