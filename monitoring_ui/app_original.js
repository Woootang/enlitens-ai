const { useState, useEffect, useMemo, useRef } = React;

const QUICK_REPLIES = [
  "Current pipeline health?",
  "Summarize LangFuse traces",
  "Any hallucination alerts today?",
  "Cost outlook for this batch",
  "Show latest latency anomalies",
];

const NAV_ITEMS = [
  { id: "logs", label: "Live Logs", icon: "📋" },
  { id: "quality", label: "Quality & Timeline", icon: "📊" },
  { id: "foreman", label: "Foreman AI", icon: "🏗️" },
];

const LEVEL_CLASS = {
  DEBUG: "debug",
  INFO: "info",
  WARNING: "warning",
  ERROR: "error",
  CRITICAL: "error",
};

function formatTime(ts) {
  if (!ts) return "--";
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return ts;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

const Header = ({ theme, onThemeChange, motionReduced, onMotionChange, connection }) => {
  const connectionLabel =
    connection === "connected" ? "Streaming" : connection === "error" ? "Error" : "Connecting";
  return (
    <header className="dashboard-header">
      <div className="header-left">
        <div className="brand-mark">⚡️</div>
        <div className="brand-title">
          <h1>Enlitens AI Control Deck</h1>
          <span>Ops · Observability · Agents</span>
        </div>
      </div>
      <div className="control-cluster">
        <div className={`connection-indicator ${connection === "connected" ? "connected" : ""}`}>
          <span className="dot" />
          <span>{connectionLabel}</span>
        </div>
        <div className="toggle-group">
          <label className="toggle">
            <input
              type="checkbox"
              checked={theme === "dark"}
              onChange={() => onThemeChange(theme === "dark" ? "light" : "dark")}
            />
            <span>{theme === "dark" ? "🌙 Dark" : "☀️ Light"}</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={motionReduced}
              onChange={() => onMotionChange(!motionReduced)}
            />
            <span>{motionReduced ? "🧘 Reduced Motion" : "⚡ Full Motion"}</span>
          </label>
        </div>
      </div>
    </header>
  );
};

const Sidebar = ({
  currentView,
  onChangeView,
  stats,
  onClearLogs,
  onTogglePause,
  isPaused,
}) => (
  <aside className="sidebar">
    <section className="nav-section">
      <h2>Navigation</h2>
      <div className="nav-menu">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-button ${currentView === item.id ? "active" : ""}`}
            onClick={() => onChangeView(item.id)}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </section>
    <section className="nav-section">
      <h2>Pulse</h2>
      <div className="metric-stack">
        <div className="metric-card">
          <div className="label">Total Logs</div>
          <div className="value">{stats.totalLogs}</div>
        </div>
        <div className="metric-card">
          <div className="label">Errors</div>
          <div className="value" style={{ color: "var(--color-danger)" }}>{stats.errors}</div>
        </div>
        <div className="metric-card">
          <div className="label">Warnings</div>
          <div className="value" style={{ color: "var(--color-warning)" }}>{stats.warnings}</div>
        </div>
        <div className="metric-card">
          <div className="label">Quality Score</div>
          <div className="value" style={{ color: "var(--color-success)" }}>{stats.qualityScore}</div>
        </div>
      </div>
    </section>
    <section className="nav-section">
      <h2>Controls</h2>
      <div className="metric-stack">
        <button className="control-button" onClick={onClearLogs}>
          🧹 Clear Logs
        </button>
        <button className="control-button" onClick={onTogglePause}>
          {isPaused ? "▶️ Resume Stream" : "⏸️ Pause Stream"}
        </button>
      </div>
    </section>
  </aside>
);

const LogsView = ({ logs, levelFilter, onLevelChange, searchTerm, onSearchChange }) => {
  const filteredLogs = useMemo(() => {
    return logs.filter((entry) => {
      if (levelFilter !== "all" && entry.level !== levelFilter) {
        return false;
      }
      if (searchTerm && !entry.message.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [logs, levelFilter, searchTerm]);

  return (
    <div className="main-panel">
      <div className="view-toolbar">
        <h2>Live Processing Stream</h2>
        <div className="filter-group">
          <select value={levelFilter} onChange={(e) => onLevelChange(e.target.value)}>
            <option value="all">All Levels</option>
            <option value="DEBUG">Debug</option>
            <option value="INFO">Info</option>
            <option value="WARNING">Warning</option>
            <option value="ERROR">Error</option>
            <option value="CRITICAL">Critical</option>
          </select>
          <input
            type="text"
            value={searchTerm}
            placeholder="Search logs"
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>
      </div>
      <div className="log-stream">
        {filteredLogs.length === 0 ? (
          <div className="empty-state">Awaiting telemetry…</div>
        ) : (
          filteredLogs.map((entry, idx) => (
            <div key={`${entry.timestamp}-${idx}`} className={`log-entry ${LEVEL_CLASS[entry.level] || "info"}`}>
              <div className="meta">
                <span className="level">{entry.level}</span>
                <span>{formatTime(entry.timestamp)}</span>
              </div>
              <div>{entry.message}</div>
              {entry.document_id && (
                <div className="meta">Document: {entry.document_id}</div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

const GanttTimeline = ({ events, theme }) => {
  const ref = useRef(null);

  useEffect(() => {
    const svgEl = ref.current;
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    if (!events.length) return;

    const parsed = events
      .map((event) => ({
        ...event,
        startDate: new Date(event.start),
        endDate: new Date(event.end),
      }))
      .filter((event) => !Number.isNaN(event.startDate.getTime()) && !Number.isNaN(event.endDate.getTime()))
      .sort((a, b) => a.startDate - b.startDate);

    if (!parsed.length) return;

    const width = svgEl.clientWidth || 720;
    const laneHeight = 28;
    const margin = { top: 24, right: 24, bottom: 40, left: 180 };
    const height = Math.max(parsed.length * laneHeight + margin.top + margin.bottom, 220);

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const xScale = d3
      .scaleTime()
      .domain([d3.min(parsed, (d) => d.startDate), d3.max(parsed, (d) => d.endDate)])
      .range([0, innerWidth])
      .nice();

    const yDomain = Array.from(new Set(parsed.map((d) => `${d.agent} · ${d.stage}`)));
    const yScale = d3.scaleBand().domain(yDomain).range([0, innerHeight]).padding(0.2);

    const chart = svg.append("g").attr("transform", `translate(${margin.left}, ${margin.top})`);

    chart
      .append("g")
      .attr("transform", `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSizeOuter(0))
      .call((g) => g.selectAll("path").attr("stroke", "var(--color-border)"))
      .call((g) => g.selectAll("line").attr("stroke", "var(--color-gridline)"));

    chart
      .append("g")
      .call(d3.axisLeft(yScale).tickSize(0))
      .selectAll("text")
      .attr("fill", "var(--color-text-muted)")
      .style("font-size", "0.7rem");

    chart
      .selectAll("rect")
      .data(parsed)
      .enter()
      .append("rect")
      .attr("x", (d) => xScale(d.startDate))
      .attr("y", (d) => yScale(`${d.agent} · ${d.stage}`))
      .attr("width", (d) => Math.max(2, xScale(d.endDate) - xScale(d.startDate)))
      .attr("height", yScale.bandwidth())
      .attr("rx", 6)
      .attr("fill", (d) => {
        if (d.status === "error") return "var(--color-danger)";
        if (d.status === "warning") return "var(--color-warning)";
        return "var(--color-primary)";
      })
      .attr("opacity", 0.85);

    chart
      .selectAll("text.duration")
      .data(parsed)
      .enter()
      .append("text")
      .attr("class", "duration")
      .attr("x", (d) => xScale(d.endDate) + 6)
      .attr("y", (d) => (yScale(`${d.agent} · ${d.stage}`) || 0) + yScale.bandwidth() / 1.4)
      .text((d) => `${((d.endDate - d.startDate) / 1000).toFixed(1)}s`)
      .attr("fill", "var(--color-text-muted)")
      .style("font-size", "0.65rem");
  }, [events, theme]);

  return <svg className="gantt-chart" ref={ref} />;
};

const CitationGraph = ({ data }) => {
  const ref = useRef(null);

  useEffect(() => {
    const svgEl = ref.current;
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    const nodes = (data.nodes || []).map((node) => ({ ...node }));
    const links = (data.edges || []).map((edge) => ({ ...edge }));
    if (!nodes.length) return;

    const width = svgEl.clientWidth || 720;
    const height = svgEl.clientHeight || 320;
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const colorByType = {
      document: "var(--color-primary)",
      concept: "var(--color-success)",
      finding: "var(--color-accent)",
    };

    const simulation = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(120).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-240))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(48));

    const link = svg
      .append("g")
      .attr("stroke", "var(--color-border)")
      .attr("stroke-width", 1.2)
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke-opacity", 0.6);

    const node = svg
      .append("g")
      .selectAll("g")
      .data(nodes)
      .enter()
      .append("g")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    node
      .append("circle")
      .attr("r", (d) => (d.type === "document" ? 16 : 12))
      .attr("fill", (d) => colorByType[d.type] || "var(--color-primary)")
      .attr("fill-opacity", 0.85);

    node
      .append("text")
      .text((d) => d.label || d.id)
      .attr("x", 20)
      .attr("y", 5)
      .attr("fill", "var(--color-text)")
      .style("font-size", "0.65rem");

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
    });

    return () => simulation.stop();
  }, [data]);

  return <svg className="citation-network" ref={ref} />;
};

const QualityView = ({ qualityLog, timeline, citationGraph, ragEvents, systemStats, theme }) => {
  const latestQuality = qualityLog.length ? qualityLog[qualityLog.length - 1] : null;
  const latestRag = ragEvents.length ? ragEvents[ragEvents.length - 1] : null;
  const qualityScore = latestQuality?.metrics?.quality_score ?? latestQuality?.metrics?.overall_score;
  const synthesisScore = latestQuality?.metrics?.synthesis_quality;
  const extractionScore = latestQuality?.metrics?.extraction_quality;
  const ragScore = latestRag?.score;

  return (
    <div className="main-panel">
      <div className="view-toolbar">
        <h2>Quality Metrics & RAG Oversight</h2>
      </div>
      <div className="metrics-grid">
        <div className="metrics-card">
          <div className="heading">Document Quality</div>
          <div className="stat">{qualityScore ? qualityScore.toFixed(2) : "--"}</div>
          <p>Latest aggregate score across extraction, entities, and synthesis.</p>
        </div>
        <div className="metrics-card">
          <div className="heading">Synthesis Confidence</div>
          <div className="stat">{synthesisScore ? synthesisScore.toFixed(2) : "--"}</div>
          <p>LangFuse-aligned trace quality from the last span.</p>
        </div>
        <div className="metrics-card">
          <div className="heading">Extraction Fidelity</div>
          <div className="stat">{extractionScore ? extractionScore.toFixed(2) : "--"}</div>
          <p>Docling + Marker extraction quality indicator.</p>
        </div>
        <div className="metrics-card">
          <div className="heading">RAG Grounding</div>
          <div className="stat">{ragScore ? ragScore.toFixed(2) : "--"}</div>
          <p>Phoenix RAG precision score from most recent synthesis.</p>
        </div>
        <div className="metrics-card">
          <div className="heading">CPU Utilization</div>
          <div className="stat">
            {systemStats?.cpu_percent !== undefined ? `${systemStats.cpu_percent.toFixed(0)}%` : "--"}
          </div>
          <p>Realtime system load from the metrics collector.</p>
        </div>
        <div className="metrics-card">
          <div className="heading">GPU Memory</div>
          <div className="stat">
            {systemStats?.gpu_memory_used_gb !== undefined
              ? `${systemStats.gpu_memory_used_gb.toFixed(2)} GB`
              : "--"}
          </div>
          <p>NVML-reported usage on the primary accelerator.</p>
        </div>
      </div>

      <section className="gantt-wrapper">
        <div className="section-heading">
          <h3>Agent Gantt Timeline</h3>
          <span>{timeline.length ? `${timeline.length} spans` : "Idle"}</span>
        </div>
        {timeline.length ? (
          <GanttTimeline events={timeline.slice(-60)} theme={theme} />
        ) : (
          <div className="empty-state">Timeline will populate once the pipeline begins processing.</div>
        )}
      </section>

      <section className="citation-graph">
        <div className="section-heading">
          <h3>Citation Network Graph</h3>
          <span>
            {citationGraph.nodes?.length ? `${citationGraph.nodes.length} nodes` : "Waiting for synthesis"}
          </span>
        </div>
        {citationGraph.nodes?.length ? (
          <CitationGraph data={citationGraph} />
        ) : (
          <div className="empty-state">Synthesis citations will appear after the first document completes.</div>
        )}
      </section>
    </div>
  );
};

const ForemanView = ({ messages, onSend, alerts }) => {
  const [draft, setDraft] = useState("");
  const streamRef = useRef(null);

  useEffect(() => {
    const el = streamRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!draft.trim()) return;
    onSend(draft.trim());
    setDraft("");
  };

  const latestAlerts = alerts.slice(-4).reverse();

  return (
    <div className="main-panel">
      <div className="view-toolbar">
        <h2>Foreman Ops Chat</h2>
      </div>
      <div className="foreman-console">
        <div className="chat-panel">
          <div className="chat-stream" ref={streamRef}>
            {messages.length ? (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`chat-message ${message.role} ${message.streaming ? "streaming" : ""}`}
                >
                  {message.content || "…"}
                </div>
              ))
            ) : (
              <div className="empty-state">No chat yet. Ask Foreman for a situational update.</div>
            )}
          </div>
          <div className="quick-replies">
            {QUICK_REPLIES.map((reply) => (
              <button key={reply} className="quick-reply" onClick={() => onSend(reply)}>
                {reply}
              </button>
            ))}
          </div>
          <form className="chat-input" onSubmit={handleSubmit}>
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask Foreman for an ops update…"
            />
            <button type="submit">Send</button>
          </form>
        </div>
        <aside className="alert-panel">
          <h3>Proactive Alerts</h3>
          {latestAlerts.length ? (
            latestAlerts.map((alert) => (
              <div key={alert.id} className="alert-item">
                <div>{alert.title}</div>
                <div>{alert.details?.description || alert.details?.message || ""}</div>
              </div>
            ))
          ) : (
            <div className="empty-state">No active alerts. Foreman will surface them here.</div>
          )}
        </aside>
      </div>
    </div>
  );
};

const AlertStack = ({ alerts }) => (
  <div className="alert-stack">
    {alerts.map((alert) => (
      <div key={alert.id} className={`alert-card ${alert.severity || "info"}`}>
        <div className="title">{alert.title}</div>
        <div>{alert.details?.description || alert.details?.message || ""}</div>
      </div>
    ))}
  </div>
);

const App = () => {
  const [currentView, setCurrentView] = useState("logs");
  const [logs, setLogs] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [qualityLog, setQualityLog] = useState([]);
  const [ragEvents, setRagEvents] = useState([]);
  const [citationGraph, setCitationGraph] = useState({ nodes: [], edges: [] });
  const [alerts, setAlerts] = useState([]);
  const [connection, setConnection] = useState("connecting");
  const [levelFilter, setLevelFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [isPaused, setIsPaused] = useState(false);
  const [systemStats, setSystemStats] = useState(null);
  const [foremanMessages, setForemanMessages] = useState([]);
  const [theme, setTheme] = useState(() => localStorage.getItem("enlitens-theme") || "dark");
  const [motionReduced, setMotionReduced] = useState(
    () => (localStorage.getItem("enlitens-motion") || "full") === "reduced"
  );

  const wsRef = useRef(null);
  const isPausedRef = useRef(isPaused);
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("enlitens-theme", theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.setAttribute("data-motion", motionReduced ? "reduced" : "full");
    localStorage.setItem("enlitens-motion", motionReduced ? "reduced" : "full");
  }, [motionReduced]);

  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
      wsRef.current = socket;

      socket.onopen = () => {
        setConnection("connected");
      };

      socket.onclose = () => {
        setConnection("disconnected");
        setTimeout(() => {
          if (wsRef.current === socket) {
            connect();
          }
        }, 2000);
      };

      socket.onerror = () => {
        setConnection("error");
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          switch (payload.type) {
            case "log":
              if (isPausedRef.current) break;
              setLogs((prev) => {
                const next = [...prev, payload];
                if (next.length > 600) next.shift();
                return next;
              });
              break;
            case "timeline":
              setTimeline((prev) => [...prev.slice(-240), payload]);
              break;
            case "quality_metrics":
              setQualityLog((prev) => [...prev.slice(-240), payload]);
              break;
            case "rag_metrics":
              setRagEvents((prev) => [...prev.slice(-120), payload]);
              break;
            case "citation_graph":
              setCitationGraph({ nodes: payload.nodes || [], edges: payload.edges || [] });
              break;
            case "system_metrics":
              setSystemStats(payload.metrics || {});
              break;
            case "alert":
              addAlert(payload);
              break;
            case "foreman_token":
              setForemanMessages((prev) => {
                const streamId = payload.stream_id || "foreman-stream";
                const existingIndex = prev.findIndex((msg) => msg.stream_id === streamId);
                if (existingIndex === -1) {
                  return [
                    ...prev,
                    {
                      id: streamId,
                      role: "assistant",
                      content: payload.token || "",
                      streaming: true,
                      stream_id: streamId,
                    },
                  ];
                }
                const next = [...prev];
                next[existingIndex] = {
                  ...next[existingIndex],
                  content: (next[existingIndex].content || "") + (payload.token || ""),
                  streaming: true,
                };
                return next;
              });
              break;
            case "foreman_response":
              setForemanMessages((prev) => {
                const streamId = payload.stream_id || `foreman-${Date.now()}`;
                const existingIndex = prev.findIndex((msg) => msg.stream_id === streamId);
                if (existingIndex === -1) {
                  return [
                    ...prev,
                    {
                      id: streamId,
                      role: "assistant",
                      content: payload.response,
                      streaming: false,
                      stream_id: streamId,
                      timestamp: payload.timestamp,
                    },
                  ];
                }
                const next = [...prev];
                next[existingIndex] = {
                  ...next[existingIndex],
                  content: payload.response,
                  streaming: false,
                  timestamp: payload.timestamp,
                };
                return next;
              });
              break;
            default:
              break;
          }
        } catch (error) {
          console.error("Failed to parse message", error);
        }
      };
    };

    const addAlert = (alert) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const enriched = { ...alert, id };
      setAlerts((prev) => [...prev, enriched]);
      setTimeout(() => {
        setAlerts((prev) => prev.filter((item) => item.id !== id));
      }, 12000);
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const addAlert = (alert) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const enriched = { ...alert, id };
    setAlerts((prev) => [...prev, enriched]);
    setTimeout(() => {
      setAlerts((prev) => prev.filter((item) => item.id !== id));
    }, 12000);
  };

  const stats = useMemo(() => {
    const errors = logs.filter((log) => log.level === "ERROR" || log.level === "CRITICAL").length;
    const warnings = logs.filter((log) => log.level === "WARNING").length;
    const totalLogs = logs.length;
    const latestQuality = qualityLog.length ? qualityLog[qualityLog.length - 1] : null;
    const qualityScore = latestQuality?.metrics?.quality_score ?? latestQuality?.metrics?.overall_score;
    return {
      totalLogs,
      errors,
      warnings,
      qualityScore: qualityScore ? qualityScore.toFixed(2) : "--",
    };
  }, [logs, qualityLog]);

  const handleClearLogs = () => setLogs([]);
  const handleTogglePause = () => setIsPaused((prev) => !prev);

  const sendForemanMessage = (message) => {
    const trimmed = message.trim();
    if (!trimmed) return;
    const timestamp = new Date().toISOString();
    setForemanMessages((prev) => [
      ...prev,
      { id: `user-${timestamp}`, role: "user", content: trimmed, streaming: false, timestamp },
    ]);

    const socket = wsRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "foreman_query", query: trimmed }));
    } else {
      addAlert({
        title: "WebSocket disconnected",
        severity: "warning",
        details: { description: "Unable to send message to Foreman. Reconnecting…" },
      });
    }
  };

  return (
    <div className="dashboard-shell">
      <Header
        theme={theme}
        onThemeChange={setTheme}
        motionReduced={motionReduced}
        onMotionChange={setMotionReduced}
        connection={connection}
      />
      <Sidebar
        currentView={currentView}
        onChangeView={setCurrentView}
        stats={stats}
        onClearLogs={handleClearLogs}
        onTogglePause={handleTogglePause}
        isPaused={isPaused}
      />
      {currentView === "logs" && (
        <LogsView
          logs={logs}
          levelFilter={levelFilter}
          onLevelChange={setLevelFilter}
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
        />
      )}
      {currentView === "quality" && (
        <QualityView
          qualityLog={qualityLog}
          timeline={timeline}
          citationGraph={citationGraph}
          ragEvents={ragEvents}
          systemStats={systemStats}
          theme={theme}
        />
      )}
      {currentView === "foreman" && (
        <ForemanView messages={foremanMessages} onSend={sendForemanMessage} alerts={alerts} />
      )}
      <AlertStack alerts={alerts} />
    </div>
  );
};

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
