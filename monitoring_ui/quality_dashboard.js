const API_ENDPOINT = "/api/quality-dashboard";
const REFRESH_INTERVAL_MS = 15000;

const selectors = {
  avgQuality: document.getElementById("avg-quality"),
  avgConfidence: document.getElementById("avg-confidence"),
  passRate: document.getElementById("pass-rate"),
  retryCount: document.getElementById("retry-count"),
  avgRetry: document.getElementById("avg-retry"),
  selfCritique: document.getElementById("self-critique-rate"),
  recentDocuments: document.getElementById("recent-documents"),
  failureReasons: document.getElementById("failure-reasons"),
  refreshBtn: document.getElementById("refresh-btn"),
};

function formatPercent(value) {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

function formatDecimal(value, digits = 2) {
  if (value === null || value === undefined) return "—";
  return Number.parseFloat(value).toFixed(digits);
}

function renderRecentDocuments(records) {
  const tbody = selectors.recentDocuments;
  tbody.innerHTML = "";

  if (!records.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No validation events recorded yet.</td></tr>';
    return;
  }

  for (const record of records) {
    const retryBadge = record.needs_retry
      ? '<span class="badge retry">Retry</span>'
      : '<span class="badge pass">Pass</span>';

    const issues = record.failure_reasons?.length
      ? record.failure_reasons.join(", ")
      : "—";

    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${record.document_id || "—"}</td>
      <td>${formatDecimal(record.quality, 3)}</td>
      <td>${retryBadge}</td>
      <td>${record.retry_attempt ?? "—"}</td>
      <td>${issues}</td>
      <td>${record.timestamp ? new Date(record.timestamp).toLocaleString() : "—"}</td>
    `;
    tbody.appendChild(row);
  }
}

function renderFailureReasons(reasons) {
  const tbody = selectors.failureReasons;
  tbody.innerHTML = "";

  if (!reasons.length) {
    tbody.innerHTML = '<tr><td colspan="2" class="empty">No failures observed.</td></tr>';
    return;
  }

  for (const item of reasons) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.reason}</td>
      <td>${item.count}</td>
    `;
    tbody.appendChild(row);
  }
}

async function fetchMetrics() {
  try {
    const response = await fetch(API_ENDPOINT, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const data = await response.json();

    selectors.avgQuality.textContent = formatDecimal(data.average_quality);
    selectors.avgConfidence.textContent = formatDecimal(data.average_confidence);
    selectors.passRate.textContent = formatPercent(data.pass_rate);
    selectors.retryCount.textContent = data.documents_requiring_retry ?? "—";
    selectors.avgRetry.textContent = formatDecimal(data.average_retry_attempt);
    selectors.selfCritique.textContent = formatPercent(data.self_critique_rate);

    renderRecentDocuments(data.recent_documents || []);
    renderFailureReasons(data.top_failure_reasons || []);
  } catch (error) {
    console.error("Failed to load quality metrics", error);
  }
}

function init() {
  if (selectors.refreshBtn) {
    selectors.refreshBtn.addEventListener("click", fetchMetrics);
  }
  fetchMetrics();
  setInterval(fetchMetrics, REFRESH_INTERVAL_MS);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
