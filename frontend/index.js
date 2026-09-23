/* ============================================================
   SIH26165 — Safety Report Analyzer  |  index.js
   ============================================================ */

// Empty string = relative URL. Works on Render (same-origin: Flask serves
// both the frontend and /api/* from the same process) and locally when
// running gunicorn or python backend/app.py directly.
const API = '';

// ─── State ───────────────────────────────────────────────────
let _allHistory = [];

// ─── Navigation ──────────────────────────────────────────────
function switchSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  document.getElementById('section-' + name).classList.remove('hidden');
  document.getElementById('nav-' + name).classList.add('active');

  if (name === 'dashboard') loadDashboard();
  if (name === 'history')   loadHistory();
  if (name === 'model')     loadModelInfo();
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => switchSection(btn.dataset.section));
});

// ─── API status check ────────────────────────────────────────
async function checkApiStatus() {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  try {
    const r = await fetch(API + '/api/health');
    if (r.ok) {
      dot.className  = 'status-dot ok';
      text.textContent = 'API online';
    } else throw new Error();
  } catch {
    dot.className  = 'status-dot err';
    text.textContent = 'API offline';
  }
}

// ─── Dashboard ───────────────────────────────────────────────
async function loadDashboard() {
  try {
    const r    = await fetch(API + '/api/stats');
    const data = await r.json();

    document.getElementById('stat-total').textContent  = data.total_analyzed;
    document.getElementById('stat-high').textContent   = data.high_risk;
    document.getElementById('stat-medium').textContent = data.medium_risk;
    document.getElementById('stat-low').textContent    = data.low_risk;

    drawDonut(data.high_risk, data.medium_risk, data.low_risk, data.total_analyzed);
    renderRecent(data.recent || []);
  } catch {
    // backend may not be up yet
  }
}

function drawDonut(high, medium, low, total) {
  const canvas = document.getElementById('risk-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2, r = 80, thick = 22;

  ctx.clearRect(0, 0, W, H);

  const segments = [
    { value: high,   color: '#ff4757' },
    { value: medium, color: '#ffa502' },
    { value: low,    color: '#2ed573' },
  ];
  const nonZero = segments.filter(s => s.value > 0);
  if (nonZero.length === 0) {
    // empty circle
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,.07)';
    ctx.lineWidth = thick;
    ctx.stroke();
    document.getElementById('donut-total').textContent = 0;
    document.getElementById('chart-legend').innerHTML = '';
    return;
  }

  let startAngle = -Math.PI / 2;
  segments.forEach(seg => {
    if (!seg.value) return;
    const slice = (seg.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + slice - 0.04);
    ctx.strokeStyle = seg.color;
    ctx.lineWidth = thick;
    ctx.lineCap = 'round';
    ctx.stroke();
    startAngle += slice;
  });

  document.getElementById('donut-total').textContent = total;

  const legend = document.getElementById('chart-legend');
  const labels = ['HIGH', 'MEDIUM', 'LOW'];
  const colors = ['#ff4757', '#ffa502', '#2ed573'];
  const vals   = [high, medium, low];
  legend.innerHTML = labels.map((l, i) => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${colors[i]}"></span>
      <span class="legend-name">${l}</span>
      <span class="legend-count">${vals[i]}</span>
    </div>
  `).join('');
}

function renderRecent(items) {
  const el = document.getElementById('recent-list');
  if (!items.length) {
    el.innerHTML = '<div class="empty-state">No analyses yet. Submit a report to begin.</div>';
    return;
  }
  el.innerHTML = items.map(r => `
    <div class="recent-item ${r.sif_potential}" onclick="switchSection('analyze')">
      <span class="recent-risk risk-${r.sif_potential}">${r.sif_potential}</span>
      <span class="recent-text">${escHtml(r.report_text)}</span>
      <span class="recent-conf">${r.confidence}%</span>
    </div>
  `).join('');
}

// ─── Analyze ─────────────────────────────────────────────────
document.querySelectorAll('.quick-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('report-input').value = btn.dataset.text;
  });
});

async function submitAnalysis() {
  const text = document.getElementById('report-input').value.trim();
  if (!text) {
    document.getElementById('report-input').focus();
    return;
  }

  // Loading state
  const btnText    = document.getElementById('analyze-btn-text');
  const spinner    = document.getElementById('analyze-spinner');
  const resultPanel = document.getElementById('result-panel');

  btnText.textContent = 'Analyzing…';
  spinner.classList.remove('hidden');
  document.getElementById('analyze-btn').disabled = true;
  resultPanel.classList.add('hidden');

  try {
    const res = await fetch(API + '/api/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ report_text: text }),
    });

    if (!res.ok) throw new Error('API error ' + res.status);
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert('Error contacting API: ' + err.message + '\n\nMake sure the Flask backend is running.');
  } finally {
    btnText.textContent = '⬡ Analyze Report';
    spinner.classList.add('hidden');
    document.getElementById('analyze-btn').disabled = false;
  }
}

function renderResult(d) {
  const panel = document.getElementById('result-panel');

  // Risk hero
  const hero = document.getElementById('risk-hero');
  hero.className = 'risk-hero ' + d.risk_level;
  document.getElementById('risk-badge').textContent = d.risk_level;

  // Safety review override banner (distinguish ML prediction from final risk;
  // also shown when ML HIGH already matches but critical evidence requires review)
  const ovBanner = document.getElementById('override-banner');
  if (d.safety_review_required) {
    ovBanner.classList.remove('hidden');
    document.getElementById('override-text').textContent =
      'ML prediction: ' + (d.ml_prediction || '—') +
      ' (' + (d.ml_confidence !== undefined ? d.ml_confidence + '%' : '—') + '). ' +
      (d.review_reason || d.override_reason ||
       'Critical safety indicators detected — safety review required.');
  } else {
    ovBanner.classList.add('hidden');
  }

  // Confidence bar
  const bar = document.getElementById('conf-bar');
  bar.style.width = '0%';
  setTimeout(() => bar.style.width = d.confidence + '%', 50);
  document.getElementById('conf-pct').textContent = d.confidence + '%';

  // Detail fields
  document.getElementById('res-rule').textContent    = d.life_saving_rule    || '—';
  document.getElementById('res-hazard').textContent  = d.hazard              || '—';
  document.getElementById('res-activity').textContent= d.activity            || '—';
  document.getElementById('res-barrier').textContent = d.barrier_failure     || '—';
  document.getElementById('res-explanation').textContent = d.explanation      || '—';
  document.getElementById('res-action').textContent  = d.recommended_action  || '—';

  // Probability bars
  const probColors = { HIGH: '#ff4757', MEDIUM: '#ffa502', LOW: '#2ed573' };
  const probs = d.class_probabilities || {};
  const probBars = document.getElementById('prob-bars');
  probBars.innerHTML = ['HIGH', 'MEDIUM', 'LOW'].map(cls => {
    const pct = probs[cls] || 0;
    return `
      <div class="prob-row">
        <span class="prob-name risk-${cls}" style="color:${probColors[cls]}">${cls}</span>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width:0%;background:${probColors[cls]}"
               data-target="${pct}"></div>
        </div>
        <span class="prob-val">${pct.toFixed(1)}%</span>
      </div>
    `;
  }).join('');

  panel.classList.remove('hidden');

  // Animate prob bars
  setTimeout(() => {
    panel.querySelectorAll('.prob-bar-fill').forEach(el => {
      el.style.width = el.dataset.target + '%';
    });
  }, 80);

  // Scroll to result
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── History ─────────────────────────────────────────────────
async function loadHistory() {
  const tbody = document.getElementById('history-tbody');
  tbody.innerHTML = '<tr><td colspan="7" class="empty-row">Loading…</td></tr>';

  try {
    const r    = await fetch(API + '/api/reports?limit=200');
    const data = await r.json();
    _allHistory = data.reports || [];
    renderHistory(_allHistory);
  } catch {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-row">Could not load reports. Is the backend running?</td></tr>';
  }
}

function renderHistory(rows) {
  const tbody = document.getElementById('history-tbody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-row">No reports found.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td style="color:var(--text3);font-family:monospace">${r.id}</td>
      <td style="max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(r.report_text)}">${escHtml(r.report_text.substring(0, 80))}…</td>
      <td><span class="risk-pill risk-${r.sif_potential}">${r.sif_potential}</span></td>
      <td style="font-family:monospace;color:var(--accent)">${r.confidence}%</td>
      <td>${r.hazard || '—'}</td>
      <td>${r.life_saving_rule || '—'}</td>
      <td style="color:var(--text3);font-size:11px">${formatDate(r.analyzed_at)}</td>
    </tr>
  `).join('');
}

function filterHistory() {
  const q   = document.getElementById('history-search').value.toLowerCase();
  const lvl = document.getElementById('history-risk-filter').value;

  const filtered = _allHistory.filter(r => {
    const matchText = !q || r.report_text.toLowerCase().includes(q) ||
                      (r.hazard || '').toLowerCase().includes(q) ||
                      (r.life_saving_rule || '').toLowerCase().includes(q);
    const matchRisk = !lvl || r.sif_potential === lvl;
    return matchText && matchRisk;
  });
  renderHistory(filtered);
}

// ─── Model Info ───────────────────────────────────────────────
async function loadModelInfo() {
  try {
    const r    = await fetch(API + '/api/model-info');
    const info = await r.json();

    // ── 4-metric mini rings (real values from backend evaluation) ──
    const METRICS = [
      { id: 'accuracy',  val: info.accuracy_percent  || 0, color: '#00d4ff' },
      { id: 'precision', val: info.precision_macro   || 0, color: '#a78bfa' },
      { id: 'recall',    val: info.recall_macro      || 0, color: '#34d399' },
      { id: 'f1',        val: info.f1_macro          || 0, color: '#fb923c' },
    ];
    const CIRC = 2 * Math.PI * 32; // r=32 → ≈ 201

    setTimeout(() => {
      METRICS.forEach(m => {
        const ring = document.getElementById('ring-' + m.id);
        const label = document.getElementById('val-' + m.id);
        if (!ring || !label) return;
        const offset = CIRC * (1 - m.val / 100);
        ring.style.strokeDashoffset = offset;
        label.textContent = m.val + '%';
      });
    }, 120);

    // ── Architecture pipeline model node ──
    const modelName = document.getElementById('pipe-model-name');
    if (modelName) modelName.textContent = info.model_type || 'TF-IDF + LogisticRegression';

    // ── Perf detail rows (live data from backend) ──
    document.getElementById('perf-detail').innerHTML = `
      <div class="perf-row"><span class="ds-label">Model</span><span class="perf-val">${escHtml(info.model_type || '—')}</span></div>
      <div class="perf-row"><span class="ds-label">Training set</span><span class="perf-val">${info.training_set_size} reports</span></div>
      <div class="perf-row"><span class="ds-label">Test set</span><span class="perf-val">${info.test_set_size} reports</span></div>
      <div class="perf-row"><span class="ds-label">Total dataset</span><span class="perf-val">${info.dataset_size} reports</span></div>
      <div class="perf-row"><span class="ds-label">High-risk recall</span><span class="perf-val">${info.high_risk_recall}%</span></div>
      <div class="perf-row"><span class="ds-label">Features</span><span class="perf-val">${escHtml(info.features || '—')}</span></div>
    `;

    // ── Confusion matrix (real values from backend) ──
    renderConfusionMatrix(info.confusion_matrix, info.class_labels || ['HIGH', 'MEDIUM', 'LOW']);

    // ── Dataset stats ──
    const dist = info.class_distribution || {};
    const total = (dist.HIGH || 0) + (dist.MEDIUM || 0) + (dist.LOW || 0);
    const colors = { HIGH: '#ff4757', MEDIUM: '#ffa502', LOW: '#2ed573' };
    document.getElementById('dataset-stats').innerHTML = `
      ${(info.class_labels || []).map(cls => `
        <div class="ds-row">
          <span class="ds-label">${cls} risk</span>
          <div class="ds-bar-wrap">
            <div class="ds-bar" style="width:${total ? ((dist[cls]||0)/total*100).toFixed(0) : 0}%;background:${colors[cls] || '#fff'}"></div>
          </div>
          <span class="ds-val">${dist[cls] || 0}</span>
        </div>
      `).join('')}
      <div class="ds-row"><span class="ds-label">Total records</span><span class="ds-val">${total}</span></div>
      <div class="ds-row"><span class="ds-label">Life-Saving Rules</span><span class="ds-val">${info.life_saving_rules}</span></div>
      <div class="ds-row"><span class="ds-label">Hazard categories</span><span class="ds-val">${info.hazard_categories}</span></div>
      <div class="ds-row"><span class="ds-label">Barrier categories</span><span class="ds-val">${info.barrier_categories}</span></div>
    `;

  } catch (e) {
    console.error('Model info load failed:', e);
  }
}

function renderConfusionMatrix(matrix, labels) {
  const grid = document.getElementById('cm-grid');
  if (!grid) return;
  if (!Array.isArray(matrix) || !matrix.length) {
    grid.innerHTML = '<span class="empty-state">No confusion matrix available.</span>';
    return;
  }

  let html = '<div class="cm-col-headers"><div class="cm-corner"></div>';
  labels.forEach(l => { html += `<div class="cm-col-head">${escHtml(l)}</div>`; });
  html += '</div>';

  matrix.forEach((row, i) => {
    html += '<div class="cm-row"><div class="cm-row-head">' + escHtml(labels[i] || '') + '</div>';
    row.forEach((val, j) => {
      const cls = (i === j) ? 'cm-hit' : 'cm-miss';
      const title = (i === j)
        ? `TRUE ${labels[i] || ''}`
        : `Predicted ${labels[j] || ''}, Actual ${labels[i] || ''}`;
      html += `<div class="cm-cell ${cls}" title="${title}">${val}</div>`;
    });
    html += '</div>';
  });

  grid.innerHTML = html;
}

// ─── Utilities ────────────────────────────────────────────────
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return iso; }
}

// ─── Init ─────────────────────────────────────────────────────
checkApiStatus();
setInterval(checkApiStatus, 10000);
loadDashboard();
