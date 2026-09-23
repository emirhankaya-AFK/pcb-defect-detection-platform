"""Industrial High-Contrast Dark Theme for Streamlit QC Terminal."""


def get_industrial_theme_css() -> str:
    return """
<style>
/* ============================================================
   INDUSTRIAL QC OPERATOR THEME TOKENS
   ============================================================ */
:root {
  --bg-deep:      #080e14;
  --bg-panel:     #0f1923;
  --bg-card:      #162332;
  --bg-card-alt:  #1d2e42;
  --border:       #243950;
  --border-glow:  #3b82f6;
  --text-main:    #f1f5f9;
  --text-muted:   #94a3b8;
  --text-dim:     #64748b;
  --pass-green:   #10b981;
  --fail-red:     #ef4444;
  --warn-amber:   #f59e0b;
  --aoi-blue:     #3b82f6;
}

[data-testid="stApp"] {
  background-color: var(--bg-deep) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', -apple-system, system-ui, sans-serif;
}

[data-testid="stHeader"] { display: none !important; }

.block-container {
  padding-top: 1.2rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  max-width: 1440px !important;
}

/* ============================================================
   OPERATOR TERMINAL HEADER
   ============================================================ */
.terminal-topbar {
  background: linear-gradient(135deg, #101c2b 0%, #16263a 100%);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 22px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.terminal-heading {
  display: flex;
  align-items: center;
  gap: 12px;
}
.terminal-title-text {
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  color: #ffffff;
}
.station-tag {
  background: rgba(59, 130, 246, 0.2);
  border: 1px solid rgba(59, 130, 246, 0.4);
  color: #60a5fa;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.status-pill-online {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.35);
  color: var(--pass-green);
  font-size: 0.75rem;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 20px;
}
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--pass-green);
  box-shadow: 0 0 6px var(--pass-green);
}

/* ============================================================
   KPI METRIC CARDS
   ============================================================ */
.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  position: relative;
  overflow: hidden;
}
.kpi-card.pass { border-left: 4px solid var(--pass-green); }
.kpi-card.fail { border-left: 4px solid var(--fail-red); }
.kpi-card.info { border-left: 4px solid var(--aoi-blue); }
.kpi-card.warn { border-left: 4px solid var(--warn-amber); }

.kpi-label {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
}
.kpi-val {
  font-size: 1.55rem;
  font-weight: 800;
  font-family: 'JetBrains Mono', monospace;
  color: #ffffff;
  line-height: 1.1;
}
.kpi-sub {
  font-size: 0.72rem;
  color: var(--text-dim);
  margin-top: 4px;
}

/* ============================================================
   STATUS BANNER
   ============================================================ */
.status-banner-pass {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.35);
  border-radius: 8px;
  padding: 12px 18px;
  color: #34d399;
  font-weight: 700;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.status-banner-fail {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.35);
  border-radius: 8px;
  padding: 12px 18px;
  color: #f87171;
  font-weight: 700;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}

/* ============================================================
   DEFECT ZOOM GALLERY CARDS
   ============================================================ */
.defect-chip-table {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  margin-top: 10px;
}
.defect-chip-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid rgba(36, 57, 80, 0.5);
  font-size: 0.82rem;
}
.defect-chip-row:last-child { border-bottom: none; }

/* Tabs styling */
[data-testid="stTabs"] > div:first-child {
  background: var(--bg-panel) !important;
  border-radius: 8px 8px 0 0 !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 8px !important;
  gap: 4px !important;
}
[data-testid="stTabs"] button {
  color: var(--text-muted) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  padding: 10px 16px !important;
  border-radius: 6px 6px 0 0 !important;
  border: none !important;
  background: transparent !important;
}
[data-testid="stTabs"] button:hover {
  color: #ffffff !important;
  background: var(--bg-card) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: #60a5fa !important;
  background: var(--bg-deep) !important;
  border-bottom: 2px solid var(--aoi-blue) !important;
}

/* Form controls */
[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] {
  background-color: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
}
</style>
"""
