"""Industrial Visual Inspection AI — PCB Automated Optical Inspection (AOI) Terminal."""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from dashboard.theme import get_industrial_theme_css  # noqa: E402
from src.generator.pcb_synthesizer import PCBSynthesizer  # noqa: E402
from src.models.schemas import DEFECT_CLASS_MAP, DEFECT_COLORS  # noqa: E402
from src.pipeline.inspector import PCBInspector  # noqa: E402
from src.vision.visualizer import draw_detections, extract_defect_crops  # noqa: E402

st.set_page_config(
    page_title="PCB Defect AOI Quality Control Station",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_industrial_theme_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ AOI Machine Parameters")
    conf_thresh = st.slider("Detection Confidence Threshold", min_value=0.20, max_value=0.95, value=0.50, step=0.05)
    iou_thresh = st.slider("NMS Overlap (IoU) Threshold", min_value=0.10, max_value=0.80, value=0.40, step=0.05)
    allow_conditional = st.checkbox("Allow Minor Copper (Conditional PASS)", value=False)

    st.markdown("---")
    st.markdown("### 🏷️ Defect Legend")
    for cid, enum_val in DEFECT_CLASS_MAP.items():
        name = enum_val.value
        color = DEFECT_COLORS.get(name, "#FFFFFF")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:0.85rem;">'
            f'<div style="width:12px;height:12px;border-radius:2px;background:{color};"></div>'
            f'<span style="font-family:monospace;">{name}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Synthetic PCB AOI Prototype v1.0.1 · Simulated IPC-A-610 Defect Categories")


# ---------------------------------------------------------------------------
# Header Topbar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="terminal-topbar">
  <div class="terminal-heading">
    <span class="station-tag">AOI Line Station #04</span>
    <span class="terminal-title-text">🔬 PCB Defect Detection & Quality Control Terminal</span>
  </div>
  <div>
    <span class="status-pill-online">
      <span class="status-dot"></span> SYSTEM ONLINE · READY
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


tabs = st.tabs([
    "🔬 Live AOI Inspection",
    "📦 Batch Inspection & Yield",
    "📈 Benchmark & Model Evaluation",
    "📚 IPC-A-610 Defect Guide",
])

inspector = PCBInspector(
    conf_threshold=conf_thresh,
    iou_threshold=iou_thresh,
    allow_minor_copper=allow_conditional,
)


# ===========================================================================
# TAB 1: Live AOI Inspection
# ===========================================================================
with tabs[0]:
    col_input, col_view = st.columns([1, 2], gap="medium")

    with col_input:
        st.markdown("#### 📥 Image Acquisition")

        sample_choice = st.selectbox(
            "Select Benchmark Sample or Upload:",
            options=[
                "Upload Custom Image",
                "Generate Fresh Golden Board (PASS)",
                "Sample: Open Circuit Defect",
                "Sample: Short Defect",
                "Sample: Missing Hole Defect",
                "Sample: Spurious Copper Defect",
                "Sample: Spur & Mouse Bite",
            ],
        )

        input_img: Image.Image | None = None
        img_name = "sample_board.png"

        if sample_choice == "Upload Custom Image":
            uploaded = st.file_uploader("Upload PCB Board image (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])
            if uploaded:
                input_img = Image.open(uploaded).convert("RGB")
                img_name = uploaded.name
        else:
            synth = PCBSynthesizer(width=800, height=800, seed=101)
            golden, meta = synth.generate_board()

            if sample_choice == "Generate Fresh Golden Board (PASS)":
                input_img = golden
                img_name = "golden_board.png"
            else:
                from src.generator.defect_injector import DefectInjector
                from src.models.schemas import DefectClass
                inj = DefectInjector(seed=202)

                if "Open Circuit" in sample_choice:
                    input_img, _ = inj.inject_defects(golden, meta, target_defects=[DefectClass.OPEN_CIRCUIT], num_defects=2)
                    img_name = "pcb_open_circuit.png"
                elif "Short" in sample_choice:
                    input_img, _ = inj.inject_defects(golden, meta, target_defects=[DefectClass.SHORT], num_defects=2)
                    img_name = "pcb_short.png"
                elif "Missing Hole" in sample_choice:
                    input_img, _ = inj.inject_defects(golden, meta, target_defects=[DefectClass.MISSING_HOLE], num_defects=3)
                    img_name = "pcb_missing_hole.png"
                elif "Spurious Copper" in sample_choice:
                    input_img, _ = inj.inject_defects(golden, meta, target_defects=[DefectClass.SPURIOUS_COPPER], num_defects=3)
                    img_name = "pcb_spurious_copper.png"
                else:
                    input_img, _ = inj.inject_defects(golden, meta, target_defects=[DefectClass.SPUR, DefectClass.MOUSE_BITE], num_defects=3)
                    img_name = "pcb_spur_mousebite.png"

        if input_img:
            st.markdown("#### ⚙️ Quick Inspection")
            run_btn = st.button("⚡ Run Optical Inspection", type="primary", use_container_width=True)

    with col_view:
        if input_img:
            # Run inspection
            res = inspector.inspect_image(input_img, filename=img_name)

            # Status Banner
            if "PASS" in res.pass_fail_status:
                st.markdown(f"""
                <div class="status-banner-pass">
                  <span>✅</span> QUALITY VERDICT: <strong>{res.pass_fail_status}</strong> (0 Critical Defect Violations)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-banner-fail">
                  <span>❌</span> QUALITY VERDICT: <strong>{res.pass_fail_status}</strong> ({res.total_defects} Defects Detected)
                </div>
                """, unsafe_allow_html=True)

            # Metric Cards Row
            card_status = "pass" if "PASS" in res.pass_fail_status else "fail"
            st.markdown(f"""
            <div class="metric-row">
              <div class="kpi-card {card_status}">
                <div class="kpi-label">Inspection Verdict</div>
                <div class="kpi-val">{res.pass_fail_status}</div>
                <div class="kpi-sub">Simulated AOI Rule Check</div>
              </div>
              <div class="kpi-card info">
                <div class="kpi-label">Defects Found</div>
                <div class="kpi-val">{res.total_defects}</div>
                <div class="kpi-sub">Total Bounding Boxes</div>
              </div>
              <div class="kpi-card info">
                <div class="kpi-label">Inference Latency</div>
                <div class="kpi-val">{res.inference_time_ms} ms</div>
                <div class="kpi-sub">Multi-scale AOI Core</div>
              </div>
              <div class="kpi-card info">
                <div class="kpi-label">Resolution</div>
                <div class="kpi-val">{res.image_width}×{res.image_height}</div>
                <div class="kpi-sub">Optical Surface</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Side-by-side or Tabbed View
            v_tab1, v_tab2 = st.tabs(["🎯 Annotated Inspection View", "🔍 Defect Zoom Gallery"])

            with v_tab1:
                annotated_img = draw_detections(input_img, res.defects)
                st.image(annotated_img, caption=f"Annotated Inspection: {img_name} ({res.total_defects} defects)", use_container_width=True)

            with v_tab2:
                crops = extract_defect_crops(input_img, res.defects, padding=25)
                if not crops:
                    st.success("✅ Clean board: Zero defects to display in zoom gallery.")
                else:
                    st.markdown("#### High-Magnification Defect Cutouts")
                    crop_cols = st.columns(min(len(crops), 4))
                    for idx, (det, crop_img) in enumerate(crops):
                        col_idx = idx % len(crop_cols)
                        with crop_cols[col_idx]:
                            st.image(crop_img, caption=f"#{idx+1}: {det.class_name} ({int(det.confidence*100)}%)", use_container_width=True)
                            st.caption(f"Severity: **{det.severity}** | Box: ({det.bbox.x_min},{det.bbox.y_min})")


# ===========================================================================
# TAB 2: Batch Inspection & Yield Analytics
# ===========================================================================
with tabs[1]:
    st.markdown("### 📦 Continuous Conveyor Batch Inspection")
    st.markdown("Simulate high-throughput surface mount assembly line inspections across multiple boards.")

    batch_btn = st.button("🚀 Run Batch Inspection on 12 Assembly Boards", type="primary")

    if batch_btn:
        with st.spinner("Processing automated conveyor batch..."):
            synth = PCBSynthesizer(width=800, height=800, seed=555)
            from src.generator.defect_injector import DefectInjector
            from src.models.schemas import DefectClass
            inj = DefectInjector(seed=777)

            batch_items = []
            for i in range(1, 13):
                b_img, meta = synth.generate_board()
                # 70% clean yield, 30% defective
                if i in (1, 2, 4, 6, 7, 9, 10, 12):
                    batch_items.append((f"PCB_LOT_A_{i:02d}.png", b_img))
                else:
                    def_type = [DefectClass.SHORT, DefectClass.OPEN_CIRCUIT, DefectClass.MISSING_HOLE][i % 3]
                    d_img, _ = inj.inject_defects(b_img, meta, target_defects=[def_type], num_defects=2)
                    batch_items.append((f"PCB_LOT_A_{i:02d}.png", d_img))

            summary = inspector.inspect_batch(batch_items)

            # Yield Metrics
            st.markdown(f"""
            <div class="metric-row">
              <div class="kpi-card pass">
                <div class="kpi-label">Production Yield</div>
                <div class="kpi-val">{summary.yield_rate_percent}%</div>
                <div class="kpi-sub">First Pass Yield (FPY)</div>
              </div>
              <div class="kpi-card pass">
                <div class="kpi-label">Passed Boards</div>
                <div class="kpi-val">{summary.passed_count} / {summary.total_inspected}</div>
                <div class="kpi-sub">Acceptable Quality Limit</div>
              </div>
              <div class="kpi-card fail">
                <div class="kpi-label">Rejected Boards</div>
                <div class="kpi-val">{summary.failed_count}</div>
                <div class="kpi-sub">Rework Required</div>
              </div>
              <div class="kpi-card info">
                <div class="kpi-label">Avg Inspection Time</div>
                <div class="kpi-val">{summary.avg_latency_ms} ms</div>
                <div class="kpi-sub">Throughput: ~{int(60000 / max(1, summary.avg_latency_ms))} boards/min</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Batch Results Table
            st.markdown("#### 📋 Board Inspection Manifest")
            table_data = []
            for item in summary.items:
                table_data.append({
                    "Board Lot ID": item.filename,
                    "Status": item.pass_fail_status,
                    "Total Defects": item.total_defects,
                    "Defect Breakdown": str(item.defect_counts) if item.defect_counts else "None",
                    "Latency (ms)": item.inference_time_ms,
                })
            st.dataframe(table_data, use_container_width=True)


# ===========================================================================
# TAB 3: Model Evaluation Benchmark
# ===========================================================================
with tabs[2]:
    st.markdown("### 📈 Quantitative Accuracy & Defect Evaluation Benchmark")
    st.markdown("Evaluate detector performance against normalized ground truth on standard benchmark test sets.")

    eval_btn = st.button("▶ Run Full Benchmark Evaluation Suite", type="primary")

    if eval_btn:
        with st.spinner("Generating benchmark evaluation report across all 6 classes..."):
            import subprocess
            res = subprocess.run(
                [sys.executable, "tests/evaluation/run_evaluation.py"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            st.code(res.stdout + res.stderr, language="text")


# ===========================================================================
# TAB 4: IPC-A-610 Defect Guide
# ===========================================================================
with tabs[3]:
    st.markdown("### 📚 Simulated IPC-A-610 PCB Defect Taxonomy")
    st.markdown("Visual reference criteria for simulated inspection of printed circuit boards.")

    guide_items = [
        ("0. Missing Hole", "#EF4444", "CRITICAL", "Drill cycle failure where a designated via or mounting hole was not drilled through the copper annular ring, causing open nets between multi-layer inner planes."),
        ("1. Mouse Bite", "#F59E0B", "MEDIUM", "Edge breakout notch or erosion in conductive track. Reduces effective cross-sectional current-carrying area and can cause localized overheating or thermal failure."),
        ("2. Open Circuit", "#DC2626", "CRITICAL", "Complete electrical discontinuity or physical gap cut through a copper trace. Results in immediate circuit operation failure."),
        ("3. Short Circuit", "#8B5CF6", "CRITICAL", "Unintended conductive copper bridge or solder bridging adjacent tracks. Causes destructive power rail shorts or signal cross-coupling."),
        ("4. Spur", "#3B82F6", "MEDIUM", "Sharp extraneous copper burr or branch protruding from a trace perimeter. Violates electrical clearance design rules and may arc at high voltages."),
        ("5. Spurious Copper", "#10B981", "LOW", "Isolated un-etched copper fleck or parasitic island on FR4 mask. Minor flaw unless bridging high-voltage insulation barriers."),
    ]

    for title, col, sev, desc in guide_items:
        st.markdown(f"""
        <div style="background:#162332;border:1px solid #243950;border-left:4px solid {col};border-radius:8px;padding:14px 18px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <strong style="color:#ffffff;font-size:1.05rem;">{title}</strong>
            <span style="font-size:0.75rem;font-weight:700;padding:2px 8px;border-radius:4px;background:rgba(255,255,255,0.1);color:{col};">{sev}</span>
          </div>
          <p style="color:#94a3b8;font-size:0.9rem;margin:0;line-height:1.5;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
