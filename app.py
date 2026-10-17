# ============================================================
# app.py
# NeuroVascular Lens
# Blood-brain barrier gene programs in human Alzheimer's brain tissue
# ============================================================

import hashlib
import io
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nvlens_core as nv

APP_VERSION = "2.0"
CONDITION_COLORS = {"AD": "#C0392B", "Control": "#2471A3", "Unknown": "#7F8C8D"}

st.set_page_config(page_title="NeuroVascular Lens", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1400px;}
      .nv-hero {padding: 1.1rem 1.4rem; border-radius: 14px;
                background: linear-gradient(120deg, #0B2545 0%, #13315C 55%, #1F4E79 100%);
                color: #F4F6F8; margin-bottom: 1rem;}
      .nv-hero h1 {font-size: 2.0rem; margin: 0 0 .25rem 0; color: #FFFFFF;}
      .nv-hero p {margin: 0; font-size: 1.02rem; color: #D6E2F0;}
      .nv-pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
                background: rgba(255,255,255,.14); margin: .55rem .35rem 0 0;}
      .nv-finding {border-left: 4px solid #1F4E79; padding: .45rem .8rem; margin: .35rem 0;
                   background: rgba(31,78,121,.06); border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nv-hero">
      <h1>🧠 NeuroVascular Lens</h1>
      <p>Measuring blood-brain barrier gene programs in human Alzheimer's brain tissue,
      the vascular biology behind cognitive decline.</p>
      <span class="nv-pill">single-nucleus RNA-seq</span>
      <span class="nv-pill">cell-type resolved</span>
      <span class="nv-pill">AD vs control statistics</span>
      <span class="nv-pill">exploratory ML signature</span>
      <span class="nv-pill">research use only</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR: DATA + PARAMETERS
# ============================================================

st.sidebar.header("1. Data")
source = st.sidebar.radio(
    "Data source",
    ["Upload 10x archives", "Local folder", "Demo cohort (synthetic)"],
    help="Upload one or more processed 10x samples (e.g. from GEO GSE163577), "
         "point to a folder on this computer, or try the app on synthetic data.",
)

st.sidebar.header("2. Quality control")
with st.sidebar.expander("QC thresholds", expanded=False):
    min_genes = st.number_input("Min genes per nucleus", 50, 2000, 200, 50)
    max_genes = st.number_input("Max genes per nucleus", 1000, 20000, 8000, 500)
    max_mt = st.number_input("Max mitochondrial %", 0.5, 50.0, 5.0, 0.5)
    min_ct = st.slider("Cell-type assignment threshold", 0.0, 1.0, 0.10, 0.05,
                       help="Minimum marker score needed to assign a nucleus to a cell type.")
qc_params = nv.QCParams(min_genes=int(min_genes), max_genes=int(max_genes), max_pct_mt=float(max_mt))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Scores describe gene-expression patterns in post-mortem brain tissue. "
    "They are not a diagnosis and must not be used for clinical decisions."
)


# ============================================================
# CACHED PROCESSING
# ============================================================

@st.cache_data(show_spinner=False, max_entries=64)
def process_bytes(data: bytes, filename: str, sample_id: str, condition: str,
                  qc: dict, min_ct_score: float, digest: str):
    adata = nv.load_10x_archive(data, filename)
    return nv.process_sample(adata, sample_id, condition, nv.QCParams(**qc), min_ct_score)


@st.cache_data(show_spinner=False, max_entries=64)
def process_path(path: str, sample_id: str, condition: str, qc: dict, min_ct_score: float, mtime: float):
    p = Path(path)
    if p.is_dir():
        adata = nv.load_10x_folder(p)
    else:
        adata = nv.load_10x_archive(p.read_bytes(), p.name)
    return nv.process_sample(adata, sample_id, condition, nv.QCParams(**qc), min_ct_score)


@st.cache_data(show_spinner=False)
def process_demo(qc: dict, min_ct_score: float, n_ad: int, n_ctrl: int):
    return [nv.process_sample(a, sid, cond, nv.QCParams(**qc), min_ct_score)
            for sid, cond, a in nv.demo_cohort(n_ad, n_ctrl)]


# ============================================================
# STEP 1: SELECT SAMPLES AND CONDITIONS
# ============================================================

qc_dict = qc_params.__dict__
manifest = []  # dicts: sample_id, condition, kind, ref

if source == "Upload 10x archives":
    files = st.file_uploader(
        "Upload processed 10x Genomics samples (.tar.gz, .tgz, .tar or .zip). Select several to compare AD and control.",
        type=["gz", "tgz", "tar", "zip"], accept_multiple_files=True,
    )
    for f in files or []:
        manifest.append({"Sample": nv.sample_id_from_filename(f.name), "Condition": nv.infer_condition(f.name),
                         "kind": "upload", "ref": f})
elif source == "Local folder":
    folder = st.text_input("Folder containing 10x archives or extracted sample folders",
                           placeholder=r"C:\Geno\GSE163577  or  /Users/me/data/GSE163577")
    if folder:
        root = Path(folder).expanduser()
        if not root.exists():
            st.error("Folder not found.")
        else:
            items = sorted([p for p in root.iterdir()
                            if p.name.lower().endswith((".tar.gz", ".tgz", ".tar", ".zip")) or
                            (p.is_dir() and nv._find_10x_dir(p) is not None)])
            if not items:
                st.warning("No 10x archives or sample folders found here.")
            for p in items:
                manifest.append({"Sample": nv.sample_id_from_filename(p.name), "Condition": nv.infer_condition(p.name),
                                 "kind": "path", "ref": str(p)})
else:
    c1, c2 = st.columns(2)
    n_ad = c1.slider("Synthetic AD samples", 3, 12, 6)
    n_ctrl = c2.slider("Synthetic control samples", 3, 12, 6)
    st.info("Demo mode uses SYNTHETIC data generated inside the app to show every feature. "
            "Results are illustrative only, not real findings.")

if source != "Demo cohort (synthetic)":
    if not manifest:
        st.markdown("#### How to use")
        st.markdown(
            "1. Download processed samples from **GEO GSE163577** (Yang et al., 2022, *Nature*), "
            "e.g. `GSM4982084_01_05_AD_filtered_feature_bc_matrix.tar.gz`.\n"
            "2. Upload several **AD and control** samples together (or point to their folder).\n"
            "3. Check the AD / Control labels, then press **Run analysis**.\n\n"
            "Each archive must contain `matrix.mtx(.gz)`, `barcodes.tsv(.gz)` and `features.tsv(.gz)`; "
            "nested folders are fine."
        )
        st.stop()
    st.markdown("#### Samples and conditions")
    st.caption("Conditions are guessed from file names (`_AD_` or `_C_`). Correct any label before running.")
    edit_df = pd.DataFrame([{"Sample": m["Sample"], "Condition": m["Condition"]} for m in manifest])
    edited = st.data_editor(
        edit_df, hide_index=True, width="stretch",
        column_config={"Condition": st.column_config.SelectboxColumn(options=["AD", "Control", "Unknown"], required=True),
                       "Sample": st.column_config.TextColumn(disabled=True)},
        key=f"editor_{len(manifest)}",
    )
    for m, cond in zip(manifest, edited["Condition"]):
        m["Condition"] = cond

run = st.button("🔬 Run analysis", type="primary", width="stretch")

if run:
    results, errors = [], []
    prog = st.progress(0.0, text="Starting...")
    if source == "Demo cohort (synthetic)":
        prog.progress(0.2, text="Generating and analysing synthetic cohort...")
        results = process_demo(qc_dict, float(min_ct), n_ad, n_ctrl)
    else:
        for i, m in enumerate(manifest):
            prog.progress(i / len(manifest), text=f"Processing {m['Sample']} ({i + 1}/{len(manifest)})...")
            try:
                if m["kind"] == "upload":
                    data = m["ref"].getvalue()
                    digest = hashlib.sha1(data).hexdigest()
                    results.append(process_bytes(data, m["ref"].name, m["Sample"], m["Condition"],
                                                 qc_dict, float(min_ct), digest))
                else:
                    p = Path(m["ref"])
                    results.append(process_path(str(p), m["Sample"], m["Condition"], qc_dict,
                                                float(min_ct), p.stat().st_mtime))
            except Exception as e:  # keep going; report at the end
                errors.append(f"{m['Sample']}: {e}")
    prog.progress(1.0, text="Done")
    st.session_state["results"] = results
    st.session_state["errors"] = errors
    st.session_state["params"] = {**qc_dict, "cell_type_threshold": float(min_ct), "source": source}

results = st.session_state.get("results")
if not results:
    st.stop()
for e in st.session_state.get("errors", []):
    st.error(e)

ft = nv.feature_table(results)
n_ad = int((ft["Condition"] == "AD").sum())
n_ctrl = int((ft["Condition"] == "Control").sum())
can_compare = n_ad >= 2 and n_ctrl >= 2
stats_df = nv.compare_groups(ft) if can_compare else pd.DataFrame()


# ============================================================
# TABS
# ============================================================

VIEWS = ["Overview", "Quality control", "Cell types", "Neurovascular programs",
         "AD vs control", "AI signature (exploratory)", "Methods & downloads", "The science"]
view = st.segmented_control("View", VIEWS, default="Overview", key="view", label_visibility="collapsed") or "Overview"
st.markdown("---")


# ---------------- OVERVIEW ----------------
if view == VIEWS[0]:
    total_nuclei = sum(r.qc["nuclei_after_qc"] for r in results)
    endo_nuclei = sum(int(r.composition.loc[r.composition["Cell type"] == "Endothelial", "Nuclei"].sum())
                      for r in results if not r.composition.empty)
    k = st.columns(5)
    k[0].metric("Samples", len(results))
    k[1].metric("AD / Control", f"{n_ad} / {n_ctrl}")
    k[2].metric("Nuclei after QC", f"{total_nuclei:,}")
    k[3].metric("Endothelial nuclei", f"{endo_nuclei:,}")
    bii_ad = ft.loc[ft["Condition"] == "AD", "Endothelial | Barrier Integrity Index"].mean()
    bii_c = ft.loc[ft["Condition"] == "Control", "Endothelial | Barrier Integrity Index"].mean()
    k[4].metric("Barrier Integrity Index (AD vs ctrl)",
                f"{bii_ad:.2f}" if pd.notna(bii_ad) else "n/a",
                delta=f"{bii_ad - bii_c:+.2f} vs control" if pd.notna(bii_ad) and pd.notna(bii_c) else None)

    st.markdown("### Key findings")
    if not can_compare:
        st.info("Load at least 2 AD and 2 control samples to see group comparisons. "
                "Single-sample results are available in the other tabs.")
    else:
        sig = stats_df[stats_df["p-value"] < 0.05]
        if sig.empty:
            st.markdown('<div class="nv-finding">No feature differed between AD and control at p &lt; 0.05 '
                        'in this cohort.</div>', unsafe_allow_html=True)
        for _, r in sig.head(6).iterrows():
            direction = "higher" if r["Difference"] > 0 else "lower"
            st.markdown(
                f'<div class="nv-finding"><b>{r["Feature"]}</b> is {direction} in AD '
                f'(effect size {r["Effect size (rank-biserial)"]:+.2f}, p = {r["p-value"]:.3g}, '
                f'FDR q = {r["FDR q-value"]:.3g}; n = {int(r["n AD"])} AD, {int(r["n Control"])} control).</div>',
                unsafe_allow_html=True)
        st.caption("Associations in this cohort only. Small sample sizes limit statistical power; "
                   "findings should be replicated in independent cohorts.")

        st.markdown("### Effect-size map")
        prog_feats = stats_df[stats_df["Feature"].str.contains(r"^(?:Endothelial|Pericyte) \| ", regex=True)].copy()
        if not prog_feats.empty:
            prog_feats[["Cell type", "Program"]] = prog_feats["Feature"].str.split(r" \| ", n=1, expand=True)
            mat = prog_feats.pivot(index="Program", columns="Cell type", values="Effect size (rank-biserial)")
            fig = px.imshow(mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                            labels={"color": "AD vs control<br>(rank-biserial)"})
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Red = higher in AD, blue = lower in AD. Values range from -1 to +1.")


# ---------------- QC ----------------
if view == VIEWS[1]:
    qc_table = pd.DataFrame([{"Sample": r.sample_id, "Condition": r.condition, **r.qc} for r in results])
    st.dataframe(qc_table, width="stretch", hide_index=True)
    pick = st.selectbox("Inspect sample", [r.sample_id for r in results], key="qc_pick")
    r = next(x for x in results if x.sample_id == pick)
    c = st.columns(3)
    for col, (field, label) in zip(c, [("n_genes_by_counts", "Genes per nucleus"),
                                       ("total_counts", "UMIs per nucleus"),
                                       ("pct_counts_mt", "Mitochondrial %")]):
        fig = px.histogram(r.obs, x=field, nbins=50, title=label)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        col.plotly_chart(fig, width="stretch")


# ---------------- CELL TYPES ----------------
if view == VIEWS[2]:
    comp_all = pd.concat([r.composition.assign(Sample=r.sample_id, Condition=r.condition)
                          for r in results if not r.composition.empty], ignore_index=True)
    fig = px.bar(comp_all, x="Sample", y="Percent", color="Cell type", title="Cell-type composition per sample",
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), barmode="stack")
    st.plotly_chart(fig, width="stretch")
    st.caption("Nuclei are annotated by canonical marker-gene scores (endothelial: CLDN5, FLT1, PECAM1, VWF, ABCB1; "
               "pericyte: PDGFRB, RGS5, KCNJ8, ABCC9; and so on). Pericyte loss relative to endothelium is one "
               "readout of neurovascular injury.")
    ratio = ft[["Condition", "Ratio | Pericyte : Endothelial"]].reset_index()
    fig = px.strip(ratio, x="Condition", y="Ratio | Pericyte : Endothelial", color="Condition",
                   hover_data=["Sample"], color_discrete_map=CONDITION_COLORS, title="Pericyte to endothelial ratio")
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Composition table"):
        st.dataframe(comp_all, width="stretch", hide_index=True)


# ---------------- PROGRAMS ----------------
if view == VIEWS[3]:
    st.markdown("Six neurovascular gene programs are scored in every nucleus "
                "(average expression of the program genes minus a matched random gene set).")
    prog_info = pd.DataFrame([{"Program": p, "Genes": ", ".join(s["genes"]), "Why it matters": s["why"]}
                              for p, s in nv.PROGRAMS.items()])
    st.dataframe(prog_info, width="stretch", hide_index=True)

    pick = st.selectbox("Sample", [r.sample_id for r in results], key="prog_pick")
    r = next(x for x in results if x.sample_id == pick)
    mat = r.by_cell_type.pivot(index="Program", columns="Cell type", values="Mean")
    vasc_first = [c for c in nv.VASCULAR_CELL_TYPES if c in mat.columns] + \
                 [c for c in mat.columns if c not in nv.VASCULAR_CELL_TYPES]
    fig = px.imshow(mat[vasc_first], color_continuous_scale="Viridis", text_auto=".2f", aspect="auto",
                    title=f"Mean program score by cell type: {pick}")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Endothelial nuclei across samples")
    prog_choice = st.selectbox("Program", list(nv.PROGRAMS), key="prog_choice")
    col = f"prog__{prog_choice}"
    rows = []
    for x in results:
        sub = x.obs.loc[x.obs["cell_type"] == "Endothelial", [col]].copy()
        if len(sub) > 1500:
            sub = sub.sample(1500, random_state=0)
        sub["Sample"], sub["Condition"] = x.sample_id, x.condition
        rows.append(sub)
    endo_df = pd.concat(rows) if rows else pd.DataFrame()
    if not endo_df.empty:
        fig = px.violin(endo_df, x="Sample", y=col, color="Condition", box=True, points=False,
                        color_discrete_map=CONDITION_COLORS, labels={col: f"{prog_choice} score"})
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Gene coverage for this sample"):
        st.dataframe(r.coverage, width="stretch", hide_index=True)


# ---------------- AD VS CONTROL ----------------
if view == VIEWS[4]:
    if not can_compare:
        st.info("Needs at least 2 AD and 2 control samples.")
    else:
        st.markdown("Each sample is summarised by its **mean program score per cell type** (a pseudobulk "
                    "approach), so statistics compare people, not individual nuclei. Groups are compared with "
                    "the Mann-Whitney U test; p-values are corrected with Benjamini-Hochberg FDR.")
        show = stats_df.copy()
        st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if show[c].dtype.kind == "f"}
                                       | {"p-value": "{:.3g}", "FDR q-value": "{:.3g}"}),
                     width="stretch", hide_index=True)
        feat = st.selectbox("Plot feature", stats_df["Feature"].tolist(), key="cmp_feat")
        d = ft[["Condition", feat]].reset_index()
        d = d[d["Condition"].isin(["AD", "Control"])]
        fig = px.box(d, x="Condition", y=feat, color="Condition", points="all", hover_data=["Sample"],
                     color_discrete_map=CONDITION_COLORS)
        rr = stats_df.set_index("Feature").loc[feat]
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
                          title=f"{feat}  (p = {rr['p-value']:.3g}, effect size {rr['Effect size (rank-biserial)']:+.2f})")
        st.plotly_chart(fig, width="stretch")


# ---------------- AI SIGNATURE ----------------
if view == VIEWS[5]:
    st.markdown(
        "Can blood-brain barrier programs **together** separate Alzheimer's from control tissue? "
        "A regularised logistic-regression model is trained on per-sample neurovascular features and "
        "tested with **leave-one-out cross-validation** (every sample is predicted by a model that never saw it). "
        "A **permutation test** (shuffling labels hundreds of times) checks whether the result beats chance."
    )
    st.warning("Exploratory research analysis on post-mortem tissue. Not a diagnostic test and not validated "
               "for any individual. With small cohorts, treat results as hypothesis-generating.")
    default_feats = [f for f in ["Endothelial | Tight junctions", "Endothelial | Nutrient and efflux transport",
                                 "Endothelial | Endothelial activation", "Pericyte | Pericyte support",
                                 "Ratio | Pericyte : Endothelial"] if f in ft.columns]
    feats = st.multiselect("Features", [c for c in ft.columns if c != "Condition"], default=default_feats)
    n_perm = st.select_slider("Permutations", options=[100, 250, 500, 1000], value=250)
    if st.button("Train and evaluate signature", disabled=not feats):
        try:
            with st.spinner("Running leave-one-out cross-validation and permutation test..."):
                clf = nv.run_classifier(ft, feats, n_perm=int(n_perm))
            st.session_state["clf"] = clf
        except Exception as e:
            st.error(str(e))
    clf = st.session_state.get("clf")
    if clf:
        m = st.columns(3)
        m[0].metric("Out-of-sample AUC", f"{clf['auc']:.2f}")
        m[1].metric("Permutation p-value", f"{clf['p_perm']:.3f}")
        m[2].metric("Samples (AD / control)", f"{clf['n_ad']} / {clf['n_ctrl']}")
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=clf["fpr"], y=clf["tpr"], mode="lines", name=f"AUC = {clf['auc']:.2f}",
                                 line=dict(width=3, color="#1F4E79")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="grey")))
        fig.update_layout(title="ROC curve (leave-one-out)", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", height=400, margin=dict(l=10, r=10, t=50, b=10))
        c1.plotly_chart(fig, width="stretch")
        fig = px.histogram(x=clf["null"], nbins=30, title="Permutation null distribution of AUC",
                           labels={"x": "AUC with shuffled labels"})
        fig.add_vline(x=clf["auc"], line_width=3, line_color="#C0392B", annotation_text="observed")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        c2.plotly_chart(fig, width="stretch")
        fig = px.bar(clf["coefs"], x="Coefficient", y="Feature", orientation="h",
                     title="Model weights (standardised; positive = pushes toward AD)",
                     color="Coefficient", color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
        with st.expander("Per-sample predictions"):
            st.dataframe(clf["pred"], width="stretch", hide_index=True)


# ---------------- METHODS & DOWNLOADS ----------------
def build_report() -> str:
    p = st.session_state.get("params", {})
    lines = [
        "# NeuroVascular Lens analysis report",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} | NeuroVascular Lens v{APP_VERSION}",
        "",
        "## Methods",
        f"- Samples: {len(results)} ({n_ad} AD, {n_ctrl} control). Source: {p.get('source', '')}.",
        f"- Quality control: nuclei kept with {p.get('min_genes')}-{p.get('max_genes')} detected genes and "
        f"<= {p.get('max_pct_mt')}% mitochondrial reads; genes detected in < 3 nuclei removed.",
        "- Normalisation: counts scaled to 10,000 per nucleus and log1p-transformed (Scanpy).",
        f"- Cell types: marker-gene scores (Scanpy score_genes); each nucleus assigned to its top-scoring "
        f"type if the score >= {p.get('cell_type_threshold')}, otherwise 'Unassigned'.",
        "- Neurovascular programs: " + "; ".join(f"{k} ({', '.join(v['genes'])})" for k, v in nv.PROGRAMS.items()) + ".",
        "- Barrier Integrity Index (endothelial): mean of tight-junction and transport scores minus endothelial "
        "activation (exploratory composite).",
        "- Statistics: per-sample means per cell type (pseudobulk); Mann-Whitney U; rank-biserial effect size; "
        "Benjamini-Hochberg FDR.",
        "- Exploratory classifier: standardised L2 logistic regression, leave-one-out cross-validation, "
        "label-permutation test.",
        f"- Software: Python {platform.python_version()}, Scanpy, scikit-learn, SciPy, Streamlit.",
        "",
        "## Limitations",
        "- Post-mortem tissue; associations only, no causal or diagnostic claims.",
        "- Marker-based annotation is approximate; small cohorts limit statistical power.",
        "",
        "## Top AD vs control differences",
    ]
    if not stats_df.empty:
        for _, r in stats_df.head(10).iterrows():
            lines.append(f"- {r['Feature']}: AD {r['Mean AD']:.3f} vs control {r['Mean Control']:.3f}, "
                         f"effect {r['Effect size (rank-biserial)']:+.2f}, p = {r['p-value']:.3g}, q = {r['FDR q-value']:.3g}")
    else:
        lines.append("- Not computed (needs >= 2 AD and >= 2 control samples).")
    clf = st.session_state.get("clf")
    if clf:
        lines += ["", "## Exploratory classifier",
                  f"- Leave-one-out AUC {clf['auc']:.2f}, permutation p = {clf['p_perm']:.3f} "
                  f"({clf['n_ad']} AD, {clf['n_ctrl']} control)."]
    lines += ["", "## References", *[f"- {ref}" for ref in nv.REFERENCES]]
    return "\n".join(lines)


if view == VIEWS[6]:
    report = build_report()
    st.markdown(report)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.md", report)
        z.writestr("sample_features.csv", ft.to_csv())
        z.writestr("qc_summary.csv", pd.DataFrame([{"Sample": r.sample_id, **r.qc} for r in results]).to_csv(index=False))
        z.writestr("cell_type_composition.csv", pd.concat(
            [r.composition.assign(Sample=r.sample_id) for r in results if not r.composition.empty]).to_csv(index=False))
        z.writestr("program_scores_by_cell_type.csv", pd.concat(
            [r.by_cell_type.assign(Sample=r.sample_id, Condition=r.condition) for r in results]).to_csv(index=False))
        if not stats_df.empty:
            z.writestr("ad_vs_control_statistics.csv", stats_df.to_csv(index=False))
        z.writestr("parameters.json", json.dumps({**st.session_state.get("params", {}), "app_version": APP_VERSION}, indent=2))
    st.download_button("⬇️ Download full results (ZIP)", buf.getvalue(), "neurovascular_lens_results.zip",
                       "application/zip", width="stretch", type="primary")
    pick = st.selectbox("Per-nucleus scores for sample", [r.sample_id for r in results], key="dl_pick")
    r = next(x for x in results if x.sample_id == pick)
    st.download_button("⬇️ Download per-nucleus scores (CSV)", r.obs.to_csv(), f"{pick}_per_nucleus.csv", "text/csv")


# ---------------- SCIENCE ----------------
if view == VIEWS[7]:
    st.markdown(
        """
        ### Why blood vessels matter for memory
        The brain is protected by the **blood-brain barrier (BBB)**: endothelial cells sealed by tight junctions,
        supported by pericytes, that decide what enters the brain. Imaging studies have shown that BBB breakdown
        appears early in people developing cognitive impairment, and the APOE4 risk gene has been linked to
        pericyte injury, barrier leakage and later cognitive decline. Most Alzheimer's research focuses on
        amyloid and tau inside neurons; this tool looks at the **vascular side**.

        ### What NeuroVascular Lens measures
        - **Which cells are present**, including endothelial cells, pericytes and smooth muscle.
        - **Six gene programs** that keep the barrier sealed, transporting, calm and supported, plus vascular
          Alzheimer's risk genes.
        - **How these programs differ** between Alzheimer's and control brains, at the level of people (pseudobulk)
          rather than individual nuclei.
        - **Whether the programs combined** carry a disease-associated signal (exploratory machine learning).

        ### What it does not do
        It does not diagnose anyone, predict an individual's cognition, or measure living patients. It analyzes
        gene expression in donated post-mortem tissue to generate hypotheses about vascular contributions to
        cognitive decline.
        """
    )
    st.markdown("### Key references")
    for ref in nv.REFERENCES:
        st.markdown(f"- {ref}")
