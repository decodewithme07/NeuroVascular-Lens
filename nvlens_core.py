"""
nvlens_core.py
NeuroVascular Lens - analysis engine.

Pure functions (no Streamlit) so the pipeline can be tested and reused:
loading 10x archives, quality control, cell-type annotation, neurovascular
program scoring, cohort statistics and an exploratory classifier.

Research use only. Scores describe gene-expression patterns in post-mortem
brain tissue; they are not a diagnosis.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

sc.settings.verbosity = 0


# ============================================================
# BIOLOGY: CELL TYPES AND NEUROVASCULAR PROGRAMS
# ============================================================

# Canonical marker genes used to annotate nuclei in human brain vascular
# snRNA-seq (vascular-enriched datasets such as GSE163577 still contain
# parenchymal cells, so those are annotated too).
CELL_TYPE_MARKERS: dict[str, list[str]] = {
    "Endothelial": ["CLDN5", "FLT1", "PECAM1", "VWF", "ABCB1"],
    "Pericyte": ["PDGFRB", "RGS5", "KCNJ8", "ABCC9"],
    "Smooth muscle": ["ACTA2", "MYH11", "TAGLN"],
    "Fibroblast": ["COL1A2", "DCN", "LUM"],
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2"],
    "Microglia": ["CX3CR1", "P2RY12", "CSF1R"],
    "Oligodendrocyte": ["MBP", "PLP1", "MOBP"],
    "Neuron": ["SNAP25", "RBFOX3", "SYT1"],
}

VASCULAR_CELL_TYPES = ["Endothelial", "Pericyte", "Smooth muscle", "Fibroblast"]

# Neurovascular gene programs. Each program has a short rationale linking it
# to blood-brain barrier (BBB) function and cognitive health.
PROGRAMS: dict[str, dict] = {
    "Tight junctions": {
        "genes": ["CLDN5", "OCLN", "TJP1", "TJP2", "CDH5", "JAM2", "JAM3", "ESAM"],
        "direction": +1,
        "why": "Seal the space between endothelial cells. Loss of junction "
               "integrity lets blood-borne molecules leak into the brain.",
    },
    "Cytoskeleton": {
        "genes": ["ACTB", "ACTN1", "MYH9", "PFN1", "ARPC2", "RHOA"],
        "direction": 0,
        "why": "Anchors junctions and controls endothelial cell shape; "
               "remodeling accompanies barrier stress.",
    },
    "Nutrient and efflux transport": {
        "genes": ["SLC2A1", "ABCB1", "SLC7A5", "MFSD2A", "SLC16A1", "ABCG2", "LRP1", "TFRC"],
        "direction": +1,
        "why": "Moves glucose and nutrients into the brain and pumps toxins "
               "and amyloid-beta out (e.g. ABCB1, LRP1).",
    },
    "Endothelial activation": {
        "genes": ["VCAM1", "ICAM1", "SELE", "SELP", "IL1R1", "NFKBIA"],
        "direction": -1,
        "why": "Inflammatory adhesion signals that recruit immune cells and "
               "accompany barrier breakdown.",
    },
    "Pericyte support": {
        "genes": ["PDGFRB", "RGS5", "ANPEP", "KCNJ8", "ABCC9", "NOTCH3"],
        "direction": +1,
        "why": "Pericytes maintain the barrier; pericyte injury has been "
               "linked to BBB breakdown and cognitive decline.",
    },
    "AD genetic risk (vascular)": {
        "genes": ["APOE", "CLU", "PICALM", "BIN1", "SORL1", "PTK2B", "CD2AP", "ABCA7", "FERMT2", "CASS4"],
        "direction": 0,
        "why": "Alzheimer's risk genes from genome-wide studies, several of "
               "which are expressed in brain vascular cells.",
    },
}

# Programs combined into the Barrier Integrity Index (in endothelial nuclei).
BII_COMPONENTS = {
    "Tight junctions": +1,
    "Nutrient and efflux transport": +1,
    "Endothelial activation": -1,
}

REFERENCES = [
    ("Yang AC et al. (2022). A human brain vascular atlas reveals diverse "
     "mediators of Alzheimer's risk. Nature 603, 885-892. (GEO: GSE163577)"),
    ("Nation DA et al. (2019). Blood-brain barrier breakdown is an early "
     "biomarker of human cognitive dysfunction. Nature Medicine 25, 270-276."),
    ("Montagne A et al. (2020). APOE4 leads to blood-brain barrier "
     "dysfunction predicting cognitive decline. Nature 581, 71-76."),
    ("Sweeney MD, Sagare AP, Zlokovic BV (2018). Blood-brain barrier breakdown "
     "in Alzheimer disease and other neurodegenerative disorders. "
     "Nature Reviews Neurology 14, 133-150."),
]


# ============================================================
# DATA LOADING
# ============================================================

def _find_10x_dir(root: Path) -> Path | None:
    """Return the first directory (root included) holding a 10x matrix."""
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        for suffix in (".gz", ""):
            if all((d / f"{n}{suffix}").exists() for n in ("matrix.mtx", "barcodes.tsv")) and (
                (d / f"features.tsv{suffix}").exists() or (d / f"genes.tsv{suffix}").exists()
            ):
                return d
    return None


def _safe_extract_tar(archive: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in archive.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Unsafe path in archive: {member.name}")
    archive.extractall(dest)


def _safe_extract_zip(archive: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for name in archive.namelist():
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise ValueError(f"Unsafe path in archive: {name}")
    archive.extractall(dest)


def load_10x_archive(data: bytes, filename: str) -> sc.AnnData:
    """Load a 10x Genomics archive (.tar.gz/.tgz/.tar/.zip) from raw bytes."""
    name = filename.lower()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                _safe_extract_zip(z, tmp_path)
        elif name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as t:
                _safe_extract_tar(t, tmp_path)
        else:
            raise ValueError("Unsupported archive. Use .tar.gz, .tgz, .tar or .zip.")
        mdir = _find_10x_dir(tmp_path)
        if mdir is None:
            raise ValueError("No 10x matrix (matrix.mtx, barcodes.tsv, features.tsv) found in archive.")
        adata = sc.read_10x_mtx(mdir, var_names="gene_symbols", make_unique=True)
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    return adata


def load_10x_folder(path: str | Path) -> sc.AnnData:
    """Load an already-extracted 10x directory (searched recursively)."""
    mdir = _find_10x_dir(Path(path))
    if mdir is None:
        raise ValueError(f"No 10x matrix found under {path}")
    adata = sc.read_10x_mtx(mdir, var_names="gene_symbols", make_unique=True)
    adata.X = sparse.csr_matrix(adata.X, dtype=np.float32)
    return adata


def infer_condition(filename: str) -> str:
    """Guess AD / Control from GSE163577-style names (e.g. 01_05_AD_..., 02_03_C_...)."""
    n = filename.lower().replace("-", "_")
    if "_ad_" in n or n.startswith("ad_") or "_ad." in n:
        return "AD"
    if "_c_" in n or "_ctrl" in n or "control" in n or "_nci_" in n:
        return "Control"
    return "Unknown"


def sample_id_from_filename(filename: str) -> str:
    base = Path(filename).name
    for ext in (".tar.gz", ".tgz", ".tar", ".zip"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base.replace("_filtered_feature_bc_matrix", "")


# ============================================================
# QC AND NORMALIZATION
# ============================================================

@dataclass
class QCParams:
    min_genes: int = 200
    max_genes: int = 8000
    max_pct_mt: float = 5.0
    min_cells_per_gene: int = 3


def run_qc(adata: sc.AnnData, p: QCParams) -> tuple[sc.AnnData, dict]:
    adata = adata.copy()
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    before = adata.n_obs
    keep = (
        (adata.obs["n_genes_by_counts"] >= p.min_genes)
        & (adata.obs["n_genes_by_counts"] <= p.max_genes)
        & (adata.obs["pct_counts_mt"] <= p.max_pct_mt)
    )
    adata = adata[keep].copy()
    sc.pp.filter_genes(adata, min_cells=p.min_cells_per_gene)
    summary = {
        "nuclei_before_qc": int(before),
        "nuclei_after_qc": int(adata.n_obs),
        "pct_retained": round(100 * adata.n_obs / max(before, 1), 1),
        "genes_detected": int(adata.n_vars),
        "median_genes_per_nucleus": float(np.median(adata.obs["n_genes_by_counts"])) if adata.n_obs else np.nan,
        "median_umi_per_nucleus": float(np.median(adata.obs["total_counts"])) if adata.n_obs else np.nan,
        "median_pct_mt": float(np.median(adata.obs["pct_counts_mt"])) if adata.n_obs else np.nan,
    }
    return adata, summary


def normalize(adata: sc.AnnData) -> sc.AnnData:
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


# ============================================================
# CELL-TYPE ANNOTATION
# ============================================================

def _score(adata: sc.AnnData, genes: list[str], name: str) -> list[str]:
    present = [g for g in genes if g in adata.var_names]
    if not present:
        adata.obs[name] = np.nan
        return []
    n_bins = 25 if adata.n_vars > 2000 else 10
    ctrl = min(50, max(5, adata.n_vars // 20))
    sc.tl.score_genes(adata, present, score_name=name, ctrl_size=ctrl, n_bins=n_bins, use_raw=False)
    return present


def annotate_cell_types(adata: sc.AnnData, min_score: float = 0.1) -> pd.DataFrame:
    """Marker-score annotation: each nucleus gets the highest-scoring cell type,
    or 'Unassigned' when no marker set scores above `min_score`."""
    cols, coverage = [], {}
    for ct, markers in CELL_TYPE_MARKERS.items():
        col = f"ct__{ct}"
        coverage[ct] = _score(adata, markers, col)
        if coverage[ct]:
            cols.append(col)
    if not cols:
        adata.obs["cell_type"] = "Unassigned"
        return pd.DataFrame()
    scores = adata.obs[cols].to_numpy(dtype=float)
    best = np.nanargmax(scores, axis=1)
    best_val = scores[np.arange(len(best)), best]
    labels = np.array([c.replace("ct__", "") for c in cols])[best]
    labels = np.where(best_val >= min_score, labels, "Unassigned")
    adata.obs["cell_type"] = pd.Categorical(labels)
    comp = adata.obs["cell_type"].value_counts().rename_axis("Cell type").reset_index(name="Nuclei")
    comp["Percent"] = (100 * comp["Nuclei"] / comp["Nuclei"].sum()).round(2)
    comp["Markers found"] = comp["Cell type"].map(lambda c: ", ".join(coverage.get(c, [])))
    return comp


# ============================================================
# NEUROVASCULAR PROGRAM SCORING
# ============================================================

def score_programs(adata: sc.AnnData) -> pd.DataFrame:
    rows = []
    for prog, spec in PROGRAMS.items():
        col = f"prog__{prog}"
        present = _score(adata, spec["genes"], col)
        rows.append({
            "Program": prog,
            "Genes found": len(present),
            "Genes expected": len(spec["genes"]),
            "Missing": ", ".join(g for g in spec["genes"] if g not in present) or "-",
        })
    return pd.DataFrame(rows)


def program_by_cell_type(adata: sc.AnnData) -> pd.DataFrame:
    """Mean program score per cell type (long format)."""
    prog_cols = [f"prog__{p}" for p in PROGRAMS if f"prog__{p}" in adata.obs]
    df = adata.obs[["cell_type", *prog_cols]].copy()
    long = df.melt(id_vars="cell_type", var_name="Program", value_name="Score")
    long["Program"] = long["Program"].str.replace("prog__", "", regex=False)
    out = (long.groupby(["cell_type", "Program"], observed=True)["Score"]
           .agg(["mean", "median", "count"]).reset_index()
           .rename(columns={"cell_type": "Cell type", "mean": "Mean", "median": "Median", "count": "Nuclei"}))
    return out


def barrier_integrity_index(adata: sc.AnnData) -> pd.Series:
    """Per-nucleus Barrier Integrity Index (BII) for endothelial nuclei:
    the average of junction and transport scores minus activation, in the
    same units as the program scores so it can be compared across samples.
    Higher = more barrier-supporting expression. Exploratory composite."""
    endo = adata.obs["cell_type"] == "Endothelial"
    bii = pd.Series(np.nan, index=adata.obs_names, name="BII")
    if endo.sum() < 10:
        return bii
    total = np.zeros(int(endo.sum()))
    used = 0
    for prog, sign in BII_COMPONENTS.items():
        col = f"prog__{prog}"
        if col not in adata.obs or adata.obs[col].isna().all():
            continue
        total += sign * adata.obs.loc[endo, col].to_numpy(dtype=float)
        used += 1
    if used:
        bii.loc[endo] = total / used
    return bii


# ============================================================
# FULL PER-SAMPLE PIPELINE
# ============================================================

@dataclass
class SampleResult:
    sample_id: str
    condition: str
    qc: dict
    composition: pd.DataFrame
    coverage: pd.DataFrame
    by_cell_type: pd.DataFrame
    features: dict = field(default_factory=dict)
    obs: pd.DataFrame | None = None


def process_sample(adata: sc.AnnData, sample_id: str, condition: str,
                   qc_params: QCParams, min_ct_score: float = 0.1,
                   keep_obs: bool = True) -> SampleResult:
    adata, qc = run_qc(adata, qc_params)
    if adata.n_obs < 20:
        raise ValueError(f"{sample_id}: only {adata.n_obs} nuclei passed QC.")
    adata = normalize(adata)
    composition = annotate_cell_types(adata, min_score=min_ct_score)
    coverage = score_programs(adata)
    by_ct = program_by_cell_type(adata)
    adata.obs["BII"] = barrier_integrity_index(adata)

    # Sample-level feature vector (used for cohort stats and the classifier)
    feats: dict[str, float] = {}
    for ct in ("Endothelial", "Pericyte"):
        sub = by_ct[by_ct["Cell type"] == ct]
        for _, r in sub.iterrows():
            feats[f"{ct} | {r['Program']}"] = float(r["Mean"])
    n_total = max(int(composition["Nuclei"].sum()), 1) if not composition.empty else 1
    for ct in VASCULAR_CELL_TYPES:
        n = int(composition.loc[composition["Cell type"] == ct, "Nuclei"].sum()) if not composition.empty else 0
        feats[f"Fraction | {ct}"] = n / n_total
    n_endo = int((adata.obs["cell_type"] == "Endothelial").sum())
    n_peri = int((adata.obs["cell_type"] == "Pericyte").sum())
    feats["Ratio | Pericyte : Endothelial"] = n_peri / n_endo if n_endo else np.nan
    feats["Endothelial | Barrier Integrity Index"] = float(np.nanmean(adata.obs["BII"])) if n_endo >= 10 else np.nan

    obs_cols = ["cell_type", "BII", "n_genes_by_counts", "total_counts", "pct_counts_mt",
                *[c for c in adata.obs.columns if c.startswith("prog__")]]
    return SampleResult(
        sample_id=sample_id, condition=condition, qc=qc, composition=composition,
        coverage=coverage, by_cell_type=by_ct, features=feats,
        obs=adata.obs[obs_cols].copy() if keep_obs else None,
    )


# ============================================================
# COHORT STATISTICS
# ============================================================

def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(pv)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def feature_table(results: list[SampleResult]) -> pd.DataFrame:
    rows = [{"Sample": r.sample_id, "Condition": r.condition, **r.features} for r in results]
    return pd.DataFrame(rows).set_index("Sample")


def compare_groups(ft: pd.DataFrame, a: str = "AD", b: str = "Control") -> pd.DataFrame:
    """Mann-Whitney U per feature, rank-biserial effect size, BH-FDR."""
    ga, gb = ft[ft["Condition"] == a], ft[ft["Condition"] == b]
    rows = []
    for col in ft.columns:
        if col == "Condition":
            continue
        x, y = ga[col].dropna().to_numpy(), gb[col].dropna().to_numpy()
        if len(x) < 2 or len(y) < 2:
            continue
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rbc = 2 * u / (len(x) * len(y)) - 1  # > 0 means higher in group a
        rows.append({
            "Feature": col,
            f"Mean {a}": x.mean(), f"Mean {b}": y.mean(),
            "Difference": x.mean() - y.mean(),
            "Effect size (rank-biserial)": rbc,
            "p-value": p, "n " + a: len(x), "n " + b: len(y),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["FDR q-value"] = _bh(df["p-value"].to_numpy())
        df = df.sort_values("p-value").reset_index(drop=True)
    return df


def nucleus_level_test(results: list[SampleResult], cell_type: str, program: str,
                       a: str = "AD", b: str = "Control") -> dict:
    """Pseudobulk-aware summary: per-sample means compared across groups."""
    vals = {a: [], b: []}
    col = f"prog__{program}"
    for r in results:
        if r.obs is None or r.condition not in vals:
            continue
        v = r.obs.loc[r.obs["cell_type"] == cell_type, col]
        if len(v) >= 5:
            vals[r.condition].append(float(v.mean()))
    if len(vals[a]) < 2 or len(vals[b]) < 2:
        return {}
    u, p = stats.mannwhitneyu(vals[a], vals[b], alternative="two-sided")
    return {"a": vals[a], "b": vals[b], "p": p}


# ============================================================
# EXPLORATORY CLASSIFIER (leave-one-out, permutation-tested)
# ============================================================

def run_classifier(ft: pd.DataFrame, features: list[str], n_perm: int = 500,
                   seed: int = 0) -> dict:
    """L2 logistic regression on per-sample features with leave-one-out CV.
    Reports out-of-sample AUC and a label-permutation p-value. With small
    cohorts this is exploratory: it asks whether vascular programs carry a
    disease-associated signal, not whether they can diagnose anyone."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = ft[ft["Condition"].isin(["AD", "Control"])][["Condition", *features]].dropna()
    y = (d["Condition"] == "AD").astype(int).to_numpy()
    X = d[features].to_numpy(dtype=float)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
        raise ValueError("Need at least 3 AD and 3 Control samples with complete features.")

    def loo_scores(X, y):
        pred = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(X):
            if len(np.unique(y[tr])) < 2:
                pred[te] = 0.5
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        return pred

    pred = loo_scores(X, y)
    auc = roc_auc_score(y, pred)
    rng = np.random.default_rng(seed)
    null = np.array([roc_auc_score(yp, loo_scores(X, yp)) for yp in (rng.permutation(y) for _ in range(n_perm))])
    p_perm = (1 + np.sum(null >= auc)) / (1 + n_perm)

    full = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    coefs = pd.DataFrame({"Feature": features, "Coefficient": full[-1].coef_[0]}).sort_values(
        "Coefficient", key=np.abs, ascending=False)
    fpr, tpr, _ = roc_curve(y, pred)
    return {
        "auc": float(auc), "p_perm": float(p_perm), "null": null,
        "pred": pd.DataFrame({"Sample": d.index, "Condition": d["Condition"].values, "LOO probability AD": pred}),
        "coefs": coefs, "fpr": fpr, "tpr": tpr, "n_ad": int(y.sum()), "n_ctrl": int((1 - y).sum()),
    }


# ============================================================
# SYNTHETIC DEMO DATA
# ============================================================

def make_demo_sample(seed: int, condition: str, n_cells: int = 1500) -> sc.AnnData:
    """Synthetic brain-vascular snRNA-seq sample for demos and testing.
    AD samples carry lower junction/transport/pericyte programs and higher
    endothelial activation in vascular cells. NOT real data."""
    rng = np.random.default_rng(seed)
    marker_genes = sorted({g for v in CELL_TYPE_MARKERS.values() for g in v})
    prog_genes = sorted({g for v in PROGRAMS.values() for g in v["genes"]})
    filler = [f"GENE{i:04d}" for i in range(1200)]
    mt = [f"MT-ND{i}" for i in range(1, 6)]
    genes = list(dict.fromkeys(marker_genes + prog_genes + filler + mt))
    gi = {g: i for i, g in enumerate(genes)}

    props = {"Endothelial": .28, "Pericyte": .14, "Smooth muscle": .07, "Fibroblast": .06,
             "Astrocyte": .15, "Microglia": .1, "Oligodendrocyte": .12, "Neuron": .08}
    if condition == "AD":
        props["Pericyte"] *= 0.7
    types = rng.choice(list(props), size=n_cells, p=np.array(list(props.values())) / sum(props.values()))

    base = rng.gamma(0.4, 0.6, size=len(genes))
    lam = np.tile(base, (n_cells, 1))
    for ct, markers in CELL_TYPE_MARKERS.items():
        idx = types == ct
        for g in markers:
            lam[idx, gi[g]] += 6.0
    vascular = np.isin(types, ["Endothelial", "Pericyte"])
    for prog, spec in PROGRAMS.items():
        for g in spec["genes"]:
            lam[vascular, gi[g]] += 2.5
    if condition == "AD":
        endo = types == "Endothelial"
        for g in PROGRAMS["Tight junctions"]["genes"] + PROGRAMS["Nutrient and efflux transport"]["genes"]:
            lam[endo, gi[g]] *= 0.55
        for g in PROGRAMS["Endothelial activation"]["genes"]:
            lam[endo, gi[g]] *= 1.9
        peri = types == "Pericyte"
        for g in PROGRAMS["Pericyte support"]["genes"]:
            lam[peri, gi[g]] *= 0.6
    lam *= rng.lognormal(0, 0.25, size=(n_cells, 1)) * (1 + rng.normal(0, 0.08))
    counts = rng.poisson(lam).astype(np.float32)
    for g in mt:
        counts[:, gi[g]] = rng.poisson(0.3, n_cells)
    ad = sc.AnnData(sparse.csr_matrix(counts))
    ad.var_names = genes
    ad.obs_names = [f"cell{i}" for i in range(n_cells)]
    return ad


def demo_cohort(n_ad: int = 6, n_ctrl: int = 6) -> list[tuple[str, str, sc.AnnData]]:
    out = []
    for i in range(n_ad):
        out.append((f"DEMO_{i+1:02d}_AD", "AD", make_demo_sample(100 + i, "AD")))
    for i in range(n_ctrl):
        out.append((f"DEMO_{i+1:02d}_C", "Control", make_demo_sample(200 + i, "Control")))
    return out
