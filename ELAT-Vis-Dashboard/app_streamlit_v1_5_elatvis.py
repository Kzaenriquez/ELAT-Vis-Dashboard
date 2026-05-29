#!/usr/bin/env python3
"""
app_streamlit_v1_6_elatvis.py

Streamlit MVP dashboard for ELAT-Vis.

GitHub/Streamlit Cloud layout supported:
    repo/
      ELAT-Vis-Dashboard/
        app_streamlit_v1_6_elatvis.py
        figures_*.py
        summary_tables_v1_elatvis.py
      data/
        <run>/
          parsed/
            timeline_table.csv
            states_table.csv
            ...
          figures/                  # optional
            Figure_100.png

Run locally:
    python -m streamlit run app_streamlit_v1_6_elatvis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

DEFAULT_MODULE_DIR = Path("/mnt/d/Thesis Masters/thesis-runs/data_viz/final")
THIS_DIR = Path(__file__).resolve().parent
APP_ROOT = THIS_DIR

# Handles both layouts:
#   repo/app_streamlit_v1_6_elatvis.py + repo/data/
#   repo/ELAT-Vis-Dashboard/app_streamlit_v1_6_elatvis.py + repo/data/
REPO_ROOT = THIS_DIR.parent if (THIS_DIR.parent / "data").exists() else THIS_DIR

DATA_ROOT_CANDIDATES = [
    THIS_DIR / "data",
    THIS_DIR.parent / "data",
    Path.cwd() / "data",
]

DATA_ROOTS = []
for _root in DATA_ROOT_CANDIDATES:
    try:
        _resolved = _root.resolve()
    except Exception:
        _resolved = _root
    if _root.exists() and _resolved not in DATA_ROOTS:
        DATA_ROOTS.append(_resolved)

for p in [THIS_DIR, DEFAULT_MODULE_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def resolve_dashboard_path(path_str: str) -> Path:
    """
    Resolve paths robustly for local use, GitHub repo layout, and Streamlit Cloud.

    Accepts:
      data/merge_extinction_ela_5ROI/parsed
      /data/merge_extinction_ela_5ROI/parsed
      ELAT-Vis-Dashboard/...
      absolute local paths
    """
    raw = str(path_str).strip().strip('"').strip("'")
    p = Path(raw)

    # Already valid absolute or relative to current working directory.
    if p.exists():
        return p.resolve()

    # Fix accidental leading slash in Streamlit sidebar: /data/... -> data/...
    if raw.startswith("/data/"):
        raw_no_slash = raw.lstrip("/")
        for base in [THIS_DIR, THIS_DIR.parent, Path.cwd()]:
            q = base / raw_no_slash
            if q.exists():
                return q.resolve()

    # Try relative to app folder, repo parent, and current working directory.
    for base in [THIS_DIR, THIS_DIR.parent, Path.cwd()]:
        q = base / raw
        if q.exists():
            return q.resolve()

    return p


def discover_parsed_dirs() -> list[Path]:
    """
    Find parsed directories under all known data/ folders.
    A valid parsed directory must contain timeline_table.csv.
    """
    candidates = []
    seen = set()

    for data_root in DATA_ROOTS:
        for p in data_root.rglob("parsed"):
            if (p / "timeline_table.csv").exists():
                rp = p.resolve()
                if rp not in seen:
                    candidates.append(rp)
                    seen.add(rp)

    return sorted(candidates, key=lambda x: str(x).lower())


def pretty_parsed_label(p: Path) -> str:
    """
    Readable dropdown label.

    Example:
      /repo/data/merge_extinction_ela_5ROI/parsed
      -> data/merge_extinction_ela_5ROI
    """
    p = Path(p)
    for base in [THIS_DIR, THIS_DIR.parent, Path.cwd()]:
        try:
            rel = p.relative_to(base)
            if rel.name == "parsed":
                return str(rel.parent)
            return str(rel)
        except Exception:
            pass

    if p.name == "parsed":
        return p.parent.name
    return str(p)


# Import generated ELAT-Vis modules.
# Keep these imports here so Streamlit fails early if a required file is missing.
try:
    import figures_timeline_v4_elatvis as timeline_mod
except Exception as e:
    timeline_mod = None
    TIMELINE_IMPORT_ERROR = e
else:
    TIMELINE_IMPORT_ERROR = None

try:
    import figures_node_map_v2_1_elatvis as node_mod
except Exception as e:
    node_mod = None
    NODE_IMPORT_ERROR = e
else:
    NODE_IMPORT_ERROR = None

try:
    import figures_radial_explorer_v1_5_elatvis as radial_mod
except Exception as e:
    radial_mod = None
    RADIAL_IMPORT_ERROR = e
else:
    RADIAL_IMPORT_ERROR = None

try:
    import summary_tables_v1_elatvis as summary_mod
except Exception as e:
    summary_mod = None
    SUMMARY_IMPORT_ERROR = e
else:
    SUMMARY_IMPORT_ERROR = None

try:
    import figures_sankey_precs_v1_2_elatvis as sankey_mod
except Exception as e:
    sankey_mod = None
    SANKEY_IMPORT_ERROR = e
else:
    SANKEY_IMPORT_ERROR = None


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ELAT-Vis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_parsed_basic(parsed_dir_str: str) -> Dict[str, object]:
    parsed_dir = resolve_dashboard_path(parsed_dir_str)
    out: Dict[str, object] = {
        "parsed_dir": parsed_dir,
        "exists": parsed_dir.exists(),
        "timeline": None,
        "states": None,
        "basins": None,
        "metadata": {},
        "summary_dir": parsed_dir / "summary_tables",
    }

    if not parsed_dir.exists():
        return out

    out["timeline"] = read_csv_if_exists(parsed_dir / "timeline_table.csv")
    out["states"] = read_csv_if_exists(parsed_dir / "states_table.csv")
    out["basins"] = read_csv_if_exists(parsed_dir / "basins_table.csv")

    meta_path = parsed_dir / "model_metadata.json"
    if meta_path.exists():
        try:
            out["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            out["metadata"] = {}

    return out


def available_values(df: Optional[pd.DataFrame], col: str) -> List[str]:
    if df is None or col not in df.columns:
        return []
    vals = sorted(df[col].dropna().astype(str).unique().tolist())
    return vals


def available_event_cols(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    cols = []
    for c in df.columns:
        lc = c.lower()
        if "trial_type" in lc or lc.startswith("event") or "stimulus_type" in lc:
            cols.append(c)
    # Put common HRF columns first.
    priority = ["trial_type", "trial_type_hrf2", "trial_type_hrf4", "trial_type_hrf6"]
    cols = sorted(cols, key=lambda x: (priority.index(x) if x in priority else 999, x))
    return cols


def choose_default_event_col(cols: List[str]) -> str:
    for c in ["trial_type_hrf4", "trial_type_hrf2", "trial_type", "trial_type_hrf6"]:
        if c in cols:
            return c
    return cols[0] if cols else "trial_type_hrf4"


def sidebar_select_optional(label: str, options: List[str], key: str, include_all: bool = True) -> Optional[str]:
    opts = ["All"] + options if include_all else options
    if not opts:
        return None
    val = st.sidebar.selectbox(label, opts, key=key)
    if val == "All":
        return None
    return val


def show_import_status():
    problems = []
    if TIMELINE_IMPORT_ERROR is not None:
        problems.append(f"Timeline module: {TIMELINE_IMPORT_ERROR}")
    if NODE_IMPORT_ERROR is not None:
        problems.append(f"Node map module: {NODE_IMPORT_ERROR}")
    if RADIAL_IMPORT_ERROR is not None:
        problems.append(f"Radial explorer module: {RADIAL_IMPORT_ERROR}")
    if SUMMARY_IMPORT_ERROR is not None:
        problems.append(f"Summary tables module: {SUMMARY_IMPORT_ERROR}")
    if SANKEY_IMPORT_ERROR is not None:
        problems.append(f"Pre-CS Sankey module: {SANKEY_IMPORT_ERROR}")

    if problems:
        st.error("Some generated modules could not be imported.")
        for p in problems:
            st.code(p)
        st.stop()


def display_table_file(path: Path, max_rows: int = 200):
    if not path.exists():
        st.warning(f"Missing table: {path.name}")
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")
        return

    st.caption(f"{path.name} — {len(df):,} rows × {len(df.columns):,} columns")
    st.dataframe(df.head(max_rows), use_container_width=True, height=420)


def find_elat_figure_search_dirs(parsed_dir: Path) -> List[Path]:
    """
    Candidate folders for ELAT figure exports.

    Works when only parsed/ was uploaded by searching:
      parsed/
      parsed/figures/
      parent/
      parent/figures/

    Recommended repo layout for figure previews:
      data/<run>/parsed/figures/Figure_100.png
    or:
      data/<run>/figures/Figure_100.png
    """
    parsed_dir = resolve_dashboard_path(str(parsed_dir))
    candidates = []

    if parsed_dir.name.lower() == "parsed":
        candidates.extend([
            parsed_dir / "figures",
            parsed_dir,
            parsed_dir.parent / "figures",
            parsed_dir.parent,
        ])
    else:
        candidates.extend([
            parsed_dir / "figures",
            parsed_dir,
        ])

    out = []
    seen = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if p.exists() and rp not in seen:
            out.append(rp)
            seen.add(rp)
    return out


def list_elat_figure_files(search_dirs: List[Path]) -> List[Path]:
    """
    Find displayable ELAT figure outputs in one or more search folders.

    Prioritizes Figure_100–Figure_103, then falls back to common image/html files.
    """
    allowed = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".html", ".htm", ".pdf"}
    candidates = []

    for fig_dir in search_dirs:
        if not fig_dir.exists():
            continue

        for pat in [
            "Figure_100*", "Figure_101*", "Figure_102*", "Figure_103*",
            "figure_100*", "figure_101*", "figure_102*", "figure_103*",
        ]:
            candidates.extend([p for p in fig_dir.glob(pat) if p.suffix.lower() in allowed])

    if not candidates:
        for fig_dir in search_dirs:
            if not fig_dir.exists():
                continue
            for pat in ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.webp", "*.html", "*.htm", "*.pdf"]:
                candidates.extend([p for p in fig_dir.glob(pat) if p.suffix.lower() in allowed])

    seen = set()
    out = []
    for p in sorted(candidates, key=lambda x: (str(x.parent).lower(), x.name.lower())):
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            out.append(rp)
            seen.add(rp)

    return out


def render_figure_file(path: Path, height: int = 720):
    """Render an ELAT figure file in Streamlit when possible."""
    suffix = path.suffix.lower()
    st.caption(str(path))

    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        st.image(str(path), use_container_width=True)
    elif suffix in {".html", ".htm"}:
        try:
            html = path.read_text(encoding="utf-8", errors="ignore")
            components.html(html, height=height, scrolling=True)
        except Exception as e:
            st.error(f"Could not render HTML figure: {e}")
    elif suffix == ".pdf":
        st.info("PDF preview is not embedded in this MVP. Use the download button below.")
    else:
        st.warning(f"Unsupported figure type: {suffix}")

    try:
        data = path.read_bytes()
        st.download_button(
            label=f"Download {path.name}",
            data=data,
            file_name=path.name,
            mime="application/octet-stream",
            key=f"download_{path.name}_{path.stat().st_mtime_ns}",
        )
    except Exception:
        pass


def sort_existing(df: pd.DataFrame, by: str, ascending: bool = False) -> pd.DataFrame:
    if df is None or df.empty or by not in df.columns:
        return df
    out = df.copy()
    out[by] = pd.to_numeric(out[by], errors="coerce")
    return out.sort_values(by, ascending=ascending)


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def display_node_highlights(
    cohort: str,
    state_df: pd.DataFrame,
    lm_df: pd.DataFrame,
    backbone_df: pd.DataFrame,
    delta_state_df: pd.DataFrame,
    delta_transition_df: pd.DataFrame,
):
    """Show immediately interpretable highlights beside/below the node map."""
    st.subheader("Node map highlights")
    st.caption("Highest-value rows extracted from the same companion tables used by this node-map panel.")

    if cohort == "PTSD-minus-HC":
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Largest state differences")
            df = delta_state_df.copy()
            if not df.empty:
                if "abs_cs_delta_difference" not in df.columns and "PTSD_minus_HC_cs_delta" in df.columns:
                    df["abs_cs_delta_difference"] = pd.to_numeric(df["PTSD_minus_HC_cs_delta"], errors="coerce").abs()
                df = sort_existing(df, "abs_cs_delta_difference", ascending=False)
                cols = [
                    "state", "basin", "basin_label_readable", "is_local_minimum",
                    "cs_delta_PTSD", "cs_delta_HC", "PTSD_minus_HC_cs_delta",
                    "occupancy_prop_PTSD", "occupancy_prop_HC",
                    "enriched_event_PTSD", "enriched_event_HC",
                    "abs_cs_delta_difference",
                ]
                st.dataframe(format_node_table(keep_existing_cols(df, cols)).head(10), use_container_width=True, height=300)
            else:
                st.info("No group-delta state table available.")

        with c2:
            st.markdown("#### Largest transition differences")
            df = delta_transition_df.copy()
            if not df.empty:
                if "abs_probability_difference" not in df.columns and "PTSD_minus_HC_probability" in df.columns:
                    df["abs_probability_difference"] = pd.to_numeric(df["PTSD_minus_HC_probability"], errors="coerce").abs()
                df = sort_existing(df, "abs_probability_difference", ascending=False)
                cols = [
                    "source", "target", "source_basin", "target_basin",
                    "source_basin_label", "target_basin_label",
                    "probability_PTSD", "probability_HC", "PTSD_minus_HC_probability",
                    "count_PTSD", "count_HC", "abs_probability_difference",
                ]
                st.dataframe(format_node_table(keep_existing_cols(df, cols)).head(10), use_container_width=True, height=300)
            else:
                st.info("No group-delta transition table available.")
        return

    # Standard cohort views.
    show_state_df = state_df.copy()
    if "cohort" in show_state_df.columns and cohort != "merged":
        show_state_df = show_state_df[show_state_df["cohort"].astype(str) == cohort].copy()
    elif "cohort" in show_state_df.columns and cohort == "merged":
        show_state_df = show_state_df[show_state_df["cohort"].astype(str) == "merged"].copy()

    show_lm_df = lm_df.copy()
    if "cohort" in show_lm_df.columns and cohort != "merged":
        show_lm_df = show_lm_df[show_lm_df["cohort"].astype(str) == cohort].copy()
    elif "cohort" in show_lm_df.columns and cohort == "merged":
        show_lm_df = show_lm_df[show_lm_df["cohort"].astype(str) == "merged"].copy()

    show_backbone_df = backbone_df.copy()
    if "cohort" in show_backbone_df.columns and cohort != "merged":
        show_backbone_df = show_backbone_df[show_backbone_df["cohort"].astype(str) == cohort].copy()
    elif "cohort" in show_backbone_df.columns and cohort == "merged":
        show_backbone_df = show_backbone_df[show_backbone_df["cohort"].astype(str) == "merged"].copy()

    c1, c2, c3 = st.columns(3)

    state_cols = [
        "state", "basin", "basin_label_readable", "is_local_minimum",
        "occupancy", "occupancy_prop", "cs_delta",
        "prop_CSplus", "prop_CSminus", "prop_ITI",
        "enriched_event", "max_enrichment", "binary_01", "sigma_pattern",
    ]

    with c1:
        st.markdown("#### Most CS+ shifted states")
        df = sort_existing(show_state_df, "cs_delta", ascending=False)
        st.dataframe(format_node_table(keep_existing_cols(df, state_cols)).head(8), use_container_width=True, height=280)

    with c2:
        st.markdown("#### Most CS− shifted states")
        df = sort_existing(show_state_df, "cs_delta", ascending=True)
        st.dataframe(format_node_table(keep_existing_cols(df, state_cols)).head(8), use_container_width=True, height=280)

    with c3:
        st.markdown("#### Strongest transition backbone")
        df = show_backbone_df.copy()
        sort_col = "probability" if "probability" in df.columns else "count"
        df = sort_existing(df, sort_col, ascending=False)
        cols = [
            "source", "target", "source_basin", "target_basin",
            "source_basin_label", "target_basin_label",
            "count", "probability", "subject_support_count", "subject_support_prop",
        ]
        st.dataframe(format_node_table(keep_existing_cols(df, cols)).head(8), use_container_width=True, height=280)

    st.markdown("#### Local minima / attractor summary")
    lm_cols = [
        "state", "basin", "basin_label_readable", "occupancy", "occupancy_prop",
        "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI",
        "enriched_event", "max_enrichment", "binary_01", "sigma_pattern",
    ]
    st.dataframe(format_node_table(keep_existing_cols(show_lm_df, lm_cols)), use_container_width=True, height=210)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def keep_existing_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in cols if c in df.columns]].copy()


def format_node_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        if any(tok in c.lower() for tok in ["prop", "delta", "enrich", "probability"]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(4)
    return out


def display_companion_dataframe(title: str, df: pd.DataFrame, height: int = 300, max_rows: int = 100):
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info("No rows available for the current filters.")
        return
    st.caption(f"{len(df):,} rows × {len(df.columns):,} columns")
    st.dataframe(format_node_table(df).head(max_rows), use_container_width=True, height=height)


def filter_df_for_overview(df: pd.DataFrame, group=None, subject=None, task=None, session=None) -> pd.DataFrame:
    out = df.copy()
    for col, val in [("group", group), ("subject", subject), ("task", task), ("session", session)]:
        if val and col in out.columns:
            out = out[out[col].astype(str) == str(val)]
    return out


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

show_import_status()

st.title("ELAT-Vis Dashboard")
st.caption("Portable visual analytics dashboard for Ezaki ELAT outputs, task/event alignment, and ELA summary tables.")

st.sidebar.header("Input")

available_parsed_dirs = discover_parsed_dirs()

if available_parsed_dirs:
    parsed_options = {pretty_parsed_label(p): p for p in available_parsed_dirs}

    selected_label = st.sidebar.selectbox(
        "Available parsed dataset",
        list(parsed_options.keys()),
        index=0,
        help="Automatically detected parsed folders under data/.",
    )

    parsed_dir = parsed_options[selected_label]

    with st.sidebar.expander("Manual path override", expanded=False):
        manual_path = st.text_input(
            "Manual parsed directory",
            value=pretty_parsed_label(parsed_dir),
            help="Use this only if the dataset is not detected automatically.",
        )
        use_manual = st.checkbox("Use manual path", value=False)
        if use_manual:
            parsed_dir = resolve_dashboard_path(manual_path)
else:
    st.sidebar.warning("No parsed datasets detected under data/. Use manual path.")
    manual_path = st.sidebar.text_input(
        "Parsed ELAT directory",
        value="data/merge_extinction_ela_5ROI/parsed",
        help="Use a relative path such as data/merge_extinction_ela_5ROI/parsed. Do not start with /data.",
    )
    parsed_dir = resolve_dashboard_path(manual_path)

parsed_dir_str = str(parsed_dir)
basic = load_parsed_basic(parsed_dir_str)

if not basic["exists"]:
    st.error("Parsed directory does not exist. Check the selected dataset or manual path.")
    st.code(str(parsed_dir))
    st.stop()

timeline_df: Optional[pd.DataFrame] = basic["timeline"]  # type: ignore
states_df: Optional[pd.DataFrame] = basic["states"]  # type: ignore
basins_df: Optional[pd.DataFrame] = basic["basins"]  # type: ignore
metadata: dict = basic["metadata"] if isinstance(basic["metadata"], dict) else {}

if timeline_df is None:
    st.error("No timeline_table.csv found in the parsed directory.")
    st.stop()

event_cols = available_event_cols(timeline_df)
default_event_col = choose_default_event_col(event_cols)

st.sidebar.header("Global filters")
event_col = st.sidebar.selectbox(
    "Event column",
    event_cols if event_cols else [default_event_col],
    index=(event_cols.index(default_event_col) if default_event_col in event_cols else 0),
)
epoch_mode = st.sidebar.selectbox("Epoch mode", ["full", "early", "late"], index=0)
preset = st.sidebar.selectbox("Basin preset", ["fear4", "none"], index=0)
preset_arg = None if preset == "none" else preset

groups = available_values(timeline_df, "group")
subjects = available_values(timeline_df, "subject")
tasks = available_values(timeline_df, "task")
sessions = available_values(timeline_df, "session")

global_group = sidebar_select_optional("Group", groups, key="global_group")
global_task = sidebar_select_optional("Task", tasks, key="global_task")
global_session = sidebar_select_optional("Session", sessions, key="global_session")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tabs = st.tabs([
    "Overview",
    "Visualization Outline",
    "ELAT Figures",
    "Timeline",
    "Event-Enriched Node Map",
    "Pre-CS Sankey",
    "Radial Basin Explorer",
    "Summary Tables",
])


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------

with tabs[0]:
    st.subheader("Overview")

    c1, c2, c3, c4 = st.columns(4)
    filtered_overview = filter_df_for_overview(
        timeline_df,
        group=global_group,
        task=global_task,
        session=global_session,
    )

    with c1:
        st.metric("Timeline rows", f"{len(filtered_overview):,}")
    with c2:
        st.metric("Subjects", f"{filtered_overview['subject'].nunique() if 'subject' in filtered_overview.columns else 0:,}")
    with c3:
        st.metric("States", f"{states_df['state'].nunique() if states_df is not None and 'state' in states_df.columns else 'NA'}")
    with c4:
        st.metric("Basins", f"{basins_df['basin'].nunique() if basins_df is not None and 'basin' in basins_df.columns else (timeline_df['basin'].nunique() if 'basin' in timeline_df.columns else 'NA')}")

    st.markdown("### Parsed directory")
    st.code(str(parsed_dir))

    st.markdown("### Available filters")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write("Groups")
        st.write(groups or ["NA"])
    with c2:
        st.write("Tasks")
        st.write(tasks or ["NA"])
    with c3:
        st.write("Sessions")
        st.write(sessions or ["NA"])
    with c4:
        st.write("Event columns")
        st.write(event_cols or ["NA"])

    st.markdown("### Parser metadata")
    if metadata:
        st.json(metadata)
    else:
        st.info("No model_metadata.json found or metadata is empty.")

    st.markdown("### Available parsed files")
    file_rows = []
    for name in [
        "timeline_table.csv",
        "states_table.csv",
        "basins_table.csv",
        "basin_graph.csv",
        "state_basin_membership.csv",
        "feature_file_manifest.csv",
        "confirmatory_file_manifest.csv",
    ]:
        p = parsed_dir / name
        file_rows.append({"file": name, "exists": p.exists(), "path": str(p)})
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)



# ---------------------------------------------------------------------
# Visualization Outline tab
# ---------------------------------------------------------------------

with tabs[1]:
    st.subheader("Visualization outline")
    st.caption(
        "A guided interpretation page for domain experts reviewing ELAT-Vis. "
        "Use this first before opening the analytic visualization tabs."
    )

    st.markdown(
        """
        ## 1. Task and event definitions

        This dashboard visualizes Energy Landscape Analysis outputs from a contextual fear-learning task.
        The core event labels are:

        - **CS+**: the conditioned stimulus category that is treated as threat-associated because it was paired with the unconditioned stimulus/shock during acquisition.
        - **CS−**: the comparison/safety stimulus category that was not paired with shock.
        - **ITI**: inter-trial interval or non-CS period.

        The event columns such as `trial_type`, `trial_type_hrf2`, `trial_type_hrf4`, and `trial_type_hrf6`
        represent different temporal alignments between the task event and the fMRI signal.
        `trial_type_hrf4` is often used as the default exploratory alignment because the BOLD response is delayed relative to stimulus onset.
        """
    )

    task_table = pd.DataFrame(
        [
            {
                "Task": "Baseline",
                "What it represents": "Pre-learning or non-reinforced reference condition.",
                "What to inspect": "Baseline basin occupancy, dwell time, and global-off/global-on locking.",
            },
            {
                "Task": "Acquisition",
                "What it represents": "CS+ becomes associated with shock/threat; CS− remains non-reinforced.",
                "What to inspect": "CS+ recruitment of threat-like or global arousal basins; CS+ staying vs movement.",
            },
            {
                "Task": "Extinction",
                "What it represents": "CS+ is presented without shock, so threat expectation should weaken over time.",
                "What to inspect": "Early vs late changes in CS+ routing; safety-like recruitment; persistence of threat-like states.",
            },
            {
                "Task": "Renewal",
                "What it represents": "Context shift after extinction; threat/safety associations may re-emerge or reorganize.",
                "What to inspect": "CS− safety retrieval, CS+ global/threat-like recruitment, and group differences.",
            },
        ]
    )
    st.dataframe(task_table, use_container_width=True, hide_index=True)

    st.markdown("## 2. Expected CS+ and CS− patterns")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            ### Expected CS+ pattern

            During acquisition, **CS+** is expected to show greater recruitment of
            threat/salience-related basin patterns, especially states involving amygdala,
            insula, dACC, or high-engagement configurations.

            During extinction, CS+ should ideally become less threat-like over time,
            but residual threat-like routing may remain, especially in late extinction
            or in PTSD-related patterns.
            """
        )
    with c2:
        st.markdown(
            """
            ### Expected CS− pattern

            **CS−** is expected to behave more like a safety or comparison condition.
            It may show stronger recruitment of safety-like, context-regulatory, or
            non-threat basin patterns.

            If CS− fails to recruit safety-like basins, this may suggest weaker
            safety-specific organization rather than only stronger CS+ threat persistence.
            """
        )

    st.info(
        "Interpretation rule: CS+ and CS− effects should be read as task-linked basin/state associations, "
        "not as direct neural activation claims unless supported by the ROI pattern and statistical summaries."
    )

    st.markdown("## 3. Visualization tools in ELAT-Vis")

    viz_table = pd.DataFrame(
        [
            {
                "Tool": "ELAT Figures",
                "What it shows": "Original ELAT outputs such as basin graph, landscape, disconnectivity graph, and state matrix if figure exports are available.",
                "Best use": "Check the original landscape structure before task/event overlays.",
            },
            {
                "Tool": "Timeline",
                "What it shows": "Epoch strip, event-aligned basin table timeline, and ROI activation heatmap.",
                "Best use": "Inspect when each basin/state occurs across TRs for one subject or filtered run.",
            },
            {
                "Tool": "Event-Enriched Node Map",
                "What it shows": "State-space graph with node size as occupancy, border as basin, fill as CS delta/enrichment, and edges as transition backbone.",
                "Best use": "Identify event-associated states, local minima, and robust transitions.",
            },
            {
                "Tool": "Pre-CS Sankey",
                "What it shows": "Directional Pre-CS → During-CS → Post-CS event-window flows plus percentage and CS+−CS− difference charts.",
                "Best use": "Summarize how basins change around CS events.",
            },
            {
                "Tool": "Radial Basin Explorer",
                "What it shows": "Per-subject animated movement through basin/state space with recent ROI activation history.",
                "Best use": "Storytelling view for explaining one subject's trajectory.",
            },
            {
                "Tool": "Summary Tables",
                "What it shows": "State summaries, local minima, transitions, group deltas, and detected confirmatory/statistical files.",
                "Best use": "Verify numerical values behind visual impressions.",
            },
        ]
    )
    st.dataframe(viz_table, use_container_width=True, hide_index=True)

    st.markdown("## 4. How to adjust and manipulate the tools")

    with st.expander("Global controls", expanded=True):
        st.markdown(
            """
            - **Available parsed dataset**: selects which ELA run/task is loaded from the `data/` folder.
            - **Event column**: chooses the event alignment, for example `trial_type`, `trial_type_hrf2`, `trial_type_hrf4`, or `trial_type_hrf6`.
            - **Epoch mode**: selects `full`, `early`, or `late` rows when epoch labels are available.
            - **Group / Task / Session filters**: restrict the dashboard to HC, PTSD, a specific task, or session if those columns exist.
            """
        )

    with st.expander("Timeline controls", expanded=False):
        st.markdown(
            """
            - Select a **subject** to inspect a single subject's event-aligned trajectory.
            - Use **event column** to compare onset-aligned versus HRF-shifted interpretations.
            - Read basin rows categorically: B1–B4 are labels, not numeric heights.
            """
        )

    with st.expander("Node map controls", expanded=False):
        st.markdown(
            """
            - **Cohort/comparison**: view merged, HC, PTSD, or PTSD-minus-HC.
            - **Fill mode**:
              - `cs_delta`: CS+ proportion minus CS− proportion.
              - `enriched_event`: event overrepresented relative to global event frequency.
              - `iti_prop`: proportion of ITI occupancy.
            - **Transition filters**:
              - minimum transition count,
              - minimum transition probability,
              - minimum subject support,
              - top-k outgoing transitions per state.
            - Use the companion tables to inspect exact transition probabilities and state summaries.
            """
        )

    with st.expander("Sankey controls", expanded=False):
        st.markdown(
            """
            - **Cohort**: merged, HC, or PTSD.
            - **Event filter**: CS+, CS−, or both.
            - **Flow level**: basin-level is cleaner; state-level is more detailed but often crowded.
            - **Pre/Post offset**: number of TRs before and after the CS-aligned anchor.
            - **Node/link color**: use basin coloring for basin interpretation; event coloring for CS+/CS− emphasis.
            - Read the percentage and CS+−CS− charts as the quantitative companion to the Sankey.
            """
        )

    with st.expander("Radial explorer controls", expanded=False):
        st.markdown(
            """
            - Requires a **subject** because task-event order is randomized per subject.
            - **Tail length** controls how much recent trajectory history is shown.
            - **Frame step** and **frame duration** control animation smoothness and speed.
            - The current radial layout uses descent-graph distance from local minima, not actual energy height.
            """
        )

    with st.expander("Interpretation caveats", expanded=True):
        st.markdown(
            """
            - The dashboard uses a shared/merged landscape as a **reference space** for visualization and comparison.
              Separate group/task landscapes should be used later as a robustness benchmark.
            - Individual state-level patterns are useful for exploration, but stronger inferential claims should focus on
              **basins, local minima, and aggregated ELA features**.
            - Transition backbones are **filtered observed transitions**, not automatically statistically significant edges.
            - Basin labels such as **threat-like** and **safety-like** are provisional and should be checked against ROI patterns
              and domain-expert feedback.
            """
        )

    st.markdown("## Suggested starting workflow")
    workflow_table = pd.DataFrame(
        [
            {"Step": 1, "Action": "Choose the parsed dataset/task.", "Reason": "Defines which ELA landscape and timeline are loaded."},
            {"Step": 2, "Action": "Open the Timeline tab.", "Reason": "Check event alignment, early/late labels, and ROI activation patterns."},
            {"Step": 3, "Action": "Open the Node Map tab.", "Reason": "Find CS+ or CS− shifted states and transition backbones."},
            {"Step": 4, "Action": "Open the Sankey tab.", "Reason": "Summarize Pre-CS → During-CS → Post-CS basin routes."},
            {"Step": 5, "Action": "Open Summary Tables.", "Reason": "Verify the exact counts, proportions, transition probabilities, and group-delta values."},
            {"Step": 6, "Action": "Use the Radial Explorer for examples.", "Reason": "Select a representative subject to explain movement through the landscape."},
        ]
    )
    st.dataframe(workflow_table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# ELAT Figures tab
# ---------------------------------------------------------------------

with tabs[2]:
    st.subheader("Original ELAT figures")
    st.caption(
        "Displays Figure_100–Figure_103 exports when available. "
        "This build searches inside the selected parsed folder first, so it works even when only parsed/ was uploaded."
    )

    figure_search_dirs = find_elat_figure_search_dirs(parsed_dir)

    with st.expander("Figure search locations", expanded=False):
        for d in figure_search_dirs:
            st.code(str(d))

        manual_fig_dir = st.text_input(
            "Optional manual figure directory",
            value="",
            help="Example: data/merge_extinction_ela_5ROI/parsed/figures",
        )
        if manual_fig_dir.strip():
            resolved_manual_fig_dir = resolve_dashboard_path(manual_fig_dir)
            if resolved_manual_fig_dir.exists():
                figure_search_dirs = [resolved_manual_fig_dir] + figure_search_dirs
            else:
                st.warning(f"Manual figure directory not found: {resolved_manual_fig_dir}")

    fig_files = list_elat_figure_files(figure_search_dirs)

    if not fig_files:
        st.warning("No displayable ELAT figure files found.")
        st.markdown(
            """
            This is expected if the GitHub repository only contains parsed CSV tables.

            To make this tab display ELAT figures, upload PNG/HTML exports into either:

            ```text
            data/<run>/parsed/figures/Figure_100.png
            data/<run>/parsed/figures/Figure_101.png
            data/<run>/parsed/figures/Figure_102.png
            data/<run>/parsed/figures/Figure_103.png
            ```

            or:

            ```text
            data/<run>/figures/Figure_100.png
            ```

            MATLAB `.fig` files will not preview directly in Streamlit; export them to PNG first.
            """
        )
    else:
        st.success(f"Found {len(fig_files)} displayable figure file(s).")
        view_mode = st.radio("View mode", ["One figure", "Gallery"], horizontal=True)
        if view_mode == "One figure":
            selected = st.selectbox(
                "Figure",
                fig_files,
                format_func=lambda p: f"{p.parent.name}/{p.name}",
            )
            render_figure_file(selected, height=760)
        else:
            cols = st.columns(2)
            for i, p in enumerate(fig_files):
                with cols[i % 2]:
                    st.markdown(f"#### {p.name}")
                    render_figure_file(p, height=520)


# ---------------------------------------------------------------------
# Timeline tab
# ---------------------------------------------------------------------

with tabs[3]:
    st.subheader("Timeline visualization")
    if timeline_mod is None:
        st.error(f"Timeline module unavailable: {TIMELINE_IMPORT_ERROR}")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            timeline_subject = sidebar_select_optional(
                "Timeline subject",
                subjects,
                key="timeline_subject",
                include_all=False,
            )
        with c2:
            show_all_event_cols = st.checkbox("Show all detected event columns", value=False)
        with c3:
            timeline_height = st.number_input("Timeline height", min_value=600, max_value=1800, value=900, step=50)

        try:
            tables = timeline_mod.load_parsed_tables(parsed_dir)
            tl = timeline_mod.get_base_timeline(tables)
            tl = timeline_mod.filter_timeline(
                tl,
                group=global_group,
                subject=timeline_subject,
                task=global_task,
                session=global_session,
            )
            states_for_tl = tables.get("states", pd.DataFrame())
            basins_for_tl = tables.get("basins", pd.DataFrame())

            fig = timeline_mod.make_elat_timeline_dashboard(
                timeline=tl,
                states=states_for_tl,
                basins=basins_for_tl,
                event_col=event_col,
                show_all_event_cols=show_all_event_cols,
                epoch_type="trialblock",
                preset=preset if preset != "none" else "default",
                basin_event_col=event_col,
                height=int(timeline_height),
                title=f"Timeline | {timeline_subject or 'selected rows'} | {event_col} | {epoch_mode}",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.exception(e)


# ---------------------------------------------------------------------
# Node map tab
# ---------------------------------------------------------------------

with tabs[4]:
    st.subheader("Event-enriched node map")

    if node_mod is None:
        st.error(f"Node map module unavailable: {NODE_IMPORT_ERROR}")
    else:
        with st.expander("Visual encoding", expanded=False):
            st.markdown(
                """
                - **Node** = binary ELAT state.
                - **Node size** = state occupancy.
                - **Node border** = basin identity.
                - **Node fill** = selected event encoding.
                - **Diamond** = local minimum.
                - **Solid edges** = observed transition backbone.
                - **Dotted edges** = ELAT descent reference, if enabled.

                The companion tables below show the numerical values behind the map:
                state occupancy, CS+/CS− proportions, enriched event, transition counts,
                transition probability, and subject support.
                """
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cohort = st.selectbox("Cohort / comparison", ["merged", "HC", "PTSD", "PTSD-minus-HC"], index=0)
        with c2:
            fill_mode = st.selectbox("Fill mode", ["cs_delta", "enriched_event", "iti_prop"], index=0)
        with c3:
            edge_mode = st.selectbox("Edge mode", ["both", "observed", "descent"], index=0)
        with c4:
            show_hulls = st.checkbox("Show basin hulls", value=False)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cs_delta_range = st.number_input("CS-delta range", min_value=0.05, max_value=2.0, value=0.3, step=0.05)
        with c2:
            min_transition_count = st.number_input("Min transition count", min_value=1, max_value=100, value=5, step=1)
        with c3:
            min_transition_prob = st.number_input("Min transition probability", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        with c4:
            min_subject_support_count = st.number_input("Min subject support count", min_value=0, max_value=100, value=10, step=1)

        c1, c2, c3 = st.columns(3)
        with c1:
            top_k_per_state = st.number_input("Top-k transitions per source", min_value=1, max_value=20, value=2, step=1)
        with c2:
            node_width = st.number_input("Figure width", min_value=900, max_value=2200, value=1450, step=50)
        with c3:
            node_height = st.number_input("Figure height", min_value=700, max_value=1800, value=920, step=50)

        analysis_mode = "group_delta" if cohort == "PTSD-minus-HC" else "standard"
        group_arg = None if cohort in ["merged", "PTSD-minus-HC"] else cohort

        try:
            fig = node_mod.make_event_enriched_node_map(
                parsed_dir=parsed_dir,
                group=group_arg,
                task=global_task,
                session=global_session,
                event_col=event_col,
                epoch_mode=epoch_mode,
                preset=preset_arg,
                fill_mode=fill_mode,
                analysis_mode=analysis_mode,
                color_range_mode="fixed",
                cs_delta_range=float(cs_delta_range),
                min_transition_count=int(min_transition_count),
                min_transition_prob=float(min_transition_prob),
                min_subject_support_count=int(min_subject_support_count),
                top_k_per_state=int(top_k_per_state),
                edge_mode=edge_mode,
                show_basin_hulls=show_hulls,
                width=int(node_width),
                height=int(node_height),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.exception(e)

        st.markdown("---")
        st.subheader("Node map companion tables")
        st.caption(
            "These tables are generated with the same event column, epoch mode, task/session filters, "
            "and transition thresholds used by the node-map panel. The transition backbone is a robust "
            "observed-transition filter, not inferential statistical significance."
        )

        node_summary_dir = parsed_dir / "summary_tables_node_panel"
        auto_refresh_node_tables = st.checkbox("Auto-refresh node companion tables", value=True)
        manual_refresh = st.button("Refresh node companion tables")

        if summary_mod is None:
            st.warning(f"Summary module unavailable: {SUMMARY_IMPORT_ERROR}")
        else:
            required_files = [
                node_summary_dir / "state_event_summary.csv",
                node_summary_dir / "transition_backbone_summary.csv",
                node_summary_dir / "transition_summary.csv",
                node_summary_dir / "local_minimum_summary.csv",
                node_summary_dir / "group_delta_state_summary.csv",
                node_summary_dir / "group_delta_transition_summary.csv",
            ]
            needs_refresh = auto_refresh_node_tables or manual_refresh or any(not p.exists() for p in required_files)

            if needs_refresh:
                try:
                    summary_mod.generate_summary_tables(
                        parsed_dir=parsed_dir,
                        out_dir=node_summary_dir,
                        event_col=event_col,
                        epoch_mode=epoch_mode,
                        group=None,
                        task=global_task,
                        session=global_session,
                        top_n=20,
                        min_transition_count=int(min_transition_count),
                        min_transition_prob=float(min_transition_prob),
                        min_subject_support_count=int(min_subject_support_count),
                        top_k_per_state=int(top_k_per_state),
                        preset=preset if preset != "none" else "fear4",
                    )
                except Exception as e:
                    st.exception(e)

            table_tabs = st.tabs([
                "States",
                "Local minima",
                "Transition backbone",
                "All transitions",
                "Group differences",
            ])

            state_df = read_table(node_summary_dir / "state_event_summary.csv")
            lm_df = read_table(node_summary_dir / "local_minimum_summary.csv")
            backbone_df = read_table(node_summary_dir / "transition_backbone_summary.csv")
            transitions_df = read_table(node_summary_dir / "transition_summary.csv")
            delta_state_df = read_table(node_summary_dir / "group_delta_state_summary.csv")
            delta_transition_df = read_table(node_summary_dir / "group_delta_transition_summary.csv")

            display_node_highlights(
                cohort=cohort,
                state_df=state_df,
                lm_df=lm_df,
                backbone_df=backbone_df,
                delta_state_df=delta_state_df,
                delta_transition_df=delta_transition_df,
            )

            st.markdown("### Detailed node-map tables")
            with table_tabs[0]:
                if cohort != "PTSD-minus-HC" and "cohort" in state_df.columns:
                    show_state_df = state_df[state_df["cohort"].astype(str) == cohort].copy()
                else:
                    show_state_df = state_df.copy()
                state_cols = [
                    "cohort", "state", "basin", "basin_label_readable", "is_local_minimum",
                    "occupancy", "occupancy_prop", "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI",
                    "enriched_event", "max_enrichment", "enrich_CSplus", "enrich_CSminus", "enrich_ITI",
                    "binary", "binary_01", "sigma_pattern",
                ]
                show_state_df = keep_existing_cols(show_state_df, state_cols)
                sort_col = "occupancy" if "occupancy" in show_state_df.columns else None
                if sort_col:
                    show_state_df = show_state_df.sort_values(sort_col, ascending=False)
                display_companion_dataframe("Observed state/event summary", show_state_df, height=360, max_rows=150)

            with table_tabs[1]:
                if cohort != "PTSD-minus-HC" and "cohort" in lm_df.columns:
                    show_lm_df = lm_df[lm_df["cohort"].astype(str) == cohort].copy()
                else:
                    show_lm_df = lm_df.copy()
                lm_cols = [
                    "cohort", "state", "basin", "basin_label_readable", "occupancy", "occupancy_prop",
                    "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI", "enriched_event",
                    "max_enrichment", "binary", "binary_01", "sigma_pattern",
                ]
                display_companion_dataframe("Local-minimum / attractor summary", keep_existing_cols(show_lm_df, lm_cols), height=260, max_rows=50)

            with table_tabs[2]:
                if cohort != "PTSD-minus-HC" and "cohort" in backbone_df.columns:
                    show_backbone_df = backbone_df[backbone_df["cohort"].astype(str) == cohort].copy()
                else:
                    show_backbone_df = backbone_df.copy()
                transition_cols = [
                    "cohort", "source", "target", "source_basin", "target_basin",
                    "source_basin_label", "target_basin_label", "count", "source_total", "probability",
                    "subject_support_count", "subject_support_prop", "most_common_source_event", "most_common_target_event",
                ]
                display_companion_dataframe("Filtered observed transition backbone", keep_existing_cols(show_backbone_df, transition_cols), height=360, max_rows=150)

            with table_tabs[3]:
                if cohort != "PTSD-minus-HC" and "cohort" in transitions_df.columns:
                    show_transitions_df = transitions_df[transitions_df["cohort"].astype(str) == cohort].copy()
                else:
                    show_transitions_df = transitions_df.copy()
                if "probability" in show_transitions_df.columns:
                    show_transitions_df = show_transitions_df.sort_values(["cohort", "probability", "count"], ascending=[True, False, False])
                transition_cols = [
                    "cohort", "source", "target", "source_basin", "target_basin",
                    "source_basin_label", "target_basin_label", "count", "source_total", "probability",
                    "subject_support_count", "subject_support_prop", "most_common_source_event", "most_common_target_event",
                ]
                display_companion_dataframe("All observed transitions", keep_existing_cols(show_transitions_df, transition_cols), height=360, max_rows=200)

            with table_tabs[4]:
                if cohort == "PTSD-minus-HC":
                    c1, c2 = st.columns(2)
                    with c1:
                        delta_state_cols = [
                            "state", "basin", "basin_label_readable", "is_local_minimum",
                            "cs_delta_PTSD", "cs_delta_HC", "PTSD_minus_HC_cs_delta",
                            "occupancy_prop_PTSD", "occupancy_prop_HC", "PTSD_minus_HC_occupancy_prop",
                            "enriched_event_PTSD", "enriched_event_HC", "binary", "binary_01", "sigma_pattern",
                            "abs_cs_delta_difference",
                        ]
                        display_companion_dataframe(
                            "State-level PTSD−HC differences",
                            keep_existing_cols(delta_state_df, delta_state_cols),
                            height=360,
                            max_rows=150,
                        )
                    with c2:
                        delta_transition_cols = [
                            "source", "target", "source_basin", "target_basin",
                            "source_basin_label", "target_basin_label",
                            "probability_PTSD", "probability_HC", "PTSD_minus_HC_probability",
                            "count_PTSD", "count_HC", "PTSD_minus_HC_count",
                            "subject_support_count_PTSD", "subject_support_count_HC",
                            "abs_probability_difference",
                        ]
                        display_companion_dataframe(
                            "Transition-level PTSD−HC differences",
                            keep_existing_cols(delta_transition_df, delta_transition_cols),
                            height=360,
                            max_rows=150,
                        )
                else:
                    st.info("Select `PTSD-minus-HC` in the cohort selector to view group-difference tables here.")


# ---------------------------------------------------------------------
# Pre-CS Sankey tab
# ---------------------------------------------------------------------

with tabs[5]:
    st.subheader("Pre-CS → During-CS → Post-CS Sankey")
    st.caption(
        "Directional event-window flow view. This avoids generic recurrent basin-to-basin Sankeys and focuses on "
        "where subjects come from before CS, where they are during CS, and where they go after CS."
    )

    if sankey_mod is None:
        st.error(f"Pre-CS Sankey module unavailable: {SANKEY_IMPORT_ERROR}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sankey_cohort = st.selectbox("Sankey cohort", ["merged", "HC", "PTSD"], index=0)
        with c2:
            sankey_event_filter = st.selectbox("Event filter", ["CS+ and CS-", "CS+", "CS-"], index=0)
        with c3:
            sankey_flow_level = st.selectbox("Flow level", ["basin", "state"], index=0)
        with c4:
            sankey_subject = sidebar_select_optional(
                "Sankey subject",
                subjects,
                key="sankey_subject",
                include_all=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pre_offset = st.number_input("Pre offset (TRs)", min_value=1, max_value=10, value=1, step=1)
        with c2:
            post_offset = st.number_input("Post offset (TRs)", min_value=1, max_value=10, value=1, step=1)
        with c3:
            sankey_min_count = st.number_input("Sankey min count", min_value=1, max_value=500, value=1, step=1)
        with c4:
            sankey_min_support = st.number_input("Sankey min subject support", min_value=0, max_value=100, value=0, step=1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sankey_top_k_raw = st.number_input("Top-k per source (0 = off)", min_value=0, max_value=20, value=0, step=1)
        with c2:
            node_color_by = st.selectbox("Node color by", ["basin", "event"], index=0)
        with c3:
            link_color_by = st.selectbox("Link color by", ["source", "target", "event"], index=0)
        with c4:
            sankey_height = st.number_input("Sankey height", min_value=500, max_value=1400, value=760, step=50)

        top_k_source = None if int(sankey_top_k_raw) == 0 else int(sankey_top_k_raw)

        try:
            sankey_edges = sankey_mod.compute_pre_during_post_edges(
                parsed_dir=parsed_dir,
                event_col=event_col,
                cohort=sankey_cohort,
                subject=sankey_subject,
                task=global_task,
                session=global_session,
                epoch_mode=epoch_mode,
                event_filter=sankey_event_filter,
                flow_level=sankey_flow_level,
                pre_offset=int(pre_offset),
                post_offset=int(post_offset),
                preset=preset_arg,
            )

            sankey_edges_filtered = sankey_mod.filter_sankey_edges(
                sankey_edges,
                min_count=int(sankey_min_count),
                min_subject_support_count=int(sankey_min_support),
                top_k_per_source=top_k_source,
            )

            if sankey_edges_filtered.empty:
                st.warning("No Sankey edges remain after filtering.")
            else:
                fig = sankey_mod.make_pre_during_post_sankey_from_edges(
                    sankey_edges,
                    title=(
                        f"Pre-CS → During-CS → Post-CS | {sankey_cohort} | "
                        f"{sankey_event_filter} | {event_col} | epoch={epoch_mode}"
                    ),
                    preset=preset_arg,
                    min_count=int(sankey_min_count),
                    min_subject_support_count=int(sankey_min_support),
                    top_k_per_source=top_k_source,
                    node_color_by=node_color_by,
                    link_color_by=link_color_by,
                    width=1250,
                    height=int(sankey_height),
                )

                stage_pct = sankey_mod.compute_stage_basin_percentages(
                    parsed_dir=parsed_dir,
                    event_col=event_col,
                    cohort=sankey_cohort,
                    subject=sankey_subject,
                    task=global_task,
                    session=global_session,
                    epoch_mode=epoch_mode,
                    event_filter=sankey_event_filter,
                    pre_offset=int(pre_offset),
                    post_offset=int(post_offset),
                    preset=preset_arg,
                )
                diff_df = sankey_mod.compute_csplus_minus_csminus_stage_difference(stage_pct)

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Stage-wise basin percentages")
                st.caption(
                    "This is the quantitative companion to the Sankey. Each bar is a Pre/During/Post stage; "
                    "segments show the percentage of event windows occupying each basin."
                )
                if stage_pct.empty:
                    st.warning("No stage percentage data available.")
                else:
                    pct_fig = sankey_mod.make_stage_basin_percentage_bar(
                        stage_pct,
                        title=f"Stage-wise basin percentages | {sankey_cohort} | epoch={epoch_mode}",
                        preset=preset_arg,
                        height=520,
                    )
                    st.plotly_chart(pct_fig, use_container_width=True)

                st.markdown("### CS+ minus CS− percentage difference")
                st.caption(
                    "Positive values mean CS+ uses that basin more than CS−. Negative values mean CS− uses that basin more than CS+."
                )
                if diff_df.empty or sankey_event_filter != "CS+ and CS-":
                    st.info("CS+ minus CS− difference requires Event filter = CS+ and CS-.")
                else:
                    diff_fig = sankey_mod.make_csplus_minus_csminus_difference_bar(
                        diff_df,
                        title=f"CS+ − CS− basin difference | {sankey_cohort} | epoch={epoch_mode}",
                        preset=preset_arg,
                        height=520,
                    )
                    st.plotly_chart(diff_fig, use_container_width=True)

                table_tabs = st.tabs(["Percentage table", "Difference table", "Sankey edge table"])
                with table_tabs[0]:
                    st.dataframe(stage_pct, use_container_width=True, height=360)
                    st.download_button(
                        "Download percentage table CSV",
                        data=stage_pct.to_csv(index=False).encode("utf-8"),
                        file_name="pre_during_post_cs_stage_basin_percentages.csv",
                        mime="text/csv",
                    )
                with table_tabs[1]:
                    st.dataframe(diff_df, use_container_width=True, height=360)
                    st.download_button(
                        "Download CS difference table CSV",
                        data=diff_df.to_csv(index=False).encode("utf-8"),
                        file_name="pre_during_post_cs_stage_cs_difference.csv",
                        mime="text/csv",
                    )
                with table_tabs[2]:
                    st.caption("These are the actual Pre→During and During→Post edge counts used by the Sankey.")
                    st.dataframe(sankey_edges_filtered, use_container_width=True, height=360)
                    st.download_button(
                        "Download Sankey edges CSV",
                        data=sankey_edges_filtered.to_csv(index=False).encode("utf-8"),
                        file_name="pre_during_post_cs_sankey_edges.csv",
                        mime="text/csv",
                    )

                with st.expander("Interpretation guide", expanded=False):
                    st.markdown(
                        """
                        - **Sankey** = common directional routes from Pre-CS → During-CS → Post-CS.
                        - **Stage-wise percentages** = how much each basin contributes at each stage.
                        - **CS+ − CS− difference** = direct contrast of event-specific basin use.
                        - **Pre** = basin/state at `anchor - pre_offset`.
                        - **During** = basin/state at the CS-labelled anchor TR from the selected event column.
                        - **Post** = basin/state at `anchor + post_offset`.
                        - This is a directional task-window summary, not a generic recurrent basin transition plot.
                        """
                    )
        except Exception as e:
            st.exception(e)


# ---------------------------------------------------------------------
# Radial explorer tab
# ---------------------------------------------------------------------

with tabs[6]:
    st.subheader("Radial basin explorer")
    st.caption("Per-subject animation. Subject selection is required because event order is randomized per subject.")

    if radial_mod is None:
        st.error(f"Radial explorer module unavailable: {RADIAL_IMPORT_ERROR}")
    else:
        # Subject options can be filtered by global group/task/session.
        radial_base = filter_df_for_overview(timeline_df, group=global_group, task=global_task, session=global_session)
        radial_subjects = available_values(radial_base, "subject")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            radial_subject = st.selectbox("Subject", radial_subjects if radial_subjects else subjects, index=0)
        with c2:
            tail_length = st.number_input("Tail length", min_value=1, max_value=20, value=3, step=1)
        with c3:
            frame_step = st.number_input("Frame step", min_value=1, max_value=10, value=1, step=1)
        with c4:
            frame_duration = st.number_input("Frame duration ms", min_value=100, max_value=3000, value=600, step=100)

        c1, c2, c3 = st.columns(3)
        with c1:
            radial_height = st.number_input("Radial figure height", min_value=800, max_value=1800, value=1200, step=50)
        with c2:
            show_state_labels = st.checkbox("Show state labels", value=True)
        with c3:
            activity_window = st.number_input("Activity history window", min_value=2, max_value=8, value=4, step=1)

        try:
            fig = radial_mod.make_radial_basin_explorer(
                parsed_dir=parsed_dir,
                subject=radial_subject,
                group=global_group,
                task=global_task,
                session=global_session,
                event_col=event_col,
                epoch_mode=epoch_mode,
                preset=preset_arg,
                tail_length=int(tail_length),
                frame_step=int(frame_step),
                frame_duration=int(frame_duration),
                activity_window=int(activity_window),
                show_state_labels=show_state_labels,
                width=1450,
                height=int(radial_height),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.exception(e)


# ---------------------------------------------------------------------
# Summary tables tab
# ---------------------------------------------------------------------

with tabs[7]:
    st.subheader("Summary tables")
    st.caption("Numerical tables supporting the visual modules: state event summaries, local minima, transition backbone, group deltas, and detected statistics files.")

    if summary_mod is None:
        st.error(f"Summary tables module unavailable: {SUMMARY_IMPORT_ERROR}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sum_top_n = st.number_input("Top N", min_value=5, max_value=50, value=12, step=1)
        with c2:
            sum_min_count = st.number_input("Summary min transition count", min_value=1, max_value=100, value=5, step=1)
        with c3:
            sum_min_prob = st.number_input("Summary min transition prob", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        with c4:
            sum_min_support = st.number_input("Summary min subject support", min_value=0, max_value=100, value=10, step=1)

        if st.button("Generate / refresh summary tables", type="primary"):
            try:
                outputs = summary_mod.generate_summary_tables(
                    parsed_dir=parsed_dir,
                    event_col=event_col,
                    epoch_mode=epoch_mode,
                    group=global_group,
                    task=global_task,
                    session=global_session,
                    top_n=int(sum_top_n),
                    min_transition_count=int(sum_min_count),
                    min_transition_prob=float(sum_min_prob),
                    min_subject_support_count=int(sum_min_support),
                    top_k_per_state=2,
                    preset=preset if preset != "none" else "fear4",
                )
                st.success(f"Generated {len(outputs)} summary outputs in {parsed_dir / 'summary_tables'}")
            except Exception as e:
                st.exception(e)

        summary_dir = parsed_dir / "summary_tables"
        if not summary_dir.exists():
            st.info("No summary_tables/ folder found yet. Click the button above to generate tables.")
        else:
            table_files = sorted(summary_dir.glob("*.csv"))
            md_file = summary_dir / "summary_report.md"

            if md_file.exists():
                with st.expander("Summary report", expanded=True):
                    st.markdown(md_file.read_text(encoding="utf-8", errors="ignore"))

            if not table_files:
                st.warning("summary_tables/ exists, but no CSV files were found.")
            else:
                preferred_order = [
                    "state_event_summary.csv",
                    "top_csplus_states.csv",
                    "top_csminus_states.csv",
                    "top_enriched_states.csv",
                    "local_minimum_summary.csv",
                    "basin_event_summary.csv",
                    "transition_backbone_summary.csv",
                    "group_delta_state_summary.csv",
                    "group_delta_transition_summary.csv",
                    "statistical_top_results.csv",
                    "detected_feature_files.csv",
                    "detected_confirmatory_files.csv",
                ]
                file_map = {p.name: p for p in table_files}
                ordered_names = [n for n in preferred_order if n in file_map] + [p.name for p in table_files if p.name not in preferred_order]

                chosen = st.selectbox("Table", ordered_names)
                display_table_file(file_map[chosen], max_rows=300)

                with open(file_map[chosen], "rb") as f:
                    st.download_button(
                        label=f"Download {chosen}",
                        data=f,
                        file_name=chosen,
                        mime="text/csv",
                    )
