#!/usr/bin/env python3
"""
app_streamlit_v1_5_elatvis.py

Streamlit MVP dashboard for ELAT-Vis.

Expected module directory:
    /mnt/d/Thesis Masters/thesis-runs/data_viz/final

Run:
    cd "/mnt/d/Thesis Masters/thesis-runs/data_viz/final"
    streamlit run app_streamlit_v1_5_elatvis.py

Expected parsed directory example:
    /mnt/d/Thesis Masters/thesis-runs/ELA_run_5ROI/_merged_fit_runs_ela_5ROI/merge_extinction_ela_5ROI/parsed
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

for p in [THIS_DIR, DEFAULT_MODULE_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


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
    parsed_dir = Path(parsed_dir_str)
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


def find_elat_output_dir(parsed_dir: Path) -> Path:
    """Infer the ELAT output directory from a parsed directory."""
    if parsed_dir.name.lower() == "parsed":
        return parsed_dir.parent
    return parsed_dir


def list_elat_figure_files(elat_dir: Path) -> List[Path]:
    """
    Find ELAT figure outputs.

    ELAT commonly writes Figure_100 ... Figure_103, but extensions vary
    depending on export settings. This function prioritizes Figure_* files
    and falls back to common image/html figure files.
    """
    if not elat_dir.exists():
        return []

    allowed = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".html", ".htm", ".pdf"}
    candidates = []
    for pat in ["Figure_100*", "Figure_101*", "Figure_102*", "Figure_103*", "figure_100*", "figure_101*", "figure_102*", "figure_103*"]:
        candidates.extend([p for p in elat_dir.glob(pat) if p.suffix.lower() in allowed])

    if not candidates:
        for pat in ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.webp", "*.html", "*.htm", "*.pdf"]:
            candidates.extend(elat_dir.glob(pat))

    # De-duplicate while preserving sorted order.
    seen = set()
    out = []
    for p in sorted(candidates, key=lambda x: x.name.lower()):
        if p.resolve() not in seen:
            out.append(p)
            seen.add(p.resolve())
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

default_parsed = (
    "/mnt/d/Thesis Masters/thesis-runs/ELA_run_5ROI/"
    "_merged_fit_runs_ela_5ROI/merge_extinction_ela_5ROI/parsed"
)

parsed_dir_str = st.sidebar.text_input(
    "Parsed ELAT directory",
    value=default_parsed,
    help="Use the parsed/ folder produced by parser.py.",
)

basic = load_parsed_basic(parsed_dir_str)
parsed_dir = Path(parsed_dir_str)

if not basic["exists"]:
    st.error("Parsed directory does not exist. Check the path in the sidebar.")
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
# ELAT Figures tab
# ---------------------------------------------------------------------

with tabs[1]:
    st.subheader("Original ELAT figures")
    st.caption("Displays Figure_100–Figure_103 outputs from the ELAT run directory when image/HTML exports are available.")

    elat_dir_default = find_elat_output_dir(parsed_dir)
    elat_dir_str = st.text_input(
        "ELAT output directory",
        value=str(elat_dir_default),
        help="Usually the parent folder of parsed/. This should contain BasinList.txt, result.txt, Figure_100–Figure_103, etc.",
    )
    elat_dir = Path(elat_dir_str)

    fig_files = list_elat_figure_files(elat_dir)
    if not fig_files:
        st.warning("No displayable ELAT figure files found. Supported: PNG/JPG/SVG/WEBP/HTML/PDF.")
        st.write("Checked:")
        st.code(str(elat_dir))
        st.info("If ELAT only exported MATLAB .fig files, export them first as .png or .html for dashboard preview.")
    else:
        st.success(f"Found {len(fig_files)} displayable figure file(s).")
        view_mode = st.radio("View mode", ["One figure", "Gallery"], horizontal=True)
        if view_mode == "One figure":
            selected = st.selectbox("Figure", fig_files, format_func=lambda p: p.name)
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

with tabs[2]:
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

with tabs[3]:
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

with tabs[4]:
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

with tabs[5]:
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

with tabs[6]:
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
