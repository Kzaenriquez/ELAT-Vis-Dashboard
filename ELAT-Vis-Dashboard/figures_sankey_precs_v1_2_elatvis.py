#!/usr/bin/env python3
"""
figures_sankey_precs_v1_2_elatvis.py

Pre-CS / During-CS / Post-CS Sankey module for ELAT-Vis.

Focus
-----
This is intentionally NOT a generic recurrent basin-to-basin Sankey.
It only summarizes directional event-window flows:

    Pre-CS basin  ->  During-CS basin  ->  Post-CS basin

This is the Sankey use case that remains interpretable for task-based ELA.

Inputs
------
A parsed ELAT directory produced by parser.py:

    parsed/
      timeline_table.csv
      states_table.csv
      model_metadata.json

CLI example
-----------
python figures_sankey_precs_v1_2_elatvis.py \
  --parsed-dir "merge_extinction_ela_5ROI/parsed" \
  --event-col trial_type_hrf4 \
  --epoch-mode late \
  --cohort HC \
  --flow-level basin \
  --pre-offset 1 \
  --post-offset 1

Streamlit-ready
---------------
from figures_sankey_precs_v1_elatvis import compute_pre_during_post_edges, make_pre_during_post_sankey
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


EVENT_COLORS = {
    "CS+": "#D95F5F",
    "CS-": "#4C78A8",
    "ITI": "#5F6368",
    "OTHER": "#CBD5E1",
    "NONE": "#D9D9D9",
}

BASIN_LABELS_FEAR4 = {
    1: "Global off",
    2: "Threat-like",
    3: "Safety-like",
    4: "Global on",
}

BASIN_COLORS_FEAR4 = {
    1: "#A8D5A2",
    2: "#D95F5F",
    3: "#4C78A8",
    4: "#2E7D32",
}

DEFAULT_BASIN_COLORS = [
    "#A8D5A2", "#D95F5F", "#4C78A8", "#2E7D32", "#B07AA1",
    "#F28E2B", "#76B7B2", "#EDC948", "#9C755F", "#7F7F7F",
]


# ---------------------------------------------------------------------
# Loading / normalization
# ---------------------------------------------------------------------

def read_csv_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def load_parsed_tables(parsed_dir: Union[str, Path]) -> Dict[str, object]:
    parsed_dir = Path(parsed_dir)
    metadata = {}
    meta_path = parsed_dir / "model_metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    return {
        "timeline": read_csv_if_exists(parsed_dir / "timeline_table.csv"),
        "states": read_csv_if_exists(parsed_dir / "states_table.csv"),
        "basins": read_csv_if_exists(parsed_dir / "basins_table.csv"),
        "metadata": metadata,
    }


def coalesce_duplicate_columns(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    matches = [i for i, c in enumerate(df.columns) if c == col]
    if len(matches) <= 1:
        return df
    merged = df.iloc[:, matches].bfill(axis=1).iloc[:, 0]
    keep = np.ones(df.shape[1], dtype=bool)
    keep[matches] = False
    out = df.iloc[:, keep].copy()
    out[col] = merged
    return out


def normalize_state_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["state", "State", "state_id", "state_label", "SN"]:
        if c in df.columns and c != "state":
            df = df.rename(columns={c: "state"})
            break
    if "state" in df.columns:
        df = coalesce_duplicate_columns(df, "state")
        df["state"] = pd.to_numeric(df["state"], errors="coerce").astype("Int64")
    return df


def normalize_basin_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["basin", "Basin", "basin_id", "basin_label", "BN"]:
        if c in df.columns and c != "basin":
            df = df.rename(columns={c: "basin"})
            break
    if "basin" in df.columns:
        df = coalesce_duplicate_columns(df, "basin")
        df["basin"] = pd.to_numeric(df["basin"], errors="coerce").astype("Int64")
    return df


def normalize_event_value(x) -> str:
    if pd.isna(x):
        return "NONE"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "n/a"}:
        return "NONE"
    s2 = s.upper().replace(" ", "").replace("_", "").replace("−", "-")
    if s2 in {"CS+", "CSP", "CSPLUS", "CSPOS", "CSPOSITIVE", "THREAT"}:
        return "CS+"
    if s2 in {"CS-", "CSMINUS", "CSNEG", "CSNEGATIVE", "SAFETY"}:
        return "CS-"
    if "CS+" in s2 or "CSPLUS" in s2:
        return "CS+"
    if "CS-" in s2 or "CSMINUS" in s2:
        return "CS-"
    if "ITI" in s2:
        return "ITI"
    return "OTHER"


def normalize_epoch_value(x) -> str:
    if pd.isna(x):
        return "none"
    s = str(x).strip().lower()
    if "early" in s:
        return "early"
    if "late" in s:
        return "late"
    return s if s else "none"


def find_epoch_col(timeline: Optional[pd.DataFrame], event_col: Optional[str] = None) -> Optional[str]:
    if timeline is None:
        return None
    cols = list(timeline.columns)
    if event_col:
        suffix = event_col.replace("trial_type_", "")
        candidates = [
            f"trialblock_epoch_{suffix}",
            f"trial_block_epoch_{suffix}",
            f"epoch_{suffix}",
            f"{event_col}_epoch",
            event_col.replace("trial_type", "trialblock_epoch"),
            event_col.replace("trial_type", "trial_block_epoch"),
        ]
        for c in candidates:
            if c in cols:
                return c
    for pat in ["trialblock_epoch", "trial_block_epoch", "block_epoch", "early_late", "epoch"]:
        matches = [c for c in cols if pat.lower() in c.lower()]
        if matches:
            return matches[0]
    return None


def choose_x_col(timeline: pd.DataFrame) -> str:
    for c in ["orig_tr", "kept_row", "tr", "TR", "row_index_0", "timepoint"]:
        if c in timeline.columns:
            return c
    timeline["row_index_0"] = np.arange(len(timeline))
    return "row_index_0"


def basin_labels_map(basins: Iterable[int], preset: Optional[str] = "fear4") -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_LABELS_FEAR4.copy()
        for b in basins:
            out.setdefault(b, f"B{b}")
        return out
    return {b: f"B{b}" for b in basins}


def basin_color_map(basins: Iterable[int], preset: Optional[str] = "fear4") -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_COLORS_FEAR4.copy()
        for i, b in enumerate(basins):
            out.setdefault(b, DEFAULT_BASIN_COLORS[i % len(DEFAULT_BASIN_COLORS)])
        return out
    return {b: DEFAULT_BASIN_COLORS[i % len(DEFAULT_BASIN_COLORS)] for i, b in enumerate(basins)}


def hex_to_rgba(hex_color: str, alpha: float = 0.30) -> str:
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return f"rgba(100,116,139,{alpha})"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------

def prepare_timeline(
    parsed_dir: Union[str, Path],
    event_col: str = "trial_type_hrf4",
    cohort: str = "merged",
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    epoch_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    tables = load_parsed_tables(parsed_dir)
    timeline = tables.get("timeline")
    if timeline is None or timeline.empty:
        raise FileNotFoundError(f"No timeline_table.csv found in {parsed_dir}")

    df = normalize_basin_col(normalize_state_col(timeline))
    if "basin" not in df.columns:
        raise ValueError("timeline_table.csv must contain a basin column.")
    if event_col not in df.columns:
        raise ValueError(f"Event column not found in timeline_table.csv: {event_col}")

    for col in ["group", "subject", "task", "session"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if cohort and cohort not in {"merged", "All", "all"} and "group" in df.columns:
        df = df[df["group"].astype(str) == str(cohort)]

    if subject and "subject" in df.columns:
        df = df[df["subject"].astype(str) == str(subject)]
    if task and "task" in df.columns:
        df = df[df["task"].astype(str) == str(task)]
    if session and "session" in df.columns:
        df = df[df["session"].astype(str) == str(session)]

    x_col = choose_x_col(df)
    epoch_col = epoch_col or find_epoch_col(df, event_col)

    df["event_norm"] = df[event_col].apply(normalize_event_value)
    df["epoch_norm"] = df[epoch_col].apply(normalize_epoch_value) if epoch_col in df.columns else "none"
    df["basin"] = pd.to_numeric(df["basin"], errors="coerce").astype("Int64")
    if "state" in df.columns:
        df["state"] = pd.to_numeric(df["state"], errors="coerce").astype("Int64")

    sort_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns] + [x_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df, x_col, epoch_col


def node_label_for_row(row: pd.Series, stage: str, event: str, flow_level: str, basin_labels: Dict[int, str]) -> str:
    b = int(row["basin"]) if pd.notna(row.get("basin")) else -1

    if flow_level == "state":
        state = int(row["state"]) if "state" in row and pd.notna(row.get("state")) else "NA"
        return f"{stage} {event} | S{state} / B{b}"

    # default = basin
    return f"{stage} {event} | B{b}: {basin_labels.get(b, f'B{b}')}"


def compute_pre_during_post_edges(
    parsed_dir: Union[str, Path],
    event_col: str = "trial_type_hrf4",
    cohort: str = "merged",
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    event_filter: str = "CS+ and CS-",
    flow_level: str = "basin",
    pre_offset: int = 1,
    post_offset: int = 1,
    preset: Optional[str] = "fear4",
) -> pd.DataFrame:
    """
    Compute windowed Sankey edges.

    Epoch filtering is applied to the DURING-CS anchor row only.
    Pre/post rows are allowed to come from the adjacent TRs even if they cross
    an epoch boundary; this preserves valid windows around anchor events.
    """
    df, x_col, epoch_col = prepare_timeline(
        parsed_dir=parsed_dir,
        event_col=event_col,
        cohort=cohort,
        subject=subject,
        task=task,
        session=session,
        epoch_col=epoch_col,
    )

    if df.empty:
        return pd.DataFrame()

    basins = sorted(df["basin"].dropna().astype(int).unique().tolist())
    basin_labels = basin_labels_map(basins, preset=preset)

    if event_filter == "CS+":
        wanted_events = {"CS+"}
    elif event_filter == "CS-":
        wanted_events = {"CS-"}
    else:
        wanted_events = {"CS+", "CS-"}

    group_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns]
    if not group_cols:
        df["_series"] = "all"
        group_cols = ["_series"]

    rows = []
    windows = []

    for _, g in df.groupby(group_cols, dropna=False):
        g = g.reset_index(drop=True)
        if len(g) < (pre_offset + post_offset + 1):
            continue

        for i, r in g.iterrows():
            ev = r["event_norm"]
            if ev not in wanted_events:
                continue

            if epoch_mode != "full":
                ep = normalize_epoch_value(r.get("epoch_norm", "none"))
                if ep != epoch_mode:
                    continue

            pre_i = i - int(pre_offset)
            post_i = i + int(post_offset)
            if pre_i < 0 or post_i >= len(g):
                continue

            pre = g.loc[pre_i]
            dur = r
            post = g.loc[post_i]

            if pd.isna(pre.get("basin")) or pd.isna(dur.get("basin")) or pd.isna(post.get("basin")):
                continue

            pre_label = node_label_for_row(pre, "Pre", ev, flow_level, basin_labels)
            dur_label = node_label_for_row(dur, "During", ev, flow_level, basin_labels)
            post_label = node_label_for_row(post, "Post", ev, flow_level, basin_labels)

            subj = str(r.get("subject", "NA"))
            group = str(r.get("group", "NA"))
            task_val = str(r.get("task", "NA"))
            session_val = str(r.get("session", "NA"))

            common = {
                "event": ev,
                "group": group,
                "subject": subj,
                "task": task_val,
                "session": session_val,
                "anchor_index": int(i),
                "anchor_x": r.get(x_col, np.nan),
                "anchor_state": int(r["state"]) if "state" in r and pd.notna(r.get("state")) else np.nan,
                "pre_state": int(pre["state"]) if "state" in pre and pd.notna(pre.get("state")) else np.nan,
                "during_state": int(dur["state"]) if "state" in dur and pd.notna(dur.get("state")) else np.nan,
                "post_state": int(post["state"]) if "state" in post and pd.notna(post.get("state")) else np.nan,
                "pre_basin": int(pre["basin"]),
                "during_basin": int(dur["basin"]),
                "post_basin": int(post["basin"]),
            }

            rows.append({
                "source_label": pre_label,
                "target_label": dur_label,
                "source_stage": "Pre",
                "target_stage": "During",
                **common,
            })
            rows.append({
                "source_label": dur_label,
                "target_label": post_label,
                "source_stage": "During",
                "target_stage": "Post",
                **common,
            })
            windows.append(common)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    edges = (
        raw.groupby(["source_label", "target_label", "source_stage", "target_stage", "event"], as_index=False)
           .agg(
               count=("event", "size"),
               subject_support_count=("subject", "nunique"),
               groups=("group", lambda x: ", ".join(sorted(set(map(str, x))))),
           )
    )

    n_subjects = max(raw["subject"].nunique(), 1) if "subject" in raw.columns else 1
    edges["subject_support_prop"] = edges["subject_support_count"] / n_subjects

    # Add common source/target basin/state decoding where possible from labels.
    edges["source_basin"] = edges["source_label"].str.extract(r"\bB(-?\d+)\b")[0]
    edges["target_basin"] = edges["target_label"].str.extract(r"\bB(-?\d+)\b")[0]
    edges["source_basin"] = pd.to_numeric(edges["source_basin"], errors="coerce").astype("Int64")
    edges["target_basin"] = pd.to_numeric(edges["target_basin"], errors="coerce").astype("Int64")

    return edges.sort_values(["source_stage", "event", "count"], ascending=[True, True, False]).reset_index(drop=True)


def filter_sankey_edges(
    edges: pd.DataFrame,
    min_count: int = 1,
    min_subject_support_count: int = 0,
    top_k_per_source: Optional[int] = None,
) -> pd.DataFrame:
    if edges is None or edges.empty:
        return pd.DataFrame()

    df = edges.copy()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    df = df[df["count"] >= int(min_count)]

    if min_subject_support_count and "subject_support_count" in df.columns:
        df = df[pd.to_numeric(df["subject_support_count"], errors="coerce").fillna(0) >= int(min_subject_support_count)]

    if top_k_per_source is not None and int(top_k_per_source) > 0 and not df.empty:
        df = df.sort_values(["source_label", "count"], ascending=[True, False])
        df = df.groupby("source_label", group_keys=False).head(int(top_k_per_source))

    return df.reset_index(drop=True)



# ---------------------------------------------------------------------
# Stage-wise percentage summaries
# ---------------------------------------------------------------------

def _event_set_from_filter(event_filter: str) -> set:
    if event_filter == "CS+":
        return {"CS+"}
    if event_filter == "CS-":
        return {"CS-"}
    return {"CS+", "CS-"}


def compute_pre_during_post_windows(
    parsed_dir: Union[str, Path],
    event_col: str = "trial_type_hrf4",
    cohort: str = "merged",
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    event_filter: str = "CS+ and CS-",
    pre_offset: int = 1,
    post_offset: int = 1,
    preset: Optional[str] = "fear4",
) -> pd.DataFrame:
    """
    Return one row per valid Pre/During/Post event window.

    Columns include pre_basin, during_basin, post_basin, event, subject, group,
    and optional states. This table is the basis for both the Sankey and the
    stage-wise percentage charts.
    """
    df, x_col, epoch_col = prepare_timeline(
        parsed_dir=parsed_dir,
        event_col=event_col,
        cohort=cohort,
        subject=subject,
        task=task,
        session=session,
        epoch_col=epoch_col,
    )

    if df.empty:
        return pd.DataFrame()

    wanted_events = _event_set_from_filter(event_filter)
    group_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns]
    if not group_cols:
        df["_series"] = "all"
        group_cols = ["_series"]

    rows = []
    for _, g in df.groupby(group_cols, dropna=False):
        g = g.reset_index(drop=True)
        if len(g) < (pre_offset + post_offset + 1):
            continue

        for i, r in g.iterrows():
            ev = r["event_norm"]
            if ev not in wanted_events:
                continue

            if epoch_mode != "full":
                ep = normalize_epoch_value(r.get("epoch_norm", "none"))
                if ep != epoch_mode:
                    continue

            pre_i = i - int(pre_offset)
            post_i = i + int(post_offset)
            if pre_i < 0 or post_i >= len(g):
                continue

            pre = g.loc[pre_i]
            dur = r
            post = g.loc[post_i]

            if pd.isna(pre.get("basin")) or pd.isna(dur.get("basin")) or pd.isna(post.get("basin")):
                continue

            rows.append({
                "cohort": cohort,
                "event": ev,
                "group": str(r.get("group", "NA")),
                "subject": str(r.get("subject", "NA")),
                "task": str(r.get("task", "NA")),
                "session": str(r.get("session", "NA")),
                "epoch": normalize_epoch_value(r.get("epoch_norm", "none")),
                "anchor_index": int(i),
                "anchor_x": r.get(x_col, np.nan),
                "pre_basin": int(pre["basin"]),
                "during_basin": int(dur["basin"]),
                "post_basin": int(post["basin"]),
                "pre_state": int(pre["state"]) if "state" in pre and pd.notna(pre.get("state")) else np.nan,
                "during_state": int(dur["state"]) if "state" in dur and pd.notna(dur.get("state")) else np.nan,
                "post_state": int(post["state"]) if "state" in post and pd.notna(post.get("state")) else np.nan,
            })

    return pd.DataFrame(rows)


def compute_stage_basin_percentages(
    parsed_dir: Union[str, Path],
    event_col: str = "trial_type_hrf4",
    cohort: str = "merged",
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    event_filter: str = "CS+ and CS-",
    pre_offset: int = 1,
    post_offset: int = 1,
    preset: Optional[str] = "fear4",
) -> pd.DataFrame:
    """
    Compute percentages for each event x stage x basin.

    Output columns:
        cohort, event, stage, basin, basin_label, count, total_event_stage, percent
    """
    windows = compute_pre_during_post_windows(
        parsed_dir=parsed_dir,
        event_col=event_col,
        cohort=cohort,
        subject=subject,
        task=task,
        session=session,
        epoch_col=epoch_col,
        epoch_mode=epoch_mode,
        event_filter=event_filter,
        pre_offset=pre_offset,
        post_offset=post_offset,
        preset=preset,
    )

    if windows.empty:
        return pd.DataFrame(columns=[
            "cohort", "event", "stage", "basin", "basin_label",
            "count", "total_event_stage", "percent"
        ])

    basin_values = sorted(
        set(windows["pre_basin"].dropna().astype(int))
        | set(windows["during_basin"].dropna().astype(int))
        | set(windows["post_basin"].dropna().astype(int))
    )
    labels = basin_labels_map(basin_values, preset=preset)

    long_rows = []
    for stage, col in [("Pre", "pre_basin"), ("During", "during_basin"), ("Post", "post_basin")]:
        tmp = windows[["cohort", "event", col]].copy()
        tmp = tmp.rename(columns={col: "basin"})
        tmp["stage"] = stage
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["basin"] = pd.to_numeric(long_df["basin"], errors="coerce").astype("Int64")

    grouped = (
        long_df.groupby(["cohort", "event", "stage", "basin"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    # Complete event x stage x basin grid so missing basin percentages display as 0.
    events = sorted(long_df["event"].dropna().astype(str).unique().tolist())
    stages = ["Pre", "During", "Post"]
    grid = pd.MultiIndex.from_product(
        [[cohort], events, stages, basin_values],
        names=["cohort", "event", "stage", "basin"]
    ).to_frame(index=False)

    grouped = grid.merge(grouped, on=["cohort", "event", "stage", "basin"], how="left")
    grouped["count"] = grouped["count"].fillna(0).astype(int)
    grouped["total_event_stage"] = grouped.groupby(["cohort", "event", "stage"])["count"].transform("sum")
    grouped["percent"] = np.where(
        grouped["total_event_stage"] > 0,
        grouped["count"] / grouped["total_event_stage"] * 100.0,
        0.0,
    )
    grouped["basin_label"] = grouped["basin"].astype(int).map(lambda b: labels.get(b, f"B{b}"))
    grouped["stage"] = pd.Categorical(grouped["stage"], categories=stages, ordered=True)
    grouped = grouped.sort_values(["event", "stage", "basin"]).reset_index(drop=True)
    grouped["stage"] = grouped["stage"].astype(str)
    return grouped


def compute_csplus_minus_csminus_stage_difference(stage_pct: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CS+ minus CS- percentage per stage x basin.

    Input should come from compute_stage_basin_percentages().
    """
    if stage_pct is None or stage_pct.empty:
        return pd.DataFrame(columns=[
            "cohort", "stage", "basin", "basin_label",
            "CSplus_percent", "CSminus_percent", "CSplus_minus_CSminus"
        ])

    df = stage_pct.copy()
    df["event_clean"] = df["event"].replace({"CS+": "CSplus", "CS-": "CSminus"})
    pivot = (
        df.pivot_table(
            index=["cohort", "stage", "basin", "basin_label"],
            columns="event_clean",
            values="percent",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )

    if "CSplus" not in pivot.columns:
        pivot["CSplus"] = 0.0
    if "CSminus" not in pivot.columns:
        pivot["CSminus"] = 0.0

    pivot = pivot.rename(columns={
        "CSplus": "CSplus_percent",
        "CSminus": "CSminus_percent",
    })
    pivot["CSplus_minus_CSminus"] = pivot["CSplus_percent"] - pivot["CSminus_percent"]

    stage_order = {"Pre": 0, "During": 1, "Post": 2}
    pivot["_stage_order"] = pivot["stage"].map(stage_order).fillna(9)
    pivot = pivot.sort_values(["_stage_order", "basin"]).drop(columns=["_stage_order"])
    return pivot.reset_index(drop=True)


def make_stage_basin_percentage_bar(
    stage_pct: pd.DataFrame,
    title: str = "Stage-wise basin percentages",
    preset: Optional[str] = "fear4",
    width: int = 1100,
    height: int = 520,
) -> go.Figure:
    """
    100% stacked bar chart:
        x = event + stage
        color = basin
        y = percent
    """
    if stage_pct is None or stage_pct.empty:
        raise ValueError("No stage percentage data available.")

    df = stage_pct.copy()
    stage_order = {"Pre": 0, "During": 1, "Post": 2}
    event_order = {"CS+": 0, "CS-": 1, "OTHER": 9}
    df["_stage_order"] = df["stage"].map(stage_order).fillna(9)
    df["_event_order"] = df["event"].map(event_order).fillna(9)
    df["x_label"] = df["event"].astype(str) + " " + df["stage"].astype(str)

    basins = sorted(df["basin"].dropna().astype(int).unique().tolist())
    colors = basin_color_map(basins, preset=preset)

    x_order = (
        df[["event", "stage", "x_label", "_event_order", "_stage_order"]]
        .drop_duplicates()
        .sort_values(["_event_order", "_stage_order"])["x_label"]
        .tolist()
    )

    fig = go.Figure()
    for b in basins:
        sub = df[df["basin"].astype(int) == b].copy()
        sub = sub.set_index("x_label").reindex(x_order).reset_index()
        label = stage_pct.loc[stage_pct["basin"].astype(int) == b, "basin_label"].dropna()
        basin_label = label.iloc[0] if len(label) else f"B{b}"

        fig.add_trace(go.Bar(
            x=x_order,
            y=sub["percent"],
            name=f"B{b}: {basin_label}",
            marker_color=colors.get(b, "#64748B"),
            customdata=np.stack([
                sub["count"].fillna(0),
                sub["total_event_stage"].fillna(0),
                sub["percent"].fillna(0),
            ], axis=-1),
            hovertemplate=(
                f"B{b}: {basin_label}<br>"
                "window=%{x}<br>"
                "count=%{customdata[0]:.0f}/%{customdata[1]:.0f}<br>"
                "percent=%{customdata[2]:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111827")),
        barmode="stack",
        width=width,
        height=height,
        yaxis=dict(title="Percent of event windows", range=[0, 100], ticksuffix="%"),
        xaxis=dict(title="Event window stage"),
        legend=dict(title="Basin"),
        margin=dict(l=40, r=30, t=70, b=60),
        font=dict(size=12, color="#111827"),
    )
    return fig


def make_csplus_minus_csminus_difference_bar(
    diff_df: pd.DataFrame,
    title: str = "CS+ minus CS− basin percentage difference",
    preset: Optional[str] = "fear4",
    width: int = 1100,
    height: int = 520,
) -> go.Figure:
    """
    Grouped bar chart:
        x = basin
        trace = stage
        y = CS+ percent - CS- percent

    Positive = CS+ occupies that basin more than CS-.
    Negative = CS- occupies that basin more than CS+.
    """
    if diff_df is None or diff_df.empty:
        raise ValueError("No CS+ minus CS- difference data available.")

    df = diff_df.copy()
    basins = sorted(df["basin"].dropna().astype(int).unique().tolist())
    stage_colors = {
        "Pre": "#94A3B8",
        "During": "#F59E0B",
        "Post": "#7C3AED",
    }
    x_labels = []
    for b in basins:
        label = df.loc[df["basin"].astype(int) == b, "basin_label"].dropna()
        basin_label = label.iloc[0] if len(label) else f"B{b}"
        x_labels.append(f"B{b}: {basin_label}")

    fig = go.Figure()
    for stage in ["Pre", "During", "Post"]:
        sub = df[df["stage"].astype(str) == stage].copy()
        sub = sub.set_index("basin").reindex(basins).reset_index()
        fig.add_trace(go.Bar(
            x=x_labels,
            y=sub["CSplus_minus_CSminus"],
            name=stage,
            marker_color=stage_colors.get(stage, "#64748B"),
            customdata=np.stack([
                sub["CSplus_percent"].fillna(0),
                sub["CSminus_percent"].fillna(0),
                sub["CSplus_minus_CSminus"].fillna(0),
            ], axis=-1),
            hovertemplate=(
                f"{stage}<br>"
                "%{x}<br>"
                "CS+ = %{customdata[0]:.1f}%<br>"
                "CS- = %{customdata[1]:.1f}%<br>"
                "CS+ − CS- = %{customdata[2]:+.1f} pp<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_width=1, line_color="#111827")
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111827")),
        barmode="group",
        width=width,
        height=height,
        yaxis=dict(title="CS+ − CS− percentage points", ticksuffix=" pp"),
        xaxis=dict(title="Basin"),
        legend=dict(title="Stage"),
        margin=dict(l=50, r=30, t=70, b=80),
        font=dict(size=12, color="#111827"),
    )
    return fig


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def parse_stage(label: str) -> str:
    label = str(label)
    if label.startswith("Pre "):
        return "Pre"
    if label.startswith("During "):
        return "During"
    if label.startswith("Post "):
        return "Post"
    return "Other"


def parse_event(label: str) -> str:
    label = str(label)
    if "CS+" in label:
        return "CS+"
    if "CS-" in label:
        return "CS-"
    if "ITI" in label:
        return "ITI"
    return "OTHER"


def parse_basin(label: str) -> Optional[int]:
    m = re.search(r"\bB(-?\d+)\b", str(label))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def sankey_label_sort(labels: Iterable[str]) -> List[str]:
    stage_order = {"Pre": 0, "During": 1, "Post": 2, "Other": 9}
    event_order = {"CS+": 0, "CS-": 1, "ITI": 2, "OTHER": 9}
    def key(label: str):
        stage = parse_stage(label)
        ev = parse_event(label)
        b = parse_basin(label)
        return (stage_order.get(stage, 9), event_order.get(ev, 9), b if b is not None else 999, str(label))
    return sorted([str(x) for x in labels], key=key)


def label_color(label: str, basin_colors: Dict[int, str], color_by: str = "basin") -> str:
    ev = parse_event(label)
    b = parse_basin(label)
    if color_by == "event":
        return EVENT_COLORS.get(ev, "#64748B")
    if b is not None:
        return basin_colors.get(b, "#64748B")
    return EVENT_COLORS.get(ev, "#64748B")



def compact_sankey_label(label: str) -> str:
    """
    Shorter, darker Sankey labels.

    Full label is kept in hover. Display label is compact to reduce visual clutter.
    """
    label = str(label)
    stage = parse_stage(label)
    ev = parse_event(label)
    b = parse_basin(label)

    state_match = re.search(r"\bS(\d+)\b", label)
    if state_match and b is not None:
        return f"{stage}<br>{ev}<br>S{state_match.group(1)} / B{b}"

    basin_name = ""
    if b is not None:
        m = re.search(rf"B{b}:\s*([^|]+)$", label)
        if m:
            basin_name = m.group(1).strip()
        if basin_name:
            return f"{stage}<br>{ev}<br>B{b}: {basin_name}"
        return f"{stage}<br>{ev}<br>B{b}"

    return label.replace(" | ", "<br>")


def label_to_color(label: str, basin_colors: Dict[int, str], mode: str = "basin") -> str:
    """Color a node/link endpoint by basin or event."""
    ev = parse_event(label)
    b = parse_basin(label)
    if mode == "event":
        return EVENT_COLORS.get(ev, "#64748B")
    if b is not None:
        return basin_colors.get(b, "#64748B")
    return "#64748B"



def make_pre_during_post_sankey_from_edges(
    edges: pd.DataFrame,
    title: str = "Pre-CS → During-CS → Post-CS Sankey",
    preset: Optional[str] = "fear4",
    min_count: int = 1,
    min_subject_support_count: int = 0,
    top_k_per_source: Optional[int] = None,
    node_color_by: str = "basin",
    link_color_by: str = "source",
    width: int = 1200,
    height: int = 720,
) -> go.Figure:
    if edges is None or edges.empty:
        raise ValueError("No Sankey edges available.")

    df = filter_sankey_edges(
        edges,
        min_count=min_count,
        min_subject_support_count=min_subject_support_count,
        top_k_per_source=top_k_per_source,
    )
    if df.empty:
        raise ValueError("No Sankey edges remain after filtering.")

    all_basins = sorted(
        set(pd.to_numeric(df.get("source_basin", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
        | set(pd.to_numeric(df.get("target_basin", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist())
    )
    basin_colors = basin_color_map(all_basins, preset=preset)

    labels = sankey_label_sort(set(df["source_label"].astype(str)).union(set(df["target_label"].astype(str))))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    source = df["source_label"].astype(str).map(label_to_idx).tolist()
    target = df["target_label"].astype(str).map(label_to_idx).tolist()
    value = pd.to_numeric(df["count"], errors="coerce").fillna(0).tolist()

    display_labels = [compact_sankey_label(lab) for lab in labels]
    node_colors = [label_to_color(lab, basin_colors, mode=node_color_by) for lab in labels]

    link_colors = []
    for _, r in df.iterrows():
        if link_color_by == "event":
            c = EVENT_COLORS.get(str(r.get("event", "OTHER")), "#64748B")
        elif link_color_by == "target":
            c = label_to_color(str(r["target_label"]), basin_colors, mode="basin")
        else:
            # default: source basin color. This makes the flows much easier to distinguish
            # than event-only coloring when CS+ and CS- are both shown.
            c = label_to_color(str(r["source_label"]), basin_colors, mode="basin")
        link_colors.append(hex_to_rgba(c, 0.42))

    # Stable 3-column node positioning.
    stage_x = {"Pre": 0.05, "During": 0.50, "Post": 0.95}
    grouped = {"Pre": [], "During": [], "Post": [], "Other": []}
    for lab in labels:
        grouped.setdefault(parse_stage(lab), []).append(lab)

    node_x, node_y = [], []
    for lab in labels:
        stage = parse_stage(lab)
        group = grouped.get(stage, [])
        idx = group.index(lab) if lab in group else 0
        n = max(len(group), 1)
        y = (idx + 0.5) / n
        node_x.append(stage_x.get(stage, 0.50))
        node_y.append(y)

    customdata = []
    for lab in labels:
        incoming = df.loc[df["target_label"].astype(str) == lab, "count"].sum()
        outgoing = df.loc[df["source_label"].astype(str) == lab, "count"].sum()
        customdata.append(f"{lab}<br>incoming={incoming}<br>outgoing={outgoing}")

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                label=display_labels,
                x=node_x,
                y=node_y,
                color=node_colors,
                pad=22,
                thickness=20,
                line=dict(color="rgba(17,24,39,0.95)", width=0.7),
                customdata=customdata,
                hovertemplate="%{customdata}<extra></extra>",
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=link_colors,
                customdata=[
                    f"{r.source_label} → {r.target_label}<br>"
                    f"event={r.event}<br>"
                    f"count={r.count}<br>"
                    f"subject support={getattr(r, 'subject_support_count', 'NA')}"
                    for r in df.itertuples(index=False)
                ],
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111827")),
        width=width,
        height=height,
        font=dict(size=12, color="#111827"),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def make_pre_during_post_sankey(
    parsed_dir: Union[str, Path],
    event_col: str = "trial_type_hrf4",
    cohort: str = "merged",
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    event_filter: str = "CS+ and CS-",
    flow_level: str = "basin",
    pre_offset: int = 1,
    post_offset: int = 1,
    preset: Optional[str] = "fear4",
    min_count: int = 1,
    min_subject_support_count: int = 0,
    top_k_per_source: Optional[int] = None,
    node_color_by: str = "basin",
    link_color_by: str = "source",
    width: int = 1200,
    height: int = 720,
) -> go.Figure:
    edges = compute_pre_during_post_edges(
        parsed_dir=parsed_dir,
        event_col=event_col,
        cohort=cohort,
        subject=subject,
        task=task,
        session=session,
        epoch_col=epoch_col,
        epoch_mode=epoch_mode,
        event_filter=event_filter,
        flow_level=flow_level,
        pre_offset=pre_offset,
        post_offset=post_offset,
        preset=preset,
    )
    title = (
        f"Pre-CS → During-CS → Post-CS Sankey | {cohort} | "
        f"{event_filter} | {event_col} | epoch={epoch_mode}"
    )
    return make_pre_during_post_sankey_from_edges(
        edges,
        title=dict(text=title, font=dict(size=16, color="#111827")),
        preset=preset,
        min_count=min_count,
        min_subject_support_count=min_subject_support_count,
        top_k_per_source=top_k_per_source,
        node_color_by=node_color_by,
        link_color_by=link_color_by,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create pre-CS/during-CS/post-CS Sankey diagram from parsed ELAT outputs.")
    p.add_argument("--parsed-dir", required=True)
    p.add_argument("--event-col", default="trial_type_hrf4")
    p.add_argument("--cohort", default="merged", choices=["merged", "HC", "PTSD"])
    p.add_argument("--subject", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--epoch-col", default=None)
    p.add_argument("--epoch-mode", default="full", choices=["full", "early", "late"])
    p.add_argument("--event-filter", default="CS+ and CS-", choices=["CS+ and CS-", "CS+", "CS-"])
    p.add_argument("--flow-level", default="basin", choices=["basin", "state"])
    p.add_argument("--pre-offset", type=int, default=1)
    p.add_argument("--post-offset", type=int, default=1)
    p.add_argument("--preset", default="fear4", choices=["fear4", "none"])
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--min-subject-support-count", type=int, default=0)
    p.add_argument("--top-k-per-source", type=int, default=0, help="0 means no top-k filtering.")
    p.add_argument("--node-color-by", default="basin", choices=["basin", "event"])
    p.add_argument("--link-color-by", default="source", choices=["source", "target", "event"])
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--output", default=None)
    p.add_argument("--edges-output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    preset = None if args.preset == "none" else args.preset
    top_k = None if args.top_k_per_source in [None, 0] else int(args.top_k_per_source)

    edges = compute_pre_during_post_edges(
        parsed_dir=parsed_dir,
        event_col=args.event_col,
        cohort=args.cohort,
        subject=args.subject,
        task=args.task,
        session=args.session,
        epoch_col=args.epoch_col,
        epoch_mode=args.epoch_mode,
        event_filter=args.event_filter,
        flow_level=args.flow_level,
        pre_offset=args.pre_offset,
        post_offset=args.post_offset,
        preset=preset,
    )

    if edges.empty:
        raise SystemExit("No edges generated. Check event column, epoch mode, and filters.")

    edges_filtered = filter_sankey_edges(
        edges,
        min_count=args.min_count,
        min_subject_support_count=args.min_subject_support_count,
        top_k_per_source=top_k,
    )

    fig = make_pre_during_post_sankey_from_edges(
        edges,
        title=f"Pre-CS → During-CS → Post-CS Sankey | {args.cohort}",
        preset=preset,
        min_count=args.min_count,
        min_subject_support_count=args.min_subject_support_count,
        top_k_per_source=top_k,
        node_color_by=args.node_color_by,
        link_color_by=args.link_color_by,
        width=args.width,
        height=args.height,
    )

    output = Path(args.output) if args.output else parsed_dir / "pre_during_post_cs_sankey_v1_2.html"
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote: {output}")

    edges_output = Path(args.edges_output) if args.edges_output else parsed_dir / "pre_during_post_cs_sankey_edges_v1_2.csv"
    edges_filtered.to_csv(edges_output, index=False)
    print(f"Wrote: {edges_output}")

    stage_pct = compute_stage_basin_percentages(
        parsed_dir=parsed_dir,
        event_col=args.event_col,
        cohort=args.cohort,
        subject=args.subject,
        task=args.task,
        session=args.session,
        epoch_col=args.epoch_col,
        epoch_mode=args.epoch_mode,
        event_filter=args.event_filter,
        pre_offset=args.pre_offset,
        post_offset=args.post_offset,
        preset=preset,
    )
    stage_pct_output = parsed_dir / "pre_during_post_cs_stage_basin_percentages_v1_2.csv"
    stage_pct.to_csv(stage_pct_output, index=False)
    print(f"Wrote: {stage_pct_output}")

    diff_df = compute_csplus_minus_csminus_stage_difference(stage_pct)
    diff_output = parsed_dir / "pre_during_post_cs_stage_cs_difference_v1_2.csv"
    diff_df.to_csv(diff_output, index=False)
    print(f"Wrote: {diff_output}")


if __name__ == "__main__":
    main()
