#!/usr/bin/env python3
"""
figures_node_map_v2_1_1_elatvis.py

Event-enriched node map for ELAT-Vis.

v2.1 changes
------------
1. Categorical event fill now uses ENRICHED event, not dominant event.
   enriched_event = argmax(state_event_prop - global_event_prop)

2. Improved legend/colorbar layout to reduce overlap.

3. Robust CS-delta scaling:
   --cs-delta-range 0.2
   --color-range-mode fixed|robust|full

4. Subject-support transition filtering:
   --min-subject-support-count N
   --min-subject-support-prop P

Core visual encodings
---------------------
- Node position: deterministic basin layout.
- Node size: state occupancy.
- Node border: basin identity.
- Node shape: diamond = local minimum, circle = other state.
- Node fill:
    cs_delta       = prop(CS+) - prop(CS-), gradient blue-white-red
    enriched_event = most overrepresented event relative to global event proportions
    iti_prop       = ITI participation, grayscale
- Edges:
    observed = observed state transitions
    descent  = ELAT descent reference
    both     = both

Streamlit-ready:
    from figures_node_map_v2_1_elatvis import make_event_enriched_node_map
    fig = make_event_enriched_node_map(parsed_dir="...", event_col="trial_type_hrf4")
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


EVENT_COLORS = {
    "CS+": "#D95F5F",
    "CS-": "#4C78A8",
    "ITI": "#5F6368",
    "NONE": "#E8E8E8",
    "MIXED": "#F7F7F7",
}

BASIN_LABELS_FEAR4 = {
    1: "Global off",
    2: "Threat-like",
    3: "Safety-like",
    4: "Global on",
}

BASIN_BORDER_COLORS_FEAR4 = {
    1: "#A8D5A2",  # light green
    2: "#D95F5F",  # red
    3: "#4C78A8",  # blue
    4: "#2E7D32",  # dark green
}

BASIN_FILL_COLORS_FEAR4 = {
    1: "rgba(168,213,162,0.13)",
    2: "rgba(217,95,95,0.10)",
    3: "rgba(76,120,168,0.10)",
    4: "rgba(46,125,50,0.10)",
}

FEAR4_BASIN_CENTERS = {
    2: (-4.2, 3.2),   # Threat-like
    1: (4.2, 3.2),    # Global off
    3: (-4.2, -3.2),  # Safety-like
    4: (4.2, -3.2),   # Global on
}

DEFAULT_BASIN_BORDER_COLORS = [
    "#A8D5A2", "#D95F5F", "#4C78A8", "#2E7D32", "#B07AA1",
    "#F28E2B", "#76B7B2", "#EDC948", "#9C755F", "#7F7F7F"
]

DEFAULT_BASIN_FILL_COLORS = [
    "rgba(168,213,162,0.10)", "rgba(217,95,95,0.10)", "rgba(76,120,168,0.10)",
    "rgba(46,125,50,0.10)", "rgba(176,122,161,0.10)", "rgba(242,142,43,0.10)",
    "rgba(118,183,178,0.10)", "rgba(237,201,72,0.10)", "rgba(156,117,95,0.10)",
    "rgba(127,127,127,0.10)"
]

CS_DELTA_COLORSCALE = [
    [0.0, "#4C78A8"],
    [0.5, "#F7F7F7"],
    [1.0, "#D95F5F"],
]

ITI_COLORSCALE = [
    [0.0, "#F4F4F4"],
    [1.0, "#5F6368"],
]

GROUP_DELTA_COLORSCALE = [
    [0.0, "#4C78A8"],
    [0.5, "#F7F7F7"],
    [1.0, "#D95F5F"],
]


def read_csv_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def load_parsed_tables(parsed_dir: Union[str, Path]) -> Dict[str, Optional[pd.DataFrame]]:
    parsed_dir = Path(parsed_dir)
    metadata = {}
    metadata_path = parsed_dir / "model_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    return {
        "states": read_csv_if_exists(parsed_dir / "states_table.csv"),
        "timeline": read_csv_if_exists(parsed_dir / "timeline_table.csv"),
        "basin_graph": read_csv_if_exists(parsed_dir / "basin_graph.csv"),
        "basins": read_csv_if_exists(parsed_dir / "basins_table.csv"),
        "membership": read_csv_if_exists(parsed_dir / "state_basin_membership.csv"),
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
    s2 = s.upper().replace(" ", "")
    if s2 in {"CS+", "CSP", "CSPLUS", "CS_POS", "THREAT"}:
        return "CS+"
    if s2 in {"CS-", "CSMINUS", "CSNEG", "CS_NEG", "SAFETY"}:
        return "CS-"
    if s2 in {"ITI", "INTERTRIAL", "INTERTRIALINTERVAL"}:
        return "ITI"
    if "CS+" in s or "CSPLUS" in s.upper() or "CS_PLUS" in s.upper():
        return "CS+"
    if "CS-" in s or "CSMINUS" in s.upper() or "CS_MINUS" in s.upper():
        return "CS-"
    if "ITI" in s.upper():
        return "ITI"
    return s


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


def filter_timeline(
    timeline: pd.DataFrame,
    subject: Optional[str] = None,
    group: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    event_col: Optional[str] = None,
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
) -> pd.DataFrame:
    df = timeline.copy()
    for col in ["subject", "group", "task", "session"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if subject and "subject" in df.columns:
        df = df[df["subject"].astype(str) == str(subject)]
    if group and "group" in df.columns:
        df = df[df["group"].astype(str) == str(group)]
    if task and "task" in df.columns:
        df = df[df["task"].astype(str) == str(task)]
    if session and "session" in df.columns:
        df = df[df["session"].astype(str) == str(session)]

    if epoch_col is None:
        epoch_col = find_epoch_col(df, event_col)
    if epoch_mode.lower() != "full" and epoch_col in df.columns:
        df = df[df[epoch_col].apply(normalize_epoch_value) == epoch_mode.lower()]

    return df


def infer_n_rois(states: Optional[pd.DataFrame], metadata: dict) -> Optional[int]:
    if isinstance(metadata, dict):
        for key in ["nodeNumber", "node_number", "n_rois", "N"]:
            if key in metadata:
                try:
                    return int(metadata[key])
                except Exception:
                    pass

    if states is not None and "binary" in states.columns:
        vals = states["binary"].dropna().astype(str)
        if len(vals):
            return int(vals.str.len().max())

    if states is not None and "state" in states.columns:
        mx = pd.to_numeric(states["state"], errors="coerce").max()
        if pd.notna(mx) and mx > 0:
            return int(math.ceil(math.log2(int(mx))))

    return None


def state_to_binary(state: int, n_rois: int) -> str:
    return format(int(state) - 1, f"0{n_rois}b")


def add_binary_if_missing(states: pd.DataFrame, n_rois: Optional[int]) -> pd.DataFrame:
    states = states.copy()
    if "binary_pattern" in states.columns and "binary" not in states.columns:
        states = states.rename(columns={"binary_pattern": "binary"})
    if n_rois is not None and "binary" not in states.columns:
        states["binary"] = states["state"].apply(
            lambda s: state_to_binary(int(s), n_rois) if pd.notna(s) else None
        )
    return states


def hamming_distance(a: str, b: str) -> Optional[int]:
    if a is None or b is None:
        return None
    a, b = str(a), str(b)
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def basin_labels_map(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_LABELS_FEAR4.copy()
        for b in basins:
            out.setdefault(b, f"B{b}")
        return out
    return {b: f"B{b}" for b in basins}


def basin_border_palette(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_BORDER_COLORS_FEAR4.copy()
        for i, b in enumerate(basins):
            out.setdefault(b, DEFAULT_BASIN_BORDER_COLORS[i % len(DEFAULT_BASIN_BORDER_COLORS)])
        return out
    return {b: DEFAULT_BASIN_BORDER_COLORS[i % len(DEFAULT_BASIN_BORDER_COLORS)] for i, b in enumerate(basins)}


def basin_fill_palette(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_FILL_COLORS_FEAR4.copy()
        for i, b in enumerate(basins):
            out.setdefault(b, DEFAULT_BASIN_FILL_COLORS[i % len(DEFAULT_BASIN_FILL_COLORS)])
        return out
    return {b: DEFAULT_BASIN_FILL_COLORS[i % len(DEFAULT_BASIN_FILL_COLORS)] for i, b in enumerate(basins)}


def find_steepest_neighbor_col(basin_graph: Optional[pd.DataFrame]) -> Optional[str]:
    if basin_graph is None:
        return None
    candidates = [
        "steepest_neighbor", "steepest_descent_neighbor", "next_state",
        "neighbor", "target_state", "to_state", "destination_state", "downhill_state"
    ]
    for c in candidates:
        if c in basin_graph.columns:
            return c

    for c in basin_graph.columns:
        lc = c.lower()
        if c != "state" and ("neighbor" in lc or "target" in lc or "next" in lc or "steepest" in lc):
            return c

    return None


def build_descent_edges(basin_graph: Optional[pd.DataFrame]) -> pd.DataFrame:
    if basin_graph is None or basin_graph.empty:
        return pd.DataFrame(columns=["source", "target"])

    bg = normalize_state_col(basin_graph)
    ncol = find_steepest_neighbor_col(bg)
    if not ncol:
        return pd.DataFrame(columns=["source", "target"])

    out = bg[["state", ncol]].copy()
    out = out.rename(columns={"state": "source", ncol: "target"})
    out["source"] = pd.to_numeric(out["source"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna().astype(int)
    out = out[out["source"] != out["target"]]
    return out.drop_duplicates().reset_index(drop=True)


def prepare_states(
    states: Optional[pd.DataFrame],
    timeline: pd.DataFrame,
    basin_graph: Optional[pd.DataFrame],
    metadata: dict,
) -> pd.DataFrame:
    timeline = normalize_basin_col(normalize_state_col(timeline))

    if states is None or states.empty:
        unique_states = sorted(pd.to_numeric(timeline["state"], errors="coerce").dropna().astype(int).unique())
        states = pd.DataFrame({"state": unique_states})
    else:
        states = normalize_basin_col(normalize_state_col(states))

    n_rois = infer_n_rois(states, metadata)
    states = add_binary_if_missing(states, n_rois)

    if "basin" not in states.columns and "basin" in timeline.columns:
        state_basin = (
            timeline.dropna(subset=["state", "basin"])
            .groupby("state")["basin"]
            .agg(lambda x: int(pd.Series(x).mode().iloc[0]) if len(x) else np.nan)
            .reset_index()
        )
        states = states.merge(state_basin, on="state", how="left")

    if "is_local_minimum" not in states.columns:
        states["is_local_minimum"] = False

    if basin_graph is not None and not basin_graph.empty:
        bg = normalize_basin_col(normalize_state_col(basin_graph))
        local_states = []

        local_cols = [c for c in bg.columns if "local" in c.lower() and "minimum" in c.lower()]
        if local_cols:
            local_states = pd.to_numeric(bg[local_cols[0]], errors="coerce").dropna().astype(int).unique().tolist()

        ncol = find_steepest_neighbor_col(bg)
        if ncol:
            tmp = bg[["state", ncol]].copy()
            tmp["state"] = pd.to_numeric(tmp["state"], errors="coerce")
            tmp[ncol] = pd.to_numeric(tmp[ncol], errors="coerce")
            self_mins = tmp[tmp["state"] == tmp[ncol]]["state"].dropna().astype(int).tolist()
            local_states = sorted(set(local_states).union(self_mins))

        if local_states:
            states["is_local_minimum"] = states["state"].astype("Int64").isin(local_states)

    return states


def compute_state_metrics(timeline: pd.DataFrame, states: pd.DataFrame, event_col: str) -> pd.DataFrame:
    df = normalize_basin_col(normalize_state_col(timeline)).copy()
    if event_col not in df.columns:
        df[event_col] = "NONE"
    df["event_norm"] = df[event_col].apply(normalize_event_value)

    counts = df.groupby(["state", "event_norm"]).size().unstack(fill_value=0).reset_index()
    for ev in ["CS+", "CS-", "ITI", "NONE"]:
        if ev not in counts.columns:
            counts[ev] = 0

    counts["occupancy"] = counts[["CS+", "CS-", "ITI", "NONE"]].sum(axis=1)

    total_n = max(len(df), 1)
    global_counts = df["event_norm"].value_counts().to_dict()
    global_prop = {ev: global_counts.get(ev, 0) / total_n for ev in ["CS+", "CS-", "ITI"]}

    rows = []
    for _, r in counts.iterrows():
        occ = max(float(r["occupancy"]), 1.0)
        props = {
            "CS+": float(r.get("CS+", 0)) / occ,
            "CS-": float(r.get("CS-", 0)) / occ,
            "ITI": float(r.get("ITI", 0)) / occ,
        }
        enrich = {ev: props[ev] - global_prop.get(ev, 0.0) for ev in ["CS+", "CS-", "ITI"]}
        dominant_event = max(props, key=props.get)
        enriched_event = max(enrich, key=enrich.get)
        max_enrichment = enrich[enriched_event]

        rows.append({
            "state": int(r["state"]),
            "occupancy": int(r["occupancy"]),
            "occupancy_prop": float(r["occupancy"]) / total_n,
            "count_CSplus": int(r.get("CS+", 0)),
            "count_CSminus": int(r.get("CS-", 0)),
            "count_ITI": int(r.get("ITI", 0)),
            "prop_CSplus": props["CS+"],
            "prop_CSminus": props["CS-"],
            "prop_ITI": props["ITI"],
            "cs_delta": props["CS+"] - props["CS-"],
            "iti_prop": props["ITI"],
            "dominant_event": dominant_event,
            "enriched_event": enriched_event,
            "max_enrichment": max_enrichment,
            "enrich_CSplus": enrich["CS+"],
            "enrich_CSminus": enrich["CS-"],
            "enrich_ITI": enrich["ITI"],
            "global_prop_CSplus": global_prop.get("CS+", 0.0),
            "global_prop_CSminus": global_prop.get("CS-", 0.0),
            "global_prop_ITI": global_prop.get("ITI", 0.0),
        })

    stats = pd.DataFrame(rows)
    out = states.merge(stats, on="state", how="left")

    for c in [
        "occupancy", "occupancy_prop", "count_CSplus", "count_CSminus", "count_ITI",
        "prop_CSplus", "prop_CSminus", "prop_ITI", "cs_delta", "iti_prop",
        "max_enrichment", "enrich_CSplus", "enrich_CSminus", "enrich_ITI",
        "global_prop_CSplus", "global_prop_CSminus", "global_prop_ITI"
    ]:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    for c in ["dominant_event", "enriched_event"]:
        if c in out.columns:
            out[c] = out[c].fillna("NONE")

    return out


def compute_group_delta_metrics(
    timeline: pd.DataFrame,
    states: pd.DataFrame,
    event_col: str,
    pos_group: str = "PTSD",
    neg_group: str = "HC",
) -> pd.DataFrame:
    if "group" not in timeline.columns:
        raise ValueError("group_delta mode requires a 'group' column in timeline_table.csv")

    t_pos = timeline[timeline["group"].astype(str) == str(pos_group)].copy()
    t_neg = timeline[timeline["group"].astype(str) == str(neg_group)].copy()

    if t_pos.empty or t_neg.empty:
        raise ValueError(f"group_delta mode requires both groups: {pos_group}, {neg_group}")

    pos = compute_state_metrics(t_pos, states, event_col).add_suffix(f"_{pos_group}")
    pos = pos.rename(columns={f"state_{pos_group}": "state"})
    neg = compute_state_metrics(t_neg, states, event_col).add_suffix(f"_{neg_group}")
    neg = neg.rename(columns={f"state_{neg_group}": "state"})

    merged = states.copy()
    merged = merged.merge(pos, on="state", how="left")
    merged = merged.merge(neg, on="state", how="left")
    merged = merged.fillna(0)

    merged["occupancy"] = (
        merged.get(f"occupancy_prop_{pos_group}", 0) -
        merged.get(f"occupancy_prop_{neg_group}", 0)
    ).abs()
    merged["signed_occupancy_delta"] = (
        merged.get(f"occupancy_prop_{pos_group}", 0) -
        merged.get(f"occupancy_prop_{neg_group}", 0)
    )
    merged["cs_delta"] = (
        merged.get(f"cs_delta_{pos_group}", 0) -
        merged.get(f"cs_delta_{neg_group}", 0)
    )
    merged["iti_prop"] = (
        merged.get(f"iti_prop_{pos_group}", 0) -
        merged.get(f"iti_prop_{neg_group}", 0)
    )
    merged["dominant_event"] = "GROUP_DELTA"
    merged["enriched_event"] = "GROUP_DELTA"
    return merged


def compute_observed_transitions(
    timeline: pd.DataFrame,
    x_col: Optional[str] = None,
    hide_self_transitions: bool = True,
) -> pd.DataFrame:
    df = normalize_state_col(timeline).copy()
    if x_col is None:
        x_col = choose_x_col(df)

    sort_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns] + [x_col]
    df = df.sort_values(sort_cols)

    group_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns]
    if not group_cols:
        df["_series"] = "all"
        group_cols = ["_series"]

    edge_rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))

        states = pd.to_numeric(g["state"], errors="coerce").dropna().astype(int).tolist()
        seen_edges_this_series = set()

        for a, b in zip(states[:-1], states[1:]):
            if hide_self_transitions and a == b:
                continue
            pair = (int(a), int(b))
            seen_edges_this_series.add(pair)
            row = {"source": int(a), "target": int(b)}
            row.update(key_dict)
            edge_rows.append(row)

    if not edge_rows:
        return pd.DataFrame(columns=[
            "source", "target", "count", "probability",
            "subject_support_count", "subject_support_prop"
        ])

    raw = pd.DataFrame(edge_rows)
    edf = raw.groupby(["source", "target"]).size().reset_index(name="count")
    totals = edf.groupby("source")["count"].sum().rename("source_total").reset_index()
    edf = edf.merge(totals, on="source", how="left")
    edf["probability"] = edf["count"] / edf["source_total"].replace(0, np.nan)

    if "subject" in raw.columns:
        support = raw.groupby(["source", "target"])["subject"].nunique().reset_index(name="subject_support_count")
        n_subjects = max(raw["subject"].nunique(), 1)
        support["subject_support_prop"] = support["subject_support_count"] / n_subjects
        edf = edf.merge(support, on=["source", "target"], how="left")
    else:
        edf["subject_support_count"] = np.nan
        edf["subject_support_prop"] = np.nan

    return edf.sort_values(["source", "probability", "count"], ascending=[True, False, False]).reset_index(drop=True)


def compute_group_delta_transitions(
    timeline: pd.DataFrame,
    pos_group: str = "PTSD",
    neg_group: str = "HC",
    x_col: Optional[str] = None,
    hide_self_transitions: bool = True,
) -> pd.DataFrame:
    if "group" not in timeline.columns:
        raise ValueError("group_delta mode requires a 'group' column")

    t_pos = timeline[timeline["group"].astype(str) == str(pos_group)].copy()
    t_neg = timeline[timeline["group"].astype(str) == str(neg_group)].copy()

    pos = compute_observed_transitions(t_pos, x_col=x_col, hide_self_transitions=hide_self_transitions)
    neg = compute_observed_transitions(t_neg, x_col=x_col, hide_self_transitions=hide_self_transitions)

    out = pos.merge(neg, on=["source", "target"], how="outer", suffixes=(f"_{pos_group}", f"_{neg_group}")).fillna(0)
    out["count"] = out.get(f"count_{pos_group}", 0) + out.get(f"count_{neg_group}", 0)
    out["probability"] = out.get(f"probability_{pos_group}", 0) - out.get(f"probability_{neg_group}", 0)
    out["abs_probability"] = out["probability"].abs()
    out["subject_support_count"] = (
        out.get(f"subject_support_count_{pos_group}", 0) +
        out.get(f"subject_support_count_{neg_group}", 0)
    )
    out["subject_support_prop"] = np.maximum(
        out.get(f"subject_support_prop_{pos_group}", 0),
        out.get(f"subject_support_prop_{neg_group}", 0),
    )
    return out.sort_values(["source", "abs_probability"], ascending=[True, False]).reset_index(drop=True)


def filter_transition_backbone(
    edges: pd.DataFrame,
    min_transition_count: int = 1,
    min_transition_prob: float = 0.0,
    top_k_per_state: Optional[int] = 3,
    visible_states: Optional[Iterable[int]] = None,
    analysis_mode: str = "standard",
    min_subject_support_count: int = 0,
    min_subject_support_prop: float = 0.0,
) -> pd.DataFrame:
    if edges is None or edges.empty:
        return pd.DataFrame(columns=edges.columns if edges is not None else ["source", "target"])

    edf = edges.copy()

    if visible_states is not None:
        visible = set(int(x) for x in visible_states)
        edf = edf[edf["source"].astype(int).isin(visible) & edf["target"].astype(int).isin(visible)]

    if "count" in edf.columns:
        edf = edf[pd.to_numeric(edf["count"], errors="coerce").fillna(0) >= min_transition_count]

    support_count = pd.to_numeric(edf.get("subject_support_count", 0), errors="coerce").fillna(0)
    support_prop = pd.to_numeric(edf.get("subject_support_prop", 0), errors="coerce").fillna(0)

    if min_subject_support_count > 0:
        edf = edf[support_count >= min_subject_support_count]
    if min_subject_support_prop > 0:
        edf = edf[support_prop >= min_subject_support_prop]

    if analysis_mode == "group_delta":
        score = pd.to_numeric(edf.get("abs_probability", 0), errors="coerce").fillna(0)
        edf = edf[score >= min_transition_prob]
        edf = edf.assign(_score=score)
    else:
        score = pd.to_numeric(edf.get("probability", 0), errors="coerce").fillna(0)
        edf = edf[score >= min_transition_prob]
        edf = edf.assign(_score=score)

    if top_k_per_state is not None and top_k_per_state > 0 and not edf.empty:
        edf = edf.sort_values(["source", "_score", "count"], ascending=[True, False, False])
        edf = edf.groupby("source", group_keys=False).head(int(top_k_per_state))

    return edf.drop(columns=["_score"], errors="ignore").reset_index(drop=True)


def get_basin_centers(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, Tuple[float, float]]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        centers = FEAR4_BASIN_CENTERS.copy()
        for i, b in enumerate(basins):
            centers.setdefault(
                b,
                (6.0 * math.cos(2 * math.pi * i / max(len(basins), 1)),
                 6.0 * math.sin(2 * math.pi * i / max(len(basins), 1)))
            )
        return centers

    return {
        b: (
            6.0 * math.cos(2 * math.pi * i / max(len(basins), 1)),
            6.0 * math.sin(2 * math.pi * i / max(len(basins), 1))
        )
        for i, b in enumerate(basins)
    }


def compute_ring_distances(sub: pd.DataFrame, descent_edges: pd.DataFrame, local_states: List[int]) -> Dict[int, int]:
    states = sub["state"].dropna().astype(int).tolist()
    if not states:
        return {}

    state_set = set(states)
    local_states = [int(s) for s in local_states if int(s) in state_set]
    if not local_states:
        local_states = [min(states)]

    # Graph distance via descent edges if possible.
    adjacency = {s: set() for s in states}
    if descent_edges is not None and not descent_edges.empty:
        for _, r in descent_edges.iterrows():
            s, t = int(r["source"]), int(r["target"])
            if s in state_set and t in state_set:
                adjacency[s].add(t)
                adjacency[t].add(s)

    distances = {}
    queue = [(lm, 0) for lm in local_states]
    seen = set()
    while queue:
        s, d = queue.pop(0)
        if s in seen:
            continue
        seen.add(s)
        distances[s] = d
        for nb in adjacency.get(s, []):
            if nb not in seen:
                queue.append((nb, d + 1))

    if len(distances) == len(states):
        return distances

    # Fallback to Hamming distance from first local minimum.
    binaries = dict(zip(sub["state"].astype(int), sub.get("binary", pd.Series([None] * len(sub))).astype(str)))
    ref = binaries.get(local_states[0], None)
    for s in states:
        if s in distances:
            continue
        if s in local_states:
            distances[s] = 0
        else:
            hd = hamming_distance(binaries.get(s, None), ref)
            distances[s] = int(hd) if hd is not None else 1

    return distances


def compute_layout(
    nodes: pd.DataFrame,
    descent_edges: pd.DataFrame,
    preset: Optional[str] = "fear4",
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, float]]]:
    nodes = normalize_basin_col(normalize_state_col(nodes)).copy()
    basins = sorted(nodes["basin"].dropna().astype(int).unique().tolist()) if "basin" in nodes.columns else [1]
    centers = get_basin_centers(basins, preset=preset)

    pos = {}
    basin_shapes = {}
    inner_radius = 0.85
    ring_spacing = 0.9

    for b in basins:
        sub = nodes[nodes["basin"].astype("Int64") == b].copy()
        if sub.empty:
            continue

        cx, cy = centers.get(int(b), (0.0, 0.0))
        local_states = sub[sub.get("is_local_minimum", False).fillna(False).astype(bool)]["state"].dropna().astype(int).tolist()
        d_map = compute_ring_distances(sub, descent_edges, local_states)

        local_states = sorted(local_states)
        if len(local_states) == 1:
            pos[local_states[0]] = (cx, cy)
        elif len(local_states) > 1:
            for i, s in enumerate(local_states):
                ang = 2 * math.pi * i / len(local_states)
                pos[s] = (cx + 0.25 * math.cos(ang), cy + 0.25 * math.sin(ang))

        non_local = sub[~sub["state"].astype(int).isin(local_states)].copy()
        if not non_local.empty:
            non_local["ring"] = non_local["state"].astype(int).map(lambda s: d_map.get(int(s), 1))
            for ring, g in non_local.groupby("ring"):
                g = g.sort_values("state")
                r = inner_radius + max(int(ring), 1) * ring_spacing
                n = len(g)
                phase = (int(b) * 0.45) + (int(ring) * 0.25)
                for j, (_, row) in enumerate(g.iterrows()):
                    ang = 2 * math.pi * j / max(n, 1) + phase
                    pos[int(row["state"])] = (cx + r * math.cos(ang), cy + r * math.sin(ang))

        max_ring = max(d_map.values()) if d_map else 1
        basin_shapes[int(b)] = {
            "cx": cx,
            "cy": cy,
            "radius": inner_radius + max(max_ring, 1) * ring_spacing + 0.85,
        }

    return pos, basin_shapes


def scale_sizes(values: pd.Series, min_size: float = 10, max_size: float = 42) -> List[float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0).astype(float)
    if vals.max() <= 0:
        return [min_size] * len(vals)
    scaled = np.sqrt(vals / vals.max())
    return (min_size + scaled * (max_size - min_size)).tolist()


def compute_color_range(values: pd.Series, fill_mode: str, analysis_mode: str, mode: str, fixed_range: float) -> Tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if fill_mode == "iti_prop" and analysis_mode != "group_delta":
        return 0.0, 1.0

    if mode == "full":
        return (-1.0, 1.0) if analysis_mode != "group_delta" else (-2.0, 2.0)

    if mode == "fixed":
        r = abs(float(fixed_range))
        return -r, r

    # robust
    if len(vals) == 0:
        r = abs(float(fixed_range))
    else:
        r = float(np.nanquantile(np.abs(vals), 0.90))
        r = max(r, 0.05)
        r = min(r, abs(float(fixed_range)) if fixed_range else r)
    return -r, r


def add_basin_hulls(fig, basin_shapes, basin_fill_colors, basin_labels):
    shapes, annotations = [], []
    for b, info in basin_shapes.items():
        cx, cy, r = info["cx"], info["cy"], info["radius"]
        shapes.append(dict(
            type="circle",
            xref="x", yref="y",
            x0=cx - r, x1=cx + r,
            y0=cy - r, y1=cy + r,
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor=basin_fill_colors.get(int(b), "rgba(0,0,0,0.05)"),
            layer="below",
        ))
        annotations.append(dict(
            x=cx, y=cy + r + 0.25,
            xref="x", yref="y",
            text=f"B{b}: {basin_labels.get(int(b), f'B{b}')}",
            showarrow=False,
            font=dict(size=12, color="#455A64"),
        ))

    fig.update_layout(shapes=shapes, annotations=annotations)


def add_descent_edges(fig, descent_edges, pos):
    if descent_edges is None or descent_edges.empty:
        return

    first = True
    for _, r in descent_edges.iterrows():
        s, t = int(r["source"]), int(r["target"])
        if s not in pos or t not in pos:
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=1.8, color="rgba(108,99,255,0.42)", dash="dot"),
            hovertemplate=f"<b>ELAT descent</b><br>{s} → {t}<extra></extra>",
            name="ELAT descent reference",
            showlegend=first,
            legendgroup="descent",
            legendrank=1,
        ))
        first = False


def add_observed_edges(fig, edges, pos, analysis_mode="standard", pos_group="PTSD", neg_group="HC"):
    if edges is None or edges.empty:
        return

    if analysis_mode == "group_delta":
        score = pd.to_numeric(edges.get("abs_probability", 0), errors="coerce").fillna(0)
    else:
        score = pd.to_numeric(edges.get("probability", 0), errors="coerce").fillna(0)

    vmax = max(float(score.max()), 1e-9)

    seen = {"std": False, "pos": False, "neg": False}
    for _, r in edges.iterrows():
        s, t = int(r["source"]), int(r["target"])
        if s not in pos or t not in pos:
            continue

        x0, y0 = pos[s]
        x1, y1 = pos[t]

        if analysis_mode == "group_delta":
            delta = float(r.get("probability", 0) or 0)
            val = abs(delta)
            color = "rgba(217,95,95,0.56)" if delta >= 0 else "rgba(76,120,168,0.56)"
            key = "pos" if delta >= 0 else "neg"
            name = f"Transition Δ: {pos_group}>{neg_group}" if delta >= 0 else f"Transition Δ: {neg_group}>{pos_group}"
            showlegend = not seen[key]
            seen[key] = True
            hover = (
                f"<b>Transition difference</b><br>"
                f"{s} → {t}<br>"
                f"Δ probability ({pos_group}−{neg_group}): {delta:.3f}<br>"
                f"support count: {float(r.get('subject_support_count', 0)):.0f}<br>"
                f"support prop: {float(r.get('subject_support_prop', 0)):.2f}"
            )
        else:
            val = float(r.get("probability", 0) or 0)
            color = "rgba(45,45,45,0.45)"
            name = "Observed state transitions"
            showlegend = not seen["std"]
            seen["std"] = True
            hover = (
                f"<b>Observed transition</b><br>"
                f"{s} → {t}<br>"
                f"count: {int(r.get('count', 0))}<br>"
                f"probability: {val:.3f}<br>"
                f"subject support count: {float(r.get('subject_support_count', 0)):.0f}<br>"
                f"subject support prop: {float(r.get('subject_support_prop', 0)):.2f}"
            )

        width = 0.9 + (val / vmax) * (7.5 - 0.9)

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=width, color=color),
            hovertemplate=hover + "<extra></extra>",
            name=name,
            showlegend=showlegend,
            legendgroup=name,
            legendrank=2,
        ))


def add_basin_legend(fig, basin_labels, basin_colors):
    for b in sorted(basin_labels):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=f"B{b}: {basin_labels[b]}",
            marker=dict(size=13, color="#FFFFFF", line=dict(color=basin_colors.get(int(b), "#777"), width=3)),
            showlegend=True,
            legendgroup=f"basin_{b}",
            legendrank=10 + int(b),
        ))


def add_categorical_fill_legend(fig):
    for ev in ["CS+", "CS-", "ITI", "NONE"]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            name=f"Enriched event: {ev}",
            marker=dict(size=12, color=EVENT_COLORS.get(ev, "#FFFFFF"), line=dict(color="#555", width=1)),
            showlegend=True,
            legendgroup="fill",
            legendrank=30,
        ))


def add_local_minimum_legend(fig):
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        name="Local minimum",
        marker=dict(symbol="diamond", size=14, color="#FFFFFF", line=dict(color="#111111", width=3)),
        showlegend=True,
        legendrank=20,
    ))


def make_event_enriched_node_map(
    parsed_dir: Union[str, Path],
    subject: Optional[str] = None,
    group: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    event_col: str = "trial_type_hrf4",
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    preset: Optional[str] = "fear4",
    fill_mode: str = "cs_delta",
    analysis_mode: str = "standard",
    delta_pos_group: str = "PTSD",
    delta_neg_group: str = "HC",
    color_range_mode: str = "fixed",
    cs_delta_range: float = 0.2,
    min_transition_count: int = 1,
    min_transition_prob: float = 0.0,
    top_k_per_state: Optional[int] = 3,
    min_subject_support_count: int = 0,
    min_subject_support_prop: float = 0.0,
    hide_self_transitions: bool = True,
    edge_mode: str = "both",
    min_occupancy: int = 0,
    min_occupancy_prop: float = 0.0,
    preserve_local_minima: bool = True,
    show_basin_hulls: bool = True,
    width: int = 1450,
    height: int = 920,
    title: Optional[str] = None,
) -> go.Figure:
    tables = load_parsed_tables(parsed_dir)
    timeline = tables["timeline"]
    if timeline is None or timeline.empty:
        raise FileNotFoundError(f"No timeline_table.csv found in {parsed_dir}")

    timeline = normalize_basin_col(normalize_state_col(timeline))
    epoch_col = epoch_col or find_epoch_col(timeline, event_col)

    filt = filter_timeline(
        timeline,
        subject=subject,
        group=group if analysis_mode == "standard" else None,
        task=task,
        session=session,
        event_col=event_col,
        epoch_col=epoch_col,
        epoch_mode=epoch_mode,
    )
    if filt.empty:
        raise ValueError("No timeline rows after filtering. Check filters.")

    states = prepare_states(tables["states"], filt, tables["basin_graph"], tables["metadata"])
    states = normalize_basin_col(normalize_state_col(states))
    states = add_binary_if_missing(states, infer_n_rois(states, tables["metadata"]))

    x_col = choose_x_col(filt)
    descent_edges = build_descent_edges(tables["basin_graph"])

    if analysis_mode == "group_delta":
        nodes = compute_group_delta_metrics(filt, states, event_col, pos_group=delta_pos_group, neg_group=delta_neg_group)
        transitions = compute_group_delta_transitions(
            filt,
            pos_group=delta_pos_group,
            neg_group=delta_neg_group,
            x_col=x_col,
            hide_self_transitions=hide_self_transitions,
        )
    else:
        nodes = compute_state_metrics(filt, states, event_col)
        transitions = compute_observed_transitions(filt, x_col=x_col, hide_self_transitions=hide_self_transitions)

    nodes = normalize_basin_col(normalize_state_col(nodes))
    if "occupancy" not in nodes.columns:
        nodes["occupancy"] = 0
    if "occupancy_prop" not in nodes.columns:
        nodes["occupancy_prop"] = 0

    keep_mask = pd.to_numeric(nodes["occupancy"], errors="coerce").fillna(0) >= min_occupancy
    keep_mask &= pd.to_numeric(nodes.get("occupancy_prop", 0), errors="coerce").fillna(0) >= min_occupancy_prop
    if preserve_local_minima and "is_local_minimum" in nodes.columns:
        keep_mask |= nodes["is_local_minimum"].fillna(False).astype(bool)
    nodes = nodes[keep_mask].copy()
    if nodes.empty:
        raise ValueError("All nodes removed by occupancy filters.")

    visible_states = nodes["state"].dropna().astype(int).tolist()
    transitions = filter_transition_backbone(
        transitions,
        min_transition_count=min_transition_count,
        min_transition_prob=min_transition_prob,
        top_k_per_state=top_k_per_state,
        visible_states=visible_states,
        analysis_mode=analysis_mode,
        min_subject_support_count=min_subject_support_count,
        min_subject_support_prop=min_subject_support_prop,
    )

    pos, basin_shapes = compute_layout(nodes, descent_edges, preset=preset)
    nodes["x"] = nodes["state"].astype(int).map(lambda s: pos.get(int(s), (np.nan, np.nan))[0])
    nodes["y"] = nodes["state"].astype(int).map(lambda s: pos.get(int(s), (np.nan, np.nan))[1])
    nodes = nodes.dropna(subset=["x", "y"]).copy()

    basins = sorted(nodes["basin"].dropna().astype(int).unique().tolist()) if "basin" in nodes.columns else []
    basin_labels = basin_labels_map(basins, preset=preset)
    basin_colors = basin_border_palette(basins, preset=preset)
    basin_fill_colors = basin_fill_palette(basins, preset=preset)

    nodes["node_size"] = scale_sizes(pd.to_numeric(nodes["occupancy"], errors="coerce").fillna(0))

    if fill_mode == "cs_delta":
        nodes["fill_value"] = pd.to_numeric(nodes["cs_delta"], errors="coerce").fillna(0)
        colorscale = GROUP_DELTA_COLORSCALE if analysis_mode == "group_delta" else CS_DELTA_COLORSCALE
        cmin, cmax = compute_color_range(nodes["fill_value"], fill_mode, analysis_mode, color_range_mode, cs_delta_range)
        colorbar_title = (
            f"CS-delta Δ<br>{delta_pos_group}−{delta_neg_group}"
            if analysis_mode == "group_delta"
            else "CS+ minus CS-<br>event enrichment"
        )
        use_continuous = True
    elif fill_mode == "iti_prop":
        nodes["fill_value"] = pd.to_numeric(nodes["iti_prop"], errors="coerce").fillna(0)
        colorscale = ITI_COLORSCALE
        cmin, cmax = compute_color_range(nodes["fill_value"], fill_mode, analysis_mode, color_range_mode, cs_delta_range)
        colorbar_title = "ITI participation"
        use_continuous = True
    else:
        # v2.1: categorical mode uses enriched_event, not dominant_event.
        nodes["fill_color"] = nodes["enriched_event"].map(lambda x: EVENT_COLORS.get(str(x), EVENT_COLORS["NONE"]))
        use_continuous = False
        colorscale = None
        cmin, cmax = None, None
        colorbar_title = ""

    if title is None:
        id_parts = []
        if group and analysis_mode == "standard":
            id_parts.append(str(group))
        if subject:
            id_parts.append(str(subject))
        if session:
            id_parts.append(str(session))
        if task:
            id_parts.append(str(task))
        id_text = " | ".join(id_parts) if id_parts else "merged"
        if analysis_mode == "group_delta":
            id_text = f"Δ {delta_pos_group}−{delta_neg_group}"
        title = f"Event-enriched ELAT node map v2.1: {id_text} | event: {event_col} | epoch: {epoch_mode}"

    fig = go.Figure()

    if show_basin_hulls:
        add_basin_hulls(fig, basin_shapes, basin_fill_colors, basin_labels)

    if edge_mode in {"descent", "both"}:
        add_descent_edges(fig, descent_edges, pos)

    if edge_mode in {"observed", "both"}:
        add_observed_edges(fig, transitions, pos, analysis_mode=analysis_mode, pos_group=delta_pos_group, neg_group=delta_neg_group)

    for b in basins:
        sub = nodes[nodes["basin"].astype("Int64") == b].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("state")

        hover = []
        for _, r in sub.iterrows():
            s = int(r["state"])
            bname = basin_labels.get(int(b), f"B{b}")
            local = "yes" if bool(r.get("is_local_minimum", False)) else "no"

            lines = [
                f"<b>State {s}</b>",
                f"binary: {r.get('binary', 'NA')}",
                f"basin: B{b} ({bname})",
                f"local minimum: {local}",
            ]

            if analysis_mode == "group_delta":
                lines.extend([
                    f"abs occupancy Δ: {float(r.get('occupancy', 0)):.4f}",
                    f"signed occupancy Δ ({delta_pos_group}−{delta_neg_group}): {float(r.get('signed_occupancy_delta', 0)):.4f}",
                    f"CS delta Δ: {float(r.get('cs_delta', 0)):.3f}",
                    f"{delta_pos_group} CS delta: {float(r.get(f'cs_delta_{delta_pos_group}', 0)):.3f}",
                    f"{delta_neg_group} CS delta: {float(r.get(f'cs_delta_{delta_neg_group}', 0)):.3f}",
                ])
            else:
                lines.extend([
                    f"occupancy: {int(r.get('occupancy', 0))}",
                    f"occupancy prop: {float(r.get('occupancy_prop', 0)):.4f}",
                    f"CS+: {int(r.get('count_CSplus', 0))} ({float(r.get('prop_CSplus', 0)):.2f})",
                    f"CS-: {int(r.get('count_CSminus', 0))} ({float(r.get('prop_CSminus', 0)):.2f})",
                    f"ITI: {int(r.get('count_ITI', 0))} ({float(r.get('prop_ITI', 0)):.2f})",
                    f"CS delta: {float(r.get('cs_delta', 0)):.3f}",
                    f"dominant event: {r.get('dominant_event', 'NA')}",
                    f"enriched event: {r.get('enriched_event', 'NA')}",
                    f"max enrichment: {float(r.get('max_enrichment', 0)):.3f}",
                    f"enrich CS+: {float(r.get('enrich_CSplus', 0)):.3f}",
                    f"enrich CS-: {float(r.get('enrich_CSminus', 0)):.3f}",
                    f"enrich ITI: {float(r.get('enrich_ITI', 0)):.3f}",
                ])

            hover.append("<br>".join(lines))

        symbols = np.where(sub.get("is_local_minimum", False).fillna(False).astype(bool), "diamond", "circle")

        marker = dict(
            size=sub["node_size"],
            symbol=symbols,
            opacity=0.95,
            line=dict(color=basin_colors.get(int(b), "#777"), width=3),
        )
        if use_continuous:
            marker["color"] = sub["fill_value"]
            marker["coloraxis"] = "coloraxis"
        else:
            marker["color"] = sub["fill_color"]

        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers+text",
            text=sub["state"].astype(int).astype(str),
            textposition="middle center",
            textfont=dict(size=10, color="#1F2D3D"),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            name=f"B{b}: {basin_labels.get(int(b), f'B{b}')}",
            marker=marker,
            showlegend=False,
        ))

    add_basin_legend(fig, basin_labels, basin_colors)
    add_local_minimum_legend(fig)
    if not use_continuous:
        add_categorical_fill_legend(fig)

    if use_continuous:
        fig.update_layout(
            coloraxis=dict(
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right", font=dict(size=11)),
                    x=1.12,
                    y=0.42,
                    len=0.48,
                    thickness=18,
                    outlinewidth=1,
                    tickfont=dict(size=10),
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=10),
            traceorder="normal",
        ),
        margin=dict(l=25, r=340, t=70, b=25),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False, scaleanchor="x", scaleratio=1)

    return fig


def export_standard_set(parsed_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, **kwargs) -> List[Path]:
    parsed_dir = Path(parsed_dir)
    output_dir = Path(output_dir) if output_dir else parsed_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    configs = [
        ("merged", None, "standard"),
        ("HC", "HC", "standard"),
        ("PTSD", "PTSD", "standard"),
        ("PTSD_minus_HC", None, "group_delta"),
    ]

    for tag, group, mode in configs:
        local_kwargs = dict(kwargs)
        local_kwargs["group"] = group
        local_kwargs["analysis_mode"] = mode
        fig = make_event_enriched_node_map(parsed_dir=parsed_dir, **local_kwargs)
        out = output_dir / f"event_enriched_node_map_v2_1_{tag}.html"
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
        outputs.append(out)

    return outputs


def list_columns(parsed_dir: Union[str, Path]) -> None:
    tables = load_parsed_tables(parsed_dir)
    timeline = tables["timeline"]
    if timeline is None:
        print("No timeline_table.csv found.")
        return

    print("Timeline columns:")
    for c in timeline.columns:
        print(f"  {c}")

    print("\nLikely event columns:")
    for c in timeline.columns:
        lc = c.lower()
        if "trial_type" in lc or "event" in lc or "stimulus" in lc:
            print(f"  {c}")

    print("\nLikely epoch columns:")
    for c in timeline.columns:
        lc = c.lower()
        if "epoch" in lc or "early" in lc or "late" in lc or "block" in lc:
            print(f"  {c}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an event-enriched ELAT node map from parsed ELAT outputs.")
    p.add_argument("--parsed-dir", required=True, help="Directory containing parser.py outputs.")
    p.add_argument("--subject", default=None, help="Subject filter, e.g. FC001.")
    p.add_argument("--group", default=None, help="Group filter for standard mode, e.g. HC or PTSD.")
    p.add_argument("--task", default=None, help="Task filter.")
    p.add_argument("--session", default=None, help="Session filter.")
    p.add_argument("--event-col", default="trial_type_hrf4", help="Event column for CS+/CS-/ITI.")
    p.add_argument("--epoch-col", default=None, help="Matched early/late epoch column. Auto-detected if omitted.")
    p.add_argument("--epoch-mode", default="full", choices=["full", "early", "late"], help="Filter to full/early/late.")
    p.add_argument("--preset", default="fear4", choices=["fear4", "none"], help="Basin label/color preset.")
    p.add_argument("--fill-mode", default="cs_delta", choices=["cs_delta", "iti_prop", "enriched_event"], help="Node fill encoding.")
    p.add_argument("--analysis-mode", default="standard", choices=["standard", "group_delta"], help="Standard map or group-delta map.")
    p.add_argument("--delta-pos-group", default="PTSD", help="Positive group for group-delta mode.")
    p.add_argument("--delta-neg-group", default="HC", help="Negative group for group-delta mode.")

    p.add_argument("--color-range-mode", default="fixed", choices=["fixed", "robust", "full"], help="Color scaling strategy.")
    p.add_argument("--cs-delta-range", type=float, default=0.2, help="Symmetric range for fixed/robust CS-delta color scaling.")

    p.add_argument("--min-transition-count", type=int, default=1, help="Minimum transition count to keep an edge.")
    p.add_argument("--min-transition-prob", type=float, default=0.0, help="Minimum transition probability to keep an edge.")
    p.add_argument("--top-k-per-state", type=int, default=3, help="Keep top-k outgoing transitions per source state.")
    p.add_argument("--min-subject-support-count", type=int, default=0, help="Keep edge only if seen in at least N subjects.")
    p.add_argument("--min-subject-support-prop", type=float, default=0.0, help="Keep edge only if seen in at least this subject proportion.")

    p.add_argument("--show-self-transitions", action="store_true", help="Keep self transitions.")
    p.add_argument("--edge-mode", default="both", choices=["observed", "descent", "both"], help="Which edges to draw.")
    p.add_argument("--min-occupancy", type=int, default=0, help="Minimum occupancy to keep a node.")
    p.add_argument("--min-occupancy-prop", type=float, default=0.0, help="Minimum occupancy proportion to keep a node.")
    p.add_argument("--drop-empty-local-minima", action="store_true", help="Allow empty local minima to be filtered out.")
    p.add_argument("--hide-basin-hulls", action="store_true", help="Hide basin background hulls.")
    p.add_argument("--width", type=int, default=1450)
    p.add_argument("--height", type=int, default=920)
    p.add_argument("--output", default=None, help="Output HTML path.")
    p.add_argument("--list-columns", action="store_true", help="List timeline columns and exit.")
    p.add_argument("--export-standard-set", action="store_true", help="Export merged, HC, PTSD, and PTSD-minus-HC maps.")
    return p.parse_args()


def main():
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    preset = None if args.preset == "none" else args.preset

    if args.list_columns:
        list_columns(parsed_dir)
        return

    common = dict(
        subject=args.subject,
        group=args.group,
        task=args.task,
        session=args.session,
        event_col=args.event_col,
        epoch_col=args.epoch_col,
        epoch_mode=args.epoch_mode,
        preset=preset,
        fill_mode=args.fill_mode,
        analysis_mode=args.analysis_mode,
        delta_pos_group=args.delta_pos_group,
        delta_neg_group=args.delta_neg_group,
        color_range_mode=args.color_range_mode,
        cs_delta_range=args.cs_delta_range,
        min_transition_count=args.min_transition_count,
        min_transition_prob=args.min_transition_prob,
        top_k_per_state=args.top_k_per_state,
        min_subject_support_count=args.min_subject_support_count,
        min_subject_support_prop=args.min_subject_support_prop,
        hide_self_transitions=not args.show_self_transitions,
        edge_mode=args.edge_mode,
        min_occupancy=args.min_occupancy,
        min_occupancy_prop=args.min_occupancy_prop,
        preserve_local_minima=not args.drop_empty_local_minima,
        show_basin_hulls=not args.hide_basin_hulls,
        width=args.width,
        height=args.height,
    )

    if args.export_standard_set:
        export_kwargs = dict(common)
        export_kwargs.pop("group", None)
        export_kwargs.pop("analysis_mode", None)
        outputs = export_standard_set(parsed_dir=parsed_dir, output_dir=parsed_dir, **export_kwargs)
        for out in outputs:
            print(f"Wrote: {out}")
        return

    fig = make_event_enriched_node_map(parsed_dir=parsed_dir, **common)
    output = Path(args.output) if args.output else parsed_dir / "event_enriched_node_map_v2_1_1.html"
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
