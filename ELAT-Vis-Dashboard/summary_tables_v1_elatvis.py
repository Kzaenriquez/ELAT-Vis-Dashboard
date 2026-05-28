#!/usr/bin/env python3
"""
summary_tables_v1_elatvis.py

Generate dashboard-ready summary tables from ELAT-Vis parser outputs.

Input
-----
A parsed ELAT directory, usually:
    <ELAT_DIR>/parsed/

Required files:
    timeline_table.csv
    states_table.csv

Optional files:
    basin_graph.csv
    local_minima_table.csv
    feature_file_manifest.csv
    confirmatory_file_manifest.csv
    dynamics_*_long.csv

Outputs
-------
By default writes to:
    <parsed_dir>/summary_tables/

Core visual-summary tables:
    state_event_summary.csv
    top_csplus_states.csv
    top_csminus_states.csv
    top_enriched_states.csv
    local_minimum_summary.csv
    basin_event_summary.csv
    transition_summary.csv
    transition_backbone_summary.csv
    group_delta_state_summary.csv
    group_delta_transition_summary.csv

Optional evidence/index tables:
    detected_feature_files.csv
    detected_confirmatory_files.csv
    statistical_results_index.csv
    statistical_top_results.csv
    summary_report.md

Interpretation note
-------------------
Transition backbone rows are robust observed transitions filtered by count,
probability, and subject support. They are not inferentially significant unless
linked to a confirmatory/statistical result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EVENT_LEVELS = ["CS+", "CS-", "ITI", "NONE"]

BASIN_LABELS_FEAR4 = {
    1: "Global off",
    2: "Threat-like",
    3: "Safety-like",
    4: "Global on",
}


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------


def read_csv_if_exists(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def write_table(df: Optional[pd.DataFrame], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False)


def safe_read_json(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_parsed_tables(parsed_dir: Union[str, Path]) -> Dict[str, object]:
    parsed_dir = Path(parsed_dir)
    return {
        "states": read_csv_if_exists(parsed_dir / "states_table.csv"),
        "timeline": read_csv_if_exists(parsed_dir / "timeline_table.csv"),
        "basin_graph": read_csv_if_exists(parsed_dir / "basin_graph.csv"),
        "basins": read_csv_if_exists(parsed_dir / "basins_table.csv"),
        "local_minima": read_csv_if_exists(parsed_dir / "local_minima_table.csv"),
        "feature_manifest": read_csv_if_exists(parsed_dir / "feature_file_manifest.csv"),
        "confirmatory_manifest": read_csv_if_exists(parsed_dir / "confirmatory_file_manifest.csv"),
        "metadata": safe_read_json(parsed_dir / "model_metadata.json"),
    }


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------


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


def clean_subject_id(subject: str) -> str:
    return re.sub(r"^sub-", "", str(subject))


def infer_group_from_subject(subject: str) -> Optional[str]:
    subj = clean_subject_id(subject)
    m = re.search(r"FC(\d+)", subj, flags=re.I)
    if not m:
        return None
    return "HC" if int(m.group(1)) < 100 else "PTSD"


def basin_label_for(basin: object, preset: str = "fear4") -> str:
    if pd.isna(basin):
        return "NA"
    try:
        b = int(basin)
    except Exception:
        return str(basin)
    if preset == "fear4":
        return BASIN_LABELS_FEAR4.get(b, f"B{b}")
    return f"B{b}"


# ---------------------------------------------------------------------
# Filtering and state prep
# ---------------------------------------------------------------------


def filter_timeline(
    timeline: pd.DataFrame,
    event_col: str,
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    group: Optional[str] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
) -> pd.DataFrame:
    df = normalize_basin_col(normalize_state_col(timeline)).copy()

    for col in ["subject", "group", "task", "session"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "group" not in df.columns and "subject" in df.columns:
        df["group"] = df["subject"].apply(lambda x: infer_group_from_subject(str(x)) or "UNKNOWN")

    if group and "group" in df.columns:
        df = df[df["group"].astype(str) == str(group)]
    if subject and "subject" in df.columns:
        df = df[df["subject"].astype(str) == str(subject)]
    if task and "task" in df.columns:
        df = df[df["task"].astype(str) == str(task)]
    if session and "session" in df.columns:
        df = df[df["session"].astype(str) == str(session)]

    if event_col not in df.columns:
        df[event_col] = "NONE"
    df["event_norm"] = df[event_col].apply(normalize_event_value)

    epoch_col = epoch_col or find_epoch_col(df, event_col)
    if epoch_col and epoch_col in df.columns:
        df["epoch_norm"] = df[epoch_col].apply(normalize_epoch_value)
    else:
        df["epoch_norm"] = "none"

    if epoch_mode.lower() != "full":
        df = df[df["epoch_norm"] == epoch_mode.lower()]

    x_col = choose_x_col(df)
    sort_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns] + [x_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def prepare_states(states: Optional[pd.DataFrame], timeline: pd.DataFrame, preset: str = "fear4") -> pd.DataFrame:
    timeline = normalize_basin_col(normalize_state_col(timeline))
    if states is None or states.empty:
        unique_states = sorted(pd.to_numeric(timeline["state"], errors="coerce").dropna().astype(int).unique())
        states = pd.DataFrame({"state": unique_states})
    else:
        states = normalize_basin_col(normalize_state_col(states))

    if "basin" not in states.columns and "basin" in timeline.columns:
        state_basin = (
            timeline.dropna(subset=["state", "basin"])
            .groupby("state")["basin"]
            .agg(lambda x: int(pd.Series(x).mode().iloc[0]) if len(x) else np.nan)
            .reset_index()
        )
        states = states.merge(state_basin, on="state", how="left")

    if "is_local_minimum" not in states.columns:
        if "local_minimum_state" in states.columns:
            states["is_local_minimum"] = states["state"].astype("Int64") == states["local_minimum_state"].astype("Int64")
        else:
            states["is_local_minimum"] = False

    if "binary" not in states.columns:
        for c in ["binary_01", "binary_pattern"]:
            if c in states.columns:
                states["binary"] = states[c]
                break

    states["basin_label_readable"] = states["basin"].apply(lambda x: basin_label_for(x, preset=preset)) if "basin" in states.columns else "NA"
    return states


# ---------------------------------------------------------------------
# State / basin event metrics
# ---------------------------------------------------------------------


def compute_state_event_summary(
    timeline: pd.DataFrame,
    states: pd.DataFrame,
    event_col: str,
    cohort: str = "merged",
    preset: str = "fear4",
) -> pd.DataFrame:
    df = normalize_basin_col(normalize_state_col(timeline)).copy()
    if event_col not in df.columns:
        df[event_col] = "NONE"
    if "event_norm" not in df.columns:
        df["event_norm"] = df[event_col].apply(normalize_event_value)

    counts = df.groupby(["state", "event_norm"]).size().unstack(fill_value=0).reset_index()
    for ev in EVENT_LEVELS:
        if ev not in counts.columns:
            counts[ev] = 0
    counts["occupancy"] = counts[EVENT_LEVELS].sum(axis=1)

    total_n = max(len(df), 1)
    global_counts = df["event_norm"].value_counts().to_dict()
    global_prop = {ev: global_counts.get(ev, 0) / total_n for ev in ["CS+", "CS-", "ITI"]}

    rows = []
    for _, r in counts.iterrows():
        occ = max(float(r["occupancy"]), 1.0)
        props = {ev: float(r.get(ev, 0)) / occ for ev in ["CS+", "CS-", "ITI"]}
        enrich = {ev: props[ev] - global_prop.get(ev, 0.0) for ev in ["CS+", "CS-", "ITI"]}
        dominant_event = max(props, key=props.get)
        enriched_event = max(enrich, key=enrich.get)
        rows.append({
            "cohort": cohort,
            "state": int(r["state"]),
            "occupancy": int(r["occupancy"]),
            "occupancy_prop": float(r["occupancy"]) / total_n,
            "count_CSplus": int(r.get("CS+", 0)),
            "count_CSminus": int(r.get("CS-", 0)),
            "count_ITI": int(r.get("ITI", 0)),
            "count_NONE": int(r.get("NONE", 0)),
            "prop_CSplus": props["CS+"],
            "prop_CSminus": props["CS-"],
            "prop_ITI": props["ITI"],
            "cs_delta": props["CS+"] - props["CS-"],
            "dominant_event": dominant_event,
            "enriched_event": enriched_event,
            "max_enrichment": enrich[enriched_event],
            "enrich_CSplus": enrich["CS+"],
            "enrich_CSminus": enrich["CS-"],
            "enrich_ITI": enrich["ITI"],
            "global_prop_CSplus": global_prop.get("CS+", 0.0),
            "global_prop_CSminus": global_prop.get("CS-", 0.0),
            "global_prop_ITI": global_prop.get("ITI", 0.0),
        })

    summary = pd.DataFrame(rows)
    state_cols = [
        c for c in [
            "state", "basin", "basin_label", "basin_label_readable", "local_minimum_state",
            "is_local_minimum", "hamming_to_min", "energy", "binary", "binary_01", "sigma_pattern"
        ] if c in states.columns
    ]
    summary = summary.merge(states[state_cols].drop_duplicates("state"), on="state", how="left")
    if "basin_label_readable" not in summary.columns and "basin" in summary.columns:
        summary["basin_label_readable"] = summary["basin"].apply(lambda x: basin_label_for(x, preset=preset))
    return summary.sort_values(["cohort", "occupancy"], ascending=[True, False]).reset_index(drop=True)


def compute_basin_event_summary(state_summary: pd.DataFrame) -> pd.DataFrame:
    if state_summary is None or state_summary.empty or "basin" not in state_summary.columns:
        return pd.DataFrame()
    rows = []
    for (cohort, basin), g in state_summary.groupby(["cohort", "basin"], dropna=False):
        occ = float(g["occupancy"].sum())
        if occ <= 0:
            continue
        count_csplus = float(g["count_CSplus"].sum())
        count_csminus = float(g["count_CSminus"].sum())
        count_iti = float(g["count_ITI"].sum())
        prop_csplus = count_csplus / occ
        prop_csminus = count_csminus / occ
        prop_iti = count_iti / occ
        rows.append({
            "cohort": cohort,
            "basin": basin,
            "basin_label_readable": basin_label_for(basin),
            "occupancy": int(occ),
            "n_states_occupied": int((g["occupancy"] > 0).sum()),
            "count_CSplus": int(count_csplus),
            "count_CSminus": int(count_csminus),
            "count_ITI": int(count_iti),
            "prop_CSplus": prop_csplus,
            "prop_CSminus": prop_csminus,
            "prop_ITI": prop_iti,
            "cs_delta": prop_csplus - prop_csminus,
        })
    return pd.DataFrame(rows).sort_values(["cohort", "basin"]).reset_index(drop=True)


def make_top_tables(state_summary: pd.DataFrame, top_n: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if state_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    core_cols = [c for c in [
        "cohort", "state", "basin", "basin_label_readable", "is_local_minimum", "occupancy",
        "occupancy_prop", "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI",
        "enriched_event", "max_enrichment", "enrich_CSplus", "enrich_CSminus", "enrich_ITI",
        "binary", "binary_01", "sigma_pattern"
    ] if c in state_summary.columns]

    top_plus = (
        state_summary.sort_values(["cohort", "cs_delta", "occupancy"], ascending=[True, False, False])
        .groupby("cohort", group_keys=False)
        .head(top_n)[core_cols]
        .reset_index(drop=True)
    )
    top_minus = (
        state_summary.sort_values(["cohort", "cs_delta", "occupancy"], ascending=[True, True, False])
        .groupby("cohort", group_keys=False)
        .head(top_n)[core_cols]
        .reset_index(drop=True)
    )
    top_enriched = (
        state_summary.sort_values(["cohort", "max_enrichment", "occupancy"], ascending=[True, False, False])
        .groupby("cohort", group_keys=False)
        .head(top_n)[core_cols]
        .reset_index(drop=True)
    )
    return top_plus, top_minus, top_enriched


def compute_local_minimum_summary(state_summary: pd.DataFrame) -> pd.DataFrame:
    if state_summary.empty or "is_local_minimum" not in state_summary.columns:
        return pd.DataFrame()
    lm = state_summary[state_summary["is_local_minimum"].astype(bool)].copy()
    cols = [c for c in [
        "cohort", "state", "basin", "basin_label_readable", "occupancy", "occupancy_prop",
        "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI", "enriched_event", "max_enrichment",
        "energy", "binary", "binary_01", "sigma_pattern"
    ] if c in lm.columns]
    return lm[cols].sort_values(["cohort", "basin"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Transition metrics
# ---------------------------------------------------------------------


def compute_transitions(timeline: pd.DataFrame, cohort: str, hide_self_transitions: bool = True) -> pd.DataFrame:
    df = normalize_basin_col(normalize_state_col(timeline)).copy()
    if df.empty:
        return pd.DataFrame()
    x_col = choose_x_col(df)
    sort_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns] + [x_col]
    df = df.sort_values(sort_cols)

    group_cols = [c for c in ["group", "subject", "session", "task"] if c in df.columns]
    if not group_cols:
        df["_series"] = "all"
        group_cols = ["_series"]

    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))
        g = g.dropna(subset=["state"]).copy()
        states = g["state"].astype(int).tolist()
        basins = g["basin"].astype("Int64").tolist() if "basin" in g.columns else [pd.NA] * len(g)
        events = g["event_norm"].tolist() if "event_norm" in g.columns else ["NONE"] * len(g)
        epochs = g["epoch_norm"].tolist() if "epoch_norm" in g.columns else ["none"] * len(g)

        for i in range(len(states) - 1):
            s, t = int(states[i]), int(states[i + 1])
            if hide_self_transitions and s == t:
                continue
            row = {
                "cohort": cohort,
                "source": s,
                "target": t,
                "source_basin": basins[i],
                "target_basin": basins[i + 1],
                "source_event": events[i],
                "target_event": events[i + 1],
                "source_epoch": epochs[i],
                "target_epoch": epochs[i + 1],
            }
            row.update(key_dict)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "cohort", "source", "target", "count", "probability", "subject_support_count", "subject_support_prop"
        ])

    raw = pd.DataFrame(rows)
    agg = raw.groupby(["cohort", "source", "target"], dropna=False).agg(
        count=("source", "size"),
        source_basin=("source_basin", lambda x: pd.Series(x).mode().iloc[0] if len(pd.Series(x).mode()) else pd.NA),
        target_basin=("target_basin", lambda x: pd.Series(x).mode().iloc[0] if len(pd.Series(x).mode()) else pd.NA),
        most_common_source_event=("source_event", lambda x: pd.Series(x).mode().iloc[0] if len(pd.Series(x).mode()) else "NA"),
        most_common_target_event=("target_event", lambda x: pd.Series(x).mode().iloc[0] if len(pd.Series(x).mode()) else "NA"),
    ).reset_index()

    totals = agg.groupby(["cohort", "source"])["count"].sum().reset_index(name="source_total")
    agg = agg.merge(totals, on=["cohort", "source"], how="left")
    agg["probability"] = agg["count"] / agg["source_total"].replace(0, np.nan)

    if "subject" in raw.columns:
        support = raw.groupby(["cohort", "source", "target"])["subject"].nunique().reset_index(name="subject_support_count")
        n_subjects = raw.groupby("cohort")["subject"].nunique().reset_index(name="n_subjects_in_cohort")
        support = support.merge(n_subjects, on="cohort", how="left")
        support["subject_support_prop"] = support["subject_support_count"] / support["n_subjects_in_cohort"].replace(0, np.nan)
        agg = agg.merge(support, on=["cohort", "source", "target"], how="left")
    else:
        agg["subject_support_count"] = np.nan
        agg["n_subjects_in_cohort"] = np.nan
        agg["subject_support_prop"] = np.nan

    agg["source_basin_label"] = agg["source_basin"].apply(basin_label_for)
    agg["target_basin_label"] = agg["target_basin"].apply(basin_label_for)
    return agg.sort_values(["cohort", "probability", "count"], ascending=[True, False, False]).reset_index(drop=True)


def filter_transition_backbone(
    transitions: pd.DataFrame,
    min_count: int = 1,
    min_probability: float = 0.0,
    min_subject_support_count: int = 0,
    min_subject_support_prop: float = 0.0,
    top_k_per_state: Optional[int] = 3,
) -> pd.DataFrame:
    if transitions is None or transitions.empty:
        return pd.DataFrame()
    df = transitions.copy()
    df = df[pd.to_numeric(df["count"], errors="coerce").fillna(0) >= min_count]
    df = df[pd.to_numeric(df["probability"], errors="coerce").fillna(0) >= min_probability]
    if min_subject_support_count > 0 and "subject_support_count" in df.columns:
        df = df[pd.to_numeric(df["subject_support_count"], errors="coerce").fillna(0) >= min_subject_support_count]
    if min_subject_support_prop > 0 and "subject_support_prop" in df.columns:
        df = df[pd.to_numeric(df["subject_support_prop"], errors="coerce").fillna(0) >= min_subject_support_prop]
    if top_k_per_state is not None and top_k_per_state > 0 and not df.empty:
        df = df.sort_values(["cohort", "source", "probability", "count"], ascending=[True, True, False, False])
        df = df.groupby(["cohort", "source"], group_keys=False).head(int(top_k_per_state))
    return df.sort_values(["cohort", "probability", "count"], ascending=[True, False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Group-delta tables
# ---------------------------------------------------------------------


def compute_group_delta_state_summary(state_summary: pd.DataFrame, pos_group: str = "PTSD", neg_group: str = "HC") -> pd.DataFrame:
    if state_summary.empty:
        return pd.DataFrame()
    pos = state_summary[state_summary["cohort"] == pos_group].copy()
    neg = state_summary[state_summary["cohort"] == neg_group].copy()
    if pos.empty or neg.empty:
        return pd.DataFrame()

    keep = [
        "state", "basin", "basin_label_readable", "is_local_minimum", "occupancy", "occupancy_prop",
        "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI", "enriched_event", "max_enrichment",
        "binary", "binary_01", "sigma_pattern"
    ]
    keep_pos = [c for c in keep if c in pos.columns]
    keep_neg = [c for c in keep if c in neg.columns]
    merged = pos[keep_pos].merge(neg[keep_neg], on="state", how="outer", suffixes=(f"_{pos_group}", f"_{neg_group}"))

    # Coalesce identity columns.
    for c in ["basin", "basin_label_readable", "is_local_minimum", "binary", "binary_01", "sigma_pattern"]:
        cp = f"{c}_{pos_group}"
        cn = f"{c}_{neg_group}"
        if cp in merged.columns or cn in merged.columns:
            merged[c] = merged.get(cp, pd.Series([np.nan] * len(merged))).combine_first(merged.get(cn, pd.Series([np.nan] * len(merged))))

    for metric in ["occupancy", "occupancy_prop", "cs_delta", "prop_CSplus", "prop_CSminus", "prop_ITI", "max_enrichment"]:
        cp = f"{metric}_{pos_group}"
        cn = f"{metric}_{neg_group}"
        if cp in merged.columns and cn in merged.columns:
            merged[f"{pos_group}_minus_{neg_group}_{metric}"] = pd.to_numeric(merged[cp], errors="coerce").fillna(0) - pd.to_numeric(merged[cn], errors="coerce").fillna(0)

    ordered = [
        "state", "basin", "basin_label_readable", "is_local_minimum",
        f"cs_delta_{pos_group}", f"cs_delta_{neg_group}", f"{pos_group}_minus_{neg_group}_cs_delta",
        f"occupancy_prop_{pos_group}", f"occupancy_prop_{neg_group}", f"{pos_group}_minus_{neg_group}_occupancy_prop",
        f"occupancy_{pos_group}", f"occupancy_{neg_group}", f"{pos_group}_minus_{neg_group}_occupancy",
        f"enriched_event_{pos_group}", f"enriched_event_{neg_group}",
        "binary", "binary_01", "sigma_pattern",
    ]
    ordered = [c for c in ordered if c in merged.columns]
    rest = [c for c in merged.columns if c not in ordered and not re.search(r"_(PTSD|HC)$", c)]
    out = merged[ordered + rest]
    sort_col = f"{pos_group}_minus_{neg_group}_cs_delta"
    if sort_col in out.columns:
        out = out.assign(abs_cs_delta_difference=out[sort_col].abs()).sort_values("abs_cs_delta_difference", ascending=False)
    return out.reset_index(drop=True)


def compute_group_delta_transition_summary(transitions: pd.DataFrame, pos_group: str = "PTSD", neg_group: str = "HC") -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    pos = transitions[transitions["cohort"] == pos_group].copy()
    neg = transitions[transitions["cohort"] == neg_group].copy()
    if pos.empty or neg.empty:
        return pd.DataFrame()
    cols = [
        "source", "target", "source_basin", "target_basin", "source_basin_label", "target_basin_label",
        "count", "source_total", "probability", "subject_support_count", "subject_support_prop",
        "most_common_source_event", "most_common_target_event"
    ]
    pos = pos[[c for c in cols if c in pos.columns]]
    neg = neg[[c for c in cols if c in neg.columns]]
    merged = pos.merge(neg, on=["source", "target"], how="outer", suffixes=(f"_{pos_group}", f"_{neg_group}")).fillna(0)

    for c in ["source_basin", "target_basin", "source_basin_label", "target_basin_label"]:
        cp = f"{c}_{pos_group}"
        cn = f"{c}_{neg_group}"
        if cp in merged.columns or cn in merged.columns:
            merged[c] = merged.get(cp, pd.Series([np.nan] * len(merged))).replace(0, np.nan).combine_first(merged.get(cn, pd.Series([np.nan] * len(merged))).replace(0, np.nan))

    for metric in ["count", "probability", "subject_support_count", "subject_support_prop"]:
        cp = f"{metric}_{pos_group}"
        cn = f"{metric}_{neg_group}"
        if cp in merged.columns and cn in merged.columns:
            merged[f"{pos_group}_minus_{neg_group}_{metric}"] = pd.to_numeric(merged[cp], errors="coerce").fillna(0) - pd.to_numeric(merged[cn], errors="coerce").fillna(0)

    sort_col = f"{pos_group}_minus_{neg_group}_probability"
    if sort_col in merged.columns:
        merged["abs_probability_difference"] = merged[sort_col].abs()
        merged = merged.sort_values("abs_probability_difference", ascending=False)
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------
# Feature/statistical result indexing
# ---------------------------------------------------------------------


def resolve_manifest_paths(manifest: Optional[pd.DataFrame], parsed_dir: Path) -> List[Path]:
    if manifest is None or manifest.empty:
        return []
    base = parsed_dir.parent
    paths = []
    path_cols = [c for c in ["path", "file_path", "relative_path", "filename", "file_name"] if c in manifest.columns]
    for _, row in manifest.iterrows():
        p = None
        for col in path_cols:
            raw = row.get(col)
            if pd.isna(raw):
                continue
            candidate = Path(str(raw))
            if candidate.is_absolute():
                p = candidate
            else:
                p = base / candidate
            if p.exists():
                break
        if p is not None and p.exists():
            paths.append(p)
    return sorted(set(paths))


def summarize_result_file(path: Path) -> Dict[str, object]:
    row: Dict[str, object] = {
        "file_name": path.name,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "readable": False,
        "n_rows": np.nan,
        "n_cols": np.nan,
        "columns": "",
        "p_col": "",
        "q_col": "",
        "sig_col": "",
        "min_p": np.nan,
        "min_q": np.nan,
        "n_p_lt_0_05": np.nan,
        "n_q_lt_0_05": np.nan,
        "n_significant_flag": np.nan,
    }
    try:
        if path.suffix.lower() in [".csv", ".tsv"]:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            return row
    except Exception:
        return row

    row["readable"] = True
    row["n_rows"] = len(df)
    row["n_cols"] = len(df.columns)
    row["columns"] = ";".join(map(str, df.columns))

    p_cols = [c for c in df.columns if re.fullmatch(r"p(_value)?|pval|p_value_raw", str(c).lower()) or "p_value" in str(c).lower()]
    q_cols = [c for c in df.columns if re.fullmatch(r"q(_value)?|qval|fdr|p_adj|padj", str(c).lower()) or "q_value" in str(c).lower()]
    sig_cols = [c for c in df.columns if str(c).lower() in {"sig", "significant", "is_significant", "reject"}]

    if p_cols:
        pcol = p_cols[0]
        vals = pd.to_numeric(df[pcol], errors="coerce")
        row["p_col"] = pcol
        row["min_p"] = vals.min()
        row["n_p_lt_0_05"] = int((vals < 0.05).sum())
    if q_cols:
        qcol = q_cols[0]
        vals = pd.to_numeric(df[qcol], errors="coerce")
        row["q_col"] = qcol
        row["min_q"] = vals.min()
        row["n_q_lt_0_05"] = int((vals < 0.05).sum())
    if sig_cols:
        scol = sig_cols[0]
        row["sig_col"] = scol
        s = df[scol]
        if s.dtype == bool:
            row["n_significant_flag"] = int(s.sum())
        else:
            row["n_significant_flag"] = int(s.astype(str).str.lower().isin(["true", "1", "yes", "sig", "*"]).sum())
    return row


def extract_top_stat_rows(paths: Sequence[Path], top_n_per_file: int = 20) -> pd.DataFrame:
    rows = []
    for path in paths:
        try:
            if path.suffix.lower() in [".csv", ".tsv"]:
                sep = "\t" if path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(path, sep=sep)
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                continue
        except Exception:
            continue
        if df.empty:
            continue
        p_cols = [c for c in df.columns if re.fullmatch(r"p(_value)?|pval|p_value_raw", str(c).lower()) or "p_value" in str(c).lower()]
        q_cols = [c for c in df.columns if re.fullmatch(r"q(_value)?|qval|fdr|p_adj|padj", str(c).lower()) or "q_value" in str(c).lower()]
        sort_col = q_cols[0] if q_cols else (p_cols[0] if p_cols else None)
        if sort_col is None:
            continue
        tmp = df.copy()
        tmp["_sort"] = pd.to_numeric(tmp[sort_col], errors="coerce")
        tmp = tmp.sort_values("_sort").head(top_n_per_file)
        for rank, (_, r) in enumerate(tmp.iterrows(), start=1):
            keep = {str(c): r[c] for c in df.columns[:40]}
            keep.update({"file_name": path.name, "path": str(path), "rank_in_file": rank, "sort_col": sort_col, "sort_value": r["_sort"]})
            rows.append(keep)
    return pd.DataFrame(rows)


def index_optional_files(parsed_dir: Path, feature_manifest: Optional[pd.DataFrame], confirm_manifest: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_paths = resolve_manifest_paths(feature_manifest, parsed_dir)
    confirm_paths = resolve_manifest_paths(confirm_manifest, parsed_dir)

    feature_index = feature_manifest.copy() if feature_manifest is not None else pd.DataFrame()
    confirm_index = confirm_manifest.copy() if confirm_manifest is not None else pd.DataFrame()

    stat_paths = [p for p in confirm_paths if p.suffix.lower() in [".csv", ".tsv", ".xlsx", ".xls"]]
    stat_rows = [summarize_result_file(p) for p in stat_paths]
    statistical_index = pd.DataFrame(stat_rows)
    if not statistical_index.empty:
        sort_cols = [c for c in ["min_q", "min_p", "n_q_lt_0_05", "n_p_lt_0_05"] if c in statistical_index.columns]
        if sort_cols:
            statistical_index = statistical_index.sort_values(sort_cols, na_position="last")

    top_stats = extract_top_stat_rows(stat_paths)
    return feature_index, confirm_index, statistical_index, top_stats


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


def format_top_list(df: pd.DataFrame, value_col: str, n: int = 5) -> str:
    if df is None or df.empty or value_col not in df.columns:
        return "No rows available."
    lines = []
    for _, r in df.head(n).iterrows():
        state = r.get("state", "NA")
        basin = r.get("basin", "NA")
        blabel = r.get("basin_label_readable", basin_label_for(basin))
        val = r.get(value_col, np.nan)
        occ = r.get("occupancy", "NA")
        lines.append(f"- State {state} | B{basin} {blabel} | {value_col}={val:.3f} | occupancy={occ}")
    return "\n".join(lines)


def write_summary_report(
    out_dir: Path,
    event_col: str,
    epoch_mode: str,
    state_summary: pd.DataFrame,
    transition_backbone: pd.DataFrame,
    group_delta_state: pd.DataFrame,
    group_delta_transition: pd.DataFrame,
    statistical_index: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# ELAT-Vis Summary Tables Report")
    lines.append("")
    lines.append(f"Event column: `{event_col}`")
    lines.append(f"Epoch mode: `{epoch_mode}`")
    lines.append("")
    lines.append("## Top CS+ shifted states")
    top_plus = state_summary.sort_values("cs_delta", ascending=False) if "cs_delta" in state_summary.columns else pd.DataFrame()
    lines.append(format_top_list(top_plus, "cs_delta"))
    lines.append("")
    lines.append("## Top CS− shifted states")
    top_minus = state_summary.sort_values("cs_delta", ascending=True) if "cs_delta" in state_summary.columns else pd.DataFrame()
    lines.append(format_top_list(top_minus, "cs_delta"))
    lines.append("")
    lines.append("## Transition backbone")
    if transition_backbone.empty:
        lines.append("No transition backbone rows passed the selected filters.")
    else:
        for _, r in transition_backbone.head(10).iterrows():
            lines.append(
                f"- {r.get('cohort', 'NA')}: {int(r['source'])} → {int(r['target'])} | "
                f"prob={float(r.get('probability', np.nan)):.3f} | count={int(r.get('count', 0))} | "
                f"subject_support={r.get('subject_support_count', 'NA')}"
            )
    lines.append("")
    lines.append("## Group-delta state highlights")
    if group_delta_state.empty:
        lines.append("No HC/PTSD group-delta state table generated.")
    else:
        delta_cols = [c for c in group_delta_state.columns if c.endswith("_cs_delta") and "minus" in c]
        if delta_cols:
            dcol = delta_cols[0]
            tmp = group_delta_state.sort_values("abs_cs_delta_difference", ascending=False) if "abs_cs_delta_difference" in group_delta_state.columns else group_delta_state
            for _, r in tmp.head(10).iterrows():
                lines.append(f"- State {r.get('state')} | B{r.get('basin')} {r.get('basin_label_readable', '')} | {dcol}={r.get(dcol):.3f}")
    lines.append("")
    lines.append("## Group-delta transition highlights")
    if group_delta_transition.empty:
        lines.append("No HC/PTSD group-delta transition table generated.")
    else:
        dcols = [c for c in group_delta_transition.columns if c.endswith("_probability") and "minus" in c]
        if dcols:
            dcol = dcols[0]
            tmp = group_delta_transition.sort_values("abs_probability_difference", ascending=False) if "abs_probability_difference" in group_delta_transition.columns else group_delta_transition
            for _, r in tmp.head(10).iterrows():
                lines.append(f"- {int(r['source'])} → {int(r['target'])} | {dcol}={float(r.get(dcol, 0)):.3f}")
    lines.append("")
    lines.append("## Statistical result index")
    if statistical_index.empty:
        lines.append("No readable confirmatory/statistical result files indexed.")
    else:
        lines.append(f"Indexed {len(statistical_index)} readable statistical result files.")
        if "min_p" in statistical_index.columns:
            best = statistical_index.sort_values("min_p", na_position="last").head(5)
            for _, r in best.iterrows():
                lines.append(f"- {r.get('file_name')} | min_p={r.get('min_p')} | min_q={r.get('min_q')}")
    lines.append("")
    lines.append("> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.")
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------


def generate_summary_tables(
    parsed_dir: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    event_col: str = "trial_type_hrf4",
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    group: Optional[str] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    top_n: int = 12,
    min_transition_count: int = 1,
    min_transition_prob: float = 0.0,
    min_subject_support_count: int = 0,
    min_subject_support_prop: float = 0.0,
    top_k_per_state: Optional[int] = 3,
    hide_self_transitions: bool = True,
    pos_group: str = "PTSD",
    neg_group: str = "HC",
    preset: str = "fear4",
) -> Dict[str, Path]:
    parsed_dir = Path(parsed_dir)
    out_dir = Path(out_dir) if out_dir else parsed_dir / "summary_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_parsed_tables(parsed_dir)
    timeline = tables["timeline"]
    states_raw = tables["states"]
    if timeline is None or timeline.empty:
        raise FileNotFoundError(f"No timeline_table.csv found in {parsed_dir}")

    filtered = filter_timeline(
        timeline,
        event_col=event_col,
        epoch_col=epoch_col,
        epoch_mode=epoch_mode,
        group=group,
        subject=subject,
        task=task,
        session=session,
    )
    if filtered.empty:
        raise ValueError("No timeline rows remain after filtering.")

    states = prepare_states(states_raw, filtered, preset=preset)

    outputs: Dict[str, Path] = {}

    # Cohorts to compute.
    cohorts: List[Tuple[str, pd.DataFrame]] = [("merged", filtered)]
    if "group" in filtered.columns:
        for g in sorted(filtered["group"].dropna().astype(str).unique()):
            if g and g != "UNKNOWN":
                cohorts.append((g, filtered[filtered["group"].astype(str) == g].copy()))

    state_parts = []
    transition_parts = []
    for cohort_name, cohort_df in cohorts:
        if cohort_df.empty:
            continue
        state_parts.append(compute_state_event_summary(cohort_df, states, event_col, cohort=cohort_name, preset=preset))
        transition_parts.append(compute_transitions(cohort_df, cohort=cohort_name, hide_self_transitions=hide_self_transitions))

    state_summary = pd.concat(state_parts, ignore_index=True) if state_parts else pd.DataFrame()
    transition_summary = pd.concat(transition_parts, ignore_index=True) if transition_parts else pd.DataFrame()

    basin_summary = compute_basin_event_summary(state_summary)
    top_plus, top_minus, top_enriched = make_top_tables(state_summary, top_n=top_n)
    local_minimum_summary = compute_local_minimum_summary(state_summary)
    transition_backbone = filter_transition_backbone(
        transition_summary,
        min_count=min_transition_count,
        min_probability=min_transition_prob,
        min_subject_support_count=min_subject_support_count,
        min_subject_support_prop=min_subject_support_prop,
        top_k_per_state=top_k_per_state,
    )
    group_delta_state = compute_group_delta_state_summary(state_summary, pos_group=pos_group, neg_group=neg_group)
    group_delta_transition = compute_group_delta_transition_summary(transition_summary, pos_group=pos_group, neg_group=neg_group)

    core_tables = {
        "state_event_summary.csv": state_summary,
        "top_csplus_states.csv": top_plus,
        "top_csminus_states.csv": top_minus,
        "top_enriched_states.csv": top_enriched,
        "local_minimum_summary.csv": local_minimum_summary,
        "basin_event_summary.csv": basin_summary,
        "transition_summary.csv": transition_summary,
        "transition_backbone_summary.csv": transition_backbone,
        "group_delta_state_summary.csv": group_delta_state,
        "group_delta_transition_summary.csv": group_delta_transition,
    }
    for name, df in core_tables.items():
        path = out_dir / name
        write_table(df, path)
        outputs[name] = path

    # Optional feature/confirmatory indexes.
    feature_index, confirm_index, statistical_index, top_stats = index_optional_files(
        parsed_dir,
        tables.get("feature_manifest"),
        tables.get("confirmatory_manifest"),
    )
    optional_tables = {
        "detected_feature_files.csv": feature_index,
        "detected_confirmatory_files.csv": confirm_index,
        "statistical_results_index.csv": statistical_index,
        "statistical_top_results.csv": top_stats,
    }
    for name, df in optional_tables.items():
        path = out_dir / name
        write_table(df, path)
        outputs[name] = path

    # Run metadata.
    run_meta = {
        "parsed_dir": str(parsed_dir),
        "out_dir": str(out_dir),
        "event_col": event_col,
        "epoch_col": epoch_col or find_epoch_col(filtered, event_col),
        "epoch_mode": epoch_mode,
        "group_filter": group,
        "subject_filter": subject,
        "task_filter": task,
        "session_filter": session,
        "n_timeline_rows": int(len(filtered)),
        "n_states_summary_rows": int(len(state_summary)),
        "n_transition_rows": int(len(transition_summary)),
        "n_transition_backbone_rows": int(len(transition_backbone)),
        "transition_backbone_filters": {
            "min_transition_count": min_transition_count,
            "min_transition_prob": min_transition_prob,
            "min_subject_support_count": min_subject_support_count,
            "min_subject_support_prop": min_subject_support_prop,
            "top_k_per_state": top_k_per_state,
            "hide_self_transitions": hide_self_transitions,
        },
    }
    meta_path = out_dir / "summary_run_metadata.json"
    meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    outputs["summary_run_metadata.json"] = meta_path

    write_summary_report(
        out_dir,
        event_col=event_col,
        epoch_mode=epoch_mode,
        state_summary=state_summary,
        transition_backbone=transition_backbone,
        group_delta_state=group_delta_state,
        group_delta_transition=group_delta_transition,
        statistical_index=statistical_index,
    )
    outputs["summary_report.md"] = out_dir / "summary_report.md"

    return outputs


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dashboard-ready ELAT-Vis summary tables from parser outputs.")
    p.add_argument("--parsed-dir", required=True, help="Directory containing parser.py outputs.")
    p.add_argument("--out-dir", default=None, help="Output directory. Defaults to parsed_dir/summary_tables.")
    p.add_argument("--event-col", default="trial_type_hrf4", help="Event column for CS+/CS-/ITI.")
    p.add_argument("--epoch-col", default=None, help="Matched early/late epoch column. Auto-detected if omitted.")
    p.add_argument("--epoch-mode", default="full", choices=["full", "early", "late"], help="Filter to full/early/late.")
    p.add_argument("--group", default=None, help="Optional group filter.")
    p.add_argument("--subject", default=None, help="Optional subject filter.")
    p.add_argument("--task", default=None, help="Optional task filter.")
    p.add_argument("--session", default=None, help="Optional session filter.")
    p.add_argument("--preset", default="fear4", choices=["fear4", "none"], help="Basin label preset.")
    p.add_argument("--top-n", type=int, default=12, help="Rows per cohort for top-state tables.")
    p.add_argument("--min-transition-count", type=int, default=1, help="Minimum transition count for backbone.")
    p.add_argument("--min-transition-prob", type=float, default=0.0, help="Minimum transition probability for backbone.")
    p.add_argument("--min-subject-support-count", type=int, default=0, help="Minimum number of subjects supporting edge.")
    p.add_argument("--min-subject-support-prop", type=float, default=0.0, help="Minimum proportion of subjects supporting edge.")
    p.add_argument("--top-k-per-state", type=int, default=3, help="Top-k outgoing transitions per source state for backbone. Use 0 to disable.")
    p.add_argument("--show-self-transitions", action="store_true", help="Keep self-transitions.")
    p.add_argument("--pos-group", default="PTSD", help="Positive group for group-delta tables.")
    p.add_argument("--neg-group", default="HC", help="Negative group for group-delta tables.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = None if args.preset == "none" else args.preset
    top_k = None if args.top_k_per_state == 0 else args.top_k_per_state
    outputs = generate_summary_tables(
        parsed_dir=args.parsed_dir,
        out_dir=args.out_dir,
        event_col=args.event_col,
        epoch_col=args.epoch_col,
        epoch_mode=args.epoch_mode,
        group=args.group,
        subject=args.subject,
        task=args.task,
        session=args.session,
        top_n=args.top_n,
        min_transition_count=args.min_transition_count,
        min_transition_prob=args.min_transition_prob,
        min_subject_support_count=args.min_subject_support_count,
        min_subject_support_prop=args.min_subject_support_prop,
        top_k_per_state=top_k,
        hide_self_transitions=not args.show_self_transitions,
        pos_group=args.pos_group,
        neg_group=args.neg_group,
        preset=preset or "none",
    )
    print("Generated summary tables:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
