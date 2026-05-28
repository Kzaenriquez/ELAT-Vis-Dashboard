#!/usr/bin/env python3
"""
figures_radial_explorer_v1_5_elatvis.py

Radial Basin Explorer for ELAT-Vis.

Purpose
-------
Create a per-subject animated radial basin explorer from parsed ELAT outputs.

Design
------
- Basin hulls = large radial circles.
- Local minima = basin centers.
- Other states = rings around the local minimum.
- Ring distance = descent-graph distance from local minimum.
- Fallback distance = Hamming distance.
- Tracker ball moves through the subject's state timeline.
- Ball color = current event label: CS+ red, CS- blue, ITI gray.
- Recent transition = solid line.
- Older tail = dotted/faded line.
- Right panel = animated t-3 to t+0 ROI activity heatmap + event row.
- Bottom panel = basin-only timeline strip colored by event.

Streamlit-ready:
    from figures_radial_explorer_v1_elatvis import make_radial_basin_explorer
    fig = make_radial_basin_explorer(parsed_dir="...", subject="FC001")
    st.plotly_chart(fig, use_container_width=True)

CLI example:
    python figures_radial_explorer_v1_5_elatvis.py \
        --parsed-dir merge_extinction_ela_5ROI/parsed \
        --subject FC008 \
        --event-col trial_type_hrf4 \
        --epoch-mode late \
        --preset fear4
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
from plotly.subplots import make_subplots


EVENT_COLORS = {
    "CS+": "#D95F5F",
    "CS-": "#4C78A8",
    "ITI": "#5F6368",
    "NONE": "#D9D9D9",
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        return hex_color
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


EPOCH_COLORS = {
    "early": "#F2E6B6",
    "late": "#CFCFCF",
    "none": "#F7F7F7",
}

BASIN_LABELS_FEAR4 = {
    1: "Global off",
    2: "Threat-like",
    3: "Safety-like",
    4: "Global on",
}

BASIN_COLORS_FEAR4 = {
    1: "#A8D5A2",  # Global off = light green
    2: "#D95F5F",  # Threat-like = red
    3: "#4C78A8",  # Safety-like = blue
    4: "#2E7D32",  # Global on = dark green
}

BASIN_FILL_FEAR4 = {
    1: "rgba(168,213,162,0.16)",
    2: "rgba(217,95,95,0.13)",
    3: "rgba(76,120,168,0.13)",
    4: "rgba(46,125,50,0.13)",
}

FEAR4_BASIN_CENTERS = {
    # Equal 2x2 spacing: top-left, top-right, bottom-left, bottom-right
    2: (-5.2, 5.2),   # Threat-like
    1: (5.2, 5.2),    # Global off
    3: (-5.2, -5.2),  # Safety-like
    4: (5.2, -5.2),   # Global on
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


def infer_roi_names(metadata: dict, n_rois: Optional[int]) -> List[str]:
    if isinstance(metadata, dict):
        for key in ["roi_names", "ROIs", "roi_order"]:
            if key in metadata and isinstance(metadata[key], list):
                return [str(x) for x in metadata[key]]
    if n_rois == 5:
        return ["Amygdala", "Hippocampus", "Insula", "dACC", "vmPFC"]
    if n_rois:
        return [f"ROI{i + 1}" for i in range(n_rois)]
    return []


# ---------------------------------------------------------------------
# State graph / layout
# ---------------------------------------------------------------------

def hamming_distance(a: str, b: str) -> Optional[int]:
    if a is None or b is None:
        return None
    a, b = str(a), str(b)
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


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


def basin_labels_map(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_LABELS_FEAR4.copy()
        for b in basins:
            out.setdefault(b, f"B{b}")
        return out
    return {b: f"B{b}" for b in basins}


def basin_color_map(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_COLORS_FEAR4.copy()
        for i, b in enumerate(basins):
            out.setdefault(b, DEFAULT_BASIN_COLORS[i % len(DEFAULT_BASIN_COLORS)])
        return out
    return {b: DEFAULT_BASIN_COLORS[i % len(DEFAULT_BASIN_COLORS)] for i, b in enumerate(basins)}


def basin_fill_map(basins: Iterable[int], preset: Optional[str] = None) -> Dict[int, str]:
    basins = sorted([int(b) for b in basins if pd.notna(b)])
    if preset == "fear4":
        out = BASIN_FILL_FEAR4.copy()
        for b in basins:
            out.setdefault(b, "rgba(127,127,127,0.12)")
        return out
    return {b: "rgba(127,127,127,0.12)" for b in basins}


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


def compute_radial_state_layout(
    states: pd.DataFrame,
    descent_edges: pd.DataFrame,
    preset: Optional[str] = "fear4",
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Dict[str, float]]]:
    """
    Branch-aware radial layout.

    Local minimum = center.
    Other states are placed by descent distance from local minimum.
    Descent children inherit their parent angle, so connected states cluster
    together instead of spreading randomly across the basin circle.
    """
    states = normalize_basin_col(normalize_state_col(states)).copy()
    basins = sorted(states["basin"].dropna().astype(int).unique().tolist()) if "basin" in states.columns else [1]
    centers = get_basin_centers(basins, preset=preset)

    pos: Dict[int, Tuple[float, float]] = {}
    basin_shapes: Dict[int, Dict[str, float]] = {}

    inner_radius = 0.72
    ring_spacing = 0.92

    for b in basins:
        sub = states[states["basin"].astype("Int64") == b].copy()
        if sub.empty:
            continue

        cx, cy = centers.get(int(b), (0.0, 0.0))
        basin_state_set = set(sub["state"].dropna().astype(int).tolist())

        local_states = (
            sub[sub.get("is_local_minimum", False).fillna(False).astype(bool)]["state"]
            .dropna()
            .astype(int)
            .tolist()
        )
        if not local_states:
            local_states = [min(basin_state_set)]
        local_states = sorted(local_states)

        d_map = compute_ring_distances(sub, descent_edges, local_states)

        # Place local minima at center or small mini-ring if multiple.
        if len(local_states) == 1:
            pos[local_states[0]] = (cx, cy)
        else:
            for i, s in enumerate(local_states):
                ang = 2 * math.pi * i / len(local_states)
                pos[s] = (cx + 0.28 * math.cos(ang), cy + 0.28 * math.sin(ang))

        # Directed descent parent map: source -> target, where target should be closer to LM.
        directed_parent: Dict[int, int] = {}
        undirected_neighbors: Dict[int, set] = {s: set() for s in basin_state_set}

        if descent_edges is not None and not descent_edges.empty:
            for _, erow in descent_edges.iterrows():
                s = int(erow["source"])
                t = int(erow["target"])
                if s not in basin_state_set or t not in basin_state_set:
                    continue
                undirected_neighbors.setdefault(s, set()).add(t)
                undirected_neighbors.setdefault(t, set()).add(s)

                ds = d_map.get(s, 999)
                dt = d_map.get(t, 999)
                if dt < ds:
                    directed_parent[s] = t
                elif ds < dt:
                    directed_parent[t] = s

        # Fallback parent: any neighbor one ring closer.
        parent: Dict[int, int] = {}
        for s in sorted(basin_state_set):
            if s in local_states:
                continue

            if s in directed_parent:
                parent[s] = directed_parent[s]
                continue

            ds = d_map.get(s, 1)
            closer = [
                nb for nb in undirected_neighbors.get(s, set())
                if d_map.get(nb, 999) < ds
            ]
            if closer:
                parent[s] = sorted(closer, key=lambda x: (d_map.get(x, 999), x))[0]
            else:
                # fallback to nearest local minimum
                parent[s] = local_states[0]

        children: Dict[int, List[int]] = {}
        for child, par in parent.items():
            children.setdefault(par, []).append(child)
        for par in children:
            children[par] = sorted(children[par])

        angles: Dict[int, float] = {}

        # Root branches are children of local minima.
        root_children = []
        for lm in local_states:
            root_children.extend(children.get(lm, []))
        root_children = sorted(set(root_children))

        if root_children:
            for i, s in enumerate(root_children):
                # Half-step avoids putting a branch exactly on cardinal axes.
                angles[s] = 2 * math.pi * (i + 0.5) / len(root_children) + int(b) * 0.10

        # Recursively place descendants close to parent angle.
        queue = list(root_children)
        while queue:
            par = queue.pop(0)
            kids = children.get(par, [])
            if not kids:
                continue

            base = angles.get(par, 2 * math.pi * 0.5)
            # Smaller spread for deeper rings to keep branches coherent.
            depth = max(d_map.get(par, 1), 1)
            spread = max(0.18, 0.62 / depth)

            if len(kids) == 1:
                offsets = [0.0]
            else:
                offsets = np.linspace(-spread / 2, spread / 2, len(kids)).tolist()

            for kid, off in zip(kids, offsets):
                angles[kid] = base + off
                queue.append(kid)

        # Any unassigned node gets placed by ring, but after assigned branches.
        unassigned = [s for s in sorted(basin_state_set) if s not in local_states and s not in angles]
        if unassigned:
            for i, s in enumerate(unassigned):
                angles[s] = 2 * math.pi * (i + 0.5) / len(unassigned) + int(b) * 0.25

        # Convert ring distance + angle to x/y.
        for s in sorted(basin_state_set):
            if s in local_states:
                continue
            dist = max(int(d_map.get(s, 1)), 1)
            r = inner_radius + dist * ring_spacing
            ang = angles.get(s, 0.0)

            # Tiny deterministic jitter to avoid exact overlaps without breaking branch grouping.
            jitter = 0.08 * ((s % 3) - 1)
            pos[s] = (
                cx + (r + jitter) * math.cos(ang),
                cy + (r + jitter) * math.sin(ang),
            )

        max_ring = max(d_map.values()) if d_map else 1
        basin_shapes[int(b)] = {
            "cx": cx,
            "cy": cy,
            "radius": inner_radius + max(max_ring, 1) * ring_spacing + 0.95,
        }

    return pos, basin_shapes


# ---------------------------------------------------------------------
# Timeline prep
# ---------------------------------------------------------------------

def filter_subject_timeline(
    timeline: pd.DataFrame,
    subject: Optional[str],
    group: Optional[str],
    task: Optional[str],
    session: Optional[str],
    event_col: str,
    epoch_col: Optional[str],
    epoch_mode: str,
) -> Tuple[pd.DataFrame, str]:
    df = normalize_basin_col(normalize_state_col(timeline)).copy()
    for c in ["subject", "group", "task", "session"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if group and "group" in df.columns:
        df = df[df["group"].astype(str) == str(group)]
    if task and "task" in df.columns:
        df = df[df["task"].astype(str) == str(task)]
    if session and "session" in df.columns:
        df = df[df["session"].astype(str) == str(session)]

    if subject and "subject" in df.columns:
        df = df[df["subject"].astype(str) == str(subject)]
    elif "subject" in df.columns:
        subjects = sorted(df["subject"].dropna().astype(str).unique().tolist())
        if len(subjects) > 1:
            print(f"[INFO] No subject specified. Using first subject: {subjects[0]}")
        if subjects:
            subject = subjects[0]
            df = df[df["subject"].astype(str) == subject]

    epoch_col = epoch_col or find_epoch_col(df, event_col)
    if epoch_mode.lower() != "full" and epoch_col in df.columns:
        df = df[df[epoch_col].apply(normalize_epoch_value) == epoch_mode.lower()]

    x_col = choose_x_col(df)
    df = df.sort_values(x_col).reset_index(drop=True)
    df["frame_idx"] = np.arange(len(df))
    df["event_norm"] = df[event_col].apply(normalize_event_value) if event_col in df.columns else "NONE"
    df["epoch_norm"] = df[epoch_col].apply(normalize_epoch_value) if epoch_col in df.columns else "none"

    if df.empty:
        raise ValueError("No rows remain after subject/group/task/session/epoch filtering.")

    return df, x_col


def state_binary_vector(state: int, states_lookup: Dict[int, str], n_rois: int) -> List[int]:
    binary = states_lookup.get(int(state))
    if binary is None or str(binary).lower() == "nan":
        binary = state_to_binary(int(state), n_rois)
    binary = str(binary).zfill(n_rois)
    return [int(ch) for ch in binary]


def make_discrete_colorscale(code_to_color: Dict[int, str], zmin: int, zmax: int) -> List[List[Union[float, str]]]:
    """
    Build a Plotly-compatible stepped colorscale for integer category codes.

    Correct logic:
    Plotly normalizes z values as (z - zmin) / (zmax - zmin).
    Therefore, each integer code must occupy the interval halfway to its
    neighboring integer codes in that normalized space.
    """
    zmin = int(zmin)
    zmax = int(zmax)
    used_codes = [int(c) for c in sorted(code_to_color) if zmin <= int(c) <= zmax]

    if not used_codes:
        return [[0.0, "#F7F7F7"], [1.0, "#F7F7F7"]]

    if zmax == zmin:
        color = code_to_color.get(zmin, code_to_color.get(used_codes[0], "#F7F7F7"))
        return [[0.0, color], [1.0, color]]

    denom = zmax - zmin
    scale = []

    for code in used_codes:
        center = (code - zmin) / denom
        lo = (code - 0.5 - zmin) / denom
        hi = (code + 0.5 - zmin) / denom
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        color = code_to_color[code]
        scale.append([lo, color])
        scale.append([center, color])
        scale.append([hi, color])

    scale = sorted(scale, key=lambda x: x[0])

    if scale[0][0] > 0:
        scale.insert(0, [0.0, scale[0][1]])
    if scale[-1][0] < 1:
        scale.append([1.0, scale[-1][1]])

    cleaned = []
    last = -1.0
    for v, color in scale:
        v = max(0.0, min(1.0, float(v)))
        if v < last:
            v = last
        cleaned.append([v, color])
        last = v

    return cleaned


def build_activity_matrix(
    timeline: pd.DataFrame,
    idx: int,
    states_lookup: Dict[int, str],
    roi_names: List[str],
    window: int = 4,
) -> Tuple[np.ndarray, List[str], List[str]]:
    n_rois = len(roi_names)
    start = max(0, idx - window + 1)
    rows = timeline.iloc[start:idx + 1].copy()

    while len(rows) < window:
        pad = pd.DataFrame([{}])
        rows = pd.concat([pad, rows], ignore_index=True)

    labels = [f"t-{window - 1 - i}" for i in range(window - 1)] + ["t+0"]
    mat = np.zeros((n_rois + 1, window), dtype=int)

    for j in range(window):
        if j >= len(rows):
            continue
        row = rows.iloc[j]
        if "state" in row and pd.notna(row.get("state")):
            vec = state_binary_vector(int(row["state"]), states_lookup, n_rois)
            mat[:n_rois, j] = vec

        ev = normalize_event_value(row.get("event_norm", "NONE"))
        mat[n_rois, j] = {"CS+": 2, "CS-": 3, "ITI": 4}.get(ev, 5)

    y_labels = roi_names + ["Event"]
    return mat, y_labels, labels


def build_timeline_strip(
    timeline: pd.DataFrame,
    basin_colors: Dict[int, str],
    basin_labels: Optional[Dict[int, str]] = None,
    basins: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[str], Dict[int, str], np.ndarray]:
    """
    Timeline strip with:
    Row 0 = Epoch
    Rows 1..N = basin table timeline.

    Active basin cell inherits event color:
        CS+ = red
        CS- = blue
        ITI = dark gray
    Inactive basin cells are near-white.
    """
    n = len(timeline)
    if basins is None:
        basins = sorted(timeline["basin"].dropna().astype(int).unique().tolist())
    basin_labels = basin_labels or {b: f"B{b}" for b in basins}

    z = np.zeros((len(basins) + 1, n), dtype=int)
    labels = ["Epoch"] + [f"B{b}: {basin_labels.get(b, f'B{b}')}" for b in basins]

    # Compact category codes.
    code_to_color = {
        0: "#F7F7F7",           # inactive
        1: EPOCH_COLORS["early"],
        2: EPOCH_COLORS["late"],
        3: EVENT_COLORS["CS+"],
        4: EVENT_COLORS["CS-"],
        5: EVENT_COLORS["ITI"],
        6: EVENT_COLORS["NONE"],
    }

    custom = np.empty((len(basins) + 1, n), dtype=object)
    basin_to_row = {b: i + 1 for i, b in enumerate(basins)}

    for col_idx, row in timeline.iterrows():
        tr_val = row.get("orig_tr", row.get("kept_row", row.get("frame_idx", col_idx)))
        ep = normalize_epoch_value(row.get("epoch_norm", "none"))
        ev = normalize_event_value(row.get("event_norm", "NONE"))
        basin = int(row["basin"]) if pd.notna(row.get("basin")) else None

        z[0, col_idx] = {"early": 1, "late": 2}.get(ep, 0)
        custom[0, col_idx] = f"row: Epoch<br>TR: {tr_val}<br>epoch: {ep}<br>event: {ev}"

        for b in basins:
            r_idx = basin_to_row[b]
            if basin == b:
                ev_code = {"CS+": 3, "CS-": 4, "ITI": 5}.get(ev, 6)
                z[r_idx, col_idx] = ev_code
                custom[r_idx, col_idx] = (
                    f"row: B{b}: {basin_labels.get(b, f'B{b}')}<br>"
                    f"TR: {tr_val}<br>"
                    f"event: {ev}<br>"
                    f"epoch: {ep}<br>"
                    f"state: {row.get('state', 'NA')}"
                )
            else:
                z[r_idx, col_idx] = 0
                custom[r_idx, col_idx] = (
                    f"row: B{b}: {basin_labels.get(b, f'B{b}')}<br>"
                    f"TR: {tr_val}<br>"
                    f"inactive<br>"
                    f"event: {ev}<br>"
                    f"epoch: {ep}"
                )

    return z, labels, code_to_color, custom


# ---------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------

def circle_polygon(cx: float, cy: float, r: float, n: int = 80) -> Tuple[List[float], List[float]]:
    theta = np.linspace(0, 2 * np.pi, n)
    return (cx + r * np.cos(theta)).tolist(), (cy + r * np.sin(theta)).tolist()



def find_timeline_value(row: pd.Series, timeline_columns: Iterable[str], base: str, event_col: Optional[str] = None):
    """
    Find a row value using exact or suffix variants.
    Examples:
        shock, shock_onset, shock_hrf4
        response, response_onset, response_hrf4
    """
    cols = list(timeline_columns)
    suffix = None
    if event_col and "trial_type_" in event_col:
        suffix = event_col.replace("trial_type_", "")

    candidates = []
    if suffix:
        candidates.extend([f"{base}_{suffix}", f"{base}_trial_type_{suffix}"])
    candidates.append(base)
    candidates.extend([c for c in cols if c.lower().startswith(base.lower() + "_")])

    for c in candidates:
        if c in row.index:
            val = row.get(c)
            if not pd.isna(val):
                return c, val
    return None, None


def build_metadata_table(
    timeline: pd.DataFrame,
    idx: int,
    x_col: str,
    states_lookup: Dict[int, str],
    basin_labels: Dict[int, str],
    event_col: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Build current TR metadata table values."""
    row = timeline.iloc[idx]
    prev = timeline.iloc[idx - 1] if idx > 0 else None

    state = int(row["state"]) if pd.notna(row.get("state")) else None
    basin = int(row["basin"]) if pd.notna(row.get("basin")) else None
    event = normalize_event_value(row.get("event_norm", "NONE"))
    epoch = normalize_epoch_value(row.get("epoch_norm", "none"))
    binary = states_lookup.get(state, "NA") if state is not None else "NA"

    if prev is not None and pd.notna(prev.get("state")):
        prev_state = int(prev["state"])
        transition = f"{prev_state} → {state}"
    else:
        transition = "start"

    if prev is not None and pd.notna(prev.get("basin")):
        prev_basin = int(prev["basin"])
        basin_transition = f"B{prev_basin} → B{basin}"
    else:
        basin_transition = "start"

    attrs = [
        "Subject",
        "Group",
        "Task",
        "Session",
        x_col,
        "Frame",
        "Current state",
        "Current basin",
        "Event",
        "Epoch",
        "State transition",
        "Basin transition",
        "Binary pattern",
    ]

    vals = [
        str(row.get("subject", "NA")),
        str(row.get("group", "NA")),
        str(row.get("task", "NA")),
        str(row.get("session", "NA")),
        str(row.get(x_col, "NA")),
        f"{idx + 1}/{len(timeline)}",
        str(state),
        f"B{basin}: {basin_labels.get(basin, f'B{basin}')}" if basin is not None else "NA",
        event,
        epoch,
        transition,
        basin_transition,
        str(binary),
    ]

    # Flexible task metadata, matching JSON event fields and HRF/onset variants.
    for base in ["stimulus", "response", "response_time", "recognition_memory", "shock", "category"]:
        col, val = find_timeline_value(row, timeline.columns, base, event_col=event_col)
        if col is not None:
            attrs.append(col)
            vals.append(str(val))

    return attrs, vals

def make_radial_basin_explorer(
    parsed_dir: Union[str, Path],
    subject: Optional[str] = None,
    group: Optional[str] = None,
    task: Optional[str] = None,
    session: Optional[str] = None,
    event_col: str = "trial_type_hrf4",
    epoch_col: Optional[str] = None,
    epoch_mode: str = "full",
    preset: Optional[str] = "fear4",
    tail_length: int = 3,
    frame_step: int = 1,
    frame_duration: int = 600,
    activity_window: int = 4,
    show_state_labels: bool = True,
    basin_timeline_color: str = "basin",
    width: int = 1400,
    height: int = 1040,
    title: Optional[str] = None,
) -> go.Figure:
    tables = load_parsed_tables(parsed_dir)
    timeline = tables["timeline"]
    if timeline is None or timeline.empty:
        raise FileNotFoundError(f"No timeline_table.csv found in {parsed_dir}")

    timeline, x_col = filter_subject_timeline(
        timeline=timeline,
        subject=subject,
        group=group,
        task=task,
        session=session,
        event_col=event_col,
        epoch_col=epoch_col,
        epoch_mode=epoch_mode,
    )

    metadata = tables["metadata"]
    states = prepare_states(tables["states"], timeline, tables["basin_graph"], metadata)
    states = normalize_basin_col(normalize_state_col(states))
    n_rois = infer_n_rois(states, metadata)
    states = add_binary_if_missing(states, n_rois)
    roi_names = infer_roi_names(metadata, n_rois)

    descent_edges = build_descent_edges(tables["basin_graph"])
    pos, basin_shapes = compute_radial_state_layout(states, descent_edges, preset=preset)

    basins = sorted(states["basin"].dropna().astype(int).unique().tolist()) if "basin" in states.columns else []
    basin_labels = basin_labels_map(basins, preset=preset)
    basin_colors = basin_color_map(basins, preset=preset)
    basin_fills = basin_fill_map(basins, preset=preset)

    states_lookup = dict(zip(states["state"].dropna().astype(int), states["binary"].astype(str)))

    # Attach positions to timeline.
    timeline["x"] = timeline["state"].astype(int).map(lambda s: pos.get(int(s), (np.nan, np.nan))[0])
    timeline["y"] = timeline["state"].astype(int).map(lambda s: pos.get(int(s), (np.nan, np.nan))[1])
    timeline = timeline.dropna(subset=["x", "y"]).reset_index(drop=True)
    timeline["frame_idx"] = np.arange(len(timeline))

    if len(timeline) == 0:
        raise ValueError("No timeline states could be positioned.")

    if title is None:
        subj_text = subject or str(timeline["subject"].iloc[0]) if "subject" in timeline.columns else "subject"
        title = f"Radial Basin Explorer: {subj_text} | event={event_col} | epoch={epoch_mode}"

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "table"}],
            [None, {"type": "heatmap"}],
            [{"type": "heatmap", "colspan": 2}, None],
        ],
        column_widths=[0.68, 0.32],
        row_heights=[0.43, 0.30, 0.27],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
        subplot_titles=(
            "Radial basin trajectory",
            "Current TR metadata",
            "State activation history + event strip",
            "Basin timeline colored by event",
        ),
    )

    # Basin hulls + labels
    for b, info in basin_shapes.items():
        cx, cy, r = info["cx"], info["cy"], info["radius"]
        xs, ys = circle_polygon(cx, cy, r)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                line=dict(color=basin_colors.get(b, "#AAAAAA"), width=2.5),
                fillcolor=basin_fills.get(b, "rgba(127,127,127,0.12)"),
                hoverinfo="skip",
                name=f"B{b}: {basin_labels.get(b, f'B{b}')}",
                legendgroup=f"basin_{b}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        # Label outside the hull to avoid overlap with states.
        fig.add_annotation(
            x=cx,
            y=cy + r + 0.45,
            text=f"<b>B{b}: {basin_labels.get(b, f'B{b}')}</b>",
            showarrow=False,
            font=dict(size=13, color=basin_colors.get(b, "#333333")),
            row=1,
            col=1,
        )

    # Intra-basin ELAT descent edges
    state_to_basin = {}
    for _, srow in states.dropna(subset=["state"]).iterrows():
        if pd.notna(srow.get("basin", np.nan)):
            state_to_basin[int(srow["state"])] = int(srow["basin"])

    first_descent_edge = True
    for _, erow in descent_edges.iterrows():
        s = int(erow["source"])
        t = int(erow["target"])
        if s not in pos or t not in pos:
            continue
        if state_to_basin.get(s) != state_to_basin.get(t):
            continue
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="rgba(166,140,216,0.50)", width=1.15),
                hovertemplate=f"ELAT descent edge<br>{s} → {t}<extra></extra>",
                name="Intra-basin descent edge",
                legendgroup="descent_edges",
                showlegend=first_descent_edge,
            ),
            row=1,
            col=1,
        )
        first_descent_edge = False

    # Static state nodes
    state_x, state_y, state_text, state_symbols = [], [], [], []
    state_sizes, state_colors, state_line_colors, state_line_widths, state_hover = [], [], [], [], []
    for _, row in states.dropna(subset=["state"]).iterrows():
        s = int(row["state"])
        if s not in pos:
            continue
        x, y = pos[s]
        b = int(row["basin"]) if pd.notna(row.get("basin", np.nan)) else 0
        is_lm = bool(row.get("is_local_minimum", False))

        state_x.append(x)
        state_y.append(y)
        state_text.append(f"LM {s}" if is_lm else str(s))
        state_symbols.append("diamond" if is_lm else "circle")
        state_sizes.append(34 if is_lm else 20)
        state_colors.append("#FFFFFF")
        state_line_colors.append(basin_colors.get(b, "#555555"))
        state_line_widths.append(3.4 if is_lm else 2.0)
        state_hover.append(
            f"<b>State {s}</b><br>"
            f"Basin: B{b} ({basin_labels.get(b, f'B{b}')})<br>"
            f"Local minimum: {'yes' if is_lm else 'no'}<br>"
            f"Binary: {row.get('binary', 'NA')}"
        )

    fig.add_trace(
        go.Scatter(
            x=state_x,
            y=state_y,
            mode="markers+text" if show_state_labels else "markers",
            text=state_text if show_state_labels else None,
            textposition="middle center",
            textfont=dict(size=10, color="#1F2D3D"),
            customdata=state_hover,
            marker=dict(
                symbol=state_symbols,
                size=state_sizes,
                color=state_colors,
                line=dict(color=state_line_colors, width=state_line_widths),
                opacity=0.95,
            ),
            name="States / local minima",
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Initial dynamic traces.
    first = timeline.iloc[0]
    first_event = normalize_event_value(first.get("event_norm", "NONE"))
    first_color = EVENT_COLORS.get(first_event, EVENT_COLORS["NONE"])

    older_tail_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color="rgba(80,80,80,0.28)", width=2, dash="dot"),
            name="Older tail",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    recent_tail_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=first_color, width=4),
            name="Most recent transition",
            hoverinfo="skip",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    tracker_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[first["x"]],
            y=[first["y"]],
            mode="markers",
            marker=dict(size=28, color=first_color, line=dict(color="#111111", width=2.6)),
            name="Current state",
            customdata=[f"State {int(first['state'])}<br>Basin B{int(first['basin'])}<br>Event: {first_event}<br>{x_col}: {first[x_col]}"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Current TR metadata table
    meta_attrs, meta_vals = build_metadata_table(timeline, 0, x_col, states_lookup, basin_labels, event_col=event_col)
    metadata_idx = len(fig.data)
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Field", "Value"],
                fill_color="#F2F4F8",
                align="left",
                font=dict(size=10, color="#243B53"),
            ),
            cells=dict(
                values=[meta_attrs, meta_vals],
                fill_color="#FFFFFF",
                align="left",
                font=dict(size=9, color="#243B53"),
                height=16,
            ),
        ),
        row=1,
        col=2,
    )

    # Activity heatmap: old design style, active = gold, inactive = light gray/white.
    act_z, act_y, act_x = build_activity_matrix(timeline, 0, states_lookup, roi_names, window=activity_window)
    act_colors = {
        0: "#F7F7F7",       # inactive
        1: "#F2C94C",       # active gold/yellow
        2: EVENT_COLORS["CS+"],
        3: EVENT_COLORS["CS-"],
        4: EVENT_COLORS["ITI"],
        5: EVENT_COLORS["NONE"],
    }
    act_colorscale = make_discrete_colorscale(act_colors, 0, 5)

    activity_idx = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            z=act_z,
            x=act_x,
            y=act_y,
            zmin=0,
            zmax=5,
            colorscale=act_colorscale,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="row=%{y}<br>lag=%{x}<br>code=%{z}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Basin-only timeline strip
    strip_z, strip_y, strip_colors, strip_custom = build_timeline_strip(
        timeline,
        basin_colors,
        basin_labels=basin_labels,
        basins=basins,
    )
    zmin, zmax = int(np.nanmin(strip_z)), int(np.nanmax(strip_z))
    strip_colorscale = make_discrete_colorscale(strip_colors, zmin, zmax)

    timeline_idx = len(fig.data)
    fig.add_trace(
        go.Heatmap(
            z=strip_z,
            x=timeline[x_col].tolist(),
            y=strip_y,
            zmin=zmin,
            zmax=zmax,
            colorscale=strip_colorscale,
            showscale=False,
            xgap=0,
            ygap=1,
            customdata=strip_custom,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    cursor_idx = len(fig.data)
    cursor_y0 = -0.5
    cursor_y1 = len(strip_y) - 0.5
    fig.add_trace(
        go.Scatter(
            x=[first[x_col], first[x_col]],
            y=[cursor_y0, cursor_y1],
            mode="lines",
            line=dict(color="#111111", width=2),
            name="Current TR",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Dummy event legend
    for ev in ["CS+", "CS-", "ITI"]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=EVENT_COLORS[ev]),
                name=ev,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Frames
    frames = []
    frame_indices = list(range(0, len(timeline), max(int(frame_step), 1)))
    if frame_indices[-1] != len(timeline) - 1:
        frame_indices.append(len(timeline) - 1)

    cursor_y0 = -0.5
    cursor_y1 = len(strip_y) - 0.5

    for idx in frame_indices:
        row = timeline.iloc[idx]
        ev = normalize_event_value(row.get("event_norm", "NONE"))
        color = EVENT_COLORS.get(ev, EVENT_COLORS["NONE"])

        tail_start = max(0, idx - tail_length)
        tail_df = timeline.iloc[tail_start:idx + 1].copy()

        # Older dotted tail excludes newest segment.
        old_x, old_y = [], []
        if len(tail_df) >= 3:
            coords = tail_df.iloc[:-1][["x", "y"]].values.tolist()
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                old_x += [x0, x1, None]
                old_y += [y0, y1, None]

        # Most recent segment.
        if idx > 0:
            prev = timeline.iloc[idx - 1]
            recent_x = [prev["x"], row["x"]]
            recent_y = [prev["y"], row["y"]]
            transition_label = f"{int(prev['state'])} → {int(row['state'])}"
        else:
            recent_x, recent_y = [], []
            transition_label = "start"

        act_z_i, _, _ = build_activity_matrix(timeline, idx, states_lookup, roi_names, window=activity_window)
        meta_attrs_i, meta_vals_i = build_metadata_table(timeline, idx, x_col, states_lookup, basin_labels, event_col=event_col)

        hover = (
            f"State {int(row['state'])}<br>"
            f"Basin B{int(row['basin'])}<br>"
            f"Event: {ev}<br>"
            f"Epoch: {row.get('epoch_norm', 'none')}<br>"
            f"Transition: {transition_label}<br>"
            f"{x_col}: {row[x_col]}<br>"
            f"Frame: {idx + 1}/{len(timeline)}"
        )

        frame = go.Frame(
            name=str(idx),
            traces=[older_tail_idx, recent_tail_idx, tracker_idx, metadata_idx, activity_idx, cursor_idx],
            data=[
                go.Scatter(x=old_x, y=old_y, line=dict(color=hex_to_rgba(color, 0.35), width=2, dash="dot")),
                go.Scatter(x=recent_x, y=recent_y, line=dict(color=color, width=4)),
                go.Scatter(
                    x=[row["x"]],
                    y=[row["y"]],
                    marker=dict(size=28, color=color, line=dict(color="#111111", width=2.6)),
                    customdata=[hover],
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                go.Table(
                    cells=dict(
                        values=[meta_attrs_i, meta_vals_i],
                        fill_color="#FFFFFF",
                        align="left",
                        font=dict(size=11, color="#243B53"),
                        height=22,
                    )
                ),
                go.Heatmap(z=act_z_i),
                go.Scatter(x=[row[x_col], row[x_col]], y=[cursor_y0, cursor_y1]),
            ],
        )
        frames.append(frame)

    fig.frames = frames

    # Animation controls
    play_button = dict(
        label="Play",
        method="animate",
        args=[
            None,
            {
                "frame": {"duration": frame_duration, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
                "mode": "immediate",
            },
        ],
    )
    pause_button = dict(
        label="Pause",
        method="animate",
        args=[
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
    )

    sliders = [
        dict(
            active=0,
            x=0.16,
            y=-0.095,
            len=0.76,
            currentvalue={"prefix": "Frame index: ", "xanchor": "right"},
            pad={"t": 55, "b": 10},
            steps=[
                dict(
                    method="animate",
                    label=str(i),
                    args=[
                        [str(i)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                )
                for i in frame_indices
            ],
        )
    ]

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin=dict(l=60, r=235, t=85, b=155),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.05,
                y=-0.165,
                xanchor="left",
                yanchor="top",
                buttons=[play_button, pause_button],
            )
        ],
        sliders=sliders,
        legend=dict(
            x=1.015,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#DDDDDD",
            borderwidth=1,
            font=dict(size=9),
        ),
    )

    # Axes formatting
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)

    fig.update_xaxes(title_text="", side="bottom", row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)

    fig.update_xaxes(title_text=x_col, row=3, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)

    return fig


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def list_columns(parsed_dir: Union[str, Path]) -> None:
    tables = load_parsed_tables(parsed_dir)
    timeline = tables["timeline"]
    if timeline is None:
        print("No timeline_table.csv found.")
        return

    print("Timeline columns:")
    for c in timeline.columns:
        print(f"  {c}")

    print("\nSubjects:")
    if "subject" in timeline.columns:
        for s in sorted(timeline["subject"].dropna().astype(str).unique().tolist())[:30]:
            print(f"  {s}")

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
    p = argparse.ArgumentParser(description="Create an animated radial basin explorer from parsed ELAT outputs.")
    p.add_argument("--parsed-dir", required=True, help="Directory containing parser.py outputs.")
    p.add_argument("--subject", default=None, help="Subject filter, e.g. FC008. Animation is per-subject.")
    p.add_argument("--group", default=None, help="Optional group pre-filter, e.g. HC or PTSD.")
    p.add_argument("--task", default=None, help="Optional task filter.")
    p.add_argument("--session", default=None, help="Optional session filter.")
    p.add_argument("--event-col", default="trial_type_hrf4", help="Event column for CS+/CS-/ITI.")
    p.add_argument("--epoch-col", default=None, help="Matched early/late epoch column. Auto-detected if omitted.")
    p.add_argument("--epoch-mode", default="full", choices=["full", "early", "late"], help="Filter to full/early/late.")
    p.add_argument("--preset", default="fear4", choices=["fear4", "none"], help="Basin labels/colors.")
    p.add_argument("--tail-length", type=int, default=3, help="Number of recent steps shown in tracker tail.")
    p.add_argument("--frame-step", type=int, default=1, help="Use every nth timeline row as animation frame.")
    p.add_argument("--frame-duration", type=int, default=600, help="Animation frame duration in ms.")
    p.add_argument("--activity-window", type=int, default=4, help="Number of recent frames in activity heatmap.")
    p.add_argument("--hide-state-labels", action="store_true", help="Hide state IDs on radial map.")
    p.add_argument("--basin-timeline-color", default="event", choices=["event"], help="Basin timeline is basin-only and active cells inherit event color.")
    p.add_argument("--width", type=int, default=1400)
    p.add_argument("--height", type=int, default=1040)
    p.add_argument("--output", default=None, help="Output HTML path. Defaults to parsed_dir/radial_basin_explorer_v1_5.html")
    p.add_argument("--list-columns", action="store_true", help="List columns/subjects and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    parsed_dir = Path(args.parsed_dir)
    preset = None if args.preset == "none" else args.preset

    if args.list_columns:
        list_columns(parsed_dir)
        return

    fig = make_radial_basin_explorer(
        parsed_dir=parsed_dir,
        subject=args.subject,
        group=args.group,
        task=args.task,
        session=args.session,
        event_col=args.event_col,
        epoch_col=args.epoch_col,
        epoch_mode=args.epoch_mode,
        preset=preset,
        tail_length=args.tail_length,
        frame_step=args.frame_step,
        frame_duration=args.frame_duration,
        activity_window=args.activity_window,
        show_state_labels=not args.hide_state_labels,
        basin_timeline_color=args.basin_timeline_color,
        width=args.width,
        height=args.height,
    )

    output = Path(args.output) if args.output else parsed_dir / "radial_basin_explorer_v1_5.html"
    fig.write_html(str(output), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
