# ELAT-Vis Parser

This package contains `parser.py`, a standalone Python parser for Ezaki/ELAT output folders. It converts raw ELAT output files into dashboard-ready CSV/JSON tables.

## Minimum expected ELAT folder

```text
ELAT_output/
  BasinList*.txt
  Dynamics*.csv
  result*.txt
  sub-*_SN.csv
  sub-*_BN.csv
  Figure_100.* to Figure_103.*   # optional
  alignment_hrfcols/timelines_merged.csv   # optional
  ELA_features*/...              # optional
  confirmatory*/...              # optional
```

## Install requirements

```bash
pip install pandas numpy
```

## Basic use

```bash
python parser.py /path/to/ELAT_output
```

This writes outputs to:

```text
/path/to/ELAT_output/parsed/
```

## Use with ROI names

```bash
python parser.py /path/to/ELAT_output \
  --roi_names "Amygdala,Hippocampus,Insula,dACC,vmPFC"
```

or:

```bash
python parser.py /path/to/ELAT_output --roi_names_file roiname.dat
```

## Copy ELAT figures into parsed folder

```bash
python parser.py /path/to/ELAT_output --copy_figures
```

## Main output tables

```text
parsed/
  file_manifest.json
  model_metadata.json
  parse_warnings.json
  available_filters.json
  basins_table.csv
  state_basin_membership.csv
  local_minima_table.csv
  h_parameters.csv
  J_parameters_long.csv
  J_parameters_matrix.csv
  basin_graph.csv
  states_table.csv
  series_table.csv
  dynamics_wide_clean.csv
  dynamics_frequency_long.csv
  dynamics_direct_transition_long.csv
  dynamics_total_transition_long.csv
  dynamics_indirect_transition_long.csv
  timeline_table.csv
```

## Notes

- The script uses ELAT-style state numbering by default: `state = decimal binary + 1`, with ROI 1 as the least significant bit.
- Use `--bit_order big` only if you intentionally want display-order reversal.
- The parser does not plot, rerun ELAT, rerun timeline creation, rerun feature extraction, or rerun confirmatory statistics.
- `timeline_table.csv` is only populated if a timeline file such as `timelines_merged.csv` is detected.
