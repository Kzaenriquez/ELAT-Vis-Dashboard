# ELAT-Vis Summary Tables Report

Event column: `trial_type_hrf4`
Epoch mode: `full`

## Top CS+ shifted states
- State 11 | B1 Global off | cs_delta=0.167 | occupancy=24
- State 13 | B2 Threat-like | cs_delta=0.166 | occupancy=163
- State 31 | B4 Global on | cs_delta=0.143 | occupancy=119
- State 13 | B2 Threat-like | cs_delta=0.142 | occupancy=325
- State 11 | B1 Global off | cs_delta=0.122 | occupancy=49

## Top CS− shifted states
- State 7 | B1 Global off | cs_delta=-0.206 | occupancy=34
- State 8 | B3 Safety-like | cs_delta=-0.184 | occupancy=49
- State 8 | B3 Safety-like | cs_delta=-0.172 | occupancy=99
- State 8 | B3 Safety-like | cs_delta=-0.160 | occupancy=50
- State 20 | B3 Safety-like | cs_delta=-0.149 | occupancy=175

## Transition backbone
- HC: 3 → 1 | prob=0.192 | count=10 | subject_support=8
- HC: 24 → 32 | prob=0.190 | count=15 | subject_support=11
- HC: 2 → 1 | prob=0.186 | count=21 | subject_support=12
- HC: 23 → 24 | prob=0.159 | count=7 | subject_support=6
- HC: 13 → 1 | prob=0.158 | count=22 | subject_support=14
- HC: 27 → 20 | prob=0.158 | count=6 | subject_support=5
- HC: 16 → 32 | prob=0.152 | count=15 | subject_support=10
- PTSD: 31 → 32 | prob=0.304 | count=34 | subject_support=18
- PTSD: 2 → 1 | prob=0.217 | count=23 | subject_support=14
- PTSD: 17 → 1 | prob=0.208 | count=20 | subject_support=11

## Group-delta state highlights
- State 32 | B4 Global on | PTSD_minus_HC_cs_delta=0.159
- State 7 | B1 Global off | PTSD_minus_HC_cs_delta=0.153
- State 23 | B4 Global on | PTSD_minus_HC_cs_delta=-0.113
- State 25 | B1 Global off | PTSD_minus_HC_cs_delta=0.112
- State 30 | B4 Global on | PTSD_minus_HC_cs_delta=-0.110
- State 29 | B2 Threat-like | PTSD_minus_HC_cs_delta=-0.102
- State 20 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.090
- State 15 | B2 Threat-like | PTSD_minus_HC_cs_delta=0.087
- State 11 | B1 Global off | PTSD_minus_HC_cs_delta=0.087
- State 27 | B4 Global on | PTSD_minus_HC_cs_delta=0.075

## Group-delta transition highlights
- 31 → 32 | PTSD_minus_HC_probability=0.193
- 17 → 1 | PTSD_minus_HC_probability=0.116
- 4 → 2 | PTSD_minus_HC_probability=0.112
- 6 → 5 | PTSD_minus_HC_probability=-0.106
- 23 → 20 | PTSD_minus_HC_probability=0.097
- 13 → 1 | PTSD_minus_HC_probability=-0.096
- 7 → 21 | PTSD_minus_HC_probability=-0.094
- 22 → 17 | PTSD_minus_HC_probability=0.091
- 23 → 16 | PTSD_minus_HC_probability=-0.091
- 22 → 1 | PTSD_minus_HC_probability=0.091

## Statistical result index
Indexed 24 readable statistical result files.
- clinical_spearman_confirmatory.csv | min_p=7.836299381394723e-05 | min_q=nan
- primary_results_for_manuscript.csv | min_p=7.836299381394723e-05 | min_q=nan
- clinical_spearman_confirmatory.csv | min_p=7.836299381394723e-05 | min_q=nan
- primary_results_for_manuscript.csv | min_p=7.836299381394723e-05 | min_q=nan
- primary_results_for_manuscript.csv | min_p=7.836299381394723e-05 | min_q=nan

> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.