# ELAT-Vis Summary Tables Report

Event column: `trial_type_hrf4`
Epoch mode: `late`

## Top CS+ shifted states
- State 13 | B2 Threat-like | cs_delta=0.178 | occupancy=118
- State 14 | B2 Threat-like | cs_delta=0.164 | occupancy=55
- State 15 | B2 Threat-like | cs_delta=0.161 | occupancy=31
- State 11 | B1 Global off | cs_delta=0.154 | occupancy=13
- State 14 | B2 Threat-like | cs_delta=0.149 | occupancy=101

## Top CS− shifted states
- State 8 | B3 Safety-like | cs_delta=-0.310 | occupancy=29
- State 8 | B3 Safety-like | cs_delta=-0.266 | occupancy=64
- State 22 | B4 Global on | cs_delta=-0.250 | occupancy=8
- State 8 | B3 Safety-like | cs_delta=-0.229 | occupancy=35
- State 23 | B4 Global on | cs_delta=-0.200 | occupancy=35

## Transition backbone
- HC: 14 → 13 | prob=0.212 | count=11 | subject_support=10
- HC: 13 → 1 | prob=0.172 | count=17 | subject_support=12
- HC: 2 → 1 | prob=0.164 | count=12 | subject_support=10
- HC: 32 → 30 | prob=0.115 | count=14 | subject_support=10
- PTSD: 31 → 32 | prob=0.277 | count=18 | subject_support=11
- merged: 14 → 13 | prob=0.200 | count=19 | subject_support=17
- merged: 31 → 32 | prob=0.194 | count=24 | subject_support=16
- merged: 2 → 1 | prob=0.187 | count=26 | subject_support=19
- merged: 17 → 1 | prob=0.163 | count=22 | subject_support=14
- merged: 24 → 32 | prob=0.155 | count=16 | subject_support=13

## Group-delta state highlights
- State 22 | B4 Global on | PTSD_minus_HC_cs_delta=0.321
- State 23 | B4 Global on | PTSD_minus_HC_cs_delta=-0.267
- State 11 | B1 Global off | PTSD_minus_HC_cs_delta=-0.221
- State 25 | B1 Global off | PTSD_minus_HC_cs_delta=0.209
- State 7 | B1 Global off | PTSD_minus_HC_cs_delta=0.201
- State 32 | B4 Global on | PTSD_minus_HC_cs_delta=0.187
- State 30 | B4 Global on | PTSD_minus_HC_cs_delta=-0.186
- State 12 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.155
- State 20 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.138
- State 13 | B2 Threat-like | PTSD_minus_HC_cs_delta=-0.121

## Group-delta transition highlights
- 31 → 32 | PTSD_minus_HC_probability=0.175
- 11 → 32 | PTSD_minus_HC_probability=-0.167
- 22 → 17 | PTSD_minus_HC_probability=0.143
- 22 → 31 | PTSD_minus_HC_probability=0.143
- 22 → 1 | PTSD_minus_HC_probability=0.143
- 17 → 1 | PTSD_minus_HC_probability=0.137
- 11 → 13 | PTSD_minus_HC_probability=0.133
- 3 → 16 | PTSD_minus_HC_probability=0.133
- 11 → 20 | PTSD_minus_HC_probability=0.133
- 23 → 16 | PTSD_minus_HC_probability=-0.133

## Statistical result index
Indexed 24 readable statistical result files.
- clinical_spearman_confirmatory.csv | min_p=7.836299381394723e-05 | min_q=nan
- primary_results_for_manuscript.csv | min_p=7.836299381394723e-05 | min_q=nan
- clinical_spearman_confirmatory.csv | min_p=7.836299381394723e-05 | min_q=nan
- primary_results_for_manuscript.csv | min_p=7.836299381394723e-05 | min_q=nan
- primary_results_for_manuscript.csv | min_p=7.836299381394723e-05 | min_q=nan

> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.