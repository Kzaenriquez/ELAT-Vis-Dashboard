# ELAT-Vis Summary Tables Report

Event column: `trial_type_hrf4`
Epoch mode: `full`

## Top CS+ shifted states
- State 11 | B1 Global off | cs_delta=0.195 | occupancy=41
- State 30 | B4 Global on | cs_delta=0.185 | occupancy=108
- State 12 | B3 Safety-like | cs_delta=0.145 | occupancy=69
- State 8 | B3 Safety-like | cs_delta=0.111 | occupancy=99
- State 15 | B2 Threat-like | cs_delta=0.103 | occupancy=87

## Top CS− shifted states
- State 7 | B1 Global off | cs_delta=-0.154 | occupancy=52
- State 22 | B4 Global on | cs_delta=-0.150 | occupancy=40
- State 21 | B1 Global off | cs_delta=-0.093 | occupancy=151
- State 22 | B4 Global on | cs_delta=-0.087 | occupancy=69
- State 21 | B1 Global off | cs_delta=-0.086 | occupancy=245

## Transition backbone
- HC: 9 → 1 | prob=0.186 | count=26 | subject_support=15
- HC: 6 → 1 | prob=0.182 | count=12 | subject_support=10
- HC: 20 → 32 | prob=0.168 | count=43 | subject_support=21
- HC: 24 → 32 | prob=0.168 | count=24 | subject_support=16
- HC: 28 → 32 | prob=0.159 | count=28 | subject_support=13
- HC: 31 → 32 | prob=0.155 | count=36 | subject_support=20
- HC: 15 → 13 | prob=0.155 | count=17 | subject_support=12
- HC: 27 → 19 | prob=0.151 | count=11 | subject_support=6
- HC: 3 → 1 | prob=0.143 | count=15 | subject_support=10
- HC: 5 → 1 | prob=0.138 | count=24 | subject_support=15

## Group-delta state highlights
- State 11 | B1 Global off | PTSD_minus_HC_cs_delta=0.219
- State 30 | B4 Global on | PTSD_minus_HC_cs_delta=-0.204
- State 7 | B1 Global off | PTSD_minus_HC_cs_delta=-0.181
- State 22 | B4 Global on | PTSD_minus_HC_cs_delta=0.150
- State 15 | B2 Threat-like | PTSD_minus_HC_cs_delta=0.136
- State 12 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.130
- State 3 | B1 Global off | PTSD_minus_HC_cs_delta=0.125
- State 8 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.120
- State 18 | B3 Safety-like | PTSD_minus_HC_cs_delta=-0.115
- State 4 | B3 Safety-like | PTSD_minus_HC_cs_delta=-0.089

## Group-delta transition highlights
- 22 → 6 | PTSD_minus_HC_probability=0.138
- 30 → 32 | PTSD_minus_HC_probability=0.126
- 22 → 31 | PTSD_minus_HC_probability=0.122
- 10 → 1 | PTSD_minus_HC_probability=0.119
- 6 → 1 | PTSD_minus_HC_probability=-0.114
- 26 → 1 | PTSD_minus_HC_probability=-0.108
- 18 → 20 | PTSD_minus_HC_probability=0.106
- 11 → 32 | PTSD_minus_HC_probability=-0.103
- 11 → 21 | PTSD_minus_HC_probability=-0.103
- 11 → 1 | PTSD_minus_HC_probability=0.099

## Statistical result index
Indexed 22 readable statistical result files.
- clinical_spearman_confirmatory.csv | min_p=0.0006390403886994 | min_q=nan
- clinical_spearman_confirmatory.csv | min_p=0.0011649049436411 | min_q=nan
- primary_results_for_manuscript.csv | min_p=0.0011649049436411 | min_q=nan
- clinical_spearman_confirmatory.csv | min_p=0.0011649049436411 | min_q=nan
- primary_results_for_manuscript.csv | min_p=0.0011649049436411 | min_q=nan

> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.