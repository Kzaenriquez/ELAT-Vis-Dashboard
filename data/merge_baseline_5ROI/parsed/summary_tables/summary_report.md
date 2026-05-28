# ELAT-Vis Summary Tables Report

Event column: `trial_type_hrf4`
Epoch mode: `late`

## Top CS+ shifted states
- State 12 | B3 Safety-like | cs_delta=0.200 | occupancy=35
- State 30 | B4 Global on | cs_delta=0.196 | occupancy=51
- State 18 | B3 Safety-like | cs_delta=0.145 | occupancy=69
- State 11 | B1 Global off | cs_delta=0.143 | occupancy=21
- State 4 | B3 Safety-like | cs_delta=0.119 | occupancy=109

## Top CS− shifted states
- State 22 | B4 Global on | cs_delta=-0.160 | occupancy=25
- State 22 | B4 Global on | cs_delta=-0.139 | occupancy=36
- State 25 | B1 Global off | cs_delta=-0.125 | occupancy=40
- State 7 | B1 Global off | cs_delta=-0.120 | occupancy=25
- State 10 | B1 Global off | cs_delta=-0.107 | occupancy=75

## Transition backbone
- HC: 31 → 32 | prob=0.196 | count=22 | subject_support=14
- HC: 28 → 32 | prob=0.184 | count=16 | subject_support=11
- HC: 20 → 32 | prob=0.180 | count=22 | subject_support=17
- HC: 5 → 1 | prob=0.178 | count=13 | subject_support=11
- HC: 24 → 32 | prob=0.172 | count=10 | subject_support=10
- HC: 19 → 20 | prob=0.128 | count=10 | subject_support=10
- HC: 16 → 32 | prob=0.125 | count=11 | subject_support=10
- HC: 2 → 1 | prob=0.119 | count=15 | subject_support=10
- HC: 4 → 1 | prob=0.117 | count=11 | subject_support=10
- HC: 13 → 14 | prob=0.116 | count=14 | subject_support=10

## Group-delta state highlights
- State 30 | B4 Global on | PTSD_minus_HC_cs_delta=-0.275
- State 18 | B3 Safety-like | PTSD_minus_HC_cs_delta=-0.250
- State 12 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.200
- State 19 | B3 Safety-like | PTSD_minus_HC_cs_delta=-0.166
- State 26 | B4 Global on | PTSD_minus_HC_cs_delta=0.133
- State 25 | B1 Global off | PTSD_minus_HC_cs_delta=0.125
- State 29 | B2 Threat-like | PTSD_minus_HC_cs_delta=-0.125
- State 13 | B2 Threat-like | PTSD_minus_HC_cs_delta=0.123
- State 27 | B4 Global on | PTSD_minus_HC_cs_delta=-0.114
- State 2 | B1 Global off | PTSD_minus_HC_cs_delta=0.101

## Group-delta transition highlights
- 30 → 32 | PTSD_minus_HC_probability=0.201
- 11 → 1 | PTSD_minus_HC_probability=0.200
- 11 → 2 | PTSD_minus_HC_probability=-0.185
- 22 → 6 | PTSD_minus_HC_probability=0.182
- 26 → 1 | PTSD_minus_HC_probability=-0.182
- 15 → 13 | PTSD_minus_HC_probability=-0.167
- 11 → 25 | PTSD_minus_HC_probability=0.150
- 30 → 2 | PTSD_minus_HC_probability=-0.140
- 25 → 32 | PTSD_minus_HC_probability=0.131
- 26 → 17 | PTSD_minus_HC_probability=-0.121

## Statistical result index
Indexed 22 readable statistical result files.
- clinical_spearman_confirmatory.csv | min_p=0.0006390403886994 | min_q=nan
- clinical_spearman_confirmatory.csv | min_p=0.0011649049436411 | min_q=nan
- primary_results_for_manuscript.csv | min_p=0.0011649049436411 | min_q=nan
- clinical_spearman_confirmatory.csv | min_p=0.0011649049436411 | min_q=nan
- primary_results_for_manuscript.csv | min_p=0.0011649049436411 | min_q=nan

> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.