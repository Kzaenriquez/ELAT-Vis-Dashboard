# ELAT-Vis Summary Tables Report

Event column: `trial_type_hrf4`
Epoch mode: `late`

## Top CS+ shifted states
- State 13 | B2 Threat-like | cs_delta=0.232 | occupancy=164
- State 11 | B1 Global off | cs_delta=0.222 | occupancy=18
- State 10 | B1 Global off | cs_delta=0.188 | occupancy=48
- State 14 | B2 Threat-like | cs_delta=0.173 | occupancy=81
- State 10 | B1 Global off | cs_delta=0.172 | occupancy=87

## Top CS− shifted states
- State 27 | B4 Global on | cs_delta=-0.258 | occupancy=31
- State 30 | B4 Global on | cs_delta=-0.238 | occupancy=63
- State 30 | B4 Global on | cs_delta=-0.138 | occupancy=109
- State 28 | B4 Global on | cs_delta=-0.130 | occupancy=77
- State 29 | B2 Threat-like | cs_delta=-0.129 | occupancy=101

## Transition backbone
- HC: 9 → 1 | prob=0.350 | count=21 | subject_support=13
- HC: 2 → 1 | prob=0.222 | count=24 | subject_support=12
- HC: 10 → 2 | prob=0.222 | count=10 | subject_support=7
- HC: 15 → 31 | prob=0.184 | count=7 | subject_support=6
- HC: 24 → 32 | prob=0.182 | count=10 | subject_support=8
- HC: 17 → 1 | prob=0.177 | count=17 | subject_support=13
- HC: 20 → 32 | prob=0.177 | count=23 | subject_support=15
- HC: 19 → 20 | prob=0.173 | count=14 | subject_support=12
- HC: 28 → 32 | prob=0.171 | count=13 | subject_support=9
- HC: 14 → 5 | prob=0.158 | count=12 | subject_support=8

## Group-delta state highlights
- State 27 | B4 Global on | PTSD_minus_HC_cs_delta=0.329
- State 30 | B4 Global on | PTSD_minus_HC_cs_delta=-0.238
- State 29 | B2 Threat-like | PTSD_minus_HC_cs_delta=0.198
- State 22 | B4 Global on | PTSD_minus_HC_cs_delta=-0.171
- State 13 | B2 Threat-like | PTSD_minus_HC_cs_delta=-0.166
- State 11 | B1 Global off | PTSD_minus_HC_cs_delta=0.160
- State 3 | B1 Global off | PTSD_minus_HC_cs_delta=-0.152
- State 28 | B4 Global on | PTSD_minus_HC_cs_delta=-0.130
- State 19 | B3 Safety-like | PTSD_minus_HC_cs_delta=-0.114
- State 18 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.104

## Group-delta transition highlights
- 11 → 32 | PTSD_minus_HC_probability=0.294
- 6 → 2 | PTSD_minus_HC_probability=-0.192
- 9 → 1 | PTSD_minus_HC_probability=-0.188
- 22 → 23 | PTSD_minus_HC_probability=0.160
- 10 → 1 | PTSD_minus_HC_probability=0.148
- 10 → 2 | PTSD_minus_HC_probability=-0.143
- 15 → 5 | PTSD_minus_HC_probability=0.133
- 18 → 1 | PTSD_minus_HC_probability=0.132
- 11 → 12 | PTSD_minus_HC_probability=-0.125
- 22 → 19 | PTSD_minus_HC_probability=0.120

## Statistical result index
Indexed 22 readable statistical result files.
- clinical_spearman_confirmatory.csv | min_p=0.0062934349159479 | min_q=nan
- group_tests_confirmatory.csv | min_p=0.0397165403494464 | min_q=nan
- group_tests_confirmatory.csv | min_p=0.0397165403494464 | min_q=nan
- primary_results_for_manuscript.csv | min_p=0.0397165403494464 | min_q=nan
- group_tests_confirmatory.csv | min_p=0.0397165403494464 | min_q=nan

> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.