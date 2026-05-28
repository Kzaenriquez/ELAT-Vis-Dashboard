# ELAT-Vis Summary Tables Report

Event column: `trial_type_hrf4`
Epoch mode: `full`

## Top CS+ shifted states
- State 13 | B2 Threat-like | cs_delta=0.175 | occupancy=240
- State 13 | B2 Threat-like | cs_delta=0.170 | occupancy=524
- State 13 | B2 Threat-like | cs_delta=0.165 | occupancy=284
- State 10 | B1 Global off | cs_delta=0.153 | occupancy=98
- State 10 | B1 Global off | cs_delta=0.138 | occupancy=218

## Top CS− shifted states
- State 11 | B1 Global off | cs_delta=-0.220 | occupancy=41
- State 11 | B1 Global off | cs_delta=-0.218 | occupancy=87
- State 11 | B1 Global off | cs_delta=-0.217 | occupancy=46
- State 9 | B1 Global off | cs_delta=-0.202 | occupancy=124
- State 3 | B1 Global off | cs_delta=-0.173 | occupancy=110

## Transition backbone
- HC: 11 → 1 | prob=0.222 | count=10 | subject_support=9
- HC: 22 → 32 | prob=0.189 | count=7 | subject_support=7
- HC: 31 → 32 | prob=0.186 | count=41 | subject_support=15
- HC: 2 → 1 | prob=0.179 | count=39 | subject_support=17
- HC: 3 → 1 | prob=0.165 | count=20 | subject_support=13
- HC: 23 → 32 | prob=0.163 | count=17 | subject_support=14
- HC: 10 → 1 | prob=0.155 | count=17 | subject_support=12
- HC: 24 → 20 | prob=0.151 | count=24 | subject_support=18
- HC: 27 → 20 | prob=0.150 | count=12 | subject_support=8
- PTSD: 31 → 32 | prob=0.218 | count=48 | subject_support=21

## Group-delta state highlights
- State 9 | B1 Global off | PTSD_minus_HC_cs_delta=-0.264
- State 24 | B4 Global on | PTSD_minus_HC_cs_delta=0.160
- State 12 | B4 Global on | PTSD_minus_HC_cs_delta=-0.125
- State 30 | B4 Global on | PTSD_minus_HC_cs_delta=0.121
- State 16 | B4 Global on | PTSD_minus_HC_cs_delta=-0.111
- State 8 | B4 Global on | PTSD_minus_HC_cs_delta=0.099
- State 29 | B2 Threat-like | PTSD_minus_HC_cs_delta=-0.099
- State 26 | B4 Global on | PTSD_minus_HC_cs_delta=0.098
- State 21 | B1 Global off | PTSD_minus_HC_cs_delta=-0.091
- State 18 | B3 Safety-like | PTSD_minus_HC_cs_delta=0.087

## Group-delta transition highlights
- 22 → 32 | PTSD_minus_HC_probability=-0.189
- 27 → 20 | PTSD_minus_HC_probability=-0.133
- 26 → 1 | PTSD_minus_HC_probability=0.131
- 11 → 2 | PTSD_minus_HC_probability=0.122
- 6 → 13 | PTSD_minus_HC_probability=-0.106
- 6 → 14 | PTSD_minus_HC_probability=-0.104
- 11 → 17 | PTSD_minus_HC_probability=0.102
- 11 → 1 | PTSD_minus_HC_probability=-0.100
- 18 → 1 | PTSD_minus_HC_probability=0.096
- 8 → 2 | PTSD_minus_HC_probability=0.085

## Statistical result index
Indexed 22 readable statistical result files.
- clinical_spearman_confirmatory.csv | min_p=0.0002294910179779 | min_q=nan
- covariate_models_confirmatory.csv | min_p=0.0005766821375421 | min_q=nan
- covariate_models_confirmatory.csv | min_p=0.0033327240638129 | min_q=nan
- covariate_models_confirmatory.csv | min_p=0.0033327240638129 | min_q=nan
- primary_results_for_manuscript.csv | min_p=0.0033327240638129 | min_q=nan

> Note: Transition backbone rows are filtered observed transitions, not formal significance tests unless linked to confirmatory outputs.