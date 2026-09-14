| case | group | policy | setting | save scalar (mean±std) | save w/o decision cost | monitored GP/step | Newton it (ref) | u_max | alpha_rel_final |
|---|---|---|---|---|---|---|---|---|---|
| notched_R4_H20_L40 | family_unseen | kappa | {"monitor_skip_kappa": 0.0} | 0.060±0.010 | 0.060 | 29219 | 102 (102) | 9.1e-14 | 2.9e-14 |
| notched_R4_H20_L40 | family_unseen | kappa | {"monitor_skip_kappa": 1.0} | 0.518±0.003 | 0.518 | 4101 | 102 (102) | 9.1e-14 | 2.9e-14 |
| notched_R4_H20_L40 | family_unseen | kappa | {"monitor_skip_kappa": 1.5} | 0.480±0.003 | 0.480 | 6232 | 102 (102) | 9.1e-14 | 2.9e-14 |
| notched_R4_H20_L40 | family_unseen | kappa | {"monitor_skip_kappa": 2.0} | 0.439±0.004 | 0.439 | 8447 | 102 (102) | 9.1e-14 | 2.9e-14 |
| notched_R4_H20_L40 | family_unseen | kappa | {"monitor_skip_kappa": 3.0} | 0.365±0.005 | 0.365 | 12505 | 102 (102) | 9.1e-14 | 2.9e-14 |
| notched_R4_H20_L40 | family_unseen | oracle | {"horizon_safety": 1.0} | 0.560±0.002 | 0.560 | 1850 | 102 (102) | 1.9e-13 | 1.6e-13 |
| notched_R4_H20_L40 | family_unseen | oracle | {"horizon_safety": 0.5} | 0.511±0.003 | 0.512 | 4480 | 102 (102) | 9.1e-14 | 2.9e-14 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.75, "name": "gbm"} | 0.585±0.001 | 0.598 | 2981 | 97 (102) | 2.2e-02 | 6.0e-02 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm"} | 0.520±0.002 | 0.539 | 4486 | 104 (102) | 1.5e-02 | 2.5e-02 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.25, "name": "gbm"} | 0.425±0.006 | 0.455 | 7664 | 104 (102) | 3.5e-03 | 8.1e-04 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.75, "name": "linear"} | 0.620±0.000 | 0.627 | 2655 | 92 (102) | 1.2e-02 | 7.0e-02 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "linear"} | 0.549±0.000 | 0.559 | 4830 | 100 (102) | 1.6e-02 | 6.7e-03 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.25, "name": "linear"} | 0.431±0.000 | 0.442 | 8755 | 108 (102) | 3.2e-03 | 2.2e-03 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 1.0, "name": "gbm_q10"} | 0.547±0.007 | 0.568 | 2136 | 118 (102) | 1.2e-02 | 6.2e-02 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm_q10"} | 0.527±0.004 | 0.553 | 4148 | 106 (102) | 7.9e-03 | 7.9e-03 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 1.0, "name": "gbm_q05"} | 0.511±0.018 | 0.538 | 2567 | 112 (102) | 5.8e-03 | 3.9e-03 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm_q05"} | 0.506±0.007 | 0.539 | 4273 | 106 (102) | 6.0e-03 | 3.5e-03 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 1.0, "name": "gbm_q02"} | 0.517±0.004 | 0.538 | 3206 | 111 (102) | 3.9e-03 | 2.8e-03 |
| notched_R4_H20_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm_q02"} | 0.483±0.002 | 0.511 | 4710 | 102 (102) | 3.6e-04 | 6.8e-05 |
| notched_R3.5_H22_L40 | family_unseen | kappa | {"monitor_skip_kappa": 0.0} | 0.076±0.010 | 0.076 | 32613 | 102 (102) | 6.7e-14 | 2.1e-14 |
| notched_R3.5_H22_L40 | family_unseen | kappa | {"monitor_skip_kappa": 1.0} | 0.525±0.001 | 0.525 | 5101 | 102 (102) | 6.7e-14 | 2.1e-14 |
| notched_R3.5_H22_L40 | family_unseen | kappa | {"monitor_skip_kappa": 1.5} | 0.483±0.000 | 0.483 | 7670 | 102 (102) | 6.7e-14 | 2.1e-14 |
| notched_R3.5_H22_L40 | family_unseen | kappa | {"monitor_skip_kappa": 2.0} | 0.439±0.001 | 0.439 | 10392 | 102 (102) | 6.7e-14 | 2.1e-14 |
| notched_R3.5_H22_L40 | family_unseen | kappa | {"monitor_skip_kappa": 3.0} | 0.357±0.003 | 0.357 | 15410 | 102 (102) | 6.7e-14 | 2.1e-14 |
| notched_R3.5_H22_L40 | family_unseen | oracle | {"horizon_safety": 1.0} | 0.575±0.002 | 0.575 | 2088 | 102 (102) | 9.4e-12 | 6.5e-12 |
| notched_R3.5_H22_L40 | family_unseen | oracle | {"horizon_safety": 0.5} | 0.527±0.001 | 0.528 | 4968 | 102 (102) | 6.7e-14 | 2.1e-14 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.75, "name": "gbm"} | 0.616±0.007 | 0.626 | 3193 | 98 (102) | 2.2e-02 | 3.8e-02 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm"} | 0.557±0.017 | 0.577 | 4907 | 104 (102) | 1.7e-02 | 1.1e-02 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.25, "name": "gbm"} | 0.439±0.004 | 0.468 | 8966 | 105 (102) | 4.8e-03 | 8.9e-04 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.75, "name": "linear"} | 0.621±0.000 | 0.628 | 2817 | 100 (102) | 1.3e-02 | 6.9e-02 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "linear"} | 0.585±0.000 | 0.593 | 4788 | 100 (102) | 1.9e-02 | 4.8e-03 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.25, "name": "linear"} | 0.445±0.000 | 0.457 | 9688 | 106 (102) | 2.8e-03 | 1.2e-03 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 1.0, "name": "gbm_q10"} | 0.607±0.016 | 0.628 | 2397 | 111 (102) | 2.6e-02 | 1.1e-01 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm_q10"} | 0.541±0.004 | 0.568 | 4558 | 108 (102) | 1.3e-02 | 1.5e-02 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 1.0, "name": "gbm_q05"} | 0.543±0.001 | 0.568 | 2841 | 112 (102) | 9.8e-03 | 1.5e-02 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm_q05"} | 0.516±0.009 | 0.550 | 5222 | 106 (102) | 9.5e-03 | 5.2e-03 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 1.0, "name": "gbm_q02"} | 0.536±0.001 | 0.557 | 3508 | 107 (102) | 2.5e-03 | 2.0e-03 |
| notched_R3.5_H22_L40 | family_unseen | learned | {"horizon_safety": 0.5, "name": "gbm_q02"} | 0.490±0.001 | 0.520 | 5611 | 102 (102) | 2.4e-04 | 1.2e-04 |
| cantilever | other | kappa | {"monitor_skip_kappa": 0.0} | 0.067±0.003 | 0.067 | 19802 | 97 (97) | 4.9e-13 | 2.2e-13 |
| cantilever | other | kappa | {"monitor_skip_kappa": 1.0} | 0.490±0.002 | 0.490 | 2291 | 97 (97) | 4.9e-13 | 2.2e-13 |
| cantilever | other | kappa | {"monitor_skip_kappa": 1.5} | 0.461±0.002 | 0.461 | 3515 | 97 (97) | 4.9e-13 | 2.2e-13 |
| cantilever | other | kappa | {"monitor_skip_kappa": 2.0} | 0.430±0.002 | 0.430 | 4774 | 97 (97) | 4.9e-13 | 2.2e-13 |
| cantilever | other | kappa | {"monitor_skip_kappa": 3.0} | 0.363±0.003 | 0.363 | 7544 | 97 (97) | 4.9e-13 | 2.2e-13 |
| cantilever | other | oracle | {"horizon_safety": 1.0} | 0.520±0.002 | 0.520 | 1170 | 96 (97) | 1.6e-11 | 3.6e-11 |
| cantilever | other | oracle | {"horizon_safety": 0.5} | 0.480±0.002 | 0.480 | 2694 | 97 (97) | 4.9e-13 | 2.2e-13 |
| cantilever | other | learned | {"horizon_safety": 0.75, "name": "gbm"} | 0.499±0.015 | 0.520 | 2533 | 99 (97) | 2.1e-02 | 2.0e-02 |
| cantilever | other | learned | {"horizon_safety": 0.5, "name": "gbm"} | 0.453±0.006 | 0.476 | 3204 | 103 (97) | 4.2e-03 | 4.8e-03 |
| cantilever | other | learned | {"horizon_safety": 0.25, "name": "gbm"} | 0.394±0.008 | 0.427 | 5423 | 98 (97) | 3.6e-03 | 1.8e-03 |
| cantilever | other | learned | {"horizon_safety": 0.75, "name": "linear"} | 0.516±0.000 | 0.522 | 2547 | 98 (97) | 2.1e-02 | 6.0e-03 |
| cantilever | other | learned | {"horizon_safety": 0.5, "name": "linear"} | 0.462±0.000 | 0.471 | 3421 | 98 (97) | 1.9e-02 | 2.4e-03 |
| cantilever | other | learned | {"horizon_safety": 0.25, "name": "linear"} | 0.405±0.000 | 0.418 | 5257 | 100 (97) | 4.1e-03 | 6.8e-04 |
| cantilever | other | learned | {"horizon_safety": 1.0, "name": "gbm_q10"} | 0.488±0.007 | 0.513 | 2130 | 97 (97) | 1.1e-02 | 1.7e-02 |
| cantilever | other | learned | {"horizon_safety": 0.5, "name": "gbm_q10"} | 0.434±0.008 | 0.467 | 3119 | 102 (97) | 2.1e-03 | 1.2e-03 |
| cantilever | other | learned | {"horizon_safety": 1.0, "name": "gbm_q05"} | 0.494±0.007 | 0.524 | 2309 | 103 (97) | 1.5e-02 | 2.1e-02 |
| cantilever | other | learned | {"horizon_safety": 0.5, "name": "gbm_q05"} | 0.450±0.001 | 0.492 | 2838 | 100 (97) | 9.0e-03 | 3.8e-03 |
| cantilever | other | learned | {"horizon_safety": 1.0, "name": "gbm_q02"} | 0.461±0.009 | 0.488 | 2272 | 102 (97) | 7.6e-03 | 2.5e-03 |
| cantilever | other | learned | {"horizon_safety": 0.5, "name": "gbm_q02"} | 0.432±0.003 | 0.467 | 3026 | 98 (97) | 7.4e-04 | 1.4e-04 |
| cyclic_notched_kin | other | kappa | {"monitor_skip_kappa": 0.0} | 0.071±0.020 | 0.071 | 30108 | 205 (205) | 8.9e-13 | 1.4e-13 |
| cyclic_notched_kin | other | kappa | {"monitor_skip_kappa": 1.0} | 0.573±0.006 | 0.573 | 6980 | 205 (205) | 8.9e-13 | 1.4e-13 |
| cyclic_notched_kin | other | kappa | {"monitor_skip_kappa": 1.5} | 0.494±0.009 | 0.494 | 10614 | 205 (205) | 8.9e-13 | 1.4e-13 |
| cyclic_notched_kin | other | kappa | {"monitor_skip_kappa": 2.0} | 0.420±0.011 | 0.420 | 14025 | 205 (205) | 8.9e-13 | 1.4e-13 |
| cyclic_notched_kin | other | kappa | {"monitor_skip_kappa": 3.0} | 0.292±0.014 | 0.292 | 19905 | 205 (205) | 8.9e-13 | 1.4e-13 |
| cyclic_notched_kin | other | oracle | {"horizon_safety": 1.0} | 0.693±0.003 | 0.693 | 1709 | 203 (205) | 1.0e-10 | 4.0e-11 |
| cyclic_notched_kin | other | oracle | {"horizon_safety": 0.5} | 0.622±0.005 | 0.623 | 4672 | 205 (205) | 8.7e-13 | 1.5e-13 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.75, "name": "gbm"} | 0.721±0.007 | 0.743 | 2997 | 173 (205) | 3.9e-01 | 6.2e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm"} | 0.651±0.006 | 0.678 | 4385 | 208 (205) | 2.1e-01 | 6.0e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.25, "name": "gbm"} | 0.523±0.003 | 0.561 | 8448 | 217 (205) | 4.8e-02 | 1.6e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.75, "name": "linear"} | 0.636±0.000 | 0.648 | 5613 | 200 (205) | 5.3e-02 | 2.6e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.5, "name": "linear"} | 0.601±0.000 | 0.615 | 6655 | 204 (205) | 1.4e-01 | 2.4e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.25, "name": "linear"} | 0.474±0.000 | 0.488 | 11240 | 205 (205) | 1.3e-02 | 2.5e-03 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 1.0, "name": "gbm_q10"} | 0.760±0.005 | 0.786 | 2649 | 191 (205) | 2.1e-01 | 1.2e-01 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm_q10"} | 0.647±0.005 | 0.681 | 4300 | 210 (205) | 1.4e-01 | 3.7e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 1.0, "name": "gbm_q05"} | 0.720±0.003 | 0.751 | 2787 | 196 (205) | 2.2e-01 | 5.6e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm_q05"} | 0.622±0.006 | 0.663 | 4295 | 210 (205) | 3.1e-02 | 1.2e-02 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 1.0, "name": "gbm_q02"} | 0.674±0.002 | 0.702 | 3015 | 198 (205) | 2.4e-02 | 7.6e-03 |
| cyclic_notched_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm_q02"} | 0.577±0.002 | 0.615 | 5318 | 210 (205) | 8.5e-03 | 6.8e-03 |
| cyclic_cantilever_kin | other | kappa | {"monitor_skip_kappa": 0.0} | 0.066±0.003 | 0.066 | 20691 | 212 (212) | 4.2e-12 | 4.0e-12 |
| cyclic_cantilever_kin | other | kappa | {"monitor_skip_kappa": 1.0} | 0.457±0.003 | 0.457 | 4306 | 212 (212) | 4.2e-12 | 4.0e-12 |
| cyclic_cantilever_kin | other | kappa | {"monitor_skip_kappa": 1.5} | 0.397±0.002 | 0.397 | 6814 | 212 (212) | 4.2e-12 | 4.0e-12 |
| cyclic_cantilever_kin | other | kappa | {"monitor_skip_kappa": 2.0} | 0.349±0.001 | 0.349 | 8813 | 212 (212) | 4.2e-12 | 4.0e-12 |
| cyclic_cantilever_kin | other | kappa | {"monitor_skip_kappa": 3.0} | 0.216±0.001 | 0.216 | 14396 | 212 (212) | 4.2e-12 | 4.0e-12 |
| cyclic_cantilever_kin | other | oracle | {"horizon_safety": 1.0} | 0.530±0.003 | 0.530 | 1320 | 211 (212) | 2.3e-11 | 1.2e-11 |
| cyclic_cantilever_kin | other | oracle | {"horizon_safety": 0.5} | 0.490±0.003 | 0.490 | 2891 | 212 (212) | 4.8e-12 | 9.8e-12 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.75, "name": "gbm"} | 0.523±0.004 | 0.549 | 2265 | 219 (212) | 3.6e-01 | 2.6e-02 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm"} | 0.454±0.006 | 0.484 | 3104 | 227 (212) | 1.1e-01 | 1.1e-02 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.25, "name": "gbm"} | 0.388±0.010 | 0.424 | 5791 | 223 (212) | 3.3e-02 | 3.5e-03 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.5, "name": "linear"} | 0.440±0.000 | 0.454 | 4510 | 237 (212) | 4.5e-02 | 3.7e-02 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.25, "name": "linear"} | 0.382±0.000 | 0.396 | 6480 | 217 (212) | 1.0e-02 | 7.1e-04 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 1.0, "name": "gbm_q10"} | 0.526±0.017 | 0.557 | 1892 | 216 (212) | 1.5e-01 | 2.5e-02 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm_q10"} | 0.442±0.006 | 0.479 | 3015 | 220 (212) | 7.5e-02 | 7.5e-03 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 1.0, "name": "gbm_q05"} | 0.475±0.015 | 0.518 | 2257 | 237 (212) | 3.4e-01 | 4.7e-02 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm_q05"} | 0.412±0.004 | 0.457 | 3319 | 224 (212) | 6.4e-02 | 2.5e-03 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 1.0, "name": "gbm_q02"} | 0.461±0.001 | 0.499 | 2445 | 215 (212) | 9.8e-02 | 1.3e-02 |
| cyclic_cantilever_kin | other | learned | {"horizon_safety": 0.5, "name": "gbm_q02"} | 0.433±0.000 | 0.471 | 3669 | 200 (212) | 5.2e-03 | 2.9e-04 |
