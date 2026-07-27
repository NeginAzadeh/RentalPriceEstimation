\# Experiment Results



This file records the results obtained while reconstructing and improving

the original university project.



\## Historical Results



Results reported in the original course project.



| Model | Original Local Result | Public Leaderboard MAE |

|---|---:|---:|

| Dummy Regressor | \~342.6\* | \~524.9 |

| Random Forest | \~119.0\* | - |

| CatBoost | \~105.0\* | - |

| LightGBM | \~96.7\* | 141.525 |



\\\* The original local values appear to correspond to MAE measured on the

log-transformed target (and displayed after scaling), so they are not directly

comparable with leaderboard MAE on the original price scale.



\---



\## Reconstructed Pipeline



Evaluation protocol:

\- 5-fold stratified cross-validation using price deciles

\- Target-dependent feature engineering fitted inside each training fold

\- MAE reported on the original rental-price scale

\- Random state: 42



\### Baseline Investigation



| Baseline | Mean MAE |

|---|---:|

| Arithmetic Mean | 545.700 |

| Median | 524.041 |

| Log-Mean + inverse transform | 524.454 |



\### Model Evaluation



| Model | Mean CV MAE | Std CV MAE | Notes |

|---|---:|---:|---|

| Dummy Regressor | 545.700 | 3.325 | Arithmetic mean baseline |

| LightGBM | 163.049 | 2.204 | Leakage-safe 5-fold CV |

| CatBoost | 189.799 | 2.817 | Leakage-safe 5-fold CV |

| Random Forest | 199.626 | 2.935 | Leakage-safe 5-fold CV |

| Linear Regression | Unstable | - | One CV fold diverged (MAE ≈ 83,095) |

| Linear Regression | Unstable | - | Extreme inverse-transformed prediction in one fold |

| Ridge Regression | Unstable | - | Same instability observed with alpha=1.0 |





\### LightGBM Fold Results



| Fold | MAE |

|---|---:|

| 1 | 160.961 |

| 2 | 160.184 |

| 3 | 165.835 |

| 4 | 165.030 |

| 5 | 163.238 |



\### CatBoost Fold Results



| Fold | MAE |

|---|---:|

| 1 | 187.256 |

| 2 | 187.430 |

| 3 | 192.016 |

| 4 | 194.233 |

| 5 | 188.061 |



LightGBM consistently outperformed CatBoost under the same leakage-safe cross-validation protocol.





\### Random Forest Fold Results



| Fold | MAE |

|---|---:|

| 1 | 196.361 |

| 2 | 195.795 |

| 3 | 202.601 |

| 4 | 202.126 |

| 5 | 201.248 |



\### Linear Regression Diagnostic



Fold MAE:



| Fold | MAE |

|---|---:|

| 1 | 232.617 |

| 2 | 83094.974 |

| 3 | 233.622 |

| 4 | 235.290 |

| 5 | 247.008 |



Ordinary least-squares regression was numerically unstable in one fold.

Because predictions are made in log-target space and transformed back with

`expm1`, extreme linear predictions can produce unrealistically large rental

prices. The aggregated mean MAE is therefore not treated as a meaningful

model-comparison result.



LightGBM       163.049 ± 2.204   ← Best

CatBoost       189.799 ± 2.817

Random Forest  199.626 ± 2.935

Dummy          545.700 ± 3.325



Linear         unstable

Ridge          unstable



LightGBM > CatBoost > Random Forest



\## Final Model



The reconstructed project selects LightGBM as the final model based on

leakage-safe 5-fold cross-validation.



\- Mean CV MAE: 163.049

\- CV standard deviation: 2.204

\- Training samples: 79,589

\- Evaluation samples: 19,898

\- Final prediction file: `outputs/submissions/submission\_lightgbm.csv`



The original course project reported a public leaderboard MAE of 141.525.

That score cannot be reproduced because the evaluation labels and original

course leaderboard are no longer available.



141.525 → نتیجه تاریخی سایت درس

163.049 → CV معتبر نسخه بازسازی‌شده

