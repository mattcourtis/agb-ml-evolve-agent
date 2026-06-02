# Gradient Boosting Investigation

Dataset: `features_iter3.parquet`, 4636 rows, 23-project LOPO CV

## Aggregate metrics

| config       |   n_features |     r2 |    rmse |     mae |   bias |
|:-------------|-------------:|-------:|--------:|--------:|-------:|
| baseline     |           80 | 0.4274 | 56.1274 | 41.2734 | 0.8703 |
| deeper       |           80 | 0.4071 | 57.1151 | 41.8762 | 1.2041 |
| very_deep    |           80 | 0.3893 | 57.9643 | 42.6908 | 1.4820 |
| stochastic   |           80 | 0.4225 | 56.3678 | 41.2982 | 0.6551 |
| regularised  |           80 | 0.4236 | 56.3124 | 41.4239 | 0.9379 |
| emb_only     |           64 | 0.4182 | 56.5767 | 41.4927 | 0.5033 |
| fast_shallow |           80 | 0.4245 | 56.2684 | 41.3090 | 0.5454 |
| ridge_all    |           80 | 0.4011 | 57.4025 | 42.5309 | 1.5250 |

## Per-quintile bias (mean pred − true, tCO₂/acre)

| config       |   q1_bias |   q2_bias |   q3_bias |   q4_bias |   q5_bias |
|:-------------|----------:|----------:|----------:|----------:|----------:|
| baseline     |      36.0 |      32.1 |      18.0 |      -9.5 |     -72.2 |
| deeper       |      37.0 |      32.7 |      18.2 |      -9.2 |     -72.6 |
| very_deep    |      37.2 |      33.9 |      18.5 |      -9.2 |     -72.9 |
| stochastic   |      35.8 |      32.2 |      17.0 |      -9.6 |     -72.1 |
| regularised  |      37.2 |      32.6 |      17.4 |      -9.7 |     -72.9 |
| emb_only     |      35.5 |      31.8 |      17.4 |     -10.1 |     -72.0 |
| fast_shallow |      34.6 |      32.0 |      18.0 |      -9.6 |     -72.2 |
| ridge_all    |      32.8 |      35.8 |      19.4 |      -8.2 |     -72.1 |

## Config details

- **baseline**: {'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
- **deeper**: {'num_leaves': 127, 'learning_rate': 0.05, 'min_child_samples': 10, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
- **very_deep**: {'num_leaves': 255, 'learning_rate': 0.03, 'min_child_samples': 5, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
- **stochastic**: {'num_leaves': 63, 'learning_rate': 0.05, 'min_child_samples': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
- **regularised**: {'num_leaves': 63, 'learning_rate': 0.03, 'min_child_samples': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
- **emb_only**: {'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 20, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
- **fast_shallow**: {'num_leaves': 15, 'learning_rate': 0.1, 'min_child_samples': 30, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
- **ridge_all**: RidgeCV(alphas=[0.1,1,10,100], cv=5)

