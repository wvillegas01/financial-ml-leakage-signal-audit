# Minimum robustness analyses for reviewer response

Generated: 2026-09-17 13:42:34

## Execution metadata

```json
{
  "feature_count": 18,
  "features": [
    "SMA_20",
    "SMA_50",
    "Vol_20",
    "Close_Lag1",
    "Close_Lag5",
    "Return_Lag1",
    "Return_Lag5",
    "Open_Lag1",
    "High_Lag1",
    "Low_Lag1",
    "Volume_Lag1",
    "DCOILWTICO",
    "VIXCLS",
    "DHHNGSP",
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "SP500"
  ],
  "train_complete_case_rows": 7776,
  "validation_complete_case_rows": 1764,
  "test_complete_case_rows": 1746,
  "common_sequential_test_rows": 1476,
  "common_sequential_test_dates": 164,
  "sequence_length": 30,
  "epochs": 30,
  "batch_size": 32,
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "loss": "mse",
  "neural_seeds": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "bootstrap_repetitions": 5000,
  "bootstrap_cluster": "Date"
}
```

## Multi-seed neural sensitivity

```
               Model  Seeds    N  MAE_mean   MAE_sd  RMSE_mean  RMSE_sd   R2_mean    R2_sd  Directional_Accuracy_mean  Directional_Accuracy_sd
                LSTM     10 1476  0.042156 0.000061   0.055764 0.000083  0.000030 0.002962                   0.499458                 0.002254
Simplified attention     10 1476  0.042239 0.000158   0.055890 0.000176 -0.004481 0.006363                   0.499390                 0.004644
```

## Simple baselines on the common sequential subset

```
               Model Seed    N      MAE     RMSE        R2  Directional_Accuracy
Zero-return baseline None 1476 0.042127 0.055767 -0.000063                   NaN
 Train-mean baseline None 1476 0.042129 0.055765 -0.000005              0.501355
   Lagged-return OLS None 1476 0.042131 0.055754  0.000385              0.502033
```

## Date-cluster bootstrap for directional accuracy

```
            Model  N_asset_date  N_dates  Observed_Directional_Accuracy  Bootstrap_CI95_Low  Bootstrap_CI95_High  Majority_Class_Baseline  Bootstrap_p_vs_0.5  Bootstrap_p_vs_majority
             LSTM          1476      164                       0.489160            0.455268             0.523713                 0.501355              0.5584                   0.5108
      Transformer          1476      164                       0.488482            0.453930             0.522358                 0.501355              0.4984                   0.4544
          XGBoost          1476      164                       0.495257            0.461382             0.529133                 0.501355              0.8216                   0.7548
         Ensemble          1476      164                       0.498645            0.466802             0.531165                 0.501355              0.9332                   0.8636
       Train_mean          1476      164                       0.501355            0.466125             0.537940                 0.501355              0.9696                   1.0000
Lagged_return_OLS          1476      164                       0.502033            0.470190             0.533198                 0.501355              0.9120                   0.9736
```
