from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import sys


predictor = TabularPredictor.load(rf"M:\OE0855\PB\Bund Project\Th\AutogluonModels\ag-20260401_141516")

X = pd.read_csv('X_1.csv')
y = pd.read_csv('y_1.csv')
y['Profitable Trade'] = y['Bid Ask PNL'] >= 0.15

Xy = X.copy()
Xy['Profitable Trade'] = y['Profitable Trade'].values

label = "Profitable Trade"   # <-- Replace with your label column


Xy_train = Xy[(Xy['Date'] >= '2014-01-01') & (Xy['Date'] <= '2020-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_validation = Xy[(Xy['Date'] >= '2021-01-01') & (Xy['Date'] <= '2022-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_test = Xy[(Xy['Date'] > '2022-12-31') & (Xy['Date'] <= '2024-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)

# ======================================================
# 7. APPLY THRESHOLDS
# ======================================================
preds_val= predictor.predict(Xy_validation)

# ======================================================
# 8. EVALUATE — SHOW ALL METRICS
# ======================================================
eval_precision_all = predictor.evaluate_predictions(
    y_true=Xy_validation[label],
    y_pred=preds_val,
    auxiliary_metrics=True
)


print("\n=== Precision-Optimized Threshold — ALL METRICS ===")
print(eval_precision_all)


# ======================================================
# 9. FEATURE IMPORTANCE — NEW
# ======================================================
print("\n=== Feature Importance ===")
feature_importance_df = predictor.feature_importance(Xy_validation)
print(feature_importance_df)


# ======================================================
# 10. LEADERBOARD (NEW)
# ======================================================
print("\n=== LEADERBOARD ===")
print(predictor.leaderboard(silent=False))

# ======================================================
# 11. TRAINING SUMMARY (NEW)
# ======================================================
print("\n=== TRAINING SUMMARY ===")
predictor.fit_summary()

# ======================================================
# 12. SAVE MODEL
# ======================================================
predictor.save("autogluon_model/")

# ======================================================
# 13. LOAD LATER
# ======================================================
# predictor = TabularPredictor.load("autogluon_model/")

