
# ======================================================
# 1. IMPORTS
# ======================================================
from autogluon.tabular import TabularDataset, TabularPredictor
from a2_precision_recall_threshold import precision_with_min_recall_scorer
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


# ======================================================
# 2. LOAD DATA
# ======================================================
X = pd.read_csv(rf'D:\Option Data 2\unscaled_features\X_1.csv')#[['Date','IV Slope', 'Vix', 'Log Market Cap', 'Idiosyncratic Vol','Implied Vol', 'Implied Move', 'Realized Move Pct (1)','RSI']]
y = pd.read_csv(rf'D:\Option Data 2\unscaled_features\y_1.csv')


y['Profitable Trade'] = y['PNL'] >= 0.15

Xy = X.copy()


Xy['Profitable Trade'] = y['Profitable Trade'].values

Xy_train = Xy[(Xy['Date'] >= '2014-01-01') & (Xy['Date'] <= '2020-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
print(Xy_train.shape)
Xy_validation = Xy[(Xy['Date'] >= '2021-01-01') & (Xy['Date'] <= '2022-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_test = Xy[(Xy['Date'] > '2022-12-31') & (Xy['Date'] <= '2024-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)



label = "Profitable Trade"   # <-- Replace with your label column

# ======================================================
# 3. HYPERPARAMETERS (DL tuning: LR + decay + focal loss)
#    AutoGluon supports deep-learning models for tabular tasks.  [1](https://priorlabs.ai/tabpfn)
# ======================================================
hyperparameters = {
    "NN_TORCH": {
        # Learning-rate tuning
        "learning_rate": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],

        # Weight decay tuning
        "weight_decay": [0.0, 0.0001, 0.001, 0.01],

        # 🔥 NEW: Focal Loss tuning
        "loss": "focal_loss",
        #"gamma": [1, 2, 3],            # focusing parameter
        #"alpha": [0.25, 0.5, 0.75],     # class balancing parameter
    },
    'GBM': {},
    'CAT': {},
    'XGB': {},
    'RF': {},
    'XT': {},
    'FASTAI': {},
    'KNN': {},
    'EBM': {}
}

# ======================================================
# 4. HYPERPARAMETER TUNING SETTINGS
# ======================================================
hyperparameter_tune_kwargs = {
    "num_trials": 20,
    "scheduler": "local",
    "searcher": "bayes",
    "time_out": 1440
}

# ======================================================
# 5. TRAIN (with tuning)
# ======================================================
predictor = TabularPredictor(
    label=label,
    problem_type="binary",
    eval_metric= precision_with_min_recall_scorer # Can change to Precision, AUC, MCC or F1
).fit(
    train_data=Xy_train,
    presets="best_quality_v150",
    calibrate_decision_threshold=True,
    #hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    time_limit = 2200,
    full_weighted_ensemble_additionally = True,
    auto_stack = True,
    dynamic_stacking=False,
    num_stack_levels = 1,
    num_bag_sets = 1 # This is a last resource for improving predictive accuracy, increase models fit by num_bag_sets * num_bag_folds only of worse case scenario
)

'''
# ======================================================
# 6. THRESHOLD OPTIMIZATION — Precision + MCC ONLY
# ======================================================
best_thresh_precision = predictor.calibrate_decision_threshol(
    test_data=Xy_validation, metric="precision"
)
best_thresh_mcc = predictor.calibrate_decision_threshol(
    test_data=Xy_validation, metric="mcc"
)

print("Best Precision Threshold:", best_thresh_precision)
print("Best MCC Threshold:", best_thresh_mcc)
'''


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
