
# ======================================================
# 1. IMPORTS
# ======================================================
from autogluon.tabular import TabularDataset, TabularPredictor
from a1_precision_recall_threshold import precision_with_min_recall_scorer

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


# ======================================================
# 2. LOAD DATA
# ======================================================
X = pd.read_csv('X_1.csv')
y = pd.read_csv('y_1.csv')

Xy = X.copy()
Xy['Realized Move Pct'] = y['Realized Move Pct'].values


Xy_train = Xy[(Xy['Date'] >= '2014-01-01') & (Xy['Date'] <= '2020-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_validation = Xy[(Xy['Date'] >= '2021-01-01') & (Xy['Date'] <= '2022-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_test = Xy[(Xy['Date'] > '2022-12-31') & (Xy['Date'] <= '2024-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)



label = "Realized Move Pct"   # <-- Replace with your label column

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
    'LR': {},
    'EBM': {}
}

# ======================================================
# 4. HYPERPARAMETER TUNING SETTINGS
# ======================================================
hyperparameter_tune_kwargs = {
    "num_trials": 20,
    "scheduler": "local",
    "searcher": "random",
    "time_out": 1080
}

# ======================================================
# 5. TRAIN (with tuning)
# ======================================================
predictor = TabularPredictor(
    label=label,
    problem_type="regression",
    eval_metric= 'root_mean_squared_error' # Can change to Precision, AUC, MCC or F1
).fit(
    train_data=Xy_train,
    presets="best_quality_v150",
    #hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    time_limit = 2160,
    full_weighted_ensemble_additionally = True,
    auto_stack = True,
    num_stack_levels = 2,
    num_bag_sets = 1 # This is a last resource for improving predictive accuracy, increase models fit by num_bag_sets * num_bag_folds only of worse case scenario
)


# ======================================================
# 6. PREDICT ON VALIDATION
# ======================================================
preds_val = predictor.predict(Xy_validation)


# ======================================================
# 7. EVALUATE REGRESSION METRICS
# ======================================================
eval_metrics = predictor.evaluate_predictions(
    y_true=Xy_validation[label],
    y_pred=preds_val,
    auxiliary_metrics=True   # shows RMSE, MAE, MAPE, R2, etc.
)

print("\n=== REGRESSION METRICS ===")
print(eval_metrics)



plt.figure(figsize=(8,6))
plt.scatter(Xy_validation[label], preds_val, alpha=0.6)
plt.xlabel("Actual Realized Move Pct")
plt.ylabel("Predicted Realized Move Pct")
plt.title("Actual vs Predicted (Validation)")
plt.grid(True)
plt.show()


# ======================================================
# 8. FEATURE IMPORTANCE
# ======================================================
print("\n=== FEATURE IMPORTANCE ===")
feature_importance_df = predictor.feature_importance(Xy_validation)
print(feature_importance_df)


# ======================================================
# 9. LEADERBOARD
# ======================================================
print("\n=== LEADERBOARD ===")
print(predictor.leaderboard(silent=False))


# ======================================================
# 10. TRAINING SUMMARY
# ======================================================
print("\n=== TRAINING SUMMARY ===")
predictor.fit_summary()


# ======================================================
# 11. SAVE MODEL
# ======================================================
predictor.save("autogluon_regressor/")
