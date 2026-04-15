
# ======================================================
# 1. IMPORTS
# ======================================================
from autogluon.tabular import TabularDataset, TabularPredictor
from a1_precision_recall_threshold import precision_with_min_recall_scorer

import pandas as pd
import numpy as np
import sys



class AutoML():
    def __init__(self, train_start_date='2016-01-01', train_end_date='2020-12-31', train_size=4, validation_size=1, step_size=1):
        self.train_start = pd.to_datetime(train_start_date)  # Get the date from which we start the walkforward
        self.train_end = pd.to_datetime(train_end_date)  # Get the date on which we end the walkforward

        self.train_size = train_size  # On how many quarters to train
        self.validation_size = validation_size  # On how many quarters to test
        self.step_size = step_size  # By how many quarters to move the sliding window
        self.generate_folds()

    def generate_folds(self):
        current_train_start = self.train_start
        current_train_end = self.train_start + pd.offsets.QuarterEnd(self.train_size)  # We subtract one day to get the last day of the quarter
        fold_index = 1
        while current_train_end + pd.offsets.QuarterEnd(self.validation_size) <= self.train_end:
            print(f'At fold {fold_index}')
            val_start = current_train_end + pd.Timedelta(days=1)  # The validation starts the day after the training ends
            val_end = val_start + pd.offsets.QuarterEnd(self.validation_size)
            # ======================================================
            # 2. LOAD DATA
            # ======================================================
            X = pd.read_csv('X_1.csv')
            X['Date'] = pd.to_datetime(X['Date'])
            y = pd.read_csv('y_1.csv')
            y['Profitable Trade'] = y['Realistic PNL'] >= 0.15
            Xy = X.copy()
            Xy['Profitable Trade'] = y['Profitable Trade'].values
            Xy_train = Xy[(Xy['Date'] >= current_train_start) & (Xy['Date'] <= current_train_end)][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
            Xy_validation = Xy[(Xy['Date'] >= val_start) & (Xy['Date'] <= val_end)][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
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
                'LR': {},
                'EBM': {}
            }
            # ======================================================
            # 4. HYPERPARAMETER TUNING SETTINGS
            # ======================================================
            hyperparameter_tune_kwargs = {
                "num_trials": 20,
                "scheduler": "local",
                "searcher": "bayes",
                "time_out": 800
            }
            # ======================================================
            # 5. TRAIN (with tuning)
            # ======================================================
            predictor = TabularPredictor(
                label=label,
                problem_type="binary",
                eval_metric= precision_with_min_recall_scorer, # Can change to Precision, AUC, MCC or F1
                path = f"autogluon_model/{current_train_start.strftime('%Y-%m-%d')}-{current_train_end.strftime('%Y-%m-%d')}",
                verbosity= 1
            ).fit(
                train_data=Xy_train,
                presets="best_quality_v150",
                calibrate_decision_threshold=True,
                hyperparameters=hyperparameters,
                hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                time_limit = 1600,
                #full_weighted_ensemble_additionally = True,
                auto_stack = True,
                num_stack_levels = 1,
                num_bag_sets = 1 # This is a last resource for improving predictive accuracy, increase models fit by num_bag_sets * num_bag_folds only of worse case scenario
            )
       
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
            # 10. LEADERBOARD (NEW)
            # ======================================================
            print("\n=== LEADERBOARD ===")
            print(predictor.leaderboard(silent=False))
            # ======================================================
            # 11. TRAINING SUMMARY (NEW)
            # ======================================================
            print("\n=== TRAINING SUMMARY ===")
            predictor.fit_summary()

            fold_index += 1
            current_train_start = current_train_start + pd.offsets.QuarterEnd(self.step_size)
            current_train_end = current_train_end + pd.offsets.QuarterEnd(self.step_size)



if __name__ == '__main__':
    AutoML()