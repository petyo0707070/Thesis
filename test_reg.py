from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import plotly.express as px


import pandas as pd
import numpy as np
import sys


predictor = TabularPredictor.load(rf"M:\OE0855\PB\Bund Project\Th\AutogluonModels\ag-20260402_132401")


X = pd.read_csv('X_1.csv')
y = pd.read_csv('y_1.csv')
y['Profitable Trade'] = y['Bid Ask PNL'] >= 0.15

Xy = X.copy()
Xy['Realized Move Pct'] = y['Realized Move Pct'].values

label = "Realized Move Pct"   # <-- Replace with your label column


Xy_train = Xy[(Xy['Date'] >= '2014-01-01') & (Xy['Date'] <= '2020-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_validation = Xy[(Xy['Date'] >= '2021-01-01') & (Xy['Date'] <= '2022-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)
Xy_test = Xy[(Xy['Date'] > '2022-12-31') & (Xy['Date'] <= '2024-12-31')][[col for col in Xy.columns if col != 'Date' and col != 'Q-String']].reset_index(drop = True)


model_names = predictor.model_names() # Get model names
y_true = Xy_validation[label]
results = [] # This will store metrics we compute for performance evaluation


for model in model_names:
    preds_m = predictor.predict(Xy_validation, model=model)
    
    rmse = root_mean_squared_error(Xy_validation[label], preds_m, squared=False)
    mae  = mean_absolute_error(Xy_validation[label], preds_m)
    r2   = r2_score(Xy_validation[label], preds_m)
    
    results.append([model, rmse, mae, r2])



df_metrics = pd.DataFrame(results, columns=["model", "RSME", "MAE", "R2"])


fig = px.scatter(
    df_metrics,
    x="RMSE",
    y="R2",
    color="model",
    hover_name="model",
    title="RMSE vs R2 by model (Interactive)",
    size_max=12
)

fig.update_traces(marker=dict(size=14, line=dict(width=1, color="black")))
fig.update_layout(
    width=900,
    height=700,
    xaxis_title="RMSE",
    yaxis_title="R2"
)

fig.show()



# ======================================================
# 7. APPLY THRESHOLDS
# ======================================================
preds_val= predictor.predict(Xy_validation)

# ======================================================
# 8. EVALUATE — SHOW ALL METRICS
# ======================================================
eval_precision_all = predictor.evaluate_predictions(
    y_true=y_true,
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

# ======================================================
# 12. SAVE MODEL
# ======================================================
predictor.save("autogluon_model/")

# ======================================================
# 13. LOAD LATER
# ======================================================
# predictor = TabularPredictor.load("autogluon_model/")

