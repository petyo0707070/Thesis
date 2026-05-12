from autogluon.core.metrics import make_scorer
from sklearn.metrics import recall_score, precision_score

def _precision_with_min_recall_func(y_true, y_pred):
    """Internal function for the logic."""
    recall = recall_score(y_true, y_pred, zero_division=0)
    if recall < 0.1:
        return 0.0
    return precision_score(y_true, y_pred, zero_division=0)

# The exported scorer object
precision_with_min_recall_scorer = make_scorer(
    name='precision_min_recall',
    score_func=_precision_with_min_recall_func,
    optimum=1.0,
    greater_is_better=True,
    needs_pred=True
)