import pandas as pd
from sklearn.metrics import average_precision_score, ConfusionMatrixDisplay, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns


def fit_prototype_model():

    

    # Load the feature matrixes
    X = pd.read_csv('X_1.csv')
    y = pd.read_csv('y_1.csv')

    X = X[X['Date'] >= '2016-01-01']
    y = y.loc[X.index, :]


    # Create the target of our classification
    y['Profitable Trade'] = y['Bid Ask PNL'] >= 0.05
    y['Implied Move'] = X['Implied Move'].values
    y['Overpriced Event'] = y['Implied Move'] > y['Realized Move Pct']


    column_to_classify = 'Profitable Trade'

    # Drop Q-String, Realized Move Pct (1) and Realized Move Pct (2)
    X.drop(['Q-String', 'Realized Move Pct (1)', 'Realized Move Pct (2)', 'Ticker'], inplace = True, axis = 'columns')

    #X = X[X['PNL Realistic (8)'] >= 0]
    #y = y.loc[X.index, :]

    X_train = X[(X['Date'] >= '2016-01-01') & (X['Date'] <= '2020-12-31')].copy()
    X_train.drop(['Date'], inplace = True, axis = 'columns')

    X_validation = X[(X['Date'] >= '2021-01-01') & (X['Date'] <= '2021-12-31')].copy()
    X_validation.drop(['Date'], inplace = True, axis = 'columns')

    y_train = y[(y['Date'] >= '2016-01-01') & (y['Date'] <= '2020-12-31')].copy()

    y_validation = y[(y['Date'] >= '2021-01-01') & (y['Date'] <= '2021-12-31')].copy()

    model = XGBClassifier(objective = 'binary:logistic',
                            tree_method = 'hist',
                            eval_metric = 'aucpr',
                            eta = 0.001,
                            n_estimators =  400,
                            max_depth = 15,
                            subsample = 0.75,
                            gamma = 0.01,
                            min_child_weight = 3)

    weights = compute_sample_weight(class_weight='balanced', y = y_train[column_to_classify])
    model.fit(X_train, y_train[column_to_classify], sample_weight = weights)



    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(y_train[column_to_classify], model.predict(X_train), cmap=plt.cm.Blues, ax=ax)
    plt.title('Training Confusion Matrix')
    plt.show()



    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(y_validation[column_to_classify], model.predict(X_validation), cmap=plt.cm.Blues, ax=ax)
    plt.title('Validation Confusion Matrix')
    plt.show()

    for col in X_train.columns.tolist():
        sns.regplot(x = X_train[col], y = y_train['Bid Ask PNL'], logistic = True, ci=None, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title(f'Probability Profitable vs {col}')
        plt.ylabel('Probability Profitable')
        plt.show()





    print(X)
    print(y)


def fit_prototype_model_with_embeddings():

    X = pd.read_csv('X_1.csv')
    y = pd.read_csv('y_1.csv')

#fit_prototype_model()
