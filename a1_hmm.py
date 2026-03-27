import pandas as pd
from hmmlearn.hmm import GaussianHMM, GMMHMM
import matplotlib.pyplot as plt
import numpy as np

def main():
    df = pd.read_csv('vix_timeseries.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['Vix'] = df['CLOSE']
    df['Vix Change'] = df['Vix'] - df['Vix'].shift(1)
    
    #df.loc[:, 'Vix'] = np.log(df['Vix'])

    X = df[['DATE', 'Vix', 'Vix Change']]

    X_train = df[(df['DATE'] >= '2014-01-01') & (df['DATE'] <= '2020-12-31')][['Vix', 'Vix Change']].reset_index(drop = True)
    X_validation = df[(df['DATE'] >= '2021-01-01') & (df['DATE'] <= '2022-12-31')][['Vix', 'Vix Change']].reset_index(drop = True)
    model = GMMHMM(n_components=3, n_mix = 1 ,covariance_type='full', n_iter= 2000)
    model.fit(X_train)

    model, order = reorder_states_by_vix(model, feature_index=0)

    hidden_states = model.predict(X_train)
    print(hidden_states)
    
    print("Means:\n", model.means_)
    print("Variances:\n", model.covars_)


    
    plt.figure(figsize=(15,5))
    plt.scatter(range(len(X_train)), X_train['Vix'], c=hidden_states, cmap='viridis', s=10)
    plt.colorbar(label="Hidden state")
    plt.title("VIX with Hidden Markov Model States Train")
    plt.show()

    plt.figure(figsize=(15,5))
    plt.scatter(range(len(X_validation)), X_validation['Vix'], c=model.predict(X_validation), cmap='viridis', s=10)
    plt.colorbar(label="Hidden state")
    plt.title("VIX with Hidden Markov Model States Validation")
    plt.show()


def reorder_states_by_vix(model, feature_index=0):
    order = np.argsort(model.means_[:, 0, feature_index])
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order][:, order]
    model.means_ = model.means_[order]
    model.covars_ = model.covars_[order]
    return model, order


if __name__ == '__main__':
    main()