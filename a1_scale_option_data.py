from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer,StandardScaler, PolynomialFeatures
from scipy.stats import norm



# This is a Pipeline for Scaling the data and dealing with entropy etc...
class ScaleOptionData():
    def __init__(self, train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date = '2024-12-31'):
        self.X = pd.read_csv(r'D:\Option Data\unscaled_features\X_1.csv')
        self.y = pd.read_csv(r'D:\Option Data\unscaled_features\y_1.csv')

        self.class_column = 'Profitable Trade'
        self.y['Profitable Trade'] = self.y['Bid Ask PNL'] >= 0.05

        self.X.drop(['Realized Move Pct Abs (1)', 'Realized Move Pct Abs (2)'], axis = 'columns', inplace = True)

        self.X_train = self.X[self.X['Date'] <= train_end_date].copy()
        self.X_validation = self.X[self.X['Date'] <= validation_end_date].copy()
        self.X_test = self.X[self.X['Date'] <= test_end_date].copy()

        self.y_train = self.y[self.y['Date'] <= train_end_date].copy()
        self.y_validation = self.y[self.y['Date'] <= validation_end_date].copy()
        self.y_test = self.y[self.y['Date'] <= test_end_date].copy()     

        #Here we can invoke all the different transformations :)
        self.X_train = self.thin_right_tail(self.X_train, ['Abnormal Volume Call', 'Abnormal Volume Put', 'Call Put Ratio', 'Kurt Delta', 'Skew Delta'], 'log', 99)
        self.X_train = self.apply_normal_scaler(self.X_train, ['Beta', 'Implied Move', 'Implied Vol', 'PNL Bid Ask (8)', 'PNL Realistic (8)', 'Realized Move Pct (1)', 'Realized Move Pct (2)'])
        self.X_train = self.apply_tail_transformation(self.X_train, ['Abnormal Volume Call', 'Abnormal Volume Put', 'Call Put Ratio', 'Kurt Delta', 'Skew Delta'], 'hyperbolic')

        self.explore_entropy()

    def explore_entropy(self):
        for col in self.X_train.columns.tolist()[3:]:

            entropy = self.shannon_entropy(self.X_train[col], 20)
            mututal_information = self.calculate_mutual_information(self.X_train[col], self.y_train[self.class_column], 20)
            plt.hist(self.X_train[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f'{col} Entropy: {round(entropy,3)}, Mutual Informativeness {round(mututal_information, 3)}')
            plt.show()



    def shannon_entropy(self, col, bins=10):
        # 1. Handle edge case: if all values are the same or column is empty

        col = col.dropna()
        if col.nunique() <= 1:
            return 0.0
        binned_data = pd.cut(col, bins=bins)
        prob = binned_data.value_counts(normalize=True)
        prob = prob[prob > 0]
        entropy = -1 * np.sum(prob * np.log2(prob))
        max_entropy = np.log2(bins)
        return entropy / max_entropy

    def calculate_mutual_information(self, col_x, col_y, bins=10):

        df_temp = pd.concat([col_x, col_y], axis=1).dropna()
        if df_temp.empty or df_temp.iloc[:, 0].nunique() <= 1 or df_temp.iloc[:, 1].nunique() <= 1:
            return 0.0
        x_clean = df_temp.iloc[:, 0]
        y_clean = df_temp.iloc[:, 1]    

        def get_probs(data):
            binned = pd.cut(data, bins=bins, duplicates='drop')
            p = binned.value_counts(normalize=True).values
            return p[p > 0]

        p_x = get_probs(x_clean)
        p_y = get_probs(y_clean)
        h_x = -np.sum(p_x * np.log2(p_x))
        h_y = -np.sum(p_y * np.log2(p_y))

        bins_x = pd.cut(x_clean, bins=bins, labels=False, duplicates='drop')
        bins_y = pd.cut(y_clean, bins=bins, labels=False, duplicates='drop')
        
        joint_probs = pd.Series(list(zip(bins_x, bins_y))).value_counts(normalize=True).values
        h_xy = -np.sum(joint_probs * np.log2(joint_probs))

        mi = h_x + h_y - h_xy
        
        # Return normalized MI (Optional: makes it 0 to 1 scale)
        return max(0.0, mi)



    def thin_right_tail(self, X, column_list, method_list, percentile_threshold = 95):
    
        if isinstance(method_list, str):
            method_list = [method_list] * len(column_list)
        
        X_ = X.copy()
        for i in range(len(method_list)):

            threshold_val = np.percentile(X_[column_list[i]], percentile_threshold)

            mask = X_[column_list[i]] > threshold_val # We are taking the indexes of all the values which are to be changed i.e. above the threshold
                
            if method_list[i] == 'square_root':

                assert (X_.loc[mask, column_list[i]] >= 0).all(), f"Column {column_list[i]} contains negative values, cannot apply square root transformation."
                X_.loc[mask, column_list[i]] = np.sqrt(X_.loc[mask, column_list[i]])

            elif method_list[i] == 'log':
                X_.loc[mask, column_list[i]] = np.log(X_.loc[mask, column_list[i]] + 1e-8)

            elif method_list[i] == 'cubic_root':
                X_.loc[mask, column_list[i]] = np.cbrt(X_.loc[mask, column_list[i]])

        return X_

    def apply_normal_scaler(self, X, column_list):

        self.standard_scaler = StandardScaler()
        X[column_list] = self.standard_scaler.fit_transform(X[column_list])
        X[column_list] = norm.cdf(X[column_list])

        return X

    def thin_left_tail(self, X, column_list, method_list, percentile_threshold = 5):
    
        if isinstance(method_list, str):
            method_list = [method_list] * len(column_list)
        
        X_ = X.copy()
        for i in range(len(method_list)):

            threshold_val = np.percentile(X_[column_list[i]], percentile_threshold)

            mask = X_[column_list[i]] < threshold_val # We are taking the indexes of all the values which are to be changed i.e. above the threshold
                
            if method_list[i] == 'square_root':

                assert (X_.loc[mask, column_list[i]] >= 0).all(), f"Column {column_list[i]} contains negative values, cannot apply square root transformation."
                X_.loc[mask, column_list[i]] = np.sqrt(X_.loc[mask, column_list[i]])

            elif method_list[i] == 'log':
                X_.loc[mask, column_list[i]] = np.log(X_.loc[mask, column_list[i]] + 1e-8)

            elif method_list[i] == 'cubic_root':
                X_.loc[mask, column_list[i]] = np.cbrt(X_.loc[mask, column_list[i]])

        return X_


    def apply_tail_transformation(self, X, column_list, method = 'hyperbolic'):
        X_ = X.copy()
        for col in column_list:
            if method == 'hyperbolic':
                X_[col] = np.tanh(X_[col])
            elif method == 'logistic':
                X_[col] = (1 / (1 + np.exp(-X_[col]) ) - 0.5) * 2

            else:
                raise ValueError(f"Unknown method {method} for tail transformation. Use 'hyperbolic' or 'logistic'.")
        
        return X_


ScaleOptionData()