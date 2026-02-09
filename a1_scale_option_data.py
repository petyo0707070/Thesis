from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer,StandardScaler, PolynomialFeatures, PowerTransformer, MinMaxScaler
from scipy.stats import norm
import os
import sys


# This is a Pipeline for Scaling the data and dealing with entropy etc...
class ScaleOptionData():
    def __init__(self, train_start_date = '2016-01-01', train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date = '2024-12-31'):
        self.train_start_date = train_start_date

        self.X = pd.read_csv(r'D:\Option Data\unscaled_features\X_1.csv') if os.path.exists(r'D:\Option Data\unscaled_features\X_1.csv') else pd.read_csv(r'M:\OE0855\PB\Bund Project\Th\X_1.csv')
        self.y = pd.read_csv(r'D:\Option Data\unscaled_features\y_1.csv') if os.path.exists(r'D:\Option Data\unscaled_features\y_1.csv') else pd.read_csv(r'M:\OE0855\PB\Bund Project\Th\y_1.csv')

        if 'Profitable Trade' not in self.y.columns:
            self.y['Profitable Trade'] = self.y['Bid Ask PNL'] >= 0.05

        # Sort By ticker than date
        self.X = self.X.sort_values(['Ticker', 'Date'])
        self.y = self.y.loc[self.X.index]

        self.X.reset_index(inplace = True, drop = True)
        self.y.reset_index(inplace = True, drop = True)

        self.right_tail_dic = {}
        self.left_tail_dic = {}
        self.tail_transformation_dic = {}

        self.normal_scaler_columns = []
        self.power_scaler_columns = []
        self.min_max_scaler_columns = []
        self.robust_scaler_columns = []
        self.normal_cdf_columns = []

        self.class_column = 'Profitable Trade'
        #self.y['Profitable Trade'] = self.y['Bid Ask PNL'] >= 0.05

        self.X.drop(['Realized Move Pct Abs (1)', 'Realized Move Pct Abs (2)'], axis = 'columns', inplace = True)

        self.X_train = self.X[self.X['Date'] <= train_end_date].copy()
        self.X_validation = self.X[(self.X['Date'] > train_end_date) & (self.X['Date'] <= validation_end_date)].copy()
        self.X_test = self.X[(self.X['Date'] > validation_end_date) & (self.X['Date'] <= test_end_date)].copy()

        self.y_train = self.y[self.y['Date'] <= train_end_date].copy()
        self.y_validation = self.y[(self.y['Date'] > train_end_date) & (self.y['Date'] <= validation_end_date)].copy()
        self.y_test = self.y[(self.y['Date'] > validation_end_date) & (self.y['Date'] <= test_end_date)].copy()     

        
        #Here we can invoke all the different transformations :)

        # Remove the shittiest outliers
        self.X_train = self.thin_right_tail(self.X_train, ['Abnormal Volume Call', 'Abnormal Volume Put','Call Put Ratio', 'Kurt Delta', 'Skew Delta'], 'log', 99)

        # Some columns have a theoretical reason / actually are remotely normally distributed, we make our life really easy by apply the Normal CDF
        self.X_train = self.apply_normal_scaler(self.X_train, ['Beta', 'Implied Move', 'Implied Vol', 'PNL Bid Ask (8)', 'PNL Realistic (8)', 'Realized Move Pct (1)', 'Realized Move Pct (2)'])
        
        # Get rid of some shitty negative outliers
        self.X_train = self.thin_left_tail(self.X_train, ['Kurt Delta', 'Skew Delta'], 'cubic_root', 1)


        # Apply Power scaler for the heavily skewed positive distributions
        self.X_train = self.apply_power_scaler(self.X_train, ['Abnormal Volume Call', 'Abnormal Volume Put','ATM Call Open Interest Ratio', 'ATM Put Open Interest Ratio', 'IV Slope', 'Vix', 'Call Put Ratio']) 
        # Some of the resulting columns become Normally Distributed so we apply CDF to map them to [0, 1] -> [-1,1]
        self.X_train = self.apply_normal_cdf(self.X_train, ['Call Put Ratio','Vix'])

        # Apply Hyperbolic tangent transformation kiils a lot of the skewness but not always produces nicely uniformally distributed features
        self.X_train = self.apply_tail_transformation(self.X_train, ['Kurt Delta', 'Skew Delta'], 'logistic')
        

        # Use the min-max scaler to map the more nicely behaved features to [-1,1]
        self.X_train = self.apply_min_max_scaler(self.X_train, ['Abnormal Volume Call', 'Abnormal Volume Put', 'ATM Call Open Interest Ratio', 'ATM Put Open Interest Ratio', 'Call Put Ratio', 'Day Of Week','IV Slope', 'Log Market Cap', 'Call Put Ratio','Vix'])

        self.X_validation = self.transform_oos_data(self.X_validation)
        self.X_test = self.transform_oos_data(self.X_test)


        df_train_validation = pd.concat([self.X_train, self.X_validation] , axis = 0)
        df_train_validation_test = pd.concat([df_train_validation, self.X_test], axis = 0)


        self.X_train_tensor = self.transform_to_tensor(self.X_train, ['Abnormal Volume Call', 'Abnormal Volume Put', 'ATM Call Open Interest Ratio', 'ATM Put Open Interest Ratio', 'Beta', 'Call Put Ratio', 'Kurt Delta', 'Skew Delta', 'Implied Move', 'Implied Vol', 'IV Slope', 'Log Market Cap', 'Vix', 'PNL Bid Ask (8)', 'Realized Move Pct (1)', 'Realized Move Pct (2)'], 'padding', train_start_date)
        self.X_validation_tensor = self.transform_to_tensor(df_train_validation, ['Abnormal Volume Call', 'Abnormal Volume Put', 'ATM Call Open Interest Ratio', 'ATM Put Open Interest Ratio', 'Beta', 'Call Put Ratio', 'Kurt Delta', 'Skew Delta', 'Implied Move', 'Implied Vol', 'IV Slope', 'Log Market Cap', 'Vix', 'PNL Bid Ask (8)', 'Realized Move Pct (1)', 'Realized Move Pct (2)'], 'padding', train_end_date )
        self.X_test_tensor = self.transform_to_tensor(df_train_validation_test, ['Abnormal Volume Call', 'Abnormal Volume Put', 'ATM Call Open Interest Ratio', 'ATM Put Open Interest Ratio', 'Beta', 'Call Put Ratio', 'Kurt Delta', 'Skew Delta', 'Implied Move', 'Implied Vol', 'IV Slope', 'Log Market Cap', 'Vix', 'PNL Bid Ask (8)', 'Realized Move Pct (1)', 'Realized Move Pct (2)'], 'padding', validation_end_date)


        #self.explore_entropy(self.X_validation, self.y_validation)

    def explore_entropy(self, X, y):
        for col in X.columns.tolist()[3:]:

            entropy = self.shannon_entropy(X[col], 20)
            mututal_information = self.calculate_mutual_information(X[col], y[self.class_column], 20)
            plt.hist(X[col], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
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
        return max(0.0, mi/h_y)


    def thin_right_tail(self, X, column_list, method_list, percentile_threshold = 95):
    
        if isinstance(method_list, str):
            method_list = [method_list] * len(column_list)
        
        X_ = X.copy()
        for i in range(len(method_list)):

            threshold_val = np.percentile(X_[column_list[i]], percentile_threshold)


            self.right_tail_dic[column_list[i]] = {'threshold': threshold_val,
                                                    'method': method_list[i]}

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

        self.normal_scaler = StandardScaler()
        self.normal_scaler_columns = column_list
        X[column_list] = self.normal_scaler.fit_transform(X[column_list])
        X[column_list] = norm.cdf(X[column_list])

        for col in column_list:
            X[col] = 2 * (X[col] - 0.5)

        return X
    
    def apply_power_scaler(self, X, column_list):
        self.power_scaler = PowerTransformer(method = 'yeo-johnson')
        self.power_scaler_columns = column_list
        X[column_list] = self.power_scaler.fit_transform(X[column_list])
        return X


    def apply_min_max_scaler(self, X, column_list):
        self.min_max_scaler = MinMaxScaler(feature_range= (-1,1))
        self.min_max_scaler_columns = column_list
        X[column_list] = self.min_max_scaler.fit_transform(X[column_list])

        return X

    def apply_normal_cdf(self, X, column_list):

        self.normal_cdf_columns = column_list
        for col in column_list:
            X[col] = norm.cdf(X[col])

        return X

    def thin_left_tail(self, X, column_list, method_list, percentile_threshold = 5):
    
        if isinstance(method_list, str):
            method_list = [method_list] * len(column_list)
        
        X_ = X.copy()
        for i in range(len(method_list)):

            threshold_val = np.percentile(X_[column_list[i]], percentile_threshold)

            mask = X_[column_list[i]] < threshold_val # We are taking the indexes of all the values which are to be changed i.e. above the threshold
                

            self.left_tail_dic[column_list[i]] = {'threshold': threshold_val,
                                                    'method': method_list[i]}


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
            self.tail_transformation_dic[col] = method
            if method == 'hyperbolic':
                X_[col] = np.tanh(X_[col])
            elif method == 'logistic':
                X_[col] = (1 / (1 + np.exp(-X_[col]) ) - 0.5) * 2

            else:
                raise ValueError(f"Unknown method {method} for tail transformation. Use 'hyperbolic' or 'logistic'.")
        
        return X_

    def apply_iqr_scaler(self, X, column_list):
        X = X.copy()

        self.robust_scaler = RobustScaler()
        self.robust_scaler_columns = column_list
        X[column_list] = self.robust_scaler.fit_transform(X[column_list])

        return X

    def transform_oos_data(self, X):

        X_ = X.copy()
        # Apply the right tail transformation
        if len(self.right_tail_dic) > 0:
            
            for key,value in self.right_tail_dic.items():
                mask = X_[key] > value['threshold']
                if value['method'] == 'log':
                    X_.loc[mask, key] = np.log(X_.loc[mask, key] + 1e-8)
                
                elif value['method'] == 'square_root':
                    X_.loc[mask, key] = np.sqrt(X_.loc[mask, key])

                elif value['method'] == 'cubic_root':
                    X_.loc[mask, key] = np.cbrt(X_.loc[mask, key])

        # Apply the left tail transformation
        if len(self.left_tail_dic) > 0:

            for key,value in self.left_tail_dic.items():
                mask = X_[key] < value['threshold']
                if value['method'] == 'log':
                    X_.loc[mask, key] = np.log(X_.loc[mask, key] + 1e-8)
                
                elif value['method'] == 'square_root':
                    X_.loc[mask, key] = np.sqrt(X_.loc[mask, key])

                elif value['method'] == 'cubic_root':
                    X_.loc[mask, key] = np.cbrt(X_.loc[mask, key])

        # Apply the Normal Scaler Transformation
        if len(self.normal_scaler_columns) > 0:
            X_[self.normal_scaler_columns] = self.normal_scaler.transform(X_[self.normal_scaler_columns])
            X_[self.normal_scaler_columns] = norm.cdf(X_[self.normal_scaler_columns])
            for col in self.normal_scaler_columns:
                X_[col] = 2 * (X_[col] - 0.5)

        # Apply the Power Scaler Transformation
        if len(self.power_scaler_columns) > 0:
            X_[self.power_scaler_columns] = self.power_scaler.transform(X_[self.power_scaler_columns])


        if len(self.robust_scaler_columns) > 0:
            X_[self.robust_scaler_columns] = self.robust_scaler.transform(X_[self.robust_scaler_columns])

        # Apply the Normal CDF as a uniform transformation
        if len(self.normal_cdf_columns) > 0:
            for col in self.normal_cdf_columns:
                X_[col] = norm.cdf(X_[col])

        
        # Apply the tail transformations on the unseen data
        if len(self.tail_transformation_dic) > 0:

            for key, value in self.tail_transformation_dic.items():
                if value == 'hyperboloc':
                    X_[key] = np.tanh(X_[key])
                if value == 'logistic':
                    X_[key] = (1 / (1 + np.exp(-X_[col]) ) - 0.5) * 2

        if len(self.min_max_scaler_columns) > 0:
            X_[self.min_max_scaler_columns] = self.min_max_scaler.transform(X_[self.min_max_scaler_columns])

        return X_

    def transform_to_tensor(self, X, column_list, na_method = 'padding', cutoff = '2016-01-01'):

        # You need to add a filter to select only after 2016

        if na_method == 'padding':
            seq_length = 9
            pad_value = 2
            cutoff = pd.Timestamp(cutoff)
            q_freq = 'Q-DEC'


            X['Date'] = pd.to_datetime(X['Date'])
            X[column_list] = X[column_list].fillna(pad_value)

            X_, auxiliary = [], []

            for ticker, g in X.groupby('Ticker', sort = False):

                g = g.sort_values('Date').reset_index(drop=True)
                g['Quarter'] = g['Date'].dt.to_period(q_freq)


                # This might be needed but likely not
                g = g.drop_duplicates('Quarter', keep = 'last').set_index('Quarter')
                full_q= pd.period_range(g.index.min(), g.index.max(), freq = q_freq)
                g = g.reindex(full_q)

                g['Ticker'] = g['Ticker'].ffill()
                g[column_list] = g[column_list].fillna(pad_value)

                feats = g[column_list].to_numpy()
                dates = g['Date']

                for i, dt in enumerate(dates):
                    if pd.isna(dt) or dt < cutoff:
                        continue

                    start = max(0, i - seq_length + 1)
                    seq = feats[start:i+1]

                    if seq.shape[0] < seq_length:
                        pad_rows = seq_length - seq.shape[0]
                        pad = np.full((pad_rows, seq.shape[1]), pad_value, dtype=feats.dtype)
                        seq = np.vstack([pad, seq])

                    
                    X_.append(seq)
                    auxiliary.append((ticker, dt, str(full_q[i])))


            X_ = np.stack(X_)

        if na_method == 'missing_column':
            seq_length = 9
            pad_value = 2

            X_ = []
            auxiliary = []

            for ticker, g in X.groupby('Ticker'):
                print(g)
                vals = g[column_list].to_numpy(copy = True)
                mask = (~np.isnan(vals)).astype(np.float32)

                vals = np.nan_to_num(vals, )
                sys.exit()

        return X_
    
    def return_tensors(self):
        return self.X_train_tensor, self.X_validation_tensor, self.X_test_tensor


    def return_X(self):
        X_train = self.X_train[self.X_train['Date'] >= self.train_start_date]

        return X_train, self.X_validation, self.X_test
    
    def return_y(self):
        y_train = self.y_train[self.y_train['Date'] >= self.train_start_date]

        return y_train, self.y_validation, self.y_test

if __name__ == '__main__':
    ScaleOptionData()