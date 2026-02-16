from sklearn.linear_model import LogisticRegression
from a1_scale_option_data import ScaleOptionData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import sys
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations_with_replacement
from sklearn.preprocessing import PolynomialFeatures

import datetime as dt
import os


class ModelPipeline():
    def __init__(self, class_column = 'Profitable Trade', plot_performance = True, classification_threshold = 0.5, polynomial_expansion_degree = 1, permutation_test = False, 
                 run_logistic = True, run_xgboost = True, run_hybrid = True, run_ada = True, synthetic_data_multiplyer = 0, visualize_synthetic_data = False, synthetic_generator = 'GAN',
                 plot_correlation_matrix = False, train_start_date = '2016-01-01', train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date = '2024-12-31'):

        # Foundational parameters for the Option scaler
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        self.test_end_date = test_end_date

        # Whether to plot the correlation mattrix of the training features
        self.plot_correlation_matrix = plot_correlation_matrix

        # This will take care of creating synthetic data
        if synthetic_data_multiplyer > 0:
            self.synthetic_data_multiplyer = synthetic_data_multiplyer
            self.use_synthetic_data = True
            self.visualize_synthetic_data = visualize_synthetic_data
            self.synthetic_generator = synthetic_generator
        else:
            self.use_synthetic_data = False


        # Housekeeping to determine whether polynomial expansion will be used
        if polynomial_expansion_degree > 1:
            self.polynomial_expansion_degree = polynomial_expansion_degree
            self.use_poly_expansion = True
        else:
            self.use_poly_expansion = False


        self.run_logistic = run_logistic # Whether to fit a Logistic Regression model
        self.run_xgboost = run_xgboost #Whether to fit an XGBoost model
        self.run_hybrid = run_hybrid # Whether to fit the Hybrid Model -> Get a seq_nn and  XGBoost with embeddings
        self.run_ada = run_ada # Whether to fit an ADA Boosted Decision Tree Model
        self.class_column = class_column # Define the class we want to classify

        self.permutation_test = permutation_test # Decide whether to run a permutation test
        self.plot_performance = plot_performance # Whether we will plot Confusion matrixes per model , NN performance over epochs etc...
        self.classification_threshold = classification_threshold # What % do we need to classify as a 1
        self.val_pred_probs = []
        self.test_pred_probs = []

        option_scaler = ScaleOptionData(train_start_date = self.train_start_date, train_end_date = self.train_end_date, validation_end_date = self.validation_end_date, test_end_date=self.test_end_date) # My own build class that converts the Option Data from feature matrexes into ready to be fead data for the models
        self.X_train_, self.X_validation_, self.X_test_ = option_scaler.return_X() # Get the X feature matrix for the train, validation and test

        # For some reason I had to reset the index
        self.X_train_.reset_index(inplace= True, drop = True)
        self.X_validation_.reset_index(inplace= True, drop = True)
        self.X_test_.reset_index(inplace= True, drop = True)
        self.y_train, self.y_validation, self.y_test = option_scaler.return_y() # Get the y for training, validation and test


        # Drop columns which we will not use for fitting the model
        self.X_train = self.X_train_[[col for col in self.X_train_.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date' and col != 'Kurt Delta' and col != 'PNL Realistic (8)']]
        self.X_validation = self.X_validation_[[col for col in self.X_validation_.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date' and col != 'Kurt Delta' and col != 'PNL Realistic (8)']]   
        self.X_test = self.X_test_[[col for col in self.X_test_.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date' and col != 'Kurt Delta' and col != 'PNL Realistic (8)']]   
   

#-----------------------------------THIS OVERWRITES THE CLASS TO USE REALISTIC PNL TO SEE HOW THE MODEL IMPROVES------------------------------------------------------------------------------------
        self.y_train[self.class_column] = self.y_train['Realistic PNL'] >= 0.25
        self.y_validation[self.class_column] = self.y_validation['Realistic PNL'] >= 0.25
        self.y_test[self.class_column] = self.y_test['Realistic PNL'] >= 0.25



#-----------------------------------------------------------------------------------------------------------------------



        # Implements the generation of synthetic data that would perhaps be useful to train better generalizable models
        if self.use_synthetic_data:
            self.Xy_train = self.X_train.copy()

            if self.synthetic_generator == 'Sequential':
                self.Xy_train['Ticker'] = self.X_train_['Ticker'].values
                self.Xy_train['Seq Count'] = self.X_train_.groupby('Ticker').cumcount()# + 1

            self.Xy_train[self.class_column] = self.y_train[self.class_column].values

            self.X_train, self.y_train = self.generate_synthetic_data(self.Xy_train, generator = self.synthetic_generator)
            print(self.X_train)
            print(self.y_train)

        # Fill NA with 2 i.e. some sort of masking if I am working with a model that can't natively handle NA such as ADA Boosted Decision Trees or a Logistic regression
        self.X_train_masked = self.X_train.fillna(2)
        self.X_validation_masked = self.X_validation.fillna(2)
        self.X_test_masked = self.X_test.fillna(2)


        self.X_train_tensor, self.X_validation_tensor, self.X_test_tensor = option_scaler.return_tensors() # Get tensors of  the training, validation and test data
        self.split_train_tensor() # Create a X_train_train and X_train_validation tensor

        if self.plot_correlation_matrix:
            self.run_correlation_matrix(self.X_train)

        print(f'y_train: {self.y_train.shape}, X_train: {self.X_train.shape}, X_train_tensor: {self.X_train_tensor.shape}')
        print(f'y_validation: {self.y_validation.shape}, X_train: {self.X_validation.shape}, X_validation_tensor: {self.X_validation_tensor.shape}')



        # Implement a polynomial expansion if I decide to
        if self.use_poly_expansion:
            self.X_train_tensor = self.get_polynomial_expanded_tensor(self.X_train_tensor, self.polynomial_expansion_degree)
            self.X_validation_tensor = self.get_polynomial_expanded_tensor(self.X_validation_tensor, self.polynomial_expansion_degree)
            self.X_test_tensor = self.get_polynomial_expanded_tensor(self.X_test_tensor, self.polynomial_expansion_degree)
            self.X_train_train_tensor = self.get_polynomial_expanded_tensor(self.X_train_train_tensor, self.polynomial_expansion_degree)
            self.X_train_validation_tensor = self.get_polynomial_expanded_tensor(self.X_train_validation_tensor, self.polynomial_expansion_degree)

            self.X_train = self.get_polynomial_expanded_features(self.X_train, self.polynomial_expansion_degree)
            self.X_validation = self.get_polynomial_expanded_features(self.X_validation, self.polynomial_expansion_degree)
            self.X_test = self.get_polynomial_expanded_features(self.X_test, self.polynomial_expansion_degree)

            self.X_train_masked = self.get_polynomial_expanded_features(self.X_train_masked, self.polynomial_expansion_degree)
            self.X_validation_masked = self.get_polynomial_expanded_features(self.X_validation_masked, self.polynomial_expansion_degree)
            self.X_test_masked = self.get_polynomial_expanded_features(self.X_test_masked, self.polynomial_expansion_degree)

        if self.run_logistic:
            self.fit_logistic_regression()

        if self.run_xgboost:
            self.fit_xgboost()

        if self.permutation_test:
            self.run_permutation_test(model = self.model_xgboost, X = self.X_train, y = self.y_train)
        
        if self.run_hybrid:
            self.fit_hybrid_model()

        self.fit_ensemble(ensemble_type = 'soft_voting')

    def fit_hybrid_model(self):
        pad_value = 2
        timesteps = self.X_train_tensor.shape[1]
        n_features = self.X_train_tensor.shape[2]


        pr_auc = tf.keras.metrics.AUC(curve = 'PR', name = 'pr_auc')

        model = models.Sequential([
            layers.Input(shape=(timesteps, n_features), name = 'input_layer'),
            layers.Masking(mask_value=pad_value, name = 'masking_layer'),
            layers.Bidirectional(layers.GRU(units=32, return_sequences=False, dropout=0.2), name = 'gru_layer'),
            layers.Dense(units = 16, activation = 'relu', name = 'embedding_layer'),
            layers.Dropout(0.2),
            layers.Dense(1, activation = 'sigmoid')
        ])

        # This is here to insure that the Tensorboard Graph actually visualizes correctly
        model.build(input_shape=(None, timesteps, n_features))

        model.compile(optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-3, weight_decay = 1e-2),
                      loss = 'binary_crossentropy',
                      metrics = ['accuracy', pr_auc, tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall')]
                      )
        
        # This is also a a line of code desinged to insure that the Tensorboard Graph appears
        _ = model(tf.zeros((1, timesteps, n_features)))

        es = callbacks.EarlyStopping(monitor = 'val_precision', patience = 2000, mode ='max', restore_best_weights = True)
        ckpt = callbacks.ModelCheckpoint('best_precision.keras', monitor='val_precision', mode='max', save_best_only=True)
        rlrop = callbacks.ReduceLROnPlateau(monitor = 'val_precision', factor = 0.1, patience = 35, min_lr = 1e-4)

        print(f'Tensor train_train: {self.X_train_tensor.shape}, y train_train {self.y_train.shape}')
        print(f'Tensor train_validation: {self.X_validation_tensor.shape}, y train_validation {self.y_validation.shape} popportion of sucessfull trades { round(100 * len(self.y_validation[self.y_validation[self.class_column] == True]) / len(self.y_validation), 2)}%')

        #This code is used for making a Tensorboard------------------------------------------------------------------------------------------------------------------------------------------------------------
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", "fit", run_id)

        
        tb_callback = callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,       # set to 0 to disable weight histograms (faster)
                write_graph=True,
                write_images=True,
                update_freq='epoch',    # or set an integer number of batches
                profile_batch=(10, 20)  # set to (start, stop) to profile a small window; or None to disable
            )


        #------------------------------------------------------------------------------------------------------------------------------------------------------------


        self.history = model.fit(self.X_train_train_tensor, self.y_train_train[self.class_column],
                            validation_data = (self.X_train_validation_tensor, self.y_train_validation[self.class_column]),
                            epochs = 100, 
                            batch_size = 32,
                            callbacks = [es, ckpt, rlrop, tb_callback],
                            verbose = 1)
        
        model.load_weights('best_precision.keras')
        self.model_seq = model

        self.y_val_pred_nn_seq_proba = self.model_seq.predict(self.X_validation_tensor)
        self.y_val_pred_nn_seq = (self.y_val_pred_nn_seq_proba >= self.classification_threshold).astype(int).flatten()

        self.val_pred_probs.append(self.y_val_pred_nn_seq_proba.flatten())

        self.y_test_pred_nn_seq_proba = self.model_seq.predict(self.X_test_tensor)
        self.y_test_pred_nn_seq = (self.y_test_pred_nn_seq_proba >= self.classification_threshold).astype(int).flatten()

        self.test_pred_probs.append(self.y_test_pred_nn_seq_proba.flatten())

        # Extract the embeddings from the Train, Validation and Test tensors
        self.training_embeddings = self.extract_embeddings(self.X_train_tensor)
        self.validation_embeddings = self.extract_embeddings(self.X_validation_tensor)
        self.test_embeddings = self.extract_embeddings(self.X_test_tensor)

        self.X_train_with_embedding = pd.concat([self.X_train, self.training_embeddings], axis = 1)
        self.X_validation_with_embedding = pd.concat([self.X_validation, self.validation_embeddings], axis = 1)
        self.X_test_with_embedding = pd.concat([self.X_test, self.test_embeddings], axis = 1)

        self.model_embeddings = XGBClassifier(eta = 0.01, n_estimators = 300, max_depth = 12, objective = 'binary:logistic', tree_method = 'hist', eval_metric = 'aucpr')
        self.model_embeddings.fit(self.X_train_with_embedding, self.y_train[self.class_column])

        self.y_val_pred_model_embeddings = self.model_embeddings.predict(self.X_validation_with_embedding)
        self.y_val_pred_model_embeddings_proba = self.model_embeddings.predict_proba(self.X_validation_with_embedding)[:, 1]

        self.y_test_pred_model_embeddings = self.model_embeddings.predict(self.X_test_with_embedding)
        self.y_test_pred_model_embeddings_proba = self.model_embeddings.predict_proba(self.X_test_with_embedding)[:, 1]


        self.val_pred_probs.append(self.y_val_pred_model_embeddings_proba)
        self.test_pred_probs.append(self.y_test_pred_model_embeddings_proba)

        cm = confusion_matrix(self.y_validation[self.class_column], self.y_val_pred_model_embeddings)

        if self.plot_performance:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: XGBoost with Embeddings (Baseline: 41%)')
            plt.show() 

            y_validation_ = self.y_validation.copy()
            y_validation_['Prediction'] = self.y_val_pred_model_embeddings
            y_validation_['Prediction Probability'] = self.y_val_pred_model_embeddings_proba
            y_validation_['Residual'] = y_validation_[self.class_column] - y_validation_['Prediction Probability']
            serial_correlation_dist = y_validation_.groupby('Ticker')['Residual'].apply(self.get_acf)
            prop_ones = y_validation_.groupby('Ticker')['Prediction'].mean()
            plot_data = pd.DataFrame({'ACF': serial_correlation_dist, 'Prop_1': prop_ones}).dropna()
            weights_red = plot_data['Prop_1']
            weights_blue = 1 - plot_data['Prop_1']

            plt.hist([plot_data['ACF'], plot_data['ACF']], bins=30, stacked=True, weights=[weights_blue, weights_red], color=['skyblue', 'red'], edgecolor='black', label=['Pred 0', 'Pred 1'], alpha=0.8)
            plt.title('La Distribuzione ACF dei Residui (Analisi Panel per Ticker) il Modello con Embeddings', fontsize=14)
            plt.show()

            self.plot_performance_over_epochs()

    def fit_xgboost(self):
        self.model_xgboost = XGBClassifier(eta = 0.01, n_estimators = 300, max_depth = 12, objective = 'binary:logistic', tree_method = 'hist', eval_metric = 'aucpr')
        self.model_xgboost.fit(self.X_train, self.y_train[self.class_column])

        self.y_val_pred_xgboost = self.model_xgboost.predict(self.X_validation)
        self.y_val_pred_xgboost_proba = self.model_xgboost.predict_proba(self.X_validation)[:, 1]

        self.y_test_pred_xgboost = self.model_xgboost.predict(self.X_test)
        self.y_test_pred_xgboost_proba = self.model_xgboost.predict_proba(self.X_test)[:, 1]

        cm = confusion_matrix(self.y_validation[self.class_column], self.y_val_pred_xgboost)

        self.val_pred_probs.append(self.y_val_pred_xgboost_proba)
        self.test_pred_probs.append(self.y_test_pred_xgboost_proba)

        if self.plot_performance:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: XGBoost(Baseline: 41%)')
            plt.show() 

            y_validation_ = self.y_validation.copy()
            y_validation_['Prediction'] = self.y_val_pred_xgboost
            y_validation_['Prediction Probability'] = self.y_val_pred_xgboost_proba
            y_validation_['Residual'] = y_validation_[self.class_column] - y_validation_['Prediction Probability']
            serial_correlation_dist = y_validation_.groupby('Ticker')['Residual'].apply(self.get_acf)
            prop_ones = y_validation_.groupby('Ticker')['Prediction'].mean()
            plot_data = pd.DataFrame({'ACF': serial_correlation_dist, 'Prop_1': prop_ones}).dropna()
            weights_red = plot_data['Prop_1']
            weights_blue = 1 - plot_data['Prop_1']

            plt.hist([plot_data['ACF'], plot_data['ACF']], bins=30, stacked=True, weights=[weights_blue, weights_red], color=['skyblue', 'red'], edgecolor='black', label=['Pred 0', 'Pred 1'], alpha=0.8)
            plt.title('La Distribuzione ACF dei Residui (Analisi Panel per Ticker) il XGBoost', fontsize=14)
            plt.show()

    def split_train_tensor(self, split_ratio = 0.8):
        split_idx = int(0.8 * self.X_train_tensor.shape[0])

        self.X_train_train_tensor = self.X_train_tensor[:split_idx]
        self.X_train_validation_tensor = self.X_train_tensor[split_idx:]

        self.y_train_train = self.y_train[:split_idx]
        self.y_train_validation = self.y_train[split_idx:]

        #print(self.X_train_train_tensor)
        #print(self.y_train_train)

    def plot_performance_over_epochs(self):
        val_pr_auc_timeseries = self.history.history.get('val_pr_auc', None)
        val_precision_timeseries = self.history.history.get('val_precision', None)
        val_recall_timeseries = self.history.history.get('val_recall', None)

        epochs = range(1, len(val_pr_auc_timeseries) + 1)
        plt.plot(epochs, val_pr_auc_timeseries, label='val_pr_auc', color='tab:blue')
        plt.plot(epochs, val_precision_timeseries, label='Precision', color='tab:red')
        plt.plot(epochs, val_recall_timeseries, label='Recall', color='tab:green')
        plt.xlabel('Epoch')
        plt.ylabel('Validation PR AUC')
        plt.title('Validation PR AUC over Epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def extract_embeddings(self, tensor_data): # This function takes care of extracting the embeddings from a given tensor according to the Neural Network which we fit
        extractor = tf.keras.Model(inputs = self.model_seq.inputs,
                                   outputs = self.model_seq.get_layer('embedding_layer').output)

        embeddings = extractor.predict(tensor_data)
        emb_df = pd.DataFrame(embeddings,
                              columns = [f'emb_{i}' for i in range(embeddings.shape[1])])

        return emb_df

    def fit_ada_boost(self):

        base_estimator = DecisionTreeClassifier(max_depth= 4, criterion='entropy')
        self.model_ada_boost = AdaBoostClassifier(estimator =base_estimator, n_estimators= 200, learning_rate= 0.01)
        self.model_ada_boost.fit(self.X_train_masked, self.y_train[self.class_column])

        self.y_val_pred_ada = self.model_ada_boost.predict(self.X_validation_masked)
        self.y_val_pred_ada_proba = self.model_ada_boost.predict_proba(self.X_validation_masked)[:, 1]

        self.y_test_pred_ada = self.model_ada_boost.predict(self.X_test_masked)
        self.y_test_pred_ada_proba = self.model_ada_boost.predict_proba(self.X_test_masked)


        self.val_pred_probs.append(self.y_val_pred_ada_proba)
        self.test_pred_probs.append(self.y_test_pred_ada_proba)

        cm = confusion_matrix(self.y_validation[self.class_column], self.y_val_pred_ada)
        if self.plot_performance:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: AdaBoost (Baseline: 41%)')
            plt.show() 


    def fit_logistic_regression(self): # Fit the simplest classifier possible, a Logistic Regression
        self.model_logistic_regression = LogisticRegression(penalty= 'elasticnet', solver = 'saga', l1_ratio=0.5, max_iter = 1000)
        self.model_logistic_regression.fit(self.X_train_masked, self.y_train[self.class_column])

        self.y_val_pred_logistic = self.model_logistic_regression.predict(self.X_validation_masked)
        self.y_val_pred_logistic_proba = self.model_logistic_regression.predict_proba(self.X_validation_masked)[:, 1]

        self.y_test_pred_logistic = self.model_logistic_regression.predict(self.X_test_masked)
        self.y_test_pred_logistic_proba = self.model_logistic_regression.predict_proba(self.X_test_masked)[:, 1]

        cm = confusion_matrix(self.y_validation[self.class_column], self.y_val_pred_logistic)

        self.val_pred_probs.append(self.y_val_pred_logistic_proba)
        self.test_pred_probs.append(self.y_test_pred_logistic_proba)

        if self.plot_performance:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: Logistic Regression (Baseline: {self.y_validation[self.class_column].mean()*100:.2f}%)')
            plt.show()

#------------------------------------- EXPERIMENTAL CODE TO TRACK AUTOCORRELATION -------------------------------------
            y_validation_ = self.y_validation.copy()
            y_validation_['Prediction'] = self.y_val_pred_logistic
            y_validation_['Prediction Probability'] = self.y_val_pred_logistic_proba
            y_validation_['Residual'] = y_validation_[self.class_column] - y_validation_['Prediction Probability']
            serial_correlation_dist = y_validation_.groupby('Ticker')['Residual'].apply(self.get_acf)
            prop_ones = y_validation_.groupby('Ticker')['Prediction'].mean()
            plot_data = pd.DataFrame({'ACF': serial_correlation_dist, 'Prop_1': prop_ones}).dropna()
            weights_red = plot_data['Prop_1']
            weights_blue = 1 - plot_data['Prop_1']

            plt.hist([plot_data['ACF'], plot_data['ACF']], bins=30, stacked=True, weights=[weights_blue, weights_red], color=['skyblue', 'red'], edgecolor='black', label=['Pred 0', 'Pred 1'], alpha=0.8)
            plt.title('La Distribuzione ACF dei Residui (Analisi Panel per Ticker) la Logistic Regression', fontsize=14)
            plt.show()
#---------------------------------------------------------------------------------------------------------------

    def fit_ensemble(self, ensemble_type = 'logistic_regression'):

        if ensemble_type == 'logistic_regression':
            X_train_ensemble = np.column_stack([self.y_val_pred_model_embeddings_proba, self.y_val_pred_xgboost_proba, self.y_val_pred_nn_seq_proba])
            X_validation_ensemble = np.column_stack([self.y_test_pred_model_embeddings_proba, self.y_test_pred_xgboost_proba, self.y_test_pred_nn_seq_proba])

            self.model_ensemble = LogisticRegression()
            self.model_ensemble.fit(X_train_ensemble, self.y_validation[self.class_column])

            y_val_pred_ensemble = self.model_ensemble.predict(X_validation_ensemble)
            y_val_pred_ensemble_proba = self.model_ensemble.predict_proba(X_validation_ensemble)[:, 1]

            cm = confusion_matrix(self.y_test[self.class_column], y_val_pred_ensemble)

            if self.plot_performance:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
                disp.plot(cmap='Blues')
                plt.title(f'Confusion Matrix: Ensemble (Baseline: 41%)')
                plt.show() 

        if ensemble_type == 'soft_voting':

            matrix_pred_probas = np.array(self.val_pred_probs)
            y_val_voting_proba = np.mean(matrix_pred_probas, axis=0)

            y_val_voting_pred = (y_val_voting_proba >= self.classification_threshold).astype(int)
            y_val_voting_pred = y_val_voting_pred

            self.y_val_prediction = y_val_voting_pred
            self.y_val_prediction_proba = y_val_voting_proba

            y_validation_ = self.y_validation.copy()
            y_validation_['Prediction'] = y_val_voting_pred

            wins_bid_ask = y_validation_[(y_validation_['Bid Ask PNL'] >= 0) & (y_validation_['Prediction'] == 1)]['Bid Ask PNL']
            loses_bid_ask = y_validation_[(y_validation_['Bid Ask PNL'] < 0) & (y_validation_['Prediction'] == 1)]['Bid Ask PNL']

            wins_realistic = y_validation_[(y_validation_['Realistic PNL'] >= 0) & (y_validation_['Prediction'] == 1)]['Realistic PNL']
            loses_realistic = y_validation_[(y_validation_['Realistic PNL'] < 0) & (y_validation_['Prediction'] == 1)]['Realistic PNL']


            print(f'Bid-Ask Total wins: {round(wins_bid_ask.sum(), 2)}, Bid-Ask Total loses: {round(loses_bid_ask.sum(), 2)}, average win: {round(wins_bid_ask.mean(), 2)}, average loss: {round(loses_bid_ask.mean(), 2)}, num wins: {len(wins_bid_ask)}, num losses: {len(loses_bid_ask)}')
            print(f'Realistic Total wins: {round(wins_realistic.sum(), 2)}, Realistic Total loses: {round(loses_realistic.sum(), 2)}, average win: {round(wins_realistic.mean(), 2)}, average loss: {round(loses_realistic.mean(), 2)}, num wins: {len(wins_realistic)}, num losses: {len(loses_realistic)}')
            

            cm = confusion_matrix(self.y_validation[self.class_column], y_val_voting_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: Voting (Baseline: {self.y_validation[self.class_column].mean()*100:.2f}%)')
            plt.show() 

            plt.plot(y_validation_[y_validation_['Prediction'] == 1]['Bid Ask PNL'].reset_index(drop=True).cumsum(), label = 'Bid-Ask PNL Equity Curve')
            plt.title('Ensemble: Bid Ask PNL equity curve')
            plt.show()

            plt.plot(y_validation_[y_validation_['Prediction'] == 1]['Realistic PNL'].reset_index(drop=True).cumsum(), label = 'Realistic PNL Equity Curve')
            plt.title('Ensemble: Realistic PNL equity curve')
            plt.show()

            # Make a plot of the serial correlation of the residuals of the trades and non trades
            y_validation_['Prediction Probability'] = y_val_voting_proba
            y_validation_['Residual'] = y_validation_[self.class_column] - y_validation_['Prediction Probability']
            serial_correlation_dist = y_validation_.groupby('Ticker')['Residual'].apply(self.get_acf)
            prop_ones = y_validation_.groupby('Ticker')['Prediction'].mean()
            plot_data = pd.DataFrame({'ACF': serial_correlation_dist, 'Prop_1': prop_ones}).dropna()

            weights_red = plot_data['Prop_1']
            weights_blue = 1 - plot_data['Prop_1']

            plt.hist([plot_data['ACF'], plot_data['ACF']], bins=30, stacked=True, weights=[weights_blue, weights_red], color=['skyblue', 'red'], edgecolor='black', label=['Pred 0', 'Pred 1'],alpha=0.8)
            plt.title('La Distribuzione ACF dei Residui (Analisi Panel per Ticker)', fontsize=14)
            plt.show()

        if ensemble_type == 'grnn':
            from a1_grnn import GRNN
            
            matrix_val_probs = np.array(self.val_pred_probs).T
            matrix_test_probs = np.array(self.test_pred_probs).T


            self.grnn = GRNN(sigma = 0.1)
            self.grnn.fit(matrix_val_probs, self.y_validation[self.class_column])
            y_test_pred = self.grnn.predict(matrix_test_probs).astype(int)

            cm = confusion_matrix(self.y_test[self.class_column], y_test_pred)
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: Voting (Baseline: 41%)')
            plt.show() 

    def run_permutation_test(self, model, X, y, n_iterations = 200):
        from sklearn.base import clone
        from sklearn.metrics import precision_score, average_precision_score

        true_probs = model.predict_proba(X)[:, 1]
        true_preds = (true_probs >= self.classification_threshold).astype(int)

        true_precision = precision_score(y[self.class_column], true_preds)
        true_prauc = average_precision_score(y[self.class_column], true_probs)

        null_precision = []
        null_prauc = []

        y_shuffled = y[self.class_column].copy()

        for i in range(n_iterations):
            print(i)
            if self.plot_performance and i % 20 == 0:
                print(f"Running permutation test iteration {i}")

            np.random.shuffle(y_shuffled.values)

            temp_model = clone(model)

            temp_model.fit(X, y_shuffled)

            shuff_probs = temp_model.predict_proba(X)[:, 1]
            shuff_preds = (shuff_probs >= self.classification_threshold).astype(int)
            
            # Store scores
            null_precision.append(precision_score(y_shuffled, shuff_preds))
            null_prauc.append(average_precision_score(y_shuffled, shuff_probs))


        p_value_precision = np.sum(np.array(null_precision) >= true_precision) / n_iterations
        p_value_prauc = np.sum(np.array(null_prauc) >= true_prauc) / n_iterations   

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # Precision Plot
        ax1.hist(null_precision, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(true_precision, color='red', linestyle='--', linewidth=2, label=f'True Score: {true_precision:.3f}')
        ax1.set_title(f'Precision Distribution (p={p_value_precision:.4f})')
        ax1.set_xlabel('Precision')
        ax1.legend()
        
        # PR-AUC Plot
        ax2.hist(null_prauc, bins=20, color='salmon', edgecolor='black', alpha=0.7)
        ax2.axvline(true_prauc, color='red', linestyle='--', linewidth=2, label=f'True Score: {true_prauc:.3f}')
        ax2.set_title(f'PR-AUC Distribution (p={p_value_prauc:.4f})')
        ax2.set_xlabel('PR-AUC')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


    def get_polynomial_expanded_features(self, X, degree_ = 3):
        X = X.fillna(1e-8)

        poly = PolynomialFeatures(degree=degree_, include_bias=False)
        poly_features = poly.fit_transform(X)

        return pd.DataFrame(poly_features, index=X.index,columns=poly.get_feature_names_out(X.columns))

    def get_polynomial_expanded_tensor(self, X_tensor, degree_ = 3):

            T, N, F = X_tensor.shape
            feature_names = [f"x{j}" for j in range(F)]

            exps_list = []
            names = []
            for d in range(1, degree_ + 1):
                for combo in combinations_with_replacement(range(F), d):
                    e = np.zeros(F, dtype=int)
                    for idx in combo:
                        e[idx] += 1
                    exps_list.append(e)

                    # Build a readable name: e.g., X^2 Y
                    parts = []
                    for j, p in enumerate(e):
                        if p == 1:
                            parts.append(f"{feature_names[j]}")
                        elif p > 1:
                            parts.append(f"{feature_names[j]}^{p}")
                    names.append(" ".join(parts))

            E = np.vstack(exps_list)  # (M, F)
            M = E.shape[0]

            X2 = X_tensor.reshape(-1, F)  # (TN, F)
            TN = X2.shape[0]

            out = np.ones((TN, M), dtype=X_tensor.dtype)

            for j in range(F):
                exp_j = E[:, j]  # (M,)
                if np.any(exp_j):  # skip if all zeros for performance
                    out *= np.power(X2[:, j][:, None], exp_j[None, :])

                # Reshape back to (T, N, M)

            X_expanded = out.reshape(T, N, M)

            return X_expanded

    def generate_synthetic_data(self, Xy, generator = 'GAN'):
        from sdv.metadata import Metadata

        if generator == 'GAN':
            from sdv.single_table import CTGANSynthesizer
        
            # First get the info about the columns such as names, dtypes, rounding etc...
            Xy_metadata = Metadata.detect_from_dataframe(data = self.Xy_train, table_name = 'Xy_train')

            # Define the GAN Synthesizer
            self.synthesizer = CTGANSynthesizer(metadata = Xy_metadata, epochs = 300, batch_size = 100, verbose = True) # Keep in mind default batch size is 500 accordingg to the documentation
            self.synthesizer.fit(Xy) # Fit the GAN to the actual data 

            synthetic_df = self.synthesizer.sample(num_rows = int(self.synthetic_data_multiplyer * len(Xy))) # Get the synthetic data

            # Those here are some diagnostic checks to insure that everything is working as expected with the GAN
            if self.visualize_synthetic_data:
                from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot

                diagnostic_report = run_diagnostic(real_data=Xy, synthetic_data=synthetic_df, metadata = Xy_metadata) # This checks that the columns are generated according to the correct datatype etc... should be 100%
                quality_report = evaluate_quality(real_data=Xy, synthetic_data=synthetic_df, metadata = Xy_metadata) # This checks how well the synthetic data resembles the real data on a similarity score between 0% and 100%, here the check is for patterns etc... 80% is good enough

                for col in self.Xy_train.columns.tolist():
                    fig = get_column_plot(real_data=Xy, synthetic_data=synthetic_df, metadata = Xy_metadata, column_name=col)
                    fig.show()

            Xy = pd.concat([Xy, synthetic_df], axis = 0)


            return Xy[[col for col in self.X_train.columns]], Xy[[self.class_column]]


        if generator == 'Sequential':
            from sdv.sequential import PARSynthesizer
            # First get the info about the columns such as names, dtypes, rounding etc...
            Xy_metadata = Metadata.detect_from_dataframe(data = self.Xy_train, table_name = 'Xy_train')            
            Xy_metadata.update_column(column_name='Ticker', sdtype='id')
            Xy_metadata.set_sequence_key('Ticker') # Set the Sequence key i.e. which company it is
            Xy_metadata.set_sequence_index('Seq Count') # Set the sequence index i.e. which earnings is it in sequence

            self.synthesizer = PARSynthesizer(metadata = Xy_metadata, epochs = 128, enforce_min_max_values = True, enforce_rounding = True, verbose = True) # This is the sequential model synthesizer PARSynthesizer
            self.synthesizer.fit(Xy)

            synthetic_df = self.synthesizer.sample(num_sequences = int(self.synthetic_data_multiplyer * self.X_train_['Ticker'].nunique()) )

            Xy = pd.concat([Xy, synthetic_df], axis = 0)

            return Xy[[col for col in self.X_train.columns]], Xy[[self.class_column]]            


    def run_correlation_matrix(self, X):
        import seaborn as sns

        corr_matrix = X.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title('Feature Correlation Matrix')
        plt.show()

    def get_acf(self, x):
        # Calcoliamo la correlazione al lag 1 (trimestre precedente)
        return x.autocorr(lag=1)

    def return_predictions_validation(self):
        return {
            'y_val_pred_logistic': self.y_val_pred_logistic if hasattr(self, 'y_val_pred_logistic') else None,
            'y_val_pred_logistic_proba': self.y_val_pred_logistic_proba if hasattr(self, 'y_val_pred_logistic_proba') else None,
            'y_val_pred_xgboost': self.y_val_pred_xgboost if hasattr(self, 'y_val_pred_xgboost') else None,
            'y_val_pred_xgboost_proba': self.y_val_pred_xgboost_proba if hasattr(self, 'y_val_pred_xgboost_proba') else None,
            'y_val_pred_nn_seq': self.y_val_pred_nn_seq if hasattr(self, 'y_val_pred_nn_seq') else None,
            'y_val_pred_nn_seq_proba': self.y_val_pred_nn_seq_proba if hasattr(self, 'y_val_pred_nn_seq_proba') else None,
            'y_val_pred_model_embeddings': self.y_val_pred_model_embeddings if hasattr(self, 'y_val_pred_model_embeddings') else None,
            'y_val_pred_model_embeddings_proba': self.y_val_pred_model_embeddings_proba if hasattr(self, 'y_val_pred_model_embeddings_proba') else None,
            'y_val_pred_ada': self.y_val_pred_ada if hasattr(self, 'y_val_pred_ada') else None,
            'y_val_pred_ada_proba': self.y_val_pred_ada_proba if hasattr(self, 'y_val_pred_ada_proba') else None,
            'y_val_prediction': self.y_val_prediction if hasattr(self, 'y_val_prediction') else None,
            'y_val_prediction_proba': self.y_val_prediction_proba if hasattr(self, 'y_val_prediction_proba') else None
        }
    
    def return_predictions_test(self):
        return {
            'y_test_pred_logistic': self.y_test_pred_logistic if hasattr(self, 'y_test_pred_logistic') else None,
            'y_test_pred_logistic_proba': self.y_test_pred_logistic_proba if hasattr(self, 'y_test_pred_logistic_proba') else None,
            'y_test_pred_xgboost': self.y_test_pred_xgboost if hasattr(self, 'y_test_pred_xgboost') else None,
            'y_test_pred_xgboost_proba': self.y_test_pred_xgboost_proba if hasattr(self, 'y_test_pred_xgboost_proba') else None,
            'y_test_pred_nn_seq': self.y_test_pred_nn_seq if hasattr(self, 'y_test_pred_nn_seq') else None,
            'y_test_pred_nn_seq_proba': self.y_test_pred_nn_seq_proba if hasattr(self, 'y_test_pred_nn_seq_proba') else None,
            'y_test_pred_model_embeddings': self.y_test_pred_model_embeddings if hasattr(self, 'y_test_pred_model_embeddings') else None,
            'y_test_pred_model_embeddings_proba': self.y_test_pred_model_embeddings_proba if hasattr(self, 'y_test_pred_model_embeddings_proba') else None,
            'y_test_pred_ada': self.y_test_pred_ada if hasattr(self, 'y_test_pred_ada') else None,
            'y_test_pred_ada_proba': self.y_test_pred_ada_proba if hasattr(self, 'y_test_pred_ada_proba') else None,
            'y_test_prediction': self.y_test_prediction if hasattr(self, 'y_test_prediction') else None,
            'y_test_prediction_proba': self.y_test_prediction_proba if hasattr(self, 'y_test_prediction_proba') else None
        }



if __name__ == '__main__':
    ModelPipeline(plot_correlation_matrix= False, plot_performance= True, class_column = 'Implied > Realized',classification_threshold= 0.5, polynomial_expansion_degree= 1, synthetic_data_multiplyer= 0, 
                  visualize_synthetic_data= False, run_hybrid= True, synthetic_generator= 'Sequential')