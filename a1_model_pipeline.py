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
                 run_logistic = True, run_xgboost = True, run_hybrid = True, run_ada = True, synthetic_data_multiplyer = 0):

        # This will take care of creating synthetic data
        if synthetic_data_multiplyer > 0:
            self.synthetic_data_multiplyer = synthetic_data_multiplyer
            self.use_synthetic_data = True
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

        option_scaler = ScaleOptionData(train_start_date = '2016-01-01', train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date='2024-12-31') # My own build class that converts the Option Data from feature matrexes into ready to be fead data for the models
        self.X_train, self.X_validation, self.X_test = option_scaler.return_X() # Get the X feature matrix for the train, validation and test

        # Drop columns which we will not use for fitting the model
        self.X_train = self.X_train[[col for col in self.X_train.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]
        self.X_validation = self.X_validation[[col for col in self.X_validation.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]   
        self.X_test = self.X_test[[col for col in self.X_test.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]   
   
        # For some reason I had to reset the index
        self.X_train.reset_index(inplace= True, drop = True)
        self.X_validation.reset_index(inplace= True, drop = True)
        self.X_test.reset_index(inplace= True, drop = True)

        # Fill NA with 2 i.e. some sort of masking if I am working with a model that can't natively handle NA such as ADA Boosted Decision Trees or a Logistic regression
        self.X_train_masked = self.X_train.fillna(2)
        self.X_validation_masked = self.X_validation.fillna(2)
        self.X_test_masked = self.X_test.fillna(2)
        self.y_train, self.y_validation, self.y_test = option_scaler.return_y() # Get the y for training, validation and test


        # Implements the generation of synthetic data that would perhaps be useful to train better generalizable models
        if self.use_synthetic_data:
            self.Xy_train = self.X_train.copy()
            self.Xy_train[self.class_column] = self.y_train[self.class_column].values

            self.X_train, self.y_train = self.generate_synthetic_data(self.Xy_train)
            print(self.X_train)
            print(self.y_train)

            sys.exit()
        
        self.X_train_tensor, self.X_validation_tensor, self.X_test_tensor = option_scaler.return_tensors() # Get tensors of  the training, validation and test data
        self.split_train_tensor() # Create a X_train_train and X_train_validation tensor


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
                            epochs = 10, 
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


        if self.plot_performance:
            self.plot_performance_over_epochs()
            plt.scatter(self.y_val_pred_nn_seq, self.y_validation[self.class_column])
            plt.title('Predicted Probabilities (NN) vs Validation Class')
            plt.show()

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
            plt.title(f'Confusion Matrix: Logistic Regression (Baseline: 41%)')
            plt.show()

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

            cm = confusion_matrix(self.y_validation[self.class_column], y_val_voting_pred)
            
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix: Voting (Baseline: 41%)')
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

    def generate_synthetic_data(self, Xy):
        from sdv.single_table  import CTGAN


        if len(Xy) < 1000:
            gen_dim = (128, 128)
            discr_dim = (128, 128)

        else:
            gen_dim = (256, 256)
            discr_dim = (256, 256)

        self.GAN = CTGAN(epochs = 300, batch_size = 64, generator_dim = gen_dim, discriminator_dim = discr_dim, verbose = True)
        self.GAN.fit(Xy, discrete_columns = self.class_column)

        synthetic_df = self.GAN.sample(int(self.synthetic_data_multiplyer * len(Xy)))

        Xy = pd.concat([Xy, synthetic_df], axis = 0)


        return Xy[[col for col in Xy.columns if col != self.class_column]], Xy[self.class_column]


if __name__ == '__main__':
    ModelPipeline(plot_performance= False, classification_threshold= 0.5, polynomial_expansion_degree= 1, synthetic_data_multiplyer= 9)