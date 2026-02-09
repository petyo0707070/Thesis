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


import datetime as dt
import os


class ModelPipeline():
    def __init__(self, class_column = 'Profitable Trade', plot_performance = True, classification_threshold = 0.5):

        self.plot_performance = plot_performance
        self.classification_threshold = classification_threshold
        self.val_pred_probs = []

        option_scaler = ScaleOptionData(train_start_date = '2016-01-01', train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date='2024-12-31')
        self.X_train, self.X_validation, self.X_test = option_scaler.return_X()

        self.X_train = self.X_train[[col for col in self.X_train.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]
        self.X_validation = self.X_validation[[col for col in self.X_validation.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]   
        self.X_test = self.X_test[[col for col in self.X_test.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]   
   

        self.X_train.reset_index(inplace= True, drop = True)
        self.X_validation.reset_index(inplace= True, drop = True)
        self.X_test.reset_index(inplace= True, drop = True)

        self.X_train_masked = self.X_train.fillna(2)
        self.X_validation_masked = self.X_validation.fillna(2)
        self.X_test_masked = self.X_test.fillna(2)


        self.X_train_tensor, self.X_validation_tensor, self.X_test_tensor = option_scaler.return_tensors()
        self.y_train, self.y_validation, self.y_test = option_scaler.return_y()
        self.class_column = class_column

        self.X_train.to_csv(r'D:\Option Data\scaled_features\X_train_1.csv', index = False)
        self.y_train.to_csv(r'D:\Option Data\scaled_features\y_train_1.csv', index = False)


        print(f'y_train: {self.y_train.shape}, X_train: {self.X_train.shape}, X_train_tensor: {self.X_train_tensor.shape}')
        print(f'y_validation: {self.y_validation.shape}, X_train: {self.X_validation.shape}, X_validation_tensor: {self.X_validation_tensor.shape}')

        self.fit_logistic_regression()
        self.fit_xgboost()
        self.run_permutation_test(model = self.model_xgboost, X = self.X_train_masked, y = self.y_train)
        self.fit_hybrid_model()
        self.fit_ensemble(ensemble_type = 'soft_voting')

    def fit_hybrid_model(self):
        pad_value = 2
        timesteps = self.X_train_tensor.shape[1]
        n_features = self.X_train_tensor.shape[2]

        self.split_train_tensor()

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

        self.val_pred_probs.append(self.y_val_pred_ada_proba)

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

        cm = confusion_matrix(self.y_validation[self.class_column], self.y_val_pred_logistic)

        self.val_pred_probs.append(self.y_val_pred_logistic_proba)

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

if __name__ == '__main__':
    ModelPipeline(plot_performance= False, classification_threshold= 0.5)