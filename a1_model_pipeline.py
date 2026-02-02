from a1_scale_option_data import ScaleOptionData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import sys
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import datetime as dt
import os


class ModelPipeline():
    def __init__(self, class_column = 'Profitable Trade'):
        option_scaler = ScaleOptionData(train_start_date = '2016-01-01', train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date='2024-12-31')
        self.X_train, self.X_validation, self.X_test = option_scaler.return_X()

        self.X_train = self.X_train[[col for col in self.X_train.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]
        self.X_validation = self.X_validation[[col for col in self.X_validation.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]   
        self.X_test = self.X_test[[col for col in self.X_test.columns if col != 'Ticker' and col != 'Q-String' and col != 'Date']]   
   

        self.X_train.reset_index(inplace= True, drop = True)
        self.X_validation.reset_index(inplace= True, drop = True)
        self.X_test.reset_index(inplace= True, drop = True)


        self.X_train_tensor, self.X_validation_tensor, self.X_test_tensor = option_scaler.return_tensors()
        self.y_train, self.y_validation, self.y_test = option_scaler.return_y()
        self.class_column = class_column


        print(f'y_train: {self.y_train.shape}, X_train: {self.X_train.shape}, X_train_tensor: {self.X_train_tensor.shape}')
        print(f'y_validation: {self.y_validation.shape}, X_train: {self.X_validation.shape}, X_validation_tensor: {self.X_validation_tensor.shape}')


        self.fit_hybrid_model()

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

        es = callbacks.EarlyStopping(monitor = 'val_pr_auc', patience = 2000, mode ='max', restore_best_weights = True)
        ckpt = callbacks.ModelCheckpoint('best_precision.keras', monitor='val_precision', mode='max', save_best_only=True)
        rlrop = callbacks.ReduceLROnPlateau(monitor = 'val_pr_auc', factor = 0.1, patience = 35, min_lr = 1e-4)

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


        self.history = model.fit(self.X_train_tensor, self.y_train[self.class_column],
                            validation_data = (self.X_validation_tensor, self.y_validation[self.class_column]),
                            epochs = 100, 
                            batch_size = 32,
                            callbacks = [es, ckpt, rlrop, tb_callback],
                            verbose = 1)
        
        model.load_weights('best_precision.keras')
        self.model = model

        # Extract the embeddings from the Train, Validation and Test tensors
        self.training_embeddings = self.extract_embeddings(self.X_train_tensor)
        self.validation_embeddings = self.extract_embeddings(self.X_validation_tensor)
        self.test_embeddings = self.extract_embeddings(self.X_test_tensor)

        self.X_train_with_embedding = pd.concat([self.X_train, self.training_embeddings], axis = 1)
        self.X_validation_with_embedding = pd.concat([self.X_validation, self.validation_embeddings], axis = 1)
        self.X_test_with_embedding = pd.concat([self.X_test, self.test_embeddings], axis = 1)

        self.model_embeddings = XGBClassifier(eta = 0.01, n_estimators = 300, max_depth = 12, objective = 'binary:logistic', tree_method = 'hist', eval_metric = 'aucpr')
        self.model_embeddings.fit(self.X_train_with_embedding, self.y_train[self.class_column])

        y_val_pred_model_embeddings = self.model_embeddings.predict(self.X_validation_with_embedding)
        cm = confusion_matrix(self.y_validation[self.class_column], y_val_pred_model_embeddings)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trade", "Trade/Success"])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix: XGBoost with Embeddings\nPrecision: 55% (Baseline: 41%)')
        plt.show() 


        print(self.training_embeddings)
        print(self.training_embeddings.shape)

        self.plot_performance_over_epochs()

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
        extractor = tf.keras.Model(inputs = self.model.inputs,
                                   outputs = self.model.get_layer('embedding_layer').output)

        embeddings = extractor.predict(tensor_data)
        emb_df = pd.DataFrame(embeddings,
                              columns = [f'emb_{i}' for i in range(embeddings.shape[1])])

        return emb_df

if __name__ == '__main__':
    ModelPipeline()