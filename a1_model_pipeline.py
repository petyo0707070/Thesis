from a1_scale_option_data import ScaleOptionData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import sys
import matplotlib.pyplot as plt

import datetime as dt
import os


class ModelPipeline():
    def __init__(self, class_column = 'Profitable Trade'):
        option_scaler = ScaleOptionData(train_start_date = '2016-01-01', train_end_date = '2020-12-31', validation_end_date = '2022-12-31', test_end_date='2024-12-31')
        self.X_train, self.X_validation, self.X_test = option_scaler.return_X()
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
            layers.Masking(mask_value = pad_value, input_shape = (timesteps, n_features)),
            layers.Bidirectional(layers.LSTM(units = 64, return_sequences = False, dropout = 0.2, recurrent_dropout = 0.2)),
            #layers.GRU(units = 32, return_sequences = False, dropout = 0.2, recurrent_dropout = 0.2 ),
            layers.Dense(units = 32, activation = 'relu'),
            layers.Dropout(0.25),
            layers.Dense(units = 16, activation = 'relu'),
            layers.Dropout(0.25),

            layers.Dense(1, activation = 'sigmoid')
        ])

        model.compile(optimizer = tf.keras.optimizers.AdamW(learning_rate = 1e-2, weight_decay = 1e-3),
                      loss = 'binary_crossentropy',
                      metrics = ['accuracy', pr_auc, tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall')]
                      )
        
        es = callbacks.EarlyStopping(monitor = 'val_pr_auc', patience = 2000, mode ='max', restore_best_weights = True)
        ckpt = callbacks.ModelCheckpoint('best_pr_auc.keras', monitor='val_pr_auc', mode='max', save_best_only=True)
        rlrop = callbacks.ReduceLROnPlateau(monitor = 'val_pr_auc', factor = 0.1, patience = 150, min_lr = 1e-4)

        print(f'Tensor train_train: {self.X_train_tensor.shape}, y train_train {self.y_train.shape}')
        print(f'Tensor train_validation: {self.X_validation_tensor.shape}, y train_validation {self.y_validation.shape} popportion of sucessfull trades { round(100 * len(self.y_validation[self.y_validation[self.class_column] == True]) / len(self.y_validation), 2)}%')

        #This code is used for making a Tensorboard------------------------------------------------------------------------------------------------------------------------------------------------------------
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", "fit", run_id)

        
        tb_callback = callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,       # set to 0 to disable weight histograms (faster)
                write_graph=True,
                write_images=False,
                update_freq='epoch',    # or set an integer number of batches
                profile_batch=(10, 20)  # set to (start, stop) to profile a small window; or None to disable
            )


        #------------------------------------------------------------------------------------------------------------------------------------------------------------


        self.history = model.fit(self.X_train_tensor, self.y_train[self.class_column],
                            validation_data = (self.X_validation_tensor, self.y_validation[self.class_column]),
                            epochs = 400, 
                            batch_size = 32,
                            callbacks = [es, ckpt, rlrop, tb_callback],
                            verbose = 1)
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



if __name__ == '__main__':
    ModelPipeline()