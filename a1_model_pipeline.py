from a1_scale_option_data import ScaleOptionData
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class ModelPipeline():
    def __init__(self):
        option_scaler = ScaleOptionData()
        self.X_train, self.X_validation, self.X_test = option_scaler.return_X()
        self.X_train_tensor, self.X_validation_tensor, self.X_test_tensor = option_scaler.return_tensors()
        self.y_train, self.y_validation, self.y_test = option_scaler.return_y()

        self.fit_hybrid_model()

    def fit_hybrid_model(self):
        pad_value = 2
        timesteps = self.X_train_tensor.shape[1]
        n_features = self.X_train_tensor.shape[2]

        print('here')
        self.split_train_tensor()

        pr_auc = tf.keras.metrics.AUC(curve = 'PR', name = 'pr_auc')

        model = models.Sequential([
            layers.Masking(mask_value = pad_value, input_shape = (timesteps, n_features)),
            layers.LSTM(units = 64, return_sequences = False),
            layers.Dense(1, activation = 'sigmoid')
        ])

        model.compile(optimizer = tf.keras.optimizers.Adam(1e-3),
                      loss = 'binary_crossentropy',
                      metrics = ['accuracy', pr_auc, tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall')]
                      )
        
        es = callbacks.EarlyStopping(monitor = 'val_pr_auc', patience = 50, mode ='max', restore_best_weights = True)
        ckpt = callbacks.ModelCheckpoint('best_pr_auc.keras', monitor='val_pr_auc', mode='max', save_best_only=True)

        history = model.fit(self.X_train_train_tensor, self.y_train_train,
                            validation_data = (self.X_train_validation_tensor, self.y_train_validation),
                            epochs = 200, 
                            batch_size = 128,
                            callbacks = [es, ckpt],
                            verbose = 1)

    def split_train_tensor(self, split_ratio = 0.8):
        split_idx = int(0.8 * self.X_train_tensor.shape[0])

        print(split_idx)

        self.X_train_train_tensor = self.X_train_tensor[:split_idx]
        self.X_train_validation_tensor = self.X_train_tensor[split_idx:]

        self.y_train_train = self.y_train[:split_idx]
        self.y_train_validation = self.y_train[split_idx:]

if __name__ == '__main__':
    ModelPipeline()