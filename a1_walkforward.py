import math
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from a1_model_pipeline import ModelPipeline


class WalkForward():  # This is a Walk Forward class that does the Level I Walkforward, here_everything is counted in quarters
    def __init__(self, train_start='2016-01-01', train_end='2020-12-31', train_size=4, validation_size=1, step_size=1):
        self.train_start = pd.to_datetime(train_start)  # Get the date from which we start the walkforward
        self.train_end = pd.to_datetime(train_end)  # Get the date on which we end the walkforward

        self.train_size = train_size  # On how many quarters to train
        self.validation_size = validation_size  # On how many quarters to test
        self.step_size = step_size  # By how many quarters to move the sliding window

        self.generate_folds()

    # The first function which generates all the folds for the Walkforward
    def generate_folds(self):
        # Initialize the first run
        current_train_start = self.train_start
        current_train_end = self.train_start + pd.offsets.QuarterEnd(
            self.train_size)  # We subtract one day to get the last day of the quarter

        # This will hold the results the Level I OoS performance
        self.oos_results_lv1 = []
        self.oos_results_lv1.append(0)  # Add 0 as the first element

        fold_index = 1
        # This loops over all the folds
        while current_train_end + pd.offsets.QuarterEnd(self.validation_size) <= self.train_end:
            val_start = current_train_end + pd.Timedelta(
                days=1)  # The validation starts the day after the training ends
            val_end = val_start + pd.offsets.QuarterEnd(self.validation_size)  #

            print(
                f'Fold {fold_index}: Train Start = {current_train_start}, Train End = {current_train_end}, Val Start = {val_start}, Val End = {val_end}')

            pipeline = ModelPipeline(train_start_date=current_train_start.strftime('%Y-%m-%d'),
                                     train_end_date=current_train_end.strftime('%Y-%m-%d'),
                                     validation_end_date=val_end.strftime('%Y-%m-%d'), test_end_date='2024-12-31',
                                     plot_performance=False, plot_correlation_matrix=False, run_logistic=True,
                                     run_xgboost=True, run_hybrid=False, run_ada=True,
                                     synthetic_data_multiplyer=0, visualize_synthetic_data=False,
                                     synthetic_generator='GAN', no_plotting=True, classification_threshold = 0.70, polynomial_expansion_degree= 1)

            oos_results = pipeline.return_predictions_validation()  # Get the LV1 OOS perforamance of the fold
            self.oos_results_lv1.extend(oos_results['y_val_returns_realistic'])
            fold_index += 1
            current_train_start = current_train_start + pd.offsets.QuarterEnd(self.step_size)
            current_train_end = current_train_end + pd.offsets.QuarterEnd(self.step_size)

        # Plot the lv1 OOS performance
        cleaned_list = [x for x in self.oos_results_lv1 if not (isinstance(x, float) and math.isnan(
            x))]  # Drop the NA values as for some reason they appear and destroy plotting
        plt.plot(np.cumsum(cleaned_list))
        plt.title('Level I Walkforward Perforamance % of Premium Sold')
        plt.ylabel('% of Premium Sold')
        plt.show()


if __name__ == '__main__':
    WalkForward(train_start='2016-01-01', train_end='2020-12-31', train_size=4, validation_size=1, step_size=1)