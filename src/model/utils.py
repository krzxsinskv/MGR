import matplotlib.pyplot as plt
import logging
import torch
import os
from datetime import datetime
import pandas as pd
import re











def plot_predictions(y_true, y_pred, samples, timestamp, save=True):
    logger = setup_logger()
    logger.info('Plotting true and predicted values')
    plt.plot(y_true[:samples], label='True')
    plt.plot(y_pred[:samples], label='Predicted')
    plt.legend()
    plt.title("True and predicted values")
    if save:
        save_dir = os.path.join("results", "predictions")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_predictions.png")

        plt.savefig(save_path)
        logger.info(f"Saved predicted values plot to {save_path}")
        plt.show()
    else:
        plt.show()


def plot_histogram(y_pred, bins, timestamp, save=True):
    logger = setup_logger()
    logger.info('Plotting histogram of predicted values')
    plt.hist(y_pred, bins)
    plt.title("Histogram of Predicted Appliance Power")
    plt.xlabel("Power (W)")
    plt.ylabel("Frequency")
    plt.title("Histogram of predicted values")
    if save:
        save_dir = os.path.join("results", "histograms")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}_histogram.png")

        plt.savefig(save_path)
        logger.info(f"Saved predicted values plot to {save_path}")
        plt.show()
    else:
        plt.show()


if __name__ == '__main__':
    model_path = 'models/2025-05-27_16-21_best_model.pth'
    timestamp = extract_timestamp(model_path)
    print(timestamp)
