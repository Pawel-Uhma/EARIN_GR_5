import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import RANDOM_SEED

# set all random seeds to make results reproducible
def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

# make sure the folder for saving plots exists
PLOTS_DIR = os.path.join(os.getcwd(), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# scatter plot showing predicted ages vs actual ones
def plot_predictions_vs_truth(y_true, y_pred, filename='pred_vs_true.png'):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)  # diagonal line for perfect predictions
    ax.set_xlabel('True Age')
    ax.set_ylabel('Predicted Age')
    ax.set_title('Predicted vs. True Age')
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[PLOT] Saved scatter plot to {path}")

# histogram showing how far off the predictions were
def plot_error_distribution(y_true, y_pred, filename='error_dist.png', bins=20):
    errors = np.abs(np.array(y_pred) - np.array(y_true))
    fig, ax = plt.subplots()
    ax.hist(errors, bins=bins, edgecolor='black')
    ax.set_xlabel('Absolute Error (years)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Absolute Errors')
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[PLOT] Saved error distribution to {path}")

# line plot showing training and validation loss over time
def plot_train_val_loss(train_losses, val_losses, filename='loss_curve.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, 'o-', label='Train Loss')
    ax.plot(epochs, val_losses, 's-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[PLOT] Saved loss curve to {path}")

# shows the difference between prediction and true value for each point
def plot_residuals_vs_true(y_true, y_pred, filename='residuals_vs_true.png'):
    residuals = np.array(y_pred) - np.array(y_true)
    fig, ax = plt.subplots()
    ax.scatter(y_true, residuals, alpha=0.6, edgecolor='k')
    ax.axhline(0, color='r', linestyle='--')  # horizontal line at zero
    ax.set_xlabel('True Age')
    ax.set_ylabel('Residual (Pred - True)')
    ax.set_title('Residuals vs. True Age')
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[PLOT] Saved residuals plot to {path}")

# histogram showing how often each true age appears in the dataset
def plot_true_age_distribution(y_true, filename='true_age_dist.png', bins=20):
    fig, ax = plt.subplots()
    ax.hist(y_true, bins=bins, edgecolor='black')
    ax.set_xlabel('True Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of True Ages')
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[PLOT] Saved true age distribution to {path}")

# boxplot showing how error changes across different age groups
def plot_error_by_age_bin(y_true, y_pred, bins=10, filename='error_by_age_bin.png'):
    ages = np.array(y_true)
    errors = np.abs(np.array(y_pred) - ages)
    bin_edges = np.linspace(ages.min(), ages.max(), bins + 1)
    bin_indices = np.digitize(ages, bin_edges) - 1
    data = [errors[bin_indices == i] for i in range(bins)]
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(bins)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xlabel('Age Bins')
    ax.set_ylabel('Absolute Error (years)')
    ax.set_title('Error Distribution by Age Bin')
    plt.xticks(rotation=45)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"[PLOT] Saved error-by-age-bin boxplot to {path}")
