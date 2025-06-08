import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import RANDOM_SEED

def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


PLOTS_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_predictions_vs_truth(y_true, y_pred, filename="pred_vs_true.png"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    ax.set_xlabel("True Age")
    ax.set_ylabel("Predicted Age")
    ax.set_title("Predicted vs. True Age")
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved predictions vs. truth to {out_path}")

def plot_error_distribution(y_true, y_pred, filename="error_dist.png", bins=20):
    errors = np.abs(np.array(y_pred) - np.array(y_true))
    fig, ax = plt.subplots()
    ax.hist(errors, bins=bins, edgecolor='black')
    ax.set_xlabel("Absolute Error (years)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Absolute Errors")
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved error distribution to {out_path}")
