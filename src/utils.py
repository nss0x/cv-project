"""
Utility functions for the project
"""
import torch
import numpy as np
import random
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import SEED, RESULTS_DIR

def set_seed(seed=SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    metrics = {}
    
    # Get number of unique classes
    num_classes_true = len(np.unique(y_true))
    num_classes_pred = len(np.unique(y_pred))
    num_classes = max(num_classes_true, num_classes_pred)
    
    # Use weighted average for all multiclass scenarios
    average_method = 'weighted' if num_classes > 2 else 'binary'
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['sensitivity'] = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    
    # Calculate specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] > 2:
        # Multiclass specificity
        specificity_list = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            if (tn + fp) > 0:
                specificity_list.append(tn / (tn + fp))
            else:
                specificity_list.append(0)
        metrics['specificity'] = np.mean(specificity_list)
    else:
        # Binary specificity
        tn = cm[0, 0]
        fp = cm[0, 1]
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)
        else:
            metrics['specificity'] = 0
    
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics

def save_results(experiment_name, config, metrics, model_path=None):
    """Save experiment results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(RESULTS_DIR, f'{experiment_name}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save metrics
    with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return result_dir

def plot_metrics(history, save_path):
    """Plot training history safely (handles missing metrics)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    if 'train_accuracy' in history and 'val_accuracy' in history:
        axes[0, 0].plot(history['train_accuracy'], label='Train')
        axes[0, 0].plot(history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 1].plot(history['train_loss'], label='Train')
        axes[0, 1].plot(history['val_loss'], label='Val')
        axes[0, 1].set_title('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Sensitivity (only if exists)
    if 'train_sensitivity' in history and 'val_sensitivity' in history:
        axes[1, 0].plot(history['train_sensitivity'], label='Train')
        axes[1, 0].plot(history['val_sensitivity'], label='Val')
        axes[1, 0].set_title('Sensitivity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].set_title('Sensitivity (Not Available)')
    
    # MCC (only if exists)
    if 'train_mcc' in history and 'val_mcc' in history:
        axes[1, 1].plot(history['train_mcc'], label='Train')
        axes[1, 1].plot(history['val_mcc'], label='Val')
        axes[1, 1].set_title('MCC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_title('MCC (Not Available)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_separator(text=''):
    """Print a formatted separator"""
    print('\n' + '='*60)
    if text:
        print(f'  {text}')
        print('='*60)
    print()
