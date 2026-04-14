"""
Configuration file for the Evolutionary Deep Learning Framework
"""
import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset Configuration
DATASETS = {
    'brain_tumor': {
        'kaggle_dataset': 'masoudnickparvar/brain-tumor-mri-dataset',
        'classes': ['glioma', 'meningioma', 'pituitary', 'notumor'],
        'num_classes': 4
    },
    'colorectal': {
        'kaggle_dataset': 'kmader/colorectal-histology-mnist',
        'classes': ['01_TUMOR', '02_STROMA', '03_COMPLEX', '04_LYMPHO', '05_DEBRIS', '06_MUCOSA', '07_ADIPOSE', '08_EMPTY', '09_NECROSIS'],
        'num_classes': 9
    },
    'chest_xray': {
        'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
        'classes': ['NORMAL', 'PNEUMONIA'],
        'num_classes': 2
    }
}

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'resnet18',
    'pretrained': True,
    'num_classes': 4,  # Will be updated based on dataset
    'input_size': 224,
}

# Training Configuration - Default
TRAINING_CONFIG = {
    'num_epochs': 12,
    'batch_size': 16,
    'learning_rate': 0.00005,
    'weight_decay': 1e-3,
    'optimizer': 'adam',
    'scheduler': 'cosine',
    'patience': 5,
}

# Optimization Configuration
OPTIMIZATION_CONFIG = {
    'method': 'pso',  # 'pso' or 'ga'
    'population_size': 20,
    'generations': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
}

# Hyperparameter Tuning Ranges
HYPERPARAMETER_RANGES = {
    'learning_rate': [1e-4, 1e-2],
    'batch_size': [16, 64],
    'weight_decay': [1e-6, 1e-3],
    'dropout': [0.2, 0.5],
}

# Evaluation Metrics
EVAL_METRICS = ['accuracy', 'sensitivity', 'specificity', 'mcc']

# Device Configuration
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Auto-detect, fallback to CPU
NUM_WORKERS = 4

# Random Seed
SEED = 42
