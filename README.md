# Evolutionary Deep Learning Framework for Medical Image Classification

A comprehensive implementation of an evolutionary deep learning framework for medical image classification using ResNet-18 with PSO and Genetic Algorithm optimization.

## Project Overview

This project implements an evolutionary deep learning framework that combines:
- **ResNet-18** neural network architecture for feature extraction and classification
- **Particle Swarm Optimization (PSO)** and **Genetic Algorithm (GA)** for automatic hyperparameter tuning
- **Multi-parameter optimization** for learning rate, batch size, and dropout

## Datasets

The framework is evaluated on three publicly available medical image datasets:

1. **Brain Tumor MRI Dataset**
   - Source: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Classes: Glioma, Meningioma, Pituitary, Normal

2. **Colorectal Histology Dataset**
   - Source: [Kaggle Colorectal Histology MNIST](https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist)
   - 9 tissue type classes

3. **Chest X-Ray Pneumonia Dataset**
   - Source: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - Classes: Normal, Pneumonia

## Project Structure

```
cv-project/
├── src/
│   ├── main.py           # Main entry point
│   ├── config.py         # Configuration settings
│   ├── dataset.py        # Dataset loading and preprocessing
│   ├── model.py          # ResNet-18 model definition
│   ├── trainer.py        # Training loop
│   ├── optimizer.py      # PSO and GA optimization algorithms
│   └── utils.py          # Utility functions
├── data/                 # Datasets directory
├── models/               # Saved models directory
├── results/              # Results and logs directory
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cv-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets from Kaggle and place them in the `data/` directory:
```
data/
├── brain_tumor/
├── colorectal/
└── chest_xray/
```

## Usage

### Baseline Training

Train the model with default hyperparameters:

```bash
python src/main.py --mode baseline --dataset brain_tumor --epochs 100 --batch-size 32
```

### Optimized Training with PSO

Train with hyperparameter optimization using Particle Swarm Optimization:

```bash
python src/main.py --mode optimized --dataset brain_tumor --optimizer pso --epochs 100
```

### Optimized Training with GA

Train with hyperparameter optimization using Genetic Algorithm:

```bash
python src/main.py --mode optimized --dataset brain_tumor --optimizer ga --epochs 100
```

### Options

- `--dataset`: Dataset to use ('brain_tumor', 'colorectal', 'chest_xray')
- `--optimizer`: Optimization algorithm ('pso' or 'ga')
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--mode`: Training mode ('baseline' or 'optimized')
- `--learning-rate`: Initial learning rate (default: 0.001)

## Architecture Details

### ResNet-18

- Lightweight residual network with 18 layers
- Skip connections to mitigate vanishing gradient problem
- Pre-trained on ImageNet for transfer learning
- Adaptable to different number of output classes

### Optimization Algorithms

**Particle Swarm Optimization (PSO)**:
- Population-based metaheuristic algorithm
- Mimics social behavior of bird flocking
- Balances exploration and exploitation
- Good for continuous optimization

**Genetic Algorithm (GA)**:
- Evolutionary algorithm using selection, crossover, and mutation
- Tournament selection for parent selection
- Single-point crossover for recombination
- Gaussian mutation for variation

### Hyperparameters Tuned

1. **Learning Rate** (1e-4 to 1e-2) - Controls gradient descent step size
2. **Batch Size** (16 to 64) - Number of samples per iteration
3. **Weight Decay** (1e-6 to 1e-3) - L2 regularization coefficient
4. **Dropout** (0.2 to 0.5) - Regularization to prevent overfitting

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **Matthews Correlation Coefficient (MCC)**: Balanced metric for multiclass classification

## Results Structure

Results are saved in the `results/` directory with the following structure:

```
results/
└── <mode>_<timestamp>/
    ├── config.json              # Training configuration
    ├── metrics.json             # Evaluation metrics
    └── training_history.png     # Training curves
```

## File Descriptions

### config.py
Central configuration file containing:
- Dataset paths and class information
- Model architecture parameters
- Training hyperparameters
- Optimization settings
- Hyperparameter search ranges

### dataset.py
Handles dataset loading and preprocessing:
- Image transforms (resize, augmentation, normalization)
- Train/validation/test splitting
- DataLoader creation with appropriate batch sizes
- Support for multiple medical image datasets

### model.py
Defines neural network architectures:
- ResNet-18: Standard architecture with dropout
- LightweightResNet18: Reduced complexity version
- Model factory function for easy instantiation

### trainer.py
Implements training loop:
- Forward pass and loss calculation
- Backward propagation and optimization
- Validation and early stopping
- Metric computation (accuracy, sensitivity, specificity, MCC)

### optimizer.py
Implements evolutionary algorithms:
- ParticleSwarmOptimizer: PSO-based hyperparameter search
- GeneticAlgorithm: GA-based hyperparameter search
- Hyperparameter bounds and scaling

### utils.py
Utility functions:
- Random seed setting for reproducibility
- Metric calculation functions
- Result saving and loading
- Visualization of training history
- Console formatting

### main.py
Main application entry point:
- Command-line argument parsing
- Dataset loading and model creation
- Baseline training workflow
- Optimized training with PSO/GA
- Result saving and reporting

## Key Features

✓ Automatic hyperparameter optimization
✓ Support for multiple medical image datasets
✓ Two evolutionary algorithms (PSO and GA)
✓ Comprehensive evaluation metrics
✓ Transfer learning with pre-trained ResNet-18
✓ Data augmentation for improved generalization
✓ Early stopping to prevent overfitting
✓ Detailed logging and result visualization
✓ Modular and extensible architecture

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for faster training
2. **Batch Size**: Larger batches faster but require more memory
3. **Learning Rate**: Start with 0.001 and adjust based on convergence
4. **Early Stopping**: Set patience=15 to avoid unnecessary training
5. **Data Augmentation**: Helps improve model generalization

## Future Improvements

- [ ] Support for other architectures (VGG, DenseNet, Vision Transformer)
- [ ] Additional optimization algorithms (Ant Colony, Differential Evolution)
- [ ] Ensemble methods
- [ ] Cross-dataset evaluation
- [ ] Model interpretability analysis
- [ ] Real-time prediction interface

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization
- Holland, J. H. (1975). Adaptation in Natural and Artificial Systems

## License

This project is for educational purposes.

## Contact

For questions or issues, please contact the project team.
