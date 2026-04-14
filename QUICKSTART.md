# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Datasets

Download the datasets from Kaggle and organize them as follows:

```
data/
├── brain_tumor/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
├── colorectal/
│   ├── 01_TUMOR/
│   ├── 02_STROMA/
│   ├── ... (other classes)
│   └── 09_NECROSIS/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## Step 3: Run Baseline Training

```bash
cd src
python main.py --mode baseline --dataset brain_tumor --epochs 50 --batch-size 32
```

## Step 4: Run Optimized Training

Using PSO:
```bash
python main.py --mode optimized --dataset brain_tumor --optimizer pso --epochs 50
```

Using GA:
```bash
python main.py --mode optimized --dataset brain_tumor --optimizer ga --epochs 50
```

## Step 5: Check Results

Results will be saved in the `results/` directory with:
- `config.json`: Training configuration used
- `metrics.json`: Final evaluation metrics
- `training_history.png`: Training curves visualization

## Common Commands

### Train on Different Datasets
```bash
# Colorectal dataset
python main.py --dataset colorectal --mode baseline --epochs 100

# Chest X-Ray dataset
python main.py --dataset chest_xray --mode baseline --epochs 100
```

### Advanced Options
```bash
# Custom learning rate and batch size
python main.py --dataset brain_tumor --epochs 100 --batch-size 64 --learning-rate 0.0005

# Longer optimization
python main.py --mode optimized --dataset brain_tumor --optimizer pso --epochs 200
```

## Expected Results

The framework typically achieves:
- **Brain Tumor MRI**: 90-95% accuracy
- **Colorectal Histology**: 85-92% accuracy
- **Chest X-Ray**: 92-98% accuracy

Results vary based on hyperparameters and optimization.

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch-size 16`
- Use fewer epochs: `--epochs 30`

### Slow Training
- Ensure GPU is available
- Check CUDA installation
- Reduce population size in optimization

### Dataset Not Found
- Verify dataset directory structure
- Check file names match configuration
- Ensure proper permissions for directory

## Next Steps

1. Experiment with different datasets
2. Tune optimization parameters
3. Try different hyperparameter ranges
4. Implement ensemble methods
5. Add model interpretability analysis
