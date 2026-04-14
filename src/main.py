"""
Main entry point for the Evolutionary Deep Learning Framework
"""
import sys
import os
import argparse
import json
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    TRAINING_CONFIG, OPTIMIZATION_CONFIG, DATASETS, 
    MODEL_CONFIG, DEVICE, SEED
)
from utils import set_seed, print_separator, save_results, plot_metrics
from dataset import create_dataloader
from model import create_model, count_parameters
from trainer import Trainer
from optimizer import ParticleSwarmOptimizer, GeneticAlgorithm

def create_objective_function(model_template, train_loader, val_loader, num_classes):
    """Create objective function for optimization"""
    
    def objective_function(config):
        """
        Evaluate a configuration
        
        Args:
            config: Hyperparameter configuration
        
        Returns:
            Validation accuracy
        """
        try:
            # Create model
            model = create_model(
                'resnet18',
                num_classes=num_classes,
                pretrained=True,
                dropout_rate=0.5
            )
            
            # Create trainer
            trainer = Trainer(model, train_loader, val_loader, config)
            
            # Train
            _ = trainer.train()
            
            # Get best validation accuracy
            accuracy = trainer.best_val_accuracy
            
            # Clean up
            del model
            del trainer
            torch.cuda.empty_cache()
            
            return accuracy
        
        except Exception as e:
            print(f"Error evaluating config: {e}")
            return 0
    
    return objective_function

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evolutionary Deep Learning Framework')
    parser.add_argument('--dataset', type=str, default='brain_tumor',
                        choices=['brain_tumor', 'colorectal', 'chest_xray'],
                        help='Dataset to use')
    parser.add_argument('--optimizer', type=str, default='pso',
                        choices=['pso', 'ga'],
                        help='Optimization algorithm')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'optimized'],
                        help='Training mode')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(SEED)
    
    print_separator("EVOLUTIONARY DEEP LEARNING FRAMEWORK")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Device: {DEVICE}")
    print_separator()
    
    # Load dataset
    print("Loading dataset...")
    dataset = create_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        image_size=224
    )
    
    if dataset is None:
        print("Failed to load dataset!")
        return
    
    train_loader, val_loader, test_loader = dataset.get_loaders()
    num_classes = dataset.get_num_classes()
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {dataset.get_class_names()}")
    
    # Create model
    print("\nCreating model...")
    model = create_model('resnet18', num_classes=num_classes, pretrained=True, dropout_rate=0.5)
    print(f"Model: ResNet-18")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    if args.mode == 'baseline':
        # Baseline training with default configuration
        print_separator("BASELINE TRAINING")
        
        config = TRAINING_CONFIG.copy()
        config['num_epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['learning_rate'] = args.learning_rate
        config['num_classes'] = num_classes
        
        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train()
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save results
        print("\nSaving results...")
        result_dir = save_results('baseline', config, test_metrics)
        
        # Plot metrics
        plot_metrics(history, os.path.join(result_dir, 'training_history.png'))
        
        print(f"\nResults saved to: {result_dir}")
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    else:  # optimized mode
        print_separator("HYPERPARAMETER OPTIMIZATION")
        
        # Optimize hyperparameters
        opt_config = OPTIMIZATION_CONFIG.copy()
        
        if args.optimizer == 'pso':
            optimizer = ParticleSwarmOptimizer(
                population_size=opt_config['population_size'],
                generations=opt_config['generations']
            )
        else:
            optimizer = GeneticAlgorithm(
                population_size=opt_config['population_size'],
                generations=opt_config['generations'],
                mutation_rate=opt_config['mutation_rate'],
                crossover_rate=opt_config['crossover_rate']
            )
        
        # Create objective function
        objective_fn = create_objective_function(model, train_loader, val_loader, num_classes)
        
        # Run optimization
        best_config, best_score = optimizer.optimize(objective_fn)
        
        print_separator("OPTIMAL HYPERPARAMETERS")
        for param, value in best_config.items():
            print(f"  {param}: {value}")
        print(f"  Best Validation Accuracy: {best_score:.4f}")
        print_separator()
        
        # Train final model with best configuration
        print("Training final model with optimal hyperparameters...")
        
        config = TRAINING_CONFIG.copy()
        config.update(best_config)
        config['num_epochs'] = args.epochs
        config['num_classes'] = num_classes
        
        # Create new model for final training
        model = create_model('resnet18', num_classes=num_classes, pretrained=True)
        trainer = Trainer(model, train_loader, val_loader, config)
        history = trainer.train()
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save results
        print("\nSaving results...")
        result_dir = save_results('optimized', config, test_metrics)
        
        # Plot metrics
        plot_metrics(history, os.path.join(result_dir, 'training_history.png'))
        
        print(f"\nResults saved to: {result_dir}")
        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == '__main__':
    main()
