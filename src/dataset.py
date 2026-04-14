"""
Dataset loading and preprocessing
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import os
from config import DATA_DIR, DATASETS, SEED

class MedicalImageDataset:
    """Class to handle medical image datasets"""
    
    def __init__(self, dataset_name, image_size=224, batch_size=32):
        """
        Initialize dataset
        
        Args:
            dataset_name: Name of dataset ('brain_tumor', 'colorectal', or 'chest_xray')
            image_size: Size to resize images to
            batch_size: Batch size for DataLoader
        """
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_config = DATASETS[dataset_name]
        
        # Define transforms - AGGRESSIVE AUGMENTATION
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def load_dataset(self):
        """Load dataset from local directory or download"""
        dataset_path = os.path.join(DATA_DIR, self.dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            print("Please download the dataset manually from Kaggle:")
            print(f"Dataset: {self.dataset_config['kaggle_dataset']}")
            print(f"and place it in: {dataset_path}")
            return False
        
        # Load training dataset
        try:
            full_dataset = datasets.ImageFolder(
                dataset_path,
                transform=self.train_transform
            )
            print(f"Loaded {len(full_dataset)} images")
            
            # Split into train, val, test
            train_size = int(0.7 * len(full_dataset))
            val_size = int(0.15 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(SEED)
            )
            
            # Update transforms for val and test
            val_dataset.dataset.transform = self.val_transforms
            test_dataset.dataset.transform = self.val_transforms
            
            # Create weighted sampler for balanced training
            targets = np.array([label for _, label in train_dataset])
            class_counts = np.bincount(targets)
            class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            # Create data loaders with weighted sampler for train
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=2,
                pin_memory=True
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def get_loaders(self):
        """Get data loaders"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_num_classes(self):
        """Get number of classes"""
        return self.dataset_config['num_classes']
    
    def get_class_names(self):
        """Get class names"""
        return self.dataset_config['classes']

def create_dataloader(dataset_name, batch_size=32, image_size=224):
    """
    Create dataset and loaders
    
    Args:
        dataset_name: Name of dataset
        batch_size: Batch size
        image_size: Image size
    
    Returns:
        Dataset object
    """
    dataset = MedicalImageDataset(dataset_name, image_size=image_size, batch_size=batch_size)
    if dataset.load_dataset():
        return dataset
    return None
