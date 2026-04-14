import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from utils import calculate_metrics
from config import DEVICE


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # ===============================
        # 🔥 ROBUST CLASS WEIGHT COMPUTATION
        # ===============================
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)

        num_classes = self.config['num_classes']

        # Count samples per class
        class_counts = np.zeros(num_classes)
        for label in all_labels:
            class_counts[int(label)] += 1

        # Avoid division by zero
        class_counts[class_counts == 0] = 1

        # Compute weights (inverse frequency)
        class_weights = len(all_labels) / (num_classes * class_counts)

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ===============================
        # OPTIMIZER
        # ===============================
        if config['optimizer'].lower() == 'adam':
            self.optimizer = Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-5)
            )
        else:
            self.optimizer = SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config.get('weight_decay', 1e-5)
            )

        # ===============================
        # SCHEDULER
        # ===============================
        if config['scheduler'].lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs']
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )

        self.best_val_accuracy = 0
        self.best_model_state = None
        self.patience = config.get('patience', 10)
        self.patience_counter = 0

    # ===============================
    # TRAIN ONE EPOCH
    # ===============================
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Train')
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.train_loader)

        return metrics

    # ===============================
    # VALIDATION
    # ===============================
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Val')
            for images, labels in pbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    # ===============================
    # TRAIN LOOP
    # ===============================
    def train(self):
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        print(f"\nTraining for {self.config['num_epochs']} epochs...")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])

            print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

            # Scheduler
            if self.config['scheduler'].lower() == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['accuracy'])

            # Save best model
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return history

    # ===============================
    # TEST
    # ===============================
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Test'):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(all_labels, all_preds)
        return metrics