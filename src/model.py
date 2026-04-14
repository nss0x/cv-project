"""
ResNet-18 model for medical image classification
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18(nn.Module):
    """
    ResNet-18 model for image classification
    """
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(ResNet18, self).__init__()
        
        # Load pretrained ResNet-18 using weights parameter
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Freeze early layers (conv1, bn1, layer1-2) to preserve pre-trained features
        for param in self.model.conv1.parameters():
            param.requires_grad = False
        for param in self.model.bn1.parameters():
            param.requires_grad = False
        for param in self.model.layer1.parameters():
            param.requires_grad = False
        for param in self.model.layer2.parameters():
            param.requires_grad = False
        
        # Allow layer3, layer4 to train (later layers adapt to medical images)
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        # Modify the final fully connected layer
        in_features = self.model.fc.in_features
        
        # Replace the final layer with dropout + FC
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
        # Final FC layer is trainable (of course)
        for param in self.model.fc.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def get_features(self, x):
        """Get features before classification"""
        # Get features from all layers except the final fc layer
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

class LightweightResNet18(nn.Module):
    """
    Lightweight ResNet-18 with reduced parameters
    """
    def __init__(self, num_classes=4, pretrained=False, dropout_rate=0.3):
        super(LightweightResNet18, self).__init__()
        
        # Load ResNet-18 using weights parameter
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)
        
        # Remove some layers to make it lighter
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            # Skip layer4 for lightweight version
        )
        
        self.avgpool = base_model.avgpool
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate input features for fc layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def create_model(model_name='resnet18', num_classes=4, pretrained=True, dropout_rate=0.5):
    """
    Create a model instance
    
    Args:
        model_name: Name of the model ('resnet18' or 'lightweight_resnet18')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
    
    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    elif model_name == 'lightweight_resnet18':
        return LightweightResNet18(num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
