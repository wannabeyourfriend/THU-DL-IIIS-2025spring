# Full-scale Experiment

## Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import custom models
sys.path.append('.')
from mlp import MLP
from cnn import CNN
from layer2NN import TwoLayerNN

## Set random seed and device
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

## Data loading and preprocessing
def get_data_loaders(data_dir='data/custom-dataset', batch_size=32, img_size=224):
    """
    Get data loaders for training, validation and testing
    
    Args:
        data_dir: Dataset directory
        batch_size: Batch size
        img_size: Image size
    
    Returns:
        Training, validation and test data loaders
    """
    # Data preprocessing and augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

## Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, 
               model_name, epochs=30, early_stopping_patience=5):
    """
    Train the model
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Training device
        model_name: Name of the model for saving
        epochs: Number of training epochs
        early_stopping_patience: Early stopping patience
    
    Returns:
        Training history and training time
    """
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_val_acc = 0
    patience_counter = 0
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': train_loss/train_total, 'acc': 100.*train_correct/train_total})
        
        train_loss = train_loss / train_total
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': val_loss/val_total, 'acc': 100.*val_correct/val_total})
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/best_{model_name}_model.pth')
            print(f'Model saved with Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    # Calculate training time
    training_time = time.time() - start_time
    
    return history, training_time

## Evaluation function
def evaluate_model(model, test_loader, criterion, device, class_names):
    """
    Evaluate model performance
    
    Args:
        model: Model instance
        test_loader: Test data loader
        criterion: Loss function
        device: Evaluation device
        class_names: Names of the classes
    
    Returns:
        Test loss, accuracy, predictions, true labels and class-wise accuracy
    """
    model.to(device)
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    class_correct = list(0. for _ in range(len(class_names)))
    class_total = list(0. for _ in range(len(class_names)))
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics for overall accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            # Collect predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Statistics for class-wise accuracy
            correct = predicted.eq(targets).cpu().numpy()
            for i in range(inputs.size(0)):
                label = targets[i].item()
                class_correct[label] += correct[i]
                class_total[label] += 1
    
    test_loss = test_loss / test_total
    test_acc = 100. * test_correct / test_total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Calculate class-wise accuracy
    class_acc = {}
    for i in range(len(class_names)):
        acc = 100 * class_correct[i] / class_total[i]
        class_acc[class_names[i]] = acc
        print(f'Accuracy of {class_names[i]}: {acc:.2f}%')
    
    return test_loss, test_acc, all_preds, all_targets, class_acc

## Plotting functions
def plot_training_history(histories, model_names, save_path):
    """
    Plot training history curves
    
    Args:
        histories: Dictionary of training histories
        model_names: List of model names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    for model_name in model_names:
        plt.plot(histories[model_name]['train_loss'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot validation loss
    plt.subplot(2, 2, 2)
    for model_name in model_names:
        plt.plot(histories[model_name]['val_loss'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    
    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for model_name in model_names:
        plt.plot(histories[model_name]['train_acc'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(2, 2, 4)
    for model_name in model_names:
        plt.plot(histories[model_name]['val_acc'], label=model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(all_preds, all_targets, class_names, model_name, save_path):
    """
    Plot confusion matrix
    
    Args:
        all_preds: All predictions
        all_targets: All true labels
        class_names: Names of the classes
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_accuracy(class_accs, model_names, class_names, save_path):
    """
    Plot class-wise accuracy comparison
    
    Args:
        class_accs: Dictionary of class-wise accuracies
        model_names: List of model names
        class_names: Names of the classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        accuracies = [class_accs[model_name][class_name] for class_name in class_names]
        plt.bar(x + i*width, accuracies, width, label=model_name)
    
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Class-wise Accuracy Comparison')
    plt.xticks(x + width, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_performance_comparison(test_accs, training_times, model_names, save_path):
    """
    Plot performance comparison
    
    Args:
        test_accs: Dictionary of test accuracies
        training_times: Dictionary of training times
        model_names: List of model names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot test accuracy
    plt.subplot(1, 2, 1)
    plt.bar(model_names, [test_accs[model] for model in model_names])
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Comparison')
    
    # Plot training time
    plt.subplot(1, 2, 2)
    plt.bar(model_names, [training_times[model] / 60 for model in model_names])  # Convert to minutes
    plt.xlabel('Model')
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time Comparison')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

## Run full experiment
def run_full_experiment():
    # Create directories for results
    os.makedirs('results', exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    print(f'Classes: {class_names}')
    
    # Create models
    input_size = 3 * 224 * 224
    models = {
        'Layer2NN': TwoLayerNN(input_size=input_size, hidden_size=2048, num_classes=len(class_names)),
        'MLP': MLP(input_size=input_size, hidden_sizes=[2048, 1024, 512], num_classes=len(class_names)),
        'CNN': CNN(num_classes=len(class_names))
    }
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # For storing results
    histories = {}
    training_times = {}
    test_accs = {}
    class_accs = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f'\nStarting {model_name} experiment')
        
        # Optimizer and scheduler
        if model_name == 'CNN':
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # Train model
        history, training_time = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            device, model_name, epochs=30, early_stopping_patience=5
        )
        
        # Store training results
        histories[model_name] = history
        training_times[model_name] = training_time
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(f'models/best_{model_name}_model.pth'))
        
        # Evaluate model
        test_loss, test_acc, all_preds, all_targets, class_acc = evaluate_model(
            model, test_loader, criterion, device, class_names
        )
        
        # Store evaluation results
        test_accs[model_name] = test_acc
        class_accs[model_name] = class_acc
        
        # Plot confusion matrix
        plot_confusion_matrix(
            all_preds, all_targets, class_names, model_name,
            f'results/confusion_matrix_{model_name}.png'
        )
        
        # Generate classification report
        report = classification_report(all_targets, all_preds, target_names=class_names)
        with open(f'results/classification_report_{model_name}.txt', 'w') as f:
            f.write(report)
        
        # Free GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Plot training history
    plot_training_history(
        histories, list(models.keys()),
        'results/training_history.png'
    )
    
    # Plot class-wise accuracy comparison
    plot_class_accuracy(
        class_accs, list(models.keys()), class_names,
        'results/class_accuracy.png'
    )
    
    # Plot performance comparison
    plot_performance_comparison(
        test_accs, training_times, list(models.keys()),
        'results/performance_comparison.png'
    )
    
    # Save summary results
    with open('results/experiment_summary.txt', 'w') as f:
        f.write(f'Experiment Summary - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('Test Accuracy:\n')
        for model_name in models.keys():
            f.write(f'{model_name}: {test_accs[model_name]:.2f}%\n')
        
        f.write('\nTraining Time:\n')
        for model_name in models.keys():
            f.write(f'{model_name}: {training_times[model_name]/60:.2f} minutes\n')
        
        f.write('\nClass-wise Accuracy:\n')
        for model_name in models.keys():
            f.write(f'{model_name}:\n')
            for class_name in class_names:
                f.write(f'  {class_name}: {class_accs[model_name][class_name]:.2f}%\n')
    
    return histories, test_accs, class_accs, training_times

# Run the full experiment
if __name__ == '__main__':
    # Check dataset path
    data_dir = 'data/custom-dataset'
    if not os.path.exists(data_dir):
        print(f'Error: Dataset path {data_dir} does not exist!')
    else:
        # Run full experiment
        histories, test_accs, class_accs, training_times = run_full_experiment()
        
        print('Full experiment completed!')
        print('\nTest Accuracy:')
        for model_name, acc in test_accs.items():
            print(f'{model_name}: {acc:.2f}%')
        
        print('\nTraining Time:')
        for model_name, time_taken in training_times.items():
            print(f'{model_name}: {time_taken/60:.2f} minutes')