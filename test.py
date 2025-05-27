import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report

from train import PetExpressionClassifier
from models.DenseNet121 import DenseNet121
from models.ResNet import ResNet50
from models.VGG import VGG16
from models.MobileNet import MobileNetV1
from models.InceptionV3 import InceptionV3

def load_checkpoint(checkpoint_path, model_class, num_classes):
    """Load a checkpoint and return the model."""
    model = model_class(num_classes=num_classes)
    classifier = PetExpressionClassifier(model, num_classes=num_classes)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['state_dict'])
    return classifier

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data and return predictions and targets."""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            all_preds.append(y_pred)
            all_targets.append(y)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    return all_preds, all_targets

def plot_confusion_matrix(cm, class_names, model_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Test Confusion Matrix - {model_name}")
    plt.savefig(save_path)
    plt.close()

def plot_metrics(metrics_df, model_name, save_path):
    """Plot all metrics from the evaluation."""
    plt.figure(figsize=(18, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    num_metrics = len(metrics)
    cols = 2
    rows = (num_metrics + 1) // 2
    
    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i + 1)
        plt.bar(range(len(metrics_df)), metrics_df[metric])
        plt.title(f'Class-wise {metric.capitalize()}')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.xticks(range(len(metrics_df)), metrics_df['class'], rotation=45)
    
    plt.suptitle(f"Test Metrics - {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()

def get_dataset(test_folder_path, is_inception=False):
    if is_inception:
        IMG_SIZE = 299
    else:
        IMG_SIZE = 224
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    test_dataset = ImageFolder(test_folder_path, transform=transform)

    return test_dataset

def main(args):
    # Model registry
    models_registry = {
        "DenseNet121": DenseNet121,
        "ResNet50": ResNet50,
        "VGG16": VGG16,
        "MobileNetV1": MobileNetV1,
        "InceptionV3": InceptionV3
    }
    
    # Determine if InceptionV3 is used
    is_inception = (args.model == "InceptionV3")
    
    # Load test dataset
    test_folder_path = os.path.join(args.data_dir, "test")
    test_dataset = get_dataset(test_folder_path, is_inception)
    
    num_classes = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"Test dataset loaded: {num_classes} classes found: {class_names}")
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           num_workers=args.num_workers)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpus > 0 else "cpu")
    
    # Create output directory for test results
    test_results_dir = os.path.join(args.root_dir, "test_results", args.model)
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Construct checkpoint path from argument
    if not args.checkpoint_file:
        print("Error: --checkpoint_file argument is required.")
        return

    checkpoint_path = args.checkpoint_file

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    print(f"\nEvaluating checkpoint: {args.checkpoint_file}")
    
    # Load model from checkpoint
    model = load_checkpoint(checkpoint_path, models_registry[args.model], num_classes)
    
    # Evaluate model
    all_preds, all_targets = evaluate_model(model, test_loader, device)
    
    # Convert predictions to class labels
    pred_labels = torch.argmax(all_preds, dim=1).cpu().numpy()
    true_labels = all_targets.cpu().numpy()
    
    # Calculate metrics
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, 
                                    target_names=class_names, 
                                    output_dict=True)
    
    # Define base filename for outputs
    base_filename = args.checkpoint_file.split('.')[0]

    # Save confusion matrix
    cm_save_path = os.path.join(test_results_dir, 
                                f"{base_filename}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, args.model, cm_save_path)
    
    # Save classification report
    report_save_path = os.path.join(test_results_dir, 
                                    f"{base_filename}_classification_report.txt")
    with open(report_save_path, 'w') as f:
        f.write(classification_report(true_labels, pred_labels, 
                                    target_names=class_names))
    
    # Prepare data for plot_metrics
    metrics_list = []
    for class_name in class_names:
        metrics_list.append({
            'class': class_name,
            'accuracy': report['accuracy'], # Overall accuracy, same for all classes in this context
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1': report[class_name]['f1-score']
        })
    metrics_df = pd.DataFrame(metrics_list)
    
    # Create and save class-wise metrics plot
    metrics_plot_save_path = os.path.join(test_results_dir, f"{base_filename}_metrics.png")
    plot_metrics(metrics_df, args.model, metrics_plot_save_path)
    
    print(f"\nTest results saved to: {test_results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DenseNet121")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--root_dir", type=str, default=".", 
                       help="Root directory for all outputs (logs, checkpoints, reports)")
    parser.add_argument("--data_dir", type=str, 
                       default=os.path.join("pets_expression", "Master Folder"),
                       help="Directory for input dataset")
    parser.add_argument("--checkpoint_file", type=str, help="Directory for input checkpoint file")
    
    args = parser.parse_args()
    main(args) 