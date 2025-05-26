from lightning import Trainer
from lightning import LightningModule
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
import torch
import os
import shutil
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from models.DenseNet121 import DenseNet121
from models.ResNet import ResNet50
from models.VGG import VGG16
from models.MobileNet import MobileNetV1
from models.InceptionV3 import InceptionV3

import argparse
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

class PetExpressionClassifier(LightningModule):
    def __init__(self, model, lr=0.001, num_classes=4):
        super(PetExpressionClassifier, self).__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes

        # metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.val_precision_metric = torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_recall_metric = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.val_f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        # For final confusion matrix
        self.all_val_preds = []
        self.all_val_targets = []

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_accuracy(y_pred, y)
        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_accuracy(y_pred, y)
        self.log("val_accuracy", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_precision_metric(y_pred, y)
        self.log("val_precision", self.val_precision_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_recall_metric(y_pred, y)
        self.log("val_recall", self.val_recall_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_f1_metric(y_pred, y)
        self.log("val_f1", self.val_f1_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.all_val_preds.append(y_pred)
        self.all_val_targets.append(y)

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class ReportMetrics(Callback):
    def __init__(self, class_names, model_name, report_subdir="custom_reports"): 
        super().__init__()
        self.class_names = class_names
        self.model_name = model_name
        self.report_subdir_name = report_subdir
        # No os.makedirs here, will be done in on_train_end relative to trainer.logger.log_dir

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):            
        print(f"\nEpoch {trainer.current_epoch} Validation Metrics ({self.model_name}):")
        metrics_to_report = {
            "Val Loss": "val_loss",
            "Val Acc": "val_accuracy",
            "Val Precision": "val_precision",
            "Val Recall": "val_recall",
            "Val F1-Score": "val_f1"
        }
        for name, key in metrics_to_report.items():
            if key in trainer.callback_metrics:
                try:
                    metric_val = trainer.callback_metrics[key].item() # .item() to get scalar
                    print(f"  {name}: {metric_val:.4f}")
                except AttributeError: # Metric might not be a tensor (e.g., if manually logged as float)
                     metric_val = trainer.callback_metrics[key]
                     print(f"  {name}: {metric_val:.4f} (raw value)")
            else:
                print(f"  {name}: N/A (key '{key}' not found in callback_metrics for epoch {trainer.current_epoch})")
        print("-" * 30)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        print(f"\nEpoch {trainer.current_epoch} Training Metrics ({self.model_name}):")
        metrics_to_report = {
            "Train Loss": "train_loss",
            "Train Acc": "train_accuracy"
        }
        for name, key in metrics_to_report.items():
            if key in trainer.callback_metrics:
                try:
                    metric_val = trainer.callback_metrics[key].item()
                    print(f"  {name}: {metric_val:.4f}")
                except AttributeError:
                     metric_val = trainer.callback_metrics[key]
                     print(f"  {name}: {metric_val:.4f} (raw value)")
            else:
                print(f"  {name}: N/A (key '{key}' not found in callback_metrics for epoch {trainer.current_epoch})")
        print("-" * 30)

    def on_train_end(self, trainer: Trainer, pl_module: PetExpressionClassifier) -> None:
        if not trainer.is_global_zero: # Ensure reports are generated only on the main process
            return

        print(f"\nTraining Ended for {self.model_name}. Generating final reports...")

        if not trainer.loggers:
            print("No loggers found in trainer. Cannot save reports.")
            return
        
        versioned_log_dir = trainer.loggers[0].log_dir 
        run_version_str = str(trainer.loggers[0].version)

        report_dir_for_run = os.path.join(versioned_log_dir, self.report_subdir_name)
        os.makedirs(report_dir_for_run, exist_ok=True)        
        print(f"Saving reports to: {report_dir_for_run}")

        if hasattr(pl_module, 'all_val_preds') and hasattr(pl_module, 'all_val_targets') and \
           pl_module.all_val_preds and pl_module.all_val_targets:
            
            all_preds_batches = torch.cat(pl_module.all_val_preds).cpu() # These are raw logits/outputs
            all_targets_batches = torch.cat(pl_module.all_val_targets).cpu()

            final_preds_tensor = torch.argmax(all_preds_batches, dim=1) 
            final_targets_tensor = all_targets_batches

            if len(final_preds_tensor) > 0 and len(final_targets_tensor) > 0:
                cm = confusion_matrix(final_targets_tensor.numpy(), final_preds_tensor.numpy())
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                            xticklabels=self.class_names, yticklabels=self.class_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Confusion Matrix - {self.model_name} (v{run_version_str})")
                cm_path = os.path.join(report_dir_for_run, f"{self.model_name}_v{run_version_str}_confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                print(f"Saved confusion matrix to {cm_path}")

                report_str = classification_report(final_targets_tensor.numpy(), final_preds_tensor.numpy(), 
                                                 target_names=self.class_names, zero_division=0)
                print(f"\nFinal Classification Report - {self.model_name} (v{run_version_str}):\n{report_str}")
                report_path = os.path.join(report_dir_for_run, f"{self.model_name}_v{run_version_str}_classification_report.txt")
                with open(report_path, "w") as f:
                    f.write(report_str)
                print(f"Saved classification report to {report_path}")
            else:
                print("No validation predictions/targets collected to generate confusion matrix.")
            
            pl_module.all_val_preds.clear()
            pl_module.all_val_targets.clear()
        else:
            print("Could not generate confusion matrix: 'all_val_preds' or 'all_val_targets' not found or empty in pl_module.")

        csv_logger_instance = None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                csv_logger_instance = logger
                break
        
        if csv_logger_instance:
            metrics_file_path = os.path.join(csv_logger_instance.log_dir, "metrics.csv")

            if os.path.exists(metrics_file_path):
                try:
                    metrics_df = pd.read_csv(metrics_file_path)
                    plt.figure(figsize=(18, 12)) 
                    
                    # Use epoch-suffixed names as generated by CSVLogger for on_epoch=True metrics
                    plotable_metrics = [
                        "val_loss_epoch", "val_accuracy_epoch", 
                        "val_precision_epoch", "val_recall_epoch", "val_f1_epoch",
                        "train_loss_epoch", "train_accuracy_epoch"
                    ]
                    
                    # Remove previous renaming logic, directly use expected column names
                    available_metrics = [m for m in plotable_metrics if m in metrics_df.columns and not metrics_df[m].dropna().empty]
                    
                    num_plots = len(available_metrics)
                    cols = 3
                    rows = (num_plots + cols -1) // cols

                    for i, metric_name in enumerate(available_metrics):
                        plt.subplot(rows, cols, i + 1)
                        epoch_metric_data = metrics_df[['epoch', metric_name]].dropna(subset=[metric_name]) # Drop rows where current metric is NaN

                        if not epoch_metric_data.empty:
                            epoch_metric_data['epoch'] = epoch_metric_data['epoch'].astype(int)
                            agg_metric_data = epoch_metric_data.groupby('epoch')[metric_name].mean().reset_index()
                            plt.plot(agg_metric_data['epoch'], agg_metric_data[metric_name], marker='o', linestyle='-')
                            plt.title(metric_name.replace("_", " ").title())
                            plt.xlabel("Epoch")
                            plt.ylabel("Value")
                            plt.grid(True)
                        else:
                            plt.text(0.5, 0.5, f'{metric_name}\n(No data to plot)', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
                            plt.title(metric_name.replace("_", " ").title())
                            plt.xlabel("Epoch")
                            plt.ylabel("Value")
                            
                    plt.suptitle(f"Metrics Over Epochs - {self.model_name} (v{run_version_str})", fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
                    plot_path = os.path.join(report_dir_for_run, f"{self.model_name}_v{run_version_str}_metrics_over_epochs.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Saved metrics plot to {plot_path}")
                except Exception as e:
                    print(f"Error plotting metrics from CSV ({metrics_file_path}): {e}")
            else:
                print(f"Metrics CSV file not found at {metrics_file_path}. Cannot plot metrics over epochs.")
        else:
            print("CSVLogger not found in trainer.loggers. Cannot plot metrics from CSV.")


def get_callbacks(args, class_names, model_name, report_subdir_name): # Added model_name, report_subdir_name
    callbacks = []

    early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        )
    callbacks.append(early_stopping_callback)

    # Checkpoint path using root_dir
    checkpoint_dir = os.path.join(args.root_dir, "checkpoints", model_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        filename=f"{model_name}_{{epoch:02d}}_{{val_loss:.2f}}", # User updated format
        save_top_k=args.save_k,
        save_last=True,
        mode="min"
    )
    callbacks.append(checkpoint_callback)

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor_callback)

    # Replaced RichProgressBar with TQDMProgressBar
    tqdm_progress_bar = TQDMProgressBar(refresh_rate=20 if args.notebook else 1)
    callbacks.append(tqdm_progress_bar)

    # New ReportMetrics callback
    report_metrics_callback = ReportMetrics(class_names=class_names, 
                                            model_name=model_name,
                                            report_subdir=report_subdir_name)
    callbacks.append(report_metrics_callback)

    return callbacks

def get_dataset(train_folder_path, val_folder_path, is_inception=False):
    if is_inception:
        IMG_SIZE = 299
    else:
        IMG_SIZE = 224
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(train_folder_path, transform=transform)
    val_dataset = ImageFolder(val_folder_path, transform=transform)

    return train_dataset, val_dataset

def get_augmentations(original_data_root, augmented_data_root, num_augmentations_per_image=5):
    # Define augmentations (customize as needed)
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomGrayscale(p=0.1),
    ])

    # --- Create augmented dataset ---
    for emotion_dir in os.listdir(original_data_root):
        original_dir = os.path.join(original_data_root, emotion_dir)
        augmented_dir = os.path.join(augmented_data_root, emotion_dir)
        
        os.makedirs(augmented_dir, exist_ok=True)  # Create output dir (e.g., /augmented_data/angry)

        for img_name in os.listdir(original_dir):
            img_path = os.path.join(original_dir, img_name)
            
            # --- Step 1: Copy original image to augmented dir ---
            original_save_path = os.path.join(augmented_dir, f"orig_{img_name}")
            shutil.copy2(img_path, original_save_path)  # Preserves metadata
            
            # --- Step 2: Generate & save augmented images ---
            image = Image.open(img_path).convert("RGB")
            
            for i in range(num_augmentations_per_image):
                augmented_image = augmentation(image)
                augmented_save_path = os.path.join(augmented_dir, f"aug_{i}_{img_name}")
                augmented_image.save(augmented_save_path)

    print("Augmentation complete! Check:", augmented_data_root)

def main(args):
    models = {
        "DenseNet121": DenseNet121,
        "ResNet50": ResNet50,
        "VGG16": VGG16,
        "MobileNetV1": MobileNetV1,
        "InceptionV3": InceptionV3
    }
    model = models[args.model]()
    if args.model == "InceptionV3":
        is_inception = True
    else:
        is_inception = False
    print(f"Loaded {args.model} model")

    if args.augment:
        get_augmentations(os.path.join(args.data_dir, "train"), 
                          os.path.join(args.root_dir, "Augmented_Folder", "train"),
                          num_augmentations_per_image=3)
    
    # Use args.data_dir for dataset path
    train_folder_path = os.path.join(args.root_dir, "Augmented_Folder", "train")
    val_folder_path = os.path.join(args.data_dir, "valid")
    train_dataset, val_dataset = get_dataset(train_folder_path, val_folder_path, is_inception)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    classifier = PetExpressionClassifier(model, args.lr)
    
    model_name_arg = args.model

    # --- Logger Setup ---
    # Loggers save_dir using args.root_dir
    logger_save_dir = os.path.join(args.root_dir, "logs")
    csv_logger = CSVLogger(save_dir=logger_save_dir, name=model_name_arg)
    tensorboard_logger = TensorBoardLogger(save_dir=logger_save_dir, name=model_name_arg, sub_dir="tensorboard_events")
    
    loggers = [csv_logger, tensorboard_logger]

    # Pass model_name and the desired subdirectory name for reports to the callback setup
    callbacks = get_callbacks(args, 
                              class_names=["happy", "neutral", "sad", "angry"], 
                              model_name=model_name_arg,
                              report_subdir_name="custom_reports_and_plots") # This is a sub-folder name

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=loggers, # Pass list of loggers
        log_every_n_steps=10,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        num_sanity_val_steps=0
    )
    trainer.fit(classifier, train_loader, val_loader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DenseNet121")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_k", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--notebook", action="store_true", help="Enable notebook-optimized TQDM progress bar display (adjusts refresh rate).")

    # Arguments for path management (Kaggle friendly)
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory for all outputs (logs, checkpoints, reports)")
    parser.add_argument("--data_dir", type=str, default=os.path.join("pets_expression", "Master Folder"), help="Directory for input dataset")

    args = parser.parse_args()

    main(args)

