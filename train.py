from lightning import Trainer
from lightning import LightningModule
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.optim as optim
import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger

from models.DenseNet121 import DenseNet121
from models.ResNet import ResNet50
from models.VGG import VGG16
from models.MobileNet import MobileNetV1
from models.InceptionV3 import InceptionV3

import argparse

class PetExpressionClassifier(LightningModule):
    def __init__(self, model, lr=0.001):
        super(PetExpressionClassifier, self).__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss)
        # print(f"Train Loss: {loss}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("val_loss", loss)
        # print(f"Val Loss: {loss}")
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
class LogToTerminal(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Train Loss at Epoch {trainer.current_epoch}: {trainer.callback_metrics['train_loss']}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"Val Loss at Epoch {trainer.current_epoch}: {trainer.callback_metrics['val_loss']}")

def get_callbacks(args):
    callbacks = []

    early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        )
    callbacks.append(early_stopping_callback)

    # Combined checkpoint callback that saves both last and best models
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.model}",
        monitor="val_loss",
        filename=f"{args.model}_epoch_{{epoch:02d}}_val_loss_{{val_loss:.2f}}",
        save_top_k=args.save_k,
        save_last=True,
        mode="min"
    )
    callbacks.append(checkpoint_callback)

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor_callback)

    rich_progress_bar_callback = RichProgressBar()
    callbacks.append(rich_progress_bar_callback)

    log_to_terminal_callback = LogToTerminal()
    callbacks.append(log_to_terminal_callback)

    return callbacks

def get_dataset(folder_path, is_inception=False):
    if is_inception:
        IMG_SIZE = 299
    else:
        IMG_SIZE = 224
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(os.path.join(folder_path, "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join(folder_path, "valid"), transform=transform)
    return train_dataset, val_dataset



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

    folder_path = "Master_Folder"
    train_dataset, val_dataset = get_dataset(folder_path, is_inception)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)


    classifier = PetExpressionClassifier(model, args.lr)
    callbacks = get_callbacks(args)
    logger = CSVLogger("logs", name=args.model)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accelerator="gpu" if args.gpus > 0 else None,
        devices=args.gpus,
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
    args = parser.parse_args()

    main(args)

