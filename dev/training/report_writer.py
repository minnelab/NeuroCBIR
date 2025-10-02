import os
from datetime import datetime
from torchsummary import summary
import torch

class TrainingReport:
    def __init__(self, model, input_shape, dataset_info, optimizer=None, lr=None, weight_decay=None, scheduler=None, report_dir="training_reports"):
        os.makedirs(report_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(report_dir, f"train_report_{self.timestamp}.txt")
        
        self.write_header(model, input_shape, dataset_info, optimizer, lr, weight_decay, scheduler)

    def write_header(self, model, input_shape, dataset_info, optimizer, lr, weight_decay, scheduler):
        with open(self.file_path, 'w') as f:
            f.write(f"Training Report\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")

            f.write("Model Architecture:\n")
            try:
                summary_str = summary(model, input_shape, device="cpu" if not next(model.parameters()).is_cuda else "cuda")
                # In case you're running in a headless/CLI env where `summary` prints instead of returning:
                f.write(str(summary_str))
            except:
                f.write(str(model))  # fallback to plain print

            f.write("\n\nDataset Info:\n")
            for key, value in dataset_info.items():
                f.write(f"  - {key}: {value}\n")

            f.write("\nTraining Settings:\n")
            if optimizer:
                f.write(f"  - Optimizer: {optimizer.__class__.__name__}\n")
            if lr:
                f.write(f"  - Learning rate: {lr}\n")
            if weight_decay is not None:
                f.write(f"  - Weight decay: {weight_decay}\n")
            if scheduler is not None:
                f.write(f"  - Scheduler: {scheduler.__class__.__name__}\n")
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    f.write(f"    • Mode: {scheduler.mode}\n")
                    f.write(f"    • Factor: {scheduler.factor}\n")
                    f.write(f"    • Patience: {scheduler.patience}\n")
                    f.write(f"    • Threshold: {scheduler.threshold}\n")
                    f.write(f"    • Cooldown: {scheduler.cooldown}\n")
                    f.write(f"    • Min LR: {scheduler.min_lrs}\n")

            f.write("\nEpoch Log:\n")
            f.write("Epoch | Train Loss | Val Loss | Val Acc | Notes\n")
            f.write("------|------------|----------|---------|------\n")

    def log_epoch(self, epoch, train_loss, val_loss=None, val_acc=None, note=""):
        with open(self.file_path, 'a') as f:
            f.write(f"{epoch:<5} | {train_loss:<10.6f} | {val_loss or -99.9:<8.6f} | {val_acc or -99.9:<7.2f} | {note}\n")

'''
Usage example:

# Assuming `model`, `train_loader`, etc. are already defined
report = TrainingReport(
    model=model,
    input_shape=(1, 64, 80, 48),
    dataset_info={"Train Samples": len(train_dataset), "Val Samples": len(val_dataset)},
    optimizer=optimizer,
    lr=0.001,
    weight_decay=1e-5
)

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(...)
    val_loss, val_acc = validate(...)

    report.log_epoch(epoch, train_loss, val_loss, val_acc)

'''