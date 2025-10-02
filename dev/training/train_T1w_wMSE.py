import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, criterion, device, max_epochs=5000, patience=15, es_warmup=150, report=None, scheduler=None):
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        note = ""
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
            images_input = batch['input'].to(device, dtype=torch.float32)
            images_gt = batch['output'].to(device, dtype=torch.float32)
            seg_mask = batch['seg'].to(device, dtype=torch.float32)

            if len(images_gt.shape) > 4:
                batch_size, num_regions, *rest = images_input.shape
                images_input = images_input.reshape(batch_size * num_regions, *rest).unsqueeze(1)
                images_gt = images_gt.reshape(batch_size * num_regions, *rest).unsqueeze(1)
                seg_mask = seg_mask.reshape(batch_size * num_regions, *rest).unsqueeze(1)

            # Compute voxel-wise weights
            weights = seg_mask

            _, images_out = model(images_input)
            loss = criterion(images_out, images_gt, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images_input = batch['input'].to(device, dtype=torch.float32)
                images_gt = batch['output'].to(device, dtype=torch.float32)
                seg_mask = batch['seg'].to(device, dtype=torch.float32)

                if len(images_gt.shape) > 4:
                    batch_size, num_regions, *rest = images_input.shape
                    images_input = images_input.reshape(batch_size * num_regions, *rest).unsqueeze(1)
                    images_gt = images_gt.reshape(batch_size * num_regions, *rest).unsqueeze(1)
                    seg_mask = seg_mask.reshape(batch_size * num_regions, *rest).unsqueeze(1)

                weights = seg_mask
                _, images_out = model(images_input)
                loss = criterion(images_out, images_gt, weight=weights)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} — Val Loss: {avg_val_loss:.6f} — lr = {lr:.2e}")

        # --- Early Stopping ---
        if es_warmup < epoch:
            if avg_val_loss < best_val_loss:
                note += "ES"
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break

        if report is not None:
            report.log_epoch(epoch + 1, avg_train_loss, val_loss=avg_val_loss, note=note)

        torch.save(model.state_dict(), f"./data/pretrained_models/CP_{report.timestamp}.pth")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses
