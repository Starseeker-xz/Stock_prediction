import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_predictions(
    model,
    loader,
    device,
    save_dir,
    scaler=None,
    filename='prediction_analysis.png',
    predict_residual: bool = False,
):
    """
    绘制预测值与真实值的对比图
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_long, x_medium, x_short, y, prev_y in loader:
            # Data is already on device if using GPUDataset, but .to(device) is idempotent (safe)
            if not x_long.is_cuda and device.type == 'cuda':
                x_long = x_long.to(device)
                x_medium = x_medium.to(device)
                x_short = x_short.to(device)
            
            output = model(x_long, x_medium, x_short)
            if predict_residual:
                output = output + prev_y
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Check if scaler is fitted before inverse transform
    if scaler is not None:
        try:
            # Simple check if scaler is fitted (sklearn scalers have mean_ attribute after fit)
            if hasattr(scaler, 'mean_') or hasattr(scaler, 'min_'):
                preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"Warning: inverse_transform failed, using raw values. Error: {e}")

    plt.figure(figsize=(15, 7))
    plt.plot(targets, label='Actual', alpha=0.7, color='blue')
    plt.plot(preds, label='Predicted', alpha=0.7, color='red', linestyle='--')
    
    # 计算偏差指标
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    plt.title(f'Prediction Deviation (MAE: {mae:.4f}, RMSE: {rmse:.4f})')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Prediction plot saved to {save_path}")
    
    # 额外画一个残差图
    plt.figure(figsize=(15, 5))
    residuals = preds - targets
    plt.plot(residuals, color='green', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Residuals (Pred - Actual)')
    plt.grid(True, alpha=0.3)
    
    res_path = os.path.join(save_dir, 'residuals.png')
    plt.savefig(res_path, dpi=300)
    plt.close()

def plot_history(train_losses, val_losses, save_dir):
    """
    绘制训练过程的Loss曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'loss_history.png')
    plt.savefig(save_path)
    plt.close()
