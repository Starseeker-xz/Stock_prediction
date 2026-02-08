import argparse
import os
import sys
import torch
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.hierarchical_model import HierarchicalStockModel
from src.dataset import StockDataset
import src.config as current_config

def load_config_from_path(path):
    spec = importlib.util.spec_from_file_location("custom_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ModelConfig

def predict(checkpoint_path, data_path=None, output_dir='predictions', device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Determine Config
    # Check if config_backup.py exists in the checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_backup_path = os.path.join(checkpoint_dir, 'config_backup.py')
    
    if os.path.exists(config_backup_path):
        print(f"Found config backup at: {config_backup_path}. Loading...")
        Config = load_config_from_path(config_backup_path)
        
        # Monkey patch src.config.ModelConfig so HierarchicalStockModel uses it
        # Note: This is a bit hacky but avoids changing model code
        import src.config
        src.config.ModelConfig = Config
    else:
        print("No config backup found. Using current src.config.ModelConfig")
        Config = current_config.ModelConfig

    # 2. Data Preparation
    if data_path is None:
        data_path = getattr(Config, 'DATA_CSV', 'data/GOOGL_processed.csv')
    
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Create dataset to get scalers and test data
    # We use mode='test' to get the test split
    test_dataset = StockDataset(data_path, Config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. Model Initialization & Loading
    print(f"Loading model from: {checkpoint_path}")
    model = HierarchicalStockModel().to(device)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()
    
    # 4. Inference
    print("Running inference...")
    preds = []
    gts = []
    
    criterion = torch.nn.MSELoss()
    test_loss = 0
    
    predict_residual = bool(getattr(Config, "PREDICT_RESIDUAL", False))

    with torch.no_grad():
        for x_long, x_medium, x_short, y, prev_y in test_loader:
            x_long = x_long.to(device)
            x_medium = x_medium.to(device)
            x_short = x_short.to(device)
            y = y.to(device)
            prev_y = prev_y.to(device)
            
            output = model(x_long, x_medium, x_short)
            if predict_residual:
                output = output + prev_y
            loss = criterion(output, y)
            test_loss += loss.item()
            
            preds.append(output.cpu().numpy())
            gts.append(y.cpu().numpy())
    
    print(f"Test Loss (MSE): {test_loss / len(test_loader):.6f}")
    
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    
    # Inverse Transform if needed
    target_normalize = getattr(Config, 'TARGET_NORMALIZE', True)
    if target_normalize and hasattr(test_dataset, 'target_scaler'):
        print("Inverse scaling predictions...")
        # target_scaler was fitted on (N, 1) shape
        preds_unscaled = test_dataset.target_scaler.inverse_transform(preds)
        gts_unscaled = test_dataset.target_scaler.inverse_transform(gts)
    else:
        print("Target normalization disabled or scaler not found. Using raw predictions.")
        preds_unscaled = preds
        gts_unscaled = gts
        
    # Calculate Metrics
    mse = mean_squared_error(gts_unscaled, preds_unscaled)
    mae = mean_absolute_error(gts_unscaled, preds_unscaled)
    rmse = np.sqrt(mse)
    
    print(f"Unscaled Metrics -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 5. Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(gts_unscaled, label='True Value', alpha=0.7)
    plt.plot(preds_unscaled, label='Prediction', alpha=0.7)
    plt.title(f'Stock Prediction (RMSE={rmse:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using HierarchicalStockModel')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint file')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--output', type=str, default='predictions', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    predict(args.checkpoint, args.data, args.output, args.device)
