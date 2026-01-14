import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import shutil
import json
import datetime
import numpy as np

from src.config import ModelConfig
from src.model.hierarchical_model import HierarchicalStockModel
from src.dataset import StockDataset
from src.utils.plot import plot_predictions, plot_history

def train():
    config = ModelConfig
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建实验记录文件夹
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment outputs will be saved to: {exp_dir}")
    
    # 备份 config.py
    try:
        shutil.copy('src/config.py', os.path.join(exp_dir, 'config_backup.py'))
    except Exception as e:
        print(f"Warning: Could not copy config file: {e}")

    # 2. 数据准备
    # 使用预处理后的数据
    csv_path = ModelConfig.DATA_CSV if hasattr(ModelConfig, 'DATA_CSV') else os.path.join('data', 'GOOGL_processed.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return

    print("Loading datasets...")
    train_dataset = StockDataset(csv_path, config, mode='train')
    val_dataset = StockDataset(csv_path, config, mode='val')
    test_dataset = StockDataset(csv_path, config, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # 3. 模型初始化
    model = HierarchicalStockModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 4. 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Early Stopping 计数器
    patience = getattr(config, 'PATIENCE', 20)
    counter = 0
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (x_long, x_medium, x_short, y) in enumerate(train_loader):
            x_long, x_medium, x_short, y = x_long.to(device), x_medium.to(device), x_short.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x_long, x_medium, x_short)
            loss = criterion(output, y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 5. 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_long, x_medium, x_short, y in val_loader:
                x_long, x_medium, x_short, y = x_long.to(device), x_medium.to(device), x_short.to(device), y.to(device)
                output = model(x_long, x_medium, x_short)
                loss = criterion(output, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            print(f"Saved best model with val_loss: {best_val_loss:.6f}")
            counter = 0 # reset counter
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
        if counter >= patience:
            print("Early Stopping triggered.")
            break

    # 训练结束，加载最佳模型进行最终评估
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for final testing.")
        
    # 6. 在测试集上评估 (Final Test)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_long, x_medium, x_short, y in test_loader:
            x_long, x_medium, x_short, y = x_long.to(device), x_medium.to(device), x_short.to(device), y.to(device)
            output = model(x_long, x_medium, x_short)
            loss = criterion(output, y)
            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss (MSE): {avg_test_loss:.6f}")

    # 保存记录和绘图
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss,
        'final_test_loss': avg_test_loss
    }
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    plot_history(train_losses, val_losses, exp_dir)
    
    # 绘制各集预测情况
    # 创建专门的图片文件夹
    plots_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    target_scaler = test_dataset.target_scaler if hasattr(test_dataset, 'target_scaler') else None
    
    print("Plotting predictions for Train, Val, and Test sets...")
    plot_predictions(model, train_loader, device, plots_dir, scaler=target_scaler, filename='train_predictions.png')
    plot_predictions(model, val_loader, device, plots_dir, scaler=target_scaler, filename='val_predictions.png')
    plot_predictions(model, test_loader, device, plots_dir, scaler=target_scaler, filename='test_predictions.png')

if __name__ == "__main__":
    train()
