import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import logging
from tqdm import tqdm
import numpy as np

# Add current dir to path
sys.path.append(os.getcwd())

from tkan_dataset_builder.kan_gnn import AlphaGammaNet
from core.tkan_model import FocalLoss

import subprocess

def auto_git_push(msg="Auto-commit: Updated model weights"):
    """Pushes weights and logs to GitHub."""
    try:
        logging.info(f"Syncing to Git: {msg}")
        subprocess.run(["git", "add", "graph_tkan_best.pth", "*.log"], check=False)
        subprocess.run(["git", "commit", "-m", msg], check=False)
        subprocess.run(["git", "push", "origin", "main"], check=False)
        logging.info("Git Sync Complete.")
    except Exception as e:
        logging.warning(f"Git Sync Failed: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TemporalTensorDataset(Dataset):
    """
    Simpler dataset wrapper for optimized tensors.
    """
    def __init__(self, data_dict, device=None):
        self.x = data_dict['x']
        self.adj = data_dict['adj']
        self.agent = data_dict['agent']
        self.y = data_dict['y']
        
        if device and device.type == 'cuda':
            logging.info("Moving dataset to VRAM for Speedup...")
            self.x = self.x.to(device)
            self.adj = self.adj.to(device)
            self.agent = self.agent.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.adj[idx], self.agent[idx], self.y[idx]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on {device}")

    # 1. Load Optimized Data
    train_path = 'graph_data_output/train_tensor.pt'
    val_path = 'graph_data_output/val_tensor.pt'
    
    if not os.path.exists(train_path):
        logging.error("Optimized tensor data not found. Run optimize_dataset.py first.")
        return

    logging.info("Loading optimized tensors...")
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    
    train_ds = TemporalTensorDataset(train_data, device=device)
    val_ds = TemporalTensorDataset(val_data, device=device)
    
    # Shuffle=True for better convergence since we don't depend on inter-sequence state anymore
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Initialize Model
    model = AlphaGammaNet(node_in=4, global_in=3, hidden_dim=32, n_classes=3).to(device)
    
    # Loss functions
    alpha_weights = torch.tensor([0.2, 2.0, 2.0]).to(device)
    class_criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 3. Training Loop
    epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, adj, agent, y in pbar:
            # If device is cuda, data is already there!
            if device.type != 'cuda':
                x, adj, agent, y = x.to(device), adj.to(device), agent.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (Stateless TBPTT removed as per user recommendation for speed)
            logits, _, _ = model(x, adj, agent)
            
            loss = class_criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, adj, agent, y in val_loader:
                x, adj, agent, y = x.to(device), adj.to(device), agent.to(device), y.to(device)
                logits, _, _ = model(x, adj, agent)
                loss = class_criterion(logits, y)
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.2f}%")
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'graph_tkan_best.pth')
            logging.info("Saved Best Model Checkpoint.")

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.2f}%")
        
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'graph_tkan_best.pth')
            logging.info("Saved Best Model Checkpoint.")

if __name__ == '__main__':
    train()
