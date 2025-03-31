import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, 
    recall_score, accuracy_score, roc_curve
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    """
    Custom dataset class for PyTorch.
    """
    def __init__(self, X, y):
        # Convert to numpy arrays if they are pandas Series/DataFrame
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.to_numpy()
            
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
        self.length = self.X.shape[0]
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.length

class NeuralNetwork(nn.Module):
    """
    Neural Network model for CHF prediction.
    """
    def __init__(self, input_size, hidden_layer_size=128):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, hidden_layer_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layer_size, hidden_layer_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layer_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.sequential(x)

def train_epoch(model, dataloader, loss_fn, optimizer):
    """
    Train model for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in dataloader:
        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
    
    return running_loss / len(dataloader.dataset)

def evaluate_epoch(model, dataloader, loss_fn):
    """
    Evaluate model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            running_loss += loss.item() * X_batch.size(0)
    
    return running_loss / len(dataloader.dataset)

def train_neural_network(model, train_dataloader, val_dataloader, epochs=1000, 
                        loss_fn=nn.BCELoss(), optimizer=None, patience=50):
    """
    Train neural network with early stopping.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train and evaluate for one epoch
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer)
        val_loss = evaluate_epoch(model, val_dataloader, loss_fn)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if epoch % 10 == 0:
            logging.info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_neural_network(model, test_dataloader):
    """
    Evaluate neural network and generate metrics.
    """
    model.eval()
    
    y_true = []
    y_pred_proba = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            y_pred = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred_proba.extend(y_pred.cpu().numpy())
    
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Get binary predictions
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    
    return metrics

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load processed data
        logging.info("Loading processed data...")
        processed_data = pd.read_csv('processed_chf_data.csv')
        logging.info(f"Loaded data shape: {processed_data.shape}")
        
        # Prepare data splits
        from train_models import prepare_data_splits  # Import from previous script
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(
            processed_data
        )
        
        # Create datasets and dataloaders
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        test_dataset = CustomDataset(X_test, y_test)
        
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize and train neural network
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size).to(device)
        
        logging.info("Training neural network...")
        model, train_losses, val_losses = train_neural_network(
            model, train_dataloader, val_dataloader
        )

        
        # Evaluate model
        logging.info("Evaluating neural network...")
        metrics = evaluate_neural_network(model, test_dataloader)
        
        # Log results
        logging.info("\nNeural Network Performance:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.3f}")
        
        # Save metrics
        pd.DataFrame([metrics], index=['NeuralNetwork']).to_csv('neural_network_results.csv')
        
        logging.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Stack trace:", exc_info=True)
        raise



    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load processed data
        logging.info("Loading processed data...")
        processed_data = pd.read_csv('processed_chf_data.csv')
        logging.info(f"Loaded data shape: {processed_data.shape}")
        
        # Prepare data splits
        from train_models import prepare_data_splits  # Import from previous script
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data_splits(
            processed_data
        )
        
        logging.info(f"Training features shape: {X_train.shape}")
        logging.info(f"Training labels shape: {y_train.shape}")
        
        # Create datasets and dataloaders
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        test_dataset = CustomDataset(X_test, y_test)
        
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize and train neural network
        input_size = X_train.shape[1]
        model = NeuralNetwork(input_size).to(device)
        
        logging.info("Training neural network...")
        model, train_losses, val_losses = train_neural_network(
            model, train_dataloader, val_dataloader
        )
        
        # Evaluate model
        logging.info("Evaluating neural network...")
        metrics = evaluate_neural_network(model, test_dataloader)
        
        # Log results
        logging.info("\nNeural Network Performance:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.3f}")
        
        # Save metrics
        pd.DataFrame([metrics], index=['NeuralNetwork']).to_csv('neural_network_results.csv')
        
        logging.info("Training and evaluation completed successfully!")
        
    except Exception as e:
        logging.error("Error occurred: %s", str(e))
        logging.error("Stack trace:", exc_info=True)
        raise