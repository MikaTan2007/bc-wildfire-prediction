import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Loading Data
data = pd.read_csv("bc_fire_ml_dataset.csv")

# Dropping rows with NA values
data = data.dropna(subset = ['total_precipitation_sum', 'temperature_c', 'dewpoint_c'])

features = ['total_precipitation_sum', 'temperature_c', 'dewpoint_c']
X_features = data[features].values
y_labels = data['label'].values

pre_data = train_test_split(X_features, y_labels, test_size=0.15, random_state=42)
X_temp = pre_data[0]
X_test = pre_data[1]
y_temp = pre_data[2]
y_test = pre_data[3]

# Train/Test Split & Normalization
all_data = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)
# Produces [X_train, X_test, y_train, y_val]
X_train = all_data[0]
X_val = all_data[1]
y_train = all_data[2]
y_val = all_data[3]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.FloatTensor(y_val).view(-1,1)

X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Checking if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Neural Network Architecture
class WildfireClassifier(nn.Module):
    def __init__(self, input_dim):
        super(WildfireClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1) # We output to a single node for binary classification
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = WildfireClassifier(len(features))
model.to(device)

# Defining Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Training Loop
epochs = 3000

print(f"Starting training loop with {epochs} epochs")

for epoch in range(epochs):
    model.train()
    # Resetting Gradients
    optimizer.zero_grad()

    inputs = X_train_tensor.to(device)
    labels = y_train_tensor.to(device)

    # Forward Pass
    train_outputs = model(inputs)
    train_loss = criterion(train_outputs, labels)

    # Backward pass and gradient step
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_inputs = X_val_tensor.to(device)
        val_labels = y_val_tensor.to(device)

        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

# Model Evaluating
model.eval()

with torch.no_grad():
    inputs = X_test_tensor.to(device)

    test_outputs = model(inputs)

    # Convert logits to probabilities using Sigmoid
    probs = torch.sigmoid(test_outputs.cpu())

    predictions = (probs >= 0.5).float()

    # Calculating accuracy
    correct = (predictions == y_test_tensor).sum().item()

    accuracy = correct / y_test_tensor.size(0)

    print(f"Test accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report

y_pred_numpy = predictions.cpu().numpy()
y_true_numpy = y_test_tensor.cpu().numpy()

cm = confusion_matrix(y_true_numpy, y_pred_numpy)

tn, fp, fn, tp = cm.ravel()

print("\n" + "="*50)
print("BC WILDFIRE MODEL CONFUSION MATRIX".center(50))
print("="*50)
print(f"{'':<20} | {'Predicted 0':<10} | {'Predicted 1':<10}")
print(f"{'':<20} | {'(No Fire)':<10}  | {'(Fire)':<10}")
print("-"*50)
print(f"{'Actual 0 (No Fire)':<20} | {tn:<10}  | {fp:<10}")
print("-"*50)
print(f"{'Actual 1 (Fire)':<20} | {fn:<10}  | {tp:<10}")
print("="*50)

print("\n")

# Prediction, Recall, and F1 Score
print("="*53)
print("Classification Report".center(53))
print("="*53)
print(classification_report(y_true_numpy, y_pred_numpy, target_names=["No Fire", "Fire"]))

# Saving model weights
model_weight_save_path = "wildfire_model.pth"
torch.save(model.state_dict(), model_weight_save_path)
print(f"Model weights saved to {model_weight_save_path}")

# Saving standard scalar
scaler_save_path = "scaler.joblib"
joblib.dump(scaler, scaler_save_path)
print(f"Standard Scaler saved to {scaler_save_path}")