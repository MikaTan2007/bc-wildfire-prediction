import torch
import pandas as pd
import joblib
import torch.nn as nn

class WildfireClassifier(nn.Module):
    def __init__(self, input_dim):
        super(WildfireClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = WildfireClassifier(input_dim=3)

# Load the Model & Scaler
model.load_state_dict(torch.load("../training/wildfire_model.pth", map_location=device))
model.to(device)
model.eval() 
scaler = joblib.load("../training/scaler.joblib")

# 4. Load your new testing data
df = pd.read_csv("../datasets/post-processed/2025/bc_weather_current_fires.csv")

X_raw = df[['total_precipitation_sum', 'temperature_c', 'dewpoint_c']].values

# 5. Preprocess the Data
X_scaled = scaler.transform(X_raw)
X_tensor = torch.FloatTensor(X_scaled).to(device)

# 6. Execute predictions!
y_pred_probs = []
y_pred_labels = []

with torch.no_grad():
    logits = model(X_tensor)
    probabilities = torch.sigmoid(logits)

predictions = (probabilities >= 0.5).int()

total_fires = len(predictions)
predicted_as_fire = sum(predictions) # Counts how many 1s
predicted_as_no_fire = total_fires - predicted_as_fire # Counts how many 0s

pred_fire_count = predicted_as_fire.item() 
pred_no_fire_count = predicted_as_no_fire.item()

accuracy_on_fires = (pred_fire_count / total_fires) * 100

print("=========================================")
print("      2025 ALL-FIRE DATASET EVALUATION    ")
print("=========================================")
print(f"Total Actual Fires Evaluated : {total_fires}")
print(f"Model Predicted as 'Fire'    : {pred_fire_count}")
print(f"Model Predicted as 'No Fire' : {pred_no_fire_count}")
print(f"Recall (Accuracy on Fires)   : {accuracy_on_fires:.2f}%")