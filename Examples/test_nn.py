import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import joblib

# Define the fully connected neural network model (same as in the training script)
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.fc5 = nn.Linear(hidden_dims[3], output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Load the best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 21
hidden_dims = [128, 64, 32, 16]
output_dim = 37  # Number of temperature values to predict
model = FullyConnectedNN(input_dim, hidden_dims, output_dim).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Load the saved scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load and preprocess the test data
test_data = pd.read_table('input_data.txt', sep='\s+', encoding='gb2312', header=None)

# Extract features (first 21 columns) and true values (next 37 columns)
X_test = test_data.iloc[:, :21].values
y_test_true = test_data.iloc[:, 21:].values  # True values

# Apply the same scaling as used for training
X_test_scaled = scaler_X.transform(X_test)

# Convert to PyTorch tensor
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# Predict the temperature values
with torch.no_grad():
    y_test_pred = model(X_test_tensor).cpu().numpy()

# Inverse transform to get the original scale for predicted values
y_test_pred = scaler_y.inverse_transform(y_test_pred)

# Combine input features, predicted temperatures, and true values for output
output = np.hstack((X_test, y_test_pred, y_test_true))

# Save the predictions and true values to a file with 5 decimal places
output_df = pd.DataFrame(output)
output_df.to_csv('output_data.txt', index=False, header=False, sep='\t', float_format='%.5f')

print("Predictions and true values for the test data have been saved to 'test_predictions.txt'.")
