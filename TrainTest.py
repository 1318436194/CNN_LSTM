import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, seq):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        _, (hidden, cell) = self.encoder(x)
        y, _ = self.decoder(seq, (hidden, cell))
        y = self.fc(y)
        return y
def define_search_space(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 50, 200)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1)
    return hidden_dim, num_layers, dropout, learning_rate
def train_model(hidden_dim, num_layers, dropout, learning_rate, num_epochs=100):
    model = CNN_LSTM(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            decoder_input = torch.zeros_like(y_batch)
            optimizer.zero_grad()
            output = model(X_batch, decoder_input)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')
    model.eval()
    with torch.no_grad():
        decoder_input = torch.zeros_like(y_val_tensor)
        preds_val = model(X_val_tensor, decoder_input)
        val_loss = criterion(preds_val, y_val_tensor)
    return val_loss.item()
def objective(trial):
    hidden_dim, num_layers, dropout, learning_rate = define_search_space(trial)
    val_loss = train_model(hidden_dim, num_layers, dropout, learning_rate, num_epochs=30)
    return val_loss
def extract(best_trial):
    hidden_dim = best_trial.params["hidden_dim"]
    num_layers = best_trial.params["num_layers"]
    dropout = best_trial.params["dropout"]
    learning_rate = best_trial.params["learning_rate"]
    return hidden_dim, num_layers, dropout, learning_rate
## Temp Part
print("Temp Part")
input_dim = 2
output_dim = 1
steps = 1
inputs = 30
tries = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv("dataAll.csv").sort_values(by='Time',ascending=True)
data = df[['Temp', 'Humi']].values
target = df['Temp'].shift(-steps).values[:-steps]
scalerdata = MinMaxScaler()
scalertarget = MinMaxScaler()
data_scaled = scalerdata.fit_transform(data)
target_scaled = scalertarget.fit_transform(target.reshape(-1, 1))

X = []
y = []
for i in range(inputs, len(data_scaled)-steps*2):
    X.append(data_scaled[i-inputs:i])
    y.append(target_scaled[i:i+steps])
    
X, y = np.array(X), np.array(y)
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=tries)
best_trial = study.best_trial
print("The best hyperparameters: {}".format(best_trial.params))
print("The best validation loss: {:.5f}".format(best_trial.value))
best_hidden_dim, best_num_layers, best_dropout, best_learning_rate = extract(best_trial)

best_model = CNN_LSTM(input_dim, best_hidden_dim, output_dim, best_num_layers, best_dropout).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_learning_rate)
num_epochs = 100
for epoch in range(num_epochs):
    best_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        decoder_input = torch.zeros_like(y_batch)
        optimizer.zero_grad()
        output = best_model(X_batch, decoder_input)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
save_path = './best_save_temp.pth'
torch.save({
    'model_state_dict': best_model.state_dict(),
    'hidden_dim': best_hidden_dim,
    'num_layers': best_num_layers,
    'dropout': best_dropout,
    'learning_rate': best_learning_rate,
    'best_val_loss': best_trial.value
}, save_path)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

best_model.eval()
with torch.no_grad():
    decoder_input_test = torch.zeros_like(y_test_tensor)
    preds = best_model(X_test_tensor, decoder_input_test)
fs_pred = preds[:, 0, :]
ls_pred = preds[:, -1, :]
fs_true = y_test_tensor[:, 0, :]
ls_true = y_test_tensor[:, -1, :]

fs_mse = torch.mean((fs_pred - fs_true) ** 2).item()
ls_mse = torch.mean((ls_pred - ls_true) ** 2).item()
fs_mae = torch.mean(torch.abs(fs_pred - fs_true)).item()
ls_mae = torch.mean(torch.abs(ls_pred - ls_true)).item()

print(f"First Step MSE: {fs_mse}, MAE: {fs_mae}")
print(f"Last Step MSE: {ls_mse}, MAE: {ls_mae}")
fs_pred_scaled = fs_pred.cpu().numpy()
fs_true_scaled = fs_true.cpu().numpy()
ls_pred_scaled = ls_pred.cpu().numpy()
ls_true_scaled = ls_true.cpu().numpy()

fs_pred_origin = scalertarget.inverse_transform(fs_pred_scaled)
fs_true_origin = scalertarget.inverse_transform(fs_true_scaled)
ls_pred_origin = scalertarget.inverse_transform(ls_pred_scaled)
ls_true_origin = scalertarget.inverse_transform(ls_true_scaled)

plt.figure(figsize=(10, 6))
plt.plot(fs_pred_origin[:, 0], 'r-', label='Predicted First Step')
plt.plot(fs_true_origin[:, 0], 'b-', label='Actual First Step')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('First Step Pred vs Actual (Temp)')
plt.legend()
plt.savefig('fs_temp.png')
plt.figure(figsize=(10, 6))
plt.plot(ls_pred_origin[:, 0], 'r-', label='Predicted Last Step')
plt.plot(ls_true_origin[:, 0], 'b-', label='Actual Last Step')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Last Step Pred vs Actual (Temp)')
plt.legend()
plt.savefig('ls_temp.png')

# ## Humi Part
# print("Humi Part")
# input_dim = 2
# output_dim = 1
# steps = 5
# tries = 1
# inputs = 30
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# df = pd.read_csv("dataAll.csv").sort_values(by='Time',ascending=True)
# data = df[['Temp', 'Humi']].values
# target = df['Humi'].shift(-steps).values[:-steps]
# scalerdata = MinMaxScaler()
# scalertarget = MinMaxScaler()
# data_scaled = scalerdata.fit_transform(data)
# target_scaled = scalertarget.fit_transform(target.reshape(-1, 1))
#
# X = []
# y = []
# for i in range(inputs, len(data_scaled)-steps*2):
#     X.append(data_scaled[i-inputs:i])
#     y.append(target_scaled[i:i+steps])
# X, y = np.array(X), np.array(y)
# train_size = int(len(X) * 0.6)
# val_size = int(len(X) * 0.2)
# test_size = len(X) - train_size - val_size
#
# X_train, y_train = X[:train_size], y[:train_size]
# X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
# X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
#
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
#
# train_data = TensorDataset(X_train_tensor, y_train_tensor)
# val_data = TensorDataset(X_val_tensor, y_val_tensor)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64)
#
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=tries)
# best_trial = study.best_trial
# print("The best hyperparameters: {}".format(best_trial.params))
# print("The best validation loss: {:.5f}".format(best_trial.value))
# best_hidden_dim, best_num_layers, best_dropout, best_learning_rate = extract(best_trial)
#
# best_model = CNN_LSTM(input_dim, best_hidden_dim, output_dim, best_num_layers, best_dropout).to(device)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(best_model.parameters(), lr=best_learning_rate)
# num_epochs = 100
# for epoch in range(num_epochs):
#     best_model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         decoder_input = torch.zeros_like(y_batch)
#         optimizer.zero_grad()
#         output = best_model(X_batch, decoder_input)
#         loss = loss_fn(output, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
# save_path = './best_save_humi.pth'
# torch.save({
#     'model_state_dict': best_model.state_dict(),
#     'hidden_dim': best_hidden_dim,
#     'num_layers': best_num_layers,
#     'dropout': best_dropout,
#     'learning_rate': best_learning_rate,
#     'best_val_loss': best_trial.value
# }, save_path)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
#
# best_model.eval()
# with torch.no_grad():
#     decoder_input_test = torch.zeros_like(y_test_tensor)
#     preds = best_model(X_test_tensor, decoder_input_test)
# fs_pred = preds[:, 0, :]
# ls_pred = preds[:, -1, :]
# fs_true = y_test_tensor[:, 0, :]
# ls_true = y_test_tensor[:, -1, :]
#
# fs_mse = torch.mean((fs_pred - fs_true) ** 2).item()
# ls_mse = torch.mean((ls_pred - ls_true) ** 2).item()
# fs_mae = torch.mean(torch.abs(fs_pred - fs_true)).item()
# ls_mae = torch.mean(torch.abs(ls_pred - ls_true)).item()
#
# print(f"First Step MSE: {fs_mse}, MAE: {fs_mae}")
# print(f"Last Step MSE: {ls_mse}, MAE: {ls_mae}")
# fs_pred_scaled = fs_pred.cpu().numpy()
# fs_true_scaled = fs_true.cpu().numpy()
# ls_pred_scaled = ls_pred.cpu().numpy()
# ls_true_scaled = ls_true.cpu().numpy()
#
# fs_pred_origin = scalertarget.inverse_transform(fs_pred_scaled)
# fs_true_origin = scalertarget.inverse_transform(fs_true_scaled)
# ls_pred_origin = scalertarget.inverse_transform(ls_pred_scaled)
# ls_true_origin = scalertarget.inverse_transform(ls_true_scaled)
#
# plt.figure(figsize=(10, 6))
# plt.plot(fs_pred_origin[:, 0], 'r-', label='Predicted First Step')
# plt.plot(fs_true_origin[:, 0], 'b-', label='Actual First Step')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('First Step Pred vs Actual (Humi)')
# plt.legend()
# plt.savefig('fs_humi.png')
# plt.figure(figsize=(10, 6))
# plt.plot(ls_pred_origin[:, 0], 'r-', label='Predicted Last Step')
# plt.plot(ls_true_origin[:, 0], 'b-', label='Actual Last Step')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('Last Step Pred vs Actual (Humi)')
# plt.legend()
# plt.savefig('ls_humi.png')
#
# ## Light Part
# print("Light Part")
# input_dim = 3
# output_dim = 1
# steps = 5
# inputs = 30
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# df = pd.read_csv("dataAll.csv").sort_values(by='Time',ascending=True)
# data = df[['Temp', 'Humi', 'Light']].values
# target = df['Light'].shift(-steps).values[:-steps]
# scalerdata = MinMaxScaler()
# scalertarget = MinMaxScaler()
# data_scaled = scalerdata.fit_transform(data)
# target_scaled = scalertarget.fit_transform(target.reshape(-1, 1))
#
# X = []
# y = []
# for i in range(inputs, len(data_scaled)-steps*2):
#     X.append(data_scaled[i-inputs:i])
#     y.append(target_scaled[i:i+steps])
# X, y = np.array(X), np.array(y)
# train_size = int(len(X) * 0.6)
# val_size = int(len(X) * 0.2)
# test_size = len(X) - train_size - val_size
#
# X_train, y_train = X[:train_size], y[:train_size]
# X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
# X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
#
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
#
# train_data = TensorDataset(X_train_tensor, y_train_tensor)
# val_data = TensorDataset(X_val_tensor, y_val_tensor)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=64)
#
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=tries)
# best_trial = study.best_trial
# print("The best hyperparameters: {}".format(best_trial.params))
# print("The best validation loss: {:.5f}".format(best_trial.value))
# best_hidden_dim, best_num_layers, best_dropout, best_learning_rate = extract(best_trial)
#
# best_model = CNN_LSTM(input_dim, best_hidden_dim, output_dim, best_num_layers, best_dropout).to(device)
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(best_model.parameters(), lr=best_learning_rate)
# num_epochs = 100
# for epoch in range(num_epochs):
#     best_model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         decoder_input = torch.zeros_like(y_batch)
#         optimizer.zero_grad()
#         output = best_model(X_batch, decoder_input)
#         loss = loss_fn(output, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")
# save_path = './best_save_light.pth'
# torch.save({
#     'model_state_dict': best_model.state_dict(),
#     'hidden_dim': best_hidden_dim,
#     'num_layers': best_num_layers,
#     'dropout': best_dropout,
#     'learning_rate': best_learning_rate,
#     'best_val_loss': best_trial.value
# }, save_path)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
#
# best_model.eval()
# with torch.no_grad():
#     decoder_input_test = torch.zeros_like(y_test_tensor)
#     preds = best_model(X_test_tensor, decoder_input_test)
# fs_pred = preds[:, 0, :]
# ls_pred = preds[:, -1, :]
# fs_true = y_test_tensor[:, 0, :]
# ls_true = y_test_tensor[:, -1, :]
#
# fs_mse = torch.mean((fs_pred - fs_true) ** 2).item()
# ls_mse = torch.mean((ls_pred - ls_true) ** 2).item()
# fs_mae = torch.mean(torch.abs(fs_pred - fs_true)).item()
# ls_mae = torch.mean(torch.abs(ls_pred - ls_true)).item()
#
# print(f"First Step MSE: {fs_mse}, MAE: {fs_mae}")
# print(f"Last Step MSE: {ls_mse}, MAE: {ls_mae}")
# fs_pred_scaled = fs_pred.cpu().numpy()
# fs_true_scaled = fs_true.cpu().numpy()
# ls_pred_scaled = ls_pred.cpu().numpy()
# ls_true_scaled = ls_true.cpu().numpy()
#
# fs_pred_origin = scalertarget.inverse_transform(fs_pred_scaled)
# fs_true_origin = scalertarget.inverse_transform(fs_true_scaled)
# ls_pred_origin = scalertarget.inverse_transform(ls_pred_scaled)
# ls_true_origin = scalertarget.inverse_transform(ls_true_scaled)
#
# plt.figure(figsize=(10, 6))
# plt.plot(fs_pred_origin[:, 0], 'r-', label='Predicted First Step')
# plt.plot(fs_true_origin[:, 0], 'b-', label='Actual First Step')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('First Step Pred vs Actual (Light)')
# plt.legend()
# plt.savefig('fs_light.png')
# plt.figure(figsize=(10, 6))
# plt.plot(ls_pred_origin[:, 0], 'r-', label='Predicted Last Step')
# plt.plot(ls_true_origin[:, 0], 'b-', label='Actual Last Step')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('Last Step Pred vs Actual (Light)')
# plt.legend()
# plt.savefig('ls_light.png')
# plt.show()
