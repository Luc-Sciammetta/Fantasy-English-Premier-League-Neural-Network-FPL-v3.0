import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import joblib

#a class that represents the dataset
class FPLDataset(Dataset):
    def __init__(self, X, y):
        self.X = X #x features
        self.y = y #what we want to predict

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FPLModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 1), 
        )
    
    def forward(self, x):
        return self.network(x)


def getDataFromCSV(filename):
    return pd.read_csv(filename)


def main():
    data = getDataFromCSV('get_historical_data/oneGW_fpl_training_data.csv')

    x_data = data.drop('actual_gw_points', axis=1)
    y_data = data['actual_gw_points']

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=(1 - train_ratio), random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    #columns we don't want to standardize
    binary_columns = (
        [col for col in x_train.columns if col.startswith('position_')] +
        ['home_away_current'] 
    )

    numerical_columns = [col for col in x_train.columns if col not in binary_columns] #columns we do want to scale
    scaler = StandardScaler() #scalar object
    #applies the scaling (Note that 'fit' gets the mean and std, so x_val and x_test don't need them since those values are reused)
    x_train[numerical_columns] = scaler.fit_transform(x_train[numerical_columns])
    x_val[numerical_columns] = scaler.transform(x_val[numerical_columns])
    x_test[numerical_columns] = scaler.transform(x_test[numerical_columns])

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    #create the individual datasets
    train_dataset = FPLDataset(x_train_tensor, y_train_tensor)
    val_dataset = FPLDataset(x_val_tensor, y_val_tensor)
    test_dataset = FPLDataset(x_test_tensor, y_test_tensor)

    #create the loaders that loads the data into the model. 48 'lines' will be inputted at a time, with all lines being randomly shuffled
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

    model = FPLModel(input_size=x_train_tensor.shape[1])
    loss_function = nn.HuberLoss() #HuberLoss works like this: for small errors it behaves like MSE, for large errors it behaves like MAE
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    """
    The gradient tells you for each weight how much to to change it to minimize the loss.
    It is basically a derrivative that depends on the layer after it (hense the backward pass)
    """
    num_epochs = 100

    #for early stopping
    best_val_loss = float('inf')
    patience = 15
    epochs_without_improvement = 0

    #for learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    for epoch in range(num_epochs):
        #training
        model.train() #sets the model to train mode (dropout active)
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad() #clears the gradients from the previous batch
            predictions = model(x_batch) #produces predictions
            loss = loss_function(predictions, y_batch) 
            loss.backward() #backwards pass, computes gradients for every weight
            optimizer.step() #uses the gradients to update the weights
            train_loss += loss.item() 

        #evaluation on data it's 'never' seen before
        model.eval() #sets the model to evaluation mode (no dropout)
        val_loss = 0
        with torch.no_grad(): #see above comment
            for x_batch, y_batch in val_loader:
                predictions = model(x_batch)
                loss = loss_function(predictions, y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        #update the learning rate
        scheduler.step(val_loss)

        #early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "points_predictors/misc/best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    #test the model
    model.load_state_dict(torch.load('points_predictors/misc/best_model.pth'))
    torch.save(model.state_dict(), "points_predictors/misc/fpl_model.pth")

    model.eval()
    with torch.no_grad():
        predictions = model(x_test_tensor)
        test_loss = loss_function(predictions, y_test_tensor)

        print(f"Test Loss: {test_loss.item():.4f}")

        preds = predictions.numpy() #convert to numpy array for mae
        actuals = y_test_tensor.numpy() 
        
    # Calculate and print the Mean Absolute Error
    mae = mean_absolute_error(actuals, preds)
    print(f"Mean Absolute Error: {mae:.4f} points")

    torch.save(model.state_dict(), f'points_predictors/one_gw_{mae:.4f}_best_model.pth')
    joblib.dump(scaler, f'points_predictors/one_gw_{mae:.4f}_scaler.pkl')  #save the scalers for later use

    # #load the model
    # model = FPLModel(input_size=x_train_tensor.shape[1])
    # model.load_state_dict(torch.load('fpl_model.pth'))

if __name__ == '__main__':
    for i in range(0, 10):
        main()