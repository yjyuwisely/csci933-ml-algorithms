import pandas as pd  
import numpy as np  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, r2_score  

# 1. Data Preparation  
def prepare_data(add_feature_engineering=True):  
    # Load dataset  
    df = pd.read_csv("bike_rental_data.csv")  
    
    # Feature Engineering 
    if add_feature_engineering:  
        df['temp_squared'] = df['temp'] ** 2  
        df['hour_temp'] = df['hour'] * df['temp']  
        df['humidity_squared'] = df['humidity'] ** 2  
    
    # Define features and target  
    X = df.drop(columns=["bikes_rented"])  
    y = df["bikes_rented"]  
    
    # Train-test split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    
    # Standardize features  
    scaler = StandardScaler()  
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)  
    
    # Convert to PyTorch tensors  
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)  
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)  
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor  

# 2. Neural Network Models  
class ImprovedLinearNN(nn.Module):  
    def __init__(self, input_dim):  
        super(ImprovedLinearNN, self).__init__()  
        self.layers = nn.Sequential(  
            nn.Linear(input_dim, 64),  
            nn.BatchNorm1d(64),  
            nn.ReLU(),  
            nn.Linear(64, 32),  
            nn.BatchNorm1d(32),  
            nn.ReLU(),  
            nn.Linear(32, 1)  
        )  

    def forward(self, x):  
        return self.layers(x)  

class ImprovedDropoutNN(nn.Module):  
    def __init__(self, input_dim, dropout_rates=[0.2, 0.5]):  
        super(ImprovedDropoutNN, self).__init__()  
        self.layers = nn.Sequential(  
            nn.Linear(input_dim, 64),  
            nn.BatchNorm1d(64),  
            nn.ReLU(),  
            nn.Dropout(dropout_rates[0]),  
            nn.Linear(64, 32),  
            nn.BatchNorm1d(32),  
            nn.ReLU(),  
            nn.Dropout(dropout_rates[1]),  
            nn.Linear(32, 1)  
        )  

    def forward(self, x):  
        return self.layers(x)  

# 3. Training and Evaluation Function  
def train_and_evaluate(model, optimizer, X_train, y_train, X_test, y_test,   
                       patience=10, epochs=500, verbose=False):  
    criterion = nn.MSELoss()  
    best_loss = float('inf')  
    patience_counter = 0  
    
    for epoch in range(epochs):  
        model.train()  
        optimizer.zero_grad()  
        y_pred = model(X_train)  
        loss = criterion(y_pred, y_train)  
        loss.backward()  
        optimizer.step()  
        
        # Early stopping  
        if loss.item() < best_loss:  
            best_loss = loss.item()  
            patience_counter = 0  
        else:  
            patience_counter += 1  
        
        if patience_counter >= patience:  
            if verbose:  
                print(f"Early stopping at epoch {epoch}")  
            break  

    model.eval()  
    with torch.no_grad():  
        y_pred_tensor = model(X_test).numpy()  
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_tensor))  
    r2 = r2_score(y_test, y_pred_tensor)  
    return rmse, r2  

# 4. Experiment Function  
def run_deep_learning_experiments():  
    # Prepare data with and without feature engineering  
    X_train_fe, y_train_fe, X_test_fe, y_test_fe = prepare_data(add_feature_engineering=True)  
    X_train_no_fe, y_train_no_fe, X_test_no_fe, y_test_no_fe = prepare_data(add_feature_engineering=False)  
    
    # Experiments with different regularization techniques  
    results = []  
    
    # 1. Weight Decay Experiment  
    weight_decays = [0, 0.0001, 0.001, 0.01, 0.1]  
    for wd in weight_decays:  
        model = ImprovedLinearNN(X_train_fe.shape[1])  
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)  
        rmse, r2 = train_and_evaluate(model, optimizer, X_train_fe, y_train_fe, X_test_fe, y_test_fe)  
        results.append({  
            "Experiment": f"Weight Decay {wd}",  
            "RMSE": rmse,  
            "R²": r2  
        })  
    
    # 2. Dropout Rates Experiment  
    dropout_rates = [0.2, 0.5, 0.7]  
    for p in dropout_rates:  
        model = ImprovedDropoutNN(X_train_fe.shape[1], [p, p])  
        optimizer = optim.Adam(model.parameters(), lr=0.001)  
        rmse, r2 = train_and_evaluate(model, optimizer, X_train_fe, y_train_fe, X_test_fe, y_test_fe)  
        results.append({  
            "Experiment": f"Dropout Rate {p}",  
            "RMSE": rmse,  
            "R²": r2  
        })  
    
    # 3. Feature Engineering Comparison  
    model_fe = ImprovedLinearNN(X_train_fe.shape[1])  
    optimizer_fe = optim.Adam(model_fe.parameters(), lr=0.001, weight_decay=0.01)  
    rmse_fe, r2_fe = train_and_evaluate(model_fe, optimizer_fe, X_train_fe, y_train_fe, X_test_fe, y_test_fe)  
    
    model_no_fe = ImprovedLinearNN(X_train_no_fe.shape[1])  
    optimizer_no_fe = optim.Adam(model_no_fe.parameters(), lr=0.001, weight_decay=0.01)  
    rmse_no_fe, r2_no_fe = train_and_evaluate(model_no_fe, optimizer_no_fe, X_train_no_fe, y_train_no_fe, X_test_no_fe, y_test_no_fe)  
    
    results.append({  
        "Experiment": "With Feature Engineering",  
        "RMSE": rmse_fe,  
        "R²": r2_fe  
    })  
    results.append({  
        "Experiment": "Without Feature Engineering",  
        "RMSE": rmse_no_fe,  
        "R²": r2_no_fe  
    })  
    
    # Convert results to DataFrame  
    results_df = pd.DataFrame(results)  
    print(results_df)  
    return results_df  

# 5. Main Execution  
if __name__ == "__main__":  
    # Set random seeds for reproducibility  
    torch.manual_seed(42)  
    np.random.seed(42)  
    
    # Run experiments  
    run_deep_learning_experiments()  
