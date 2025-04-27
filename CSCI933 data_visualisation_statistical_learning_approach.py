import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


## Data visualisation 
import seaborn as sns  

def load_data(file_path):  
    """Load the bike rental dataset"""  
    return pd.read_csv("bike_rental_data.csv")  

def create_correlation_heatmap(df, output_path='images/correlation_heatmap.png'):  
    """Create and save correlation heatmap"""  
    plt.figure(figsize=(12, 10))  
    correlation_matrix = df.corr()  
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)  
    plt.title('Feature Correlation Heatmap')  
    plt.tight_layout()  
    plt.savefig(output_path)  
    plt.close()  

def plot_feature_distributions(df, output_path='images/feature_distributions.png'):  
    """Plot distributions of all numerical features"""  
    plt.figure(figsize=(15, 10))  
    df.hist(bins=50, figsize=(20,15))  
    plt.suptitle('Feature Distributions')  
    plt.tight_layout()  
    plt.savefig(output_path)  
    plt.close()  

# def plot_bikes_by_hour(df, output_path='images/bikes_by_hour.png'):  
#     """Scatter plot of bikes rented by hour"""  
#     plt.figure(figsize=(12, 6))  
#     plt.scatter(df['hour'], df['bikes_rented'], alpha=0.1)  
#     plt.title('Bikes Rented by Hour of Day')  
#     plt.xlabel('Hour of Day')  
#     plt.ylabel('Number of Bikes Rented')  
#     plt.tight_layout()  
#     plt.savefig(output_path)  
#     plt.close()  

df = pd.read_csv("bike_rental_data.csv")
hourly_avg = df.groupby("hour")["bikes_rented"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o')
plt.title("Average Bikes Rented by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Rentals")
plt.grid(True)
plt.tight_layout()
plt.savefig("images/avg_bikes_by_hour.png")
plt.show()

def plot_bikes_by_season(df, output_path='images/bikes_by_season.png'):  
    """Box plot of bikes rented by season"""  
    plt.figure(figsize=(12, 6))  
    sns.boxplot(x='season', y='bikes_rented', data=df)  
    plt.title('Bikes Rented by Season')  
    plt.xlabel('Season (1:Winter, 2:Spring, 3:Summer, 4:Autumn)')  
    plt.ylabel('Number of Bikes Rented')  
    plt.tight_layout()  
    plt.savefig(output_path)  
    plt.close()  


def main():  
    # Load data  
    df = load_data('data/bike_rental_data.csv')  
    
    # Generate visualizations  
    create_correlation_heatmap(df)  
    plot_feature_distributions(df)  
 #  plot_bikes_by_hour(df)  
    plot_bikes_by_season(df)  
    
    # Print descriptive statistics  
    print("Dataset Descriptive Statistics:")  
    print(df.describe())  

if __name__ == "__main__":  
    main()  


## Statistical learning approach 
# Load dataset
df = pd.read_csv("bike_rental_data.csv")

# Features and target
X = df.drop(columns=["bikes_rented"])
y = df["bikes_rented"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# 2. Ridge Regression (Cross-Validation) L2 regularisation
print(">> Fitting RidgeCV...")
ridge_cv = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 30)}, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge = ridge_cv.best_estimator_
print("Best alpha for Ridge Regression:", ridge_cv.best_params_["alpha"])

# 3. Lasso Regression (Crosss-Validation) L1 regularisation
lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': np.logspace(-3, 1, 30)}, cv=5)
lasso_cv.fit(X_train_scaled, y_train)
lasso = lasso_cv.best_estimator_
print("Best alpha for Lasso Regression:", lasso_cv.best_params_["alpha"])

# 4. Elastic Net Regression (Cross-Validation)
elastic_cv = GridSearchCV(
    ElasticNet(max_iter=10000),
    {'alpha': np.logspace(-3, 1, 10), 'l1_ratio': np.linspace(0.1, 1.0, 10)}, #controls the balance between L1 and L2
    cv=5
)
elastic_cv.fit(X_train_scaled, y_train)
elastic_net = elastic_cv.best_estimator_
print("Best alpha for Elastic Net:", elastic_cv.best_params_["alpha"])
print("Best l1_ratio for Elastic Net:", elastic_cv.best_params_["l1_ratio"])

# Evaluation
def evaluate(model, X, y):
    y_pred = model.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred)), r2_score(y, y_pred)

models = [lin_reg, ridge, lasso, elastic_net]
names = ["Linear", "Ridge", "Lasso", "ElasticNet"]

# Results table
results = []
for name, model in zip(names, models):
    rmse, r2 = evaluate(model, X_test_scaled, y_test)
    results.append([name, rmse, r2])
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R²"])
print(results_df)

# Plot coefficients
plt.figure(figsize=(10, 6))
# for name, model in zip(names, models):
#     plt.plot(model.coef_, label=name)
for name, model in zip(names, models):
    plt.plot(model.coef_, marker='o', label=name)
plt.xticks(ticks=np.arange(len(X.columns)), labels=X.columns, rotation=45)
plt.ylabel("Coefficient Value")
plt.title("Feature Coefficients by Model")
plt.legend()
plt.tight_layout()
plt.savefig("images/model_coefficients.png")
plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=len(X.columns), figsize=(18, 4), sharey=True)
# for i, feature in enumerate(X.columns):
#     for name, model in zip(names, models):
#         axes[i].bar(name, model.coef_[i])
#     axes[i].set_title(feature)
#     axes[i].tick_params(axis='x', rotation=45)

# fig.suptitle("Coefficient Comparison per Feature")
# plt.tight_layout()
# plt.savefig("images/model_coefficients_subplots.png")
# plt.show()

# Print coefficients for detailed comparison
for name, model in zip(names, models):
    print(f"\n{name} Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature:>12}: {coef:.4f}")