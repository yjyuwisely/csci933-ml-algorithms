import pandas as pd  
import matplotlib.pyplot as plt  
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