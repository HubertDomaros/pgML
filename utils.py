import kagglehub
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def download_and_prepare_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("arashnic/dynamic-pricing-dataset")
    data = pd.read_csv(os.path.join(path, "dynamic_pricing.csv"))
    data.rename(columns={
        'Number_of_Riders': 'Riders',
        'Number_of_Drivers': 'Drivers',
        'Location_Category': 'Location',
        'Customer_Loyalty_Status': 'Loyalty',
        'Number_of_Past_Rides': 'PastRides',
        'Average_Ratings': 'Ratings',
        'Time_of_Booking': 'BookingTime',
        'Vehicle_Type': 'Vehicle',
        'Expected_Ride_Duration': 'Duration',
        'Historical_Cost_of_Ride': 'Cost'
    }, inplace=True)
    return data

def plot_z_scored_num_distributions(data: pd.DataFrame) -> None:
    numerical_columns = data.select_dtypes(include=['float64', 'int']).columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    fig, axes = plt.subplots(nrows=len(numerical_columns), ncols=1, figsize=(10, 15))
    for col, ax in zip(numerical_columns, axes):
        sns.kdeplot(data=data, x=col, ax=ax, fill=True)
        ax.set_title(f'Z-Scored KDE of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
    plt.tight_layout()
    plt.show()