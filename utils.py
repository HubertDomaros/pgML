from typing import Dict, Any

import kagglehub
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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

def print_feature_importances(model: xgboost.XGBRegressor, feature_names: list[str]) -> None:
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f'Feature: {name}, Importance: {importance}')


def train_and_evaluate_model(data: pd.DataFrame, target_column: str, test_size: float = 0.2,
                             random_state: int = 42) -> dict[str, float | dict[str, Any] | Any]:
    # Separating features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Splitting dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initializing and fitting the XGBoost Regressor
    model = xgboost.XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict(X_test)

    # Calculating and printing metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    feature_importances = {'feature': X.columns.tolist()}
    return {"mse": mse, "r2": r2}
