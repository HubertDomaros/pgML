import kagglehub
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import dython
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Download dataset
path = kagglehub.dataset_download("arashnic/dynamic-pricing-dataset")
src = pd.read_csv(os.path.join(path, "dynamic_pricing.csv"))
src.rename(columns={
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

print(f"Dataset info:\n"
      f"shape: \n{src.shape}\n"
      f"null values: \n{src.isnull().sum()}\n"
      f"dtypes: \n{src.dtypes}")
# No null values. Nice.

# Number of Drivers and Riders separately doesn't give us any info.
# Let's combine them by creating ratio.
src['RidersPerDriver'] = pd.DataFrame(src['Riders'] / src['Drivers'],
                                      columns=['RidersPerDriver'])
src.drop(columns=['Riders', 'Drivers'], inplace=True)

# Let's see distributions of numerical columns. We'll use KDE plots for this.
numerical_columns = src.select_dtypes(include=['float64', 'int']).columns
# Create subplots
fig, axes = plt.subplots(nrows=len(numerical_columns), ncols=1, figsize=(10, 15))
for col, ax in zip(numerical_columns, axes):
    sns.kdeplot(data=src, x=col, ax=ax, fill=True)
    ax.set_title(f'KDE of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
# Adjusting layout to prevent overlap
plt.tight_layout()
plt.show()
# We can see few things here:
# 1. 


dython.nominal.associations(src, plot=True, cmap='coolwarm')

# Duration and Cost have very high correlation (0.93). We will do PCA on these components.
scaler = StandardScaler()
src[['Duration', 'Cost']] = scaler.fit_transform(src[['Duration', 'Cost']])
pca = PCA(n_components=1)
pca_results = pca.fit_transform(src[['Duration', 'Cost']])
src.drop(columns=['Duration', 'Cost'], inplace=True)
src['DurationCostPCA'] = pca_results[:, 0]

dython.nominal.associations(src, plot=True, cmap='coolwarm')
plt.show()



src['Loyalty'] = src['Loyalty'].map({
    'Regular': -1,
    'Silver': 0,
    'Gold': 1
}).astype(int)

label_encoder = LabelEncoder()
categorical_columns = src.select_dtypes(include=['object']).columns.values
for col in categorical_columns:
    src[col] = LabelEncoder().fit_transform(src[col])

import xgboost

xgb_model = xgboost.XGBRegressor()
feature_names = np.array(src.drop(columns=['DurationCostPCA']).columns)
X = src.drop(columns=['DurationCostPCA']).to_numpy()
y = src['DurationCostPCA'].to_numpy()
xgb_model.fit(X, y)

xgb_model.predict(X)
score = xgb_model.score(X, y)
print(score)
for name, importance in zip(feature_names, xgb_model.feature_importances_):
    print(f'Feature: {name}, Importance: {importance}')

# Reduce skewness of RidersPerDriver using a log transformation and re-run the model

# Transform RidersPerDriver to reduce skewness
src['RidersPerDriver'] = np.log1p(src['RidersPerDriver'])
# Plot KDE for RidersPerDriver
plt.figure(figsize=(8, 6))
sns.kdeplot(data=src, x='RidersPerDriver', fill=True)
plt.title('KDE of RidersPerDriver')
plt.xlabel('RidersPerDriver')
plt.ylabel('Density')
plt.show()

# Recalculate the model
X = src.drop(columns=['DurationCostPCA']).to_numpy()
y = src['DurationCostPCA'].to_numpy()

# Re-fit the model
xgb_model.fit(X, y)

# Predict and score the model
xgb_model.predict(X)
new_score = xgb_model.score(X, y)
print(f'Score after transformation: {new_score}')

# Print feature importances
for name, importance in zip(feature_names, xgb_model.feature_importances_):
    print(f'Feature: {name}, Importance: {importance}')

print(score, new_score, score - new_score)