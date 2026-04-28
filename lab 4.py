# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import fetch_california_housing instead of load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset - Using California Housing as suggested by the error message
housing= fetch_california_housing()
X=pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal') # Target name is 'MedHouseVal' in California Housing

# Select relevant features
# Adjust feature selection for California Housing dataset.
# We'll use MedInc (Median Income~ a proxy for wealth/location), HouseAge, and AveRooms
#(Average number of rooms ~ related to size)
X_selected =X[['MedInc', 'HouseAge','AveRooms']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train-test split
X_train, X_test, y_train, y_test=train_test_split(X_scaled,y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred=model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:",r2_score(y_test, y_pred))

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, edgecolors='k', alpha=0.7)
# Adjust plot range for California Housing data if necessary, or keep dynamic
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)],'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("California Housing - Actual vs Predicted Prices") # Update title
plt.grid(True)
plt.show()