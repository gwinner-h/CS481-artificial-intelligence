import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import warnings

# Load the data from a csv
def loadData(fname):    
    print("Loading Data..........")
    arr = np.loadtxt(fname, delimiter=",", dtype=str)
    headers = arr[0, :]
    data = np.delete(arr, 0, 0)
    data = data.astype(float)
    
    # Print information about the data loaded
    print()
    print(f"Feature Names: \n{headers}\n")
    print(f"Data Loaded: \n{data}\n")
    print(f"Target Data: \n{data[:, -1]}\n")
    print(f"Correlation Coefficients Table:\n{np.corrcoef(data)}\n")
    print(f"Target Name: {headers[-1]}")
    print(f"Data Shape: {arr.shape}")
    print("\nData Loaded!")
    print()
        
    return data

# Load the data
data = loadData('')
rows, cols = data.shape

# Split the data into training and test sets
train, test = train_test_split(data, random_state=42, test_size=.2)
trainX = train[:, :-1]
trainY = train[:, -1]
testX  =  test[:, :-1]
testY  =  test[:, -1]

# Create a dict to store alpha values used for regression
gridSearchParams = {'alpha': [0, 0.1, 1.0, 10.0, 20.0, 50.0, 100.0]}



# Search for the best alpha values
# Suppress warnings stemming from the use of alpha = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # Perform ridge grid search 
    ridgeAlphaSearch = GridSearchCV(Ridge(random_state=42), gridSearchParams, cv=5)
    ridgeAlphaSearch.fit(trainX, trainY)

    # Perform lasso grid search 
    lassoAlphaSearch = GridSearchCV(Lasso(random_state=42), gridSearchParams, cv=5)
    lassoAlphaSearch.fit(trainX, trainY)
    


# Store the best alphas
bestAlphaRidge = ridgeAlphaSearch.best_params_['alpha']
bestAlphaLasso = lassoAlphaSearch.best_params_['alpha']

# Train the models
ridgeReg = Ridge(bestAlphaRidge, random_state=42)
ridgeReg.fit(trainX, trainY)

lassoReg = Lasso(bestAlphaLasso, random_state=42)
lassoReg.fit(trainX, trainY)

linearReg = LinearRegression()
linearReg.fit(trainX, trainY)

# Predict on the test set
ridgePredictions = ridgeReg.predict(testX)
lassoPredictions = lassoReg.predict(testX)
linearPredictions = linearReg.predict(testX)



# Calculate R2 and RMSE values
ridge_r2 = r2_score(testY, ridgePredictions)
ridge_rmse = sqrt(mean_squared_error(testY, ridgePredictions))
lasso_r2 = r2_score(testY, lassoPredictions)
lasso_rmse = sqrt(mean_squared_error(testY, lassoPredictions))
linear_r2 = r2_score(testY, linearPredictions)
linear_rmse = sqrt(mean_squared_error(testY, linearPredictions))

# Print the metrics
print(f"Best alpha for Ridge Regression: {bestAlphaRidge}")
print(f"Ridge Beta Coefficients: {ridgeReg.coef_}")
print(f"R2 for Ridge Regression: {ridge_r2:.4f}")
print(f"RMSE for Ridge Regression: {ridge_rmse:.4f}\n")

print(f"Best alpha for Lasso Regression: {bestAlphaLasso}")
print(f"Ridge Beta Coefficients: {lassoReg.coef_}")
print(f"R2 for Lasso Regression: {lasso_r2:.4f}")
print(f"RMSE for Lasso Regression: {lasso_rmse:.4f}\n")

print(f"R2 for Linear Regression: {linear_r2:.4f}")
print(f"RMSE for Linear Regression: {linear_rmse:.4f}")



# Plot the coefficients
fig, fig1 = plt.subplots()
fig1.set_title("Coefficient Plot") 
fig1.set_xlabel("Coefficient Index") 
fig1.set_ylabel("Coefficient Magnitude")
fig1.set_ylim(-1, 1)
fig1.axhline(y=0, color='k', linestyle='--', label="y=0") 
indexes = np.linspace(1, 6, 6)
fig1.scatter(indexes, ridgeReg.coef_, marker="s", label=f"Ridge with Alpha = {bestAlphaRidge}")
fig1.scatter(indexes, lassoReg.coef_, marker="o", label=f"Lasso with Alpha = {bestAlphaLasso}")
fig1.scatter(indexes, linearReg.coef_, marker="^", label="Linear")
fig1.legend()