import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.svm import SVR


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
    print(f"Correlation Coefficients Table:\n{np.corrcoef(np.transpose(data))}\n")
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

# Plot some properties of the training data
# Histogram of the LotFrontage
fig, fig1 = plt.subplots()
fig1.set_title("LotFrontage Measurements") 
fig1.set_xlabel('LotFrontage')
fig1.set_ylabel('Frequency')
fig1.hist(train[:,0], bins = 20, color = "blue", edgecolor = "black")
plt.show()

# Histogram of the LotArea
fig, fig2 = plt.subplots()
fig2.set_title("LotArea Measurements") 
fig2.set_xlabel('LotArea')
fig2.set_ylabel('Frequency')
fig2.hist(train[:,1], bins = 20,color = "blue",  edgecolor = "black")
plt.show()

# Find the threshold of bottom 30% correlation
correlations = np.corrcoef(np.transpose(data))[-1,:]
threshold = np.percentile(correlations, 30)

print("\nRemoving least correlated feartures...")
numRemoved = 0
# drop 30% least correlated columns
for i in range(len(correlations) - 1, -1, -1):
    if correlations[i] <= threshold:
        data = np.delete(data, i, axis=1)
        numRemoved += 1

rows, cols = data.shape
print(f"Threshold for removal: {threshold}")
print(f"{numRemoved} columns (bottom 30%) were dropped from the array, leaving {cols} columns.")

# Split the data into training and test sets
train, test = train_test_split(data, random_state=42, test_size=.2)
trainX = train[:, :-1]
trainY = train[:, -1]
testX  =  test[:, :-1]
testY  =  test[:, -1]

# Train Lasso Models to 
lassoReg1 = Lasso(1, random_state=42)
lassoReg1.fit(trainX, trainY)

lassoReg5 = Lasso(5, random_state=42)
lassoReg5.fit(trainX, trainY)
# Alpha 10 is the best alpha value
lassoReg10 = Lasso(10, random_state=42)
lassoReg10.fit(trainX, trainY)

# Plot the coefficients
fig, fig1 = plt.subplots()
fig1.set_title("Coefficient Plot") 
fig1.set_xlabel("Coefficient Index") 
fig1.set_ylabel("Coefficient Magnitude")
fig1.set_ylim(-50000, 50000)
fig1.axhline(y=0, color='k', linestyle='--', label="y=0") 
indexes = np.linspace(1, 169, 169)
fig1.scatter(indexes, lassoReg1.coef_, marker="o", label="Lasso with Alpha = 1")
fig1.scatter(indexes, lassoReg5.coef_, marker="^", label="Lasso with Alpha = 5")
fig1.scatter(indexes, lassoReg10.coef_, marker="s", label="Lasso with Alpha = 10")
fig1.legend()


threshold = np.percentile(lassoReg10.coef_, 70)

print("\nRemoving feartures with highest beta values...")
numRemoved = 0
# drop 30% columns with the highest beta values
for i in range(len(lassoReg10.coef_) - 1, -1, -1):
    if lassoReg10.coef_[i] >= threshold:
        data = np.delete(data, i, axis=1)
        numRemoved += 1

rows, cols = data.shape
print(f"Threshold for removal: {threshold}")
print(f"{numRemoved} columns (top 30%) were dropped from the array, leaving {cols} columns.")

# Split the data into training and test sets
train, test = train_test_split(data, random_state=42, test_size=.2)
trainX = train[:, :-1]
trainY = train[:, -1]
testX  =  test[:, :-1]
testY  =  test[:, -1]

# Neural Network
nn_model = MLPClassifier(max_iter=1000, random_state=42)

# Define the parameter grid for Neural Network
param_grid_nn = {'hidden_layer_sizes': [(x, y) for x in range(5, 16) for y in range(5, 16)]}

# Perform grid search with cross-validation
nn_grid_search = GridSearchCV(nn_model, param_grid_nn, cv=5)
nn_grid_search.fit(trainX, trainY)

# Get the best parameters
best_nn_params = nn_grid_search.best_params_

print("Best Neural Network Parameters:", best_nn_params)


# SVM with RBF Kernel
svm_model = SVR(kernel='rbf')

# Define the parameter grid for SVM
param_grid_svm = {'C': range(10, 110, 10), 'gamma': np.arange(1, 11, 5)}

# Perform grid search with cross-validation
svm_grid_search = GridSearchCV(svm_model, param_grid_svm, cv=5)
svm_grid_search.fit(trainX, trainY)

# Get the best parameters
best_svm_params = svm_grid_search.best_params_

print("Best SVM Parameters:", best_svm_params)

bestSVM = SVR(C=100, gamma=1, kernel='rbf')
bestSVM.fit(trainX, trainY)