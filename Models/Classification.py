import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the data from a csv
def loadData(fname):    
    arr = np.loadtxt(fname, delimiter=",", dtype=str)
    headers = arr[0, :]
    data = np.delete(arr, 0, 0)
    
    # Print information about the data loaded
    print()
    print(f"Feature Names: \n{headers}")
    print(f"Data Loaded: \n{data}")
    print(f"Target Data: \n{data[:, -1]}\n")
    print(f"Target Name: {headers[-1]}")
    print(f"Data Shape: {arr.shape}")
    print()
    
    return data.astype(float)


# Load the data
data = loadData('./ConnellLozonData.csv')
rows, cols = data.shape

# Split the data into training and test sets
train, test = train_test_split(data, random_state=42, test_size=.2)
trainX = train[:, :-1]
trainY = train[:, -1]
testX  =  test[:, :-1]
testY  =  test[:, -1]



# Plot some properties of the training data
# Histogram of the radius1 measurements
fig, fig1 = plt.subplots()
fig1.set_title("Radius 1 Measurements") 
fig1.set_xlabel('Radius 1')
fig1.set_ylabel('Frequency')
fig1.hist(train[:,0], bins = 20, color = "blue", edgecolor = "black")
plt.show()

# Histogram of the radius1 measurements
fig, fig2 = plt.subplots()
fig2.set_title("Texture 1 Measurements") 
fig2.set_xlabel('Texture 1')
fig2.set_ylabel('Frequency')
fig2.hist(train[:,1], bins = 20,color = "blue",  edgecolor = "black")
plt.show()

# Scatterplot of some influential properties
# Split the array into malignant and benign diagnoses
mask = train[:, -1] == 1
malignant = train[mask]
benign = train[~mask]
# Create the plot
fig, fig3 = plt.subplots()
fig3.set_title('Benign vs Malignant Tumors')
fig3.set_xlabel("Tumor Area 1")
fig3.set_ylabel("Concave Points 1")
fig3.scatter(benign[:, 3], benign[:, 7], label='Benign Tumors', color='blue', marker='o')
fig3.scatter(malignant[:, 3], malignant[:, 7], label='Malignant Tumors', color='red', marker='o')
fig3.legend()
plt.show()



# Create variables to store model data
neighborValues = []
classificationTestAccuracies = []
classificationTrainAccuracies = []
bestNN = 0
bestAcc = 1
bestClassifier = None

# Classify the data
for neighbors in range(1, int(np.sqrt(rows) + 3), 2):
    # Create a classifier and train it with current NN value
    classifier = kn(n_neighbors=neighbors)
    classifier.fit(trainX, trainY)

    # Calculate and store this model's accuracy with the NN used
    neighborValues.append(neighbors)
    trainAcc = accuracy_score(trainY, classifier.predict(trainX))
    classificationTrainAccuracies.append(trainAcc)
    testAcc = accuracy_score(testY, classifier.predict(testX))
    classificationTestAccuracies.append(testAcc)
    
    # If the current NN value is the best, overwrite the previous best
    diff = abs(trainAcc-testAcc)
    if(diff < bestAcc):
        bestAcc = diff
        bestNN = neighbors
        bestClassifier = classifier

    # Print the accuracy of each NN value and note the best one
    print(f"Neighbors: {neighbors}, ",
          f"Training Accuracy: {trainAcc:.3f} ", 
          f"Test Accuracy: {testAcc:.3f}")
print(f"Best NN value: {bestNN}, ",
      f"Best Accuracy Difference: {bestAcc:.3f}\n")

# Plot each tested value of NN against its accuracy
fig, fig4 = plt.subplots()
fig4.set_title("k-NN Diagnosis") 
fig4.set_xlabel('Nearest Neighbors') 
fig4.set_ylabel('Prediction Accuracy') 
fig4.plot(neighborValues, classificationTestAccuracies, label="Accuracy on Test Data")
fig4.plot(neighborValues, classificationTrainAccuracies, label="Accuracy on Training Data")
fig4.legend()



# Test the model with 5-Fold Cross Validation 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create lists to store training and test accuracies for each fold
foldTrainAccuracies = []
foldTestAccuracies = []
currentFold = 0

# Use StratifiedKFold with 5 splits and 20% test set
for trainIndex, testIndex in skf.split(data[:, :-2], data[:, -1]):
    currentFold += 1
    
    # Split the data into training and test sets for the current fold
    foldTrainX, foldTestX = data[trainIndex, :-2], data[testIndex, :-2]
    foldTrainY, foldTestY = data[trainIndex, -1], data[testIndex, -1]
    
    # Train the classifier on the training data
    bestClassifier.fit(foldTrainX, foldTrainY)

    # Calculate and store accuracies for this fold
    trainAcc = accuracy_score(foldTrainY, bestClassifier.predict(foldTrainX))
    testAcc = accuracy_score(foldTestY, bestClassifier.predict(foldTestX))
    
    foldTrainAccuracies.append(trainAcc)
    foldTestAccuracies.append(testAcc)
    
    # Print the training and test accuracy for each fold
    print(f"Fold {currentFold}: ",
          f"Training Accuracy: {trainAcc:.3f}, ",
          f"Test Accuracy: {testAcc:.3f} ")
    # Print the mean accuracies
print(f"Mean Training Accuracy: {np.mean(foldTrainAccuracies):.3f}")
print(f"Mean Test Accuracy: {np.mean(foldTestAccuracies):.3f}")