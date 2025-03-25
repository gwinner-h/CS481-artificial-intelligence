import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

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

#Logistic regression
classifier = lr(random_state=42)
classifier.fit(trainX, trainY)
print("Test set score: {:.2f}".format(classifier.score(testX,testY)))



#Metrics
predictions = classifier.predict(testX)
tn, fp, fn, tp = confusion_matrix(testY, predictions).ravel()

accuracy = (tp+tn) / (tp+tn+fp+fn)
recall = tp / (tp+fn)
precision = tp / (tp+fp)
specificity = tn / (tn+fp)
fstat = 2 * ((precision*recall) / (precision+recall))

print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"Specificity: {specificity}")
print(f"F-Stat: {fstat}")



# ROC Curve
fpr, tpr, thresholds = roc_curve(testY, classifier.predict_proba(testX)[:, 1])
roc_auc = auc(fpr, tpr)
bestIndex = np.argmax(tpr-fpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Skilless')
plt.plot(fpr[bestIndex], tpr[bestIndex], 'o', markersize=8, fillstyle="none", c='k', label='Best Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

print(f"\nfpr: {fpr}\n\ntpr: {tpr}\n\nthresholds: {thresholds}")
print()