import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

# load datasets
train_data = np.loadtxt('classify_train_2D.txt')
test_data = np.loadtxt('classify_test_2D.txt')

# sploit into features and labels
X_train, y_train = train_data[:, :2], train_data[:, 2]
X_test, y_test = test_data[:, :2], test_data[:, 2]

# compute class priors
classes, class_counts = np.unique(y_train, return_counts=True)
priors = {cls: count / len(y_train) for cls, count in zip(classes, class_counts)}

# compute class means and covariances
class_params = {}
for cls in classes:
    X_cls = X_train[y_train == cls]
    mean = np.mean(X_cls, axis=0)
    cov = np.cov(X_cls, rowvar=False)
    class_params[cls] = {'mean': mean, 'cov': cov}

# Bayes Plug-in Classifier
predictions = []
for x in X_test:
    posteriors = {}
    for cls in classes:
        likelihood = multivariate_normal.pdf(x, mean=class_params[cls]['mean'], cov=class_params[cls]['cov'])
        posteriors[cls] = likelihood * priors[cls]
    predictions.append(max(posteriors, key=posteriors.get))

y_pred = np.array(predictions)

# compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Bayes Plug-in Rule Classifier Accuracy: {accuracy:.2f}')
