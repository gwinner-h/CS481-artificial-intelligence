import numpy as np

# load data from files
def load_data(filename):
    data = np.loadtxt(filename)
    X = data[:, :2]  # Features
    y = data[:, 2]   # Labels
    return X, y

# estimate class priors
def estimate_priors(y):
    classes, counts = np.unique(y, return_counts=True)
    priors = counts / len(y)
    return dict(zip(classes, priors))

# estimate mean and covariance to get class-conditional densities
def estimate_class_conditional(X, y):
    classes = np.unique(y)
    class_params = {}
    for c in classes:
        X_c = X[y == c]
        mean = np.mean(X_c, axis=0)
        cov = np.cov(X_c, rowvar=False)
        class_params[c] = {'mean': mean, 'cov': cov}
    return class_params

# discriminant function
def discriminant_function(x, mean, cov, prior):
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
    normalization = -0.5 * np.log(np.linalg.det(cov))
    return exponent + normalization + np.log(prior)

# classify test data
def classify(X_test, class_params, priors):
    y_pred = []
    for x in X_test:
        scores = []
        for c in class_params:
            mean = class_params[c]['mean']
            cov = class_params[c]['cov']
            prior = priors[c]
            score = discriminant_function(x, mean, cov, prior)
            scores.append(score)
        y_pred.append(np.argmax(scores))
    return np.array(y_pred)

# load training and test data
X_train, y_train = load_data('classify_train_2D.txt')
X_test, y_test = load_data('classify_test_2D.txt')

# estimate priors and class-conditional densities
priors = estimate_priors(y_train)
class_params = estimate_class_conditional(X_train, y_train)

# classify test data
y_pred = classify(X_test, class_params, priors)

# calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# output the predictions
print('Predictions:', y_pred)