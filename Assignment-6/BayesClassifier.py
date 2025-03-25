import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# load training data
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # Only process lines with 3 values
                x1, x2, y = float(parts[0]), float(parts[1]), int(parts[2])
                data.append([x1, x2, y])
    return np.array(data)

# seaparte data by class
def separate_by_class(data):
    class_dict = {}
    for row in data:
        x = row[:-1]
        y = int(row[-1])
        if y not in class_dict:
            class_dict[y] = []
        class_dict[y].append(x)
    
    # convert lists to arrays
    for class_value in class_dict:
        class_dict[class_value] = np.array(class_dict[class_value])
    
    return class_dict

# calculate class priors
def calculate_priors(data):
    classes = np.unique(data[:, -1])
    priors = {}
    n_samples = len(data)
    
    for c in classes:
        priors[c] = np.sum(data[:, -1] == c) / n_samples
    
    return priors

# calculate mean vector for each class
def calculate_means(class_dict):
    means = {}
    for class_value, features in class_dict.items():
        means[class_value] = np.mean(features, axis=0)
    return means

# calculate covariance matrix for each class
def calculate_covariances(class_dict):
    covariances = {}
    for class_value, features in class_dict.items():
        covariances[class_value] = np.cov(features, rowvar=False)
    return covariances

# calculate discriminant function
def discriminant_function(x, mean, covariance, prior):
    mvn = multivariate_normal(mean=mean, cov=covariance)
    return np.log(prior) + mvn.logpdf(x)

# predict class for a sample
def predict(x, means, covariances, priors):
    discriminants = {}
    for class_value in means:
        discriminants[class_value] = discriminant_function(
            x, means[class_value], covariances[class_value], priors[class_value]
        )
    
    # return the class with the maximum discriminant value
    return max(discriminants, key=discriminants.get)

# evaluate the model
def evaluate(X, y, means, covariances, priors):
    predictions = []
    for sample in X:
        predictions.append(predict(sample, means, covariances, priors))
    
    # calculate accuracy
    accuracy = np.sum(predictions == y) / len(y)
    return predictions, accuracy

def main():
    # Load data
    train_data = load_data('classify_train_2D.dat')
    test_data = load_data('classify_test_2D.dat')
    
    # Separate data by class
    train_class_dict = separate_by_class(train_data)
    
    # Calculate priors from training data
    priors = calculate_priors(train_data)
    
    # Calculate means and covariances from training data
    means = calculate_means(train_class_dict)
    covariances = calculate_covariances(train_class_dict)
    
    # Evaluate on training data
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int)
    train_predictions, train_accuracy = evaluate(X_train, y_train, means, covariances, priors)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Print model parameters
    print("\nModel parameters:")
    for class_value in sorted(means.keys()):
        print(f"\nClass {class_value}:")
        print(f"Prior probability: {priors[class_value]:.4f}")
        print(f"Mean vector: {means[class_value]}")
        print(f"Covariance matrix:\n{covariances[class_value]}")
    
    # Evaluate on test data
    if len(test_data) > 0:
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1].astype(int)
        test_predictions, test_accuracy = evaluate(X_test, y_test, means, covariances, priors)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        # Show detailed predictions for test data
        print("\nTest predictions:")
        print("X1\tX2\tTrue\tPredicted")
        for i, (sample, true_label, pred_label) in enumerate(zip(X_test, y_test, test_predictions)):
            print(f"{sample[0]:.4f}\t{sample[1]:.4f}\t{true_label}\t{pred_label}")

if __name__ == "__main__":
    main()