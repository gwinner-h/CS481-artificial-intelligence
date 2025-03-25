import numpy as np

def load_data(train_file, test_file):
    # load training data
    train_data = np.loadtxt(train_file, usecols=(0, 1))
    train_labels = np.loadtxt(train_file, usecols=(2))
    
    # load test data
    test_data = np.loadtxt(test_file, usecols=(0, 1))
    test_labels = np.loadtxt(test_file, usecols=(2))
    
    return train_data, train_labels, test_data, test_labels

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def knn_classifier(train_data, train_labels, test_point, k):
    # calculate distances from test point to all training points
    distances = [euclidean_distance(test_point, train_point) for train_point in train_data]
    
    # get indices & labels of k nearest neighbors
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = train_labels[k_indices]
    
    # return the most common label
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    return unique_labels[np.argmax(counts)]

def evaluate_classifier(train_data, train_labels, test_data, test_labels, k):
    # predict labels for test data
    predictions = [knn_classifier(train_data, train_labels, test_point, k) for test_point in test_data]
    
    # calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    return accuracy

def main():
    # load data
    train_data, train_labels, test_data, test_labels = load_data(
        'classify_train_2D.dat', 
        'classify_test_2D.dat')
    
    # test different k values
    k_values = [1, 3, 15]
    
    print("K-NN Classification Results:")
    for k in k_values:
        accuracy = evaluate_classifier(train_data, train_labels, test_data, test_labels, k)
        print(f"k = {k}: Accuracy = {accuracy:.2%}")

if __name__ == "__main__":
    main()