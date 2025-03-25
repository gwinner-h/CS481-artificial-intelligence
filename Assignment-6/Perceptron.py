import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load data
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:  # Only process lines with 3 values
                x1, x2, y = float(parts[0]), float(parts[1]), int(parts[2])
                data.append([x1, x2, y])
    return np.array(data)

# Separate data by class for plotting
def separate_by_class(data):
    class_dict = {}
    for row in data:
        x = row[:-1]
        y = int(row[-1])
        if y not in class_dict:
            class_dict[y] = []
        class_dict[y].append(x)
    
    # Convert lists to numpy arrays
    for class_value in class_dict:
        class_dict[class_value] = np.array(class_dict[class_value])
    
    return class_dict

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=1000, batch_mode=True):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_mode = batch_mode
        self.weights = None
        self.bias = None
        self.errors_history = []
    
    def _initialize_weights(self, n_features):
        """Initialize weights and bias to small random values"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.random.randn() * 0.01
    
    def _activation(self, x):
        """Step function"""
        return 1 if x >= 0 else -1
    
    def _predict_raw(self, X):
        """Calculate the weighted sum for each sample"""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """Predict class labels"""
        raw_output = self._predict_raw(X)
        # Apply activation function
        return np.array([self._activation(x) for x in raw_output])
    
    def _update_weights_incremental(self, X, y):
        """Update weights using incremental (online) learning"""
        errors = 0
        for i, x_i in enumerate(X):
            # Convert y from {1, 2} to {-1, 1}
            y_i = 1 if y[i] == 2 else -1
            
            # Predict
            activation = self._activation(np.dot(x_i, self.weights) + self.bias)
            
            # Update weights only if prediction is wrong
            if y_i != activation:
                self.weights += self.learning_rate * y_i * x_i
                self.bias += self.learning_rate * y_i
                errors += 1
        
        return errors
    
    def _update_weights_batch(self, X, y):
        """Update weights using batch learning"""
        # Convert y from {1, 2} to {-1, 1}
        y_transformed = np.where(y == 2, 1, -1)
        
        # Predict
        activations = np.array([self._activation(np.dot(x_i, self.weights) + self.bias) for x_i in X])
        
        # Calculate misclassified samples
        misclassified = y_transformed != activations
        errors = np.sum(misclassified)
        
        # Compute weight updates based on all misclassified samples
        if errors > 0:
            delta_w = self.learning_rate * np.dot(y_transformed[misclassified], X[misclassified])
            delta_b = self.learning_rate * np.sum(y_transformed[misclassified])
            
            self.weights += delta_w
            self.bias += delta_b
        
        return errors
    
    def fit(self, X, y):
        """Train the perceptron"""
        self._initialize_weights(X.shape[1])
        
        # Track errors over epochs
        self.errors_history = []
        
        for epoch in range(self.max_epochs):
            if self.batch_mode:
                errors = self._update_weights_batch(X, y)
            else:
                errors = self._update_weights_incremental(X, y)
            
            self.errors_history.append(errors)
            
            # Stop if no errors (convergence)
            if errors == 0:
                print(f"Converged after {epoch+1} epochs")
                break
        
        if epoch == self.max_epochs - 1:
            print(f"Reached max epochs ({self.max_epochs}) without convergence")
        
        return self
    
    def score(self, X, y):
        """Calculate accuracy"""
        # Convert y from {1, 2} to {-1, 1} for comparison
        y_transformed = np.where(y == 2, 1, -1)
        predictions = self.predict(X)
        return np.sum(predictions == y_transformed) / len(y)

# Function to evaluate and print results
def evaluate_perceptron(model, X_train, y_train, X_test, y_test):
    # Training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Test accuracy
    test_accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Print test predictions
    print("\nTest predictions:")
    print("X1\t\tX2\t\tTrue\tPredicted")
    
    # Convert {-1, 1} predictions back to {1, 2}
    predictions = np.where(model.predict(X_test) == 1, 2, 1)
    
    for i, (sample, true_label, pred_label) in enumerate(zip(X_test, y_test, predictions)):
        print(f"{sample[0]:.4f}\t{sample[1]:.4f}\t{true_label}\t{pred_label}")
    
    return train_accuracy, test_accuracy, predictions

# Plot decision boundary
def plot_decision_boundary(model, train_data, test_data, title):
    # Create a mesh grid
    x_min, x_max = min(train_data[:, 0].min(), test_data[:, 0].min()) - 1, max(train_data[:, 0].max(), test_data[:, 0].max()) + 1
    y_min, y_max = min(train_data[:, 1].min(), test_data[:, 1].min()) - 1, max(train_data[:, 1].max(), test_data[:, 1].max()) + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict class for each point in the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = np.where(Z == 1, 2, 1)  # Convert back to original labels {1, 2}
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.unique(Z).tolist())
    
    # Plot the decision line (w·x + b = 0)
    plt.contour(xx, yy, (np.dot(np.c_[xx.ravel(), yy.ravel()], model.weights) + model.bias).reshape(xx.shape), 
                levels=[0], colors='k', linestyles='--')
    
    plt.title(title)
    
    # Plot training data
    train_class_dict = separate_by_class(train_data)
    for class_value, features in train_class_dict.items():
        plt.scatter(features[:, 0], features[:, 1], label=f'Class {class_value} (Train)', marker='o')
    
    # Plot test data
    test_class_dict = separate_by_class(test_data)
    for class_value, features in test_class_dict.items():
        plt.scatter(features[:, 0], features[:, 1], label=f'Class {class_value} (Test)', marker='x')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# Plot convergence
def plot_convergence(incremental_errors, batch_errors):
    plt.figure(figsize=(12, 6))
    
    if incremental_errors:
        plt.plot(incremental_errors, label='Incremental Learning')
    
    if batch_errors:
        plt.plot(batch_errors, label='Batch Learning')
    
    plt.title('Perceptron Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Errors')
    plt.legend()
    plt.grid(True)
    plt.savefig('perceptron_convergence.png')
    plt.show()

def main():
    # Load data
    train_data = load_data('classify_train_2D.dat')
    test_data = load_data('classify_test_2D.dat')
    
    # Split features and labels
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(int)
    
    print("Data Information:")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Results dictionary to store accuracies for comparison
    results = {}
    
    # 1. Incremental Perceptron
    print("\n------------- Incremental Perceptron -------------")
    incremental_perceptron = Perceptron(learning_rate=0.01, max_epochs=1000, batch_mode=False)
    incremental_perceptron.fit(X_train, y_train)
    
    print("\nIncremental Perceptron Results:")
    train_acc, test_acc, _ = evaluate_perceptron(incremental_perceptron, X_train, y_train, X_test, y_test)
    results['Incremental Perceptron'] = {'train': train_acc, 'test': test_acc}
    
    # Plot decision boundary
    plot_decision_boundary(incremental_perceptron, train_data, test_data, "Incremental Perceptron")
    
    # Store errors for convergence plot
    incremental_errors = incremental_perceptron.errors_history
    
    # 2. Batch Perceptron
    print("\n------------- Batch Perceptron -------------")
    batch_perceptron = Perceptron(learning_rate=0.01, max_epochs=1000, batch_mode=True)
    batch_perceptron.fit(X_train, y_train)
    
    print("\nBatch Perceptron Results:")
    train_acc, test_acc, _ = evaluate_perceptron(batch_perceptron, X_train, y_train, X_test, y_test)
    results['Batch Perceptron'] = {'train': train_acc, 'test': test_acc}
    
    # Plot decision boundary
    plot_decision_boundary(batch_perceptron, train_data, test_data, "Batch Perceptron")
    
    # Store errors for convergence plot
    batch_errors = batch_perceptron.errors_history
    
    # Plot convergence
    plot_convergence(incremental_errors, batch_errors)
    
    # Compare results
    print("\n------------- Comparison of All Methods -------------")
    print("Method\t\t\tTraining Accuracy\tTest Accuracy")
    for method, accuracies in results.items():
        print(f"{method}\t{accuracies['train']:.4f}\t\t{accuracies['test']:.4f}")

if __name__ == "__main__":
    main()