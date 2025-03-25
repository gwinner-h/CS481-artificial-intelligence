import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# load data
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                x1, x2, y = float(parts[0]), float(parts[1]), int(parts[2])
                data.append([x1, x2, y])
    return np.array(data)

# separate data by class for plotting
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

# build a single hidden layer neural network
def build_single_layer_model(input_dim, hidden_units, output_dim):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_dim=input_dim),
        Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# build a two hidden layer neural network
def build_two_layer_model(input_dim, hidden_units1, hidden_units2, output_dim):
    model = Sequential([
        Dense(hidden_units1, activation='relu', input_dim=input_dim),
        Dense(hidden_units2, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Plot decision boundaries
def plot_decision_boundary(model, scaler, train_data, test_data, title):
    # Create a mesh grid for the decision boundary
    x_min, x_max = min(train_data[:, 0].min(), test_data[:, 0].min()) - 1, max(train_data[:, 0].max(), test_data[:, 0].max()) + 1
    y_min, y_max = min(train_data[:, 1].min(), test_data[:, 1].min()) - 1, max(train_data[:, 1].max(), test_data[:, 1].max()) + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict class for each point in the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = model.predict(mesh_points_scaled)
    Z = np.argmax(Z, axis=1) + 1  # Add 1 to match original class labels
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=np.unique(Z).tolist())
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

# Calculate and print detailed predictions
def print_test_predictions(model, X_test_scaled, X_test_original, y_test_original):
    predictions = model.predict(X_test_scaled)
    pred_labels = np.argmax(predictions, axis=1) + 1  # Add 1 to match original class labels
    
    # Print predictions
    print("\nTest predictions:")
    print("X1\t\tX2\t\tTrue\tPredicted")
    for i, (sample, true_label, pred_label) in enumerate(zip(X_test_original, y_test_original, pred_labels)):
        print(f"{sample[0]:.4f}\t{sample[1]:.4f}\t{int(true_label)}\t{pred_label}")
    
    # Calculate accuracy
    accuracy = np.mean(pred_labels == y_test_original)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    return accuracy

def main():
    # Load data
    train_data = load_data('classify_train_2D.dat')
    test_data = load_data('classify_test_2D.dat')
    
    # Split features and labels
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].astype(int)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(int)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert labels to categorical for multi-class classification
    # Note: Since our classes are 1 and 2, we need to subtract 1 to make them 0 and 1
    y_train_cat = to_categorical(y_train - 1, num_classes=2)
    y_test_cat = to_categorical(y_test - 1, num_classes=2)
    
    # Model parameters
    input_dim = X_train.shape[1]
    output_dim = 2  # Number of classes
    
    print("Data Information:")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # 1. Single Hidden Layer Network
    print("\n------------- Single Hidden Layer Network -------------")
    hidden_units = 8
    
    single_layer_model = build_single_layer_model(input_dim, hidden_units, output_dim)
    print(single_layer_model.summary())
    
    # Train the model
    history_single = single_layer_model.fit(
        X_train_scaled, y_train_cat,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on training data
    train_loss, train_acc = single_layer_model.evaluate(X_train_scaled, y_train_cat, verbose=0)
    print(f"\nSingle Layer Network - Training accuracy: {train_acc:.4f}")
    
    # Evaluate and print detailed predictions on test data
    print("\nSingle Layer Network - Test Results:")
    single_layer_test_acc = print_test_predictions(single_layer_model, X_test_scaled, X_test, y_test)
    
    # Plot decision boundary
    plot_decision_boundary(single_layer_model, scaler, train_data, test_data, 
                          f"Single Hidden Layer NN (Units: {hidden_units})")
    
    # 2. Two Hidden Layer Network
    print("\n------------- Two Hidden Layer Network -------------")
    hidden_units1 = 8
    hidden_units2 = 4
    
    two_layer_model = build_two_layer_model(input_dim, hidden_units1, hidden_units2, output_dim)
    print(two_layer_model.summary())
    
    # Train the model
    history_two = two_layer_model.fit(
        X_train_scaled, y_train_cat,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate on training data
    train_loss, train_acc = two_layer_model.evaluate(X_train_scaled, y_train_cat, verbose=0)
    print(f"\nTwo Layer Network - Training accuracy: {train_acc:.4f}")
    
    # Evaluate and print detailed predictions on test data
    print("\nTwo Layer Network - Test Results:")
    two_layer_test_acc = print_test_predictions(two_layer_model, X_test_scaled, X_test, y_test)
    
    # Plot decision boundary
    plot_decision_boundary(two_layer_model, scaler, train_data, test_data, 
                          f"Two Hidden Layer NN (Units: {hidden_units1}, {hidden_units2})")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_single.history['accuracy'], label='Single Layer Train')
    plt.plot(history_single.history['val_accuracy'], label='Single Layer Validation')
    plt.plot(history_two.history['accuracy'], label='Two Layer Train')
    plt.plot(history_two.history['val_accuracy'], label='Two Layer Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_single.history['loss'], label='Single Layer Train')
    plt.plot(history_single.history['val_loss'], label='Single Layer Validation')
    plt.plot(history_two.history['loss'], label='Two Layer Train')
    plt.plot(history_two.history['val_loss'], label='Two Layer Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('neural_network_training_history.png')
    plt.show()
    
    # Compare results
    print("\n------------- Comparison -------------")
    print(f"Single Hidden Layer Network - Test Accuracy: {single_layer_test_acc:.4f}")
    print(f"Two Hidden Layer Network - Test Accuracy: {two_layer_test_acc:.4f}")

if __name__ == "__main__":
    main()