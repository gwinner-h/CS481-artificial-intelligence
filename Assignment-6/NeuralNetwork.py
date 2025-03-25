import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# load datasets
train_data = np.loadtxt('classify_train_2D.txt')
test_data = np.loadtxt('classify_test_2D.txt')

# split into features and labels
X_train, y_train = train_data[:, :2], train_data[:, 2]
X_test, y_test = test_data[:, :2], test_data[:, 2]

# normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build and train single hidden layer model
def create_model(hidden_layers=1, neurons_per_layer=10):
    model = keras.Sequential()
    model.add(keras.layers.Dense(neurons_per_layer, activation='relu', input_shape=(2,)))
    if hidden_layers == 2:
        model.add(keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# train and evaluate single hidden layer model
model_1 = create_model(hidden_layers=1)
model_1.fit(X_train, y_train, epochs=50, verbose=0)
y_pred_1 = (model_1.predict(X_test) > 0.5).astype(int)
accuracy_1 = accuracy_score(y_test, y_pred_1)
print(f'Single hidden layer model accuracy: {accuracy_1:.2f}')

# train and evaluate two hidden layers model
model_2 = create_model(hidden_layers=2)
model_2.fit(X_train, y_train, epochs=50, verbose=0)
y_pred_2 = (model_2.predict(X_test) > 0.5).astype(int)
accuracy_2 = accuracy_score(y_test, y_pred_2)
print(f'Two hidden layers model accuracy: {accuracy_2:.2f}')
