from matplotlib import pyplot as plt
from helpers import Loader, Image
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.src.utils import load_img, img_to_array
from keras.src.saving.saving_api import load_model

model_filename = 'final_cnn-50-epochs-85.keras'
image_filename = 'demo_image.png'

# define the convultional neural network
def build_cnn():
    """
    Builds and compiles a Convolutional Neural Network (CNN) model for image classification.
    The CNN consists of three convolutional blocks, each followed by batch normalization, 
    max pooling, and dropout layers. The model ends with a fully connected dense layer 
    and a softmax output layer for multi-class classification.
    Returns:
        keras.models.Sequential: A compiled CNN model.
    Architecture:
        - Block 1:
            - Conv2D: 32 filters, 3x3 kernel, ReLU activation, He uniform initializer, same padding
            - BatchNormalization
            - Conv2D: 32 filters, 3x3 kernel, ReLU activation, He uniform initializer, same padding
            - BatchNormalization
            - MaxPooling2D: 2x2 pool size
            - Dropout: 0.2
        - Block 2:
            - Conv2D: 64 filters, 3x3 kernel, ReLU activation, He uniform initializer, same padding
            - BatchNormalization
            - Conv2D: 64 filters, 3x3 kernel, ReLU activation, He uniform initializer, same padding
            - BatchNormalization
            - MaxPooling2D: 2x2 pool size
            - Dropout: 0.3
        - Block 3:
            - Conv2D: 128 filters, 3x3 kernel, ReLU activation, He uniform initializer, same padding
            - BatchNormalization
            - Conv2D: 128 filters, 3x3 kernel, ReLU activation, He uniform initializer, same padding
            - BatchNormalization
            - MaxPooling2D: 2x2 pool size
            - Dropout: 0.4
        - Fully Connected Layers:
            - Flatten
            - Dense: 128 units, ReLU activation, He uniform initializer
            - BatchNormalization
            - Dropout: 0.5
            - Dense: 10 units, softmax activation (output layer)
    Optimizer:
        - Adam optimizer with a learning rate of 0.001 and EMA momentum of 0.9.
    Loss Function:
        - Categorical Crossentropy
    Metrics:
        - Accuracy
    """
    model = Sequential()
    
    # 3-block 
    model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# plot diagnostic learning curves
def summarize(history):
    """
    Generate and save plots for model training history.
    This function takes the training history of a model and creates two subplots:
    one for the cross-entropy loss and another for classification accuracy. It
    plots both training and validation metrics, saves the plot to a file, and
    then closes the plot.
    Args:
        history (keras.callbacks.History): The history object returned by the
            `fit` method of a Keras model. It contains the training and validation
            metrics for each epoch.
    Note:
        Ensure that the variable `filename` is defined in the global scope before
        calling this function. The plot will be saved as `<filename>_plot.png`.
    """
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.legend('true')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.legend('true')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')

    # save plot to file
    plt.savefig(model_filename + '_plot.png')
    plt.close()

# print accuracy
def print_accuracy(acc):
    """
    Prints the accuracy as a percentage with three decimal places.
    Args:
        acc (float): The accuracy value as a decimal (e.g., 0.85 for 85%).
    """
    print('> %.3f' % (acc * 100.0))

# for building and training model
def build_model():
    # load test and train data
    X_train, y_train, X_test, y_test = Loader.load_dataset(one_hot_encode=True, print_data_details=True)

    # prepare the image pixels
    X_train = Image.prepare_image(X_train)
    X_test = Image.prepare_image(X_test)

    # build and fit model
    model = build_cnn()
    history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                        validation_data=(X_test, y_test), verbose=1)
    
    # evaluate the model
    _, acc = model.evaluate(X_test, y_test, verbose=1)
    print_accuracy(acc)

    # print learning curves and save model
    summarize(history)
    model.save('final_cnn-10-epochs.keras')

# make prediction for a new image
def make_prediction():
    # load the image
    img = Loader.load_image(image_filename)

    # load model
    model = load_model(model_filename)

    # predict the class
    result = model.predict(img)

    # print results
    print(result[0])


def main():
    make_prediction()


if __name__ == "__main__":
    main()