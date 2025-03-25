import numpy as np
from matplotlib import pyplot as plt
from keras.src.datasets import cifar10 as cifar
from keras.src.utils import to_categorical
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.src.layers import preprocessing
from keras.src.optimizers import Adam, Nadam, SGD

MAX_PIXEL_VALUE = 255
FLOAT32 = 'float32'
filename = 'test1'
demo_image = 'demo_image.png'

# load train and test dataset
def load_dataset():
    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar.load_data()

    # summarize loaded dataset
    print('Train: X = %s, y = %s' % (X_train.shape, y_train.shape))
    print('Test: X = %s, y = %s' % (X_test.shape, y_test.shape))

    # one hot encode target values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test

# load and prepare an image
def load_image(filename):
    # load the image
    img = preprocessing.load_img(filename, target_size=(32,32))
    
    # convert to an array
    img = preprocessing.img_to_array(img)

    # reshape into a single sample with 3 channels
    img = img.reshape(1,32,32,3)

    # prepare pixel data
    img = img.astype(FLOAT32)
    img = img / MAX_PIXEL_VALUE

    return img

# for testing: plot first few images
def peek_data(train):
    for i in range(9):
        plt.subplot(330 + 1 + i) # define subplot
        plt.imshow(train[i]) # plot raw pixel data

    # show the figure
    plt.axis('off')
    plt.show()

# load image
def run_image_demo():
    img = load_image(demo_image) # load the image
    model = preprocessing.load_model('cnn_model.h5') # load the model

    # predict class and print the result
    result = model.predict_classes(img)
    print(result[0])

# predict class from uploaded image
def predict_class(img):
    model = preprocessing.load_model('cnn_model.h5')

    # predict the class
    result = model.predict_classes(img)
    print(result[0])

# scale image pixels by normalizing their value btwn 0 and 1
def prepare_image(train, test):
    # convert from integers to floats
    train_norm = train.astype(FLOAT32)
    test_norm = test.astype(FLOAT32)

    # normalize to range btwn 0 and 1
    train_norm = train_norm / MAX_PIXEL_VALUE
    test_norm = test_norm / MAX_PIXEL_VALUE

    # return normalized images
    return train_norm, test_norm

# define the convultional neural network
def build_cnn():
    model = Sequential()
    
    # 3-block 
    model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
    model.add(Conv2D(32,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0,2))
    model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0,2))
    model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128,(3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0,2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0,2))
    model.add(Dense(10, activation='softmax'))

    # def adam optimizer
    opt = Adam(learning_rate=0.001, ema_momentum=0.9)

    # def nadam optimizer
    # opt = Nadam(learning_rate=0.001, ema_momentum=0.9)

    # def SGD optimizer
    # opt = SGD(learning_rate=0.001, ema_momentum=0.9)

    # compile model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# plot diagnostic learning curves
def summarize(history):
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
    plt.savefig(filename + '_plot.png')
    plt.close()

# evaluate model
def evaluate_model(model, acc):
    print('> %.3f' % (acc * 100.0))

    # learning curves
    summarize(model)

    model.save('cnn_model.h5')

# test for evaluating model
def test_model():
    # load test and train data
    X_train, y_train, X_test, y_test = load_dataset()

    # peek at the images
    peek_data(X_train)

    # prepare the image pixels
    X_train, X_test = prepare_image(X_train, X_test)

    model = build_cnn()

    # fit model
    history = model.fit(X_train, y_train, epochs=150, batch_size=64,
                        validation_data=(X_test, y_test), verbose=1)
    
    _loss, acc = model.evaluate(X_test, y_test, verbose=1)
    
    return history, acc

# testing purposes for conv2d
def example_conv2d():
    x = np.random.rand(4, 10, 10, 128)
    y = Conv2D(32, 3, activation='relu')(x)
    print(y.shape)


def main():
    model, acc = test_model()
    evaluate_model(model, acc)
    run_image_demo()

if __name__ == "__main__":
    main()