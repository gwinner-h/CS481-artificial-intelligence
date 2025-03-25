from matplotlib import pyplot as plt
from keras.src.datasets import cifar10 as cifar
from keras.src.utils import to_categorical, load_img, img_to_array
from keras.src.layers import preprocessing, Conv2D
import numpy as np

MAX_PIXEL_VALUE = 255
FLOAT32 = 'float32'
filename = 'test1'
demo_image = 'demo_image.png'

# for testing: plot first few images
def peek_data(train):
    """
    Display a 3x3 grid of images from the given dataset.
    This function takes a dataset of images and visualizes the first 9 images
    in a 3x3 grid layout. Each image is displayed without axes for a cleaner view.
    Args:
        train (numpy.ndarray or list): A dataset containing image data. It is 
            expected to be an array-like structure where each element represents 
            an image.
    Returns:
        None: This function does not return any value. It displays the images 
        using matplotlib.
    """
    for i in range(9):
        plt.subplot(330 + 1 + i) # define subplot
        plt.imshow(train[i]) # plot raw pixel data

    # show the figure
    plt.axis('off')
    plt.show()

class Image:
    """
    A helper class for image preprocessing and classification.
    Methods:
    --------
    prepare_image(train, test):
        Scales image pixel values by normalizing them to a range between 0 and 1.
        Converts integer pixel values to floating-point values and normalizes
        them using a maximum pixel value constant.
        Parameters:
        - train: ndarray
            The training dataset containing image pixel values.
        - test: ndarray
            The testing dataset containing image pixel values.
        Returns:
        - tuple: (train_norm, test_norm)
            Normalized training and testing datasets.
    predict_class(img):
        Predicts the class of an uploaded image using a pre-trained CNN model.
        Parameters:
        - img: ndarray
            The image data to classify.
        Returns:
        - None
            Prints the predicted class index of the image.
    """
    # scale image pixels by normalizing their value btwn 0 and 1
    def prepare_image(img):
        # convert from integers to floats
        to_norm = img.astype(FLOAT32)

        # normalize to range btwn 0 and 1
        norm = to_norm / MAX_PIXEL_VALUE

        # return normalized images
        return norm

    # predict class from uploaded image
    def predict_class(img, filename):
        model = preprocessing.load_model(filename)

        # predict the class
        result = model.predict_classes(img)
        print(result[0])


class Loader:
    """
    A helper class for loading datasets and images, specifically designed for 
    tasks like CIFAR-10 photo classification. This class provides methods to 
    load datasets and preprocess images for machine learning models.
    Methods
    -------
    load_dataset(filename=False, is_csv=False, is_txt=False, is_cifar=False, 
        Loads a dataset based on the specified parameters. Currently set up 
        for CIFAR-10 by default. Supports optional one-hot encoding and 
        printing dataset details.
        Parameters:
        - filename (str, optional): Path to the dataset file. Default is False.
        - is_csv (bool, optional): Whether the dataset is in CSV format. Default is False.
        - is_txt (bool, optional): Whether the dataset is in TXT format. Default is False.
        - is_cifar (bool, optional): Whether the dataset is CIFAR-10. Default is False.
        - one_hot_encode (bool, optional): Whether to one-hot encode the labels. Default is False.
        - print_data_details (bool, optional): Whether to print dataset details. Default is False.
        Returns:
        - tuple: (X_train, y_train, X_test, y_test), where X and y are the 
          features and labels for training and testing datasets.
    load_image(filename):
        Loads and preprocesses an image for prediction. The image is resized, 
        converted to an array, reshaped, and normalized.
        Parameters:
        - filename (str): Path to the image file.
        Returns:
        - numpy.ndarray: Preprocessed image ready for input into a model.
    """
    # load data | todo: check file type
    def load_dataset(filename=False, is_csv=False, is_txt=False, is_cifar=False, 
                     one_hot_encode=False, print_data_details=False):
        (X_train, y_train), (X_test, y_test) = cifar.load_data() # change this! its only set up for AI final

        if print_data_details:
            # summarize loaded dataset
            print('Train: X = %s, y = %s' % (X_train.shape, y_train.shape))
            print('Test: X = %s, y = %s' % (X_test.shape, y_test.shape))

        if one_hot_encode:
            # one hot encode target values
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

        return X_train, y_train, X_test, y_test
    
    # load and prepare an image
    def load_image(filename):
        # load the image
        img = load_img(filename, target_size=(32,32))
        
        # convert to an array
        img = img_to_array(img)

        # reshape into a single sample with 3 channels
        img = img.reshape(1,32,32,3)

        # prepare image data
        img = Image.prepare_image(img)

        return img

class Tester:
    # testing purposes for conv2d
    def example_conv2d():
        """
        Demonstrates the use of a 2D convolutional layer (Conv2D) in a neural network.
        This function creates a random 4D input tensor with shape (4, 10, 10, 128),
        applies a Conv2D layer with 32 filters, a kernel size of 3x3, and ReLU activation,
        and prints the shape of the resulting output tensor.
        Note:
            - The function assumes that the necessary libraries (e.g., numpy and TensorFlow/Keras)
              are imported and available in the environment.
            - This is an example function and does not return any value.
        Example:
            >>> example_conv2d()
            (4, 8, 8, 32)
        """
        x = np.random.rand(4, 10, 10, 128)
        y = Conv2D(32, 3, activation='relu')(x)
        print(y.shape)