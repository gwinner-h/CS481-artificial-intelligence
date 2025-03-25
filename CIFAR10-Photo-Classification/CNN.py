from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.src.optimizers import Adam

class CNN:
    # define the convultional neural network
    def build_model():
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

        # def adam optimizer
        opt = Adam(learning_rate=0.001, ema_momentum=0.9)

        # compile model
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model