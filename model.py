# importing the required libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# image preprocessing dependencies
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class CNN:
    def modelCreation(self):
        '''
            This function creates and compiles an CNN model
                args: none
                return: CNN model
        '''

        # Initialising the CNN
        model = Sequential()
        # Step 1 - Convolution - Input Shape specified*
        model.add(Conv2D(32, (3, 3), input_shape= (64, 64, 3), activation='relu'))
        # Step 2 - Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Step 3 - One more Convolution layer
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Step 4 - Flatening
        model.add(Flatten())
        # Step 5 - Fully Connecting Layers
        model.add(Dense(units = 128, activation = 'relu'))
        model.add(Dense(units = 1, activation = 'sigmoid'))

        # compiling the model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        # generating training and validation set
        train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
        validation_datagen = ImageDataGenerator(rescale = 1./255)
        # loading the data from the directory
        training_set = train_datagen.flow_from_directory('data/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
        validation_set = validation_datagen.flow_from_directory('data/testing_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

        # training the model
        model.fit_generator(training_set, steps_per_epoch = 200, epochs = 2, validation_data = validation_set, validation_steps = 80)

        return model

