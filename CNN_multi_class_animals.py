# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 13:08:23 2020

@author: seknayu
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing CNN

model = Sequential()

# Step1 : Convolutional layter
model.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))

# Step: MaxPooling layting
model.add(MaxPooling2D(pool_size=(2,2)))

#Step3 : Convolutional layer
model.add(Conv2D(32, (3,3), activation = 'relu'))

#step4 : MaxPooling layer
model.add(MaxPooling2D(pool_size=(2,2)))


# Step5: Flattening
model.add(Flatten())

# Step 6 - Full connection
model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 4, activation = 'softmax'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# part 2: Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:/DL_Datasets/training/animals/data',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
classes = training_set.class_indices
print(classes)

test_set = test_datagen.flow_from_directory('D:/DL_Datasets/training/animals/data',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# model.fit_generator(training_set,
#                     steps_per_epoch = 8000,
#                     epochs = 1,
#                     validation_data = test_set,
#                     validation_steps = 2000)

#model.save('model.h5')


