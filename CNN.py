
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
chars74k_classifier = Sequential()

# Adding the first convolutional layer
chars74k_classifier.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (299, 299, 3)))

# Adding the max pooling layer
chars74k_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the second convolutional layer
chars74k_classifier.add(Conv2D(32, (3, 3), activation='relu'))

# Adding a second max pooling layer
chars74k_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the third convolutional layer
chars74k_classifier.add(Conv2D(64, (3, 3), activation='relu'))

# Adding a third max pooling layer
chars74k_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the fourth convolutional layer
chars74k_classifier.add(Conv2D(128, (3, 3), activation='relu'))

# Adding a fourth max pooling layer
chars74k_classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding the flattening layer
chars74k_classifier.add(Flatten())

# Adding the fully connected layers (Normal ANN)
chars74k_classifier.add(Dense(activation = 'relu', units = 128))
chars74k_classifier.add(Dense(activation = 'relu', units = 512))
chars74k_classifier.add(Dropout(0.25))
chars74k_classifier.add(Dense(activation = 'relu', units = 512))
chars74k_classifier.add(Dense(activation = 'relu', units = 128))
chars74k_classifier.add(Dense(activation = 'softmax', units = 26))

# Compiling the CNN
chars74k_classifier.compile(optimizer='Adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Pre-processing the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(299, 299),
                                                batch_size=32,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(299, 299),
                                                        batch_size=32,
                                                        class_mode='categorical')

history = chars74k_classifier.fit_generator(training_set,
                    steps_per_epoch=3860,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=1342)

# Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/Z.jpg', target_size = (299, 299))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = chars74k_classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'A'
elif result[0][1] == 1:
    prediction = 'B'
elif result[0][2] == 1:
    prediction = 'C'
elif result[0][3] == 1:
    prediction = 'D'
elif result[0][4] == 1:
    prediction = 'E'
elif result[0][5] == 1:
    prediction = 'F'
elif result[0][6] == 1:
    prediction = 'G'
elif result[0][7] == 1:
    prediction = 'H'
elif result[0][8] == 1:
    prediction = 'I'
elif result[0][9] == 1:
    prediction = 'J'
elif result[0][10] == 1:
    prediction = 'K'
elif result[0][11] == 1:
    prediction = 'L'
elif result[0][12] == 1:
    prediction = 'M'
elif result[0][13] == 1:
    prediction = 'N'
elif result[0][14] == 1:
    prediction = 'O'
elif result[0][15] == 1:
    prediction = 'P'
elif result[0][16] == 1:
    prediction = 'Q'
elif result[0][17] == 1:
    prediction = 'R'
elif result[0][18] == 1:
    prediction = 'S'
elif result[0][19] == 1:
    prediction = 'T'
elif result[0][20] == 1:
    prediction = 'U'
elif result[0][21] == 1:
    prediction = 'V'
elif result[0][22] == 1:
    prediction = 'W'
elif result[0][23] == 1:
    prediction = 'X'
elif result[0][24] == 1:
    prediction = 'Y'
elif result[0][25] == 1:
    prediction = 'Z'
    
training_set.class_indices
# Saving the trained model
chars74k_classifier.save('chars74kV3.0.h5')

# Saving model as JSON
model_json = chars74k_classifier.to_json()
with open("model.json", "w") as chars74KJson:
    chars74KJson.write(model_json)
    
# Saving the models weights
chars74k_classifier.save_weights('mnistClassifierWeights.h5')

# Show the classes
training_set.class_indices

# Plotting the charts
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
