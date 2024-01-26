# CONVOLUTION NEURAL NETWORK Skin Cancer Dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



# Preprocessing the Training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(128, 128),  # Adjusting image size
    batch_size=32,
    class_mode='binary'
)

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'validation',
    target_size=(128, 128),  # Adjusting image size
    batch_size=32,
    class_mode='binary'
)

# Initialising the CNN
cnn = Sequential()

# Step 1 - Convolution
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))

# Step 2 - Pooling
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(Flatten())

# Step 4 - Full Connection
cnn.add(Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(Dense(units=1, activation='sigmoid'))


# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


cnn.summary()

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)


import matplotlib.pyplot as plt

# Accessing the history object from model training
history_dict = cnn.history.history

# Extracting accuracy and validation accuracy
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

# Extracting loss and validation loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Plotting accuracy
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'gold', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'purple', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.plot(epochs, loss, 'pink', label='Training loss')
plt.plot(epochs, val_loss, 'orange', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


from tensorflow.keras.preprocessing import image
import numpy as np

def predict_single_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Adjust target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = cnn.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Malignant"
    else:
        return "Benign"

# Example usage
single_image_path = 'ISIC_0330896.JPG'  # Replace with your image path
prediction_result = predict_single_image(single_image_path)
print(f"The tumor in the single image is predicted as {prediction_result}.")


from tensorflow.keras.preprocessing import image
import numpy as np

def predict_single_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Adjust target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = cnn.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Malignant"
    else:
        return "Benign"

# Example usage
single_image_path = 'ISIC_5309744.JPG'  # Replace with your image path
prediction_result = predict_single_image(single_image_path)
print(f"The tumor in the single image is predicted as {prediction_result}.")


