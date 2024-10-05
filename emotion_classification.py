
import cv2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

MAK_model = Sequential()

MAK_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
MAK_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
MAK_model.add(MaxPooling2D(pool_size=(2, 2)))
MAK_model.add(Dropout(0.25))

MAK_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
MAK_model.add(MaxPooling2D(pool_size=(2, 2)))
MAK_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
MAK_model.add(MaxPooling2D(pool_size=(2, 2)))
MAK_model.add(Dropout(0.25))

MAK_model.add(Flatten())
MAK_model.add(Dense(1024, activation='relu'))
MAK_model.add(Dropout(0.5))
MAK_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

MAK_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Train the neural network/model
MAK_model_info = MAK_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in json file
model_json = MAK_model.to_json()
with open("MAK_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
MAK_model.save_weights('MAK_model.h5')

