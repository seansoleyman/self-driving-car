import csv
import os
import numpy as np
import cv2
import keras
import sklearn
import sklearn.model_selection
import sklearn.utils

DATA_DIR = "data"
SIDE_CAMERA_ANGLE_CORRECTION = 0.10
BATCH_SIZE = 32

def load_instances(data_dir):
    """Open a CSV file and use it to create a list of instances."""
    csv_path = os.path.join(data_dir, "driving_log.csv")
    images_dir = os.path.join(data_dir, "IMG")
    images_paths = []
    images_flip_boolean = []
    measurements = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            for column in range(0, 3):
                image_filename = os.path.split(line[column])[1]
                image_path = os.path.join(images_dir, image_filename)
                measurement = float(line[3])
                if column==1:
                    measurement += SIDE_CAMERA_ANGLE_CORRECTION
                elif column==2:
                    measurement -= SIDE_CAMERA_ANGLE_CORRECTION
                images_paths.append(image_path)
                images_flip_boolean.append(False)
                measurements.append(measurement)
                images_paths.append(image_path)
                images_flip_boolean.append(True)
                measurements.append(-measurement)
    return list(zip(images_paths, images_flip_boolean, measurements))

# Create instances from each subdirectory (multiple training runs). 
data_subdirectories = next(os.walk(DATA_DIR))[1]
instances = []
for data_subdirectory in data_subdirectories:
    instances += load_instances(os.path.join(DATA_DIR, data_subdirectory))
print("Number of Training Instances:", len(instances))

# Split the data into training and validation sets. 
instances_train, instances_val = sklearn.model_selection.train_test_split(
    instances, test_size=0.2)

def generator(samples, batch_size=BATCH_SIZE):
    """Produces batches of data by loading images on the fly."""
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples-batch_size+1, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for [image_path, image_flip_boolean, measurement] in batch_samples:
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                if image_flip_boolean:
                    image = cv2.flip(image, 1)
                images.append(image)
                measurements.append(measurement)

            X_train = np.array(images, np.uint8)
            y_train = np.array(measurements, np.float32)
            yield [X_train, y_train]

# Create two generators to be used during training.
train_generator = generator(instances_train)
val_generator = generator(instances_val)

# Model definition. 
model = keras.models.Sequential()
model.add(keras.layers.core.Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))
model.add(keras.layers.Cropping2D(cropping=((70, 25), (0, 0))))
model.add(keras.layers.convolutional.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(keras.layers.convolutional.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(keras.layers.convolutional.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(keras.layers.convolutional.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.convolutional.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the model. 
model.fit_generator(
    train_generator, 
    steps_per_epoch=len(instances_train)//BATCH_SIZE, 
    validation_data=val_generator, 
    validation_steps=len(instances_val)//BATCH_SIZE,
    epochs=2)

# Save the model. 
model.save("model.h5")

