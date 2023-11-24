import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

# define input image size and number of images that will be imported for training
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
INPUT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/CSUF/2023/2023 Fall/CPSC 483 - Intro to Machine Learning/Project/leaf_health/Project Code/Preprocessed Data/"

# pull in data
file_names = tf.constant([os.path.join(INPUT_DIR, fname) for fname in os.listdir(INPUT_DIR)])
dataset = tf.data.Dataset.from_tensor_slices((file_names))
data_size = dataset.cardinality().numpy()
print("Total dataset dataset size: " + str(data_size))

# let's define data splits
training_size = int(0.7 * data_size)
validation_size = int(0.15 * data_size)
testing_size = int(0.15 * data_size)

# let's shuffle the dataset and split it
# buffer_size set to BATCH_SIZE since it is just a sample, and 123 set as seed to get the same results every time
dataset = dataset.shuffle(BATCH_SIZE, seed = 123)
training_dataset = dataset.take(training_size)
testing_dataset = dataset.skip(training_size)
validation_dataset = testing_dataset.skip(validation_size)
testing_dataset = testing_dataset.take(testing_size)

# review data splits
print("Training dataset size: " + str(training_dataset.cardinality().numpy()))
print("Testing dataset size: " + str(testing_dataset.cardinality().numpy()))
print("Validation dataset size: " + str(validation_dataset.cardinality().numpy()))

# build model
# Look at below URL
# https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01
