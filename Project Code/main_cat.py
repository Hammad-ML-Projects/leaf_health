import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
#%matplotlib inline
import pandas as pd
import numpy as np
import os
import PIL
import PIL.Image as pilim
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import cv2
import glob
import pickle
from tensorflow import keras
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# patch_sklearn()


# define input image size and number of images that will be imported for training
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
INPUT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/CSUF/2023/2023 Fall/CPSC 483 - Intro to Machine Learning/Project/leaf_health/Project Code/Preprocessed Data/"
CATEGORY_INPUT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/CSUF/2023/2023 Fall/CPSC 483 - Intro to Machine Learning/Project/leaf_health/Project Code/Preprocessed Data - Categorized/"
RESIZED_CATEGORY_INPUT_DIR = "/Users/hammadsheikh/Desktop/Documents/Studies/CSUF/2023/2023 Fall/CPSC 483 - Intro to Machine Learning/Project/leaf_health/Project Code/Preprocessed Data - Resized and Categorized/"

CATEGORIES = ['Alstonia Scholaris - Diseased', 'Alstonia Scholaris - Healthy', 'Arjun - Diseased', 'Arjun - Healthy', 'Bael - Diseased',
              'Bael - Healthy', 'Basil - Diseased', 'Basil - Healthy', 'Chinar - Diseased', 'Chinar - Healthy', 'Guava - Diseased',
              'Guava - Healthy', 'Jamun - Diseased', 'Jamun - Healthy', 'Jatropha - Diseased', 'Jatropha - Healthy', 'Lemon - Diseased',
              'Lemon - Healthy', 'Mango - Diseased', 'Mango - Healthy', 'Pomegranate - Diseased', 'Pomegranate - Healthy',
              'Pongamia Pinnata - Diseased', 'Pongamia Pinnata - Healthy']

# input array
flat_data_arr = []

# output array
target_arr = []

# load the images
for i in CATEGORIES:
    print(f'Loading category: {i} ...')
    path = os.path.join(RESIZED_CATEGORY_INPUT_DIR,i)
    for img in os.listdir(path):
        if img != '.DS_Store':
            img_array = imread(os.path.join(path,img))
            flat_data_arr.append(img_array.flatten())
            target_arr.append(CATEGORIES.index(i))
    print(f'Category: {i} loaded successfully!')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data) # dataframe
df['Target'] = target
x = df.iloc[:,:-1] # input data
y = df.iloc[:,-1] # output data

#--------------------------------------------------------------------#
# data split

testing_ratio = 0.15
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 77, stratify = y) # this is the original

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = testing_ratio, random_state = 1, stratify = y) # testing data is 15%
# x_val, x_test, y_val, y_test = train_test_split(x, y, test_size = testing_ratio / (1 - testing_ratio), random_state = 1, stratify = y) # validation data is 15% [x% of 85%]
# print('x info;')
# print(x_train, x_val, x_test)
# print('y info;')
# print(y_train, y_val, y_test)
print('Data split successfully!')


#--------------------------------------------------------------------#
# model construction
param_grid = {'C':[0.1,1,10,100], 'gamma':[0.0001,0.001,0.1,1], 'kernel':['rbf']}
# svc = svm.SVC(kernel = 'linear', probability = True)
# original
# param_grid = {'C':[0.1,1,10,100], 'gamma':[0.0001,0.001,0.1,1], 'kernel':['rbf','poly']}
svc = svm.SVC(kernel = 'linear', probability = True)

#--------------------------------------------------------------------#
# model training
print("The training of the model has started. This will take some time.\nGet some coffee, some food, or maybe go do some gardening!")
model = GridSearchCV(svc, param_grid)
model.fit(x_train, y_train)
print('The model is trained well with the given images')
# model.best_params_ # contains the best parameters obtained from GridSearchCV

#--------------------------------------------------------------------#
# model testing
y_pred = model.predict(x_test) # original training_test model; model.predict(x_test) --> x_val
print("The predicted Data is:")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred, y_test)*100}% accurate")

#--------------------------------------------------------------------#
# save model to disk
pickle.dump(model,open('img_model.p','wb'))
print("Pickle is dumped successfully!")

#--------------------------------------------------------------------#
# model evaluation

# load model
# model = pickle.load(open('img_model.p','rb'))
"""
url = input('Enter URL of Image :')
img = imread(url)
plt.imshow(img)
plt.show()
img_resize = resize(img,(256,256,3))
l = [img_resize.flatten()]
probability = model.predict_proba(l)
for ind,val in enumerate(CATEGORIES):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : " + CATEGORIES[model.predict(l)[0]])
"""
# Resource: https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01
# same resource on github: https://github.com/ShanmukhVegi/Image-Classification/blob/main/Shanmukh_Classification.ipynb
