# ENGR418 UAV Landing support

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # ; sns.set()
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from PIL import ImageOps
from PIL import Image
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Image pre-processing code (can refer to a different script, just need to be run once...)]

# Dataset was normalized to 64x64 grayscale images in a different script

# Vectorizing images
def imageList(path):
    addresses = glob.glob(path)
    images = []
    for path in addresses:
        img = Image.open(path).convert('L')
        images.append(img)
    return images

def vectorizeImg(img):
    flatten = np.asmatrix(img).flatten() #converts to a matrix then flattens to vector
    flatten = np.divide(flatten,255.0) # normalizes the elements from 0-255 to 0-1
    return np.asarray(flatten)[0]

def toVectors(imgList):
    return [vectorizeImg(img) for img in imgList]

# Building dataset frame with Pandas

path_notflat = "C:/Users/andy9/Documents/GitHub/Project_MachineLearning/ImageProcessing/dataset/notflat/*.png" #Change this to the files we need and use forward slashes
listOfImages = imageList(path_notflat)
vectorInput_notflat = toVectors(listOfImages)

path_flat = "C:/Users/andy9/Documents/GitHub/Project_MachineLearning/ImageProcessing/dataset/flat/*.png" #Change this to the files we need and use forward slashes
listOfImages = imageList(path_flat)
vectorInput_flat = toVectors(listOfImages)

# Nonflat images

numberOfNotFlat = np.size(vectorInput_notflat)/np.size(vectorInput_notflat[1]) #number of notflat images

df_notflat = pd.DataFrame(vectorInput_notflat) #converts the list into a dataframe

df_notflat.insert(0, "flat", [0] * int(numberOfNotFlat), True) #add new column to data fram to denote flat vs nonflat

#Flat images

numberOfFlat = np.size(vectorInput_flat)/np.size(vectorInput_flat[1]) #number of notflat images

df_flat = pd.DataFrame(vectorInput_flat) #converts the list into a dataframe

#add first column to data fram to denote flat vs nonflat
df_flat.insert(0, "flat", [1] * int(numberOfFlat), True)

# Combine flat and nonflat dataframes

frames = [df_notflat, df_flat]
dataset = pd.concat(frames) #this is our full dataset, it has 4096 columns where the first is the label and the rest are the pixel values, and the rows are the number of images

X = dataset.drop('flat', axis=1) #these are all of our input images
Y = dataset['flat'] #these are all of the flat vs nonflat labels, flat = 1, nonflat = 0

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)#,random_state=0) #create a training and testing dataset
# Import dataset (Vectorized images) as pandas dataframe?


# 2. Processing dataset through

# PCA/Autoencoder/anything else?


# 3. Classifier (MLP), Act function, optimizer
# X is the dataset array
# Y is the label

#X_train, X_test, y_train, y_test = train_test_split(dataset, dataset.target, test_size=.2, random_state=0)
model = MLPClassifier(hidden_layer_sizes=(10000,5000,1000,100), activation='relu', solver='adam',
                      max_iter=100000)  # 10,10,10 can play with these and iterations, 3 layers of 10 nodes each)

model.fit(X_train, y_train)
ypred = model.predict(X_test)

print(metrics.classification_report(ypred, y_test))

# 4. Some plots to help with the report/presentation?

mat = confusion_matrix(y_test, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
