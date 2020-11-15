

from PIL import ImageOps
from PIL import Image
import numpy as np
from PIL import Image

#added by David
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

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

path_notflat = "C:/Users/jojo5/Documents/GitHub/MachineLearning/ImageProcessing/dataset/notflat/*.png" #Change this to the files we need and use forward slashes
listOfImages = imageList(path_notflat)
vectorInput_notflat = toVectors(listOfImages)

path_flat = "C:/Users/jojo5/Documents/GitHub/MachineLearning/ImageProcessing/dataset/flat/*.png" #Change this to the files we need and use forward slashes
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

excess = list(range(int(numberOfNotFlat), int(numberOfFlat))) #normalizing the number of flat vs nonflat images
df_flat.drop(excess, axis = 0)

# Combine flat and nonflat dataframes

frames = [df_notflat, df_flat]
dataset = pd.concat(frames) #this is our full dataset, it has 4096 columns where the first is the label and the rest are the pixel values, and the rows are the number of images

X = dataset.drop('flat', axis=1) #these are all of our input images
Y = dataset['flat'] #these are all of the flat vs nonflat labels, flat = 1, nonflat = 0

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8) #create a training and testing dataset

