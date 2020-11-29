# ENGR418 UAV Landing support

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import time
import seaborn as sns

sns.set()
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

start_time = time.time()
plt.style.use('seaborn')
# 1. Image pre-processing code (can refer to a different script, just need to be run once...)]

# Dataset was normalized to 64x64 grayscale images in a different script

# 2. Processing normalized dataset

# Vectorizing images

def imageList(path):
    addresses = glob.glob(path)
    images = []
    for path in addresses:
        img = Image.open(path).convert('L')
        images.append(img)
    return images


def vectorizeImg(img):
    flatten = np.asmatrix(img).flatten()  # converts to a matrix then flattens to vector
   # flatten = np.divide(flatten, 255.0)  # normalizes the elements from 0-255 to 0-1 ; Andres: Actually want to wait after mean centering data for PCA
    return np.asarray(flatten)[0]


def toVectors(imgList):
    return [vectorizeImg(img) for img in imgList]


path_notflat = "C:/Users/andy9/Documents/GitHub/Project_MachineLearning/ImageProcessing/dataset/gnotflat/*.png"
# Change path_notflat to the files we need and use forward slashes
listOfImages = imageList(path_notflat)
vectorInput_notflat = toVectors(listOfImages)

path_flat = "C:/Users/andy9/Documents/GitHub/Project_MachineLearning/ImageProcessing/dataset/gflat/*.png"
# Change path_flat to the files we need and use forward slashes
listOfImages = imageList(path_flat)
vectorInput_flat = toVectors(listOfImages)

# Nonflat images

numberOfNotFlat = np.size(vectorInput_notflat) / np.size(vectorInput_notflat[1])  # number of notflat images

df_notflat = pd.DataFrame(vectorInput_notflat)  # converts the list into a dataframe

df_notflat.insert(0, "flat", [0] * int(numberOfNotFlat), True)  # add new column to data fram to denote flat vs nonflat

# Flat images

numberOfFlat = np.size(vectorInput_flat) / np.size(vectorInput_flat[1])  # number of notflat images

df_flat = pd.DataFrame(vectorInput_flat)  # converts the list into a dataframe

# add first column to dataframe to denote flat vs nonflat
df_flat.insert(0, "flat", [1] * int(numberOfFlat), True)

excess = list(range(int(numberOfNotFlat), int(numberOfFlat)))  # normalizing the number of flat vs nonflat images
df_flat = df_flat.drop(excess, axis=0)

# Combine flat and nonflat dataframes

frames = [df_notflat, df_flat]
dataset = pd.concat(frames)  # this is our full dataset, it has 4096 columns where the first is the label and the...
# rest are the pixel values, and the rows are the number of images

X = dataset.drop('flat', axis=1)  # these are all of our input images
Y = dataset['flat']  # these are all of the flat vs nonflat labels, flat = 1, nonflat = 0

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=2)  # create a training and testing dataset
# 3. Reducing image dimensionality through PCA
# PCA uses all of the data points and relates them to get the basis vectors(c1,c2,...,cn), we have 960 images, and...
# 4096 pixels, (diff dimensions)

# You are supposed to do to the test data exactly the same transformation that you did to the training data; this
# includes centering -- it should be done using the mean values obtained on the training set. If you standardized the
# training set, then you would also divide your test set by the standard deviations obtained on the training set.
# After that, you can project your test set onto the PCs of the training set.
# Can also be found on Lecture 10 slide 13?, should've seen that before...
# https://stats.stackexchange.com/questions/142216/zero-centering-the-testing-set-after-pca-on-the-training-set
# https://stats.stackexchange.com/questions/55718/pca-and-the-train-test-split
# https://datascience.stackexchange.com/questions/45900/when-to-use-standard-scaler-and-when-normalizer
# X_std_train = StandardScaler().fit_transform(X_train)
# X_nml_train = Normalizer().fit_transform(X_train)
meanTrainSet = X_train.mean(axis=0)  #Axis 0 is column means in 2D matrix, which is what we want
StdTrainSet = X_train.std(axis=0)
X_Centered_Train = (X_train - meanTrainSet)/StdTrainSet
pca = PCA(n_components=200)
X_pca_train = pca.fit_transform(X_Centered_Train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
Xtest_centered = (X_test - meanTrainSet)/StdTrainSet
X_pca_test_centered = pca.transform(Xtest_centered) #pca extracted from training set applied to test set
# 3. Classifier (MLP), Act function, optimizer
# X is the dataset array
# Y is the label

model = MLPClassifier(hidden_layer_sizes=(60, 40, 20), activation='relu', solver='adam', max_iter=100000,
                      random_state=0)  # 10,10,10 can play with these and iterations, 3 layers of 10 nodes each)

model.fit(X_pca_train, y_train)
ypred = model.predict(X_pca_test_centered)

print(metrics.classification_report(ypred, y_test))

# 4. Some plots to help with the report/presentation?

mat = confusion_matrix(y_test, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_pca_train[:, 0], X_pca_train[:, 1], X_pca_train[:, 2], c=y_train, alpha=0.8,
           cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
ax.set_zlabel('component 3')
plt.show()

plt.scatter(X_Centered_Train[0], X_Centered_Train[1], c=y_train)
plt.xlim(-1, 10)
plt.ylim(-1, 10)
plt.show()

X_recovered = pca.inverse_transform(X_pca_test_centered)
X_test_np = X_test.values

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("original")
plt.imshow(X_test_np[3].reshape((64, 64)), cmap='gray')
f.add_subplot(1, 2, 2)

plt.title("PCA compressed")
plt.imshow(X_recovered[3].reshape((64, 64)), cmap='gray')
plt.show(block=True)

total_time = time.time() - start_time
