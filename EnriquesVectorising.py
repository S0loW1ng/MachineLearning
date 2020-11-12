from PIL import ImageOps
from PIL import Image
import numpy as np
from PIL import Image

def imageList(path):
    addresses = glob.glob(path)
    images = []
    for path in addresses:
        img = Image.open(path).convert('L')
        images.append(img)
    return images

def vectorizeImg(img):
    flatten = np.asmatrix(img).flatten()
    flatten = np.divide(flatten,255.0)
    return np.asarray(flatten)[0]

def toVectors(imgList):
    return [vectorizeImg(img) for img in imgList]

path = "/home/pointrain/Documents/MachineLearning/ImageProcessing/dataset/flat/*.png" #Change this to the files we need
listOfImages = imageList(path)
vectorInput = toVectors(listOfImages)
