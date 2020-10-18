import cv2
import image_slicer
import glob 

pathsNotFlat = glob.glob('/home/pointrain/notflat/*.png')
pathsFlat = glob.glob('/home/pointrain/flat/*.png')
counter = 0

#Slice the images 

for i in pathsNotFlat:
    tiles = image_slicer.slice(i, 5, save = False)
    image_slicer.save_tiles(tiles,directory='./notflat',prefix = 'notflat' + str(counter),format='png')
    counter = counter +1

counter = 0
for i in pathsFlat:
    tiles = image_slicer.slice(i, 7, save = False)
    image_slicer.save_tiles(tiles,directory='./flat',prefix = 'flat' + str(counter),format='png')
    counter = counter +1



slicedFlat = glob.glob('./flat/*.png')
slicedNotFlat = glob.glob('./notflat/*.png')
counter = 0
dim = (64,64)
for i in slicedFlat:
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED) 
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayImage,dim,interpolation = cv2.INTER_AREA)
    cv2.imwrite('./dataset/flat/flat' + str(counter) + '.png',resized)
    laplace = cv2.Laplacian(resized,cv2.CV_64F)
    cv2.imwrite('./dataset/gflat/gflat' + str(counter) + '.png',laplace)
    counter = counter +1
counter = 0
for i in slicedNotFlat:
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED) 
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayImage,dim,interpolation = cv2.INTER_AREA)
    cv2.imwrite('./dataset/notflat/notflat' + str(counter) + '.png',resized)
    laplace = cv2.Laplacian(resized,cv2.CV_64F)
    cv2.imwrite('./dataset/gnotflat/notgflat' + str(counter) + '.png',laplace)
    counter = counter +1