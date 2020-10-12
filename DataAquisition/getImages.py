import image_slicer
import ee
import os
import glob
import cv2

def get_region(geom):
    """Get the region of a given geometry, needed for exporting tasks.

    Parameters:
        geom (ee.Geometry, ee.Feature, ee.Image): region of interest

    Returns:
        region (list)
    """
    if isinstance(geom, ee.Geometry):
        region = geom.getInfo()["coordinates"]
    elif isinstance(geom, ee.Feature, ee.Image):
        region = geom.geometry().getInfo()["coordinates"]
    elif isinstance(geom, list):
        condition = all([isinstance(item) == list for item in geom])
        if condition:
            region = geom
    return region
def obtain_image_landsat_composite(collection, time_range, area):
    """ Selection of Landsat cloud-free composites in the Earth Engine library
    See also: https://developers.google.com/earth-engine/landsat

    Parameters:
        collection (): name of the collection
        time_range (['YYYY-MT-DY','YYYY-MT-DY']): must be inside the available data
        area (ee.geometry.Geometry): area of interest

    Returns:
        image_composite (ee.image.Image)
     """
    collection = ee.ImageCollection(collection)

    ## Filter by time range and location
    collection_time = collection.filterDate(time_range[0], time_range[1])
    image_area = collection_time.filterBounds(area)
    image_composite = ee.Algorithms.Landsat.simpleComposite(image_area, 75, 3)
    return image_composite
def get_url(name, image, scale, region):
    """It will open and download automatically a zip folder containing Geotiff data of 'image'.
    If additional parameters are needed, see also:
    https://github.com/google/earthengine-api/blob/master/python/ee/image.py

    Parameters:
        name (str): name of the created folder
        image (ee.image.Image): image to export
        scale (int): resolution of export in meters (e.g: 30 for Landsat)
        region (list): region of interest

    Returns:
        path (str)
     """
    path = image.getDownloadURL({
        'name':(name),
        'scale': scale,
        'region':(region)
        })

   # webbrowser.open_new_tab(path)
    return path
def downloadFile(url):
   outputDir ='./' 
   print('Downloading..')
   os.system('wget ' + url + ' -O files:getpixels')
   print('Success!') 
   print('Extracting files..')
   os.system('mkdir files')
   os.system('unzip files:getpixels -d ./files')
   print('Extraced!')
def imageResized():
    image_slicer.slice('./files/dresden.B5.tif' , 10) # We can change the name of the images 
    listOfFiles = glob.glob('./files/*.png')
    imageNum = 0
    for i in listOfFiles:
        img = cv2.imread(i,cv2.IMREAD_UNCHANGED) 
        dim = (60,60)
        resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
        writenBack = cv2.imwrite('./resized/' + str(imageNum) + '.png', resized)
        if writenBack:
            print("saved " + str(imageNum))
        imageNum = imageNum +1
    



ee.Initialize()
image = ee.Image('srtm90_v4')
image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318'); #Landsat 8 image, with Top of Atmosphere processing, on 2014/03/18

#Landsat_composite in Dresden area
area_dresden = list([(13.6, 50.96), (13.9, 50.96), (13.9, 51.12), (13.6, 51.12), (13.6, 50.96)])
area_dresden = ee.Geometry.Polygon(area_dresden)
time_range_dresden = ['2002-07-28', '2002-08-05']

collection_dresden = ('LANDSAT/LE07/C01/T1')


region_dresden = get_region(area_dresden)
composite_dresden = obtain_image_landsat_composite(collection_dresden, time_range_dresden, area_dresden)
url_dresden = get_url('dresden', composite_dresden, 30, region_dresden)
print(url_dresden)

#downloadFile(url_dresden)
imageResized()


