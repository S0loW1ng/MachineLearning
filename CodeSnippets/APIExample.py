import ee
import webbrowser
ee.Initialize()
image = ee.Image('srtm90_v4')
print(image.getInfo())
image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318'); #Landsat 8 image, with Top of Atmosphere processing, on 2014/03/18

#Landsat_composite in Dresden area
area_dresden = list([(13.6, 50.96), (13.9, 50.96), (13.9, 51.12), (13.6, 51.12), (13.6, 50.96)])
area_dresden = ee.Geometry.Polygon(area_dresden)
time_range_dresden = ['2002-07-28', '2002-08-05']

collection_dresden = ('LANDSAT/LE07/C01/T1')
print(type(area_dresden))

#Population density in Switzerland
list_swiss = list([(6.72, 47.88),(6.72, 46.55),(9.72, 46.55),(9.72, 47.88),(6.72, 47.88)])
area_swiss = ee.Geometry.Polygon(list_swiss)
time_range_swiss=['2002-01-01', '2005-12-30']

collection_swiss = ee.ImageCollection('CIESIN/GPWv4/population-density')
print(type(collection_swiss))

#Sentinel 2 cloud-free image in Zürich
collection_zurich = ('COPERNICUS/S2')
list_zurich = list([(8.53, 47.355),(8.55, 47.355),(8.55, 47.376),(8.53, 47.376),(8.53, 47.355)])
area_zurich = ee.Geometry.Polygon(list_swiss)
time_range_zurich = ['2018-05-01', '2018-07-30']


#Landcover in Europe with CORINE dataset
dataset_landcover = ee.Image('COPERNICUS/CORINE/V18_5_1/100m/2012')
landCover_layer = dataset_landcover.select('landcover')
print(type(landCover_layer))
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

def obtain_image_median(collection, time_range, area):
    """ Selection of median from a collection of images in the Earth Engine library
    See also: https://developers.google.com/earth-engine/reducers_image_collection

    Parameters:
        collection (): name of the collection
        time_range (['YYYY-MT-DY','YYYY-MT-DY']): must be inside the available data
        area (ee.geometry.Geometry): area of interest

    Returns:
        image_median (ee.image.Image)
     """
    collection = ee.ImageCollection(collection)

    ## Filter by time range and location
    collection_time = collection.filterDate(time_range[0], time_range[1])
    image_area = collection_time.filterBounds(area)
    image_median = image_area.median()
    return image_median

def obtain_image_sentinel(collection, time_range, area):
    """ Selection of median, cloud-free image from a collection of images in the Sentinel 2 dataset
    See also: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2

    Parameters:
        collection (): name of the collection
        time_range (['YYYY-MT-DY','YYYY-MT-DY']): must be inside the available data
        area (ee.geometry.Geometry): area of interest

    Returns:
        sentinel_median (ee.image.Image)
     """
#First, method to remove cloud from the image
    def maskclouds(image):
        band_qa = image.select('QA60')
        cloud_mask = ee.Number(2).pow(10).int()
        cirrus_mask = ee.Number(2).pow(11).int()
        mask = band_qa.bitwiseAnd(cloud_mask).eq(0) and(
            band_qa.bitwiseAnd(cirrus_mask).eq(0))
        return image.updateMask(mask).divide(10000)

    sentinel_filtered = (ee.ImageCollection(collection).
                         filterBounds(area).
                         filterDate(time_range[0], time_range[1]).
                         filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).
                         map(maskclouds))

    sentinel_median = sentinel_filtered.median()
    return sentinel_median
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

    webbrowser.open_new_tab(path)
    return path

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


region_dresden = get_region(area_dresden)
region_swiss = get_region(area_swiss)
region_zurich= get_region(area_zurich)


composite_dresden = obtain_image_landsat_composite(collection_dresden, time_range_dresden, area_dresden)
median_swiss = obtain_image_median(collection_swiss, time_range_swiss, area_swiss)
zurich_median = obtain_image_sentinel(collection_zurich, time_range_zurich, area_zurich)

#Selection of specific bands from an image
zurich_band = zurich_median.select(['B4','B3','B2'])


print(composite_dresden.getInfo())
print(type(median_swiss))
print(type(zurich_band))
url_swiss = get_url('swiss_pop', median_swiss, 900, region_swiss)
url_dresden = get_url('dresden', composite_dresden, 30, region_dresden)
url_landcover = get_url('landcover_swiss', landCover_layer, 100, region_swiss)

#For the example of Zürich, due to size, it doesn't work on Jupyter Notebook but it works on Python
#url_zurich = get_url('sentinel', zurich_band, 10, region_zurich)

print(url_swiss)
print(url_dresden)
print(url_landcover)
