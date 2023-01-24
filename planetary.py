import pandas as pd
from datetime import timedelta
import planetary_computer as pc
from pystac_client import Client
import geopy.distance as distance
import rioxarray
from IPython.display import Image
from PIL import Image as PILImage
from rioxarray.exceptions import NoDataInBounds
import odc.stac
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import data_load
import os

IMAGE_ARRAY_DIR = "images/"


def select_best_item(items, date, latitude, longitude):
    """
    Select the best satellite item given a sample's date, latitude, and longitude.
    If any Sentinel-2 imagery is available, returns the closest sentinel-2 image by
    time. Otherwise, returns the closest Landsat imagery.

    Returns a tuple of (STAC item, item platform name, item date)
    """
    # get item details
    item_details = pd.DataFrame(
        [
            {
                "datetime": item.datetime.strftime("%Y-%m-%d"),
                "platform": item.properties["platform"],
                "min_long": item.bbox[0],
                "max_long": item.bbox[2],
                "min_lat": item.bbox[1],
                "max_lat": item.bbox[3],
                "item_obj": item,
            }
            for item in items
        ]
    )

    # filter to items that contain the point location, or return None if none contain the point
    item_details["contains_sample_point"] = (
            (item_details.min_lat < latitude)
            & (item_details.max_lat > latitude)
            & (item_details.min_long < longitude)
            & (item_details.max_long > longitude)
    )
    item_details = item_details[item_details["contains_sample_point"] == True]
    if len(item_details) == 0:
        return np.nan, np.nan, np.nan

    # add time difference between each item and the sample
    item_details["time_diff"] = pd.to_datetime(date) - pd.to_datetime(
        item_details["datetime"]
    )

    # if we have sentinel-2, filter to sentinel-2 images only
    item_details["sentinel"] = item_details.platform.str.lower().str.contains(
        "sentinel"
    )
    if item_details["sentinel"].any():
        item_details = item_details[item_details["sentinel"] == True]

    # return the closest imagery by time
    best_item = item_details.sort_values(by="time_diff", ascending=True).iloc[0]

    return best_item["item_obj"], best_item["platform"], best_item["datetime"]


def get_date_range(date, time_buffer_days=15):
    """Get a date range to search for in the planetary computer based
    on a sample's date. The time range will include the sample date
    and time_buffer_days days prior

    Returns a string"""
    datetime_format = "%Y-%m-%dT"
    range_start = pd.to_datetime(date) - timedelta(days=time_buffer_days)
    date_range = f"{range_start.strftime(datetime_format)}/{pd.to_datetime(date).strftime(datetime_format)}"

    return date_range


def get_bounding_box(latitude, longitude, meter_buffer=50000):
    """
    Given a latitude, longitude, and buffer in meters, returns a bounding
    box around the point with the buffer on the left, right, top, and bottom.

    Returns a list of [minx, miny, maxx, maxy]
    """
    distance_search = distance.distance(meters=meter_buffer)

    # calculate the lat/long bounds based on ground distance
    # bearings are cardinal directions to move (south, west, north, and east)
    min_lat = distance_search.destination((latitude, longitude), bearing=180)[0]
    min_long = distance_search.destination((latitude, longitude), bearing=270)[1]
    max_lat = distance_search.destination((latitude, longitude), bearing=0)[0]
    max_long = distance_search.destination((latitude, longitude), bearing=90)[1]

    return [min_long, min_lat, max_long, max_lat]


def crop_sentinel_image(item, bounding_box):
    """
    Given a STAC item from Sentinel-2 and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = rioxarray.open_rasterio(pc.sign(item.assets["visual"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

    return image.to_numpy()


def crop_landsat_image(item, bounding_box):
    """
    Given a STAC item from Landsat and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    image = odc.stac.stac_load(
        [pc.sign(item)], bands=["red", "green", "blue"], bbox=[minx, miny, maxx, maxy]
    ).isel(time=0)
    image_array = image[["red", "green", "blue"]].to_array().to_numpy()

    # normalize to 0 - 255 values
    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)

    return image_array


def main():
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    metadata = data_load.load_metadata()
    example_row = metadata[metadata.uid == "garm"].iloc[0]
    date_range = get_date_range(example_row.date)
    bbox = get_bounding_box(example_row.latitude, example_row.longitude, meter_buffer=50000)

    search = catalog.search(
        collections=["sentinel-2-l2a", "landsat-c2-l2"], bbox=bbox, datetime=date_range
    )
    items = [item for item in search.item_collection()]
    print(items)

    # get details of all of the items returned
    item_details = pd.DataFrame(
        [
            {
                "datetime": item.datetime.strftime("%Y-%m-%d"),
                "platform": item.properties["platform"],
                "min_long": item.bbox[0],
                "max_long": item.bbox[2],
                "min_lat": item.bbox[1],
                "max_lat": item.bbox[3],
                "bbox": item.bbox,
                "item_obj": item,
            }
            for item in items
        ]
    )

    # check which rows actually contain the sample location
    item_details["contains_sample_point"] = (
            (item_details.min_lat < example_row.latitude)
            & (item_details.max_lat > example_row.latitude)
            & (item_details.min_long < example_row.longitude)
            & (item_details.max_long > example_row.longitude)
    )

    print(
        f"Filtering from {len(item_details)} returned to {item_details.contains_sample_point.sum()} items that contain the sample location"
    )

    item_details = item_details[item_details["contains_sample_point"]]
    item_details[["datetime", "platform", "contains_sample_point", "bbox"]].sort_values(
        by="datetime"
    )
    # filter to sentinel and take the closest date
    best_item = (
        item_details[item_details.platform.str.contains("Sentinel")]
        .sort_values(by="datetime", ascending=False)
        .iloc[0]
    )
    item = best_item.item_obj
    # get a smaller geographic bounding box
    minx, miny, maxx, maxy = get_bounding_box(
        example_row.latitude, example_row.longitude, meter_buffer=3000
    )

    # get the zoomed in image array
    bbox = (minx, miny, maxx, maxy)
    zoomed_img_array = crop_sentinel_image(item, bbox)

    fig, ax = plt.subplots()
    ax.imshow(np.rollaxis(zoomed_img_array, 0, 3))
    plt.show()


def download():
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )

    # save outputs in dictionaries
    selected_items = {}
    features_dict = {}
    errored_ids = []

    metadata = data_load.load_metadata()

    for row in tqdm(metadata.itertuples(), total=len(metadata)):
        # check if we've already saved the selected image array
        image_array_pth = os.path.join(IMAGE_ARRAY_DIR, f"{row.uid}.npy")

        if not os.path.isfile(image_array_pth):
            try:
                # QUERY STAC API
                # get query ranges for location and date
                search_bbox = get_bounding_box(
                    row.latitude, row.longitude, meter_buffer=50000
                )
                date_range = get_date_range(row.date, time_buffer_days=15)

                # search the planetary computer
                search = catalog.search(
                    collections=["sentinel-2-l2a", "landsat-c2-l2"],
                    bbox=search_bbox,
                    datetime=date_range,
                )
                items = [item for item in search.item_collection()]

                # GET BEST IMAGE
                if len(items) == 0:
                    raise Exception
                else:
                    best_item, item_platform, item_date = select_best_item(
                        items, row.date, row.latitude, row.longitude
                    )
                    # add to dictionary tracking best items
                    selected_items[row.uid] = {
                        "item_object": best_item,
                        "item_platform": item_platform,
                        "item_date": item_date,
                    }

                feature_bbox = get_bounding_box(
                    row.latitude, row.longitude, meter_buffer=100
                )

                # crop the image
                if "sentinel" in item_platform.lower():
                    image_array = crop_sentinel_image(best_item, feature_bbox)
                else:
                    image_array = crop_landsat_image(best_item, feature_bbox)

                # save image array so we don't have to rerun
                with open(image_array_pth, "wb") as f:
                    np.save(image_array_pth, image_array)

            # keep track of any that ran into errors without interrupting the process
            except NoDataInBounds:
                errored_ids.append(row.uid)
                print(row.uid)


if __name__ == '__main__':
    # print(df[df.split == "train"])  # 17060 training samples
    # main()
    download()
