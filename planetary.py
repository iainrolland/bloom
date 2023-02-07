import pandas as pd
from datetime import timedelta
import planetary_computer as pc
from pystac_client import Client
import geopy.distance as distance
import json
import rioxarray
from rioxarray.exceptions import NoDataInBounds
import odc.stac
import cv2
from tqdm import tqdm
import numpy as np

import data_load
import os
from time import time

IMAGE_ARRAY_DIR = "data/"


def dec2bin_bitwise2(x, bits=8):
    """2d input"""
    shp = x.shape
    return ((x.ravel()[:, None] & (1 << np.arange(bits - 1, -1, -1))) > 0).astype(np.uint8).reshape((*shp, bits))


def is_water(qa):
    return dec2bin_bitwise2(qa, bits=16)[..., -8] == 1


def is_cloud(qa):
    return dec2bin_bitwise2(qa, bits=16)[..., -4] == 1


def is_cloud_shadow(qa):
    return dec2bin_bitwise2(qa, bits=16)[..., -5] == 1


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
    item_details = item_details[item_details["contains_sample_point"] == True]  # filter to only those which contain
    if len(item_details) == 0:
        return np.nan, np.nan, np.nan

    # add time difference between each item and the sample
    item_details["time_diff"] = pd.to_datetime(date) - pd.to_datetime(
        item_details["datetime"]
    )

    # flag: sentinel-2
    item_details["sentinel"] = item_details.platform.str.lower().str.contains(
        "sentinel"
    )
    # flag: landsat
    item_details["landsat"] = item_details.platform.str.lower().str.contains(
        "landsat"
    )
    if item_details["sentinel"].any():  # if sentinel exists, filter to just those
        sentinel = item_details[item_details["sentinel"] == True].sort_values(by="time_diff", ascending=True)
    else:
        sentinel = None
    if item_details["landsat"].any():  # if landsat exists, filter to just those
        landsat = item_details[item_details["landsat"] == True].sort_values(by="time_diff", ascending=True)
    else:
        landsat = None

    # return best_item["item_obj"], best_item["platform"], best_item["datetime"]
    return sentinel, landsat


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


def crop_sentinel_image(item_df, bounding_box):
    """
    Given a STAC item from Sentinel-2 and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    for item in item_df.iterrows():
        image = odc.stac.stac_load([pc.sign(item[1]["item_obj"])],
                                   bands=[f"B0{i}" if i < 10 else f"B{i}" for i in range(1, 13) if i != 10] + ["SCL"],
                                   bbox=(minx, miny, maxx, maxy)).isel(time=0)
        if np.mean(np.isin(image["SCL"].to_numpy(), [0, 1, 2, 3, 8, 9])) > .4:
            # if excl. classes make up >40% of pixels, ignore
            continue  # do the next row in item_df
        elif np.mean(np.isnan(image["SCL"].to_numpy())) > .4:
            # if >40% of pixels are nan
            continue  # do the next row in item_df

        return image, item[1]["item_obj"], item[1]["platform"], item[1]["datetime"]

    return None, None, None, None  # if no images <=40% excluded pixels


def crop_landsat_image(item_df, bounding_box):
    """
    Given a STAC item from Landsat and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box
    bands = {
        "landsat-7": ["blue", "green", "red", "nir08", "swir16", "lwir", "swir22", "qa_pixel"],
        "landsat-8": ["blue", "green", "red", "nir08", "swir16", "lwir11", "swir22", "qa_pixel"],
        "landsat-9": ["blue", "green", "red", "nir08", "swir16", "lwir11", "swir22", "qa_pixel"]
    }

    for item in item_df.iterrows():
        image = odc.stac.stac_load(
            [pc.sign(item[1]["item_obj"])],
            bands=bands[item[1].platform],
            bbox=(minx, miny, maxx, maxy)
        ).isel(time=0)
        cloud_mask = is_cloud(image["qa_pixel"].to_numpy())
        cloud_shadow_mask = is_cloud_shadow(image["qa_pixel"].to_numpy())
        if np.mean(np.bitwise_or(cloud_mask, cloud_shadow_mask)) > 0.4:  # excluded if clouds/shadows >40% of image
            continue  # do the next row in item_df
        elif np.mean(image.to_array()[:-1, ...] == 0) > .1:  # if >10% RGB band pixels have zero data, disregard
            continue  # image ignored - fails test for missing values
        return image.astype(np.int32), item[1]["item_obj"], item[1]["platform"], item[1]["datetime"]

    return None, None, None, None


def download():
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )

    metadata = data_load.load_metadata()

    for row in tqdm(metadata.itertuples(), total=len(metadata)):
        sen_pth = os.path.join(IMAGE_ARRAY_DIR, "sentinel", f"{row.uid}.nc")
        landsat_pth = os.path.join(IMAGE_ARRAY_DIR, "landsat", f"{row.uid}.nc")
        sen_metadata_pth = os.path.join(IMAGE_ARRAY_DIR, "sentinel", f"{row.uid}_metadata.json")
        landsat_metadata_pth = os.path.join(IMAGE_ARRAY_DIR, "landsat", f"{row.uid}_metadata.json")

        if not os.path.isfile(sen_pth) or not os.path.isfile(landsat_pth):  # if we haven't already got images
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

                if len(items) == 0:
                    raise NoDataInBounds
                else:
                    sentinel, landsat = select_best_item(items, row.date, row.latitude, row.longitude)

                feature_bbox = get_bounding_box(
                    row.latitude, row.longitude, meter_buffer=100
                )

                if sentinel is not None:
                    sentinel_data, best_item, item_platform, item_date = crop_sentinel_image(
                        sentinel,
                        feature_bbox
                    )
                    if sentinel_data is not None:
                        sentinel_data.to_netcdf(sen_pth)
                        with open(sen_metadata_pth, 'w') as f:
                            json.dump({
                                "item_object": best_item.id,
                                "item_platform": item_platform,
                                "item_date": item_date,
                            }, f)

                if landsat is not None:
                    landsat_data, best_item, item_platform, item_date = crop_landsat_image(
                        landsat,
                        feature_bbox
                    )
                    if landsat_data is not None:
                        landsat_data.to_netcdf(landsat_pth)
                        with open(landsat_metadata_pth, 'w') as f:
                            json.dump({
                                "item_object": best_item.id,
                                "item_platform": item_platform,
                                "item_date": item_date,
                            }, f)

            # keep track of any that ran into errors without interrupting the process
            except NoDataInBounds:
                print(row.uid)


if __name__ == '__main__':
    download()
