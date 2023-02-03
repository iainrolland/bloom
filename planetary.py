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

IMAGE_ARRAY_DIR = "images/"


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
        image = rioxarray.open_rasterio(pc.sign(item[1]["item_obj"].assets["visual"].href)).rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            crs="EPSG:4326",
        ).to_numpy()
        cloud = rioxarray.open_rasterio(pc.sign(item[1]["item_obj"].assets["SCL"].href)).rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            crs="EPSG:4326",
        ).to_numpy()
        if np.mean(np.isin(cloud, [0, 1, 2, 3, 8, 9])) > .4:  # if excluded classes account for >40% of pixels, ignore
            continue  # do the next row in item_df

        return image, cloud, item[1]["item_obj"], item[1]["platform"], item[1]["datetime"]

    return None, None, None, None, None  # if no images <=40% excluded pixels


def crop_landsat_image(item_df, bounding_box):
    """
    Given a STAC item from Landsat and a bounding box tuple in the format
    (minx, miny, maxx, maxy), return a cropped portion of the item's visual
    imagery in the bounding box.

    Returns the image as a numpy array with dimensions (color band, height, width)
    """
    (minx, miny, maxx, maxy) = bounding_box

    for item in item_df.iterrows():
        image = odc.stac.stac_load(
            [pc.sign(item[1]["item_obj"])], bands=["red", "green", "blue", "qa_pixel"], bbox=[minx, miny, maxx, maxy]
        ).isel(time=0)
        image_array = image[["red", "green", "blue", "qa_pixel"]].to_array().to_numpy()
        cloud_mask = is_cloud(image_array[-1, ...])
        cloud_shadow_mask = is_cloud_shadow(image_array[-1, ...])
        water_mask = is_water(image_array[-1, ...])
        if np.mean(np.bitwise_or(cloud_mask, cloud_shadow_mask)) > 0.4:  # excluded if clouds/shadows >40% of image
            continue  # do the next row in item_df
        elif np.mean(image_array[:3] == 0) > .1:  # if >10% RGB band pixels have zero data, disregard
            continue  # image ignored - fails test for missing values

        # normalize to 0 - 255 values
        image_array = cv2.normalize(image_array[:3], None, 0, 255, cv2.NORM_MINMAX)
        return image_array, water_mask, item[1]["item_obj"], item[1]["platform"], item[1]["datetime"]

    return None, None, None, None, None


def download():
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )

    metadata = data_load.load_metadata()

    for row in tqdm(metadata.itertuples(), total=len(metadata)):
        sen_img_pth = os.path.join(IMAGE_ARRAY_DIR, f"sen2_{row.uid}_img.npy")
        sen_scl_pth = os.path.join(IMAGE_ARRAY_DIR, f"sen2_{row.uid}_scl.npy")
        sen_metadata_pth = os.path.join(IMAGE_ARRAY_DIR, f"sen2_{row.uid}_metadata.json")
        landsat_img_pth = os.path.join(IMAGE_ARRAY_DIR, f"landsat_{row.uid}_img.npy")
        landsat_water_pth = os.path.join(IMAGE_ARRAY_DIR, f"landsat_{row.uid}_water_mask.npy")
        landsat_metadata_pth = os.path.join(IMAGE_ARRAY_DIR, f"landsat_{row.uid}_metadata.json")

        if not os.path.isfile(sen_img_pth) or not os.path.isfile(landsat_img_pth):  # if we haven't already got images
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
                    sentinel_img, sentinel_cloud, best_item, item_platform, item_date = crop_sentinel_image(
                        sentinel,
                        feature_bbox
                    )
                    if sentinel_img is not None:
                        np.save(sen_img_pth, sentinel_img)
                        np.save(sen_scl_pth, sentinel_cloud)
                        with open(sen_metadata_pth, 'w') as f:
                            json.dump({
                                "item_object": best_item.id,
                                "item_platform": item_platform,
                                "item_date": item_date,
                            }, f)

                if landsat is not None:
                    landsat_img, landsat_water_mask, best_item, item_platform, item_date = crop_landsat_image(
                        landsat,
                        feature_bbox
                    )
                    if landsat_img is not None:
                        np.save(landsat_img_pth, landsat_img)
                        np.save(landsat_water_pth, landsat_water_mask)
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
