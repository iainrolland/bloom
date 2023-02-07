import code
import matplotlib.pyplot as plt
import cv2
import json
import xarray as xr
import pickle
import odc.stac
import planetary_computer as pc
from planetary import is_water
import numpy as np
import os
from glob import glob

import data_load


class NoDataError(Exception):
    pass


def shape_standardise(array, interpolation):
    return cv2.resize(array, (21, 21), interpolation=interpolation)


def get_landsat_image(uid):
    try:
        bands, qa = get_file(os.path.join("data", "landsat", f"{uid}.nc"))
        return bands, is_water(qa)
    except NoDataError:
        return None


def nvdi(array, platform):
    supported = ["sentinel", "landsat"]
    if platform not in supported:
        raise ValueError(f"platform must be one of {supported} but was '{platform}'")
    else:
        # TODO: check bands, especially if landsat 7/8/9 are different from one another
        if platform == "sentinel":
            plus_band, minus_band = 7, 3
        else:
            plus_band, minus_band = 3, 2
        return (array[..., plus_band] - array[..., minus_band]) / (array[..., plus_band] + array[..., minus_band])


def get_sen2_image(uid):
    try:
        bands, scl = get_file(os.path.join("data", "sentinel", f"{uid}.nc"))
        return bands, scl == 6
    except NoDataError:
        return None


def get_landsat_metadata(uid):
    with open(os.path.join("data", "landsat", f"{uid}_metadata.json"), 'r') as f:
        metadata = json.load(f)
    return metadata


def get_sen2_metadata(uid):
    with open(os.path.join("data", "sentinel", f"{uid}_metadata.json"), 'r') as f:
        metadata = json.load(f)
    return metadata


def get_file(path):
    if os.path.isfile(path):
        data = xr.open_dataset(path)
        bands, qa = np.split(
            np.stack([data.data_vars[key].to_numpy() for key in data.data_vars.keys() if key != "spatial_ref"], axis=-1)
            , [-1], axis=-1)
        bands = bands.astype(float)  # move band axis to last
        # bands = np.stack([cv2.normalize(b, None, 0, 1, cv2.NORM_MINMAX) for b in bands], axis=-1)
        return (
            shape_standardise(bands, cv2.INTER_LINEAR),  # the data bands
            shape_standardise(qa, cv2.INTER_NEAREST).squeeze()  # the qa/scl mask
        )
    else:  # if we don't have data from that source for that uid
        raise NoDataError


def get_user_ip():
    label = input("Use (S)entinel 2 mask or (L)andat mask or (I)gnore?")
    if label in ["S", "L", "I"]:
        return label
    else:
        get_user_ip()


def show_data(uid):
    landsat_image = get_landsat_image(uid)
    sen2_image = get_sen2_image(uid)
    if landsat_image is not None or sen2_image is not None:
        fig, ax = plt.subplots(3, 2, figsize=(20, 20))
        if landsat_image is not None:
            ax[1, 0].set_title("NVDI")
            ax[1, 0].matshow(nvdi(landsat_image[0], "landsat"), vmin=-1, vmax=1, cmap="bwr")  # water NVDI blue
            ax[0, 0].imshow(cv2.normalize(landsat_image[0][..., [2, 1, 0]], None, 0.001, .999, cv2.NORM_MINMAX))  # RGB
            ax[0, 0].set_title(f"{landsat_image[1].shape}")
            ax[2, 0].matshow(landsat_image[1], vmin=0, vmax=1)  # 1 if is water, 0 otherwise
        else:
            ax[0, 0].set_title("No Landsat")
        if sen2_image is not None:
            ax[1, 1].set_title("NVDI")
            ax[1, 1].matshow(nvdi(sen2_image[0], "sentinel"), vmin=-1, vmax=1, cmap="bwr")  # water NVDI blue
            ax[0, 1].imshow(cv2.normalize(sen2_image[0][..., [3, 2, 1]], None, 0.001, .999, cv2.NORM_MINMAX))  # RGB
            ax[0, 1].set_title(f"{sen2_image[1].shape}")
            ax[2, 1].matshow(sen2_image[1], vmin=0, vmax=1)  # 1 if is water, 0 otherwise
        else:
            ax[1, 1].set_title("No Sentinel")
        plt.waitforbuttonpress(0)  # this will wait for indefinite time
        plt.close(fig)


def main():
    uid = data_load.load_metadata().uid.values
    np.random.shuffle(uid)
    for one in uid:
        show_data(one)


if __name__ == '__main__':
    main()
