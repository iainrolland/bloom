import os
import matplotlib.pyplot as plt
import numpy as np

import data_load


def show_data(uid, plot=True):
    landsat_water_mask_path = os.path.join("images", f"landsat_{uid}_water_mask.npy")
    landsat_path = os.path.join("images", f"landsat_{uid}_img.npy")
    sentinel_path = os.path.join("images", f"sen2_{uid}_img.npy")
    sentinel_scl_path = os.path.join("images", f"sen2_{uid}_scl.npy")
    if os.path.isfile(landsat_path) and os.path.isfile(sentinel_path):
        landsat_image = np.load(landsat_path)
        landsat_water_mask = np.load(landsat_water_mask_path)
        sen2_image = np.load(sentinel_path)
        sen2_scl = np.load(sentinel_scl_path)
        if plot:
            fig, ax = plt.subplots(2, 2)
            fig.suptitle(uid)
            ax[0, 0].imshow(np.rollaxis(landsat_image, 0, 3))
            ax[0, 1].matshow(landsat_water_mask, vmin=0, vmax=1)
            ax[1, 0].imshow(np.rollaxis(sen2_image, 0, 3))
            ax[1, 1].matshow(sen2_scl[0])
            plt.show()


def main():
    # uid = data_load.load_metadata().uid.values
    good_image_pairs = ["aaff", "aafl", "aays", "abzk", "aclb"]
    for one in good_image_pairs:
        show_data(one, plot=True)


if __name__ == '__main__':
    main()
