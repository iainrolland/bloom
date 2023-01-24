import os
import pandas as pd

DATA_DIR = "/media/imr27/SharedDataPartition/Datasets/Tick Tick Bloom"


def load_metadata():
    return pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))


def load_train_labels():
    return pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))


def load_train_merged():
    return load_train_labels().merge(load_metadata(), how="left", on="uid")
