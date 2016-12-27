########################################################################
#
# Functions for downloading a custom data-set from the internet
# and loading it into memory. Note that this only loads the file-names
# for the images in the data-set and does not load the actual images.
#
# Implemented in Python 3.5
#
########################################################################
#
# This file is part of the TensorFlow Tutorials
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
# It has been modified for this repository
#
########################################################################

from dataset import load_cached
import download
import os

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_dir = "data/"

# URL for the data-set on the internet.
data_url = ""

# Dataset name
dataset_name = ""

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 200

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 3

########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract():
    """
    Download and extract the data-set if it doesn't already exist
    in data_dir (set this variable first to the desired directory).
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


def load():
    """
    Load the data-set into memory.

    This uses a cache-file which is reloaded if it already exists,
    otherwise the data-set is created and saved to
    the cache-file. The reason for using a cache-file is that it
    ensure the files are ordered consistently each time the data-set
    is loaded. This is important when the data-set is used in
    combination with Transfer Learning as is done in Tutorial #09.

    :return:
        A DataSet-object for the data-set.
    """

    # Path for the cache-file.
    cache_path = os.path.join(data_dir, dataset_name + ".pkl")

    # If the DataSet-object already exists in a cache-file
    # then load it, otherwise create a new object and save
    # it to the cache-file so it can be loaded the next time.
    dataset = load_cached(cache_path=cache_path,
                          in_dir=data_dir)

    return dataset


########################################################################

if __name__ == '__main__':
    # Download and extract the data-set if it doesn't already exist.
    maybe_download_and_extract()

    # Load the data-set.
    dataset = load()

    # Get the file-paths for the images and their associated class-numbers
    # and class-labels. This is for the training-set.
    image_paths_train, cls_train, labels_train = dataset.get_training_set()

    # Get the file-paths for the images and their associated class-numbers
    # and class-labels. This is for the test-set.
    image_paths_test, cls_test, labels_test = dataset.get_test_set()

    # Check if the training-set looks OK.

    # Print some of the file-paths for the training-set.
    for path in image_paths_train[0:5]:
        print(path)

    # Print the associated class-numbers.
    print(cls_train[0:5])

    # Print the class-numbers as one-hot encoded arrays.
    print(labels_train[0:5])

    # Check if the test-set looks OK.

    # Print some of the file-paths for the test-set.
    for path in image_paths_test[0:5]:
        print(path)

    # Print the associated class-numbers.
    print(cls_test[0:5])

    # Print the class-numbers as one-hot encoded arrays.
    print(labels_test[0:5])

########################################################################
