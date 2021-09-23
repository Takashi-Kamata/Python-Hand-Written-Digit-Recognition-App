import time
import sys
from os import path
import os


# calculate downloaded mnist data folder size, outputs in percentage
# called for updating progress bar
# INPUT: Folder location, default set to mnist_data
# OUTPUT: download progress in %
def get_size(start_path='mnist_data/'):
    total_size = 121999248  # mnist final size
    temp_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                temp_size += os.path.getsize(fp)

    return temp_size/total_size * 100
