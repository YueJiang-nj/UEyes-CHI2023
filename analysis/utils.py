#!/usr/bin/env python3
# coding: utf-8

'''
Utility functions.

External dependencies, to be installed e.g. via pip:
- Pillow
- numpy
- imagesize

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import os
import csv
import numpy as np
import imagesize
from PIL import Image


def memoize(f):
    '''
    Memoization helper.
    '''
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper


@memoize
def load_screenshot(filepath, mode='RGB'):
    '''
    Load screenshot as a pixel matrix.
    '''
    return Image.open(filepath).convert(mode)


def load_files(dirname, mode='dict', allow_ext=('.jpg','.jpeg','.png','.csv')):
    '''
    Read valid files in dir, according ot their file extension.
    Default allowed file extensions are: .png, .jpg, .jpeg, .csv
    '''
    img_list = []
    img_dict = {}
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if not f.endswith(tuple(allow_ext)) or f in img_dict:
                continue

            img_path = os.path.join(root, f)
            if mode == 'dict':
                img_dict[f] = img_path
            else:
                img_list.append(img_path)

    if mode == 'dict':
        return img_dict
    return img_list


@memoize
def image_resolution(filepath):
    '''
    Get image size as a (width, height) tuple, without having to open the file.
    '''
    return imagesize.get(filepath)


def cv2_color(color_tuple):
    '''
    Transform color tuple (e.g. `(0,0,0,)`) to open-cv format.
    '''
    return np.array(color_tuple).astype('uint8').tolist()


def mean_std(values):
    '''
    Compute mean and std of a list of values.
    '''
    v = np.array(values)
    return v.mean(), v.std()


def extract_scanpaths(csvfile, delim=','):
    '''
    Parse dataset CSV with fixation data.
    '''
    res = []
    cur = None

    with open(csvfile) as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            # We have a unique combination of image and username per experiment trial.
            uniq = row['image'] + row['username']

            if uniq != cur:
                if 'obj' in locals() and cur is not None:
                    # Store entry.
                    res.append(obj)

                # Init new entry.
                obj = {'image': row['image'], 'width': int(row['width']), 'height': int(row['height']), 'scanpath':[]}

            x, y = float(row['x']), float(row['y'])
            obj['scanpath'].append([x,y])

            # Flag already seen image.
            cur = uniq

    # Flush last entry.
    if len(obj['scanpath']) > 0:
        res.append(obj)

    return res


def lin_map(values, start, end):
    '''
    Create a linear mapping of a list of values to a [start,end] range.
    '''
    def rescale(x):
        # Rescale values in the [0,1] range.
        x_min, x_max = min(x), max(x)
        return [(i - x_min) / (x_max - x_min) for i in x]

    return [i * (end - start) + start for i in rescale(values)]
