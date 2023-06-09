#!/usr/bin/env python3
# coding: utf-8

'''
Get color histograms at fixations from a CSV eye-tracking dataset.

Usage: colorhist_fixations.py --csv_file dataset.csv --img_dir /path/to/images [--delim " "]

Note: This program will print a (large) JSON string to stdout, so you should redirect the output to a file.

External dependencies, to be installed e.g. via pip:
- Pillow

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import os
import csv
import json
import argparse
from utils import load_screenshot, load_files, extract_scanpaths, lin_map
from collections import defaultdict


parser = argparse.ArgumentParser(description='compute histogram of fixated colors from eye-tracking dataset',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--csv_file', required=True, help='input CSV file')
parser.add_argument('--delim', default=',', help='CSV column delimiter')
parser.add_argument('--img_dir', required=True, help='directory with the experiment images')
parser.add_argument('--resize_width', type=int, default=400, help='resizing image width')
parser.add_argument('--resize_height', type=int, default=400, help='resizing image height')

args = parser.parse_args()

colorhist = defaultdict(int)

# Map images (media IDs) to absolute paths.
img_dict = load_files(args.img_dir)
scanlist = extract_scanpaths(args.csv_file, delim=args.delim)

# Iterate over the fixations dataset (large file!) and analyze image colors.
for row in scanlist:
    # CVS rows are: `image, width, height, username, x, y, timestamp, duration`.
    filename = row['image']
    if filename not in img_dict:
        continue

    filepath = img_dict[filename]
    # Downsample images to speed up computation time.
    sz = (args.resize_width, args.resize_height)
    im = load_screenshot(filepath).resize(sz)

    # TODO: weigh fixation importance by duration. Currently not done in `extract_scanpaths()`.
    x_values = [x for x,y in row['scanpath']]
    y_values = [y for x,y in row['scanpath']]

    # Instead of linear mapping, make fixations relative to the new viewport (elastic mapping).
    x_factor = args.resize_width / row['width']
    y_factor = args.resize_height / row['height']
    # Note that we subtract 1 px to the provided sizes, to avoid an IndexError.
    x_values = [i * x_factor for i in x_values]
    y_values = [i * y_factor for i in y_values]

    for x,y in zip(x_values, y_values):
        coord = (int(x), int(y))
        # Still, some fixations may be reported as outside of the image, so ignore them.
        try:
            r, g, b = im.getpixel(coord)
        except:
            pass
        colorhist[(r, g, b)] += 1

print(json.dumps([{'color':rgb, 'count':count} for rgb, count in colorhist.items()]))
