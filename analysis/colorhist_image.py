#!/usr/bin/env python3
# coding: utf-8

'''
Get color histogram of ALL pixels in a given list of images.

Usage: colorhist_image.py --img_dir /path/to/imgs

NB: This program will print a (large) JSON string to stdout, so you should redirect the output to a file.

External dependencies, to be installed e.g. via pip:
- Pillow

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import sys
import json
import argparse
from collections import defaultdict
from utils import load_screenshot, load_files
from PIL import Image


parser = argparse.ArgumentParser(description='compute histogram of image colors',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_dir', help='directory with the experiment images')
parser.add_argument('--img_files',  nargs='+', help='directory with the experiment images', metavar='filename')
parser.add_argument('--resize_width', type=int, default=400, help='resizing image width')
parser.add_argument('--resize_height', type=int, default=400, help='resizing image height')

args = parser.parse_args()

if not args.img_dir and not args.img_files:
    parser.print_help()
    exit(1)

colorhist = defaultdict(int)
img_files = load_files(args.img_dir, mode='list') if args.img_dir else args.img_files

for f in img_files:
    # Downsample images to speed up computation time.
    sz = (args.resize_width, args.resize_height)
    im = load_screenshot(f).resize(sz)
    # The source screenshots are not padded, so we just count pixel colors.
    for rgb in im.getdata():
        colorhist[rgb] += 1

print(json.dumps([{'color':rgb, 'count':count} for rgb, count in colorhist.items()]))
