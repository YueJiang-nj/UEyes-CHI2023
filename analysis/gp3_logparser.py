#!/usr/bin/env python3
# coding: utf-8

'''
Parse GP3 HD eye-tracking logs and create a consolidated dataset in CSV format with the following columns:
`image, width, height, username, x, y, timestamp, duration`

Usage: gp3_logparser.py --log_dir /path/to/logs --img_dir /path/to/images

It is possible to filter out fixations that happened after or before N seconds,
using the `--t_min` and `--t_max` options.

Note: This program will print a (large) CSV string to stdout, so you should redirect the output to a file.

External dependencies, to be installed e.g. via pip:
- none

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import argparse
import warnings
import os
import csv
import io
from collections import defaultdict
from datetime import datetime
from time import mktime
from utils import load_files, image_resolution


def ts_col_parse(s):
    '''
    Convert "TIME(2022/04/13 16:49:44.313)" (str) to 1649861384.0658 (float).
    '''
    t = s[s.find('(')+1:s.find(')')]
    d = datetime.strptime(t, '%Y/%m/%d %H:%M:%S.%f')
    return mktime(d.timetuple())


def get_timestamp(row):
    '''
    Read current timestamp from CSV row.
    GP3 logs have a particular format: the actual date is shown in a "TIME(d)" column name,
    where d is something like "2022/04/13 16:49:44.313". Therefore, we must parse that date
    and then add the elapsed time according to that date (i.e., the row value).
    '''
    for col in row.keys():
        if col.startswith('TIME('):
            return ts_col_parse(col) + float(row[col])
    return 0


def get_fixations(log_file, img_dict, delim=',', rel_coords=False, **kwargs):
    res = defaultdict(list)

    with open(log_file) as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            # FPOGX and FPOGY are the fixation coordinates. FPOGD is the fixation duration.
            x, y = float(row['FPOGX']), float(row['FPOGY'])
            duration = row['FPOGD']

            # NB: The eye-tracker's user manual says this:
            # """
            # The X and Y coordinates are reported as a fraction of the screen size.
            # (0,0) is top left, (0.5,0.5) is the screen center, and (1.0,1.0) is bottom right.
            # """
            # It seems that the refer to image size, actually, because some fixations have negative values.
            if x < 0 or y < 0:
                continue

            image = row['MEDIA_NAME']

            if not rel_coords:
                # Compute absolute coordinates, relative to image size.
                w, h = image_resolution(img_dict[image])
                x *= w
                y *= h

            # Cast values to get the final CSV right.
            t = get_timestamp(row) # already float
            d = float(duration)    # in seconds

            res[image].append((x, y, t, d))

    return res


def file_generator(log_file, img_dict, delim=',', cat_dict=None, category=None,
                   t_min=None, t_max=None, f_min=None, f_max=None, **kwargs):
    basename = os.path.basename(log_file)
    username, _ = os.path.splitext(basename)

    fix_dict = get_fixations(log_file, img_dict, **kwargs)
    for image, rows in fix_dict.items():
        # Exclude non-stimulus images.
        if image not in img_dict:
            warnings.warn(f'Image {image} not found!', stacklevel=2)
            continue

        # Exclude images in different categories.
        if category and cat_dict[image] != category:
            continue

        # Read stimulus image size.
        w, h = image_resolution(img_dict[image])

        # Allow to filter out fixations, if desired.
        # This way we can analyze fixations that happened e.g.
        # - "during the first 2 seconds of free viewing" -> --t_max=2
        # - "after 1 second of free viewing" -> --t_min=1
        # etc.
        x0, y0, t0, d0 = rows[0]

        for n, (x, y, t, d) in enumerate(rows):
            # Get number of seconds since the beginning of the scanpath.
            elapsed = t - t0
            if t_min is not None and elapsed < t_min:
                continue
            if t_max is not None and elapsed > t_max:
                continue

            # Remember that `n` is zero-indexed.
            num_fixations = n + 1
            if f_min is not None and num_fixations < f_min:
                continue
            if f_max is not None and num_fixations > f_max:
                continue

            yield (image, w, h, username, x, y, t, d)


def dir_generator(dirname, img_dict, delim=',', cat_dict=None, category=None,
                  t_min=None, t_max=None, f_min=None, f_max=None, **kwargs):
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if not f.endswith('.csv'):
                continue

            log_file = os.path.join(root, f)
            yield from file_generator(log_file, img_dict, delim=delim, cat_dict=cat_dict, category=category, \
                          t_min=t_min, t_max=t_max, f_min=f_min, f_max=f_max, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse GP3 HD eye-tracking logs',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', required=True, help='path to eye-tracking logs')
    parser.add_argument('--img_dir', required=True, help='path to image stimuli')
    parser.add_argument('--category', choices=['desktop','mobile','poster','web'], help='image category to only consider')
    parser.add_argument('--delim', default=',', help='CSV log file delimiter')
    parser.add_argument('--t_min', type=int, help='minimum free-viewing time, in seconds')
    parser.add_argument('--t_max', type=int, help='maximum free-viewing time, in seconds')
    parser.add_argument('--f_min', type=int, help='minimum number of fixations')
    parser.add_argument('--f_max', type=int, help='maximum number of fixations')

    args = parser.parse_args()

    # Map images (media IDs) to absolute paths.
    img_dict = load_files(args.img_dir)

    # Also map images to categories.
    categories = {}
    with open('categories.csv') as f:
        reader = csv.DictReader(f, delimiter=args.delim)
        for row in reader:
            k, v = row['image'], row['category']
            categories[k] = v

    # Keep CSV data in memory, so that we can redirect the output of this program to stdout.
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)

    header = ['image', 'width', 'height', 'username', 'x', 'y', 'timestamp', 'duration']
    writer.writerow(header)

    log_data = dir_generator(args.log_dir, img_dict, cat_dict=categories, **vars(args))
    for values in log_data:
        writer.writerow(values)

    print(output.getvalue())
