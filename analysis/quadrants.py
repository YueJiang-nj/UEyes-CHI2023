#!/usr/bin/env python3
# coding: utf-8

'''
Compute how many fixations happened in each of the four screen quadrants.

Usage: quadrants.py --log_dir /path/to/logs --img_dir /path/to/images

It is possible to filter out fixations that happened after or before N seconds,
using the `--t_min` and `--t_max` options.

External dependencies, to be installed e.g. via pip:
- none

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import argparse
import io
import csv
from gp3_logparser import dir_generator
from utils import load_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute fixations in screen quadrants',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', required=True, help='path to eye-tracking logs')
    parser.add_argument('--img_dir', required=True, help='path to image stimuli')
    parser.add_argument('--delim', default=',', help='CSV log file delimiter')
    parser.add_argument('--t_min', type=int, help='minimum free-viewing time, in seconds')
    parser.add_argument('--t_max', type=int, help='maximum free-viewing time, in seconds')
    parser.add_argument('--f_min', type=int, help='minimum number of fixations')
    parser.add_argument('--f_max', type=int, help='maximum number of fixations')

    args = parser.parse_args()

    # Map images (media IDs) to absolute paths.
    img_dict = load_files(args.img_dir)

    # Keep CSV data in memory, so that we can redirect the output of this program to stdout.
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)

    header = ['image', 'username', 'quadrant']
    writer.writerow(header)

    log_data = dir_generator(args.log_dir, img_dict, rel_coords=True, **vars(args))
    for (image, w, h, username, x, y, t, d) in log_data:
        # Add an epsilon to avoid using `<=` or `>=` comparisons later.
        th = 0.5 + 1e-6

        if x > th and y < th:
            q = 'q1'
        elif x < th and y < th:
            q = 'q2'
        elif x < th and y > th:
            q = 'q3'
        elif x > th and y > th:
            q = 'q4'

        values = (image, username, q)
        writer.writerow(values)

    print(output.getvalue())
