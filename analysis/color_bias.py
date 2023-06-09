#!/usr/bin/env python3
# coding: utf-8

'''
Plot color histograms from saved JSON file [{color, count}] dataset. See colorhist_*.py files.

Will save the resulting plot as a PNG file with the same name as the --json_file option.
If the --frequency option is provided, will print to stdout the frequency of the top-K colors in CSV format.

Usage: colorbar.py --json_file colorhist.json [--frequency --num_colors 16 --width 200 --height 10]

External dependencies, to be installed e.g. via pip:
- numpy
- cv2
- matplotlib
- sklearn

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import cv2_color


parser = argparse.ArgumentParser(description='generate color bar from JSON file',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--json_file', required=True, help='input file')
parser.add_argument('--frequency', action='store_true', help='print most frequent colors to stdout')
parser.add_argument('--num_colors', type=int, default=16, help='number of colors to summarize')
parser.add_argument('--width', type=int, default=400, help='output bar width')
parser.add_argument('--height', type=int, default=40, help='output bar height')
parser.add_argument('--dpi', type=int, default=96, help='outfile resolution, in dots per inch')
parser.add_argument('--outline', action='store_true', help='draw outline around thumbnail squares')
parser.add_argument('--save', action='store_true', help='save result to {IMG}_color_bias.png')

args = parser.parse_args()


colordic = {}
with open(args.json_file) as f:
    for entry in json.load(f):
        colordic[tuple(entry['color'])] = entry['count']

# TODO: Normalize by color count.
# NB 1: Most colors are singletons.
# NB 2: In mobile UIs, the most common color is white.
colors = list(colordic.keys())


def centroid_histogram(clustering):
    # Based on the number of pixels assigned to each cluster.
    num_labels = np.arange(0, len(np.unique(clustering.labels_)) + 1)
    hist, _ = np.histogram(clustering.labels_, bins=num_labels)
    # Normalize the histogram, such that it sums to one.
    hist = hist.astype('float')
    hist /= hist.sum()

    return hist


def plot_colors(histogram, centroids, width=100, height=10, report=False, outline=False):
    # Allocate some space for drawing text.
    voffset = 50
    halfbin = len(centroids) // 2

    # Discount some space to the final viewport width, to fit exactly all color bars.
    vp_width = width - halfbin
    vp_height = height + voffset
    image = np.zeros((vp_height, vp_width, 3), dtype='uint8')

    # Set background color.
    cv2.rectangle(image, (0,0), (vp_width,vp_height), [255,255,255], -1)

    # Print CSV header.
    if report:
        print('"frequency","r","g","b"')

    x_start = 0
    # Sort quantized colors by more to less frequency.
    color_data = sorted(zip(histogram, centroids), key=lambda x:x[0], reverse=True)
    for i, (frequency, color) in enumerate(color_data):
        # Print CSV rows.
        if report:
            print(frequency, ','.join(map(str, color)), sep=',')

        x_end = x_start + int(frequency * width)
        cv2.rectangle(image, (x_start, voffset), (x_end, voffset + height), cv2_color(color), -1)

        # Draw text every N-th bars, with lines indicatin bar width.
        if i % halfbin == 0:
            perc = frequency * 100
            sep = 8
            mar = 10
            cv2.putText(image, f'{perc:.1f}%', (x_start, voffset - sep - mar), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,0], 1, cv2.LINE_AA)
#            # Line decorators.
#            cv2.line(image, (x_start, voffset - sep), (x_start, voffset - mar), [0,0,0], 1)
#            cv2.line(image, (x_start, voffset - mar), (x_end,   voffset - mar), [0,0,0], 1)
#            cv2.line(image, (x_end,   voffset - mar), (x_end,   voffset - sep), [0,0,0], 1)
            # Better: enlarge current bar.
            cv2.rectangle(image, (x_start, voffset - mar), (x_end - 1, voffset), cv2_color(color), -1)

        x_start = x_end

    if outline:
        thickness = 1
        #cv2.rectangle(image, (0, voffset), (vp_width - thickness, voffset + height - thickness), [0,0,0], thickness)
        # Better: draw only bottom line.
        cv2.line(image, (0, vp_height - thickness), (x_end, vp_height - thickness), [0,0,0], thickness)

    return image


# Perform color quantization.
seed = np.random.seed(123456)
data = KMeans(n_clusters=args.num_colors, random_state=seed)
data.fit(colors)

hist = centroid_histogram(data)
centers = list(data.cluster_centers_)
bar = plot_colors(hist, centers, width=args.width, height=args.height, report=args.frequency, outline=args.outline)

plt.figure(dpi=args.dpi)
plt.axis('off')
plt.imshow(bar)
plt.tight_layout()
if args.save:
    plt.savefig(args.json_file + '_color_bias.png')
plt.show()
