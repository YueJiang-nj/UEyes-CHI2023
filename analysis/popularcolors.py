#!/usr/bin/env python3
# coding: utf-8

'''
Plot color palette from image file.

Usage: popularcolors.py --img image.png

External dependencies, to be installed e.g. via pip:
- colorthief
- numpy
- cv2
- matplotlib

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorthief import ColorThief
from utils import cv2_color

parser = argparse.ArgumentParser(description='extract popular colors from image',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img', required=True, help='input image file')
parser.add_argument('--thumb_size', type=int, default=50, help='size of the thumbnail squares')
parser.add_argument('--num_colors', type=int, default=5, help='number of colors to extract')
parser.add_argument('--quality', type=int, default=10, help='accuracy of color extraction')
parser.add_argument('--outline', action='store_true', help='draw outline around thumbnail squares')
parser.add_argument('--save', action='store_true', help='save result to {IMG}_palette.png')

args = parser.parse_args()


thief = ColorThief(args.img)

dominant = thief.get_color(quality=args.quality)
palette = thief.get_palette(color_count=args.num_colors, quality=args.quality)

print('dominant:', dominant)
print('palette:', palette)


def plot_color(source, color, x=0, y=0, width=50, height=50, linewidth=0):
    # Plot filled square (cv2 does it by setting `thickness = -1`.
    cv2.rectangle(source, (x,y), (x+width,y+height), cv2_color(color), -1)
    # Then draw line around square, if set.
    if linewidth > 0:
        cv2.rectangle(source, (x, y), (x + width, y + height - linewidth), cv2_color((0,0,0)), linewidth)


# Allocate final image size.
img_h = args.thumb_size
img_w = args.thumb_size * (args.num_colors + 2)
image = np.zeros((img_h, img_w, 3), dtype='uint8')

# Fill image with white background color.
plot_color(image, [255,255,255], width=img_w, height=img_h)

# Plot dominant color.
plot_color(image, dominant, width=args.thumb_size, height=args.thumb_size, linewidth=int(args.outline))

# Plot the rest of the palette separated from the dominant color.
for n, color in enumerate(palette):
    # Exclude the dominant color, since it's already plotted.
    if n == 0:
        continue

    x_pos = args.thumb_size * (n+1)
    plot_color(image, color, x=x_pos, width=args.thumb_size, height=args.thumb_size, linewidth=int(args.outline))


plt.figure()
plt.axis('off')
plt.imshow(image)
if args.save:
    plt.savefig(args.img + '_palette.png')
plt.show()
