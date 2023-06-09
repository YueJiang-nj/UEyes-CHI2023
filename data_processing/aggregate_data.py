import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import pickle 
from generate_heatmaps import *


# output binary fixmap and heatmap
def create_agg_fixmap_and_heatmap(w, h, coords, img_name, img,
                    outpath, num_khs, WEIGHTED, sigma=25, duration=7): 


    # scalar for scaling to the screen size
    scalar = min(1920/w, 1200/h)

    # the radius should be one visual angle divided by the resize ratio.
    sigma /= scalar

    # output directories
    fixmaps_dir = outpath + "fixmaps_" + str(duration) + 's'
    heatmaps_dir = outpath + "heatmaps_" + str(duration) + 's'

    # create output directories
    os.makedirs(fixmaps_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    # initialize binary fixation maps and heatmaps
    xs = tuple([elt[0] for elt in coords])
    ys = tuple([elt[1] for elt in coords])
    times = tuple([elt[2] for elt in coords])
    time_durations = tuple([elt[3] for elt in coords])
    bitmap = np.zeros((w, h))
    fixations = np.zeros((w, h))

    # compute binary heatmaps
    for c in coords: 
        x, y, time_duration = int(c[0]), int(c[1]), float(c[3])
        if x < w and y < h and x >= 0 and y >= 0:
            if WEIGHTED:
                fixations[x,y] += 1 * time_duration 
            else:
                fixations[x,y] += 1
            bitmap[x,y] = 1
    heatmap = ndimage.gaussian_filter(fixations, sigma=[sigma, sigma])
    if np.max(heatmap) != 0:
        heatmap = 255*heatmap/float(np.max(heatmap))
    
    # compute binary fixation maps
    bitmap = ndimage.gaussian_filter(bitmap, [min(w, h) / 400, min(w, h) / 400])
    bitmap = (bitmap > 0.001).astype(int)
    if np.max(bitmap) != 0:
        bitmap = 255 * np.ceil(bitmap/float(np.max(bitmap)))

    # save binary fixation maps and heatmaps
    fixmap_img = Image.fromarray(np.uint8(np.transpose(bitmap)), "L") 
    heatmap_img = Image.fromarray(np.uint8(np.transpose(heatmap)), "L") 
    fixmap_img.save(os.path.join(fixmaps_dir, img_name))
    heatmap_img.save(os.path.join(heatmaps_dir, img_name))


def create_agg_eye_path(user_ids, w, h, coords_list, img_name, 
                    outpath, img, sigma=25, duration=7): 

    # output directories
    paths_dir = outpath + "paths_" + str(duration) + 's'

    # create output directories
    os.makedirs(paths_dir, exist_ok=True)
    individual_dir = paths_dir + '/{}/'.format(
                '.'.join(img_name.split('/')[-1].split('.')[:-1]))
    os.makedirs(individual_dir, exist_ok=True)

    # render scanpath
    colors = ['olive', 'blue', 'brown', 'm',
                'darkorange', 'black', 'red', 'green', 'lime', 'pink', 'yellow', 'cyan', 
                'purple', 'palegreen', 'deepskyblue', 'gold', 'white', 'salmon',
                 'slategray', 'beige', 'silver', 'navy',
                'violet', 'lime', 'lightgreen', 'y']

    if len(img.shape) < 3:
        return
    if img.shape[2] == 3: 
        img = np.insert(
            img,
            3, #position in the pixel value [ r, g, b, a <-index [3]  ]
            255/2, # or 1 if you're going for a float data type as you want the alpha to be fully white otherwise the entire image will be transparent.
            axis=2, #this is the depth where you are inserting this alpha channel into
        )
    else:
        img[:,:,3] = 255/2

    for index in range(len(coords_list)):
        coords = coords_list[index]
        xs = tuple([elt[0] for elt in coords])
        ys = tuple([elt[1] for elt in coords])
        ts = tuple([elt[2] for elt in coords])
        plt.gray()
        plt.axis('off')

        img = Image.fromarray(img)
        img.putalpha(int(255/2))
        img = np.array(img)

        
        ax = plt.imshow(img)
        for i in range(len(xs)):
            if i > 0:
                ax.axes.arrow(
                    xs[i - 1],
                    ys[i - 1],
                    xs[i] - xs[i - 1],
                    ys[i] - ys[i - 1],
                    width=min(w, h) / 300,
                    color='blue',
                    alpha=0.5,
                )
        for i in range(len(xs)):
            cir_rad = min(w, h) / 40
            if i == 0:
                circle = plt.Circle(
                (xs[i], ys[i]),
                radius=min(w, h) / 35,
                edgecolor="blue",
                facecolor="lime",
                )
            elif i == len(xs) - 1:
                circle = plt.Circle(
                (xs[i], ys[i]),
                radius=min(w, h) / 35,
                edgecolor="blue",
                facecolor="red",
                )
            else:
                circle = plt.Circle(
                    (xs[i], ys[i]),
                    radius=cir_rad,
                    edgecolor="blue",
                    facecolor="blue",
                    alpha=0.5,
                )
            ax.axes.add_patch(circle)
   
        # render scanpath on the image
        ax.figure.savefig(os.path.join(individual_dir, '{}.png'.format(user_ids[index])), dpi=120, bbox_inches="tight")
        plt.close(ax.figure)



def main(args):

    WEIGHTED = args["weight"] # 0 or 1

    # define directories
    base_dir = './output/'
    if WEIGHTED:
        output_dir = './weighted_results/'
    else: 
        output_dir = './unweighted_results/'
    print(output_dir)

    # aggregate heatmaps and scanpaths
    data_files_save = []
    for block_idx in range(53,56):
        if block_idx == 51:
            continue

        print('Block:', block_idx)
        exist_kh_list = []
        num_first_dir_files = 0
        block_folder = 'block ' + str(block_idx) + '/'
        for kh_idx in range(66):
            data_files = sorted(glob.glob(base_dir + "kh0" + f'{kh_idx:02}' + "/block " \
                                            + str(block_idx) + "/data/*.pkl"))
            if data_files != []:
                exist_kh_list.append(kh_idx)
                if len(exist_kh_list) == 1:
                    num_first_dir_files = len(data_files)

                # some users did not see all the images in the block
                # we need to find largest block result
                else:
                    if len(data_files) > num_first_dir_files:
                        tmp_first = exist_kh_list[0]
                        tmp_last = exist_kh_list[-1]
                        exist_kh_list[0] = tmp_last
                        exist_kh_list[-1] = tmp_first

        data_files = sorted(glob.glob(base_dir + "kh0" + f'{exist_kh_list[0]:02}' + "/block " \
                                        + str(block_idx) + "/data/*.pkl"))

        # loop over all the results
        for data_f1 in data_files:
            all_image_coords_list = []
            data_image_coords_list = [] 
            all_image_coords_list_3s = []
            data_image_coords_list_3s = [] 
            all_image_coords_list_1s = []
            data_image_coords_list_1s = [] 
            f1 = open(data_f1, 'rb')
            data1 = pickle.load(f1)
            all_image_coords_list += data1.image_coords
            data_image_coords_list.append(data1.image_coords)
            all_image_coords_list_3s += data1.image_coords_3s
            data_image_coords_list_3s.append(data1.image_coords_3s)
            all_image_coords_list_1s += data1.image_coords_1s
            data_image_coords_list_1s.append(data1.image_coords_1s)
            for idx in exist_kh_list[1:]:
                data_f2 = data_f1[:12] + f'{idx:02}' + data_f1[14:]
                try:
                    f2 = open(data_f2, 'rb')
                    data2 = pickle.load(f2)
                    all_image_coords_list += data2.image_coords
                    data_image_coords_list.append(data2.image_coords)
                    all_image_coords_list_3s += data2.image_coords_3s
                    data_image_coords_list_3s.append(data2.image_coords_3s)
                    all_image_coords_list_1s += data2.image_coords_1s
                    data_image_coords_list_1s.append(data2.image_coords_1s)
                except:
                    pass

            # aggregate 7s results
            create_agg_fixmap_and_heatmap(data1.image_width, data1.image_height, 
                      all_image_coords_list, data1.img_name, data1.img, 
                      base_dir + output_dir + block_folder, 
                      len(data_image_coords_list), WEIGHTED, sigma=40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate 3s and 7s heatmaps and scanpaths from participant data."
    )
    parser.add_argument(
        "-w",
        "--weight",
        help="whether the result is based on time duration, 0: unweighted, 1:weighted.",
        type=int,
        default=1,
    )
    args = vars(parser.parse_args())
    main(args)