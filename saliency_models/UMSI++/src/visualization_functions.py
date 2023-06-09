from enum import Enum

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from PIL import Image
from scipy.ndimage.filters import gaussian_filter


class VisType(Enum):
    HEATMAP = 1
    HEATMAP_OVERLAY = 2
    SPOTLIGHT = 3
    SPOTLIGHT_LEVEL_SETS = 4


def spotlight(im, heatmap, toplot=False):
    im = Image.fromarray(im)
    h = Image.fromarray(heatmap)
    h = h.resize(im.size)
    h = np.double(h) / 255
    a = np.zeros(h.shape)
    a = np.zeros(h.shape)
    h_flat = h

    for i in [80, 85, 90, 95, 100]:
        h50 = np.percentile(h_flat, i)
        h50mat = np.copy(h)
        h50mat[h50mat < h50] = 0
        h50mat[h50mat > 0] = 1
        a += h50mat
    a = a / 4
    a = gaussian_filter(a, sigma=30)
    a[a < 0.2] = 0.2
    singleim = np.single(im) / 255
    b = np.array([a, a, a])
    b = b.transpose((1, 2, 0))
    r_channel = singleim[:, :, 0]
    g_channel = singleim[:, :, 1]
    b_channel = singleim[:, :, 2]
    r_channel = np.multiply(a, r_channel)
    g_channel = np.multiply(a, g_channel)
    b_channel = np.multiply(a, b_channel)
    rgb = np.dstack((r_channel, g_channel, b_channel))
    rgb[rgb > 1] = 1
    rgb_uint8 = (rgb * 255.999).astype(np.uint8)
    z = Image.fromarray(rgb_uint8)

    if toplot:
        plt.figure(figsize=(20, 20))
        plt.imshow(z, vmin=0, vmax=1)
        plt.axis('off')
        plt.show()

    return z


def heatmap_overlay(im, heatmap, colmap='hot'):
    cm_array = cm.get_cmap(colmap)
    im_array = np.asarray(im)
    heatmap_norm = (
        heatmap - np.min(heatmap)) / float(np.max(heatmap) - np.min(heatmap))
    heatmap_hot = cm_array(heatmap_norm)
    res_final = im_array.copy()
    heatmap_rep = np.repeat(heatmap_norm[:, :, np.newaxis], 3, axis=2)
    res_final[...] = heatmap_hot[..., 0:3] * 255.0 * heatmap_rep + im_array[
        ...] * (1 - heatmap_rep)

    return res_final


def heatmap_patches(im, heatmap, alpha=.6, colmap='hot'):
    cm_array = cm.get_cmap(colmap)
    im_array = np.asarray(im)
    heatmap_norm = (
        heatmap - np.min(heatmap)) / float(np.max(heatmap) - np.min(heatmap))
    inds = heatmap_norm > (np.mean(heatmap_norm) + np.std(heatmap_norm))
    heatmap_hot = cm_array(heatmap_norm)
    res_final = im_array.copy()
    res_final[inds, ...] = heatmap_hot[inds, 0:3] * 255.0 * alpha + im_array[
        inds, ...] * (1 - alpha)

    return res_final


def spotlight_custom(im,
                     heatmap,
                     levels=3,
                     most_salient_nlevel=3,
                     mask_darkness=0.8,
                     smoothness=3,
                     brightness=0.8,
                     percentile_based=True,
                     toplot=True):
    """
    spotlight visualization adapted from https://github.com/cvzoya/fixation-visualization/blob/master/plotSpotlight.m
    highlights parts of the image given the saliency heatmap, darkening the least salient regions
    enhanced with more options to define the discretization of the visualization

    for percentile_based = False, recommended to use levels=5, most_salient_nlevel=3, smoothness=0

    Parameters
    ----------
    im : array_like or PIL image object
        original image
    heatmap : array_like or PIL image object
        saliency heatmap
    levels: integer greater than 0, optional
        overall discretization for the visualization, default 3
    most_salient_nlevel: integer greater than 0, optional
        specifies the granularity of the most salient level
        adds more sub levels to the most salient level
        for more fine-grained visualization, default 3
        total number of levels = levels + most_salient_nlevel - 1
    mask_darkness: float between 0.0 and 1.0, optional
        specifies how dark the darkned area should be, default 0.8
        a value of 1.0 completely masks the irrelevant regions
    smoothness: float, optional
        specifies the sigma in gaussian blurring
        when making the discretization between levels smoother, default 3
        when set to 0, no smoothing is applied
    brightness: float, optional
        specifies the brightness of the image, default 0.8
        useful when the image is too bright
        and is difficult to visualize the levels
    percentile_based: bool, optional
        specifies whether the values for each level
        should be split based on percentile, default True
        if set to false, values are split uniformly out of 255
    toplot: bool, optional
        plot the result
    """

    # convert map from 0 to 255
    heatmap = 255.0 * (heatmap - np.min(heatmap)) / (np.max(heatmap) -
                                                     np.min(heatmap))
    heatmap = heatmap.astype('uint8')

    total_levels = levels + most_salient_nlevel - 1
    # if not equal
    h = heatmap

    if (heatmap.size != im.size):
        h = Image.fromarray(heatmap) if isinstance(heatmap,
                                                   np.ndarray) else heatmap
        im = Image.fromarray(im) if isinstance(im, np.ndarray) else im
        h = h.resize(im.size)
    h = np.asarray(h)

    if (percentile_based):
        # split levels by percentile
        # start percentile from non-zero values
        start_percentile = int(stats.percentileofscore(h.flatten(), 1))
        perc_list = np.array(
            [np.percentile(h, i) for i in range(start_percentile, 101)])
        val_list = perc_list
    else:
        # uniformly split 255
        div, rem = int(255 / total_levels), 255 % total_levels
        val_list = [
            val + i if i < rem else val + rem
            for i, val in enumerate(range(0, 256, div))
        ]
    nvals = len(val_list)
    # readjust total levels based on unique values
    total_levels = min(total_levels, nvals, len(np.unique(val_list)))
    levels = total_levels - most_salient_nlevel + 1

    if (levels < 0):
        levels = total_levels
        most_salient_nlevel = 1

    # bigger group size corresponding to levels
    group_size = int(len(val_list) / levels)
    group_val = int(255 / total_levels)
    new_h = h.copy().astype(int)
    # group values
    # up to second most salient

    for i in range(levels - 1):
        smaller_idx, larger_idx = i * group_size, (i + 1) * group_size
        less_val, greater_val = val_list[smaller_idx], val_list[larger_idx]
        new_h[(less_val < h) & (h <= greater_val)] = group_val * (i + 1)
    # most salient group_size
    # corresponding to most_salient_nlevel
    first_group_size = int(group_size / most_salient_nlevel)
    i = levels - 1
    # fine grained level for the most salient

    for j in range(most_salient_nlevel):
        smaller_idx, larger_idx = i * group_size + j * first_group_size, i * group_size + (
            j + 1) * first_group_size
        larger_idx = larger_idx if larger_idx < nvals else nvals - 1
        less_val, greater_val = val_list[smaller_idx], val_list[larger_idx]
        new_h[(less_val < h) & (h <= greater_val)] = group_val * (i + j + 1)
    # remaining
    new_h[greater_val < h] = 255
    # make edges look smoother

    if (smoothness > 0):
        new_h = gaussian_filter(new_h, sigma=smoothness)

    # create dark mask based on h grouped values
    dark_mask = np.zeros_like(im)
    inverted_hm = abs(new_h - 256)
    new_dark = np.dstack((dark_mask, inverted_hm))
    darkening = np.where(new_dark[:, :, 3] > 0, new_dark[:, :, 3], 255) / 255
    # invert dark mask for transparency
    darkening = abs(1 - darkening * mask_darkness)
    new_im = (im * np.repeat(darkening[..., None], 3, axis=-1) *
              brightness).astype('uint8')

    if toplot:
        plt.figure(figsize=(20, 20))
        plt.imshow(new_im, vmin=0, vmax=1)
        plt.axis('off')
        plt.show()

    return new_im


def plot_attention(filename, im, impim, plottype=2, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    if plottype == VisType.HEATMAP:
        ax.imshow(impim)
    elif plottype == VisType.HEATMAP_OVERLAY:
        ax.imshow(heatmap_overlay(im, impim))
    elif plottype == VisType.SPOTLIGHT:
        spotlight_res = spotlight(im, impim, toplot=False)
        ax.imshow(spotlight_res)
    elif plottype == VisType.SPOTLIGHT_LEVEL_SETS:
        spotlight_res = spotlight_custom(im,
                                         impim,
                                         toplot=False,
                                         percentile_based=False,
                                         levels=5,
                                         most_salient_nlevel=5,
                                         smoothness=0)
        ax.imshow(spotlight_res)

    ax.set_axis_off()

    if not title:
        title = filename
    ax.set_title(title)
