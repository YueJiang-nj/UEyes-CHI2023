#!/usr/bin/env python3
# coding: utf-8
# pylint: disable=E1101,C0103

'''
A set of metrics for evaluatation of eye-gaze scanpaths,
adapted from https://github.com/rAm1n/saliency

Every function expects at least two arguments:
- Predicted heatmap image, as a Numpy array.
- Reference heatmap image, as a Numpy array.

External dependencies, to be installed e.g. via pip:
- cv2
- numpy

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

from scipy.stats import entropy
import numpy as np


EPS = np.finfo(np.float32).eps

def rescale(any_map):
    '''
    Apply mimaxscaler algorithm.
    See https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    '''
    return (any_map - any_map.min()) / (EPS + any_map.max() - any_map.min())


def normalize(any_map):
    '''
    Normalize heatmap values in the [0,1] range.
    '''
    any_map = rescale(any_map)
    any_map /= (EPS + any_map.sum())
    return any_map


def whiten(any_map):
    '''
    Apply whitening algorithm.
    See https://en.wikipedia.org/wiki/Whitening_transformation
    '''
    return (any_map - any_map.mean()) / (EPS + any_map.std())


def auc(sal_map, ref_map, fixation_threshold=0.7):
    '''
    Compute Judd's AUC score.
    '''
    def area_under_curve(predicted, actual, labelset):
        def roc_curve(predicted, actual, cls):
            si = np.argsort(-predicted)
            tp = np.cumsum(np.single(actual[si]==cls))
            fp = np.cumsum(np.single(actual[si]!=cls))
            tp = tp/np.sum(actual==cls)
            fp = fp/np.sum(actual!=cls)
            # print(predicted, actual, cls)
            tp = np.hstack((0.0, tp, 1.0))
            fp = np.hstack((0.0, fp, 1.0))

            return tp, fp

        # integration
        def auc_from_roc(tp, fp):
            # print(fp)
            h = np.diff(fp)
            # print(h,tp[1:]+tp[:-1])
            return np.sum(h*(tp[1:]+tp[:-1]))/2.0

        tp, fp = roc_curve(predicted, actual, np.max(labelset))
        return auc_from_roc(tp, fp)

    ref_map = (ref_map > fixation_threshold).astype(int)
    salShape = sal_map.shape
    fixShape = ref_map.shape
    # print(np.max(sal_map))
    
    predicted = sal_map.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
    actual = ref_map.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
    labelset = np.arange(2)

    return area_under_curve(predicted, actual, labelset)



def auc_shuff(sal_map, ref_map, rand_map=None, step_size=.01):
    '''
    Compute shuffled AUC score.
    '''
    sal_map -= np.min(sal_map)
    ref_map = np.vstack(np.where(ref_map != 0)).T

    if np.max(sal_map) > 0:
        sal_map = sal_map / np.max(sal_map)
    Sth = np.asarray([ sal_map[y-1][x-1] for y,x in ref_map ])
    Nfixations = len(ref_map)

    if rand_map is None:
        height, width = sal_map.shape
        rand_map = np.zeros((height, width))

    others = np.copy(rand_map)
    for y,x in ref_map:
        others[y-1][x-1] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = rand_map[ind]
    randfix = sal_map[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1] = 1.0
    fp[-1] = 1.0
    tp[1:-1] = [float(np.sum(Sth >= thresh)) / Nfixations for thresh in allthreshes]
    fp[1:-1] = [float(np.sum(nFix[randfix >= thresh])) / Nothers for thresh in allthreshes]

    return np.trapz(tp,fp)


def nss(sal_map, ref_map):
    '''
    Compute Normalized Scanpath Saliency score.
    '''
    # print(sal_map)
    sal_map = whiten(sal_map)
    # print(sal_map)

    mask = ref_map.astype(np.bool)
    # print(mask)
    return sal_map[mask].mean()


def infogain(sal_map, ref_map, rand_map=None):
    '''
    Compute InfoGain score.
    '''
    sal_map = normalize(sal_map)

    if rand_map is None:
        height, width = sal_map.shape
        rand_map = np.zeros((height, width))

    rand_map = normalize(rand_map)

    fixs = ref_map.astype(np.bool)

    return (np.log2(EPS + sal_map[fixs]) \
          - np.log2(EPS + rand_map[fixs])).mean()


def similarity(sal_map, ref_map):
    '''
    Compute Similarity score.
    '''
    sal_map = normalize(sal_map)
    ref_map = normalize(ref_map)

    return np.minimum(sal_map, ref_map).sum()


def cc(sal_map, ref_map):
    '''
    Compute Coefficient of Correlation score.
    '''
    sal_map = whiten(sal_map)
    ref_map = whiten(ref_map)
    score = np.corrcoef(sal_map.flatten(), ref_map.flatten())
    return score[0][1]


def kldiv(sal_map, ref_map):
    '''
    Compute Kullback-Leibler divergence.
    '''
    return entropy(sal_map.flatten() + EPS, ref_map.flatten() + EPS)
