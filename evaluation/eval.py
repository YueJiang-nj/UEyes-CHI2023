#!/usr/bin/env python3
# coding: utf-8

'''
Evaluate eye-gaze heatmaps or scanpaths.

ASSUMPTIONS:
(1) Every predicted heatmap has one and only one grounthruth counterpart
    AND both filenames are the same but stored in different folders.
(2) Every predicted scanpath usually has more one grounthruth counterpart.

Usage: eval.py [--heatmaps --scanpaths] --ref_dir groundtruth-folder --pred_dir predictions-folder

External dependencies, to be installed e.g. via pip:
- numpy
- cv2

example: python evaluation/eval.py --scanpaths --ref_file ./output/weighted_results/ours.csv --pred_file ./paper_figures/scanpath_category/scanpath_Itti-Koch-based_poster.csv 


Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import sys
import os
import csv
import argparse
import warnings
import eval_heatmaps as heat
import eval_scanpaths as scan
from utils import load_files, mean_std, extract_scanpaths
from collections import defaultdict
from operator import itemgetter
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='evaluate saliency heatmaps or scanpaths',
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--heatmaps', action='store_true', help='evaluate heatmaps')
parser.add_argument('--scanpaths', action='store_true', help='evaluate scanpaths')
parser.add_argument('--ref_dir', help='reference (groundtruth) folder')
parser.add_argument('--pred_dir', help='prediction folder')
parser.add_argument('--ref_files', nargs='+', help='reference (groundtruth) files')
parser.add_argument('--pred_files', nargs='+', help='prediction files')
parser.add_argument('--delim', default=',', help='CSV column delimiter, for scanpath files')

args = parser.parse_args()


def exit_if_not(condition, message):
    '''
    Exit program (and print CLI usage) if condition is not met.
    '''
    if not condition:
        print(message)
        parser.print_help()
        sys.exit(1)

# Silly checks.
exit_if_not(args.heatmaps or args.scanpaths, 'Please specify --heatmaps or --scanpaths option\n')
exit_if_not(args.ref_dir or args.ref_files, 'Missing --ref_dir or --ref_files option\n')
exit_if_not(args.pred_dir or args.pred_files, 'Missing --pred_dir or --pred_files option\n')

# Sort all files, to get the paired comparisons right.
ref_files = sorted(load_files(args.ref_dir, mode='list') if args.ref_dir else args.ref_files)
pred_files = sorted(load_files(args.pred_dir, mode='list') if args.pred_dir else args.pred_files)

# Each heatmap file must be provided as an image path.
if args.heatmaps:

    # Ensure we have exactly the same number of heatmaps to compare.
    if len(ref_files) > len(pred_files):
        pred_dir = '/'.join(pred_files[0].split('/')[:-1]) + '/'
        ref_dir = '/'.join(ref_files[0].split('/')[:-1]) + '/'
        for f_path in ref_files:
            f_name = f_path.split('/')[-1]
            if pred_dir + f_name not in pred_files:
                print(ref_dir + f_name)
                ref_files.remove(ref_dir + f_name)

    assert len(ref_files) == len(pred_files), 'The number of reference and prediction files do not match.'


    # Initialize result arrays.
    a_auc, a_nss, a_inf, a_sim, a_cor, a_kld = [], [], [], [], [], []

    for prediction, reference in zip(pred_files, ref_files):
        print(f'Comparing {prediction} vs {reference} heatmaps ...')

        sal_map = cv2.imread(prediction, 0)
        if np.max(sal_map) > 1:
            sal_map = np.divide(sal_map, np.max(sal_map))
        ref_map = cv2.imread(reference, 0)
        if np.max(ref_map) > 1:
            ref_map = np.divide(ref_map, np.max(sal_map))

        # Ensure that both heatmaps were produced over the same source image.
        if ref_map.shape != sal_map.shape:
            warnings.warn('The heatmap shapes do not match!', stacklevel=2)

        auc = heat.auc(sal_map, ref_map)
        nss = heat.nss(sal_map, ref_map)
        inf = heat.infogain(sal_map, ref_map)
        sim = heat.similarity(sal_map, ref_map)
        cor = heat.cc(sal_map, ref_map)
        kld = heat.kldiv(sal_map, ref_map)

        # Finally store the values in the result arrays.
        a_auc.append(auc)
        a_nss.append(nss)
        a_inf.append(inf)
        a_sim.append(sim)
        a_cor.append(cor)
        a_kld.append(kld)

    # Finally report stats.
    print('AUC (judd) :', mean_std(a_auc))
    print('       NSS :', mean_std(a_nss))
    print('  infogain :', mean_std(a_inf))
    print('similarity :', mean_std(a_sim))
    print('        CC :', mean_std(a_cor))
    print('    KL div :', mean_std(a_kld))


# Each scanpath file must be provided in CSV format with the following columns:
# `image, width, height, username, x, y, timestamp, duration`.
if args.scanpaths:

    # Note that there may be multiple *groundtruth* scanpaths for the same source image,
    # but there is only one *predicted* scanpath. So let's create an ad-hoc mapping between reference and predicted scanpaths.
    def extract_scanpaths(csvfile):
        res = []
        cur = None
        obj = None

        with open(csvfile) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                print(row)

                # We have a unique combination of image and username per experiment trial.
                uniq = row['image'] + row['username']

                if uniq != cur:
                    if 'obj' in locals() and cur is not None:
                        # Store entry.
                        res.append(obj)

                    # Init new entry.
                    obj = {'image': row['image'], 'width': int(row['width']), 'height': int(row['height']), 'scanpath':[]}

                # Normalize the scanpath
                # x, y = float(row['x']), float(row['y']) ####### 
                x, y = float(row['x']) / int(row['width']), float(row['y']) / int(row['height']) ####### 
                obj['scanpath'].append([x,y])

                # Flag already seen image.
                cur = uniq

        # Flush last entry.
        if len(obj['scanpath']) > 0:
            res.append(obj)

        return res


    # Initialize result arrays.
    # TODO: Simplify! We don't need to report all the metrics.
    # See Table 1 in https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
    a_dtw, a_tde, a_eye, a_fre, a_euc, a_hau, a_rec, a_det, a_lam, a_cor, a_lev, a_man, a_sma = [], [], [], [], [], [], [], [], [], [], [], [], []

    for reference in ref_files:
        
        print(reference)
        ref_objs = extract_scanpaths(reference)
        print('-===---')

        ref_images = defaultdict(list)
        for r in ref_objs:
            ref_images[r['image']].append(r)

        for prediction in pred_files:
            print('####', prediction)
            pred_objs = extract_scanpaths(prediction)
            print('------')

            print(f'Comparing {prediction} vs {reference} scanpaths ...')

            for pred in pred_objs:
                refs = ref_images[pred['image']]

                for ref in refs:
                    # # Ensure that both scanpaths were produced over the same source image.
                    # # Another way of testing this requirement is to compare the "source" property.
                    # if ref['width'] != pred['width']:
                    #     warnings.warn('The viewport width of groundtruth and predicted scanpaths do not match!', stacklevel=2)
                    # if ref['height'] != pred['height']:
                    #     warnings.warn('The viewport height of groundtruth and predicted scanpaths do not match!', stacklevel=2)

                    r_seq = np.array(ref['scanpath'])
                    p_seq = np.array(pred['scanpath'])
                    if len(r_seq) <= 3:
                        continue 

                    # Compute evaluation metrics that don't depend on the viewport size.
                    dtw = scan.DTW(p_seq, r_seq)
                    tde = scan.TDE(p_seq, r_seq)
                    eye = scan.eyenalysis(p_seq, r_seq)
                    fre = scan.frechet_distance(p_seq, r_seq)
                    euc = scan.euclidean_distance(p_seq, r_seq)
                    hau = scan.hausdorff_distance(p_seq, r_seq)
                    man = scan.mannan_distance(p_seq, r_seq)
                    rec = scan.recurrence(p_seq, r_seq)
                    det = scan.determinism(p_seq, r_seq)
                    lam = scan.laminarity(p_seq, r_seq)
                    cor = scan.CORM(p_seq, r_seq)

                    # Now compute evaluation metrics that depend on the viewport size.
                    width = ref['width']
                    height = ref['height']

                    lev = scan.levenshtein_distance(p_seq, r_seq, width, height)
#                    # FIXME: ScanMatch implementation is buggy.
#                    sma = scan.scan_match(p_seq, r_seq, width, height)

                    # Finally store the values in the result arrays.
                    a_dtw.append(dtw)
                    a_tde.append(tde)
                    a_eye.append(eye)
                    a_fre.append(fre)
                    a_euc.append(euc)
                    a_hau.append(hau)
                    a_rec.append(rec)
                    a_det.append(det)
                    a_lam.append(lam)
                    a_cor.append(cor)
                    a_lev.append(lev)
                    a_man.append(man)
#                    a_sma.append(sma)

    # Finally report stats.
    print('        DTW :', mean_std(a_dtw))
    print('        TDE :', mean_std(a_tde))
    print(' eyenalysis :', mean_std(a_eye))
    print('    frechet :', mean_std(a_fre))
    print('  euclidean :', mean_std(a_euc))
    print('  hausdorff :', mean_std(a_hau))
    print(' recurrence :', mean_std(a_rec))
    print('determinism :', mean_std(a_det))
    print(' laminarity :', mean_std(a_lam))
    print('       CORM :', mean_std(a_cor))
    print('levenshtein :', mean_std(a_lev))
    print('     mannan :', mean_std(a_man))
#    print('  scanmatch :', mean_std(a_sma))