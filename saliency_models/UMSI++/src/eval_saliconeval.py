"""
Eval functions used by the SUN 2017 challenge. 
"""
import numpy as np
import scipy
from sal_imp_utilities import *
import tqdm

#def get_stats_saliconeval(model, gen_eval, imsize=(480, 640), n=None, blur_sigma=7, verbose=True): 
#    tot_cc = []
#    tot_nss = []
#    tot_auc = []
#    #tot_sauc = []
#    c = 0
#
#    if not n: 
#        data = [elt for elt in gen_eval]
#    else: 
#        data = [next(gen_eval) for _ in range(n)]
#    print("loaded data")
#    print("len data", len(data))
#
#    # build shuf map (used for sauc)
##    shufMap = np.zeros(imsize)
##    for _, _, fixmap, _ in data: 
##        shufMap += fixmap
##    print("shufmap", shufMap.shape)
#    
#    for img, gt_map, gt_fix_map, gt_fix_points in data:
#        # get img names
#        if c%100 == 0: 
#           print(c)
#        c+=1
#
#        b, w, h, ch = img.shape
#        preds = model.predict(img)
#        map_idx = 0
#        fix_idx = 1
#        for time in range(len(preds)): 
#        #for time in [1]: 
#            for batch in range(len(preds[time])):
#                p = preds[time][batch][map_idx] # the 0 is to get rid of the duplicated dimension 
#                if (blur_sigma): 
#                    p = ndimage.filters.gaussian_filter(p, [blur_sigma, blur_sigma, 1])
#
#                p = (p-np.min(p))/np.max(p)
#
#                tot_cc.append(calc_cc(gt_map, p))
#                tot_nss.append(calc_nss(gt_fix_points, p))
#                tot_auc.append(calc_auc(gt_fix_points, p))
#                #tot_sauc.append(calc_sauc(gt_fix_points, p, shufMap))
#    
#    ret_cc = np.mean(tot_cc)
#    ret_nss = np.mean(tot_nss)
#    ret_auc = np.mean(tot_auc)
#    #ret_sauc = np.mean(tot_sauc)
#    if (verbose):
#        print()
#        print("CC:", ret_cc)
#        print("NSS:", ret_nss)
#        print("AUC:", ret_auc)
#        #print("SAUC:", ret_sauc)
#    return ret_cc, ret_nss, ret_auc
#
#def fix_coords_from_map(fixmap): 
#    return np.argwhere(fixmap>0) + np.ones(shape=fixmap.shape)
#
def cc_saliconeval(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def nss_saliconeval(gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    return np.mean([ salMap[y-1][x-1] for x,y in gtsAnn ])

def auc_saliconeval(gtsAnn, resAnn, stepSize=.01, Nrand=100000):
    """
    Computer AUC score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    S = salMap.reshape(-1)
    Sth = np.asarray([ salMap[y-1][x-1] for x,y in gtsAnn ])

    Nfixations = len(gtsAnn)
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    np.concatenate((Sth, randfix), axis=0)
    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(randfix >= thresh))/Nrand for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc

def sauc_saliconeval(gtsAnn, resAnn, shufMap, stepSize=.01):
    # TODO: CURRENTLY BROKEN
    """
    Computer SAUC score. A simple implementation
    :param gtsAnn : list of fixation annotataions
    :param resAnn : list only contains one element: the result annotation - predicted saliency map
    :return score: int : score
    """
    #print("shufmap 2", shufMap.shape)

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)
    Sth = np.asarray([ salMap[y-1][x-1] for x,y in gtsAnn ])
    Nfixations = len(gtsAnn)

    others = np.copy(shufMap)
    #print("others shape", others.shape)
    for x,y in gtsAnn:
        others[y-1][x-1] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    #print("ind", ind[0].shape)
    nFix = shufMap[ind]
    #print("nFix", nFix.shape)
    randfix = salMap[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    #print("fp shape", fp.shape)
    #print("allthreshes", allthreshes.shape)
    tp[-1]=1.0
    fp[-1]=1.0
    #print("randfix", randfix.shape)
    #print("allthreshes[0]", allthreshes[0].shape)
    #print("nFix", nFix)
    #randfix>=allthreshes[0]
    nFix[randfix >= allthreshes[0]] # this throws an error
    val = [float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]
    #print("val shape", val.shape)
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc
