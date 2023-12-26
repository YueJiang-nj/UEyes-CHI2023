#!/usr/bin/env python3
# coding: utf-8
# pylint: disable=E1101,C0103

'''
A set of metrics for evaluatation of eye-gaze scanpaths,
borrowed from https://github.com/rAm1n/saliency

Every function expects at least two arguments:
- P: Predicted scanpath: Numpy array of (x,y) values.
- Q: Reference scanpath: Numpy array of (x,y) values.

See Table 1 in https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
for a comprehensive summary of each metric.

External dependencies, to be installed e.g. via pip:
- editdistance
- fastdtw
- scipy
- numpy

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import re
from fastdtw import fastdtw
from scipy.spatial.distance import directed_hausdorff, euclidean
import numpy as np
import editdistance


EPS = np.finfo(np.float32).eps


def global_align(P, Q, SubMatrix=None, gap=0, match=1, mismatch=-1):
    '''
    Compute alignment between two Numpy sequences.
    '''
    UP, LEFT, DIAG, NONE = range(4)
    max_p = len(P)
    max_q = len(Q)
    score = np.zeros((max_p + 1, max_q + 1), dtype='f')
    pointer = np.zeros((max_p + 1, max_q + 1), dtype='i')

    pointer[0, 0] = NONE
    score[0, 0] = 0.0
    pointer[0, 1:] = LEFT
    pointer[1:, 0] = UP

    score[0, 1:] = gap * np.arange(max_q)
    score[1:, 0] = gap * np.arange(max_p).T

    for i in range(1, max_p + 1):
        ci = P[i - 1]
        for j in range(1, max_q + 1):
            cj = Q[j - 1]
            if SubMatrix is None:
                diag_score = score[i - 1, j - 1] + \
                    (cj == ci and match or mismatch)
            else:
                diag_score = score[i - 1, j - 1] + SubMatrix[cj][ci]
            up_score = score[i - 1, j] + gap
            left_score = score[i, j - 1] + gap

            if diag_score >= up_score:
                if diag_score >= left_score:
                    score[i, j] = diag_score
                    pointer[i, j] = DIAG
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT
            else:
                if up_score > left_score:
                    score[i, j] = up_score
                    pointer[i, j] = UP
                else:
                    score[i, j] = left_score
                    pointer[i, j] = LEFT

    align_j = ""
    align_i = ""
    while True:
        p = pointer[i, j]
        if p == NONE:
            break

        s = score[i, j]
        if p == DIAG:
            # align_j += Q[j - 1]
            # align_i += P[i - 1]
            i -= 1
            j -= 1
        elif p == LEFT:
            # align_j += Q[j - 1]
            # align_i += "-"
            j -= 1
        elif p == UP:
            # align_j += "-"
            # align_i += P[i - 1]
            i -= 1
        else:
            raise ValueError

    # return align_j[::-1], align_i[::-1]
    return score.max()


def scanpath_to_string(scanpath, height, width, Xbins=10, Ybins=10):
    '''
    Create a string representation of a scanpath.
    Example: np.array([(512, 362), (860, 250)]) in a 1024x724 display becomes the string 'fFdI' using 10x10 bins.
    '''
    height_step, width_step = height // Ybins, width // Xbins
    string = ''
    num = list()
    for i in range(scanpath.shape[0]):
        fixation = scanpath[i].astype(np.int32)
        xbin = fixation[0] // width_step
        ybin = ((height - fixation[1]) // height_step)
        corrs_x = chr(65 + xbin)
        corrs_y = chr(97 + ybin)
        string += (corrs_y + corrs_x)
        num += [(ybin * Xbins) + xbin]
    return string, num


def euclidean_distance(P, Q):
    '''
    Compute Euclidean distance between two Numpy arrays of the same length.
    '''
    if P.shape == Q.shape:
        return np.sqrt(np.sum((P - Q)**2))
    return np.nan


def hausdorff_distance(P, Q):
    '''
    Compute Hausdorff distance between two Numpy arrays.
    '''
    return max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])


def mannan_distance(P, Q):
    '''
    Compute Mannan distance between two Numpy arrays.
    '''
    dist = np.zeros((P.shape[0], Q.shape[0]))
    for idx_1, fix_1 in np.ndenumerate(P):
        for idx_2, fix_2 in np.ndenumerate(Q):
            dist[idx_1, idx_2] = euclidean_distance(fix_1, fix_2)

    return (1 / (P.shape[0] + Q.shape[0])) * \
        (np.power(dist.min(axis=0).sum(), 2) + np.power(dist.min(axis=1).sum(), 2))


def eyenalysis(P, Q):
    '''
    Compute the eyenalysis distance between two Numpy arrays of the same length.
    '''

    dist = np.zeros((P.shape[0], Q.shape[0]))
    for idx_1, fix_1 in np.ndenumerate(P):
        for idx_2, fix_2 in np.ndenumerate(Q):
            dist[idx_1, idx_2] = euclidean_distance(fix_1, fix_2)

    return (1 / (P.shape[0] + Q.shape[0])) * \
        (dist.min(axis=0).sum() + dist.min(axis=1).sum())


def levenshtein_distance(P, Q, height, width, Xbins=10, Ybins=10):
    '''
    Compute Levenshtein distance between two Numpy arrays.
    '''
    P, P_num = scanpath_to_string(P, height, width, Xbins, Ybins)
    Q, Q_num = scanpath_to_string(Q, height, width, Xbins, Ybins)

    return editdistance.eval(P, Q)


def scan_match(P, Q, height, width, Xbins=10, Ybins=10,
               SubMatrix=None, threshold=5, GapValue=0):
    '''
    Compute ScanMatch distance between two Numpy arrays.
    '''
    def _create_sub_matrix(Xbins, Ybins, threshold):
        mat = np.zeros((Xbins * Ybins, Xbins * Ybins))
        idx_i = 0
        idx_j = 0

        for i in range(Ybins):
            for j in range(Xbins):
                for ii in range(Ybins):
                    for jj in range(Xbins):
                        mat[idx_i, idx_j] = np.sqrt((j - jj)**2 + (i - ii)**2)
                        idx_i += 1

                        if idx_i == Ybins:
                            idx_i = Ybins - 1
            idx_i = 0
            idx_j += 1

            if idx_j == Xbins:
                idx_j = Xbins - 1

        max_sub = mat.max()
        return np.abs(mat - max_sub) - (max_sub - threshold)

    P, P_num = scanpath_to_string(P, height, width, Xbins, Ybins)
    Q, Q_num = scanpath_to_string(Q, height, width, Xbins, Ybins)

    if SubMatrix is None:
        SubMatrix = _create_sub_matrix(Xbins, Ybins, threshold)

    score = global_align(P_num, Q_num, SubMatrix, GapValue)
    scale = SubMatrix.max() * max(len(P_num), len(Q_num)) + EPS

    return score / scale


def frechet_distance(P, Q):
    '''
    Compute Frechet distance between two Numpy arrays.
    See http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
    '''
    def _c(ca, i, j, P, Q):
        if ca[i, j] > -1:
            return ca[i, j]

        if i == 0 and j == 0:
            ca[i, j] = euclidean(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euclidean(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euclidean(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(_c(ca, i - 1, j, P, Q), _c(ca, i - 1,
                    j - 1, P, Q), _c(ca, i, j - 1, P, Q)),
                    euclidean(P[i], Q[j]))
        else:
            ca[i, j] = float("inf")

        return ca[i, j]

    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)

    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)


def DTW(P, Q):
    '''
    Compute Dynamic Time Warping distance between two Numpy arrays.
    '''
    dist, _ = fastdtw(P, Q, dist=euclidean)
    return dist


def TDE(P, Q, k=2, distance_mode='Mean'):
    '''
    Compute Time Delay Embedding distance between two Numpy arrays.
    See https://github.com/dariozanca/FixaTons/
    '''
    # k must be shorter than both lists lenghts
    if len(P) < k or len(Q) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return np.nan

    # create time-embedding vectors for both scanpaths
    P_vectors = []
    for i in np.arange(0, len(P) - k + 1):
        P_vectors.append(P[i:i + k])
    Q_vectors = []
    for i in np.arange(0, len(Q) - k + 1):
        Q_vectors.append(Q[i:i + k])

    distances = []

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k
    for s_k_vec in Q_vectors:
        norms = []
        for h_k_vec in P_vectors:
            d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
            norms.append(d)
        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.
    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)

    print(f'Unknown distance mode: {distance_mode}.')
    return np.nan


def coincidence_matrix(P, Q, threshold=0.05):
    '''
    Compute alignment matrix between two Numpy sequences of the same length.
    '''
    assert P.shape == Q.shape
    s = P.shape[0]
    c = np.zeros((s, s))

    for i in range(s):
        for j in range(s):
            if euclidean(P[i], Q[j]) < threshold:
                c[i, j] = 1
    return c


def recurrence(P, Q, threshold=0.05):
    '''
    Compute Recurrence distance between two Numpy arrays.
    See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
    '''
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = coincidence_matrix(P, Q, threshold)
    R = np.triu(c, 1).sum()

    return 100 * (2 * R) / (min_len * (min_len - 1))


def determinism(P, Q, threshold=0.05):
    '''
    Compute Determinism distance between two Numpy arrays.
    See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
    '''
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = coincidence_matrix(P, Q, threshold)
    R = np.triu(c, 1).sum() + EPS

    counter = 0
    for i in range(1, min_len):
        data = c.diagonal(i)
        data = ''.join([str(int(item)) for item in data])
        similar_subsequences = re.findall('1{2,}', data)
        for seq in similar_subsequences:
            counter += len(seq)

    return 100 * (counter / R)


def laminarity(P, Q, threshold=0.05):
    '''
    Compute Laminarity distance between two Numpy arrays.
    See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
    '''
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = coincidence_matrix(P, Q, threshold)
    R = np.triu(c, 1).sum() + EPS

    HL = 0
    HV = 0

    for i in range(min_len):
        data = c[i, :]
        data = ''.join([str(item) for item in data])
        similar_subsequences = re.findall('1{2,}', data)
        for seq in similar_subsequences:
            HL += len(seq)

    for j in range(min_len):
        data = c[:, j]
        data = ''.join([str(item) for item in data])
        similar_subsequences = re.findall('1{2,}', data)
        for seq in similar_subsequences:
            HV += len(seq)

    return 100 * ((HL + HV) / (2 * R))


def CORM(P, Q, threshold=0.05):
    '''
    Compute Center of recurrence mass (CORM) between two Numpy arrays.
    See https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
    '''
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = coincidence_matrix(P, Q, threshold)
    R = np.triu(c, 1).sum() + EPS

    counter = 0

    for i in range(0, min_len - 1):
        for j in range(i + 1, min_len):
            counter += (j - i) * c[i, j]

    return 100 * (counter / ((min_len - 1) * R))