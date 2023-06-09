import keras.backend as K
import numpy as np
from sal_imp_utilities import *
from tensorflow.keras.losses import KLDivergence, MeanSquaredError




def rescale(heatmap):
    '''
    Apply mimaxscaler algorithm.
    See https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    '''
    return (heatmap - K.min(heatmap)) / (K.epsilon() + K.max(heatmap) - K.min(heatmap))


def normalize(heatmap):
    '''
    Normalize heatmap values in the [0,1] range.
    '''
    heatmap = rescale(heatmap)
    heatmap /= (K.epsilon() + K.sum(heatmap))
    return heatmap

def whiten(heatmap):
    '''
    Apply whitening algorithm.
    See https://en.wikipedia.org/wiki/Whitening_transformation
    '''
    return (heatmap - K.mean(heatmap)) / (K.epsilon() + K.std(heatmap))


# def nss(ref_map, sal_map):
#     '''
#     Compute Normalized Scanpath Saliency score.
#     '''
#     # print(sal_map)
#     sal_map = whiten(sal_map)
#     print(sal_map)

#     mask =  K.cast(ref_map.)
#     # print(mask)
#     return K.mean(sal_map[mask])


def similarity(y_true, y_pred):
    '''
    Compute Similarity score.
    '''
    y_pred = normalize(y_pred)
    y_true = normalize(y_true)

    return K.sum(K.minimum(y_pred, y_true))



# KL-Divergence Loss
def kl_divergence(y_true, y_pred):

    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)

    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())


    # This constant was defined by Cornia et al. and is a bit arbitrary
    return K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=1), axis=1)

def kl_time(y_true, y_pred):
    if len(y_true.shape) == 5:
        ax = 2
    else:
        ax = 1
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)

    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    kl_out = K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=ax), axis=ax)

    if len(y_true.shape) == 5:
        kl_out = K.mean(kl_out, axis = 1)

    return kl_out

# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=1), axis=1)
    sum_x = K.sum(K.sum(y_true, axis=1), axis=1)
    sum_y = K.sum(K.sum(y_pred, axis=1), axis=1)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=1), axis=1)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=1), axis=1)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return num / den

def cc_time(y_true, y_pred):
    if len(y_true.shape) == 5:
        ax = 2
    else:
        ax = 1
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=ax), axis=ax), axis=ax),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=ax), axis=ax)
    sum_x = K.sum(K.sum(y_true, axis=ax), axis=ax)
    sum_y = K.sum(K.sum(y_pred, axis=ax), axis=ax)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=ax), axis=ax)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=ax), axis=ax)


    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    if len(y_true.shape) == 5:
        cc_out = K.mean(num / den, axis = 1)
    else:
        cc_out = num / den

    return cc_out

# Normalized Scanpath Saliency Loss
def nss_time(y_true, y_pred):
    if len(y_true.shape) == 5:
        ax = 2
    else:
        ax = 1

    maxi = K.max(K.max(y_pred, axis=ax), axis=ax)
    first_rep = K.repeat_elements(K.expand_dims(maxi, axis=ax),shape_r_out, axis=ax)
    max_y_pred = K.repeat_elements(K.expand_dims(first_rep, axis=ax+1), shape_c_out, axis=ax+1)
    y_pred /= max_y_pred

    if len(y_true.shape) == 5:
        y_pred_flatten = K.reshape(y_pred, (K.shape(y_pred)[0],K.shape(y_pred)[1],K.shape(y_pred)[2]*K.shape(y_pred)[3]*K.shape(y_pred)[4])) #K.batch_flatten(y_pred)
    else:
        y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    num = K.sum(K.sum(y_true * y_pred, axis=ax), axis=ax)
    den = K.sum(K.sum(y_true, axis=ax), axis=ax) + K.epsilon()

    if len(y_true.shape) == 5:
        nss_out = K.mean(num/den, axis = 1)
    else:
        nss_out = num/den

    return nss_out


def nss(y_true, y_pred):

    ax = 1

    if K.sum(K.sum(y_true, axis=ax), axis=ax) == 0:
        return 0

    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=ax), axis=ax), axis=ax+1),
                                                                   shape_r_out, axis=ax), axis=ax+1), shape_c_out, axis=ax+1)

    y_pred /= max_y_pred


    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)),
                                                               shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)),
                                                              shape_r_out, axis=ax)), shape_c_out, axis=ax+1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    den = K.sum(K.sum(y_true * y_pred, axis=ax), axis=ax)
    nom = K.sum(K.sum(y_true, axis=ax), axis=ax) + K.epsilon()

    nss_out = den/nom

    return nss_out


def cc_match(y_true, y_pred):
    '''Calculates CC between initial, mid and final timestep from both y_true and y_pred
    and calculates the mean absolute error between the CCs from y_true and from y_pred.
    Requires a y_true and y_pred to be tensors of shape (bs, t, r, c, 1)'''

    mid = 1 # y_true.shape[1].value//2
    ccim_true = cc_time(y_true[:,0,...], y_true[:,mid,...])
    ccmf_true = cc_time(y_true[:,mid,...], y_true[:,-1,...])

    ccim_pred = cc_time(y_pred[:,0,...], y_pred[:,mid,...])
    ccmf_pred = cc_time(y_pred[:,mid,...], y_pred[:,-1,...])

    return  (K.abs(ccim_true-ccim_pred) + K.abs(ccmf_true-ccmf_pred) )/2 

def kl_cc_nss_combined(lw=[10,-3,-10]):
    # DEPRECATED
    '''Loss function that combines cc, nss and kl. Beacuse nss receives a different ground truth than kl and cc (maps),
        the function requires y_true to contains both maps. It has to be a tensor with dimensions [bs, 2, r, c, 1]. y_pred also
        has to be a tensor of the same dim, so the model should add a 5th dimension between bs and r and repeat the predict map
        twice along that dim.
    '''
    def loss(y_true, y_pred):

        map_true = y_true[:,0,...]
        fix_true = y_true[:,1,...]
        pred = y_pred[:,0,...]


        k = kl_divergence(map_true, pred)
        c = correlation_coefficient(map_true, pred)
        n = nss(fix_true, pred)


        return lw[0]*k+lw[1]*c+lw[2]*n

    return loss


def loss_wrapper(loss, input_shape):
    shape_r_out, shape_c_out = input_shape
    print("shape r out, shape c out", shape_r_out, shape_c_out)
    def _wrapper(y_true, y_pred):
        return loss(y_true, y_pred)
    return _wrapper


def kl_new(y_true, y_pred):
    '''
    This function is for singleduration model. The old kl_divergence() may cause nan in training.
    '''
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=1), axis=1), axis=1),
                                                                   shape_r_out, axis=1), axis=2), shape_c_out, axis=2)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())
    kl = tf.keras.losses.KLDivergence()
    return kl(y_true,y_pred)

def kl_cc_combined(y_true, y_pred):
    # For Singleduration
    '''Loss function that combines cc, nss and kl. Beacuse nss receives a different ground truth than kl and cc (maps),
        the function requires y_true to contains both maps. It has to be a tensor with dimensions [bs, 2, r, c, 1]. y_pred also
        has to be a tensor of the same dim, so the model should add a 5th dimension between bs and r and repeat the predict map
        twice along that dim.
    '''

    #k = kl_time(y_true, y_pred)
    k = kl_new(y_true, y_pred)
    print('k=',k)
    #c = cc_time(y_true, y_pred)
    c = correlation_coefficient(y_true, y_pred)
    print('c=', c)
    return 10*k-3*c


def kl_cc_nss_combined_new(y_true, y_pred):
    # For Singleduration
    '''Loss function that combines cc, nss and kl. Beacuse nss receives a different ground truth than kl and cc (maps),
        the function requires y_true to contains both maps. It has to be a tensor with dimensions [bs, 2, r, c, 1]. y_pred also
        has to be a tensor of the same dim, so the model should add a 5th dimension between bs and r and repeat the predict map
        twice along that dim.
    '''
    print(y_true.shape, y_pred.shape)
    #k = kl_time(y_true, y_pred)
    k = kl_new(y_true, y_pred)
    print('k=',k)
    #c = cc_time(y_true, y_pred)
    c = correlation_coefficient(y_true, y_pred)
    n = nss(y_true, y_pred)
    s = similarity(y_true, y_pred)
    # mse = tf.keras.losses.MeanSquaredError()
    # m = mse(y_true, y_pred)
    
    # print('m=', m)
    return 10*k-3*c -1*s - 0.5 *n #+ 1 * m 

