import numpy as np
import keras
import sys
import os
from keras.layers import Layer, Input, Multiply, Dropout,DepthwiseConv2D, TimeDistributed, LSTM, Activation, Lambda, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, BatchNormalization, Concatenate
import keras.backend as K
from keras.models import Model
import tensorflow as tf
from keras.utils import Sequence
import cv2
import scipy.io
import math
from attentive_convlstm_new import AttentiveConvLSTM2D
from dcn_resnet_new import dcn_resnet
from gaussian_prior_new import LearningPrior
from sal_imp_utilities import *
from multiduration_models import decoder_block_timedist
from xception_custom import Xception_wrapper
from keras.applications import keras_modules_injection



def xception_cl(input_shape = (None, None, 3),
                 verbose=True,
                 print_shapes=True,
                 n_outs=1,
                 ups=8,
                 freeze_enc=False,
                 dil_rate = (2,2),
                 freeze_cl=True,
                 append_classif=True,
                 num_classes=5):
    """Xception with classification capabilities"""
    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
	        layer.trainable = False

    ### CLASSIFIER ###
    cl = GlobalAveragePooling2D(name='gap_cl')(xception.output)
    cl = Dense(512,name='dense_cl')(cl)
    cl = Dropout(0.3, name='dropout_cl')(cl)
    cl = Dense(num_classes, activation='softmax', name='dense_cl_out')(cl)

    ## DECODER ##
    outs_dec = decoder_block(xception.output, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=512, prefix='decoder')

    outs_final = [outs_dec]*n_outs

    if append_classif:
        outs_final.append(cl)

    # Building model
    m = Model(inp, outs_final) # Last element of outs_final is classification vector
    if verbose:
        m.summary()

    if freeze_cl:
        print('Freezing classification dense layers')
        m.get_layer('dense_cl').trainable = False
        m.get_layer('dense_cl_out').trainable = False

    return m

def xception_cl_fus(input_shape=(None, None, 3),
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                     dil_rate=(2,2),
                     freeze_enc=False,
                     freeze_cl=True,
                     internal_filts=256,
                     num_classes=5,
                     dp=0.3):
    """Xception with classification capabilities that fuses representations from both tasks"""
    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
            layer.trainable = False

    ### GLOBAL FEATURES ###
    g_n = global_net(xception.output, nfilts=internal_filts, dp=dp)
    if print_shapes: print('g_n shapes:', g_n.shape)

    ### CLASSIFIER ###
    # We potentially need another layer here
    out_classif = Dense(num_classes, activation='softmax', name='out_classif')(g_n)

    ### ASPP (MID LEVEL FEATURES) ###
    aspp_out = app(xception.output, internal_filts)
    if print_shapes: print('aspp out shapes:', aspp_out.shape)

    ### FUSION ###
    dense_f = Dense(internal_filts, name = 'dense_fusion')(g_n)
    if print_shapes: print('dense_f shapes:', dense_f.shape)
    reshap = Lambda(lambda x: K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), K.int_shape(aspp_out)[2], axis=1), axis=1), K.int_shape(aspp_out)[1], axis=1),
                lambda s: (s[0], K.int_shape(aspp_out)[1], K.int_shape(aspp_out)[2], s[1]))(dense_f)
    if print_shapes: print('after lambda shapes:', reshap.shape)

    conc = Concatenate()([aspp_out,reshap])

    ### Projection ###
    x = Conv2D(internal_filts, (1, 1), padding='same', use_bias=False, name='concat_projection')(conc)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(dp)(x)


    ### DECODER ###
    outs_dec = decoder_block(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=internal_filts, dp=dp)

    outs_final = [outs_dec]*n_outs
    outs_final.append(out_classif)

    # Building model
    m = Model(inp, outs_final) # Last element of outs_final is classification vector

    if freeze_cl:
        m.get_layer('out_classif').trainable = False
        # for l in g_n.layers:
        #     l.trainable=False

    if verbose:
        m.summary()

    return m



def xception_cl_fus_aspp(input_shape=(None, None, 3),
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                     dil_rate=(2,2),
                     freeze_enc=False,
                     freeze_cl=True,
                     internal_filts=256,
                     num_classes=4,
                     dp=0.3,
                     lambda_layer_for_save=False):

    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
            layer.trainable = False

    ### GLOBAL FEATURES ###
    g_n = global_net(xception.output, nfilts=internal_filts, dp=dp)
    if print_shapes: print('g_n shapes:', g_n.shape)

    ### CLASSIFIER ###
    # We potentially need another layer here
    out_classif = Dense(num_classes, activation='softmax', name='out_classif')(g_n)

    ### ASPP (MID LEVEL FEATURES) ###
    aspp_out = aspp(xception.output, internal_filts)
    if print_shapes: print('aspp out shapes:', aspp_out.shape)

    ### FUSION ###
    dense_f = Dense(internal_filts, name = 'dense_fusion')(g_n)
    if print_shapes: print('dense_f shapes:', dense_f.shape)

    if not lambda_layer_for_save:
        reshap = Lambda(lambda x: K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), K.int_shape(aspp_out)[2], axis=1), axis=1), K.int_shape(aspp_out)[1], axis=1),
                    lambda s: (s[0], K.int_shape(aspp_out)[1], K.int_shape(aspp_out)[2], s[1]))(dense_f)
    else: # Use this lambda layer if you want to be able to use model.save() (set lambda_layer_for_save to True)
        print("Using lambda layer adapted to model.save()")
        reshap = Lambda(lambda x: K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), 40, axis=1), axis=1), 30, axis=1),
                    lambda s: (s[0], 30, 40, s[1]))(dense_f)
        # reshap = FusionReshape()(dense_f)

    if print_shapes: print('after lambda shapes:', reshap.shape)

    conc = Concatenate()([aspp_out,reshap])

    ### Projection ###
    x = Conv2D(internal_filts, (1, 1), padding='same', use_bias=False, name='concat_projection')(conc)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(dp)(x)


    ### DECODER ###
    outs_dec = decoder_block(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=internal_filts, dp=dp)

    outs_final = [outs_dec]*n_outs
    outs_final.append(out_classif)

    # Building model
    m = Model(inp, outs_final,name = 'xception_cl_fus_aspp') # Last element of outs_final is classification vector

    if freeze_cl:
        m.get_layer('out_classif').trainable = False
        # for l in g_n.layers:
        #     l.trainable=False

    if verbose:
        m.summary()

    return m




def umsi(input_shape=(None, None, 3),
                     verbose=True,
                     print_shapes=True,
                     n_outs=1,
                     ups=8,
                     dil_rate=(2,2),
                     freeze_enc=False,
                     freeze_cl=True,
                     internal_filts=256,
                     num_classes=4,
                     dp=0.3,
                     lambda_layer_for_save=False):

    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
    if freeze_enc:
        for layer in xception.layers:
            layer.trainable = False
            
#     xception.summary()
    
    skip_layers = ['block3_sepconv2_bn','block1_conv1_act']
    # sizes: 119x159x32, 59x79x256
    skip_feature_maps = [xception.get_layer(n).output for n in skip_layers]

    ### GLOBAL FEATURES ###
    g_n = global_net(xception.output, nfilts=internal_filts, dp=dp)
    if print_shapes: print('g_n shapes:', g_n.shape)

    ### CLASSIFIER ###
    # We potentially need another layer here
    out_classif = Dense(num_classes, activation='softmax', name='out_classif')(g_n)

    ### ASPP (MID LEVEL FEATURES) ###
    aspp_out = aspp(xception.output, internal_filts)
    if print_shapes: print('aspp out shapes:', aspp_out.shape)

    ### FUSION ###
    dense_f = Dense(internal_filts, name = 'dense_fusion')(g_n)
    if print_shapes: print('dense_f shapes:', dense_f.shape)

    if not lambda_layer_for_save:
        reshap = Lambda(lambda x: K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), K.int_shape(aspp_out)[2], axis=1), axis=1), K.int_shape(aspp_out)[1], axis=1),
                    lambda s: (s[0], K.int_shape(aspp_out)[1], K.int_shape(aspp_out)[2], s[1]))(dense_f)
    else: # Use this lambda layer if you want to be able to use model.save() (set lambda_layer_for_save to True)
        print("Using lambda layer adapted to model.save()")
        reshap = Lambda(lambda x: K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), 40, axis=1), axis=1), 30, axis=1),
                    lambda s: (s[0], 32, 32, s[1]))(dense_f)
        # reshap = FusionReshape()(dense_f)

    if print_shapes: print('after lambda shapes:', reshap.shape)

    conc = Concatenate()([aspp_out,reshap])

    ### Projection ###
    x = Conv2D(internal_filts, (1, 1), padding='same', use_bias=False, name='concat_projection')(conc)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(dp)(x)

    ### DECODER ###
#     outs_dec = decoder_block(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=internal_filts, dp=dp)
    
    outs_dec = decoder_with_skip(x, 
                                 skip_feature_maps,  
                                 print_shapes=print_shapes, 
                                 dec_filt=internal_filts, 
                                 dp=dp)

    outs_final = [outs_dec]*n_outs
    outs_final.append(out_classif)
    

    # Building model
    m = Model(inp, outs_final, name = 'umsi') # Last element of outs_final is classification vector

    if freeze_cl:
        m.get_layer('out_classif').trainable = False
        # for l in g_n.layers:
        #     l.trainable=False

    if verbose:
        m.summary()

    return m

def xception_cl_fus_skipdec(input_shape=(None, None, 3),
                             verbose=True,
                             print_shapes=True,
                             n_outs=1,
                             ups=8,
                             dil_rate=(2,2),
                             freeze_enc=False,
                             freeze_cl=True,
                             internal_filts=256,
                             num_classes=5,
                             dp=0.3):

    inp = Input(shape=input_shape)

    ### ENCODER ###
    xception = Xception_wrapper(include_top=False, weights='imagenet', input_tensor=inp, pooling=None)
    if print_shapes: print('xception output shapes:',xception.output.shape)
        
    xception.summary()    
    
    if freeze_enc:
        for layer in xception.layers:
            layer.trainable = False

    ### GLOBAL FEATURES ###
    g_n = global_net(xception.output, nfilts=internal_filts, dp=dp)
    if print_shapes: print('g_n shapes:', g_n.shape)

    ### CLASSIFIER ###
    # We potentially need another layer here
    out_classif = Dense(num_classes, activation='softmax', name='out_classif')(g_n)

    ### ASPP (MID LEVEL FEATURES) ###
    aspp_out = aspp(xception.output, internal_filts)
    if print_shapes: print('aspp out shapes:', aspp_out.shape)

    ### FUSION ###
    dense_f = Dense(internal_filts, name = 'dense_fusion')(g_n)
    if print_shapes: print('dense_f shapes:', dense_f.shape)
    reshap = Lambda(lambda x: K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), K.int_shape(aspp_out)[2], axis=1), axis=1), K.int_shape(aspp_out)[1], axis=1),
                lambda s: (s[0], K.int_shape(aspp_out)[1], K.int_shape(aspp_out)[2], s[1]))(dense_f)
    if print_shapes: print('after lambda shapes:', reshap.shape)

    conc = Concatenate()([aspp_out,reshap])

    ### Projection ###
    x = Conv2D(internal_filts, (1, 1), padding='same', use_bias=False, name='concat_projection')(conc)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(dp)(x)

    ### DECODER ###
    outs_dec = decoder_block(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=internal_filts, dp=dp)
    
#     outs_dec = decoder_with_skip(x, dil_rate=dil_rate, print_shapes=print_shapes, dec_filt=internal_filts, dp=dp)

    outs_final = [outs_dec]*n_outs
    outs_final.append(out_classif)

    # Building model
    m = Model(inp, outs_final) # Last element of outs_final is classification vector

    if freeze_cl:
        m.get_layer('out_classif').trainable = False
        # for l in g_n.layers:
        #     l.trainable=False

    if verbose:
        m.summary()

    return m


def global_net(x, nfilts=512, dp=0.1, print_shapes = True):

    x = Conv2D(nfilts, (3, 3), strides=3, padding='same', use_bias=False, name='global_conv')(x)
    if print_shapes: print('Shape after global net conv:', x.shape)
    x = BatchNormalization(name='global_BN',epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(dp)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nfilts, name='global_dense')(x)
    x = Dropout(dp)(x)

    return x


def app(x, nfilts=256, prefix='app', dils=[6,12,18]):

    x1 = Conv2D(nfilts, 1, padding='same', activation='relu', dilation_rate=(1,1), name=prefix+'_c1x1')(x)
    x2 = Conv2D(nfilts, 3, padding='same', activation='relu', dilation_rate=(dils[0],dils[0]), name=prefix+'_c3x3d'+str(dils[0]))(x)
    x3 = Conv2D(nfilts, 3, padding='same', activation='relu', dilation_rate=(dils[1],dils[1]), name=prefix+'_c3x3d'+str(dils[1]))(x)
    x4 = Conv2D(nfilts, 3, padding='same', activation='relu', dilation_rate=(dils[2],dils[2]), name=prefix+'_c3x3d'+str(dils[2]))(x)

    x = Concatenate()([x1,x2,x3,x4])

    return x

def aspp(x, nfilts=256, prefix='aspp', dils=[6,12,18]):

    x1 = Conv2D(nfilts, (1, 1), padding='same', use_bias=False, name=prefix+'_csep0')(x)
    x1 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(x1)
    x1 = Activation('relu', name='aspp0_activation')(x1)

    # rate = 6
    x2 = SepConv_BN(x, nfilts, prefix+'_csepd'+str(dils[0]), rate=dils[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    x3 = SepConv_BN(x, nfilts, prefix+'_csepd'+str(dils[1]),rate=dils[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    x4 = SepConv_BN(x, nfilts, prefix+'_csepd'+str(dils[2]),rate=dils[2], depth_activation=True, epsilon=1e-5)

    x = Concatenate()([x1,x2,x3,x4])

    return x


def decoder_with_skip(x, skip_tensors, dil_rate=1, print_shapes=True, dec_filt=1024, dp=0.2, ups=16, prefix='decskip'):

        # sizes of input skip connections from Xception: 119x159x32, 117x157x128, 59x79x256
        
        for i, sk in enumerate(skip_tensors, start=1):
            # Upsample
            x = UpSampling2D((2,2), interpolation='bilinear', name=prefix+'_ups%d'%i)(x)
            if x.shape[1] != sk.shape[1] or x.shape[2] != sk.shape[2]:
                x = Lambda(lambda t: tf.image.resize(t, (K.int_shape(sk)[1], K.int_shape(sk)[2])))(x)

            # Concatenate
            x = Concatenate()([x, sk])
            
            # Convolve to reduce feature dimensionality
            x = Conv2D(dec_filt//2**i, (1, 1), padding='same', use_bias=False, name=prefix+'_proj_%d'%i)(x) 
            x = BatchNormalization(name=prefix+'_bn_%d'%i, epsilon=1e-5)(x)
            x = Activation('relu', name=prefix+'_act_%d'%i)(x)            
            
            # Convolve with depth sep convs
            x = SepConv_BN(x, 
                           dec_filt//2**i, 
                           kernel_size=3,
                           depth_activation=True, 
                           epsilon=1e-5, 
                           rate=dil_rate, 
                           prefix=prefix+'_sepconvA_%d'%i)
            x = SepConv_BN(x, 
                           dec_filt//2**i, 
                           kernel_size=3,
                           depth_activation=True, 
                           epsilon=1e-5, 
                           rate=dil_rate, 
                           prefix=prefix+'_sepconvB_%d'%i)
            x = Dropout(dp, name=prefix+'_dp%d'%i)(x)
    
            
            print("shape after block %d of dec:"%i, x.shape)
        
        
        # Upsampling and normal conv
#         i+=1
#         x = UpSampling2D((2,2), interpolation='bilinear', name=prefix+'_ups_prefinal')(x)
#         x = Conv2D(dec_filt//2**i, (3, 3), padding='same', use_bias=True, name=prefix+'_conv_%d'%i)(x) 
#         x = BatchNormalization(name=prefix+'_bn_%d'%i, epsilon=1e-5)(x)
#         x = Activation('relu', name=prefix+'_act_%d'%i)(x)          
        
        # Final upsample to get to desired output size (480x640)
        x = UpSampling2D((4,4), interpolation='bilinear', name=prefix+'_ups_final')(x)
        if x.shape[1] != shape_r_out or x.shape[2] != shape_c_out:
            x = Lambda(lambda t: tf.image.resize(t, (shape_r_out, shape_c_out)))(x)

        if print_shapes: print('Shape after last ups and resize:',x.shape)
            
        # Final conv to get to a heatmap
        x = Conv2D(1, kernel_size=1, padding='same', activation='relu', name=prefix+'_c_out')(x)
        if print_shapes: print('Shape after 1x1 conv:',x.shape)

        return x

def decoder_block(x, dil_rate=(2,2), print_shapes=True, dec_filt=1024, dp=0.2, ups=16, prefix='dec'):

    # Dilated convolutions
    x = Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=dil_rate, name=prefix+'_c1')(x)
    x = Conv2D(dec_filt, 3, padding='same', activation='relu', dilation_rate=dil_rate, name=prefix+'_c2')(x)
    x = Dropout(dp, name=prefix+'_dp1')(x)
    x = UpSampling2D((2,2), interpolation='bilinear', name=prefix+'_ups1')(x)

    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=dil_rate, name=prefix+'_c3')(x)
    x = Conv2D(dec_filt//2, 3, padding='same', activation='relu', dilation_rate=dil_rate, name=prefix+'_c4')(x)
    x = Dropout(dp, name=prefix+'_dp2')(x)
    x = UpSampling2D((2,2), interpolation='bilinear', name=prefix+'_ups2')(x)

    x = Conv2D(dec_filt//4, 3, padding='same', activation='relu', dilation_rate=dil_rate, name=prefix+'_c5')(x)
    x = Dropout(dp, name=prefix+'_dp3')(x)
    x = UpSampling2D((4,4), interpolation='bilinear', name=prefix+'_ups3')(x)

    if print_shapes: print('Shape after last ups:',x.shape)

    # Final conv to get to a heatmap
    x = Conv2D(1, kernel_size=1, padding='same', activation='relu', name=prefix+'_c_out')(x)
    if print_shapes: print('Shape after 1x1 conv:',x.shape)

    return x




class FusionReshape(Layer):

    def __init__(self, **kwargs):
        super(FusionReshape, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FusionReshape, self).build(input_shape) 

    def call(self, x):
        return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(x, axis=1), 40, axis=1), axis=1), 30, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 30, 40, input_shape[1])


##### DEEPLAB V3 CODE #####

def SepConv_BN(x, filters, prefix='scb', stride=1, kernel_size=3, rate=1,
                depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x
