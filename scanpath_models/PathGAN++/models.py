from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import VGG16

import keras
import h5py
import numpy as np
import scipy.io as io
import tensorflow as tf
import argparse

def decoder(lstm_activation=None, optimizer=None, weights=None, reduced =True):
    print ("Setting up decoder")
    # Decoder -------------------------------------------
    # 1. Scanpath input
    main_input = None
    shape = None
    if not reduced:
        main_input = Input(shape=(32,4))
        shape=(32,4)
    else:
        main_input = Input(shape=(32,3))
        shape = (32,3)
    x = LSTM(500, input_shape=shape, activation=lstm_activation, return_sequences=True)(main_input)
    x = BatchNormalization()(x)

    # 2. Image input
    aux_input = Input(shape=(224, 224, 3))
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=aux_input)
    vgg.trainable=False
    z = vgg.output
    z = Conv2D(100, (3,3), activation='linear')(z)
    z = LeakyReLU(alpha=0.3)(z)
    z = Flatten()(z) #1600
    z = RepeatVector(32)(z)
    z = Reshape((32,2500))(z)

    # 3. Merge
    x = concatenate([x,z])
    x = LSTM(100, activation=lstm_activation, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(100, activation=lstm_activation, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(100, activation=lstm_activation, return_sequences=True)(x)
    x = BatchNormalization()(x)
    output = LSTM(1, activation='sigmoid', return_sequences=True)(x)

    decoder = Model(inputs=[main_input, aux_input], outputs=output)

    # Don't train VGG layers
    for i in range(19):
        #print decoder.layers[i]
        decoder.layers[i].trainable = False

    decoder.compile(optimizer=optimizer, loss='binary_crossentropy', sample_weight_mode='temporal', metrics=['accuracy'])

    if weights != "-":
        print("Loading discriminator weights")
        decoder.load_weights(weights)

    return decoder

class DTWLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size: int = 32):
        super(DTWLoss, self).__init__()
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        tmp = []
        for item in range(self.batch_size):
            tf.print(f'Working on batch: {item}\n')
            s = y_true[item, :]
            t = y_pred[item, :]
            n, m = len(s), len(t)
            dtw_matrix = []
            for i in range(n + 1):
                line = []
                for j in range(m + 1):
                    if i == 0 and j == 0:
                        line.append(0)
                    else:
                        line.append(np.inf)
                dtw_matrix.append(line)

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = tf.abs(s[i - 1] - t[j - 1])
                    last_min = tf.reduce_min([dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]])
                    dtw_matrix[i][j] = tf.cast(cost, dtype=tf.float32) + tf.cast(last_min, dtype=tf.float32)

            temp = []
            for i in range(len(dtw_matrix)):
                temp.append(tf.stack(dtw_matrix[i]))

            tmp.append(tf.stack(temp)[n, m])
        return tf.reduce_mean(tmp)

def generator(n_hidden_gen=None, lstm_activation=None, dropout=None, optimizer=None, loss=None, weights=None, G=None, loss_weights=None, reduced=True):
    # Encoder -------------------------------------------
    print ("Setting up generator")

    generator = Sequential()
    main_input = Input(shape=(224, 224, 3))
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=main_input)
    vgg.trainable=False

    generator.add(vgg)
    generator.add(BatchNormalization())
    generator.add(Flatten())
    generator.add(Dense(n_hidden_gen, activation='linear'))
    generator.add(LeakyReLU(alpha=0.3))
    generator.add(RepeatVector(32))
    generator.add(Reshape((32, n_hidden_gen)))
    generator.add(LSTM(n_hidden_gen, activation=lstm_activation, return_sequences=True, dropout=dropout, recurrent_dropout=dropout,))
    generator.add(BatchNormalization())
    generator.add(LSTM(n_hidden_gen, activation=lstm_activation, return_sequences=True, dropout=dropout, recurrent_dropout=dropout,))
    generator.add(BatchNormalization())
    generator.add(LSTM(n_hidden_gen, activation=lstm_activation, return_sequences=True, dropout=dropout, recurrent_dropout=dropout,))
    generator.add(BatchNormalization())     #dopo qua si potrebbe aggiungere la saliency map
    if not reduced:
        generator.add(Dense(4, activation='sigmoid'))
    else:
        generator.add(Dense(3, activation='sigmoid'))
    
    # Compile
    generator.compile(loss=[loss, DTWLoss], optimizer=optimizer, sample_weight_mode='temporal', metrics=['accuracy', 'mae'], loss_weights=loss_weights)
    # generator.compile(loss=[loss], optimizer=optimizer, sample_weight_mode='temporal', metrics=['accuracy', 'mae'], loss_weights=loss_weights)

    # Load weights
    if weights != "-":
        print("Loading generator weights")
        generator.load_weights(weights)

    generator_parallel = generator

    return generator, generator_parallel


def gen_dec(content_loss=None, optimizer=None, loss_weights=None, generator=None, decoder=None, G=None, shape=(224, 224, 3)):

    print ("Setting up combined net")
    generator_input = Input(shape=shape)
    dec_img_input = Input(shape=shape)

    x = generator(generator_input)

    decoder.trainable=False

    output = decoder([x, dec_img_input])

    gen_dec = Model(inputs=[generator_input, dec_img_input], outputs=[output, x])

    gen_dec.compile(
                    loss=['binary_crossentropy', content_loss], 
                    optimizer=optimizer, 
                    sample_weight_mode='temporal',
                    metrics=['accuracy', 'mae'],
                    loss_weights=loss_weights
                   )

    gen_dec_parallel = gen_dec

    return gen_dec, gen_dec_parallel

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

def gradient_penalty_loss(y_true, y_pred,averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_square_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_square_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1-gradient_l2_norm)
    return K.mean(gradient_penalty)