# (c) Copnano yright 2017 Marc Assens. All Rights Reserved.
__author__ = "Marc Assens"
__version__ = "0.1"
# Modified by Paul Houssel

import models
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import VGG16
from time import gmtime, strftime
import keras
import h5py
import numpy as np
import scipy.io as io
import tensorflow as tf
import argparse
import os
import cv2
import utils

def prepare_image(scanpaths):
    a = []
    for i in range(scanpaths.shape[0]):
      if i == 0:
        a = scanpaths[i,:,:]
      else:
        a = np.vstack([a, scanpaths[i,:,:], i+1])
    return a

def predict(img_path, reduced):
    loss_weights            = [1., 0.05]
    adversarial_iteration   = 2
    batch_size              = 40
    mini_batch_size         = 800
    G                       = 1
    epochs                  = 200
    n_hidden_gen            = 1000
    lr                      = 1e-4
    content_loss            = 'mse'
    lstm_activation         = 'tanh'
    dropout                 = 0.1
    weights_generator       = 'weights/generator_PathGAN++.h5'
    opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)


    # Load image
    img = utils.load_image_predict(img_path)

    # Get the model
    params = {
        'n_hidden_gen':n_hidden_gen,
        'lstm_activation':lstm_activation,
        'dropout':dropout,
        'optimizer':opt,
        'loss':content_loss,
        'weights':weights_generator,
        'G':G,
        'reduced': reduced
    }
    _, generator_parallel = models.generator(**params)

    # Predict with a model
    n_sp_per_image = 1

    #provisional
    if reduced:
        output = np.zeros((n_sp_per_image, 32, 3))
    else:
        output = np.zeros((n_sp_per_image, 32, 4))
    for i in range(n_sp_per_image):
        print("Calculating observer %d" % i)
        noise  = np.random.normal(0,3,img.shape)
        noisy_img = img + noise
        prediction = generator_parallel.predict(noisy_img)
        output[i] = prediction

    # Prepare the predictions for matlab and save it on individual files
    output = prepare_image(output[:,:,:])
    
    return output


def save_scanpath_as_csv(scanpaths, out_path, in_path, reduced):
    name = '.'.join(in_path.split('/')[-1].split('.')[:-1])

    out_path = out_path + '%s.csv' % name
    if reduced:
        with open(out_path, "w") as saveFile:
            saveFile.write("x, y, timestamp\n")
            for i in range(scanpaths.shape[0]):
                lon = scanpaths[i, 0]
                lat = scanpaths[i, 1]
                tim = scanpaths[i, 2]
                saveFile.write("{}, {}, {}\n".format(
                    lon, lat, tim
                    )
                )
            print('Saved scanpaths from image %s in file %s' % (in_path, out_path))
    else:
        with open(out_path, "w") as saveFile:
            saveFile.write("x, y, timestamp, duration\n")
            for i in range(scanpaths.shape[0]):
                lon = scanpaths[i, 0]
                lat = scanpaths[i, 1]
                tim = scanpaths[i, 2]
                duration = scanpaths[i,3]
                saveFile.write("{}, {}, {}, {}\n".format(
                    lon, lat, tim, duration
                    )
                )
            print('Saved scanpaths from image %s in file %s' % (in_path, out_path))

def predict_and_save(imgs_path, out_path, reduced):

    # Preproces and load images
    paths = utils.file_paths_for_images(imgs_path)
    for i, path in enumerate(paths):
        print('Working on image %d of %d' % (i+1, len(paths)))

        # Predict the scanpaths
        scanpaths = predict(path, reduced)

        # Turn into a float np.array
        scanpaths = np.array(scanpaths, dtype=np.float32)

        # Save in output folder
        save_scanpath_as_csv(scanpaths, out_path, path, reduced)

        # Save as an image
        first = True
        xold = 0
        yold = 0

        prednum = 0
        # TODO: The predicted duration of the fixation is not visualised
        image = cv2.imread(path)
        width = image.shape[1]
        height = image.shape[0]
        for row in scanpaths:
            x = row[0]*width
            y = row[1]*height
            xint = int(np.round(x))
            yint = int(np.round(y))
            if xint < width and yint < height and xint >= 0 and yint >= 0:
                if first:
                    first = False
                    image[yint - 2:yint + 2, xint - 2:xint + 2] = [0, 255, 0]
                    xold = xint
                    yold = yint
                else:
                    image[yint - 2:yint + 2, xint - 2:xint + 2] = [255, 255, 255]
                    cv2.line(image, (xold, yold), (xint, yint), (0, 100, 255), 3)
                    xold = xint
                    yold = yint
        cv2.imwrite(out_path + str( path.split('/')[-1][:-4]) + ".png", image)

    print('Done!')
    return True


if __name__ == "__main__":
    # Set this to False if you trained the model to output 4 features for each fixation: x, y, t, duration
    reduced = True

    img_path = './inputs/'
    out_path = './outputs/'

    print('\n\n###########################')
    predict_and_save(img_path, out_path, reduced)
    print('\n\n###########################')
