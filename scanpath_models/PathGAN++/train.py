from tensorflow.keras.optimizers import SGD, RMSprop
import numpy as np
import models
import utils
import os
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from math import floor, ceil
import pickle
import tensorflow.keras.backend as K


def preprocess_images(images):
    means = np.zeros((1,1,3),dtype=np.float32)
    means[0,0,0] = 123.68
    means[0,0,1] = 116.78
    means[0,0,2] = 103.94
    images = images - means
    return images

def preprocess_scanpaths(scanpaths):
    return scanpaths

def train(reduced):
    loss_weights = [1., 0.05]
    adversarial_iteration = 2
    batch_size = 16
    mini_batch_size = 20
    G = 1
    epochs = 200
    n_hidden_gen = 1000
    lrgen = 1e-4
    lrdiscr = 1e-4
    content_loss = 'mse'
    lstm_activation = 'tanh'
    dropout = 0.1
    weights_generator = '-'
    weights_discriminator = '-'
    optgen = RMSprop(lr=lrgen, rho=0.9, epsilon=1e-08, decay=0.0)
    optdiscr = RMSprop(lr=lrdiscr, rho=0.9, epsilon=1e-08, decay=0.0)

    # Load image
    imgs = None
    scanpaths = None
    scannoadd = None
    if not os.path.isfile('data/scanpaths.npy'):
        print(reduced)

        imgs, scanpaths, imgsname = utils.load_data(reduced)
        scanpaths = preprocess_scanpaths(scanpaths)
        # print(scanpaths.shape)
        # exit()
        print("...Saving the Dataset...")
        np.save("data/images", imgs)
        np.save("data/scanpaths", scanpaths)
        with open('data/imgsnames.pkl', 'wb') as f:
            pickle.dump(imgsname, f)
        print("...Dataset Saved...")
    else:
        print("...Loading the Dataset...")
        imgs = np.load("data/images.npy")
        scanpaths = np.load("data/scanpaths.npy")
        with open('data/imgsnames.pkl', 'rb') as f:
            imgsname = pickle.load(f)
        print("...Dataset Loaded...")
    print("...Verifying the presences of Nan's...")
    print(np.isnan(np.sum(imgs)))
    print(np.isnan(np.sum(scanpaths)))
    print("...Done...")
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    # Creating a training and testing set with a 90-10 split
    if not os.path.isfile('data/x_train.npy'):
        print("...Creating the training and testing sets...")
        X_train, X_test, Y_train, Y_test = utils.train_test(imgs, scanpaths, imgsname)
        np.save("data/x_train", X_train)
        np.save("data/x_test", X_test)
        np.save("data/y_train", Y_train)
        np.save("data/y_test", Y_test)
        print("...Done...")
    else:
        print("...Loading the training and testing sets...")
        X_train = np.load("data/x_train.npy")
        X_test = np.load("data/x_test.npy")
        Y_train = np.load("data/y_train.npy")
        Y_test = np.load("data/y_test.npy")
        print("...Done...")

    # uncomment for training from previous weigths
    # weights_generator = 'weights/generator++.h5'
    # weights_discriminator = 'weights/discriminator++.h5'
    # step = 0

    step = 0 # comment for pursuing training from previous weigths
    # Get the model
    params = {
        'n_hidden_gen': n_hidden_gen,
        'lstm_activation': lstm_activation,
        'dropout': dropout,
        'optimizer': optgen,
        'loss': content_loss,
        'weights': weights_generator,
        'G': G,
        'reduced' : reduced
    }
    _, generator = models.generator(**params)
    discriminator = models.decoder(lstm_activation=lstm_activation, optimizer=optdiscr, weights=weights_discriminator, reduced=reduced)
    _, gan = models.gen_dec(generator=generator, decoder=discriminator, content_loss=content_loss, optimizer=optgen, loss_weights=loss_weights, G=1)
    print("discriminator metrics")
    print(discriminator.metrics_names)
    print("gan metrics")
    print(gan.metrics_names)

    print("validation")
    print(X_test.shape, Y_test.shape)
    # gen_val = generator.evaluate(X_test, Y_test, batch_size=16)
    # print(gen_val)

    discr_loss = []
    gen_loss = []
    gen_accuracy = []
    step_list = []
    batch_count = ceil(X_train.shape[0] / batch_size)
    epoch_num = step / batch_count

    print("Starting Training")

    loss = 1
    while epoch_num<51:
        epoch_num = step / batch_count
        print(epoch_num)
        sample_interval = 50
        y_discr, y_gan = utils.training_output(batch_size)
        idx = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
        #fix the randomness
        d_loss_avg = 0
        loop = 4
        if loss < 0.05:
            loop = 1
        else:
            loop = 4
        for i in range(loop):
            noise = np.random.normal(0, 3, X_train[idx].shape)
            imgs_train = X_train[idx]#images
            scanpath_train = Y_train[idx] #real scanpaths

            gen_scanpath_train = generator.predict(imgs_train)

            imgs_train += noise

            X_scanpath = np.concatenate([gen_scanpath_train, scanpath_train]) #input scanpaths
            X_image = np.concatenate([imgs_train, imgs_train])

            discriminator.trainable = True
            d_loss = discriminator.train_on_batch([X_scanpath, X_image], y_discr)
            loss = d_loss[0]
            if loss < 0.05:
                K.set_value(discriminator.optimizer.lr, 5e-5)
            else:
                K.set_value(discriminator.optimizer.lr, 1e-4)
            d_loss_avg += d_loss[0]

        discriminator.trainable = False

        idx = np.random.randint(low=0, high=X_train.shape[0], size=batch_size)
        g_loss_avg = 0
        for i in range(16):
            imgs_train = X_train[idx]  # images
            scanpath_train = Y_train[idx]  # real scanpaths
            imgs_train_discr = imgs_train.copy()

            g_loss = gan.train_on_batch([imgs_train, imgs_train_discr], [y_gan, scanpath_train])
            g_loss_avg += g_loss[2]

        gen_loss.append(g_loss_avg/8)
        discr_loss.append(d_loss_avg/loop)
        print('g: ', g_loss_avg/8)
        print('d: ', d_loss_avg/loop)
        step += 1
        step_list.append(step)
        print("validation")

        utils.sample_images(epoch_num, step, generator, discriminator, X_test, Y_test, reduced, "training-proof-1.png")
        utils.sample_images(epoch_num, step, generator, discriminator, X_train, Y_train, reduced, "training-proof-2.png")

# Set this to False if you trained the model to output 4 features (including the duration)
reduced = False

train(reduced)
