# (c) Copyright 2017 Marc Assens. All Rights Reserved.

__author__ = "Marc Assens"
__version__ = "1.0"
# Modified by Paul Houssel

from time import time
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import glob
import scipy.io
import os
import math
import cv2
import pickle


def load_image(img_path):
    img = image.load_img('images/' + img_path, target_size=(224, 224))
    
    x = image.img_to_array(img)
    
    x = preprocess_input(x)
    return x

def load_image_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def file_paths_for_images(path):
    extensions = ("*.png","*.jpg","*.jpeg",)
    paths = []
    for extension in extensions:
        paths.extend(glob.glob(path+"/*"+extension))
    return paths

    
def training_output(mini_batch_size):
    # Data for training
    y_decoder = 0.9*np.ones((2*mini_batch_size,32,1))
    y_decoder[:mini_batch_size,:,:] = 0.                        
    y_gen_dec = np.ones((mini_batch_size,32,1))                 

    return y_decoder, y_gen_dec

                                                            
def save_history(G, history, encoder_loss, encoder_acc, encoder_loss_mse, encoder_mae):
    print (history)

    if G > 1 : 
        encoder_loss.append(history['concatenate_5_loss'][0])
        encoder_acc.append(history['concatenate_5_acc'][0])
        encoder_loss_mse.append(history['concatenate_6_loss'][0])
        encoder_mae.append(history['concatenate_6_mean_absolute_error'][0])
        # encoder_loss.append(history['discriminator_loss'][0])
        # encoder_acc.append(history['discriminator_acc'][0])
        # encoder_loss_mse.append(history['generator_loss'][0])
        # encoder_mae.append(history['generator_mean_absolute_error'][0])
    else:
        encoder_loss.append(history['discriminator_loss'][0])
        encoder_acc.append(history['discriminator_acc'][0])
        encoder_loss_mse.append(history['generator_loss'][0])
        encoder_mae.append(history['generator_mean_absolute_error'][0])

    return encoder_loss, encoder_acc, encoder_loss_mse, encoder_mae

#loads path as x,y,timestamp with dummy nodes on the edges
def load_data(reduced):
    images = list()
    scanpaths = list()
    imagesname = list()

    with open('data/scanpaths.csv', 'r') as f:
        first = True
        scanpath = None
        image_file = None
        user = None
        scanpaths = []
        print("...Preprocessing all fixation points (can take up too 2 minutes)...")
        if reduced:
            for line in f:
                if 'ui_file,  username,  x,  y,  timestamp' not in line:
                    values = line.split(',')
                    if image_file == values[0] and user == values[1] :
                        # the same image and user as in the previous iteration
                        image_file = values[0]
                        username = values[1]
                        x_coordinate = float(values[2])
                        y_coordinate = float(values[3])
                        timestamp = int(values[4])    
                        if x_coordinate != None and y_coordinate != None:
                            x = 0 if float(x_coordinate) < 0 else float(x_coordinate)
                            y = 0 if float(y_coordinate) < 0 else float(y_coordinate)
                            x = 1 if x > 1 else x
                            y = 1 if y > 1 else y
                        scanpath.append([x, y, float(timestamp)]) # add the new point to the scanpath
                    else:

                        # A new fixation collection
                        image_file = values[0]
                        user = values[1]
                        x_coordinate = float(values[2])
                        y_coordinate = float(values[3])
                        timestamp = int(values[4])

                        if scanpath != None:
                            # Do not take into account
                            if len(scanpath) != 1:
                                scanpaths.append(scanpath)
                                img = load_image(image_file)
                                imagesname.append(image_file)
                                images.append(img)
                            scanpath = []
                        else :
                            img = load_image(image_file)
                            imagesname.append(image_file)
                            images.append(img)
                        scanpath = []
                        # Normalisation of the coordinates
                        if x_coordinate!=None and y_coordinate!=None:
                            x = 0 if float(x_coordinate) < 0 else float(x_coordinate)
                            y = 0 if float(y_coordinate) < 0 else float(y_coordinate)
                            x = 1 if x > 1 else x
                            y = 1 if y > 1 else y
                        scanpath.append([x, y, float(timestamp)])
        else: 

            for line in f:
                if 'ui_file,  username,  x,  y,  timestamp, duration' not in line:
                    values = line.split(',')
                    if image_file == values[0] and user == values[1] :
                        # the same image and user as in the previous iteration
                        image_file = values[0]
                        username = values[1]
                        x_coordinate = float(values[2])
                        y_coordinate = float(values[3])
                        timestamp = int(values[4])
                        duration = float(values[5])
                        if x_coordinate != None and y_coordinate != None:
                            x = 0 if float(x_coordinate) < 0 else float(x_coordinate)
                            y = 0 if float(y_coordinate) < 0 else float(y_coordinate)
                            x = 1 if x > 1 else x
                            y = 1 if y > 1 else y
                        scanpath.append([x, y, float(timestamp), float(duration)]) # add the new point to the scanpath
                    else:
                        # A new fixation collection
                        image_file = values[0]
                        user = values[1]
                        x_coordinate = float(values[2])
                        y_coordinate = float(values[3])
                        timestamp = int(values[4])
                        duration = float(values[5])

                        if scanpath != None:
                            # Do not take into account
                            if len(scanpath) != 1:
                                scanpaths.append(scanpath)
                                img = load_image(image_file)
                                imagesname.append(image_file)
                                images.append(img)
                            scanpath = []
                        else :
                            img = load_image(image_file)
                            imagesname.append(image_file)
                            images.append(img)
                        scanpath = []
                        # Normalisation of the coordinates
                        if x_coordinate!=None and y_coordinate!=None:
                            x = 0 if float(x_coordinate) < 0 else float(x_coordinate)
                            y = 0 if float(y_coordinate) < 0 else float(y_coordinate)
                            x = 1 if x > 1 else x
                            y = 1 if y > 1 else y
                        scanpath.append([x, y, float(timestamp), float(duration)])
                
        scanpaths.append(scanpath)
    print("...Preprocessing all fixation points DONE...")
    
    # print(len(images))
    # print(len(scanpaths))
    # print(len(imagesname))

    # print("...Applying Linear Interpolation to obtain fixation points of 32 coordinates...")
    # images = np.array(images)
    # scanpaths = np.array(scanpaths)
    # scanpaths_mod = list()
    # fraction = 1/31
    # for scan in scanpaths:
    #     scanpath = list()
    #     first = True
    #     length = len(scan) #-1 per ottenere numero archi
    #     toadd = 32-length # number of points to add
    #     forarch = math.floor(toadd/(length-1)) #punti da aggiungere per arco
    #     remain = toadd-(forarch*(length-1)) #punti rimanenti
    #     oldtimestamp = 0
    #     oldx = None
    #     oldy = None
    #     addedremain = False
    #     for fixation in scan:
    #         if first:
    #             first = False
    #             oldx = fixation[0]
    #             oldy = fixation[1]
    #             oldtimestamp = fixation[2]
    #         else:
    #             duration = fixation[2] - oldtimestamp
    #             scanpath.append([oldx, oldy, oldtimestamp])
    #             if forarch!=0 or (forarch==0 and remain!=0 and not addedremain):
    #                 newx = []
    #                 newy = []
    #                 newt = []
    #                 if remain!=0:
    #                     xspan = abs(fixation[0] - oldx) / (forarch+2)
    #                     dtime = duration / (forarch+2) #ragionamento dei punti intermedi, se non aggiungo uno si sovrappone all'ultimo tempo
    #                     for i in range(forarch+1):
    #                         if oldx<fixation[0]:
    #                             newx.append(oldx + ((i+1)*xspan))
    #                             newt.append(oldtimestamp + (i+1)*dtime)
    #                         else:
    #                             newx.append(oldx - ((i + 1) * xspan))
    #                             newt.append(oldtimestamp + (i + 1) * dtime)
    #                     remain -= 1
    #                 else:
    #                     xspan = abs(fixation[0]-oldx)/(forarch+1)
    #                     dtime = duration / (forarch+1)
    #                     for i in range(forarch):
    #                         if oldx < fixation[0]:
    #                             newx.append(oldx + ((i+1)*xspan))
    #                             newt.append(oldtimestamp + (i + 1) * dtime)
    #                         else:
    #                             newx.append(oldx - ((i + 1) * xspan))
    #                             newt.append(oldtimestamp + (i + 1) * dtime)
    #                 if oldx<fixation[0]:
    #                     newy = np.interp(newx,[oldx, fixation[0]], [oldy, fixation[1]])
    #                 else:
    #                     newy = np.interp(newx, [fixation[0], oldx], [fixation[1], oldy])
    #                 for x,y,t in zip(newx, newy, newt):
    #                     scanpath.append([x,y,t])
    #             oldx = fixation[0]
    #             oldy = fixation[1]
    #             oldtimestamp = fixation[2]
    #     fixation = scan[-1]
    #     scanpath.append([fixation[0], fixation[1], 1.0])

    #     scanpaths_mod.append(scanpath)

    # scanpaths_mod = np.array(scanpaths_mod)
    # # scanpaths = np.array()
    # print(len(scanpaths), len(scanpaths[0]))
    # exit()
    # print(scanpaths_mod.shape,scanpaths.shape)
    # exit()

    print("...Applying Linear Interpolation to obtain fixation points of 32 coordinates...")
    images = np.array(images)
    scanpaths = np.array(scanpaths)
    # print(scanpaths[0].shape)
    # exit()
    scanpaths_mod = list()

    fraction = 1/32
    epsilon = 1e-8
    for scan in scanpaths:
        print(scan)
        # exit()
        print("primo")
        print(len(scan))
        scanpath = list()
        first = True
        length = len(scan) #-1 per ottenere numero archi
        toadd = 32-length #punti da aggiungere
        #forarch = math.floor(toadd/(length-1)) #punti da aggiungere per arco
        fornode = toadd // length
        remain = toadd-(fornode*length)
        for fixation in scan:
            scanpath.append([fixation[0], fixation[1], fixation[2], fixation[3]])
            for i in range(fornode):
                if fixation[2]!=1:
                    scanpath.append([fixation[0], fixation[1], fixation[2]+(i*epsilon), fixation[3]])
                else:
                    scanpath.append([fixation[0], fixation[1], fixation[2], fixation[3]])
            if remain>0:
                scanpath.append([fixation[0], fixation[1], fixation[2] + (fornode * epsilon), fixation[3]])
                remain -= 1
        scanpaths_mod.append(scanpath)

    scanpaths_mod = np.array(scanpaths_mod)

    # print("...Applied Linear Interpolation to obtain fixation points of 32 coordinates...")

    # return images, scanpaths_mod, imagesname ############### without duration
    return images, scanpaths_mod, imagesname

def sample_images(epoch, step, generator, discriminator, X_test, Y_test, reduced, filename):

    print('epoch: ', epoch, 'step: ', step)

    idx = np.random.randint(low=0, high=X_test.shape[0], size=1)
    img_val = X_test[idx]
    scanpath_val = Y_test[idx][0]

    #discriminator.evaluate(X_test, Y_test)

    gen_scanpath = generator.predict(img_val)[0]

    gen_scanpath[:,0] *= 224
    gen_scanpath[:,1] *= 224
    scanpath_val[:,0] *= 224
    scanpath_val[:,1] *= 224

    # print(gen_scanpath)

    # print("next")

    # print(scanpath_val)

    penalty = False
    first = True
    xold = 0
    yold = 0
    skipped = False
    img_val = img_val[0]
    for row in gen_scanpath:
        if not skipped:
            xint = int(np.round(row[0]))
            yint = int(np.round(row[1]))
            if first:
                first = False
                img_val[yint - 2:yint + 2, xint - 2:xint + 2] = [0, 255, 0]
                xold = xint
                yold = yint
            else:
                img_val[yint - 2:yint + 2, xint - 2:xint + 2] = [255, 255, 255]
                cv2.line(img_val, (xold, yold), (xint, yint), (0, 255, 255), 1)
                xold = xint
                yold = yint
        elif skipped and first:
            penalty = True
        else:
            skipped = True
    first = True
    xold = 0
    yold = 0
    for row in scanpath_val:
        xint = int(round(row[0]))
        yint = int(round(row[1]))
        if first:
            first = False
            img_val[yint - 2:yint + 2, xint - 2:xint + 2] = [0, 255, 0]
            xold = xint
            yold = yint
        else:
            if row[2]==1.e-12:
                img_val[yint - 2:yint + 2, xint - 2:xint + 2] = [0, 0, 255]
            else:
                img_val[yint - 4:yint + 4, xint - 4:xint + 4] = [0, 0, 0]
            cv2.line(img_val, (xold, yold), (xint, yint), (255, 255, 0), 1)
            xold = xint
            yold = yint

    cv2.imwrite(filename, img_val)
    if (epoch % 1 == 0 and epoch!=0) or step % 10 == 0:
        generator.save_weights('weights/ours_generator_weights_'+ str(epoch) +'_'+str(step)+'.h5')
        discriminator.save_weights('weights/ours_discriminator_weights_' + str(epoch) + '_'+str(step)+'.h5')
    return penalty


def train_test(images, scanpaths, imagesname):
    if os.path.isfile('data/shuffledindices.pkl'):
        with open('data/shuffledindices.pkl', 'rb') as f:
            indices = pickle.load(f)
    else:
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        with open('data/shuffledindices.pkl', 'wb') as f:
            pickle.dump(indices, f)
    images = images[indices]
    print(images.shape)
    scanpaths = scanpaths[indices]
    print(len(imagesname))
    imagesname = [imagesname[j] for j in indices]
    namelist = sorted(set(imagesname), key=imagesname.index)
    numberfortraining = math.ceil(len(namelist)*0.9)
    listtrain = namelist[:numberfortraining]
    listtest = namelist[numberfortraining:]

    indexestrain = list()
    indexestest = list()
    for i in range(len(imagesname)):
        if imagesname[i] in listtrain:
            indexestrain.append(i)
        else:
            indexestest.append(i)
    X_train = np.array([images[i] for i in indexestrain])
    Y_train = np.array([scanpaths[i] for i in indexestrain])
    X_test = np.array([images[i] for i in indexestest])
    Y_test = np.array([scanpaths[i] for i in indexestest])
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    return X_train, X_test, Y_train, Y_test