import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
import math


class Unfreeze(keras.callbacks.Callback):
  def __init__(self,it_to_unfreeze):
      self.it_to_unfreeze = it_to_unfreeze
      self.c=0
      self.frozen=True

  def on_batch_end(self, batch, logs=None):
      self.c+=1
      if not self.c > self.it_to_unfreeze and self.frozen:
          print('Iteration %d reached: UNFREEZING ENCODER' % self.c)
          self.frozen=False
          for layer in self.model.layers:
              layer.trainable=True

class InteractivePlot(keras.callbacks.Callback):
  def __init__(self):
      pass

  def on_train_begin(self, logs={}):
      self.losses = []
      self.logs = []
      self.batchnr = 0
      self.icount = 0

  def on_train_end(self, logs={}):
      pass

  def on_epoch_end(self, epoch, logs={}):
      self.batchnr = 0
      loss_train = logs.get('loss')

      self.losses.append(loss_train)

      self.icount+=1
      clear_output(wait=True)
      plt.figure(figsize=(14,10))
      train_vals = [self.losses]
      desc = ['loss']
      for i in range(len(train_vals)):
          #plt.subplot(2, 3, i+1)
          plt.plot(range(self.icount), train_vals[i], label=desc[i])
          plt.legend()
      #plt.savefig(self.logfile.replace('.txt', '.png'), bbox_inches='tight', format='png')
      plt.show()

  def on_batch_end(self, batch, logs=None):
      self.batchnr+=1
      if self.batchnr % 10 == 0:
          self.on_epoch_end(epoch=0, logs=logs)



def ckpt_callback(model_name, dataset, l_str, bs, extra_str='',
                period=1, save_weights_only=True,
                ckpt_folder_path = '../../predimportance_shared/models/ckpt/'):
    path = os.path.join(ckpt_folder_path, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, model_name+'_'+dataset+'_'+l_str+'_bs'+str(bs)+extra_str+'_ep{epoch:02d}_valloss{val_loss:.4f}.hdf5')
    cb_chk = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=1)
    return cb_chk


def step_decay(init_lr = 0.0001, drop = 0.1, epochs_drop = 3.0):
    def inner(epoch):
        lrate = init_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        if not (epoch+1)%epochs_drop:
            print('Reducing lr. New lr is:', lrate)
        return lrate
    return inner
