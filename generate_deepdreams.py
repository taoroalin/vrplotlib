import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflowjs as tfjs
import numpy as np
from PIL import Image
import time
import sys
import cProfile, pstats

stime = time.time()

single = '--single' in sys.argv
fromback = '--back' in sys.argv
limitmag = '--limitmag' in sys.argv
one_batch = '--one-batch' in sys.argv

model = keras.applications.resnet.ResNet50(
    include_top=True, weights='imagenet', classes=1000,
    classifier_activation='softmax')
shp = (1,)+model.input.shape[1:]

lossdiffthreshold = 0.1
sval = 0.0 # only works if power of 2?
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = "/tfprofile",
      histogram_freq = 1,
      profile_batch = '1,1')
def optneuron(model, layer, idx, num, single):
  new_model = tf.keras.Model(inputs=model.inputs, outputs=[layer.output])
  step_size = 0.15
  momentum = 0.9
  momentum_vector = tf.fill((num,)+shp[1:], 0.0)
  img = tf.fill((num,)+shp[1:], sval)
  best_loss = -9999999999900.9
  best_loss_step = 0
  now_ldthresh = lossdiffthreshold
  for step in range(200):
    with tf.GradientTape() as tape:
      tape.watch(img)
      activation = new_model(img)
      lst = []
      for j in range(num):
        begin = [j,activation.shape[1]//2,activation.shape[2]//2,idx+j] if single else [j,0,0,idx+j]
        size = [1,1,1,1] if single else [1,-1,-1,1]
        lst.append(tf.slice(activation, begin=begin, size=size))
      loss = tf.math.reduce_sum(tf.math.add_n(lst))
      if limitmag:
        loss-=tf.math.reduce_mean(img*img)*300
      # print(loss)
      gradient = tape.gradient(loss, img)
      gradient /= tf.math.reduce_std(gradient) + 1e-8
      momentum_vector = momentum*momentum_vector+step_size*gradient
      img += momentum_vector
      img= tf.clip_by_value(img, -1,1)
      if loss>best_loss+now_ldthresh:
        best_loss = loss
        best_loss_step = step
      elif step-best_loss_step>=20:
        print("stopped early after", step, "steps")
        break
  return img

def cropsval(batch):
  left = 0
  for i in range(shp[1]):
    if batch[0,i,batch.shape[2]//2, 0]!=sval:
      left = i
      break
  right = 0
  for i in range(shp[1]-1, -1,-1):
    if batch[0,i,batch.shape[2]//2, 0]!=sval:
      right = i
      break
  top = 0
  for i in range(shp[1]):
    if batch[0,batch.shape[2]//2,i, 0]!=sval:
      top = i
      break
  bottom = 0
  for i in range(shp[1]-1, -1,-1):
    if batch[0,batch.shape[2]//2,i, 0]!=sval:
      bottom = i
      break
  cropped = tf.slice(batch, [0, top, left, 0], [-1, bottom-top, right-left, -1])
  return cropped
  
    
spath = "./deepdream"
if not os.path.exists(spath):
  os.mkdir(spath)
spath+= "/single" if single else "/filter"
if not os.path.exists(spath):
  os.mkdir(spath)
  
mpath =spath+ '/'+'ResNet50'
if not os.path.exists(mpath):
  os.mkdir(mpath)

max_batch = 96

def __main__():
  layers = reversed(list(model.layers)) if fromback else list(model.layers)
  # for l in layers:
  #   print(l.name)
  for layer in layers:
    oldcrit = (layer.weights and layer.trainable and layer.name[-2:]!='bn')
    if (layer.name[-3:]in ['add']) and len(layer.output.shape)==4:
      lpath = mpath+"/"+layer.name
      if not os.path.exists(lpath):
        os.mkdir(lpath)
      nfilters = layer.output.shape[-1]
      for start in range(0,nfilters, max_batch):
        num = min(nfilters, start+max_batch)-start
        img = optneuron(model, layer, start, num, single)
        cropped = cropsval(img)
        if cropped.shape[1]==0: 
          print("cropped size zero!")
          continue
        npy = (cropped*255).numpy().astype(np.uint8)
        print(npy.shape)
        for j in range(num):
          pilimg = Image.fromarray(npy[j])
          # pilimg.show()
          fname = lpath+"/"+str(j+start)+".png"
          try: # try catch so it doesn't fail if you had that image open in windows
            pilimg.save(fname)
          except:
            print("couldn't write", fname)
            pass
          # with open(fname, 'wb') as f:
        #   f.write(img.numpy().tobytes())  
        print(layer.name+"/"+str(start+num))
        if one_batch: break
# cProfile.run("__main__()")
if __name__=="__main__":
  __main__()
  print("took", time.time()-stime)