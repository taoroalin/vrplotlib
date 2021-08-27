import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
import tensorflowjs as tfjs
import numpy as np
from PIL import Image

def optneuron(model, layer, idx):
  new_model = tf.keras.Model(inputs=model.inputs, outputs=[layer.output])
  step_size = 0.1
  shp = (1,)+model.input.shape[1:]
  img = tf.random.uniform(shp)
  prev_loss = -99999999
  for i in range(100):
    with tf.GradientTape() as tape:
      tape.watch(img)
      activation = new_model(img)
      justfilter = tf.slice(activation, begin=[0,0,0,idx], size=[-1,-1, -1, 1])
      loss = -tf.math.reduce_sum(justfilter)
      # print(loss)
      gradient = tape.gradient(loss, img)
      gradient /= tf.math.reduce_std(gradient) + 1e-8 
      img += gradient*step_size
      img= tf.clip_by_value(img, -1,1)
      if loss-prev_loss<1: break
  return img
    
model = keras.applications.ResNet50(
    include_top=True, weights='imagenet', classes=1000,
    classifier_activation='softmax')

if not os.path.exists('./deepdream'):
  os.mkdir('./deepdream')
mpath = './deepdream/'+'resnet50'
if not os.path.exists(mpath):
  os.mkdir(mpath)
for layer in model.layers:
  if (layer.weights and layer.trainable and layer.name[-2:]!='bn'):
    if not os.path.exists(mpath+"/"+layer.name):
      os.mkdir(mpath+"/"+layer.name)
    for i in range(layer.output.shape[-1]):
      img = optneuron(model, layer, i)
      
      pilimg = Image.fromarray((tf.reshape(img, [224,224,3]).numpy()*255).astype(np.uint8))
      # pilimg.show()
      fname = mpath+"/"+layer.name+"/"+str(i)
      pilimg.save(fname+".png")
      print(fname)
      # with open(fname, 'wb') as f:
      #   f.write(img.numpy().tobytes())  