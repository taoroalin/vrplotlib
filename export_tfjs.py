import tensorflow as tf
import tensorflow.keras as keras
import tensorflowjs as tfjs
model = keras.applications.ResNet50(
    include_top=True, weights='imagenet', classes=1000,
    classifier_activation='softmax')

for layer in model.layers:
    print(layer.name)
    newname = layer.name.replace("_bn","")

tfjs.converters.save_keras_model(model, "./models/ResNet50")