import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
def jitter(d):
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)
    def inner(img):
        dx = np.random.choice(d*2+1)
        dy = np.random.choice(d*2+1)
        img = tf.roll(img,dx-d,1)
        img = tf.roll(img,dy-d,2)
        return img
    return inner

def unpad(w):
    def inner(image_t):
        l = image_t.shape[1]-w*2
        l2 = image_t.shape[2]-w*2
        return tf.slice(image_t, [0,w,w,0],[-1,l, l2, -1])
    return inner
  
def pad(w, mode="REFLECT", constant_value=0.5):
    def inner(image_t):
        return tf.pad(image_t, [[0,0],[w,w],[w,w],[0,0]],  mode=mode,constant_values=constant_value)

    return inner


def random_scale(scales):
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[1:-1]
        scale_shape = [_roundup(scale * d) for d in shp]
        resized = tf.image.resize(image_t,
            size=scale_shape)
        return tf.image.resize_with_crop_or_pad(resized, shp[0],shp[1])
        upsampled = tf.pad(resized, [[0,0],[pad_x,pad_x],[pad_y,pad_y], [0,0]])


    return inner


def random_rotate(angles, units="degrees"):
    def inner(image_t):
        return tfa.image.rotate(image_t, np.random.choice(angles) * np.pi / 180.0)

    return inner


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def _roundup(value):
    return np.ceil(value).astype(int)


def _rads2angle(angle, units):
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle