from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras import backend as K

img_rows, img_cols = 34, 56

# adjusts input format for Keras
def adjust_input(X):
    if K.image_dim_ordering() == 'th':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    X = X.astype('float32')
    X /= 255
    
    return (X, input_shape)

# adjusts input and output format for Keras
def adjust_input_output(X, y, num_classes=2):
    # convert class vectors to binary class matrices
    Y = np_utils.to_categorical(y, num_classes)
    (X, input_shape) = adjust_input(X)
    
    return (input_shape, X, Y)