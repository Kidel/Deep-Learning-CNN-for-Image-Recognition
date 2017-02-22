from keras import backend as K
from keras.utils import np_utils 
import numpy as np

def preprocess_data(X, y, nb_classes, img_rows=28, img_cols=28, verbose=1):

        X = np.array(X)
        y = np.array(y)

        input_shape = None

        if K.image_dim_ordering() == 'th':
            X = X.reshape(X.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X = X.reshape(X.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        
        X = X.astype('float32')
        X /= 255
        
        if verbose == 1:
            print('X shape:', X.shape)
            print(X.shape[0], 'samples')
        
        # convert class vectors to binary class matrices
        y = np_utils.to_categorical(y, nb_classes)
        
        return (X,y,input_shape)