# Import stuff
import sys
import math
import keras
import tensorflow as tf

# Additional imports from keras
from keras           import optimizers
from keras.models    import Model
from keras.layers    import Input
from keras.layers    import Conv2D
from keras.layers    import MaxPooling2D
from keras.layers    import Flatten
from keras.layers    import Dense
from keras.layers    import Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler

# Custom imports
from dataset         import *
###############################################
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
############################################
### ************************************************
def entry_flow(inputs) :

    x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x

    for size in [128, 256, 728] :

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = tensorflow.keras.layers.Add()([x, residual])
        previous_block_activation = x

    return x
### ************************************************
def middle_flow(x, num_blocks=8) :

    previous_block_activation = x

    for _ in range(num_blocks) :

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = tensorflow.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x

    return x

### ************************************************
def exit_flow(x) :

    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x) 
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = tensorflow.keras.layers.Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='linear')(x)

    return x

### ************************************************
### Exception network
def Exception(train_im,
        train_sol,
        valid_im,
        valid_sol,
        test_im,
        h, w,
        channels,
        outputs,
        learning_rate,
        decay,
        batch_size,
        n_epochs):

    # Define VGG16 parameters
    nb_fltrs  = 32
    conv_knl  = 3
    pool_knl  = 2
    conv_str  = 1
    pool_str  = 2
    nb_fcn    = 64

    # Input
    c0 = Input(shape=(h, w, 1))

    
    x = exit_flow(middle_flow(entry_flow(c0)))
    # Print info about model
    model = Model(inputs=c0, outputs=x)
    model.summary()
    

    optim = tf.keras.optimizers.Adam(lr    = learning_rate, decay = decay)
    
    # Set training parameters
    model.compile(loss      = 'mean_squared_error',
                  optimizer = optim)
    early = EarlyStopping(monitor  = 'val_loss',
                          mode     = 'min',
                          verbose  = 0,
                          patience = 10)
    check = ModelCheckpoint('best.h5',
                            monitor           = 'val_loss',
                            mode              = 'min',
                            verbose           = 0,
                            save_best_only    = True,
                            save_weights_only = False)

    # Train network
    with tf.device('/gpu:0'):
        train_model = model.fit(train_im,
                                train_sol,
                                batch_size      = batch_size,
                                epochs          = n_epochs,
                                validation_data = (valid_im, valid_sol),
                                callbacks       = [early, check])

    return(model, train_model)
