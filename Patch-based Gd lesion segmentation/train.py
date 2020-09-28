# coding=utf-8
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import warnings
import scipy.ndimage as ndi
import tensorflow.keras as keras


K.set_image_data_format('channels_last')  # format of the images, W*H*L*channel
smooth=1. # used in dice loss to avoid divide zero

#  voxel-wise cross-entropy
def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)   # ground truth
    y_pred_f = K.flatten(y_pred)   # prediction
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# define dice loss function
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

# conv, batch normalization and relu
def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

# crop the input to fit the size of input layer
def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw//2), int(cw//2) + 1
        else:
            cw1, cw2 = int(cw//2), int(cw//2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch//2), int(ch//2) + 1
        else:
            ch1, ch2 = int(ch//2), int(ch//2)

        return (ch1, ch2), (cw1, cw2)

#define U-Net architecture
def get_unet(img_shape = None): # e.g. input size 200*200
        inputs = Input(shape = img_shape)    # 200*200*2
        concat_axis = -1
        filters = 3  # filter size, could be further investigated

        # downsampling
        conv1 = conv_bn_relu(64, filters, inputs)
        conv1 = conv_bn_relu(64, filters, conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = conv_bn_relu(96, 3, pool1)
        conv2 = conv_bn_relu(96, 3, conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = conv_bn_relu(128, 3, pool2)
        conv3 = conv_bn_relu(128, 3, conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = conv_bn_relu(256, 3, pool3)
        conv4 = conv_bn_relu(256, 4, conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = conv_bn_relu(512, 3, pool4)
        conv5 = conv_bn_relu(512, 3, conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        conv6 = conv_bn_relu(1024, 3, pool5)
        conv6 = conv_bn_relu(1024, 3, conv6)

        # upsampling
        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = get_crop_shape(conv5, up_conv6)
        crop_conv5 = Cropping2D(cropping=(ch, cw))(conv5)
        up7 = concatenate([up_conv6, crop_conv5], axis=concat_axis)
        conv7 = conv_bn_relu(512, 3, up7)
        conv7 = conv_bn_relu(512, 3, conv7)
        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = get_crop_shape(conv4, up_conv7)
        crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
        up8 = concatenate([up_conv7, crop_conv4], axis=concat_axis)
        conv8 = conv_bn_relu(256, 3, up8)
        conv8 = conv_bn_relu(256, 3, conv8)
        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = get_crop_shape(conv3, up_conv8)
        crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
        up9 = concatenate([up_conv8, crop_conv3], axis=concat_axis)
        conv9 = conv_bn_relu(128, 3, up9)
        conv9 = conv_bn_relu(128, 3, conv9)
        up_conv9 = UpSampling2D(size=(2, 2))(conv9)
        ch, cw = get_crop_shape(conv2, up_conv9)
        crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
        up10 = concatenate([up_conv9, crop_conv2], axis=concat_axis)
        conv10 = conv_bn_relu(96, 3, up10)
        conv10 = conv_bn_relu(96, 3, conv10)
        up_conv10 = UpSampling2D(size=(2, 2))(conv10)
        ch, cw = get_crop_shape(conv1, up_conv10)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
        up11 = concatenate([up_conv10, crop_conv1], axis=concat_axis)
        conv11 = conv_bn_relu(64, 3, up11)
        conv11 = conv_bn_relu(64, 3, conv11)
        ch, cw = get_crop_shape(inputs, conv11)
        conv11 = ZeroPadding2D(padding=(ch, cw))(conv11)
        conv12 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv11)  #  kernel_initializer='he_normal'
        model = Model(inputs=inputs, outputs=conv12)

        return model

# train U-Net model
def train_UNet(images, masks, verbose=False):

    samples_num = images.shape[0]    # number of images
    row = images.shape[1]   # size of patch
    col = images.shape[2]   # size of patch

    epoch = 600     # number of epochs
    batch_size = 60 # number of batch size
    img_shape = (row, col, 5)   # 5 modalities

    # build multi-gpu model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_unet(img_shape)
        model.compile(optimizer=Adam(lr=(1e-4)), loss=dice_coef_loss)

    current_epoch = 1
    model_path = 'models/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    while current_epoch <= epoch:
        print('Epoch ', str(current_epoch), '/', str(epoch))
        image = images.copy()
        mask = masks.copy()
        print(image.shape)
        history = model.fit(image, mask, batch_size=batch_size, epochs=1, verbose=verbose, shuffle=True)
        current_epoch += 1
        if current_epoch % 100 == 0:  # save model per 100 epochs
            modelsave = model_path + str(current_epoch) + '.h5'
            model.save_weights(modelsave)
            print('Model saved to ', modelsave)

# train model
def main():
    warnings.filterwarnings("ignore")    # never print warnings
    images = np.load('patch_train.npy')  # contains 5 modalities,training data
    masks = np.load('patch_mask.npy')    # ground truth label
    print(images.shape)
    print(masks.shape)
    train_UNet(images, masks, verbose=True)


if __name__ == '__main__':
    main()





