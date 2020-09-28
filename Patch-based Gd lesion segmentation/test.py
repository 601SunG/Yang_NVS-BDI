import os
import time
import numpy as np
import warnings
import scipy.ndimage as ndi
import scipy
import SimpleITK as sitk
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages
K.set_image_data_format('channels_last')
rows_standard = 200
cols_standard = 200
smooth=1.
print('='*60)


def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef_for_training(y_true, y_pred)

def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs) #, kernel_initializer='he_normal'
    #bn = BatchNormalization()(conv)
    relu = Activation('relu')(conv)
    return relu

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
# process input images with smaller size
def less_preprocessing(image):
    channel_num = 1
    start_cut = 46
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    image = np.float32(image)
    image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    #Gaussion Normalization
    image -= np.mean(image)
    image /= np.std(image)
    image_suitable[...] = np.min(image)
    image_suitable[:, :,(cols_standard - image_cols_Dataset) // 2:(cols_standard + image_cols_Dataset) // 2] = image[:,start_cut:start_cut + rows_standard, :]
    image_suitable = image_suitable[:,:,:]
    image_suitable = image_suitable[..., np.newaxis]
    return image_suitable

# process input images with larger size
def over_preprocessing(image):
    channel_num = 1
    num_selected_slice = np.shape(image)[0]  # number of slices, 48
    image_rows_Dataset = np.shape(image)[1]  # row 240
    image_cols_Dataset = np.shape(image)[2]  # column 240
    image = np.float32(image)
    image = image[:,(image_rows_Dataset // 2 - rows_standard // 2):(image_rows_Dataset // 2 + rows_standard // 2),
                  (image_cols_Dataset // 2 - cols_standard // 2):(image_cols_Dataset // 2 + cols_standard // 2)]
    #Gaussion Normalization
    image -= np.mean(image)
    image /= np.std(image)
    image = image[5:55, :, :]
    image = image[..., np.newaxis]
    return image

# post-process the larger size output
def over_postprocessing(image, pred):
    start_slice = 6
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    original_pred = np.ndarray(np.shape(image), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:,(image_rows_Dataset-rows_standard)//2:(image_rows_Dataset+rows_standard)//2,(image_cols_Dataset-cols_standard)//2:(image_cols_Dataset+cols_standard)//2] = pred[:,:,:,0]
    original_pred[0:start_slice, :, :] = 0 # 48*200*200
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

# post-process the smaller size output
def less_postprocessing(image, pred):
    start_slice = 5
    start_cut = 55
    num_selected_slice = np.shape(image)[0]
    image_rows_Dataset = np.shape(image)[1]
    image_cols_Dataset = np.shape(image)[2]
    original_pred = np.ndarray(np.shape(image), dtype=np.float32)
    original_pred[...] = 0
    original_pred[:, start_cut:start_cut+rows_standard,:] = pred[:,:, (rows_standard-image_cols_Dataset)//2:(rows_standard+image_cols_Dataset)//2,0]
    original_pred[0:start_slice, :, :] = 0
    original_pred[(num_selected_slice-start_slice-1):(num_selected_slice-1), :, :] = 0
    return original_pred

# test stage
def test(numbers=0, verbose=True):
    # all path of folders with test data
    dirt1 = 'T1test'
    dirst1 = os.listdir(dirt1)
    dirst1.sort()
    dirt2 = 'T2test'
    dirst2 = os.listdir(dirt2)
    dirst2.sort()
    dirgd = 'Gdtest'
    dirsgd = os.listdir(dirgd)
    dirsgd.sort()
    dirfl = 'FLtest'
    dirsfl = os.listdir(dirfl)
    dirsfl.sort()
    dirpd = 'PDtest'
    dirspd = os.listdir(dirpd)
    dirspd.sort()

    nums = int(len(dirst1))
    dirmask ='masktest'
    dirmasks= os.listdir(dirmask)
    dirmasks.sort()
    T1_image = sitk.ReadImage(dirt1+'/'+dirst1[numbers])        # load T1
    T2_image = sitk.ReadImage(dirt2+'/'+dirst2[numbers])        # load T2
    Gd_image = sitk.ReadImage(dirgd+'/'+dirsgd[numbers])        # load Gd
    FL_image = sitk.ReadImage(dirfl+'/'+dirsfl[numbers])        # load FLAIR
    PD_image = sitk.ReadImage(dirpd+'/'+dirspd[numbers])        # load PD
    T1_array = sitk.GetArrayFromImage(T1_image)     # load array of T1 image
    T2_array = sitk.GetArrayFromImage(T2_image)     # load array of T2 image
    Gd_array = sitk.GetArrayFromImage(Gd_image)     # load array of Gd image
    FL_array = sitk.GetArrayFromImage(FL_image)     # load array of FLAIR image
    PD_array = sitk.GetArrayFromImage(PD_image)     # load array of PD image

# process images
    if T1_array.shape[2] > 200:
        T1_train = over_preprocessing(T1_array)
    else:
        T1_train = less_preprocessing(T1_array)

    if T2_array.shape[2] > 200:
        T2_train = over_preprocessing(T2_array)
    else:
        T2_train = less_preprocessing(T2_array)

    if Gd_array.shape[2] > 200:
        Gd_train = over_preprocessing(Gd_array)
    else:
        Gd_train = less_preprocessing(Gd_array)

    if FL_array.shape[2] > 200:
        FL_train = over_preprocessing(FL_array)
    else:
        FL_train = less_preprocessing(FL_array)

    if PD_array.shape[2] > 200:
        PD_train = over_preprocessing(PD_array)
    else:
       PD_train = less_preprocessing(PD_array)

# extract images
    if T1_train.shape[0] > 50:
        T1_train = T1_train[5:55, :, :]
    if T2_train.shape[0] > 50:
        T2_train = T2_train[5:55, :, :]
    if Gd_train.shape[0] > 50:
        Gd_train = Gd_train[5:55, :, :]
    if FL_train.shape[0] > 50:
        FL_train = FL_train[5:55, :, :]
    if PD_train.shape[0] > 50:
        PD_train = PD_train[5:55, :, :]

    imgs_two_channels = np.concatenate((T1_train, T2_train, Gd_train, FL_train, PD_train),axis=3)  # concatenate
    imgs_test = imgs_two_channels
    # input size 100*100
    img_shape = (100, 100, 5)
    model = get_unet(img_shape)
    model_path = 'models/'
    result_path = 'results/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    model.load_weights(model_path + '600.h5')  # load trained model

    # split input images into 4 patches, lt: left-top, rt: right-top,lb: left-bottom, rb: right-bottomo
    imgs_testlt = imgs_test[:, 0:100, 0:100]
    imgs_testrt = imgs_test[:, 100:200, 0:100]
    imgs_testlb = imgs_test[:, 0:100, 100:200]
    imgs_testrb = imgs_test[:, 100:200, 100:200]
    # predict patches separately
    predlt = model.predict(imgs_testlt, batch_size=1, verbose=verbose)  # image test 48*200*200*2
    predrt = model.predict(imgs_testrt, batch_size=1, verbose=verbose)  # image test 48*200*200*2
    predlb = model.predict(imgs_testlb, batch_size=1, verbose=verbose)  # image test 48*200*200*2
    predrb = model.predict(imgs_testrb, batch_size=1, verbose=verbose)  # image test 48*200*200*2
    # build empty array to store predictions
    pred = np.full([50, 200, 200, 1], None)
    # combine the patches to reconstruct same size as input
    pred[:, 0:100, 0:100, :] = predlt
    pred[:, 100:200, 0:100, :] = predrt
    pred[:, 0:100, 100:200, :] = predlb
    pred[:, 100:200, 100:200, :] = predrb
    # threshold
    pred[pred >= 1] = 1.
    pred[pred < 1] = 0.

    # process prediction
    Gd = Gd_array[5:55, :, :]
    if Gd.shape[2] > 200:
        original_pred = over_postprocessing(Gd, pred)
    else:
        original_pred = less_postprocessing(Gd, pred)
    # save prediction
    filename_resultImage = result_path + dirst2[numbers]
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage)
    # load the name of ground truth
    filename_testImage = os.path.join(dirmask + '/' + dirmasks[numbers])  # load results and GT
    # load ground truth and saved prediction
    testImage, resultImage = getImages(filename_testImage, filename_resultImage)
    # evaluate
    dsc = getDSC(testImage, resultImage)
    avd = getAVD(testImage, resultImage)
    h95 = getHausdorff(testImage, resultImage)
    recall, f1 = getLesionDetection(testImage, resultImage)
    return dsc, h95, avd, recall, f1

def main():
    dir = 'T2test'
    dirs = os.listdir(dir)
    nums = int(len(dirs))
    result = np.ndarray((nums,5), dtype = 'float32')   # generate empty array to store results
    for numbers in range(nums):
        dsc, h95, avd, recall, f1 = test(numbers, first5=True, verbose=True)#
        print('Result of patient ' + str(numbers))
        print('Dice',                dsc,	'(higher is better, max=1)')
        print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,	'(higher is better, max=1)')
        print('Lesion F1',            f1,	'(higher is better, max=1)')
        result[numbers, 0] = dsc
        result[numbers, 1] = h95
        result[numbers, 2] = avd
        result[numbers, 3] = recall
        result[numbers, 4] = f1

    np.save('results.npy', result)
    #  average values
    dsc_avg = np.mean(result[:, 0])
    h95_avg = np.mean(result[:, 1])
    avd_avg = np.mean(result[:, 2])
    recall_avg = np.mean(result[:, 3])
    f1_avg = np.mean(result[:, 4])
    print('=' * 60)
    print('=' * 60)
    print('=' * 60)
    print('Dice', dsc_avg, '(higher is better, max=1)')
    print('HD', h95_avg, 'mm', '(lower is better, min=0)')
    print('AVD', avd_avg, '%', '(lower is better, min=0)')
    print('Lesion detection', recall_avg, '(higher is better, max=1)')
    print('Lesion F1', f1_avg, '(higher is better, max=1)')


if __name__ == '__main__':
    main()

