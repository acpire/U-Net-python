

import numpy as np
import os 
import cv2
from PIL import Image
from skimage import color
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.callbacks import TensorBoard  
from keras.models import Sequential

K.set_image_data_format('channels_first')

tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images = True)





def plot(Xs, predictions):
    nrows = len(Xs)
    ncols = len(predictions)

    fig = plt.figure(figsize=(16, 8))
    fig.canvas.set_window_title('Clustering data from ' + URL)
    for row, (row_label, X_x, X_y) in enumerate(Xs):
        for col, (col_label, y_pred) in enumerate(predictions):
            ax = plt.subplot(nrows, ncols, row * ncols + col + 1)
            if row == 0:
                plt.title(col_label)
            if col == 0:
                plt.ylabel(row_label)
            plt.scatter(X_x, X_y, c=y_pred.astype(np.float), cmap='prism', alpha=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.tight_layout()
    plt.show()
    plt.close()


def dice_coef(y_true, y_pred):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sum = K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) 
    return (2. * intersection + smooth) / ( sum + smooth)
        
def dice_coef_loss(y_true, y_pred):
        return 1.-dice_coef(y_true, y_pred)

def jaccard_distance(y_true, y_pred):
    smooth=100
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac * smooth

def jaccard_distance_loss(y_true, y_pred):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    """
    smooth=100
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
#keras 1
def make_unet(img_channels, image_width , image_height, depth):

    
    if K.image_data_format() == 'channels_first':
        input_shape = (img_channels, image_width, image_height)
    else:
        input_shape = (image_width, image_height, img_channels)
    depth = depth - 1
    index_node = 0
    number_convolutions = 32;
    step_convolutions = number_convolutions
    model = Sequential()
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    index_node += 5
    number_convolutions = number_convolutions + step_convolutions
    for i in range(depth) :
        model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        index_node += 5
        number_convolutions = number_convolutions + step_convolutions
        
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    index_node += 4
    number_convolutions = number_convolutions - step_convolutions
    
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    index_node += 5
    if depth == 0:
        concate = [model.get_input_at(index_node - 1), model.get_input_at(3)]
        model.add( merge(concate, mode='concat'))
        index_node += 1
    else:
        concate = [model.get_layer(None,index_node - 1), model.get_layer(None,index_node - 6)]
        model.add( merge(concate, mode='concat'))
        index_node += 1
    
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    index_node += 4
    number_convolutions = number_convolutions - step_convolutions

    index_convolution = 21
    for i in range(depth) :
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(number_convolutions, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add( Merge([model.get_layer(None,index_node - 1), model.get_layer(None,index_node  - index_convolution)], mode='concat'))
        index_convolution += 10
        index_node += 5
        number_convolutions = number_convolutions - step_convolutions

    model.add(Conv2D(1, 1, 1, padding='same'))
    model.add(Activation('relu'))


  

  #  model = Model(input=[inputs], output=[conv4])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    
    return model

def make_unet_functions(img_channels, image_rows , image_cols, depth):
    number_convolutions = 32;
    step_convolutions = number_convolutions
    index_convolution = 0
    index_pool = 0
    index_upSampling = 0
    index_concat = 0
    if depth > 0 :
        convolution_id = np.ndarray((3 + depth * 4), dtype=Conv2D)
        concat_id = np.ndarray((depth), dtype= np.ndarray)
        pool_id = np.ndarray((depth), dtype=MaxPooling2D)
        upSampling_id = np.ndarray((depth), dtype=UpSampling2D)
        depth = depth - 1

        inputs = Input(( img_channels, image_rows,  image_cols )) 
      #  resize = ZeroPadding2D(padding=(int(modify_x), int(modify_y)))(inputs)
        convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(inputs)
        index_convolution += 1
        convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(convolution_id[index_convolution - 1])
        pool_id[index_pool] = MaxPooling2D(pool_size=(2, 2))(convolution_id[index_convolution] )
        index_convolution += 1
        index_pool += 1
        number_convolutions += step_convolutions

        for i in range(depth):
            convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(pool_id[index_pool-1])
            index_convolution += 1
            convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(convolution_id[index_convolution - 1])
            pool_id[index_pool] = MaxPooling2D(pool_size=(2, 2), border_mode='valid')(convolution_id[index_convolution] )
            index_convolution += 1
            index_pool += 1
            number_convolutions += step_convolutions

        convolution_id[index_convolution] =  Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')( pool_id[index_pool - 1])
        index_convolution += 1
        convolution_id[index_convolution] =  Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(convolution_id[index_convolution - 1] )
        index_convolution += 1
        number_convolutions -= step_convolutions
        
        upSampling_id[index_upSampling] = UpSampling2D(size=(2, 2))(convolution_id[index_convolution - 1])
        index_upSampling += 1
        concat_id[index_concat] = concatenate( [Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(  upSampling_id[index_upSampling-1] ), convolution_id[index_convolution - 3]  ], axis=1)
        index_concat += 1
    
        convolution_id[index_convolution] =  Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(concat_id[index_concat - 1])
        index_convolution += 1
        convolution_id[index_convolution] =  Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(convolution_id[index_convolution - 1])
        index_convolution += 1
        number_convolutions -= step_convolutions
    
        index = 0
        for i in range(depth):
            upSampling_id[index_upSampling] = UpSampling2D(size=(2, 2))(convolution_id[index_convolution - 1])
            index_upSampling += 1
            concat_id[index_concat]  = concatenate([Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(upSampling_id[index_upSampling - 1]), convolution_id[index_convolution - index - 7]], axis=1)
            index_concat += 1
            convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')(concat_id[index_concat - 1])
            index_convolution += 1
            convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation='relu', border_mode='same')( convolution_id[index_convolution - 1])
            index_convolution += 1
            number_convolutions -= step_convolutions
            index = index + 4

        convolution_id[index_convolution] = Conv2D(1, 1, 1, activation='sigmoid')(convolution_id[index_convolution - 1])

        model = Model(input=[inputs], output=[ convolution_id[index_convolution]])

        model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
        model.summary()
    
    return model


def make_unet_200(img_channels, image_rows , image_cols):
    inputs = Input(( img_channels, image_rows,  image_cols ))

    conv1 = Conv2D(25, (3, 3), activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(25, (3, 3), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(50, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(50, (3, 3), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(100, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(100, (3, 3), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(200, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(200, (3, 3), activation='relu', border_mode='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = [Conv2D(100, (3, 3), activation='relu', border_mode='same')(up5), conv3]
    up5 = concatenate(up5, axis=1)
    

    conv5 = Conv2D(100, (3, 3), activation='relu', border_mode='same')(up5)
    conv5 = Conv2D(100, (3, 3), activation='relu', border_mode='same')(conv5)

    
    up6 =UpSampling2D(size=(2, 2))(conv5)
    up6 =  [Conv2D(50, (3, 3), activation='relu', border_mode='same')(up6), conv2]
    up6 = concatenate(up6, axis=1)

    conv6 = Conv2D(50, (3, 3), activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(50, (3, 3), activation='relu', border_mode='same')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 =  [Conv2D(25, (3, 3), activation='relu', border_mode='same')(up7), conv1]
    up7 = concatenate(up7, axis=1)

    conv7 = Conv2D(25, (3, 3), activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(25, (3, 3), activation='relu', border_mode='same')(conv7)


    conv8 = Conv2D(1, 1, 1, activation='sigmoid')(conv7)

    model = Model(input=[inputs], output=[conv8])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()
    
    return model

def get_size(width, height, depth):
        image_w = width
        image_h = height
        modify_x = 0
        modify_y = 0
        index = 0
        for i in range(depth):
            if image_w % 2 != 0:
                image_w = width
                while image_w > 1 :
                    image_w /= 2
                    index += 1
                i = depth
                width = pow(2 ,index) 
                break
            image_w/=2
        index =0 
        for i in range(depth):
            if image_h % 2 != 0:
                image_h = height
                while image_h > 1 :
                    image_h /= 2
                    index += 1
                i = depth
                height = pow(2 ,index) 
                break
            image_h /= 2

        return width, height

def resize(images, img_rows, img_cols):
    imgs_p = np.ndarray((images.shape[0], images.shape[1], img_rows, img_cols), dtype=np.float)
    for i in range(images.shape[0]):
        imgs_p[i, 0] = cv2.resize(images[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def load_test_data(directory):
    image_rows = 200;
    image_cols = 200

    test_data_path = os.path.join(directory)
    dir_images = os.listdir(test_data_path)
    total = len(dir_images)
    images = np.ndarray((int(total),1, image_rows, image_cols), dtype=np.float_)
    images_id = np.ndarray((total, ), dtype=np.int32)
    
    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in dir_images:
        if i == int(total):
            return images, imgs_id
        image_mask_name = int(i)
        image = cv2.imread(os.path.join(test_data_path, image_name))
        image = cv2.resize(image, (image_rows, image_cols))
        image = color.rgb2gray(image)
        image = np.array([image])
        l_0 = np.size(image, 0)
        l_1 = np.size(image, 1)
        l_2 = np.size(image, 2)

        images[i] = image
        images_id[i] = np.array([image_mask_name])
        
        print('Load test : {0}/{1} images'.format(i, total))
        i += 1
    return images, images_id

def create_train_data(directory_train, directory_mask):
    image_rows = 200;
    image_cols = 200
    train_data_path = os.path.join(directory_train)
    mask_data_path = os.path.join(directory_mask)
    
    images = os.listdir(train_data_path)
    total = len(images)    

    imgs = np.ndarray((int(total),1, image_rows, image_cols), dtype=np.float_)
    imgs_mask = np.ndarray((int(total),1, image_rows, image_cols), dtype=np.float_)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if i == int(total):
            return imgs, imgs_mask
        image = cv2.imread(os.path.join(train_data_path, image_name))
        img = color.rgb2gray(image)
        image = cv2.imread(os.path.join(mask_data_path, image_name))
        img_mask = color.rgb2gray(image)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        print('Load train : {0}/{1} images'.format(i, total))
        i += 1
    return imgs, imgs_mask

    
    
if __name__ == '__main__':
    depth = 3
    width = 200
    height = 200
    directory_test = r'C:\Users\human\source\repos\ImageRecognition\ImageRecognition\test'
    directory_train = r'C:\Users\human\source\repos\ImageRecognition\ImageRecognition\train'
    directory_mask = r'C:\Users\human\source\repos\ImageRecognition\ImageRecognition\segmentation'
    directory_result = r'C:\Users\human\source\repos\ImageRecognition\ImageRecognition\result'
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = create_train_data(directory_train, directory_mask)
    imgs_test, imgs_id_test = load_test_data(directory_test)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train) 
    std = np.std(imgs_train)  
    max = np.max(imgs_train)
   # imgs_train -= mean
   # imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test) 
    std = np.std(imgs_test)  
 #   imgs_test -= mean
#    imgs_test /= std

    
    for image, image_id in zip(imgs_test, imgs_id_test):
        image = (image[:, :, :] * 255.).astype(np.uint8)
        image = Image.fromarray(image[0])
        if image.im != 'RGB':
            image = image.convert('RGB')
        image.save(os.path.join(directory_result, str(image_id) + '.png'))
    width, height = get_size(width, height, depth)
    imgs_train = resize(imgs_train, width, height)
    imgs_mask_train = resize(imgs_mask_train, width, height)
    imgs_test = resize(imgs_test, width, height)
    
    print('-' * 30)
    print('Make U-Net...')
    print('-' * 30)
    model = make_unet_functions(1 , width, height, depth)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    print('-' * 30)
    print('Fit U-Net...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=120, verbose=1, shuffle=True, validation_split=0.05, callbacks=[tensorboard])
    model.load_weights('weights.h5')
    print('-' * 30)
    print('Predict U-Net...')
    print('-' * 30)
    mode
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    if not os.path.exists(directory_result):
        os.mkdir(directory_result)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, :] * 255.).astype(np.uint8)
        image = Image.fromarray(image[0])
        if image.im != 'RGB':
            image = image.convert('RGB')
        image.save(os.path.join(directory_result, str(image_id) + '.png'))
