

import numpy as np
import os 
import cv2
from tensorflow import ConfigProto
from tensorflow import Session
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


def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
    number_convolutions = 32;
    step_convolutions = number_convolutions
    index_convolution = 0
    index_pool = 0
    index_upSampling = 0
    index_concat = 0

    nb_filter = [32,64,128,256,512]

    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
      
    if depth > 0 :
        convolution_id = np.ndarray((3 + depth * 4), dtype=Conv2D)
        concat_id = np.ndarray((depth), dtype= np.ndarray)
        pool_id = np.ndarray((depth), dtype=MaxPooling2D)
        upSampling_id = np.ndarray((depth), dtype=UpSampling2D)
        depth = depth - 1
      
        x = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(dropout_rate)(x)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(x)
        number_convolutions+=step_convolution

        conv2_1 = standard_unit(pool1, stage='21', nb_filter=number_convolutions)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)
        number_convolutions+=step_convolution

        up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
        conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
        conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

        conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

        up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
        conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
        conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

        up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
        conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
        conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

        conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

        up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
        conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
        conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

        up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
        conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
        conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

        up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
        conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
        conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

        conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

        up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
        conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
        conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

        up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
        conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
        conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

        up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
        conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
        conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

        up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
        conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
        conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
        nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

        if deep_supervision:
            model = Model(input=img_input, output=[nestnet_output_1,
                                                   nestnet_output_2,
                                                   nestnet_output_3,
                                                   nestnet_output_4])
        else:
            model = Model(input=img_input, output=[nestnet_output_4])

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
    config = ConfigProto()
    config.intra_op_parallelism_threads = 44
    config.inter_op_parallelism_threads = 44
    Session(config=config)

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
    config = ConfigProto()
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4
    
    Session(config=config)

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
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    print('-' * 30)
    print('Fit U-Net...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=30, verbose=1, shuffle=True, validation_split=0.05, callbacks=[tensorboard])
    #model.load_weights('weights.h5')
    print('-' * 30)
    print('Predict U-Net...')
    print('-' * 30)
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
