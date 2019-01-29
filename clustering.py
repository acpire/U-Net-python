

import os
import cv2
import numpy as np
import numpy
import tensorflow as tf
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

#tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images = True)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=1, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

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

def sum(n):
    if n == 0:
        return 0
    else:
        return n + sum(n-1)

def Nest_Net(img_rows, img_cols, depth, deep_supervision, color_type=1, num_class=1):
    number_convolutions = 32;
    number_conc_convolutions = 32;
    step_convolution = number_convolutions
    index_convolution = 0
    index_pool = 0
    index_upSampling = 0
    index_concat = 0
    index_dropout = 0
    index_convolution_concat = 0
    dropout_rate = 0.0


    global data_axis
    if K.image_dim_ordering() == 'tf':
      data_axis = 3
      image_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      data_axis = 1
      image_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
      
    if depth > 0 :
        number_conv = depth+1
        number_conv = sum(number_conv)
        convolution_id = np.ndarray((number_conv * 2), dtype=Conv2D)
        concat_id = np.ndarray(sum(depth+1) , dtype= np.ndarray)
        drop_id = np.ndarray((sum(depth+1) *2), dtype=Dropout)
        pool_id = np.ndarray((depth+1), dtype=MaxPooling2D)
        depth = depth - 1
      
        convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(image_input)
        index_convolution += 1
        convolution_id[index_convolution]  = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(convolution_id[index_convolution - 1])
        pool_id[index_pool] = MaxPooling2D((2, 2), strides=(2, 2))(convolution_id[index_convolution])
        number_convolutions+=step_convolution
        index_convolution += 1
        index_pool += 1

        for i in range(depth):
            convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(pool_id[index_pool-1] )
            index_convolution += 1
            convolution_id[index_convolution]  = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(convolution_id[index_convolution - 1])
            pool_id[index_pool] = MaxPooling2D((2, 2), strides=(2, 2))(convolution_id[index_convolution])
            number_convolutions+=step_convolution
            index_pool += 1
            index_convolution += 1

        convolution_id[index_convolution] = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(pool_id[index_pool-1])
        index_convolution += 1
        convolution_id[index_convolution]  = Conv2D(number_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(convolution_id[index_convolution - 1])
        number_convolutions+=step_convolution
        index_convolution += 1
        stride = (depth + 2) * 2
        offset = 0
        for j in range(depth+1):
            for i in range(depth + 1 - j):
                number_conc_convolutions=step_convolution+step_convolution * i
                if j == 0:
                    conc_data = [UpSampling2D(size=(2, 2))(convolution_id[i*2+3])]
                else:
                    conc_data = [UpSampling2D(size=(2, 2))(drop_id[i*2+3+offset])]
                conc_data.append(convolution_id[i*2+1])
                _stride = (depth + 2) * 2
                _offset = 0
                for k in range(j):
                     conc_data.append(drop_id[i*2+1+_offset])
                     _stride -= 2
                     _offset += _stride
                concat_id[index_concat] = concatenate(conc_data, axis=data_axis)
                convolution_id[index_convolution] = Conv2D(number_conc_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(concat_id[index_concat])
                drop_id[index_dropout] = Dropout(dropout_rate)(convolution_id[index_convolution])
                index_convolution += 1
                convolution_id[index_convolution]  = Conv2D(number_conc_convolutions, (3, 3), activation="relu", kernel_initializer = 'he_normal', padding='same')(drop_id[index_dropout] )
                index_dropout += 1
                drop_id[index_dropout]  = Dropout(dropout_rate)(convolution_id[index_convolution])
                index_dropout += 1
                index_convolution += 1
                index_concat+=1
            if j > 0:
                offset += stride
            stride -= 2
            
        _stride = (depth + 2) * 2
        _offset = 0
        conc_data=[]
        for k in range(depth+1):
             conc_data.append(Conv2D(num_class, (1, 1), activation='sigmoid', padding='same')(drop_id[1+_offset]))
             _stride -= 2
             _offset += _stride
             
        if deep_supervision:
            x = Average()(conc_data)
            model = Model(input=image_input, output=x)
        else:
            model = Model(input=image_input, output=conc_data[depth])
            
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
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
         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         
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
     #     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = color.rgb2gray(image)
        image = cv2.imread(os.path.join(mask_data_path, image_name))
       #   img_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    directory_test = r'C:\Users\human\Source\Repos\acpire\U-Net-python\test'
    directory_train = r'C:\Users\human\Source\Repos\acpire\U-Net-python\train'
    directory_mask = r'C:\Users\human\Source\Repos\acpire\U-Net-python\segmentation'
    directory_result = r'C:\Users\human\Source\Repos\acpire\U-Net-python\result'
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train, imgs_mask_train = create_train_data(directory_train, directory_mask)
    imgs_test, imgs_id_test = load_test_data(directory_test)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train) 
    std = np.std(imgs_train)  
    max = np.max(imgs_train)

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_test = imgs_test.astype('float32')
    mean = np.mean(imgs_test) 
    std = np.std(imgs_test)  

    
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
    #model = Nest_Net(width, height, depth, True, color_type=1, num_class=1)
   # model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    print('-' * 30)
    print('Fit U-Net...')
    print('-' * 30)
    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=250, verbose=1, shuffle=True, validation_split=0.06, callbacks=[tensorboard])
  #  model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=40, verbose=1, shuffle=True, validation_split=0.10, callbacks=[tensorboard])
    #model.load_weights('weights.h5')
    print('-' * 30)
    print('Predict U-Net...')
    print('-' * 30)
   #  imgs_mask_test = model.predict(imgs_test, verbose=1)
    if not os.path.exists(directory_result):
        os.mkdir(directory_result)
    index =0
    for one_image in imgs_test:
        test = np.ndarray((int(1),1, width, height), dtype=np.float_)
        test[0] = one_image
        imgs_mask_test = model.predict(test, verbose=1)
        image = imgs_mask_test
        image_id = imgs_id_test[index]
       # for image, image_id in zip(imgs_mask_test, imgs_id_test[index]):
        image = (image[:, :, :] * 255.).astype(np.uint8)
        _image =cv2.cvtColor(numpy.array(image[0][0]), cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(directory_result, str(image_id) + '.png'), _image)
     #    image.save(os.path.join(directory_result, str(image_id) + '.png'))
        index+=1
