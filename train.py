import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Lambda
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

def dice(y_pred, y_true):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def fbeta(y_pred, y_true):

    pred0 = Lambda(lambda x : x[:,:,:,0])(y_pred)
    pred1 = Lambda(lambda x : x[:,:,:,1])(y_pred)
    true0 = Lambda(lambda x : x[:,:,:,0])(y_true)
    true1 = Lambda(lambda x : x[:,:,:,1])(y_true) # channel last?
    
    y_pred_0 = K.flatten(pred0)
    y_true_0 = K.flatten(true0)
    
    y_pred_1 = K.flatten(pred1)
    y_true_1 = K.flatten(true1)
    
    intersection0 = K.sum(y_true_0 * y_pred_0)
    intersection1 = K.sum(y_true_1 * y_pred_1)

    precision0 = intersection0/(K.sum(y_pred_0)+K.epsilon())
    recall0 = intersection0/(K.sum(y_true_0)+K.epsilon())
    
    precision1 = intersection1/(K.sum(y_pred_1)+K.epsilon())
    recall1 = intersection1/(K.sum(y_true_1)+K.epsilon())
    
    fbeta0 = (1.0+0.25)*(precision0*recall0)/(0.25*precision0+recall0+K.epsilon())
    fbeta1 = (1.0+4.0)*(precision1*recall1)/(4.0*precision1+recall1+K.epsilon())
    
    return ((fbeta0+fbeta1)/2.0)

def fbeta_loss(y_true, y_pred):
    return 1-fbeta(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
def weighted_categorical_crossentropy(y_true, y_pred):    
    #weights = K.variable([0.5,2.0,0.0])
    weights = K.variable([0.5,4.0,0.0])
        
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    
    
    return loss

def cat_dice_loss(y_true, y_pred):
#    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    #return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)+fbeta_loss(y_true, y_pred)
    return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    #return weighted_categorical_crossentropy(y_true, y_pred) +fbeta_loss(y_true, y_pred)
    #return dice_loss(y_true, y_pred)
def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=2):
	'''
	This U-Net implementation is derived from: 	https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge
	'''
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    #classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)
    classify = Conv2D(3, (1, 1), activation='sigmoid')(up0) #using dataGen means 1,3,4 channels only

    model = Model(inputs=inputs, outputs=classify)

    #model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    #model.compile(optimizer = Adam(lr = 1e-5), loss = 'sparse_categorical_crossentropy', metrics = [dice])
    #model.compile(optimizer = Adam(lr = 1e-5), loss = 'categorical_crossentropy', metrics = [dice])
    
    for layer in model.layers:#freeze the conv layers? # only want to learn input weights (and output weights?)
        layer.trainable = False
    lr = 1e-4
    model.compile(optimizer = Adam(lr = lr, decay=1e-5), loss = cat_dice_loss, metrics = [dice, fbeta])
    #keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    return model
    
class TrainModel(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train, mean = mydata.load_train_data()
		#imgs_mask_train = to_categorical(imgs_mask_train)
		#imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train#, imgs_test

	def train(self):
		'''
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		#model = self.get_unet()
		model = get_unet_128((128,128,3),2)
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet_new.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=8, nb_epoch=8, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
        

		#print('predict test data')
		#imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		#np.save('../results/imgs_mask_test.npy', imgs_mask_test)
		'''
		'''data_gen_args1 = dict(featurewise_center=False,featurewise_std_normalization=False, rotation_range=25.,width_shift_range=0.3, 
                             height_shift_range=0.3,zoom_range=[0.5,1.5], horizontal_flip=True, fill_mode = 'constant', cval = 0.0, shear_range = 0.3, zca_whitening= True)'''

        
		#data_gen_args = dict(featurewise_center=False,featurewise_std_normalization=False, rotation_range=25.,width_shift_range=0.2, 
                            # height_shift_range=0.2,zoom_range=[0.6,1.4], horizontal_flip=True, fill_mode = 'constant', cval = 0.0, shear_range = 0.0)        
		#data_gen_args = dict(featurewise_center=False,featurewise_std_normalization=False, rotation_range=10.0,width_shift_range=0.1, 
                             #height_shift_range=0.1,zoom_range=[0.8,1.2], horizontal_flip=False, fill_mode = 'constant', cval = 0.0, shear_range = 0.0)
		data_gen_args = dict(featurewise_center=False,featurewise_std_normalization=False, rotation_range=0.,width_shift_range=0, 
                             height_shift_range=0.0,zoom_range=0.0, horizontal_flip=True, fill_mode = 'constant', cval = 0.0, shear_range = 0)
        
		image_datagen = ImageDataGenerator(**data_gen_args)#data generator works on both image and masks
		mask_datagen = ImageDataGenerator(**data_gen_args)
        
		X_train, X_test, y_train, y_test = train_test_split(imgs_train, imgs_mask_train, test_size=0.20, random_state=42)
        
		seed = 1 
		'''      
		image_datagen.fit(imgs_train, augment=True, seed=seed)
		mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)
        
		image_generator=image_datagen.flow(imgs_train,batch_size = 12,shuffle = True, seed = seed)
		mask_generator=mask_datagen.flow(imgs_mask_train,batch_size = 12,shuffle = True, seed = seed)
        
		#image_generator = image_datagen.flow_from_directory('Train/train-128', class_mode=None,seed=seed)
		#mask_generator = mask_datagen.flow_from_directory('Train/train_masks-128',class_mode=None,seed=seed)
        
		'''  
        
		image_datagen.fit(X_train, augment=True, seed=seed)
		mask_datagen.fit(y_train, augment=True, seed=seed)
		image_generator=image_datagen.flow(X_train,batch_size = 4,shuffle = True, seed = seed)
		mask_generator=mask_datagen.flow(y_train,batch_size = 4,shuffle = True, seed = seed)

		valid_image_generator=image_datagen.flow(X_test,batch_size = 4,shuffle = True, seed = seed)
		valid_mask_generator=mask_datagen.flow(y_test,batch_size = 4,shuffle = True, seed = seed)        
        
		train_generator = zip(image_generator, mask_generator)
		valid_generator = zip(valid_image_generator, valid_mask_generator)
		model = get_unet_128((self.img_rows,self.img_cols,3),2)
        
		model_checkpoint = ModelCheckpoint('unet_Datgen_256Frozen.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
		#model.load_weights("unet_1chan.hdf5")
		#model.load_weights("unet_Datgen_NewDat.hdf5")
		model.load_weights("unet_Datgen_256Full.hdf5")      
		#model.load_weights("unet_Datgen_fBeta.hdf5")   
		#model.fit_generator(train_generator, steps_per_epoch=200, epochs=40, validation_data = valid_generator, validation_steps=50, verbose = 1, callbacks=[model_checkpoint])
		model.fit(imgs_train, imgs_mask_train, batch_size=12, epochs=20, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])


if __name__ == '__main__':
	unet = TrainModel(256,256)
	#unet = TrainModel(288,288)
	#unet = TrainModel(512,512)
	#unet = TrainModel(448,640)
	imgs_train, imgs_mask_train = unet.load_data()
	unet.train()