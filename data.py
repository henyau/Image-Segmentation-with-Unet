from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import exposure
#from skimage.morphology import disk
#from skimage.filters import unsharp_mask
from skimage import filters
import numpy as np 
import os
import glob
from numpy import zeros, newaxis
#import cv2

class dataProcess(object):
	'''
	images and labels are stored into np arrays and save to disk, images are normalized and mean centered when loaded
	'''
    
	def __init__(self, out_rows, out_cols, data_path = "./Train/train-256", label_path = "./Train/train_masks-256", test_path = "./test", npy_path = "./npydata", img_type = "png"):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = './Train/train-'f'{str(out_rows)}'
		self.label_path = './Train/train_masks-'f'{str(out_rows)}'
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

	def create_train_data(self):
		i = 0
		num_img = 3000
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))

        
		imgdatas = np.ndarray((num_img,self.out_cols,self.out_rows,3), dtype=np.uint8)
		imglabels = np.ndarray((num_img,self.out_cols,self.out_rows,1), dtype=np.uint8)

        
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.data_path + "/" + midname,grayscale = False)
			label = load_img(self.label_path + "/" + midname,grayscale = False)
			print(midname)
            
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
			#imgdatas[i] = img
			tmp2 = label[...,0]
			tmp2 = tmp2[...,newaxis] #still need 128,128,1
			#imglabels[i] = tmp2
			

            #if there is a car to the left add it
			'''
			cars_b = False
			car_px_cnt = 0
			for ii in range(128):
				for jj in range(64, 110):
					if tmp2[ii,jj,0] == 10:
						car_px_cnt+=1
					if car_px_cnt >10:
						cars_b = True                        
						print(midname)
						break
				if cars_b == True:
					break;                    
			if cars_b:
				imglabels[i] = tmp2
				imgdatas[i] = img
				i += 1
			'''
                
			imglabels[i] = tmp2
			imgdatas[i] = img
			i += 1
                
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			#i += 1
			if i>=num_img:
				break
		print('loading done')
		np.save(self.npy_path + '/imgs_train_'f'{str(self.out_rows)}.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train_'f'{str(self.out_rows)}.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = img_to_array(img)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			imgdatas[i] = img 
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data_1chan(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean	
		#print('mean: '+ repr(mean))
		#imgs_mask_train = imgs_mask_train.astype('float32')
		#imgs_mask_train /= 255
		
		#imgs_mask_train[imgs_mask_train > 0] = 1 # do binary for now
		dim_mask = imgs_mask_train.shape
		
		imgs_mask_train2 = np.zeros((  dim_mask[0] , dim_mask[1], dim_mask[2]  , 1 )) # two classes in 3 channels?
		
		#for categorical
		for i in range(dim_mask[0]):
			for indc, c in enumerate([7,10,15]):                                  
				imgs_mask_train2[i, : , : , 0 ] += (imgs_mask_train[i, : , :, 0] == c ).astype(int)

		#for sparse
		
		
		#imgs_mask_train[imgs_mask_train <= 0.5] = 0
		#the mask should have two channels, road and vehicle, then just repaint after
		#return imgs_train,imgs_mask_train2
		
		#test sparse categorical for now?
		return imgs_train,imgs_mask_train2,mean
    
	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+'/imgs_train_'f'{str(self.out_rows)}.npy')
		imgs_mask_train = np.load(self.npy_path+'/imgs_mask_train_'f'{str(self.out_rows)}.npy')
		
		
		#imgs_train = exposure.equalize_hist(imgs_train) #global equalize
		#selem = disk(30)
		#imgs_train = rank.equalize(imgs_train, selem=selem) # local equalize       
		#imgs_train = exposure.adjust_gamma(imgs_train, gamma=2, gain=1)
        
		imgs_train = imgs_train.astype('float32')
        
		#imgs_train /= 127
		imgs_train /= 255
		#imgs_train -= 1
		#imgs_train = filters.unsharp_mask(imgs_train)
		mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean	 #test without subtracting mean
        
        
		#print('mean: '+ repr(mean))
		#imgs_mask_train = imgs_mask_train.astype('float32')
		#imgs_mask_train /= 255
		#imgs_mask_train[imgs_mask_train > 0] = 1 # do binary for now
		dim_mask = imgs_mask_train.shape
		
		imgs_mask_train2 = np.zeros((  dim_mask[0] , dim_mask[1], dim_mask[2]  , 3 )) # two classes vehicle or road
		
		#for categorical
		for i in range(dim_mask[0]):
			for indc, c in enumerate([7,10,15]): 
				imgs_mask_train2[i, : , : , indc ] = (imgs_mask_train[i, : , :, 0] == c ).astype(int)

		#for sparse
		#imgs_mask_train[imgs_mask_train <= 0.5] = 0
		#the mask should have two channels, road and vehicle, then just repaint after
		#return imgs_train,imgs_mask_train2
		
		#test sparse categorical for now?
		return imgs_train,imgs_mask_train2,mean
    

	def create_mean(self):
		print('-'*30)
		print('load train to compute mean images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+'/imgs_train_'f'{str(self.out_rows)}.npy')
		imgs_train = imgs_train.astype('float32')		
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)		
		np.save(self.npy_path + '/train_mean_'f'{str(self.out_rows)}.npy', mean)

		
	def get_mean(self):
		mean = np.load(self.npy_path+'/train_mean_'f'{str(self.out_rows)}.npy')
		return mean


if __name__ == "__main__":
	#traindata = dataProcess(448,448, data_path = "./Train/train-448", label_path = "./Train/train_masks-448"
	traindata = dataProcess(256,256, data_path = "./Train/train-256", label_path = "./Train/train_masks-256")
	#traindata = dataProcess(288,288, data_path = "./Train/train-288", label_path = "./Train/train_masks-288")    
	#traindata = dataProcess(512,512, data_path = "./Train/train-512", label_path = "./Train/train_masks-512")
	traindata.create_train_data()
	traindata.create_mean()

