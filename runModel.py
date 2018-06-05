from keras.models import load_model
import sys, skvideo.io, json, base64
from skimage import exposure
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
from train import *
import cv2
import os
from concurrent.futures import ThreadPoolExecutor

#sz = 256
#sx = 320
#sy = 224

#sx = 448
#sy = 640
sx = 256
sy = 256

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

file = sys.argv[-1]

if file == 'runModel.py':
  print ("Error loading video")
  quit


# Define encoder function
def encode(array):
	pil_img = Image.fromarray(array)
	buff = BytesIO()
	pil_img.save(buff, format="PNG")
	return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

# Load model
#model = get_unet_128((sx,sy,3),3)
#model.load_weights("unet_Datgen.hdf5")
#model.load_weights("unet_Datgen_NoMean448.hdf5")
model = load_model('./models/unet_Datgen_256Frozen.hdf5')

#model.load_weights("unet_1chan.hdf5")


#imgs_train, imgs_mask_train, train_mean = mydata.load_train_data()
train_mean= np.load("./npydata/train_mean_256.npy") #mydata.get_mean()
#train_mea n = np.load("./npydata/train_mean.npy") #mydata.get_mean()

# Process video with trained Unet model


#process all the frames in the video, resize frame in array [x,128,128,3], then process at once
dim = video.shape[0]

test_array = np.zeros((dim, sx,sy,3))
#print(test_array.shape)
i = 0



for rgb_frame in video:
    #use PIL for test
    #ims_PIL = Image.fromarray(rgb_frame);
    #ims = ims_PIL.resize((sz, sz),Image.NEAREST)
    #ims = np.array(ims)
    ims = cv2.resize(rgb_frame,(sy,sx) ,interpolation = cv2.INTER_CUBIC)  
    #ims = exposure.adjust_gamma(ims, gamma=2, gain=1)
    #ims = exposure.adjust_gamma(ims) #global equalize
    ims = ims.astype('float32')
    
    #ims /=127.0
    #ims -= 1.0
    ims /= 255
    #ims -=train_mean
    
    test_array[i, ...] = ims
    i+=1

#datagen = ImageDataGenerator(zca_whitening=True)
#datagen.fit(test_array)
#for i in range(len(test_array)):
    #test_array[i] = datagen.standardize(test_array[i])
    
wholeBatch = True
if wholeBatch == True:
    res = model.predict(test_array)

    for ylab in res:
        
        for i in range(0,sx): #remove hood
            for j in range(0,sy):
                if j>(sy*0.82):
                    ylab[i,j,1] = 0
                    
        res_road = cv2.resize(ylab[...,0],(800,600) ,interpolation = cv2.INTER_CUBIC)
        res_vehicle = cv2.resize(ylab[...,1],(800,600) ,interpolation = cv2.INTER_LINEAR)

        res_road = (res_road>0.5).astype('uint8')
        res_vehicle = (res_vehicle>0.5).astype('uint8')

        answer_key[frame] = [encode(res_vehicle), encode(res_road)]

        frame+=1
else:
    for test_img in test_array:
        
        ylab = model.predict(test_img[newaxis,...])      
        for i in range(0,sx):
            for j in range(0,sy):
                if j>(sy*0.82): # remove hood
                    ylab[0,i,j,1] = 0

        res_road = cv2.resize(ylab[0,...,0],(800,600) ,interpolation = cv2.INTER_LINEAR)
        res_vehicle = cv2.resize(ylab[0,...,1],(800,600) ,interpolation = cv2.INTER_LINEAR)

        res_road = (res_road>0.5).astype('uint8')
        res_vehicle = (res_vehicle>0.5).astype('uint8')
        
        answer_key[frame] = [encode(res_vehicle[...]), encode(res_road[...])]

        frame+=1
        
# Print output in proper json format
print (json.dumps(answer_key))
