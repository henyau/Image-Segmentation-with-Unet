#from fastai.conv_learner import *
#from fastai.dataset import *
from sklearn.model_selection import train_test_split
#from fastai.models.resnet import vgg_resnet50
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
#torch.cuda.set_device(0)
import shutil
import numpy as np

import cv2

PATH = Path('./Train/')
list(PATH.iterdir())
#sx = 256
sz = 256

TRAIN_DN = 'CameraRGB'
MASKS_DN = 'CameraSeg'

def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax

#list((PATH/TRAIN_DN).iterdir())[:5]
list((PATH/MASKS_DN).iterdir())[:5]


(PATH/'train_masks-'f'{str(sz)}').mkdir(exist_ok=True)
(PATH/'train-'f'{str(sz)}').mkdir(exist_ok=True)

def resize_mask(fn):
    tmpimg = Image.open(fn).resize((sz,sz))
    oneChanAr = np.zeros((sz, sz));
    for i in range(0,sz):
        for j in range(0,sz):
            r, g, b = tmpimg.getpixel((i, j))
            if r == 6:
                #tmpimg.putpixel((i, j), (7,0,0))
                oneChanAr[j][i] = 7
            elif r!= 7 and r!= 10 and r!=0:
                #tmpimg.putpixel((i, j), (15,0,0))
                oneChanAr[j][i] = 15
            elif j>sz*0.82 and r ==10: # remove hood, use 100 for 128x128 images
                #tmpimg.putpixel((i, j), (0,0,0)) #can actually extend the road
                oneChanAr[j][i] = 0
            else:
                #tmpimg.putpixel((i, j), (r,0,0)) #can actually extend the road
                oneChanAr[j][i] = r
                

    oneChanIm = Image.fromarray(np.uint8(oneChanAr))
    oneChanIm.save((fn.parent.parent)/'train_masks-'f'{str(sz)}'/fn.name, mode='L')

def resize_img(fn):
    Image.open(fn).resize((sz,sz)).save((fn.parent.parent)/'train-'f'{str(sz)}'/fn.name)

#resize_mask(PATH/MASKS_DN/'200.png')
#ims = cv2.imread(str(PATH/TRAIN_DN/'200.png'))
#im_masks = cv2.imread(str(PATH/'train_masks-128/200.png'))
#print(im_masks.shape)
#ax = show_img(ims)
#ax = show_img(im_masks[...,0])    

         
		 
files = list((PATH/f'{MASKS_DN}').iterdir())
with ThreadPoolExecutor(8) as e: 
    e.map(resize_mask, files)

files = list((PATH/f'{TRAIN_DN}').iterdir())
with ThreadPoolExecutor(8) as e: 
    e.map(resize_img, files)
