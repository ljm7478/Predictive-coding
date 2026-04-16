'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import requests
from bs4 import BeautifulSoup
#import urllib.request
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl
from step00_kitti_settings import *
import matplotlib.pyplot as plt
import cv2


desired_im_sz = (128, 160) #(720,480) 
categories = ['train','val'] #['city', 'residential', 'road']

val_recordings = [('val','frames')] #('city', '2011_09_26_drive_0005_sync')]
train_recordings = [('train', 'frames')] #[('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

def process_data():
    splits = {s: [] for s in ['train', 'val']}
    splits['val'] = val_recordings
    splits['train'] = train_recordings
    
    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        print('Processing split: {}'.format(split))
        
        for category, folder in splits[split]:
            im_dir = os.path.join(Concat_DATA_DIR, category,'frames/') # ,  'raw/', category, folder, folder[:10], folder, 'image_03/data/')
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category  + '-' + folder] * len(files)

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),)+ desired_im_sz + (3,), np.uint8)   # + desired_im_sz

        for i, im_file in enumerate(im_list):
            print('im_file is {}'.format(im_file))
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(Concat_HICKLE_DUMP, 'X_' + split +  '.hkl'))
        hkl.dump(source_list, os.path.join(Concat_HICKLE_DUMP, 'sources_'+ split + '.hkl'))


# resize and crop image
def process_im(im, desired_sz):
    # #jlee: only when making RGB =1
    # if im.shape[2] == 4:
    #     # Convert to grayscale
    #     im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    # else:
    #     # Convert to grayscale without considering alpha channel
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    
    # #jlee: if RGB=1
    # im = np.expand_dims(im, -1)

    return im


if __name__ == '__main__': 
    process_data()
