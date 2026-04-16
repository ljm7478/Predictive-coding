'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''
##conda activate prednet
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

# original-> replace 'frame' to 'test' and 'seg3' to 'frame'

desired_im_sz = (128, 160) #(720,480) 
# desired_im_sz = (227, 227) #(720,480) 
# desired_im_sz = (226, 226) #(720,480) 

# categories = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7'] #['city', 'residential', 'road'] 
# categories = ['Frequency/Prediction_Error/frames', 'SC/Prediction_Error/frames', 'Orientation/Prediction_Error/frames', 'Phase/Prediction_Error/frames'] #['1fps_clockwise_test'] #'ball_clockwise', 'ball_random', 'ball_counter_clockwise'] 
# categories = ['Frequency/Prediction/frames', 'SC/Prediction/frames', 'Orientation/Prediction/frames', 'Phase/Prediction/frames']
categories = ['12_Cam-CAN_movie'] #

def process_data():
    for cat in categories:
        print('category is ' + cat)
        test_recordings = [(cat,'frames')] #[('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

        splits = {s: [] for s in [cat]}
        splits[cat] = test_recordings
        not_train = splits[cat] 
            
        for split in splits:
            im_list = []
            source_list = []  # corresponds to recording that image came from
            for category, folder in splits[split]:
                im_dir = os.path.join(CC_DATA_DIR, category,'frames/') # ,  'raw/', category, folder, folder[:10], folder, 'image_03/data/')
                print('im_dir: ' + im_dir)
                files = list(os.walk(im_dir, topdown=False))[-1][-1]
                im_list += [im_dir + f for f in sorted(files)]
                source_list += [category  + '-' + folder] * len(files)
            
            print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
            X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
            print('begin iterating for # of frames')
            for i, im_file in enumerate(im_list):
                print('im_file is {}'.format(im_file))
                im = imread(im_file)
                # im = im[:,:,:3]
                print(im.shape)
                X[i] = process_im(im, desired_im_sz)
                # X[i] = im
                print(X[i].shape)
                print('done creating frame # ' + str(i))
        
            print('begin saving hkl file')
            
            hkl.dump(X, os.path.join(CC_HICKLE_DUMP,'X_' + split +  '.hkl'))
            hkl.dump(source_list, os.path.join(CC_HICKLE_DUMP, 'sources_' + split +  '.hkl'))


#original def process_im
def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    # im = np.expand_dims(im, -1)
    return im

if __name__ == '__main__':
    process_data()
    
        
    #jlee: only when making RGB =1
    # resize and crop image
    # def process_im(im, desired_sz):
    #     if im.shape[2] == 4:
    #         # Convert to grayscale
    #         im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    #     else:
    #         # Convert to grayscale without considering alpha channel
    #         im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     target_ds = float(desired_sz[0])/im.shape[0]
    #     im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    #     d = int((im.shape[1] - desired_sz[1]) / 2)
    #     im = im[:, d:d+desired_sz[1]]
    #     im = np.expand_dims(im, -1)
    # return im
    # resize and crop image


