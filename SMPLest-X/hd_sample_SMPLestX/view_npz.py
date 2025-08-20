import numpy as np
import glob
import random
import cv2
import os

import pdb

files = glob.glob(r'*.npz')

for file in files:
    param = dict(np.load(file, allow_pickle=True))

    print(param.keys())
    for key in param.keys():
        if key.startswith('__'):
            print(key, param[key])
        elif key.startswith('keypoints'):
            print(key, param[key].shape)
        elif 'bbox' in key:
            print(key, param[key].shape)
        elif key.startswith('smpl'):
            for smpl_key in param[key].item().keys():
                print(key, smpl_key, param[key].item()[smpl_key].shape)
        elif key.startswith('meta'):
            for meta_key in param[key].item().keys():
                print(key, meta_key, f'len={len(param[key].item()[meta_key])}')
        elif key.startswith('misc'):
            for misc_key in param[key].item().keys():
                print(key, misc_key, param[key].item()[misc_key])
        else:
            try:
                print(key, param[key].shape, param[key][:5])
            except IndexError:
                print(key, param[key])

        print('-------------------')




