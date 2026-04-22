import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import h5py
from imsize import *


def data_augmentation(label, mode=0):
    if mode == 0:
        return label
    elif mode == 1:
        return np.flipud(label)
    elif mode == 2:
        return np.rot90(label)
    elif mode == 3:
        return np.flipud(np.rot90(label))
    elif mode == 4:
        return np.rot90(label, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        return np.rot90(label, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(label, k=3))

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, n_scale=2, augment=None, use_3D=False):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_mat_file(x)]
        self.augment = augment
        self.use_3Dconv = use_3D
        self.n_scale = n_scale
        
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
            
        load_dir = self.image_files[file_index]
        
        try:
            data = sio.loadmat(load_dir)
            gt = np.array(data['gt'], dtype=np.float32)
        except NotImplementedError:
            with h5py.File(load_dir, 'r') as f:
                gt = np.array(f['gt'][...], dtype=np.float32)

        gt = gt.transpose(2, 1, 0)

        height, width, channels = gt.shape
        

        ms_h, ms_w = height // self.n_scale, width // self.n_scale
        
        ms = imresize(gt, output_shape=(ms_h, ms_w))
        
        lms = imresize(ms, output_shape=(height, width))
        
        ms = data_augmentation(ms, mode=aug_num)
        lms = data_augmentation(lms, mode=aug_num)
        gt = data_augmentation(gt, mode=aug_num)


        ms = ms.astype(np.float32)
        lms = lms.astype(np.float32)
        gt = gt.astype(np.float32)

        if self.use_3Dconv:
            ms = ms[np.newaxis, :, :, :]
            lms = lms[np.newaxis, :, :, :]
            gt = gt[np.newaxis, :, :, :]
            
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            
        ms = torch.clamp(ms,0,1)
        lms = torch.clamp(lms,0,1)
        gt = torch.clamp(gt,0,1)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files) * self.factor