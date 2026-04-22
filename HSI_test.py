import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import h5py


try:
    from imsize import imresize
except ImportError:
    print("Warning: imsize.py not found. Please ensure it is in the directory.")

class HSTestData(data.Dataset):
    def __init__(self, image_dir, n_scale=2, use_3D=False):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.mat')]
        self.use_3Dconv = use_3D
        self.n_scale = n_scale

    def __getitem__(self, index):
        load_dir = self.image_files[index]
        
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
        return len(self.image_files)