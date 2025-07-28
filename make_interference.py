import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2


if __name__ == '__main__':
    data_path = './dataset/NUDT-SIRST'
    save_path = './dataset/patches'
    img_dir = os.path.join(data_path, 'images')
    msk_dir = os.path.join(data_path, 'masks')
    os.makedirs(save_path, exist_ok=True)

    ce_patchsize = 16
    img_files = np.sort(glob.glob(os.path.join(img_dir, '*.png')))
    for img_f in img_files:
        img = Image.open(img_f)
        img = np.array(img).astype(float)
        if img.ndim == 3:
            img = img[:,:,0]
        msk = Image.open(img_f.replace('images', 'masks'))
        msk = np.array(msk).astype(float)
        if msk.ndim == 3:
            msk = msk[:,:,0]

        img_unfold = F.unfold(torch.from_numpy(img).unsqueeze(0), kernel_size=ce_patchsize, stride=ce_patchsize,
                              padding=0).reshape(ce_patchsize, ce_patchsize, -1)
        msk_unfold = F.unfold(torch.from_numpy(msk).unsqueeze(0), kernel_size=ce_patchsize, stride=ce_patchsize, padding=0).sum(0)
        patch_remain = img_unfold[:,:, msk_unfold==0]
        for i in range(patch_remain.size(2)):
            patch = np.array(patch_remain[:,:,i])
            name = os.path.join(save_path, img_f.split('/')[-1].replace('.png', '_%04d.png' % i))
            cv2.imwrite(name, patch)


