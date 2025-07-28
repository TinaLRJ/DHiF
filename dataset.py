from utils import *
import matplotlib.pyplot as plt
import cv2
import os
from skimage import measure
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.dataset_name = dataset_name
        # with open(self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
        with open(self.dataset_dir+'/' + dataset_name + '/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        
    def __getitem__(self, idx):
        # img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        # mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        img = Image.open(self.dataset_dir + '/' + self.dataset_name + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/' + self.dataset_name + '/masks/' + self.train_list[idx] + '.png')
        # img = cv2.equalizeHist(np.array(img, dtype=np.uint8))
        # img = (img.astype(np.float32) - 127.5) / 73.5
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch
    def __len__(self):
        if self.dataset_name == 'NUDT-SIRST-Sea':
            return int(len(self.train_list)/3)
        return len(self.train_list)

class TrainSetLoader_aug(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader_aug).__init__()
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size
        self.dataset_name = dataset_name
        # with open(self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
        with open(self.dataset_dir+'/' + dataset_name + '/train.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()

    def __getitem__(self, idx):
        # img = Image.open(self.dataset_dir + '/images/' + self.train_list[idx] + '.png').convert('I')
        # mask = Image.open(self.dataset_dir + '/masks/' + self.train_list[idx] + '.png')
        img = Image.open(self.dataset_dir + '/' + self.dataset_name + '/images/' + self.train_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/' + self.dataset_name + '/masks/' + self.train_list[idx] + '.png')

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0

        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        img_patch_interf = self.std_patch(img_patch, mask_patch)
        return torch.cat([img_patch, img_patch_interf], 0), mask_patch
    def __len__(self):
        return len(self.train_list)

    def std_patch(self, img, mask, patch_size=16, interference_number=20):
        img_unfold = F.unfold(img.unsqueeze(0), kernel_size=patch_size, stride=patch_size)
        mask_unfold = F.unfold(mask.unsqueeze(0), kernel_size=patch_size, stride=patch_size).squeeze(0)
        img_unfold_ori = img_unfold.clone()
        mask_ann = mask_unfold.sum(0)
        img_unfold[0, :, mask_ann>0] = 0
        img_std = torch.std(img_unfold[0,:,:], 0)
        k = 50  # 5*interference_number  #int(len(img_std)*0.2)
        _, idx = torch.topk(img_std, k)
        chosen = torch.randperm(k)[:interference_number]
        img_unfold_ori[0, :, idx[chosen]] = img_unfold_ori[0, :, idx[chosen]].mean(0).unsqueeze(0)
        # img_unfold_ori[:, idx[chosen]] = 0
        img_fold = F.fold(img_unfold_ori, img.shape[-2:], kernel_size=patch_size, stride=patch_size)
        return img_fold.squeeze(0)

    # plt.figure()
    # plt.subplot(1,2,1); plt.imshow(np.array(img[0,:,:]), cmap='gray')
    # plt.subplot(1,2,2); plt.imshow(np.array(img_fold[0,:,:]), cmap='gray')
    # plt.show()



class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.testset_name = test_dataset_name
        # with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
        with open(self.dataset_dir+'/' + test_dataset_name + '/test.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        # img = Image.open(self.dataset_dir + '/images/' + self.test_list[idx] + '.png').convert('I')
        # mask = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')
        img = Image.open(self.dataset_dir + '/' + self.testset_name + '/images/' + self.test_list[idx] + '.png').convert('I')
        mask = Image.open(self.dataset_dir + '/' + self.testset_name + '/masks/' + self.test_list[idx] + '.png')
        # mask = Image.open(self.dataset_dir + '/' + self.testset_name + '/masks/' + self.test_list[idx].split('_')[0] + '.png')

        # img = cv2.equalizeHist(np.array(img, dtype=np.uint8))
        # img = (img.astype(np.float32) - 127.5) / 73.5
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        h, w = img.shape
        m, n = mask.shape
        if m!= h and n!= w:
            mask = np.resize(mask, img.shape)
            mask[mask>0] = 1.0
        # img = np.pad(img, ((0, ((h-1)//32+1)*32-h),(0, ((w-1)//32+1)*32-w)), mode='constant')
        img = np.pad(img, ((0, 512-h), (0, 512-w)), mode='constant')
        # img = np.pad(img, ((0, max(((h-1)//32+1)*32-h, 256-h)), (0, max(((w-1)//32+1)*32-w, 256-w))), mode='reflect')
        # img = np.pad(img, ((0, ((h-1)//32+1)*32-h), (0, ((w-1)//32+1)*32-w)), mode='reflect')
        # mask = np.pad(mask, ((0,((h-1)//32+1)*32-h),(0,((w-1)//32+1)*32-w)), mode='constant')
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open(self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png')
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
