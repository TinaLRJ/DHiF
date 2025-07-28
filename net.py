import math
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
import shutil
from loss import *
from model import *
# from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode, conv='usual', save_path=None):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            self.model = DNANet(mode=mode, conv=conv)
            if save_path is not None:
                shutil.copy('./model/DNANet/model_DNANet.py', save_path)
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            self.model = ISNet(mode=mode, conv=conv)
            if save_path is not None:
                shutil.copy('./model/ISNet/model_ISNet.py', save_path)
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            self.model = UIUNet(mode=mode, conv=conv)
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN(conv=conv)
        elif model_name == 'RPCANet':
            self.model = RPCANet(mode=mode, conv=conv)
            self.cal_loss = RPCANetLoss()
        elif model_name == 'MSH':
            self.model = MSHNet(1, mode=mode, conv=conv)
            if save_path is not None:
                shutil.copy('./model/MSHNet/MSHNet.py', save_path)
        elif model_name == 'DCANet':
            self.model = DCANet(mode=mode, conv=conv)
        elif model_name == 'ILNet':
            self.model = ILNet(1, mode=mode, conv=conv)
            if save_path is not None:
                shutil.copy('./model/ILNet/ilnet.py', save_path)
        elif model_name == 'MTU':
            self.model = MTUNet(1, 1, conv=conv)
        elif model_name == 'IRGraphormer':
            self.model = IRGraphormer(1, 1, conv=conv, imgsize=512, mode=mode)
            if save_path is not None:
                shutil.copy('./model/IRGraphormer/IRGraphormer.py', save_path)
                shutil.copy('./model/IRGraphormer/graph_vit.py', save_path)
        elif model_name == 'SCTransNet':
            config_vit = config.get_SCTrans_config()
            self.model = SCTransNet(config_vit, mode=mode, deepsuper=True, conv=conv)
        elif model_name == 'APT':
            self.model = APTNet(1, conv=conv)
            if save_path is not None:
                shutil.copy('./model/APTNet.py', save_path)
        
    def forward(self, img, epoch=-1):
        if self.model_name == 'MSH' or self.model_name == 'APT':
            if epoch > 5:
                return self.model(img, True)
            else:
                return self.model(img, False)
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss

    def loss_interf(self, pred, pred_interf, gt_mask):
        loss1 = self.cal_loss(pred, gt_mask)
        # loss2 = self.cal_loss(pred_interf, gt_mask)
        if isinstance(pred, tuple) or isinstance(pred, list):
            loss_interf = 0
            loss_interf_l1 = 0
            loss_interf_posneg = 0
            for i in range(len(pred)):
                # loss_interf += torch.clip((torch.sum(pred_interf[i]*gt_mask) - torch.sum(pred[i]*gt_mask))/torch.sum(gt_mask), 0, 1)
                # loss_interf += torch.sigmoid(10*torch.clip((torch.sum(pred_interf[i]*gt_mask) - torch.sum(pred[i]*gt_mask))/torch.sum(gt_mask), 0, 1))
                loss_interf += torch.clip((torch.sum(pred_interf[i]*gt_mask) - torch.sum(pred[i]*gt_mask))/torch.sum(gt_mask), 0, 1)**2
                # loss_interf_posneg += torch.sigmoid(10*(torch.sum(pred_interf[i]*gt_mask) - torch.sum(pred[i]*gt_mask))/torch.sum(gt_mask))
                # loss_interf_l1 += (torch.sum(pred_interf[i]*gt_mask) - torch.sum(pred[i]*gt_mask)) / torch.sum(gt_mask) + 1
            loss_interf = loss_interf / len(pred)
            # loss_interf_posneg = loss_interf_posneg / len(pred)
            # loss_interf_l1 = loss_interf_l1 / len(pred)
        else:
            # loss_interf = torch.clip((torch.sum(pred_interf*gt_mask) - torch.sum(pred*gt_mask))/torch.sum(gt_mask), 0)
            # loss_interf = torch.sigmoid(10*torch.clip((torch.sum(pred_interf*gt_mask) - torch.sum(pred*gt_mask))/torch.sum(gt_mask), 0, 1))
            loss_interf = torch.clip((torch.sum(pred_interf*gt_mask) - torch.sum(pred*gt_mask))/torch.sum(gt_mask), 0, 1)**2
            # loss_interf_posneg = torch.sigmoid(10*(torch.sum(pred_interf*gt_mask) - torch.sum(pred*gt_mask))/torch.sum(gt_mask))
            # loss_interf_l1 = (torch.sum(pred_interf*gt_mask) - torch.sum(pred*gt_mask))/torch.sum(gt_mask) + 1
        # loss_interf = math.pow(loss_interf, 0.15) + loss_interf_l1
        # loss_interf = math.pow(loss_interf, 0.15)  # v3
        # loss = loss1 + loss2 + loss_interf
        # loss = loss1 + loss_interf  # v2
        loss = loss1 + (1 - loss_interf)  # v2
        # loss = loss1 + loss_interf_posneg  # posneg
        return loss, loss1

    def loss_interf2(self, pred, pred_interf, gt_mask):
        loss1 = self.cal_loss(pred, gt_mask)
        # loss2 = self.cal_loss(pred_interf, gt_mask)
        if isinstance(pred, tuple):
            loss_interf = 0
            loss_interf_l1 = 0
            for i in range(len(pred)):
                loss_interf += 1 - abs((torch.sum(pred_interf[i]*gt_mask) - torch.sum(pred[i]*gt_mask))/torch.sum(gt_mask))
            loss_interf = loss_interf / len(pred)
        else:
            loss_interf = 1 - abs((torch.sum(pred_interf*gt_mask) - torch.sum(pred*gt_mask))/torch.sum(gt_mask))
        # loss_interf = math.pow(loss_interf, 0.15) + loss_interf_l1
        loss = loss1 + loss_interf
        return loss, loss1
