import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
from tqdm import tqdm
from attribution.core import IR_Integrated_gradient, IR_Integrated_gradient_CE, MeanLinearPath, ZeroLinearPath

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['DNANet'], type=list,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', "
                         "'RISTDnet', 'SCTransNet', 'MTU', 'MSH', 'RPCANet', 'APT', 'ILNet', 'MAF', 'DCANet ")
parser.add_argument("--conv", default='usual', type=str, help="convolution types: Dual, usual")
parser.add_argument("--pth_dirs", default=None, type=list, help="log dir, default=None")
parser.add_argument("--dataset_dir", default='./dataset', type=str, help="train_dataset_dir")
parser.add_argument("--train_dataset_name", default=None, type=str, help="train_dataset_name")
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_txt", default=False, type=bool, help="save txt of results or not")
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")

parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--ce_patchsize", type=int, default=16)

global opt
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
mark = ''
# mark = '_2DualConvFC-Tanh-light+'
# mark = '_DualConvFC-Tanh-light3-ReLuConvSig'
# mark = ''
path_interpolation_func = ZeroLinearPath(fold=50)
epoch = 400

def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test', conv=opt.conv).cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()

    opt.model_name = opt.model_name + mark
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, img_dir) in enumerate(tqdm(test_loader)):
        img = Variable(img).cuda()
        if opt.ce_patchsize is None:
            pred = IR_Integrated_gradient(img, gt_mask, (img_dir, opt.model_name, opt.test_dataset_name, epoch), net,
                                          path_interpolation_func)
        else:
            pred = IR_Integrated_gradient_CE(img, gt_mask, opt.ce_patchsize, (img_dir, opt.model_name, opt.test_dataset_name, epoch), net,
                                          path_interpolation_func)
        # pred = net.forward(img)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
        
        ### save img
        if opt.save_img == True:
            img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
            if opt.dataset_name != opt.train_dataset_name:
                save_dir = opt.save_img_dir + opt.train_dataset_name + '__' + opt.dataset_name + '/' + opt.model_name
            else:
                save_dir = opt.save_img_dir + opt.dataset_name + '/' + opt.model_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img_save.save(save_dir + '/' + img_dir[0] + '.png')
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    if opt.save_txt:
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    if opt.save_txt:
        opt.f = open('./test_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                if opt.train_dataset_name == None:
                    opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(opt.model_name)
                print(dataset_name)
                if opt.save_txt:
                    opt.f.write(opt.model_name + '\n')
                    opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.train_dataset_name + '/' + opt.model_name + mark + '/%d.pth.tar' % epoch
                test()
            print('\n')
            if opt.save_txt:
                opt.f.write('\n')
        if opt.save_txt:
            opt.f.close()
    else:
        for dataset_name in opt.dataset_names:
            opt.test_dataset_name = dataset_name
            for pth_dir in opt.pth_dirs:
                for model_name in opt.model_names:
                    if model_name in pth_dir:
                        opt.model_name = model_name
                train_dataset_name = pth_dir.split('/')[0]
                print(opt.model_name)
                print(opt.test_dataset_name)
                if opt.save_txt:
                    opt.f.write(opt.model_name + '\n')
                    opt.f.write(opt.test_dataset_name + '\n')
                opt.pth_dir = pth_dir
                test()
                print('\n')
                if opt.save_txt:
                    opt.f.write('\n')
        if opt.save_txt:
            opt.f.close()
        
