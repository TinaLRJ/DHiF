import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from torch.backends import cudnn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['MTU'], type=list,
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', "
                         "'RISTDnet', 'SCTransNet', 'MTU', 'MSH', 'RPCANet', 'APT', 'ILNet', 'MAF', 'DCANet', 'IRGraphormer' ")
parser.add_argument("--conv", default='Dyf', type=str, help="convolution types: usual, Dyf, WTC, SDC, PC, FD, Ref")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='./dataset', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--seed", type=int, default=1337, help="Random seed")
parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--interference", default=False, type=bool, help="Dataset Augment")
parser.add_argument("--save", default='./log_new', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [100, 150, 200, 250, 300, 350, 400, 450, 500], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")

global opt
opt = parser.parse_args()
cuda = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
mark = '_500epoch_' + opt.conv

print('cuda:', cuda)
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        # torch.use_deterministic_algorithms(False)


def train():
    interference = opt.interference
    if not interference:
        train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    else:
        train_set = TrainSetLoader_aug(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
        total_loss_pred_epoch = []
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=False, persistent_workers=False)
    save_path = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + mark
    os.makedirs(save_path, exist_ok=True)
    net = Net(model_name=opt.model_name, mode='train', conv=opt.conv, save_path=save_path).cuda()
    net.train()


    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    
    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                # for i in range(len(opt.step)):
                #     opt.step[i] = opt.step[i] - ckpt['epoch']
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings['lr'] = 0.05
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings['epochs'] = 1500
        opt.scheduler_settings['min_lr'] = 1e-3
        
        opt.nEpochs = opt.scheduler_settings['epochs']
        
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)
    
    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            if not interference:
                pred = net.forward(img, idx_epoch)
                loss = net.loss(pred, gt_mask)
            else:
                pred = net.forward(img[:,0,:,:].unsqueeze(1), idx_epoch)
                pred_interf = net.forward(img[:,1,:,:].unsqueeze(1), idx_epoch)
                loss, loss_pred = net.loss_interf(pred, pred_interf, gt_mask)
                total_loss_pred_epoch.append(loss_pred.detach().cpu())
            total_loss_epoch.append(loss.detach().cpu())
            loss = loss / opt.grad_accum_steps
            loss.backward()

            if (idx_iter + 1) % opt.grad_accum_steps == 0 or (idx_iter + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        if (idx_epoch + 1) % 10 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            if not interference:
                print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,'
                      % (idx_epoch + 1, total_loss_list[-1]))
                opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                      % (idx_epoch + 1, total_loss_list[-1]))
                total_loss_epoch = []
            else:
                print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, total_loss_pred---%f,'
                      % (idx_epoch + 1, total_loss_list[-1], float(np.array(total_loss_pred_epoch).mean())))
                opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, total_loss_pred---%f,\n'
                      % (idx_epoch + 1, total_loss_list[-1], float(np.array(total_loss_pred_epoch).mean())))
                total_loss_epoch = []
                total_loss_pred_epoch = []
            
        if (idx_epoch + 1) % 50 == 0 and (idx_epoch + 1) != opt.nEpochs:
            save_pth = save_path + '/' + str(idx_epoch + 1) + '.pth.tar'  #
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)
            
        if (idx_epoch + 1) == opt.nEpochs:
            save_pth = save_path + '/' + str(idx_epoch + 1) + '.pth.tar'  #
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            test(save_pth)

def test(save_pth):
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test', conv=opt.conv).cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img, 100)
        pred = pred[:,:,:size[0],:size[1]]
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("IoU, nIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("IoU, nIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    if opt.seed is not None:
        init_seeds(opt.seed)
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + mark + '_' + (time.ctime()).replace(' ', '_').replace(':', '-') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name + mark)
            train()
            print('\n')
            opt.f.close()