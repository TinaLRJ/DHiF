import os
import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
import scipy.io as scio
# matplotlib.use('agg')


def PIL2Tensor(pil_image):
    if isinstance(pil_image, list):
        pils = []
        for img in pil_image:
            pils.append(F.to_tensor(img))
        return torch.stack(pils)
    return F.to_tensor(pil_image)


def Tensor2PIL(tensor_image, mode='RGB'):
    if len(tensor_image.size()) == 4 and tensor_image.size()[0] == 1:
        tensor_image = tensor_image.view(tensor_image.size()[1:])
    return F.to_pil_image(tensor_image.detach(), mode=mode)



def cv2_to_pil(img):

    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def make_pil_grid(pil_image_list):
    sizex, sizey = pil_image_list[0].size
    for img in pil_image_list:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    target = Image.new('RGB', (sizex * len(pil_image_list), sizey))
    left = 0
    right = sizex
    for i in range(len(pil_image_list)):
        target.paste(pil_image_list[i], (left, 0, right, sizey))
        left += sizex
        right += sizex
    return target


def vis_saliency_kde(map, image, path, alpha=0.5):
    path, dataset, iter_num = path
    savepath = os.path.join('./results/Attribution_ZeroLinearPath/', dataset, 'On_Img')
    os.makedirs(savepath, exist_ok=True)
    b,h,w = map.shape
    for i in range(b):
        map_i = map[i, :, :]
        img_i = np.array(image[i, 0, :, :].data.cpu())
        png_name = path[i].split('/')[-1].split('.')[0] + '_Epoch_%d.png' % iter_num

        grad_flat = map_i.reshape((-1))
        datapoint_y, datapoint_x = np.mgrid[0:h:1, 0:w:1]
        Y, X = np.mgrid[0:h:1, 0:w:1]
        positions = np.vstack([X.ravel(), Y.ravel()])
        pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
        kernel = stats.gaussian_kde(pixels, weights=grad_flat)
        Z = np.reshape(kernel(positions).T, map_i.shape)
        Z = Z / Z.max()
        cmap = plt.get_cmap('seismic')
        # cmap = plt.get_cmap('Purples')
        map_color = cv2.cvtColor((255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # map_color = (255 * cmap(Z)).astype(np.uint8)
        img_i_rgb = cv2.cvtColor(((img_i-img_i.min())/(img_i.max()-img_i.min())*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(img_i_rgb, alpha, map_color, 1-alpha, 0)
        cv2.imwrite(savepath+'/'+png_name, blended)
    return


def color(gray):
    rgb = np.expand_dims(np.array([255, 0, 0]), axis=(0,1,2))
    return rgb*np.expand_dims(gray, axis=3).astype(np.uint8)


def vis_saliency(map, image, result, label, path, alpha=0.5):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    path, model_name, dataset, iter_num = path
    savepath = os.path.join('./results/Attribution_ZeroLinearPath/', dataset, model_name, 'dConv_LAM')  # dConv_LAM_FA1
    os.makedirs(savepath, exist_ok=True)
    savepath_show = os.path.join('./results/Attribution_ZeroLinearPath/', dataset, model_name, 'dConv_Show')  # dConv_Show_FA1
    os.makedirs(savepath_show, exist_ok=True)
    b,h,w = map.shape
    for i in range(b):
        map_i = map[i, :, :]
        img_i = np.array(image[i, 0, :, :])
        res_i = cv2.cvtColor(np.array(result[i, 0, :, :])*255, cv2.COLOR_GRAY2BGR)
        lab_i = cv2.cvtColor(np.array(label[i, 0, :, :])*255, cv2.COLOR_GRAY2BGR)
        png_name = path[i].split('/')[-1].split('.')[0] + '_Epoch_%d.png' % iter_num

        cmap = plt.get_cmap('seismic')
        map_color = cv2.cvtColor((255*cmap(map_i*0.5+0.5)).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(savepath+'/'+png_name, map_color)
        img_i_rgb = cv2.cvtColor(((img_i-img_i.min())/(img_i.max()-img_i.min())*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(img_i_rgb, alpha, map_color, 1-alpha, 0)
        show = np.hstack([lab_i, img_i_rgb, map_color, blended, res_i])
        cv2.imwrite(savepath_show + '/' + png_name, show)

        save_name = os.path.join(savepath, png_name).replace('.png', '.mat')
        scio.savemat(save_name, {'map_attribution': map_i})
    return


def vis_saliency_CE(map, cim, image, result, label, path, lam_ready, ce_patchsize, alpha=0.5, inference_form=None):
    """
    :param map: the saliency map, 2D, norm to [0, 1]
    :param zoomin: the resize factor, nn upsample
    :return:
    """
    path, model_name, dataset, iter_num = path
    savepath = os.path.join('./results/Attribution_ZeroLinearPath_CE/', dataset, model_name, 'LAM')  # dConv_LAM_FA1
    os.makedirs(savepath, exist_ok=True)
    savepath_show = os.path.join('./results/Attribution_ZeroLinearPath_CE/', dataset, model_name, 'Show')  # dConv_Show_FA1
    os.makedirs(savepath_show, exist_ok=True)
    if cim is not None:
        savepath_cim = os.path.join('./results/Attribution_ZeroLinearPath_CE/', dataset, model_name, 'CIM%d_%s' % (ce_patchsize, inference_form))  # dConv_Show_FA1
        os.makedirs(savepath_cim, exist_ok=True)
        savepath_cim_show = os.path.join('./results/Attribution_ZeroLinearPath_CE/', dataset, model_name, 'CIM%d_%s_show' % (ce_patchsize, inference_form))  # dConv_Show_FA1
        os.makedirs(savepath_cim_show, exist_ok=True)
    b,h,w = map.shape
    cmap = plt.get_cmap('seismic')
    for i in range(b):
        map_i = map[i, :, :]
        img_i = np.array(image[i, 0, :, :])
        res_i = cv2.cvtColor(np.array(result[i, 0, :, :])*255, cv2.COLOR_GRAY2BGR)
        lab_i = cv2.cvtColor(np.array(label[i, 0, :, :])*255, cv2.COLOR_GRAY2BGR)
        png_name = path[i].split('/')[-1].split('.')[0] + '_Epoch_%d.png' % iter_num

        img_i_rgb = cv2.cvtColor(((img_i - img_i.min()) / (img_i.max() - img_i.min()) * 255).astype(np.uint8),
                                 cv2.COLOR_GRAY2BGR)
        if not lam_ready:
            map_color = cv2.cvtColor((255*cmap(map_i*0.5+0.5)).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(savepath+'/'+png_name, map_color)
            blended = cv2.addWeighted(img_i_rgb, alpha, map_color, 1-alpha, 0)
            show = np.hstack([lab_i, img_i_rgb, map_color, blended, res_i])
            cv2.imwrite(savepath_show + '/' + png_name, show)

            save_name = os.path.join(savepath, png_name).replace('.png', '.mat')
            scio.savemat(save_name, {'map_attribution': map_i})

        if cim is not None:
            cim_i = cim[i, 0, :, :].astype(np.float32)
            save_name = os.path.join(savepath_cim, png_name).replace('.png', '.mat')
            scio.savemat(save_name, {'cim': cim_i})

            # cim_i = cim_i/max(abs(cim_i))
            cim_i[0, 0] = -0.00001
            cim_i[0, 1] = 0.00001
            cim_color = cim_i
            cim_color[cim_color>0] = cim_color[cim_color>0] / cim_i.max()
            cim_color[cim_color<0] = cim_color[cim_color<0] / abs(cim_i.min())
            cim_color = cv2.cvtColor((255*cmap(cim_color*0.5+0.5)).astype(np.uint8), cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(img_i_rgb, alpha, cim_color, 1-alpha, 0)
            norm = TwoSlopeNorm(vmin=cim_i.min(), vcenter=0, vmax=cim_i.max())
            plt.imshow(cim_i, cmap=cmap, norm=norm)
            plt.colorbar()
            plt.savefig(savepath_cim + '/' + png_name)
            plt.close()
            cim_show = np.hstack([lab_i, img_i_rgb, cim_color, blended, res_i])
            cv2.imwrite(savepath_cim_show + '/' + png_name, cim_show)
    return


def prepare_images(hr_path, scale=4):
    hr_pil = Image.open(hr_path)
    sizex, sizey = hr_pil.size
    hr_pil = hr_pil.crop((0, 0, sizex - sizex % scale, sizey - sizey % scale))
    sizex, sizey = hr_pil.size
    lr_pil = hr_pil.resize((sizex // scale, sizey // scale), Image.BICUBIC)
    return lr_pil, hr_pil


def grad_abs_norm(grad):
    """

    :param grad: numpy array
    :return:
    """
    grad_2d = np.abs(grad.sum(axis=1))
    grad_2d = np.sqrt(grad_2d)
    #平方根


    grad_max = grad_2d.max()

    grad_norm = grad_2d / (grad_max+1e-10)

    grad_abs = np.sqrt(np.abs(grad))
    return grad_norm


def interpolation(x, x_prime, fold, mode='linear'):
    diff = x - x_prime
    l = torch.linspace(0, 1, fold).reshape((fold, 1, 1, 1, 1)).to(x.device)
    interp_list = l * diff + x_prime
    return interp_list


def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    ax = torch.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel / torch.sum(kernel)  # [1,1,l,l]
