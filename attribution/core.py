import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attribution.utils import grad_abs_norm, vis_saliency, vis_saliency_kde, vis_saliency_CE
from attribution.utils import interpolation, isotropic_gaussian_kernel
import scipy.io as scio
import os
from tqdm import tqdm




# def GaussianBlurPath(sigma, fold, l=5):
#     def path_interpolation_func(cv_numpy_image):
#         h, w, c = cv_numpy_image.shape
#         kernel_interpolation = np.zeros((fold + 1, l, l))
#         image_interpolation = np.zeros((fold, h, w, c))
#         lambda_derivative_interpolation = np.zeros((fold, h, w, c))
#         sigma_interpolation = np.linspace(sigma, 0, fold + 1)
#         for i in tqdm(range(fold + 1)):
#             kernel_interpolation[i] = isotropic_gaussian_kernel(l, sigma_interpolation[i])
#         for i in tqdm(range(fold)):
#             image_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, kernel_interpolation[i + 1])
#             lambda_derivative_interpolation[i] = cv2.filter2D(cv_numpy_image, -1, (
#                     kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold)
#         return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
#             np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)
#
#     return path_interpolation_func

# 另一种积分路径函数
def MeanLinearPath(fold=50, l=9):
    def path_interpolation_func(images):
        med_image = images.median(1)[0].unsqueeze(1)
        # template = isotropic_gaussian_kernel(l, sigma)
        template = torch.ones([1, 1, l, l])/(l*l)
        baseline_image = F.conv2d(med_image, template, stride=1, padding=l//2)
        baseline_images = baseline_image.repeat(1, images.size(1), 1, 1)
        image_interpolation = interpolation(images, baseline_images, fold, mode='linear').type(torch.float32)
        lambda_derivative_interpolation = torch.unsqueeze(images - baseline_images, dim=0).repeat(fold, 1,1,1,1)
        return image_interpolation, lambda_derivative_interpolation

    return path_interpolation_func

# 另一种积分路径函数
def ZeroLinearPath(fold=50):
    def path_interpolation_func(images):
        # baseline_images = torch.zeros_like(images)
        baseline_images = torch.ones_like(images)*images.min()
        image_interpolation = interpolation(images, baseline_images, fold, mode='linear').type(torch.float32)
        lambda_derivative_interpolation = torch.unsqueeze(images - baseline_images, dim=0).repeat(fold, 1,1,1,1)
        return image_interpolation, lambda_derivative_interpolation

    return path_interpolation_func


## attribution
def IR_Integrated_gradient(image, label, path, model, path_interpolation_func, cuda=True):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_tensor_images
    :return:
    """
    b,c,h,w = label.size()
    m,n = image.shape[-2:]
    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(image.data.cpu())
    grad_accumulate_list = np.zeros_like(image_interpolation.data.cpu().numpy())
    grad_accumulate_list = grad_accumulate_list[:,:,:,:h,:w]
    result_list = []
    with torch.set_grad_enabled(True):
        for i in range(image_interpolation.shape[0]):
            img_tensor = image_interpolation[-i-1].cuda()
            img_tensor.requires_grad_(True)
            results = model(img_tensor)
            if isinstance(results, list):
                result = results[-1]
            else:
                result = results
            result = result[:,:,:h,:w]
            # if i == 0:
            #     lab_fa = FA(result, label.cuda())
            # target = torch.sum(result*lab_fa)
            target = torch.sum(result*label.cuda())
            target.backward()
            grad = img_tensor.grad[:,:,:h,:w].cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

            grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i][:,:,:h,:w].cpu().numpy()
            result_list.append(results)

    final_grad, results = saliency_map_PG(grad_accumulate_list, result_list)
    abs_normed_grad_numpy = grad_abs_norm(final_grad)

    # 可视化
    # Visualize saliency
    if isinstance(results, list):
        result = results[-1]
    else:
        result = results
    result = result[:,:,:h,:w].data.cpu()
    if abs_normed_grad_numpy.sum() == 0:
        return result
    else:
        vis_saliency(abs_normed_grad_numpy, image[:,:,:h,:w].data.cpu(), result, label, path)
        # vis_saliency_kde(abs_normed_grad_numpy, image, path)
        return result


def blur_processing(image, blur_ks=5):
    blur_kernel = torch.ones([1, 1, blur_ks, blur_ks]).to(image.device) / (blur_ks*blur_ks)
    padding = nn.ReflectionPad2d((blur_ks-1)//2).to(image.device)
    image_blur = F.conv2d(padding(image), blur_kernel, stride=1)
    return image_blur



## attribution
def IR_Integrated_gradient_CE(image, label, ce_patchsize, path, model, path_interpolation_func, interference=True, cuda=True):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_tensor_images
    :return:
    """
    b,c,h,w = label.size()
    m,n = image.shape[-2:]
    image_blur = blur_processing(image, 5)
    lam_ready = True
    if not lam_ready:
        image_interpolation, lambda_derivative_interpolation = path_interpolation_func(image.data.cpu())
        grad_accumulate_list = np.zeros_like(image_interpolation.data.cpu().numpy())
        grad_accumulate_list = grad_accumulate_list[:,:,:,:h,:w]
        result_list = []
        with torch.set_grad_enabled(True):
            for i in range(image_interpolation.shape[0]):
                img_tensor = image_interpolation[-i-1].cuda()
                img_tensor.requires_grad_(True)
                results = model(img_tensor)
                if isinstance(results, list):
                    result = results[-1]
                else:
                    result = results
                result = result[:,:,:h,:w]
                # if i == 0:
                #     lab_fa = FA(result, label.cuda())
                # target = torch.sum(result*lab_fa)
                target = torch.sum(result*label.cuda())
                target.backward()
                grad = img_tensor.grad[:,:,:h,:w].cpu().numpy()
                if np.any(np.isnan(grad)):
                    grad[np.isnan(grad)] = 0.0

                grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i][:,:,:h,:w].cpu().numpy()
                result_list.append(results)

        final_grad, results = saliency_map_PG(grad_accumulate_list, result_list)
        abs_normed_grad_numpy = grad_abs_norm(final_grad)
        normed_grad_numpy = abs_normed_grad_numpy * ((final_grad[:,0,:,:]>0)*2-1)
    else:
        img_path, model_name, dataset, iter_num = path
        lampath = os.path.join('./results/Attribution_ZeroLinearPath_CE/', dataset, model_name, 'LAM_PosNeg')
        lam_name = img_path[0].split('/')[-1].split('.')[0] + '_Epoch_%d.mat' % iter_num
        normed_grad_numpy = scio.loadmat(os.path.join(lampath, lam_name))['map_attribution'][None, :, :]
        abs_normed_grad_numpy = abs(normed_grad_numpy)
        results = model(image)


    if isinstance(results, list):
        result = results[-1]
    else:
        result = results
    result = result[:,:,:h,:w]

    inference_form = None
    if interference:
        abs_normed_grad = torch.zeros_like(image.data.cpu())
        abs_normed_grad[:,0,:h,:w] = torch.from_numpy(abs_normed_grad_numpy)
        abs_normed_grad_unfold = F.unfold(abs_normed_grad, kernel_size=ce_patchsize, stride=ce_patchsize, padding=0).sum(1)
        abs_normed_grad_unfold = torch.reshape(abs_normed_grad_unfold, (b, 1, m//ce_patchsize, n//ce_patchsize))
        # grad_thred = 1/64*ce_patchsize*ce_patchsize
        grad_thred = 1/200*ce_patchsize*ce_patchsize
        big_grad_idx = torch.nonzero(abs_normed_grad_unfold>grad_thred)

        inference_form = 'mean'
        cim = np.zeros([b,c,h,w])
        for i in range(big_grad_idx.shape[0]):
            idx = big_grad_idx[i,:]
            img_inter = image.clone()
            if inference_form == 'blur':
                img_inter[idx[0],idx[1],idx[2]*ce_patchsize:(idx[2]+1)*ce_patchsize,idx[3]*ce_patchsize:(idx[3]+1)*ce_patchsize] = \
                    image_blur[idx[0],idx[1],idx[2]*ce_patchsize:(idx[2]+1)*ce_patchsize,idx[3]*ce_patchsize:(idx[3]+1)*ce_patchsize]
            elif inference_form == 'mean':
                img_inter[idx[0],idx[1],idx[2]*ce_patchsize:(idx[2]+1)*ce_patchsize,idx[3]*ce_patchsize:(idx[3]+1)*ce_patchsize] = \
                    torch.mean(img_inter[idx[0],idx[1],idx[2]*ce_patchsize:(idx[2]+1)*ce_patchsize,idx[3]*ce_patchsize:(idx[3]+1)*ce_patchsize])
            elif inference_form == '0':
                img_inter[idx[0],idx[1],idx[2]*ce_patchsize:(idx[2]+1)*ce_patchsize,idx[3]*ce_patchsize:(idx[3]+1)*ce_patchsize] = 0
            res_inter = model(img_inter)
            if isinstance(res_inter, list):
                re_inter = res_inter[-1]
            else:
                re_inter = res_inter
            re_inter = re_inter[:,:,:h,:w]
            tar_inter = torch.sum(re_inter*label.cuda())
            cim[idx[0],idx[1],idx[2]*ce_patchsize:(idx[2]+1)*ce_patchsize,idx[3]*ce_patchsize:(idx[3]+1)*ce_patchsize] = \
                (torch.sum(result*label.cuda()).data.cpu() - tar_inter.data.cpu())/torch.sum(label)
    else:
        cim = None


    # 可视化
    # Visualize saliency
    result = result.data.cpu()
    if abs_normed_grad_numpy.sum() == 0:
        return result
    else:
        # lam_ready = False
        vis_saliency_CE(abs_normed_grad_numpy, cim, image[:,:,:h,:w].data.cpu(), result, label, path, lam_ready, ce_patchsize, inference_form=inference_form)
        # vis_saliency_kde(abs_normed_grad_numpy, image, path)
        return result


def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[0]


def FA(result, label):
    extend = 5
    result1 = result.clone()
    result1[result1<0.5] = 0
    template = torch.ones(1, 1, 2*extend+1, 2*extend+1).cuda()    ## [1,1,5,5]
    large_lab = F.conv2d(label.float(), template, stride=1, padding=extend)          ## [2,1,512,512]
    large_lab = (large_lab > 0).float()
    lab_fa = result1 * (1-large_lab)
    lab_fa = (lab_fa > 0).float()
    return lab_fa

