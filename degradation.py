# -*- coding: utf-8 -*-

import os
import random
import torch
import numpy as np
import os.path as osp
import realesrgan.data
import realesrgan.models
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from basicsr.utils.options import parse_options
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.utils import DiffJPEG, USMSharp
from torch.nn import functional as F
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import augment, paired_random_crop

'''
degradate mosaic images to scale size
'''

def main():
    root_path = osp.abspath(osp.join(__file__, '..'))
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    dataset_opt = opt['datasets']['train']
    train_set = build_dataset(dataset_opt)
    dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
    train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
    train_loader = build_dataloader(
        train_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'])


    for data in train_loader:
        device = 'cuda'
        gt = data['gt'].to(device)
        kernel1 = data['kernel1'].to(device)
        kernel2 = data['kernel2'].to(device)
        sinc_kernel = data['sinc_kernel'].to(device)

        jpeger = DiffJPEG(differentiable=False).cuda()
        usm_sharpener = USMSharp().cuda()
        gt_usm = usm_sharpener(gt)

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt_usm, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob']
        if np.random.uniform() < opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob2']
        if np.random.uniform() < opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = opt['gt_size']
        #(gt, gt_usm), lq = paired_random_crop([gt, gt_usm], lq, gt_size, opt['scale'])

        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        gt_usm = usm_sharpener(gt)
        lq = lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        # save images to disk
        lq_list = [os.path.join(root_path,dataset_opt['lq_info'],os.path.basename(path)) for path in data['gt_path']]
        for i,d in enumerate(lq):
            d = d.mul(255.).clamp(0, 255.).cpu()
            ToPILImage()(d).save(lq_list[i])



if __name__ == '__main__':
    main()