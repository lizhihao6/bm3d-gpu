import cv2
import torch
import numpy as np

from pytorch_bm3d import BM3D

scale = 2**6 - 1 
channels = 4

if __name__ == '__main__':
    lq = cv2.imread('test/lena_20.png', cv2.IMREAD_UNCHANGED)[..., ::-1]    
    gt = cv2.imread('test/lena.png', cv2.IMREAD_UNCHANGED)[..., ::-1]
    lq, gt = np.ascontiguousarray(lq), np.ascontiguousarray(gt)
    lq = torch.from_numpy(lq)[None, None].repeat(1, channels, 1, 1)
    gt = torch.from_numpy(gt)[None, None].repeat(1, channels, 1, 1)
    lq, gt = lq.cuda(), gt.cuda()
    lq, gt = lq.int(), gt.int()
    variance = 20 * 20

    lq, gt = lq * scale, gt * scale
    variance = variance * (scale ** 2) + 0.0001

    bm3d = BM3D(two_step=True)

    pred = bm3d(lq, variance=variance)

    mse = torch.mean((pred.float() / 255. / scale - gt.float() / 255. / scale) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    
    print("PSNR: {:.2f}".format(psnr.item()))
