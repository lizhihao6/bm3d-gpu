BM3D Pytorch Version
========

🌟 We support RAW image denoising with multi-channel and int type! 🌟

Has been tested in pytorch=1.10.1, python=3.8, CUDA=11.1

# Install

```bash
export CUDA_HOME=/usr/local/cuda #use your CUDA instead
sh install.sh
```

# Test

```bash
python test/test.py
```

# Usage

```python
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
    lq = torch.from_numpy(lq)[None, None].repeat(1, channels, 1, 1) # [1, C, H, W]
    gt = torch.from_numpy(gt)[None, None].repeat(1, channels, 1, 1) # [1, C, H, W]
    lq, gt = lq.cuda(), gt.cuda()
    lq, gt = lq.int(), gt.int()
    variance = 20 * 20

    lq, gt = lq * scale, gt * scale
    variance = variance * (scale ** 2)    

    bm3d = BM3D(two_step=True)

    pred = bm3d(lq, variance=variance)

    mse = torch.mean((pred.float() / 255. / scale - gt.float() / 255. / scale) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    
    print("PSNR: {:.2f}".format(psnr.item()))
```

# Thanks
Most of the code is based on the implementation of David Honzátko <david.honzatko@epfl.ch> in [bm3d-gpu](https://github.com/DawyD/bm3d-gpu)

If you find this implementation useful please cite the following paper in your work:

    @article{bm3d-gpu,
        author = {Honzátko, David and Kruliš, Martin},
        year = {2017}, month = {11},
        title = {Accelerating block-matching and 3D filtering method for image denoising on GPUs},
        booktitle = {Journal of Real-Time Image Processing}
    }
